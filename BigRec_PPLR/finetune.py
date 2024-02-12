import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, LlamaForCausalLM, LlamaTokenizer 
from transformers import Trainer, TrainingArguments
from copy import deepcopy
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import pickle
import time
from peft import PeftModel
import torch.distributed as dist
import math
from utils import split_dataset, aggregate, LoggingCallback, get_aggregate_lora_weight, merge_models, split_client_server
os.environ['LD_LIBRARY_PATH'] = ''
import logging
dist.init_process_group(backend='nccl', init_method='env://')

def softmax_with_temperature(x, temperature=0.05):
    x = np.array(x) / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: List[str] = [""],
    val_data_path: List[str] = [""],
    test_data_path: List[str] = [""],
    output_dir: str = "./lora-alpaca",
    pretrain_emb_path: str = "../data/games/group_ckpt.pth.tar",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # federated params
    client_num: int = 3,
    patience: int = 5,
    round: int = 1,
    alpha: float = 0.7,
    beta: int = 1,
    k: int = 20,
):
    # print the hyperparameters
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f'client_num: {client_num}\n'
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_DISABLED"] = "true"
    # create client
    client = {}
    client[0] = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model_server, model_client = split_client_server(client[0], k)
    merge_model = merge_models(model_client, model_server)
    del model_client, model_server
    weights = merge_model.state_dict()
    del merge_model 
    client[0].load_state_dict(weights)
    del weights
    dist.barrier()

    # del base_client
    # dist.barrier()
    # # load dict for base_client
    # client[0].load_state_dict(weights)
    # del model_client, model_server
    # torch.cuda.empty_cache()
    # dist.barrier()

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    client[0] = prepare_model_for_int8_training(client[0])

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    client[0] = get_peft_model(client[0], config)

    train_data_list = []
    val_data_list = []
    test_data_list = []
   
    for path in train_data_path:
        if path.endswith(".json"):
            train_data_list.append(load_dataset("json", data_files=path))
        else:
            train_data_list.append(load_dataset(path))

    for path in val_data_path:
        if path.endswith(".json"):
            val_data_list.append(load_dataset("json", data_files=path))
        else:
            val_data_list.append(load_dataset(path))

    for path in test_data_path:
        if path.endswith(".json"):
            test_data_list.append(load_dataset("json", data_files=path))
        else:
            test_data_list.append(load_dataset(path))

    for i in range(len(train_data_list)):
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    for i in range(len(val_data_list)):
        val_data_list[i] = val_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    for i in range(len(test_data_list)):
        test_data_list[i] = test_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    val_data = concatenate_datasets([_["train"] for _ in val_data_list])
    test_data = concatenate_datasets([_["train"] for _ in test_data_list])

    # get client dataset if pkl file doens't exist
    if not os.path.exists('./data/train_client_data.pkl') or not os.path.exists('./data/valid_client_data.pkl') or not os.path.exists('./data/test_client_data.pkl'):
        client_data, val_data, test_data = split_dataset(train_data, client_num, val_data, test_data, pretrain_emb_path)
        with open('./data/train_client_data.pkl', 'wb') as file:
            pickle.dump(client_data, file)
        with open('./data/valid_client_data.pkl', 'wb') as file:
            pickle.dump(val_data, file)
        with open('./data/test_client_data.pkl', 'wb') as file:
            pickle.dump(test_data, file)
    else:
        with open('./data/train_client_data.pkl', 'rb') as file:
            client_data = pickle.load(file)
        with open('./data/valid_client_data.pkl', 'rb') as file:
            val_data = pickle.load(file)
        with open('./data/test_client_data.pkl', 'rb') as file:
            test_data = pickle.load(file)
            
    if dist.get_rank() == 0:
        logging.info(f'Client num: {client_num}')
        for cnt, client_ in enumerate(client_data):
            logging.info(f'Client {cnt} has {len(client_)} samples')
        for cnt, client_ in enumerate(val_data):
            logging.info(f'Client {cnt} has {len(client_)} vlidation samples')
        for cnt, client_ in enumerate(test_data):
            logging.info(f'Client {cnt} has {len(client_)} test samples')
        
    client[0].print_trainable_parameters()  # Be more transparent about the % of trainable params.

    best_eval_loss = [1e5 for _ in range(client_num)]
    best_eval_loss_all = 1e5
    warmup_step = 20
    num_update_steps_per_epoch = len(client_data[0]) // gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    warm_weight = [0 for _ in range(len(client_data))]
    # begin training
    for epoch in range(num_epochs):
        train_loss = []
        client_trainer = {}
        eval_trainer = {}
        eval_results = []
        if epoch % 2 == 0:
            save_name = f'ori'
            update_name = f'update'
        else:
            save_name = f'update'
            update_name = f'ori'
        for i in range(client_num):
            if epoch == 0 and i != 0:
                client[i] = LlamaForCausalLM.from_pretrained(
                        base_model,
                        load_in_8bit=True,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                    )
                model_server, model_client = split_client_server(client[i], k)
                merge_model = merge_models(model_client, model_server)
                del model_client, model_server
                weights = merge_model.state_dict()
                del merge_model 
                client[i].load_state_dict(weights)
                del weights
                client[i] = prepare_model_for_int8_training(client[i])
                client[i] = get_peft_model(client[i], config)
            if epoch != 0:
                warmup_step = 0
                client[i] = LlamaForCausalLM.from_pretrained(
                    base_model,
                    load_in_8bit=True,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
                client[i] = prepare_model_for_int8_training(client[i])
                client[i] = get_peft_model(client[i], config)
                state_dict = torch.load(f'{output_dir}/client{i}_{save_name}/adapter_model.bin')
                client[i] = set_peft_model_state_dict(client[i], state_dict)

            if not ddp and torch.cuda.device_count() > 1:
                client[i].is_parallelizable = True
                client[i].model_parallel = True
            if dist.get_rank() == 0:
                print(f"Training client {i} for epoch {epoch}")
                logging.info(f'Training client {i} for epoch {epoch}')
            client_trainer[i] = Trainer(
                model=client[i],
                train_dataset=client_data[i],
                callbacks=[LoggingCallback],
                args=transformers.TrainingArguments(
                    per_device_train_batch_size=micro_batch_size,
                    per_device_eval_batch_size=micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=warmup_step,
                    num_train_epochs=round,
                    learning_rate=learning_rate,
                    fp16=True,
                    logging_strategy="steps",
                    logging_steps=4,
                    optim="adamw_torch",
                    save_strategy="steps",
                    output_dir=output_dir,
                    save_total_limit=1,
                    ddp_find_unused_parameters=False if ddp else None,
                    group_by_length=group_by_length,
                    report_to=None,
                ),
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ))
            client[i].config.use_cache = False
            client_trainer[i].train()
            client[i].save_pretrained(f'{output_dir}/client{i}_{save_name}')
            train_loss.append(client_trainer[i].state.log_history[-1]['train_loss'])
            # delete used client model to save gpu
            del client[i], client_trainer[i]

        sim_matrix, accumulated_params = aggregate(output_dir, device_map, client_num, save_name, base_model)
        train_loss = softmax_with_temperature(train_loss)
        # update each client according to the cluster result
        for i in range(client_num):
            print(f'update client{i} model')
            client[i] = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,
            )

            client[i] = PeftModel.from_pretrained(
                        client[i],
                        f'{output_dir}/client{i}_{save_name}',
                        torch_dtype=torch.float16,
                        device_map=device_map,
                    )
            warm_weight[i] =  math.tanh(alpha/(train_loss[i]**(epoch+1/beta)))
            lora_weight = get_aggregate_lora_weight(i, sim_matrix, accumulated_params, warm_weight[i], beta)
            client[i].load_state_dict(lora_weight, strict=False)
            client[i].save_pretrained(f'{output_dir}/client{i}_{update_name}')

            # eval client i result
            client[i].eval()
            eval_trainer[i] = Trainer(
                                model=client[i],
                                eval_dataset=val_data[i],
                                args=transformers.TrainingArguments(
                                    per_device_train_batch_size=micro_batch_size,
                                    per_device_eval_batch_size=micro_batch_size,
                                    gradient_accumulation_steps=gradient_accumulation_steps,
                                    warmup_steps=warmup_step,
                                    num_train_epochs=1,
                                    learning_rate=learning_rate,
                                    fp16=True,
                                    logging_steps=8,
                                    optim="adamw_torch",
                                    evaluation_strategy="steps",
                                    # eval_steps=1,
                                    save_strategy="steps",
                                    # save_steps=1,
                                    output_dir=output_dir,
                                    save_total_limit=1,
                                    # load_best_model_at_end=True,
                                    ddp_find_unused_parameters=False if ddp else None,
                                    group_by_length=group_by_length,
                                    report_to=None,),
                                    data_collator=transformers.DataCollatorForSeq2Seq(
                                                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))
            eval_info = eval_trainer[i].evaluate()
            eval_results.append(eval_info["eval_loss"])
            if dist.get_rank() == 0:
                logging.info(f'Epoch {epoch}: client {i} eval_results: {eval_info["eval_loss"]}')
            if eval_info["eval_loss"] < best_eval_loss[i]:
                best_eval_loss[i] = eval_info["eval_loss"]
                # save best model
                old_state_dict = client[i].state_dict
                client[i].state_dict = (
                    lambda self, *_, **__: get_peft_model_state_dict(
                        self, old_state_dict()
                    )
                ).__get__(client[i], type(client[i]))

                if torch.__version__ >= "2" and sys.platform != "win32":
                    client[i] = torch.compile(client[i])
                client[i].save_pretrained(f'{output_dir}/best_client{i}_model')
            del eval_trainer[i], client[i]
            torch.cuda.empty_cache()
        
        sum_ = 0
        all_eval_num = 0
        for cnt, client_eval in enumerate(eval_results):
            sum_ += client_eval * len(val_data[cnt])
            all_eval_num += len(val_data[cnt])
        eval_results = sum_ / all_eval_num
        if dist.get_rank() == 0:
            logging.info(f'Epoch {epoch}: Overall eval_results: {eval_results}')

        # early stop acording to eval loss
        if eval_results < best_eval_loss_all:
            best_eval_loss_all = eval_results
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= patience and epoch > 10:
            print("Early stop!")
            break
        torch.cuda.empty_cache()
        

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)