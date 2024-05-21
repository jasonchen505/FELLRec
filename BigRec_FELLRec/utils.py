import numpy as np
import torch
import utils
import math
from pprint import pprint
from time import time
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import roc_auc_score
import copy
from collections import defaultdict
from datasets import Dataset
from transformers import EarlyStoppingCallback, LlamaForCausalLM, LlamaTokenizer 
from peft import PeftModel
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import TrainerCallback
import logging
import torch.distributed as dist
from copy import deepcopy
import torch.nn as nn

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 检查是否有训练损失
        if dist.get_rank() == 0:
            if 'loss' in logs:
                logging.info(f"Step: {state.global_step}, Loss: {logs['loss']}")


def split_dataset(train_data, n, val_data, test_data, pretrain_emb_path):
    # group data by user id emb
    # load model weight
    MF_model = torch.load(pretrain_emb_path)
    if dist.get_rank() == 0:
        logging.info(f"Load pretrain model from {pretrain_emb_path}")
        logging.info(f"Pretrain model weight: {MF_model}")

    # cluster the data
    all_user = set()
    for data in train_data:
        all_user.add(data['user'])
    for data in val_data:
        all_user.add(data['user'])
    for data in test_data:
        all_user.add(data['user'])
    all_user = list(all_user)
    all_user_emb = []
    for user in all_user:
        all_user_emb.append(MF_model['embedding_user.weight'][user].cpu())
    all_user_emb = torch.stack(all_user_emb)
    # print(all_user_emb.shape)
    kmeans = KMeans(n_clusters=n).fit(all_user_emb)
    # get the client map
    client_map = {}
    for user, label in zip(all_user, kmeans.labels_):
        client_map[user] = label
    
    # split train data
    new_train_datasets = [[] for _ in range(n)]
    if dist.get_rank() == 0:
        logging.info(f"Splitting train dataset into {n} clients")
    for data in train_data:
        user = data['user'] 
        # del data['user']
        new_train_datasets[client_map[user]].append(data)

 
    # split val data
    new_val_datasets = [[] for _ in range(n)]
    if dist.get_rank() == 0:
        logging.info(f"Splitting valid dataset into {n} clients")
    for data in val_data:
        user = data['user'] 
        # del data['user']
        new_val_datasets[client_map[user]].append(data)

    # split test data
    new_test_datasets = [[] for _ in range(n)]
    if dist.get_rank() == 0:
        logging.info(f"Splitting test dataset into {n} clients")
    for data in test_data:
        # del data['user'] 
        new_test_datasets[client_map[data['user']]].append(data)

    return new_train_datasets, new_val_datasets, new_test_datasets

def cluster_clients(model_params):
    # Convert the list of PyTorch tensors to a single NumPy array
    param_1d = []
    # cnt = 0
    for param in model_params:
        param_1d.append(torch.cat(tuple(p.view(-1).cpu() for p in param.values())).detach().numpy())
    # calculate the cos similarity between each client
    params_matrix = np.vstack(param_1d)
    similarity_matrix = cosine_similarity(params_matrix)
    normalized_matrix = (similarity_matrix + 1) / 2
    return normalized_matrix
    
def extract_params(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}
    # return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad]).detach().cpu().numpy()

def aggregate(output_dir, device, client_num, save_name, base_model):
    """
    计算模型列表的平均模型。
    """
    # get cluster according to the model weight
    models = {}
    accumulated_params = []
    for i in range(client_num):
        print(f'aggregate client{i} model')
        models[i] = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        models[i] = PeftModel.from_pretrained(
            models[i],
            f'{output_dir}/client{i}_{save_name}',
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # print(models[i].named_parameters())
        accumulated_params.append(extract_params(models[i]))
        del models[i]
    sim_matrix = cluster_clients(accumulated_params)
    # del accumulated_params
    return sim_matrix, accumulated_params

def get_aggregate_lora_weight(client_index, sim_matrix, accumulated_params, weight, beta):
    for i, value in enumerate(sim_matrix[client_index]):
        if i != client_index:
            if beta != 1:
                sim_matrix[client_index][i] = sim_matrix[client_index][i] * weight
    # get the lora weight of each client
    with torch.no_grad():
        for name, param_ in accumulated_params[client_index].items():
            weighted_param = sum(param[name] * sim_matrix[client_index][cnt] for cnt, param in enumerate(accumulated_params)) / sum(sim_matrix[client_index])
            param_.copy_(weighted_param)
    return accumulated_params[client_index]
    

def get_lr_in_this_epoch(max_lr, warmup_steps, total_steps, epoch, current_step):
    # lienar decrease
    lr = max_lr - (max_lr * (current_step - warmup_steps) / (total_steps - warmup_steps))
    return lr


def split_client_server(original_model, k):
    new_model = deepcopy(original_model)
    num_layers = len(original_model.model.layers)
    server_layer = nn.ModuleList(
        [deepcopy(original_model.model.layers[i]) for i in range(k,num_layers-1)]
    )
    client_layers = nn.ModuleList(
        [deepcopy(original_model.model.layers[i]) for i in range(k)] + 
        [deepcopy(original_model.model.layers[-1])]
    )
    new_model.model.layers = client_layers
    return server_layer, new_model


def merge_models(model_front_and_last, model_middle):

    # new_model = deepcopy(model_front_and_last)
    
    front_layers = model_front_and_last.model.layers[:-1] 
    
    middle_layers = model_middle
    
    last_layer = model_front_and_last.model.layers[-1:]
    
    combined_layers = nn.ModuleList(front_layers + middle_layers + last_layer)
    
    model_front_and_last.model.layers = combined_layers
    del model_middle
    return model_front_and_last

