import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.cluster import KMeans
import torch.distributed as dist
from pytorch_lightning import seed_everything
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import math
from utils import read_json, read_json_client, aggregate, get_aggregate_lora_weight, AverageMeterSet, Ranker, computeTopNAccuracy, merge_models
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig, RecformerForSeqRec_client, RecformerForSeqRec_server
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset
import logging
import numpy as np
os.environ['LD_LIBRARY_PATH'] = ''\

def softmax_with_temperature(x, temperature=0.05):
    x = np.array(x) / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def load_data(args):

    client_map = get_user_map()
    train_client = read_json_client(os.path.join(args.data_path, args.train_file), client_map, True)
    valid_client = read_json_client(os.path.join(args.data_path, args.dev_file), client_map, True)
    test_client = read_json_client(os.path.join(args.data_path, args.test_file), client_map, True)
    train = read_json(os.path.join(args.data_path, args.train_file), True)
    # val = read_json(os.path.join(args.data_path, args.dev_file), True)
    # test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v:k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train_client, train, valid_client, test_client, item_meta_dict_filted, item2id, id2item

def get_user_map():
    pretrain_emb_path = "../data/games/group_ckpt.pth.tar"
    MF_model = torch.load(pretrain_emb_path)
    n_user = MF_model['embedding_user.weight'].shape[0]
    all_user = list(range(n_user))
    all_user_emb = []
    for user_ in all_user:
        all_user_emb.append(MF_model['embedding_user.weight'][user_].cpu())
    all_user_emb = torch.stack(all_user_emb)
    kmeans = KMeans(n_clusters=5).fit(all_user_emb)
    # get the client map
    user_map = {}
    client_map = {}
    for user_, label in zip(all_user, kmeans.labels_):
        user_map[user_] = label
        if label not in client_map:
            client_map[label] = set()
        client_map[label].add(user_)
    return client_map
        
tokenizer_glb: RecformerTokenizer = None
def _par_tokenize_doc(doc):
    
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids

def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):

    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc='Encode all items'):

            item_batch = [[item] for item in items[i:i+args.batch_size]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)#.cpu()

    return item_embeddings


def eval(model, dataloader, args, mode='val'):
    rating_list = []
    groundTrue_list = []
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    for i in range(len(model)):
        model[i].eval()

        ranker = Ranker(args.metric_ks)
        average_meter_set = AverageMeterSet()
        groundTrue_list_client = []
        rating_list_client = []
        for batch, labels in tqdm(dataloader[i], ncols=100, desc='Evaluate'):

            for k, v in batch.items():
                batch[k] = v.to(args.device)

            with torch.no_grad():
                scores = model[i](**batch)

            rating = ranker(scores)
            rating_list.extend(rating) # shape: n_batch, user_bs, max_k
            groundTrue_list.extend(labels)
            rating_list_client.extend(rating)
            groundTrue_list_client.extend(labels)

        precision_client, recall_client, NDCG_client, MRR_client = computeTopNAccuracy(groundTrue_list_client,rating_list_client,[5,10,20,50])
        if local_rank == 0:
            if mode == 'val':
                logging.info(f'Client {i} Eval Result: Precision: {precision_client}, Recall: {recall_client}, NDCG: {NDCG_client}, MRR: {MRR_client}')
            else:
                logging.info(f'Client {i} Test Result: Precision: {precision_client}, Recall: {recall_client}, NDCG: {NDCG_client}, MRR: {MRR_client}')
    precision, recall, NDCG, MRR = computeTopNAccuracy(groundTrue_list,rating_list,[5,10,20,50])
    return precision, recall, NDCG, MRR

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args, print_loss=False):

    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        if args.fp16:
            with autocast():
                loss = model(**batch)
        else:
            loss = model(**batch)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()

            else:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()
    if print_loss:
        return loss

def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument('--pretrain_ckpt', type=str, default=None, required=True)
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default='best_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='training_dict.npy')
    parser.add_argument('--dev_file', type=str, default='validation_dict.npy')
    parser.add_argument('--test_file', type=str, default='testing_dict.npy')
    parser.add_argument('--item2id_file', type=str, default='asin_to_id_map.npy')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")

    # train
    parser.add_argument('--num_train_epochs', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)
    parser.add_argument('--round', type=int, default=1)

    # PPLR specific
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=int, default=5)
    parser.add_argument('--k', type=int, default=10)

    args = parser.parse_args()
    logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
    print(args)
    seed_everything(42)

    alpha = args.alpha
    beta = args.beta

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))
    args.device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", init_method='env://', world_size=world_size, rank=local_rank)

    train_client, train, valid_client, test_client, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = {}
    valid_data = {}
    test_data = {}
    for i in range(len(train_client)):
        train_data[i] = RecformerTrainDataset(train_client[i], collator=finetune_data_collator)
        valid_data[i] = RecformerEvalDataset(train_client[i], valid_client[i], test_client[i], mode='val', collator=eval_data_collator)
        test_data[i] = RecformerEvalDataset(train_client[i], valid_client[i], test_client[i], mode='test', collator=eval_data_collator)
        if local_rank == 0:
            logging.info(f'Client {i} has {len(train_data[i])} train users.')
            logging.info(f'Client {i} has {len(valid_data[i])} valid users.')
            logging.info(f'Client {i} has {len(test_data[i])} test users.')

    train_loader = {}
    ddp_sampler = {}
    dev_loader = {}
    test_loader = {}
    for i in range(len(train_client)):
        ddp_sampler[i] = DistributedSampler(train_data[i], num_replicas=world_size, rank=local_rank, drop_last=True)
        train_loader[i] = DataLoader(train_data[i], 
                              batch_size=args.batch_size, 
                              sampler=ddp_sampler[i], 
                              collate_fn=train_data[i].collate_fn,
                              pin_memory=True)
        dev_loader[i] = DataLoader(valid_data[i], 
                                batch_size=args.batch_size, 
                                collate_fn=valid_data[i].collate_fn)
        test_loader[i] = DataLoader(test_data[i], 
                                batch_size=args.batch_size, 
                                collate_fn=test_data[i].collate_fn)
    
    model = {}
    for i in range(len(train_client)):
        model[i] = RecformerForSeqRec(config)
        model_client = RecformerForSeqRec_client(config, model[i], args.k)
        model_server = RecformerForSeqRec_server(config, model[i], args.k)
        model[i] = merge_models(config, model_client, model_server)
        pretrain_ckpt = torch.load(args.pretrain_ckpt)
        model[i].load_state_dict(pretrain_ckpt, strict=False)

        if args.fix_word_embedding:
            print('Fix word embeddings.')
            for param in model[i].longformer.embeddings.word_embeddings.parameters():
                param.requires_grad = False

        path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
        if path_item_embeddings.exists():
            print(f'[Item Embeddings] Use cache: {path_tokenized_items}')
        else:
            print(f'Encoding items.')
            item_embeddings = encode_all_items(model[i].longformer, tokenizer, tokenized_items, args)
            torch.save(item_embeddings, path_item_embeddings)
        
        item_embeddings = torch.load(path_item_embeddings)
        model[i].init_item_embedding(item_embeddings)
        model[i].to(args.device)
        model[i] = DistributedDataParallel(model[i], device_ids=[local_rank])

    num_train_optimization_steps = {}
    for i in range(len(train_client)):
        num_train_optimization_steps[i] = int(len(train_loader[i]) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = {}, {}
    for i in range(len(train_client)):
        optimizer[i], scheduler[i] = create_optimizer_and_scheduler(model[i], num_train_optimization_steps[i], args)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    best_target = float('-inf')
    patient = 5
    warm_weight = [0 for _ in range(len(train_client))]
    for epoch in range(args.num_train_epochs):
        loss_list = []
        for i in range(len(train_client)):
            for round_num in range(args.round):
                item_embeddings = encode_all_items(model[i].module.longformer, tokenizer, tokenized_items, args)
                model[i].module.init_item_embedding(item_embeddings.to(args.device))
                dist.barrier()
                if round_num == args.round - 1:
                    loss = train_one_epoch(model[i], train_loader[i], optimizer[i], scheduler[i], scaler, args, print_loss=True)
                else:
                    train_one_epoch(model[i], train_loader[i], optimizer[i], scheduler[i], scaler, args, print_loss=False)
            loss /= len(train_data[i])
            loss_list.append(loss.detach().cpu())
            if local_rank == 0:
                logging.info(f'Stage 1: Epoch {epoch} - Client {i} loss: {loss}')
            dist.barrier()
        dist.barrier()
        # aggregate the model param to get server
        sim_matrix, accumulated_params = aggregate(model)
        loss_list = softmax_with_temperature(loss_list)
        if local_rank == 0:
            logging.info(f'Epoch {epoch} - loss_list: {loss_list}')

        # save server model
        for i in range(len(train_client)):
            warm_weight[i] =  math.tanh(alpha/(loss_list[i]**(epoch+1/beta)))
            if local_rank == 0:
                logging.info(f'Client {i} warm weight: {warm_weight[i]}')
            weight = get_aggregate_lora_weight(i, sim_matrix, accumulated_params, warm_weight[i])
            model[i].load_state_dict(weight)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')
            if local_rank == 0:
                logging.info(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics[1][0] > best_target:
                print('Save the best model.')
                best_target = dev_metrics[1][0]
                patient = 5
                test_metrics = eval(model, test_loader, args, mode='test')
                print(f'Epoch: {epoch}. Test set: {test_metrics}')
                if local_rank == 0:
                    logging.info(f'Epoch: {epoch}. Test set: {test_metrics}')
                for i in range(len(train_client)):
                    path_ckpt = 'checkpoints/' + f'client{i}_' + args.ckpt
                    torch.save(model[i].state_dict(), path_ckpt)
            else:
                patient -= 1
                if patient == 0:
                    break
        dist.barrier()
    
    print('Load best model in stage 1.')
    logging.info('Load best model in stage 1.')
    for i in range(len(train_client)):
        path_ckpt = 'checkpoints/' + f'client{i}_' + args.ckpt
        model[i].load_state_dict(torch.load(path_ckpt, map_location=args.device))
    dist.barrier()
    patient = 3

    for epoch in range(args.num_train_epochs):
        for i in range(len(train_client)):
            for round_num in range(args.round):
                if round_num == args.round - 1:
                    loss = train_one_epoch(model[i], train_loader[i], optimizer[i], scheduler[i], scaler, args, print_loss=True)
                else:
                    train_one_epoch(model[i], train_loader[i], optimizer[i], scheduler[i], scaler, args, print_loss=False)
            loss /= len(train_data[i])
            if local_rank == 0:
                logging.info(f'Stage 2: Epoch {epoch} - Client {i} loss: {loss}')
            dist.barrier()
        dist.barrier()
        # aggregate the model param to get server
        sim_matrix, accumulated_params = aggregate(model)
        loss_list = softmax_with_temperature(loss_list)
        if local_rank == 0:
            logging.info(f'Epoch {epoch} - loss_list: {loss_list}')

        # save server model
        for i in range(len(train_client)):
            warm_weight[i] =  math.tanh(alpha/(loss_list[i]**(epoch+1/beta)))
            if local_rank == 0:
                logging.info(f'Client {i} warm weight: {warm_weight[i]}')
            weight = get_aggregate_lora_weight(i, sim_matrix, accumulated_params, warm_weight[i])
            model[i].load_state_dict(weight)
        dist.barrier()

        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')
            if local_rank == 0:
                logging.info(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics[1][0] > best_target:
                print('Save the best model.')
                best_target = dev_metrics[1][0]
                patient = 3
                for i in range(len(train_client)):
                    path_ckpt = 'checkpoints/' + f'client{i}_' + args.ckpt
                    torch.save(model[i].state_dict(), path_ckpt)
            else:
                patient -= 1
                if patient == 0:
                    break
        dist.barrier()

    print('Test with the best checkpoint.')  
    logging.info('Test with the best checkpoint.')
    for i in range(len(train_client)):
        path_ckpt = 'checkpoints/' + f'client{i}_' + args.ckpt
        model[i].load_state_dict(torch.load(path_ckpt, map_location=args.device))
    dist.barrier()
    test_metrics = eval(model, test_loader, args)
    print(f'Test set: {test_metrics}')
    if local_rank == 0:
        logging.info(f'Test set: {test_metrics}')
               
if __name__ == "__main__":
    main()