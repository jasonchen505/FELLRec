from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import numpy as np
import torch
import os
import math
import json
import torch
import ipdb
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import pickle
from utils import computeTopNAccuracy, print_results
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
args = parse.parse_args()

path = []
for root, dirs, files in os.walk(args.input_dir):
    for name in files:
            path.append(os.path.join(args.input_dir, name))

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

base_model = " "
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.half()  # seems to fix bugs for some users.

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2


movies = np.load('../../../data/games/id_to_title_map.npy', allow_pickle=True).item()
movie_dict = {value.strip(" "): key for key, value in movies.items()}
movie_names = list(movies.values())

tokenizer.padding_side = "left"
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

movie_embeddings = []
from tqdm import tqdm

model.eval()
for i, id in tqdm(enumerate(batch(torch.arange(len(movie_names)), 4))):
    name = [movie_names[_] for _ in id]
    input = tokenizer(name, return_tensors="pt", padding=True).to(device)
    input_ids = input.input_ids
    attention_mask = input.attention_mask
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    movie_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
movie_embeddings = torch.cat(movie_embeddings, dim=0).cuda()

# save movie_embeddings
torch.save(movie_embeddings, './movie_embeddings.pt')

# load test_client_data.pkl
with open('../test_client_data.pkl', 'rb') as file:
    test_client_data = pickle.load(file)
path = ['../../games_client4.json']
f = open(path[0], 'r')
import json
test_data_all = json.load(f)
f.close()

test_num = [len(client) for client in test_client_data]
client_result = []
for cnt, client in enumerate(test_client_data):
    if cnt == 0:
        test_data = test_data_all[:len(client)]
    else:
        begin = 0
        for i in range(cnt):
            begin += len(test_client_data[i]) 
        end = begin + len(client)
        test_data = test_data_all[begin:end]   
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    text = [_["predict"].strip("\"") for _ in test_data]
    # text = [_["output"].strip("\"") for _ in test_data]
    tokenizer.padding_side = "left"

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    predict_embeddings = []
    from tqdm import tqdm
    for i, batch_input in tqdm(enumerate(batch(text, 4))):
        input = tokenizer(batch_input, return_tensors="pt", padding=True).to(device)
        input_ids = input.input_ids
        attention_mask = input.attention_mask
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    torch.save(predict_embeddings, './predict_embeddings.pt')
    

    dist = torch.cdist(predict_embeddings, movie_embeddings, p=2)
    dist = dist.float()
    ground_truth = []


    # load train_dict.npy
    train_dict = np.load('../../../data/games/training_dict.npy', allow_pickle=True).item()
    valid_dict = np.load('../../../data/games/validation_dict.npy', allow_pickle=True).item()
    # for every user
    for i in range(len(test_data)):
        user_gt = []
        parts = test_data[i]['output'].split('\", \"')
        # print(parts)
        target_item = [part.strip(" ") for part in parts if part.strip(" ") != '' and part.strip(" ") != ',']
        # print(target_item)
        for cnt, item in enumerate(target_item):
            if cnt == 0:
                item = item[1:] 
            if cnt == len(target_item) - 1:
                item = item[:-1]
            item = item.strip(" ")
            _ = movie_dict[item]
            user_gt.append(_)
        user = test_data[i]['user']
        if user in valid_dict and user in train_dict:
            history_item = train_dict[user] + valid_dict[user]
        elif user in valid_dict:
            history_item = valid_dict[user]
        elif user in train_dict:
            history_item = train_dict[user]
        else:
            history_item = []
        dist[i][history_item] = 1e6 
        ground_truth.append(user_gt)

    values, predicted_indices = dist.topk(50, largest=False, sorted=True, dim=-1)
    topk_list = [10, 20]
    results = computeTopNAccuracy(ground_truth, predicted_indices, topk_list)
    print(f'client {cnt}')
    print_results(results)
    client_result.append(results)

# get the overall results according to the test_num
overall_result = []
for metric, value in enumerate(results):
    for k in range(len(results[metric])):
        overall_result.append(np.average([client_result[i][metric][k] for i in range(len(client_result))], weights=test_num))
print('Overall results:')
print(overall_result)            
        


