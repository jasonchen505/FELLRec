import json
import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils import parameters_to_vector
MAX_VAL = 1e4
import logging
from copy import deepcopy
from recformer import RecformerForSeqRec
def read_json(path, as_int=False):
    # load npy
    raw = np.load(path, allow_pickle=True).item()
    if as_int:
        data = dict((int(key), value) for (key, value) in raw.items())
    else:
        data = dict((key, value) for (key, value) in raw.items())
    return data

def read_json_client(path, client_map, as_int=False):
    # load npy
    raw = np.load(path, allow_pickle=True).item()
    data = {}
    for client_idx in range(len(client_map)):
        if client_idx not in data:
            data[client_idx] = {}
        for user in client_map[client_idx]:
            if user in raw:
                data[client_idx][user] = raw[user]
    return data

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class Ranker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, scores):

        predicts = scores
        _, rating_K = torch.topk(predicts, k=50)

        return rating_K
    
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1 
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1              
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                cnt += 1
            # else:
            #     print('OPS')
        precision.append(round(sumForPrecision / cnt, 4))
        recall.append(round(sumForRecall / cnt, 4))
        NDCG.append(round(sumForNdcg / cnt, 4))
        MRR.append(round(sumForMRR / cnt, 4))
        
    return precision, recall, NDCG, MRR

def cluster_clients(model_params, num_clusters):
    # Convert the list of PyTorch tensors to a single NumPy array
    param_1d = []
    # cnt = 0
    for param in model_params:
        # param_1d.append(parameters_to_vector([p for p in param.values()]).unsqueeze(0).detach().cpu().numpy())
        param_1d.append(torch.cat(tuple(p.view(-1).cpu() for p in param.values())).detach().cpu().numpy())
    # calculate the cos similarity between each client
    params_matrix = np.vstack(param_1d)
    similarity_matrix = cosine_similarity(params_matrix)
    normalized_matrix = (similarity_matrix + 1) / 2
    return normalized_matrix
    
def extract_params(model):
    # add "longformer.embeddings.position_ids", "item_embedding.weight"
    param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
    param_dict['module.longformer.embeddings.position_ids'] = model.module.longformer.embeddings.position_ids
    param_dict['module.item_embedding.weight'] = model.module.item_embedding.weight
    # logging.info(f'param_dict_keys: {param_dict.keys()}')
    return param_dict
    # return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad]).detach().cpu().numpy()

def aggregate(model_list):
    """
    aggregate the model of different clients
    """
    accumulated_params = []
    with torch.no_grad():
        for i in range(len(model_list)):
            accumulated_params.append(extract_params(model_list[i]))
            # logging.info('Client {}: {}'.format(i, extract_params(model_list[i])))
    sim_matrix = cluster_clients(accumulated_params, len(model_list))
    return sim_matrix, accumulated_params 

def get_aggregate_lora_weight(client_index, sim_matrix, accumulated_params, weight):
    for i, value in enumerate(sim_matrix[client_index]):
        if i != client_index:
            sim_matrix[client_index][i] = sim_matrix[client_index][i] * weight
    # get the lora weight of each client
    with torch.no_grad():
        for name, param_ in accumulated_params[client_index].items():
            weighted_param = sum(param[name] * sim_matrix[client_index][cnt] for cnt, param in enumerate(accumulated_params)) / sum(sim_matrix[client_index])
            param_.copy_(weighted_param)
    return accumulated_params[client_index]


def merge_models(config, model_front_and_last, model_middle):
    # combing two models from clients and servers

    class MergedModel(RecformerForSeqRec):
        def __init__(self, config, front_and_last, middle):
            super().__init__(config)
            self.longformer = deepcopy(front_and_last.longformer)
            
            front_layers = self.longformer.encoder.layer[:-1]
            
            middle_layers = middle.longformer.encoder.layer
            
            last_layer = self.longformer.encoder.layer[-1:]
            
            self.longformer.encoder.layer = nn.ModuleList(front_layers + middle_layers + last_layer)
            
            self.sim = deepcopy(front_and_last.sim)

    return MergedModel(config, model_front_and_last, model_middle)
