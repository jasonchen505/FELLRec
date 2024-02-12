import numpy as np
import torch
import utils
import math
from pprint import pprint
from time import time
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import roc_auc_score

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

def print_results(test_result):
    """output the evaluation results."""
    print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                        '-'.join([str(x) for x in test_result[0]]), 
                        '-'.join([str(x) for x in test_result[1]]), 
                        '-'.join([str(x) for x in test_result[2]]), 
                        '-'.join([str(x) for x in test_result[3]])))