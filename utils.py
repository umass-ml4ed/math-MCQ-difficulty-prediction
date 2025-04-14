import xmltodict
import re
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_seeds(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

def seed_worker(worker_id):
    worker_seed = 45  # Replace with your seed number
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_for_LLM(data):
    return (
    data[:int(.65 * len(data))],
    data[int(.65 * len(data)):int(.8 * len(data))],
    data[int(.8 * len(data)):])
    

def count_greater_pairs(lst):
    pair_vals = []
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            if lst[i] > lst[j]:
                pair_vals.append(True)
            else:
                pair_vals.append(False)
    return pair_vals

def calculate_r2(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)

    
def check_correlation(predictions, labels, logits, entropys):
    predictions = np.array(predictions)
    labels = np.array(labels)
    logits = np.array(logits)
    entropys = np.array(entropys)
    # Calculate correlations
    entropy_label_corr = np.corrcoef(entropys, labels)[0, 1]
    first_logit_label_corr = np.corrcoef(logits[:, 0], labels)[0, 1]
    entropy_pred_corr = np.corrcoef(entropys, predictions)[0, 1]
    first_logit_pred_corr = np.corrcoef(logits[:, 0], predictions)[0, 1]
    print(f'Entropy-label correlation: {entropy_label_corr}')
    print(f'First logit-label correlation: {first_logit_label_corr}')
    print(f'Entropy-prediction correlation: {entropy_pred_corr}')
    print(f'First logit-prediction correlation: {first_logit_pred_corr}')
    # print max and min value of predictions
    print(f'Max prediction: {predictions.max()}')
    print(f'Min prediction: {predictions.min()}')
    # print max and min value of labels
    print(f'Max label: {labels.max()}')
    print(f'Min label: {labels.min()}')
    # print max and min value of entropy
    print(f'Max entropy: {entropys.max()}')
    print(f'Min entropy: {entropys.min()}')

    