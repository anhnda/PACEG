import json
import math
import time
import tqdm
import torch
import random
import inspect
import argparse
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Literal, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from captum.attr._utils.common import _reshape_and_sum, _validate_input
from xai_metrics import *
from pace_gradients import pace_gradient_classification
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
cache = {}
cache_ = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert', help='Model name or path')
    parser.add_argument('--dataset', choices=['sst2', 'imdb', 'rotten'])
    parser.add_argument('--steps', type=int, default=100, help='Number of steps for Riemann sum')

    args = parser.parse_args()
    a, b = 0, 1
    steps = args.steps
    model = args.model
    dataset_name = args.dataset
    if model == 'distilbert':
        from distilbert_helper import *
        if dataset_name == 'sst2':
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        if dataset_name == 'imdb':
            model_name = "textattack/distilbert-base-uncased-imdb"
        elif dataset_name == 'rotten':
            model_name = "textattack/distilbert-base-uncased-rotten-tomatoes"
    elif model == 'bert':
        from bert_helper import *
        if dataset_name == 'sst2':
            model_name = "textattack/bert-base-uncased-SST-2"
        elif dataset_name == 'imdb':
            model_name = "textattack/bert-base-uncased-imdb"
        elif dataset_name == 'rotten':
            model_name = "textattack/bert-base-uncased-rotten-tomatoes"
    elif model == 'roberta':
        from roberta_helper import *
        if dataset_name == 'sst2':
            model_name = "textattack/roberta-base-SST-2"
        elif dataset_name == 'imdb':
            model_name = "textattack/roberta-base-imdb"
        elif dataset_name == 'rotten':
            model_name = "textattack/roberta-base-rotten-tomatoes"
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using range [{a}, {b}] for epsilon with {steps} steps")
    text = "This is a really bad movie, although it has a promising start, it ended on a very low note."
    res_pace = pace_gradient_classification(text, a = a, b = b, steps = steps, model_name = model_name, show_special_tokens = False)
    for tok, val in zip(res_pace["tokens"], res_pace["attributions"]):
        print(f"{tok:>12s} : {val.detach().cpu().numpy():+.6f}")
    if args.dataset == 'imdb':
        dataset	= load_dataset('imdb')['test']
        data	= list(zip(dataset['text'], dataset['label']))
        data	= random.sample(data, 2000)
    elif args.dataset == 'sst2':
        dataset	= load_dataset('glue', 'sst2')['test']
        data	= list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
    elif args.dataset == 'rotten':
        dataset	= load_dataset('rotten_tomatoes')['test']
        data	= list(zip(dataset['text'], dataset['label']))
    count = 0
    print('Starting PACE Gradient attribution computation...')
    log_odds, comps, suffs, count, total_time = 0, 0, 0, 0, 0
    print_step = 100
    for row in tqdm(data):
        text = row[0]
        res_pace = pace_gradient_classification(sentence = text, a = a, b = b, steps = steps, model_name = model_name, show_special_tokens = False)
        log_odds += res_pace['log_odd']
        comps += res_pace['comp']
        suffs += res_pace['suff']
        total_time += res_pace['time']
        count += 1
        if count % print_step == 0:
            print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
                'Sufficiency: ', np.round(suffs / count, 4), "Time: ", np.round(total_time / count, 4))
            
    print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
        'Sufficiency: ', np.round(suffs / count, 4), "Time: ", np.round(total_time / count, 4))