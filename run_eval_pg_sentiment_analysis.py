# eval_pace.py
import json
import math
import time
from tqdm import tqdm
import torch
import random
import inspect
import argparse
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Literal, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from xai_metrics import *
from pace_gradients import pace_gradient_classification

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

cache = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset",  type=str, choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--steps",    type=int, default=100)
    parser.add_argument("--baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding strategy for PACE integration path")
    args = parser.parse_args()

    a, b         = 0, 1
    steps        = args.steps
    model        = args.model
    dataset_name = args.dataset
    baseline     = args.baseline

    if model == "distilbert":
        from distilbert_helper import *
        if dataset_name == "sst2":
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        elif dataset_name == "imdb":
            model_name = "textattack/distilbert-base-uncased-imdb"
        elif dataset_name == "rotten":
            model_name = "textattack/distilbert-base-uncased-rotten-tomatoes"
    elif model == "bert":
        from bert_helper import *
        if dataset_name == "sst2":
            model_name = "textattack/bert-base-uncased-SST-2"
        elif dataset_name == "imdb":
            model_name = "textattack/bert-base-uncased-imdb"
        elif dataset_name == "rotten":
            model_name = "textattack/bert-base-uncased-rotten-tomatoes"
    elif model == "roberta":
        from roberta_helper import *
        if dataset_name == "sst2":
            model_name = "textattack/roberta-base-SST-2"
        elif dataset_name == "imdb":
            model_name = "textattack/roberta-base-imdb"
        elif dataset_name == "rotten":
            model_name = "textattack/roberta-base-rotten-tomatoes"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device   : {device}")
    print(f"Baseline : {baseline}")
    print(f"Range    : [{a}, {b}]  steps={steps}")

    # Quick smoke test
    text = "This is a really bad movie, although it has a promising start, it ended on a very low note."
    res  = pace_gradient_classification(
        text, a=a, b=b, steps=steps,
        model_name=model_name,
        show_special_tokens=False,
        baseline=baseline,
    )
    for tok, val in zip(res["tokens"], res["attributions"]):
        print(f"{tok:>12s} : {val.item():+.6f}")

    # Dataset
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, 2000)
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")["test"]
        data    = list(zip(dataset["sentence"], dataset["label"], dataset["idx"]))
    elif dataset_name == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))

    log_odds, comps, suffs, count, total_time = 0, 0, 0, 0, 0
    print_step = 100
    print("Starting PACE attribution computation...")

    for row in tqdm(data):
        text = row[0]
        res  = pace_gradient_classification(
            sentence=text, a=a, b=b, steps=steps,
            model_name=model_name,
            show_special_tokens=False,
            baseline=baseline,
        )
        log_odds   += res["log_odd"]
        comps      += res["comp"]
        suffs      += res["suff"]
        total_time += res["time"]
        count      += 1
        if count % print_step == 0:
            print(
                f"[{count}] "
                f"Log-odds: {log_odds/count:.4f}  "
                f"Comp: {comps/count:.4f}  "
                f"Suff: {suffs/count:.4f}  "
                f"Time: {total_time/count:.4f}s"
            )

    print(
        f"\nFinal  "
        f"Log-odds: {log_odds/count:.4f}  "
        f"Comp: {comps/count:.4f}  "
        f"Suff: {suffs/count:.4f}  "
        f"Time: {total_time/count:.4f}s"
    )