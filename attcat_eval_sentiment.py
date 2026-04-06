"""
attcat_eval_sentiment.py

Evaluation of AttCAT (Attentive Class Activation Tokens) attributions on
sentiment classification datasets, following the structure of the PACE
gradient evaluation script.

AttCAT paper: "AttCAT: Explaining Transformers via Attentive Class Activation
Tokens", Qiang et al., NeurIPS 2022.
https://github.com/qiangyao1988/AttCAT

AttCAT Algorithm (per token i, summed over all L layers):
  1. CAT_i^l  = grad(h_i^l) ⊙ h_i^l           (Hadamard product, no ReLU)
  2. AttCAT_i^l = mean_over_heads( alpha_i^l @ CAT_i^l )
  3. score_i  = sum_l  sum_d  AttCAT_i^l        (scalar per token)

Metrics (identical to pace_gradients.py):
  calculate_log_odds / calculate_comprehensiveness / calculate_sufficiency
  from xai_metrics, using the same helper functions from *_helper.py.
"""

import time
import tqdm
import torch
import random
import argparse
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from xai_metrics import (
    calculate_log_odds,
    calculate_comprehensiveness,
    calculate_sufficiency,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ---------------------------------------------------------------------------
# Model / tokenizer cache  (same pattern as pace_gradients.py)
# ---------------------------------------------------------------------------
cache = {}


def _get_cached(model_name: str, device: str):
    if model_name not in cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
        ).to(device)
        model.eval()
        cache[model_name] = {"model": model, "tokenizer": tokenizer}
    return cache[model_name]["model"], cache[model_name]["tokenizer"]


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def _get_encoder_layers(model):
    if hasattr(model, "bert"):
        return list(model.bert.encoder.layer)
    if hasattr(model, "distilbert"):
        return list(model.distilbert.transformer.layer)
    if hasattr(model, "roberta"):
        return list(model.roberta.encoder.layer)
    raise RuntimeError("Unsupported model architecture.")


def _get_attn_submodule(layer):
    if hasattr(layer, "attention"):
        return layer.attention
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    return None


# ---------------------------------------------------------------------------
# Core AttCAT computation
# ---------------------------------------------------------------------------
def attcat_classification(
    sentence: str,
    model_name: str,
    show_special_tokens: bool = False,
    device: str = "cpu",
) -> Dict:
    t0 = time.perf_counter()
    model, tokenizer = _get_cached(model_name, device)

    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, nn_forward_func
    else:
        raise NotImplementedError(f"No helper for {model_name}")

    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len        = input_ids.shape[1]

    hidden_states_list: List[torch.Tensor] = []
    attn_weights_list:  List[torch.Tensor] = []
    hooks = []

    encoder_layers = _get_encoder_layers(model)

    def make_layer_hook(idx: int):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                h = None
                for t in reversed(out):
                    if isinstance(t, torch.Tensor) and t.dim() == 3:
                        h = t
                        break
                if h is None:
                    h = out[0]
            else:
                h = out
            hidden_states_list.append(h)
        return fn

    def make_attn_hook(idx: int):
        def fn(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                if out[1].dim() == 4:
                    attn_weights_list.append(out[1].detach())
        return fn

    for idx, layer in enumerate(encoder_layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
        attn_mod = _get_attn_submodule(layer)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(make_attn_hook(idx)))

    with torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    logits     = outputs.logits
    pred_class = int(logits.argmax(dim=-1).item())
    target     = logits[0, pred_class]

    if len(hidden_states_list) == 0 and outputs.hidden_states is not None:
        hidden_states_list = list(outputs.hidden_states[1:])
    if len(attn_weights_list) == 0 and outputs.attentions is not None:
        attn_weights_list = [a.detach() for a in outputs.attentions if a is not None]

    n_layers = len(hidden_states_list)
    attcat_scores = torch.zeros(seq_len, device=device)

    for l_idx in range(n_layers):
        h_l = hidden_states_list[l_idx]
        try:
            (grad_h_l,) = torch.autograd.grad(
                target, h_l,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
        except RuntimeError:
            continue
        if grad_h_l is None:
            continue

        cat_l = (grad_h_l * h_l.detach()).squeeze(0)   # [seq, d]

        if l_idx < len(attn_weights_list):
            alpha_l  = attn_weights_list[l_idx].squeeze(0)
            attcat_l = torch.einsum("hij,jd->hid", alpha_l, cat_l).mean(dim=0)
        else:
            attcat_l = cat_l

        attcat_scores = attcat_scores + attcat_l.sum(dim=-1)

    tokens_raw      = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    special_ids_set = set(tokenizer.all_special_ids)

    if show_special_tokens:
        tokens = tokens_raw
        attr   = attcat_scores
    else:
        keep   = [i for i, tid in enumerate(input_ids[0].tolist())
                  if tid not in special_ids_set]
        tokens = [tokens_raw[i] for i in keep]
        attr   = attcat_scores[keep]

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)   # [1, seq, d]

    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    return {
        "tokens":          tokens,
        "attributions":    attr.detach().cpu(),
        "pred_class":      pred_class,
        "time":            time.perf_counter() - t0,
        # raw tensors for eval script
        "model":           model,
        "nn_forward_func": nn_forward_func,
        "input_embed":     X,
        "attention_mask":  attention_mask,
        "position_embed":  position_embed,
        "type_embed":      type_embed,
        "attr_full":       attcat_scores.detach(),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AttCAT attributions on sentiment datasets."
    )
    parser.add_argument("--model",         type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset",       type=str, required=True,
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--n_samples",     type=int, default=2000)
    parser.add_argument("--print_step",    type=int, default=100)
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding used to replace tokens in faithfulness metrics")
    args = parser.parse_args()

    MODEL_MAP = {
        "distilbert": {
            "sst2":   "distilbert-base-uncased-finetuned-sst-2-english",
            "imdb":   "textattack/distilbert-base-uncased-imdb",
            "rotten": "textattack/distilbert-base-uncased-rotten-tomatoes",
        },
        "bert": {
            "sst2":   "textattack/bert-base-uncased-SST-2",
            "imdb":   "textattack/bert-base-uncased-imdb",
            "rotten": "textattack/bert-base-uncased-rotten-tomatoes",
        },
        "roberta": {
            "sst2":   "textattack/roberta-base-SST-2",
            "imdb":   "textattack/roberta-base-imdb",
            "rotten": "textattack/roberta-base-rotten-tomatoes",
        },
    }
    model_name    = MODEL_MAP[args.model][args.dataset]
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    eval_baseline = args.eval_baseline

    print(f"Model         : {model_name}")
    print(f"Dataset       : {args.dataset}")
    print(f"Device        : {device}")
    print(f"Eval baseline : {eval_baseline}")

    # Build eval_base_token_emb once
    from pace_gradients import get_baseline_embedding
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    eval_model.eval()
    embed = eval_model.get_input_embeddings()

    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, d)

    eval_base_token_emb = get_baseline_embedding(
        eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0:1, :]   # (1, d)

    # Demo
    demo_text = (
        "This is a really bad movie, although it has a promising start, "
        "it ended on a very low note."
    )
    print("\n--- AttCAT demo attribution ---")
    res_demo = attcat_classification(
        demo_text, model_name=model_name,
        show_special_tokens=False, device=device,
    )
    for tok, val in zip(res_demo["tokens"], res_demo["attributions"]):
        print(f"  {tok:>15s} : {val.item():+.6f}")

    # Dataset
    print("\nLoading dataset ...")
    if args.dataset == "imdb":
        ds   = load_dataset("imdb")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(args.n_samples, len(data)))
    elif args.dataset == "sst2":
        ds   = load_dataset("glue", "sst2")["validation"]
        data = list(zip(ds["sentence"], ds["label"]))
    elif args.dataset == "rotten":
        ds   = load_dataset("rotten_tomatoes")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(args.n_samples, len(data)))

    print(f"Evaluating {len(data)} samples with AttCAT ...\n")

    log_odds_sum = comps_sum = suffs_sum = total_time_sum = 0.0
    count = 0

    for row in tqdm.tqdm(data):
        text = row[0]
        try:
            res = attcat_classification(
                text, model_name=model_name,
                show_special_tokens=False, device=device,
            )
            log_odd, _ = calculate_log_odds(
                res["nn_forward_func"], res["model"],
                res["input_embed"], res["position_embed"], res["type_embed"],
                res["attention_mask"], eval_base_token_emb,
                res["attr_full"], topk=20,
            )
            comp = calculate_comprehensiveness(
                res["nn_forward_func"], res["model"],
                res["input_embed"], res["position_embed"], res["type_embed"],
                res["attention_mask"], eval_base_token_emb,
                res["attr_full"], topk=20,
            )
            suff = calculate_sufficiency(
                res["nn_forward_func"], res["model"],
                res["input_embed"], res["position_embed"], res["type_embed"],
                res["attention_mask"], eval_base_token_emb,
                res["attr_full"], topk=20,
            )
            log_odds_sum   += log_odd
            comps_sum      += comp
            suffs_sum      += suff
            total_time_sum += res["time"]
            count += 1
        except Exception as e:
            print(f"[WARN] skipped sample: {e}")
            continue

        if count % args.print_step == 0:
            print(
                f"[{count:>5d}]  "
                f"Log-odds: {log_odds_sum/count:.4f}  "
                f"Comp: {comps_sum/count:.4f}  "
                f"Suff: {suffs_sum/count:.4f}  "
                f"Time/sample: {total_time_sum/count:.4f}s"
            )

    print("\n=== Final Results ===")
    n = max(count, 1)
    print(
        f"Log-odds         : {log_odds_sum/n:.4f}\n"
        f"Comprehensiveness: {comps_sum/n:.4f}\n"
        f"Sufficiency      : {suffs_sum/n:.4f}\n"
        f"Time/sample      : {total_time_sum/n:.4f}s\n"
        f"Total samples    : {count}"
    )