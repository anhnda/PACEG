"""
evaluate_slalom.py

Benchmark SLALOM explanations with the same interface and metrics
as the PACE gradient evaluation scripts (log-odds, comprehensiveness,
sufficiency).

Usage:
    python slalom_eval.py --model distilbert --dataset sst2
    python slalom_eval.py --model bert --dataset imdb --num_samples 500
    python slalom_eval.py --model distilbert --dataset sst2 --attr_mode value
"""

import time
import random
import argparse
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from slalom_explanations import SLALOMLocalExplanantions

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_NAMES = {
    ("distilbert", "sst2"):   "distilbert-base-uncased-finetuned-sst-2-english",
    ("distilbert", "imdb"):   "textattack/distilbert-base-uncased-imdb",
    ("distilbert", "rotten"): "textattack/distilbert-base-uncased-rotten-tomatoes",
    ("bert",       "sst2"):   "textattack/bert-base-uncased-SST-2",
    ("bert",       "imdb"):   "textattack/bert-base-uncased-imdb",
    ("bert",       "rotten"): "textattack/bert-base-uncased-rotten-tomatoes",
    ("roberta",    "sst2"):   "textattack/roberta-base-SST-2",
    ("roberta",    "imdb"):   "textattack/roberta-base-imdb",
    ("roberta",    "rotten"): "textattack/roberta-base-rotten-tomatoes",
}


# ── Output structure detection (run once, cached) ─────────────────────────
_SLALOM_FORMAT = None   # detected lazily on first call

def _detect_and_unpack(res):
    """
    Auto-detect SLALOM output format and return (tokens, values, imps).
    Runs detection once, then uses cached format for all subsequent calls.

    Observed formats across SLALOM versions:
      A) dict  with keys "tokens","value","imp"         → dict, arrays shape (L,) or (L,C)
      B) list  of (token_str, value_vec, imp_vec)       → 3-tuple per token
      C) list  of (token_str, score_vec)                → 2-tuple, score_vec shape (2,C) modes stacked
      D) list  of (token_str, score_scalar)             → 2-tuple, scalar (single mode)
    """
    global _SLALOM_FORMAT

    if _SLALOM_FORMAT is None:
        # ── Detect format ──
        if isinstance(res, dict):
            _SLALOM_FORMAT = "dict"
            print(f"[SLALOM format detected] dict, keys={list(res.keys())}")
        elif isinstance(res, (list, tuple)) and len(res) > 0:
            elem = res[0]
            if isinstance(elem, (list, tuple)):
                n = len(elem)
                if n >= 3:
                    _SLALOM_FORMAT = "list_3tuple"
                elif n == 2:
                    # discriminate: is elem[1] a 2D array (modes stacked) or 1D?
                    v = np.array(elem[1])
                    if v.ndim == 2:
                        _SLALOM_FORMAT = "list_2tuple_stacked"
                    else:
                        _SLALOM_FORMAT = "list_2tuple_single"
                else:
                    raise ValueError(f"Unexpected tuple length {n}: {elem}")
            else:
                raise ValueError(f"Unexpected element type {type(elem)}: {elem}")
        else:
            raise ValueError(f"Unexpected SLALOM output type {type(res)}: {res}")
        print(f"[SLALOM format] {_SLALOM_FORMAT}")

    # ── Unpack according to detected format ──
    def _to_1d(x):
        """(L,) or (L,C) → (L,) by taking class1-class0 for binary, or [0] for scalar."""
        x = np.array(x, dtype=np.float32)
        if x.ndim == 1:
            return x
        elif x.shape[-1] == 1:
            return x[..., 0]
        else:
            return x[..., 1] - x[..., 0]   # signed: positive = favors class 1

    if _SLALOM_FORMAT == "dict":
        tokens = res["tokens"]
        values = _to_1d(np.array(res["value"], dtype=np.float32))
        imps   = _to_1d(np.array(res["imp"],   dtype=np.float32))

    elif _SLALOM_FORMAT == "list_3tuple":
        # (token, value_vec, imp_vec) per token
        tokens = [r[0] for r in res]
        values = _to_1d(np.stack([np.array(r[1], dtype=np.float32) for r in res]))
        imps   = _to_1d(np.stack([np.array(r[2], dtype=np.float32) for r in res]))

    elif _SLALOM_FORMAT == "list_2tuple_stacked":
        # (token, stacked_array) where stacked_array shape (num_modes, num_labels)
        # row 0 = value, row 1 = imp
        tokens = [r[0] for r in res]
        stacked = np.stack([np.array(r[1], dtype=np.float32) for r in res])
        # stacked shape: (L, num_modes, num_labels)
        values = _to_1d(stacked[:, 0, :])
        imps   = _to_1d(stacked[:, 1, :]) if stacked.shape[1] > 1 else np.zeros(len(tokens), dtype=np.float32)

    elif _SLALOM_FORMAT == "list_2tuple_single":
        # (token, score_vec) — only one mode returned
        tokens = [r[0] for r in res]
        values = _to_1d(np.stack([np.array(r[1], dtype=np.float32) for r in res]))
        imps   = np.zeros(len(tokens), dtype=np.float32)

    return tokens, values, imps


# ── Metrics ────────────────────────────────────────────────────────────────
def _get_base_emb(model, tokenizer, device):
    mask_id = tokenizer.mask_token_id or tokenizer.pad_token_id
    with torch.no_grad():
        return model.get_input_embeddings()(
            torch.tensor([[mask_id]], device=device)
        ).squeeze(0)


def _forward_prob(model, embed_input, attention_mask, pred_id, extra_kwargs):
    with torch.no_grad():
        logits = model(
            inputs_embeds=embed_input,
            attention_mask=attention_mask,
            **extra_kwargs
        ).logits[0]
    return F.softmax(logits, dim=-1)[pred_id]

def compute_metrics(model, tokenizer, device, input_ids, attention_mask,
                    extra_kwargs, attr, base_emb, topk=20):
    embed = model.get_input_embeddings()
    with torch.no_grad():
        X       = embed(input_ids)
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask,
                        **extra_kwargs).logits[0]
    pred_id   = int(logits0.argmax().item())
    prob_orig = F.softmax(logits0, dim=-1)[pred_id]
    L         = X.shape[1]

    special_ids = set(tokenizer.all_special_ids)
    fixed = torch.tensor(
        [tid in special_ids for tid in input_ids[0].tolist()],
        device=device, dtype=torch.bool
    )

    attr_rank        = attr.clone().to(device).float()
    attr_rank[fixed] = -float('inf')
    k                = max(1, int((~fixed).sum().item() * topk / 100))
    topk_idx         = torch.topk(attr_rank, k, sorted=False).indices

    # log-odds
    X_lo              = X.clone()
    X_lo[0, topk_idx] = base_emb
    prob_lo           = _forward_prob(model, X_lo, attention_mask, pred_id, extra_kwargs)
    log_odd           = (torch.log(prob_lo + 1e-10) - torch.log(prob_orig + 1e-10)).item()

    # comprehensiveness
    X_comp              = X.clone()
    X_comp[0, topk_idx] = base_emb
    prob_comp           = _forward_prob(model, X_comp, attention_mask, pred_id, extra_kwargs)
    comp                = (prob_orig - prob_comp).item()

    # sufficiency
    keep            = torch.zeros(L, dtype=torch.bool, device=device)
    keep[topk_idx]  = True
    keep[fixed]     = True
    X_suff          = X.clone()
    X_suff[0, ~keep] = base_emb
    prob_suff       = _forward_prob(model, X_suff, attention_mask, pred_id, extra_kwargs)
    suff            = (prob_orig - prob_suff).item()

    return log_odd, comp, suff, pred_id

# ── Single-sample wrapper ──────────────────────────────────────────────────
def slalom_explain_and_eval(
    text, slalom_explainer, model, tokenizer,
    device, extra_kwargs, base_emb, topk=20, attr_mode="lin",
):
    t0  = time.perf_counter()
    raw = slalom_explainer.tokenize_and_explain(text)
    t1  = time.perf_counter()

    tokens_out, values, imps = _detect_and_unpack(raw)

    if attr_mode == "value":
        attr = torch.tensor(values)
    elif attr_mode == "imp":
        attr = torch.tensor(imps)
    else:
        attr = torch.tensor(values * np.exp(np.clip(imps, -20, 20)))

    enc            = tokenizer(text, return_tensors="pt", truncation=True,
                               return_special_tokens_mask=True)
    enc            = {k: v.to(device) for k, v in enc.items()}
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    L              = input_ids.shape[1]

    if attr.shape[0] != L:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx        = [i for i, tid in enumerate(input_ids[0].tolist())
                           if tid not in special_ids_set]
        full_attr       = torch.zeros(L, dtype=torch.float32)
        if len(keep_idx) == attr.shape[0]:
            full_attr[keep_idx] = attr.float()
        attr = full_attr

    log_odd, comp, suff, pred_id = compute_metrics(
        model, tokenizer, device,
        input_ids, attention_mask, extra_kwargs,
        attr, base_emb, topk=topk,
    )

    return {
        "tokens":          tokens_out,
        "value":           values.tolist(),
        "imp":             imps.tolist(),
        "lin":             (values * np.exp(np.clip(imps, -20, 20))).tolist(),
        "predicted_label": pred_id,
        "time":            t1 - t0,
        "log_odd":         log_odd,
        "comp":            comp,
        "suff":            suff,
    }


# ── Benchmark loop ─────────────────────────────────────────────────────────
def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Model         : {args.model} / {args.dataset}")
    print(f"SLALOM mode   : {args.attr_mode}")
    print(f"Top-k         : {args.topk}%")
    print(f"Eval baseline : {args.eval_baseline}")

    model_name = MODEL_NAMES[(args.model, args.dataset)]
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_name
    ).to(device)
    model.eval()

    fwd_params   = inspect.signature(model.forward).parameters
    extra_kwargs = {}

    # Build base_emb once from eval_baseline
    from pace_gradients import get_baseline_embedding
    embed = model.get_input_embeddings()
    with torch.no_grad():
        dummy_ids = torch.tensor([[tokenizer.cls_token_id or 0]], device=device)
        dummy_X   = embed(dummy_ids)   # (1, 1, d)

    base_emb = get_baseline_embedding(
        args.eval_baseline, embed, tokenizer, dummy_X, device
    )[0, 0, :]   # (d,) — compute_metrics uses base_emb directly as a row vector

    slalom_explainer = SLALOMLocalExplanantions(
        model, tokenizer, modes=["value", "imp"]
    )

    if args.dataset == "imdb":
        dataset = load_dataset("imdb")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, min(args.num_samples * 2, len(data)))
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")["validation"]
        data    = list(zip(dataset["sentence"], dataset["label"]))
    elif args.dataset == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))

    if len(data) > args.num_samples:
        data = random.sample(data, args.num_samples)
    print(f"Samples       : {len(data)}")

    total_log_odd = total_comp = total_suff = total_time = 0.0
    count = errors = 0

    for row in tqdm(data):
        text = row[0]
        try:
            res = slalom_explain_and_eval(
                text=text,
                slalom_explainer=slalom_explainer,
                model=model,
                tokenizer=tokenizer,
                device=device,
                extra_kwargs=extra_kwargs,
                base_emb=base_emb,
                topk=args.topk,
                attr_mode=args.attr_mode,
            )
            total_log_odd += res["log_odd"]
            total_comp    += res["comp"]
            total_suff    += res["suff"]
            total_time    += res["time"]
            count         += 1

            if count % args.print_step == 0:
                print(f"\n[{count}/{len(data)}]"
                      f"  log-odds={total_log_odd/count:.4f}"
                      f"  comp={total_comp/count:.4f}"
                      f"  suff={total_suff/count:.4f}"
                      f"  time={total_time/count:.4f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                import traceback; traceback.print_exc()

    if count > 0:
        print(f"\n{'─'*52}")
        print(f"SLALOM ({args.attr_mode})  |  {args.model} / {args.dataset}")
        print(f"  Log-odds         : {total_log_odd/count:.6f}")
        print(f"  Comprehensiveness: {total_comp/count:.6f}")
        print(f"  Sufficiency      : {total_suff/count:.6f}")
        print(f"  Avg time/sample  : {total_time/count:.4f}s")
        print(f"  Evaluated        : {count}  |  Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        choices=["distilbert", "bert", "roberta"],
                        default="distilbert")
    parser.add_argument("--dataset",      choices=["sst2", "imdb", "rotten"],
                        default="sst2")
    parser.add_argument("--num_samples",  type=int, default=1000)
    parser.add_argument("--topk",         type=int, default=20)
    parser.add_argument("--attr_mode",    choices=["value", "imp", "lin"],
                        default="lin")
    parser.add_argument("--print_step",   type=int, default=100)
    parser.add_argument("--eval-baseline", type=str, default="mask",
                        choices=["mask", "pad", "zero", "mean", "random"],
                        help="Baseline embedding used to replace tokens in faithfulness metrics")
    args = parser.parse_args()
    run_benchmark(args)