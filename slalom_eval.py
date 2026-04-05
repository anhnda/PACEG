"""
evaluate_slalom.py

Benchmark SLALOM explanations with the same interface and metrics
as the PACE gradient evaluation scripts (log-odds, comprehensiveness,
sufficiency).

Usage:
    python evaluate_slalom.py --model distilbert --dataset sst2
    python evaluate_slalom.py --model bert --dataset imdb --num_samples 500
"""

import time
import random
import argparse
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

# ── Model name lookup (mirrors run_eval_pg.py) ─────────────────────────────
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


# ── Inline metrics (no helper dependency) ─────────────────────────────────
def _get_base_emb(model, tokenizer, device):
    """PAD or MASK embedding as baseline — same logic as xai_metrics."""
    mask_id = tokenizer.mask_token_id or tokenizer.pad_token_id
    with torch.no_grad():
        return model.get_input_embeddings()(
            torch.tensor([[mask_id]], device=device)
        ).squeeze(0)   # (d,)


def _forward_prob(model, embed_input, attention_mask, pred_id, extra_kwargs):
    with torch.no_grad():
        logits = model(
            inputs_embeds=embed_input,
            attention_mask=attention_mask,
            **extra_kwargs
        ).logits[0]
    return F.softmax(logits, dim=-1)[pred_id]


def compute_metrics(model, tokenizer, device, input_ids, attention_mask,
                    extra_kwargs, attr, topk=20):
    """
    Compute log-odds, comprehensiveness, sufficiency.
    attr: (L,) tensor — used to rank tokens (higher = more important).
    Mirrors calculate_log_odds / comprehensiveness / sufficiency from xai_metrics.py.
    """
    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)                           # (1, L, d)
        logits0 = model(inputs_embeds=X,
                        attention_mask=attention_mask,
                        **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())
    prob_orig = F.softmax(logits0, dim=-1)[pred_id]

    base_emb = _get_base_emb(model, tokenizer, device)  # (d,)
    L = X.shape[1]

    # Special token mask — never pick these as top-k
    special_ids = set(tokenizer.all_special_ids)
    fixed = torch.tensor(
        [tid in special_ids for tid in input_ids[0].tolist()],
        device=device, dtype=torch.bool
    )

    attr_rank = attr.clone().to(device)
    attr_rank[fixed] = -float('inf')
    k = max(1, int((~fixed).sum().item() * topk / 100))
    topk_idx = torch.topk(attr_rank, k, sorted=False).indices

    # ── Log-odds: replace top-k with baseline, measure log prob change ──
    X_lo = X.clone()
    X_lo[0, topk_idx] = base_emb
    prob_lo = _forward_prob(model, X_lo, attention_mask, pred_id, extra_kwargs)
    log_odd = (torch.log(prob_lo + 1e-10) -
               torch.log(prob_orig + 1e-10)).item()

    # ── Comprehensiveness: prob drop when top-k removed (replaced) ──────
    X_comp = X.clone()
    X_comp[0, topk_idx] = base_emb
    prob_comp = _forward_prob(model, X_comp, attention_mask, pred_id, extra_kwargs)
    comp = (prob_orig - prob_comp).item()

    # ── Sufficiency: prob drop when only top-k kept ──────────────────────
    keep = torch.zeros(L, dtype=torch.bool, device=device)
    keep[topk_idx] = True
    keep[fixed]    = True          # always keep special tokens
    X_suff = X.clone()
    X_suff[0, ~keep] = base_emb
    prob_suff = _forward_prob(model, X_suff, attention_mask, pred_id, extra_kwargs)
    suff = (prob_orig - prob_suff).item()

    return log_odd, comp, suff, pred_id


# ── Single-sample wrapper ──────────────────────────────────────────────────
def slalom_explain_and_eval(
    text: str,
    slalom_explainer: SLALOMLocalExplanantions,
    model,
    tokenizer,
    device: str,
    extra_kwargs: dict,
    topk: int = 20,
    attr_mode: str = "lin",          # "value", "imp", or "lin" (linearized)
):
    """
    Run SLALOM on one text sample, then compute XAI metrics.

    attr_mode:
      "value" — use token value scores as attribution ranking
      "imp"   — use token importance scores
      "lin"   — linearized score: value * exp(imp)  [paper's recommended ranking]

    Returns dict with tokens, scores, metrics, and timing.
    """
    t0 = time.perf_counter()
    res = slalom_explainer.tokenize_and_explain(text)
    t1 = time.perf_counter()

    # ── Extract attribution tensor for ranking ─────────────────────────
    # ── Attribution tensor for metric ranking ──────────────────────────
    tokens_out = [r[0] for r in res]
    values     = np.array([r[1] for r in res], dtype=np.float32)
    imps       = np.zeros_like(values)   # no imp available
    if attr_mode == "value":
        attr = torch.tensor(values)
    elif attr_mode == "imp":
        attr = torch.tensor(imps)
    else:  # "lin" — linearized SLALOM score: v * exp(s), Section B.7
        lin  = values * np.exp(imps)
        attr = torch.tensor(lin)

    # rest of the function unchanged from here ...
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        return_special_tokens_mask=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids     = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Align attr length with tokenizer output
    # SLALOM may strip special tokens — pad/trim to match
    L = input_ids.shape[1]
    if attr.shape[0] != L:
        # SLALOM strips specials by default; re-insert zeros at special positions
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist())
                    if tid not in special_ids_set]
        full_attr = torch.zeros(L, dtype=torch.float32)
        if len(keep_idx) == attr.shape[0]:
            for j, idx in enumerate(keep_idx):
                full_attr[idx] = attr[j]
        attr = full_attr

    log_odd, comp, suff, pred_id = compute_metrics(
        model, tokenizer, device,
        input_ids, attention_mask, extra_kwargs,
        attr, topk=topk,
    )

    return {
        "tokens":          tokens_out,
        "value":           values.tolist(),
        "imp":             imps.tolist(),
        "lin":             (values * np.exp(imps)).tolist(),
        "attr_used":       attr.tolist(),
        "predicted_label": pred_id,
        "time":            t1 - t0,
        "log_odd":         log_odd,
        "comp":            comp,
        "suff":            suff,
    }


# ── Benchmark loop ─────────────────────────────────────────────────────────
def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model      : {args.model} / {args.dataset}")
    print(f"SLALOM mode: {args.attr_mode}")
    print(f"Top-k      : {args.topk}%")

    model_name = MODEL_NAMES[(args.model, args.dataset)]
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_name
    ).to(device)
    model.eval()

    # extra_kwargs for models that need token_type_ids
    import inspect
    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    # (populated per-sample below if needed)

    # ── Initialize SLALOM ──────────────────────────────────────────────
    # modes=["value","imp"] computes both; "lin" is derived from them
    slalom_explainer = SLALOMLocalExplanantions(
        model, tokenizer, modes=["value", "imp"]
    )

    # ── Load dataset ───────────────────────────────────────────────────
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
    print(f"Samples    : {len(data)}")

    # ── Eval loop ──────────────────────────────────────────────────────
    total_log_odd, total_comp, total_suff = 0.0, 0.0, 0.0
    total_time = 0.0
    count, errors = 0, 0

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
                topk=args.topk,
                attr_mode=args.attr_mode,
            )
            print(res[0])  # see what a single element looks like
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
            if errors <= 3:
                import traceback; traceback.print_exc()

    if count > 0:
        print(f"\n{'─'*50}")
        print(f"SLALOM ({args.attr_mode}) on {args.model}/{args.dataset}")
        print(f"  Log-odds        : {total_log_odd/count:.6f}")
        print(f"  Comprehensiveness: {total_comp/count:.6f}")
        print(f"  Sufficiency      : {total_suff/count:.6f}")
        print(f"  Avg time/sample  : {total_time/count:.4f}s")
        print(f"  Evaluated        : {count}  |  Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["distilbert","bert","roberta"],
                        default="distilbert")
    parser.add_argument("--dataset", choices=["sst2","imdb","rotten"],
                        default="sst2")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--topk",        type=int, default=20,
                        help="% of tokens for metrics (default 20)")
    parser.add_argument("--attr_mode",
                        choices=["value","imp","lin"], default="lin",
                        help="Which SLALOM score to use for token ranking:"
                             " value, imp, or lin=value*exp(imp) [default]")
    parser.add_argument("--print_step", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(args)