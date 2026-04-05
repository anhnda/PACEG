"""
run_eval_reagent_gpt2.py
========================
Benchmark ReAGent attribution on decoder-only GPT-2 using the TellMeWhy
dataset loaded from a local raw-text file.

Mirrors run_eval_pg_gpt2.py exactly so that results (Soft-NC, Soft-NS,
Log-odds) are directly comparable between PACE Gradient and ReAGent.

Dataset format  (datasets2/tellmewhy2.txt)
------------------------------------------
One sample per line.  Format:

    <narrative sentences>  Why did <subject> <verb>?[<TAB><gold answer>]

The full line (left of any tab) is used as the prompt Q.
If --use_gold is set and a tab-separated answer exists it is used as A.

Metrics  (xai_metrics_gpt2.py, same as PACE run)
-------------------------------------------------
    Soft-NC  -- Soft Normalised Comprehensiveness  (Hellinger, Eq.15)
    Soft-NS  -- Soft Normalised Sufficiency        (Hellinger, Eq.14)
    Log-odds -- log-probability drop after hard top-k Q-token masking

Usage
-----
    # Quick smoke test (gold answers, 20 samples, gpt2-small)
    python run_eval_reagent_gpt2.py --num_samples 20 --use_gold --verbose

    # Full evaluation matching the PACE benchmark
    python run_eval_reagent_gpt2.py \\
        --model_name gpt2-medium \\
        --num_samples 200 \\
        --top_k 3 \\
        --use_gold

    # Custom paths
    python run_eval_reagent_gpt2.py \\
        --data_path /data/tellmewhy2.txt \\
        --model_name ./model--gpt2 \\
        --mlm_name  roberta-base
"""

import random
import argparse
import traceback

import numpy as np
import torch
from tqdm import tqdm

from reagent_gpt2 import reagent_gpt2, get_model_tokenizer, _get_mlm
from xai_metrics_gpt2 import calculate_all_metrics_gpt2

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader  (local .txt — identical to run_eval_pg_gpt2.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_tellmewhy_txt(path: str, num_samples: int, use_gold: bool) -> list:
    """
    Load TellMeWhy samples from the local plain-text file.

    Each line is one sample:
        "<narrative> Why did X?"  [TAB  "<gold answer>"]

    Parameters
    ----------
    path        : path to datasets2/tellmewhy2.txt
    num_samples : maximum number of samples to return
    use_gold    : if True and a tab-separated answer exists, use it

    Returns
    -------
    list of dicts:  {'question': str, 'gold_answer': str | None}
    """
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts    = line.split("\t")
            prompt   = parts[0].strip()
            gold_ans = parts[1].strip() if (use_gold and len(parts) > 1) else None
            if not prompt:
                continue
            samples.append({"question": prompt, "gold_answer": gold_ans})

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    print(f"Loaded {len(samples)} samples from {path}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_single_example(
    question:       str,
    gold_answer,
    model_name:     str,
    mlm_name:       str,
    device:         str,
    top_k:          int,
    topk:           int,
    max_new_tokens: int,
    n_samples:      int,
) -> dict:
    """
    Full ReAGent pipeline for one TellMeWhy sample:
        1. reagent_gpt2()           — importance scores via MLM oracle
        2. calculate_all_metrics_gpt2() — Soft-NC, Soft-NS, Log-odds

    Returns dict with soft_nc, soft_ns, log_odds, predicted_answer, time,
    tokens, q_len, attributions  (same keys as run_eval_pg_gpt2.py).
    """
    # ── ReAGent attribution ───────────────────────────────────────────────
    res = reagent_gpt2(
        question=question,
        model_name=model_name,
        mlm_name=mlm_name,
        device=device,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        gold_answer=gold_answer,
    )

    # ── Faithfulness metrics ──────────────────────────────────────────────
    metrics = calculate_all_metrics_gpt2(
        model=res["model"],
        input_embed=res["input_embed"],
        base_embed=res["base_embed"],
        attributions=res["attributions"],
        answer_ids=res["answer_ids"],
        answer_positions=res["answer_positions"],
        topk=topk,
        n_samples=n_samples,
        device=device,
    )

    return {
        "tokens":           res["tokens"],
        "q_len":            res["q_len"],
        "attributions":     res["attributions"],
        "predicted_answer": res["predicted_answer"],
        "time":             res["time"],
        "soft_nc":          metrics["soft_nc"].item(),
        "soft_ns":          metrics["soft_ns"].item(),
        "log_odds":         metrics["log_odds"].item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ReAGent  |  GPT-2  |  TellMeWhy")
    print("=" * 60)
    print(f"Device        : {device}")
    print(f"GPT-2 model   : {args.model_name}")
    print(f"MLM oracle    : {args.mlm_name}")
    print(f"Dataset       : {args.data_path}")
    print(f"Samples       : {args.num_samples}")
    print(f"top_k         : {args.top_k}  (MLM candidates per token)")
    print(f"topk %        : {args.topk}   (metric ablation)")
    print(f"Gold answer   : {args.use_gold}")
    print(f"MC samples    : {args.n_samples}")
    print("=" * 60)

    # Pre-load both models once (cached)
    print("\nLoading GPT-2 ...")
    get_model_tokenizer(args.model_name, device)
    print("Loading RoBERTa MLM oracle ...")
    _get_mlm(args.mlm_name, device)
    print("Models loaded.\n")

    samples = load_tellmewhy_txt(
        args.data_path,
        num_samples=args.num_samples,
        use_gold=args.use_gold,
    )
    if not samples:
        print("No samples found — check --data_path.")
        return

    # Accumulators
    total_soft_nc  = 0.0
    total_soft_ns  = 0.0
    total_log_odds = 0.0
    total_time     = 0.0
    count          = 0
    errors         = 0

    for idx, sample in enumerate(tqdm(samples, desc="ReAGent")):
        try:
            res = run_single_example(
                question       = sample["question"],
                gold_answer    = sample["gold_answer"],
                model_name     = args.model_name,
                mlm_name       = args.mlm_name,
                device         = device,
                top_k          = args.top_k,
                topk           = args.topk,
                max_new_tokens = args.max_new_tokens,
                n_samples      = args.n_samples,
            )

            total_soft_nc  += res["soft_nc"]
            total_soft_ns  += res["soft_ns"]
            total_log_odds += res["log_odds"]
            total_time     += res["time"]
            count          += 1

            if args.verbose and count <= 3:
                _print_sample(sample["question"], res)

            if count % args.print_step == 0:
                _print_running(count, len(samples),
                               total_soft_nc, total_soft_ns,
                               total_log_odds, total_time)

        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"\n[Error sample {idx}]: {str(exc)[:120]}")
                traceback.print_exc()
            continue

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS  —  ReAGent / GPT-2 / TellMeWhy")
    print("=" * 60)
    if count > 0:
        print(f"  Soft-NC  (Comprehensiveness) : {total_soft_nc  / count:.6f}")
        print(f"  Soft-NS  (Sufficiency)       : {total_soft_ns  / count:.6f}")
        print(f"  Log-odds                     : {total_log_odds / count:.6f}")
        print(f"  Avg time / sample            : {total_time     / count:.4f}s")
        print(f"  Successful samples           : {count} / {len(samples)}")
        print(f"  Errors                       : {errors}")
    else:
        print("  No samples processed successfully.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers  (identical style to run_eval_pg_gpt2.py)
# ─────────────────────────────────────────────────────────────────────────────

def _print_running(count, total, snc, sns, lo, t):
    print(f"\n[{count}/{total}] Running averages:")
    print(f"  Soft-NC  : {snc / count:.4f}")
    print(f"  Soft-NS  : {sns / count:.4f}")
    print(f"  Log-odds : {lo  / count:.4f}")
    print(f"  Avg time : {t   / count:.4f}s")


def _print_sample(question: str, res: dict):
    tokens = res["tokens"]
    scores = res["attributions"].tolist()
    q_len  = res["q_len"]

    print(f"\n{'─' * 60}")
    print(f"Q : {question[:120]}")
    print(f"A : {res['predicted_answer']}")

    q_scores = list(zip(tokens[:q_len], scores[:q_len]))
    q_scores.sort(key=lambda x: x[1], reverse=True)
    print("Top-5 Q tokens by ReAGent attribution:")
    for tok, sc in q_scores[:5]:
        print(f"    {tok!r:20s}  {sc:.4f}")

    print(f"Soft-NC={res['soft_nc']:.4f}  "
          f"Soft-NS={res['soft_ns']:.4f}  "
          f"Log-odds={res['log_odds']:.4f}  "
          f"Time={res['time']:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ReAGent Attribution on GPT-2 + TellMeWhy"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="datasets2/tellmewhy2.txt",
        help="Path to local TellMeWhy raw-text file "
             "(default: datasets2/tellmewhy2.txt)"
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2",
        help="GPT-2 variant or local path: gpt2 | gpt2-medium | ./model--gpt2"
    )
    parser.add_argument(
        "--mlm_name", type=str, default="roberta-base",
        help="RoBERTa MLM oracle (default: roberta-base)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Max samples to evaluate (default: 200)"
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="MLM replacement candidates per token (paper default: 3)"
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help="Percentage of top Q-tokens to mask for log-odds (default: 20)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=30,
        help="Max tokens to generate for each answer (default: 30)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Monte-Carlo draws for Soft-NC/NS Bernoulli perturbation "
             "(default: 10)"
    )
    parser.add_argument(
        "--use_gold", action="store_true",
        help="Use tab-separated gold answers from the txt file if available"
    )
    parser.add_argument(
        "--print_step", type=int, default=50,
        help="Print running averages every N samples (default: 50)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print attribution details for first 3 samples"
    )

    args = parser.parse_args()
    run_benchmark(args)