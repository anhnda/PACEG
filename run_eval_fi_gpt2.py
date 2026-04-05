"""
run_eval_fi_gpt2.py
===================
Benchmark Functional Information (FI) Attribution on decoder-only GPT-2
using the TellMeWhy dataset loaded from a local raw-text file.

Mirrors run_eval_pg_gpt2.py in structure so metrics are directly comparable.

Methods benchmarked
-------------------
    fi           -- standard FI:       E[(∇f)² / f]   (our main method)
    fi_cov       -- covariance FI:     E[(Σ∇f)·∇f / f]
    smooth_grad  -- SmoothGrad:        E[∇f]
    smooth_grad_sq -- SmoothGradSQ:    E[(∇f)²]

Usage
-----
    python run_eval_fi_gpt2.py --method fi --num_samples 200 --n 20
    python run_eval_fi_gpt2.py --method fi_cov --num_samples 100 --n 10
    python run_eval_fi_gpt2.py --method fi --use_gold --verbose
    python run_eval_fi_gpt2.py --method all --num_samples 50
"""

import time
import random
import argparse
import traceback

import numpy as np
import torch
from tqdm import tqdm

from fi_gpt2 import fi_gradient_gpt2, get_model_tokenizer
from xai_metrics_gpt2 import calculate_all_metrics_gpt2

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

ALL_METHODS = ["fi", "fi_cov", "smooth_grad", "smooth_grad_sq"]


# ---------------------------------------------------------------------------
# Dataset loader  (identical to run_eval_pg_gpt2.py)
# ---------------------------------------------------------------------------

def load_tellmewhy_txt(path: str, num_samples: int, use_gold: bool) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts   = line.split("\t")
            full_text = parts[0].strip()
            gold_ans  = parts[1].strip() if (use_gold and len(parts) > 1) else None
            if not full_text:
                continue
            samples.append({"question": full_text, "gold_answer": gold_ans})

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    print(f"Loaded {len(samples)} samples from {path}")
    return samples


# ---------------------------------------------------------------------------
# Single-sample pipeline
# ---------------------------------------------------------------------------

def run_single_example(
    question: str,
    gold_answer,
    model_name: str,
    device: str,
    method: str,
    n: int,
    var_spread: float,
    topk: int,
    max_new_tokens: int,
    mc_samples: int,
) -> dict:
    """
    Full pipeline for one sample:
        1. FI attribution   (fi_gpt2.py)
        2. Faithfulness metrics (xai_metrics_gpt2.py)
    """
    res = fi_gradient_gpt2(
        question=question,
        model_name=model_name,
        device=device,
        n=n,
        var_spread=var_spread,
        max_new_tokens=max_new_tokens,
        gold_answer=gold_answer,
        method=method,
    )

    metrics = calculate_all_metrics_gpt2(
        model=res["model"],
        input_embed=res["input_embed"],
        base_embed=res["base_embed"],
        attributions=res["attributions"],
        answer_ids=res["answer_ids"],
        answer_positions=res["answer_positions"],
        topk=topk,
        n_samples=mc_samples,
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


# ---------------------------------------------------------------------------
# Benchmark loop for a single method
# ---------------------------------------------------------------------------

def run_benchmark_single(args, method: str, samples: list) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_soft_nc  = 0.0
    total_soft_ns  = 0.0
    total_log_odds = 0.0
    total_time     = 0.0
    count, errors  = 0, 0

    for idx, sample in enumerate(tqdm(samples, desc=f"[{method}]")):
        try:
            res = run_single_example(
                question       = sample["question"],
                gold_answer    = sample["gold_answer"],
                model_name     = args.model_name,
                device         = device,
                method         = method,
                n              = args.n,
                var_spread     = args.var_spread,
                topk           = args.topk,
                max_new_tokens = args.max_new_tokens,
                mc_samples     = args.mc_samples,
            )

            total_soft_nc  += res["soft_nc"]
            total_soft_ns  += res["soft_ns"]
            total_log_odds += res["log_odds"]
            total_time     += res["time"]
            count          += 1

            if args.verbose and count <= 3:
                _print_sample(sample["question"], res)

            if count % args.print_step == 0:
                _print_running(method, count, len(samples),
                               total_soft_nc, total_soft_ns,
                               total_log_odds, total_time)

        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"\n[Error sample {idx}]: {str(exc)[:120]}")
                traceback.print_exc()

    return {
        "method":    method,
        "count":     count,
        "errors":    errors,
        "soft_nc":   total_soft_nc  / max(count, 1),
        "soft_ns":   total_soft_ns  / max(count, 1),
        "log_odds":  total_log_odds / max(count, 1),
        "avg_time":  total_time     / max(count, 1),
    }


def run_benchmark(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ALL_METHODS if args.method == "all" else [args.method]

    print(f"Device      : {device}")
    print(f"Model       : {args.model_name}")
    print(f"Methods     : {methods}")
    print(f"Samples     : {args.num_samples}")
    print(f"MC draws n  : {args.n}")
    print(f"var_spread  : {args.var_spread}")

    print("\nLoading model ...")
    get_model_tokenizer(args.model_name, device)
    print("Model loaded.\n")

    samples = load_tellmewhy_txt(args.data_path, args.num_samples, args.use_gold)
    if not samples:
        print("No samples found — check --data_path.")
        return

    all_results = []
    for method in methods:
        res = run_benchmark_single(args, method, samples)
        all_results.append(res)

    # Final comparison table
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} {'Soft-NC':>10} {'Soft-NS':>10} {'Log-odds':>10} {'Time(s)':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['method']:<20} {r['soft_nc']:>10.6f} {r['soft_ns']:>10.6f}"
              f" {r['log_odds']:>10.6f} {r['avg_time']:>10.4f}")
        print(f"  Successful: {r['count']} / {len(samples)}    Errors: {r['errors']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_running(method, count, total, snc, sns, lo, t):
    print(f"\n[{method}] [{count}/{total}]  "
          f"Soft-NC={snc/count:.4f}  Soft-NS={sns/count:.4f}  "
          f"Log-odds={lo/count:.4f}  AvgTime={t/count:.4f}s")


def _print_sample(question: str, res: dict):
    tokens  = res["tokens"]
    scores  = res["attributions"].tolist()
    q_len   = res["q_len"]

    print(f"\n{'─' * 60}")
    print(f"Q : {question[:120]}")
    print(f"A : {res['predicted_answer']}")

    q_scores = sorted(zip(tokens[:q_len], scores[:q_len]), key=lambda x: x[1], reverse=True)
    print("Top-5 Q tokens by FI attribution:")
    for tok, sc in q_scores[:5]:
        print(f"    {tok!r:20s}  {sc:.6f}")

    print(f"Soft-NC={res['soft_nc']:.4f}  "
          f"Soft-NS={res['soft_ns']:.4f}  "
          f"Log-odds={res['log_odds']:.4f}  "
          f"Time={res['time']:.2f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Functional Information Attribution on GPT-2 + TellMeWhy"
    )
    parser.add_argument("--data_path",      type=str, default="datasets2/tellmewhy2.txt")
    parser.add_argument("--model_name",     type=str, default="gpt2",
                        help="GPT-2 variant: gpt2 | gpt2-medium | gpt2-large | gpt2-xl")
    parser.add_argument("--method",         type=str, default="fi",
                        choices=ALL_METHODS + ["all"],
                        help="Attribution method (default: fi)")
    parser.add_argument("--num_samples",    type=int, default=200)
    parser.add_argument("--n",              type=int, default=20,
                        help="Monte-Carlo perturbation draws (default: 20)")
    parser.add_argument("--var_spread",     type=float, default=0.15,
                        help="Variance multiplier for noise (default: 0.15)")
    parser.add_argument("--topk",           type=int, default=20,
                        help="Percent of top Q-tokens masked for log-odds (default: 20)")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--mc_samples",     type=int, default=10,
                        help="Monte-Carlo draws for soft Bernoulli metric (default: 10)")
    parser.add_argument("--use_gold",       action="store_true")
    parser.add_argument("--print_step",     type=int, default=50)
    parser.add_argument("--verbose",        action="store_true")

    args = parser.parse_args()
    run_benchmark(args)