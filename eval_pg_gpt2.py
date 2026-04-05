"""
run_eval_pg_gpt2.py

Benchmark PACE Gradient Attribution on decoder-only GPT-2 models
using the TellMeWhy dataset.

Metrics (following ReAGent, Zhao & Shan 2024):
    • Soft-NC  — Soft Normalised Comprehensiveness  (Hellinger-based)
    • Soft-NS  — Soft Normalised Sufficiency        (Hellinger-based)
    • Log-odds — token-level log-probability drop after masking top-k Q tokens

Usage:
    python run_eval_pg_gpt2.py --model_name gpt2 --num_samples 200 --steps 100
    python run_eval_pg_gpt2.py --model_name gpt2-medium --num_samples 200
"""

import time
import random
import argparse
import traceback

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

from paceg_gpt2 import pace_gradient_gpt2, get_model_tokenizer
from xai_metrics_gpt2 import calculate_all_metrics_gpt2

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_single_example(
    question: str,
    gold_answer: str | None,
    model_name: str,
    device: str,
    steps: int,
    topk: int,
    max_new_tokens: int,
    n_samples: int,
) -> dict:
    """
    Full pipeline for one TellMeWhy sample:
        1. PACE gradient attribution  (paceg_gpt2.py)
        2. Faithfulness metrics       (xai_metrics_gpt2.py)

    Args:
        question       : Narrative + "Why did …?" string.
        gold_answer    : If not None, use gold instead of generating.
        model_name     : GPT-2 variant.
        device         : 'cuda' or 'cpu'.
        steps          : Riemann-sum steps for PACE integration.
        topk           : Percentage of Q-tokens masked for log-odds.
        max_new_tokens : Max generation length when not using gold answer.
        n_samples      : Monte-Carlo draws for soft Bernoulli perturbation.

    Returns:
        dict with attribution results + all three metrics.
    """
    # ── PACE attribution ──────────────────────────────────────────────────
    res = pace_gradient_gpt2(
        question=question,
        model_name=model_name,
        device=device,
        steps=steps,
        max_new_tokens=max_new_tokens,
        gold_answer=gold_answer,
    )

    model        = res["model"]
    input_embed  = res["input_embed"]
    base_embed   = res["base_embed"]
    attributions = res["attributions"]
    answer_ids   = res["answer_ids"]
    answer_positions = res["answer_positions"]

    # ── Faithfulness metrics ──────────────────────────────────────────────
    metrics = calculate_all_metrics_gpt2(
        model=model,
        input_embed=input_embed,
        base_embed=base_embed,
        attributions=attributions,
        answer_ids=answer_ids,
        answer_positions=answer_positions,
        topk=topk,
        n_samples=n_samples,
    )

    return {
        "tokens":           res["tokens"],
        "q_len":            res["q_len"],
        "attributions":     attributions,
        "predicted_answer": res["predicted_answer"],
        "time":             res["time"],
        "soft_nc":          metrics["soft_nc"].item(),
        "soft_ns":          metrics["soft_ns"].item(),
        "log_odds":         metrics["log_odds"].item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tellmewhy(num_samples: int, use_gold: bool) -> list[dict]:
    """
    Load TellMeWhy validation split and build question strings.

    TellMeWhy format:
        story     : narrative paragraph
        question  : "Why did …?" question
        answer    : gold answer string (possibly multi-sentence)

    We concatenate story + question into a single prompt, following
    the format used by ReAGent (Zhao & Shan 2024, Table 1).

    Returns:
        List of dicts: {'question': str, 'gold_answer': str | None}
    """
    print("Loading TellMeWhy dataset …")
    dataset = load_dataset("lal-nlp/tellmewhy", split="validation")

    samples = []
    for item in dataset:
        narrative = item.get("story", item.get("narrative", "")).strip()
        question  = item.get("question", "").strip()
        answer    = item.get("answer",   item.get("answers", "")).strip()

        if not narrative or not question:
            continue

        # Format matching ReAGent Table 1 example
        prompt = f"{narrative} {question}"

        samples.append({
            "question":    prompt,
            "gold_answer": answer if (use_gold and answer) else None,
        })

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    print(f"Loaded {len(samples)} TellMeWhy samples.")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device      : {device}")
    print(f"Model       : {args.model_name}")
    print(f"Samples     : {args.num_samples}")
    print(f"Steps       : {args.steps}")
    print(f"Top-k %     : {args.topk}")
    print(f"Gold answer : {args.use_gold}")
    print(f"MC samples  : {args.n_samples}")

    # Pre-load model once
    print("\nLoading model …")
    get_model_tokenizer(args.model_name, device)
    print("Model loaded.\n")

    samples = load_tellmewhy(args.num_samples, use_gold=args.use_gold)

    # Accumulators
    total_soft_nc  = 0.0
    total_soft_ns  = 0.0
    total_log_odds = 0.0
    total_time     = 0.0
    count          = 0
    errors         = 0

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            res = run_single_example(
                question       = sample["question"],
                gold_answer    = sample["gold_answer"],
                model_name     = args.model_name,
                device         = device,
                steps          = args.steps,
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

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
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
# Printing helpers
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

    print(f"\n{'─'*60}")
    print(f"Q : {question[:120]}")
    print(f"A : {res['predicted_answer']}")

    # Top-5 attributed question tokens
    q_scores = list(zip(tokens[:q_len], scores[:q_len]))
    q_scores.sort(key=lambda x: x[1], reverse=True)
    top5 = q_scores[:5]
    print(f"Top-5 Q tokens by attribution:")
    for tok, sc in top5:
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
        description="Benchmark PACE Gradient Attribution on GPT-2 + TellMeWhy"
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2",
        help="GPT-2 variant: gpt2 | gpt2-medium | gpt2-large | gpt2-xl"
    )
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Number of TellMeWhy validation samples to evaluate (default: 200)"
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Riemann-sum steps for PACE integration (default: 100)"
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
        help="Monte-Carlo draws for soft Bernoulli perturbation (default: 10)"
    )
    parser.add_argument(
        "--use_gold", action="store_true",
        help="Use gold answers from dataset instead of generating"
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