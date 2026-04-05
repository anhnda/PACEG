"""
Benchmark Script for PACE Gradient Attribution on Question Answering

This script evaluates the PACE Gradient attribution method on the SQuADv2 dataset,
computing XAI metrics: log-odds, comprehensiveness, and sufficiency.

Usage:
    python run_eval_pg_qa.py --model_name deepset/bert-base-cased-squad2 --num_samples 1000 --steps 101
    python run_eval_pg_qa.py --demo  # Run demo with a few examples
"""

import time
import random
import argparse
import numpy as np
import torch
import traceback
from tqdm import tqdm
from datasets import load_dataset

# Import PACE gradient implementation and metrics
from pace_gradients import pace_gradient_qa, get_model_tokenizer
from xai_metrics import (
    calculate_log_odds_qa,
    calculate_comprehensiveness_qa,
    calculate_sufficiency_qa,
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def run_single_example(question: str, context: str, model_name: str, 
                       a: float, b: float, steps: int, device: str, topk: int = 20):
    """
    Run PACE gradient attribution on a single QA example and compute metrics.
    
    Args:
        question: The question string
        context: The context/passage
        model_name: HuggingFace model name
        a, b: Integration range for epsilon
        steps: Number of Riemann sum steps
        device: Computation device
        topk: Percentage of top tokens for metrics
    
    Returns:
        Dictionary with attribution results and metrics
    """
    # Run PACE gradient attribution
    res = pace_gradient_qa(
        question=question,
        context=context,
        a=a,
        b=b,
        steps=steps,
        model_name=model_name,
        device=device,
        show_special_tokens=True,
    )
    
    # Extract data for metrics
    model = res["model"]
    input_embed = res["input_embed"]
    attention_mask = res["attention_mask"]
    special_tokens_mask = res["special_tokens_mask"]
    token_type_ids = res["token_type_ids"]
    base_token_emb = res["base_token_emb"]
    attr_start = res["attributions_start"]
    attr_end = res["attributions_end"]
    start_idx = res["start_idx"]
    end_idx = res["end_idx"]
    start_logit = res["start_logit"]
    end_logit = res["end_logit"]
    prob_start_orig = res["start_prob"]
    prob_end_orig = res["end_prob"]

    log_odd_start, log_odd_end = calculate_log_odds_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        base_token_emb, attr_start, attr_end, start_idx, end_idx, 
        prob_start_orig, prob_end_orig,
        topk=topk
    )

    comp_start, comp_end = calculate_comprehensiveness_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        base_token_emb, attr_start, attr_end, start_idx, end_idx, 
        prob_start_orig, prob_end_orig,
        topk=topk
    )

    suff_start, suff_end = calculate_sufficiency_qa(
        model, input_embed, attention_mask, special_tokens_mask, token_type_ids,
        base_token_emb, attr_start, attr_end, start_idx, end_idx, 
        prob_start_orig, prob_end_orig,
        topk=topk
    )
    
    return {
        "tokens": res["tokens"],
        "attributions_start": res["attributions_start"],
        "attributions_end": res["attributions_end"],
        "predicted_answer": res["predicted_answer"],
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_logit": res["start_logit"],
        "end_logit": res["end_logit"],
        "time": res["time"],
        # Metrics for start position
        "log_odd_start": log_odd_start,
        "comp_start": comp_start,
        "suff_start": suff_start,
        # Metrics for end position
        "log_odd_end": log_odd_end,
        "comp_end": comp_end,
        "suff_end": suff_end,
    }


def run_benchmark(args):
    """
    Run the full benchmark on SQuADv2 dataset.
    
    Args:
        args: Command line arguments
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Top-k percentage for metrics: {args.topk}%")

    print(f"\nLoading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="validation")
    data= list(zip(dataset['question'], dataset['context'] , dataset['answers'], dataset['id']))
    upd_data=[]
    for question, context, ans, idx in data:
        if len((question+context).split(" ")) < 80:
            upd_data.append((question, context, ans, idx))
    print(f"Length of updated data: {len(upd_data)}")
        # Filter to only answerable questions (has at least one answer)
    answerable_data = [
        {"context": item[1], "question": item[0], "answers": item[2]}
        for item in upd_data
    ]
    
    # Sample the data
    if len(answerable_data) > args.num_samples:
        sampled_data = random.sample(answerable_data, args.num_samples)
    else:
        sampled_data = answerable_data
        print(f"Warning: Only {len(answerable_data)} answerable samples available")
    
    print(f"Loaded {len(sampled_data)} answerable QA pairs for evaluation")
    print("\nLoading model...")
    get_model_tokenizer(args.model_name, device, type="qa")
    print("Model loaded successfully")

    print("\nStarting QA attribution benchmark...")
    
    total_log_odds_start = 0.0
    total_log_odds_end = 0.0
    total_comp_start = 0.0
    total_comp_end = 0.0
    total_suff_start = 0.0
    total_suff_end = 0.0
    total_time = 0.0
    count = 0
    errors = 0
    
    a, b = 0, 1
    
    for idx, example in enumerate(tqdm(sampled_data)):
        context = example["context"]
        question = example["question"]
        
        try:
            res = run_single_example(
                question=question,
                context=context,
                model_name=args.model_name,
                a=a,
                b=b,
                steps=args.steps,
                device=device,
                topk=args.topk,
            )
            
            # Accumulate metrics (separate for start and end)
            total_log_odds_start += res['log_odd_start']
            total_log_odds_end += res['log_odd_end']
            total_comp_start += res['comp_start']
            total_comp_end += res['comp_end']
            total_suff_start += res['suff_start']
            total_suff_end += res['suff_end']
            total_time += res['time']
            count += 1
            
            # Print progress every print_step samples
            if count % args.print_step == 0:
                avg_time = total_time / count
                print(f"\n[{count}/{len(sampled_data)}] Running averages:")
                print(f"  Log-odds (start): {total_log_odds_start / count:.4f}")
                print(f"  Log-odds (end):   {total_log_odds_end / count:.4f}")
                print(f"  Comp (start):     {total_comp_start / count:.4f}")
                print(f"  Comp (end):       {total_comp_end / count:.4f}")
                print(f"  Suff (start):     {total_suff_start / count:.4f}")
                print(f"  Suff (end):       {total_suff_end / count:.4f}")
                print(f"  Avg time:         {avg_time:.4f}s")
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Only print first 5 errors
                print(f"\nError processing sample {idx}: {str(e)[:100]}")
                traceback.print_exc()
            continue
    
    if count > 0:
        print(f"  Log-odds (start):          {total_log_odds_start / count:.6f}")
        print(f"  Comprehensiveness (start): {total_comp_start / count:.6f}")
        print(f"  Sufficiency (start):       {total_suff_start / count:.6f}")
        print(f"  Log-odds (end):            {total_log_odds_end / count:.6f}")
        print(f"  Comprehensiveness (end):   {total_comp_end / count:.6f}")
        print(f"  Sufficiency (end):         {total_suff_end / count:.6f}")
        print(f"Average Time per sample:     {total_time / count:.4f}s")
        print(f"Total evaluation time:       {total_time:.2f}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Gated Gradient Attribution for Question Answering"
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='deepset/bert-base-cased-squad2',
        help='QA model name (default: deepset/bert-base-cased-squad2)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='squad',
        help='Dataset name from HuggingFace datasets (default: squad)'
    )
    parser.add_argument(
        '--steps', 
        type=int, 
        default=100,
        help='Number of steps for Riemann sum'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to evaluate'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=50,
        help='Percentage of top tokens for metrics calculation'
    )
    parser.add_argument(
        '--print_step',
        type=int,
        default=100,
        help='Print metrics every N samples'
    )
    
    args = parser.parse_args()
    run_benchmark(args)
