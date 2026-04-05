"""
PACE Gradient Attribution for Question Answering Task
"""
import time
import torch
import random
import inspect
import numpy as np
import torch.nn.functional as F
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from xai_metrics import *
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Disable Flash SDP for deterministic behavior
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Global cache for model/tokenizer to avoid reloading
cache = {}

def get_model_tokenizer(model_name: str, device: str, type: str):
    """
    Load or reuse a cached (model, tokenizer)
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda' or 'cpu')
        type: Type of model ('qa' or 'classification')
    
    Returns:
        Tuple of (model, tokenizer)
    """
    key = (model_name, device, type)
    if key in cache:
        return cache[key]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if type == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    elif type == "classification":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    else:
        raise ValueError(f"Unknown model type: {type}")
    
    cache[key] = (model, tokenizer)
    return model, tokenizer

def pace_gradient_qa(
    question: str,
    context: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 101,
    model_name: str = "deepset/bert-base-cased-squad2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
) -> Dict[str, Any]:
    """
    Compute PACE (Prediction-Aware Consistency-Enhanced Gated Gradients) attributions for Question Answering.
    
    Uses Riemann-sum integration of gradients along the path ε(t) = t * 1
    from baseline (t=a) to original (t=b) embeddings.
    
    Computes SEPARATE attributions for start and end logits, allowing 
    analysis of which tokens contribute to predicting the answer start
    vs. the answer end position.
    
    Args:
        question: The question string
        context: The context/passage containing the answer
        a: Start of interpolation range (0 = baseline)
        b: End of interpolation range (1 = original)
        steps: Number of Riemann sum steps
        model_name: HuggingFace QA model name
        device: Computation device
        show_special_tokens: Whether to include [CLS], [SEP] in output
    
    Returns:
        Dictionary containing:
        - tokens: List of token strings
        - attributions_start: Token attribution scores for start logit (L,)
        - attributions_end: Token attribution scores for end logit (L,)
        - predicted_answer: The model's predicted answer string
        - start_idx, end_idx: Answer span indices
        - time: Computation time in seconds
    """
    model, tokenizer = get_model_tokenizer(model_name, device, type="qa")

    enc = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].to(device)           # (1, L)
    attention_mask = enc["attention_mask"].to(device) # (1, L)
    token_type_ids = enc.get("token_type_ids", None)  # (1, L) - 0 for question, 1 for context
    special_tokens_mask = enc.get("special_tokens_mask", torch.zeros_like(input_ids)).to(device)
    offset_mapping = enc.get("offset_mapping", None)  # For extracting answer text
    
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    # Only pass token_type_ids if model accepts it
    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)  # (1, L, d) - original input embeddings
        
        # Forward pass to get start/end logits for answer span
        outputs = model(inputs_embeds=X, attention_mask=attention_mask, **extra_kwargs)
        start_logits = outputs.start_logits[0]  # (L,)
        end_logits = outputs.end_logits[0]      # (L,)
    
    L, d = X.shape[1], X.shape[2]
    
    start_idx = int(start_logits.argmax().item())
    end_idx = int(end_logits.argmax().item())
    start_prob = F.softmax(start_logits, dim=0)[start_idx]
    end_prob = F.softmax(end_logits, dim=0)[end_idx]
    
    # Ensure valid span (end >= start)
    if end_idx < start_idx:
        end_idx = start_idx
    
    # Compute target scores (the logits we want to attribute)
    target_start_logit = start_logits[start_idx]
    target_end_logit = end_logits[end_idx]
    
    # Extract predicted answer text
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_answer_tokens = tokens[start_idx:end_idx + 1]
    pred_answer = tokenizer.convert_tokens_to_string(pred_answer_tokens)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # Fallback to PAD token if MASK not available
        mask_token_id = tokenizer.pad_token_id
    
    mask_token_tensor = torch.tensor([[mask_token_id]], device=device)
    with torch.no_grad():
        mask_embedding = embed(mask_token_tensor)  # (1, 1, d)
    X_baseline = mask_embedding.repeat(1, L, 1)    # (1, L, d) - baseline for all positions

    ids = input_ids[0]
    is_special = special_tokens_mask[0].bool()
    is_pad = (attention_mask[0] == 0)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    is_cls = (ids == cls_id) if cls_id is not None else torch.zeros_like(ids, dtype=torch.bool)
    is_sep = (ids == sep_id) if sep_id is not None else torch.zeros_like(ids, dtype=torch.bool)
    fixed_mask = (is_special | is_pad | is_cls | is_sep).view(L, 1)  # (L, 1)

    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)
    
    attr_start = torch.zeros(L, device=device, dtype=X.dtype)
    attr_end = torch.zeros(L, device=device, dtype=X.dtype)

    start_time = time.perf_counter()
    
    # Track previous scores for computing differences (separate for start/end)
    prev_start_score = None
    prev_end_score = None

    for i in range(len(t_vals)):
        t = t_vals[i]
        ones_L = torch.ones(L, device=device, dtype=X.dtype)
        interpolate_v = t * ones_L
        interpolate_coef = interpolate_v.view(L, 1).requires_grad_(True)
        
        # Expand to embedding dimension: (L, 1) -> (L, d)
        interpolate_expanded = interpolate_coef.tile((1, d))
        
        padding_mask = torch.ones((L, 1), device=device, dtype=X.dtype)
        padding_mask[fixed_mask] = 0
        interpolate_expanded[fixed_mask.expand(-1, d)] = 1  # Keep original for fixed

        X_inter = X * interpolate_expanded + X_baseline * (1 - interpolate_expanded)

        outputs = model(
            inputs_embeds=X_inter,
            attention_mask=attention_mask,
            **extra_kwargs
        )
        start_logits_t = outputs.start_logits[0]
        end_logits_t = outputs.end_logits[0]
        
        # Get the start and end logits at the predicted positions
        start_score = start_logits_t[start_idx]
        end_score = end_logits_t[end_idx]
        
        if i == 0:
            prev_start_score = start_score.detach()
            prev_end_score = end_score.detach()
            # continue  # Skip first step (no delta yet)
        
        delta_start = start_score - prev_start_score
        delta_end = end_score - prev_end_score
        prev_start_score = start_score.detach()
        prev_end_score = end_score.detach()
        
        (grad_start,) = torch.autograd.grad(
            start_score, 
            interpolate_coef, 
            retain_graph=True,  # Need to retain for end gradient
            create_graph=False
        )
        (grad_end,) = torch.autograd.grad(
            end_score, 
            interpolate_coef, 
            retain_graph=False, 
            create_graph=False
        )
        
        grad_start_normalized = grad_start / (torch.sum(grad_start) + 1e-10)
        grad_start_normalized = grad_start_normalized.squeeze()  # (L,)
        
        grad_end_normalized = grad_end / (torch.sum(grad_end) + 1e-10)
        grad_end_normalized = grad_end_normalized.squeeze()  # (L,)
        
        attri_start = grad_start_normalized * delta_start
        attri_end = grad_end_normalized * delta_end

        attr_start += attri_start
        attr_end += attri_end

    end_time = time.perf_counter()

    base_token_emb = mask_embedding.squeeze(0)  # (1, d)
    special_tokens_mask = fixed_mask.squeeze()  # (L,) boolean tensor

    tokens_output = tokens.copy()
    attr_start_output = attr_start.clone()
    attr_end_output = attr_end.clone()
    
    if not show_special_tokens:
        special_ids = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids]
        tokens_output = [tokens[i] for i in keep_idx]
        attr_start_output = attr_start[keep_idx]
        attr_end_output = attr_end[keep_idx]

    return {
        # Token-level outputs (filtered)
        "tokens": tokens_output,
        "attributions_start": attr_start_output,
        "attributions_end": attr_end_output,
        "time": end_time - start_time,
        # QA-specific outputs
        "predicted_answer": pred_answer,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_logit": float(target_start_logit.item()),
        "end_logit": float(target_end_logit.item()),
        # Raw data for metrics calculation (unfiltered, on device)
        "model": model,
        "input_embed": X,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "base_token_emb": base_token_emb,
        "special_tokens_mask": special_tokens_mask,
        "start_prob": start_prob,
        "end_prob": end_prob
    }

def pace_gradient_classification(
    sentence: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 100,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
) -> Dict[str, Any]:
    global cache

    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, get_base_token_emb, nn_forward_func
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    if cache.get(model_name) is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        cache[model_name] = {"model": model, "tokenizer": tokenizer}

    tokenizer = cache[model_name]["tokenizer"]
    model = cache[model_name]["model"]
    model.eval()

    enc = tokenizer(sentence, return_tensors="pt", truncation=True, return_special_tokens_mask=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", None)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)  # (1, L, d)

    L, d = X.shape[1], X.shape[2]

    # --- Baseline: mask embedding repeated for all positions ---
    mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id
    mask_tensor = torch.tensor([[mask_token_id]], device=device)
    with torch.no_grad():
        mask_emb = embed(mask_tensor)           # (1, 1, d)
    X_baseline = mask_emb.expand(1, L, d)      # (1, L, d)

    # Fixed positions (CLS, SEP, PAD) — interpolation coef stays 1
    ids = input_ids[0]
    special_ids_set = set(tokenizer.all_special_ids)
    fixed = torch.tensor([tid in special_ids_set for tid in ids.tolist()],
                         device=device, dtype=torch.bool)  # (L,)

    # --- Get predicted label once ---
    with torch.no_grad():
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask, **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())

    # === BATCHED integration ===
    # t_vals: (steps,)
    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)

    # coefs: (steps, L) — fixed positions always get 1.0
    coefs_base = t_vals.unsqueeze(1).expand(steps, L).clone()  # (steps, L)
    coefs_base[:, fixed] = 1.0

    # coefs as leaf with grad: (steps, L)
    coefs = coefs_base.detach().requires_grad_(True)

    # Expand to embedding dim: (steps, L, d)
    coefs_exp = coefs.unsqueeze(-1).expand(steps, L, d)

    # X_inter: (steps, L, d)  — broadcast X and X_baseline
    X_inter = X.squeeze(0) * coefs_exp + X_baseline.squeeze(0) * (1 - coefs_exp)

    # Reshape for model: (steps, L, d) → run as batch of `steps` sequences
    # Expand attention_mask to (steps, L)
    attn_batch = attention_mask.expand(steps, -1)
    extra_kwargs_batch = {}
    if "token_type_ids" in extra_kwargs:
        extra_kwargs_batch["token_type_ids"] = extra_kwargs["token_type_ids"].expand(steps, -1)

    start_time = time.perf_counter()

    out = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_kwargs_batch)
    logits_batch = out.logits[:, pred_id]  # (steps,)
    probs_batch   = F.softmax(out.logits, dim=-1)[:, pred_id]  # (steps,)

    # PACE deltas: score[i] - score[i-1], with score[-1] = score[0] (so delta[0]=0)
    score_for_delta = logits_batch
    delta = score_for_delta - torch.cat([score_for_delta[:1], score_for_delta[:-1]])  # (steps,)

    # Gradient of sum(logits_batch) w.r.t. coefs — shape (steps, L)
    # We need grad per step, so use diagonal trick: sum over steps weighted by delta
    # grad of logits_batch[i] w.r.t. coefs[i] — backprop through the batch jointly
    grad_sum = torch.autograd.grad(logits_batch.sum(), coefs)[0]  # (steps, L)
    # This gives ∂(Σ_i logit_i)/∂coef_ij = ∂logit_i/∂coef_ij (cross terms are 0 for independent rows)

    # Normalize each step's gradient, then weight by delta
    grad_norm = grad_sum / (grad_sum.sum(dim=1, keepdim=True) + 1e-10)  # (steps, L)
    attr = (grad_norm * delta.unsqueeze(1)).sum(dim=0)  # (L,)

    end_time = time.perf_counter()

    # --- Metrics: reuse X, attention_mask from above (no get_inputs re-call) ---
    base_token_emb = get_base_token_emb(model, tokenizer, device)
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, attn_for_metrics = inp

    log_odd, pred_label = calculate_log_odds(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attr.detach(), topk=20
    )
    comp = calculate_comprehensiveness(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attr.detach(), topk=20
    )
    suff = calculate_sufficiency(
        nn_forward_func, model, X, position_embed, type_embed,
        attention_mask, base_token_emb, attr.detach(), topk=20
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    if not show_special_tokens:
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids_set]
        tokens = [tokens[i] for i in keep_idx]
        attr = attr[keep_idx]

    return {
        "tokens": tokens,
        "attributions": attr.detach().cpu(),
        "time": end_time - start_time,
        "log_odd": log_odd,
        "comp": comp,
        "suff": suff,
        "predicted_label": pred_id,
    }