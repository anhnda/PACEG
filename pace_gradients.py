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
    sentence : str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 101,
    model_name: str = "deepset/bert-base-cased-squad2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
) -> Dict[str, Any]:
    """
    Compute PACE (Prediction-Aware Consistency-Enhanced Gated Gradients) attributions for Sequence Classification.
    """
    global cache
    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, get_base_token_emb, nn_forward_func
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    if cache.get(model_name, None) is None:
        print(f"Model {model_name} not found in cache, loading from stratch")
        tmp = {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tmp["model"] = model
        tmp["tokenizer"] = tokenizer
        cache[model_name] = tmp
    else:
        tokenizer = cache[model_name]["tokenizer"]
        model = cache[model_name]["model"]
    model.to(device)
    model.eval()

    enc = tokenizer(sentence, return_tensors="pt", truncation=True, return_special_tokens_mask=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"].to(device)           # (1, L)
    attention_mask = enc["attention_mask"].to(device) # (1, L)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids
    
    # Base embeddings X: (1, L, d)
    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)  # (1, L, d)
        final_logits = model(inputs_embeds=X, attention_mask=attention_mask).logits[0]
    pred_id = int(final_logits.argmax(dim=-1).item())
    L, d = X.shape[1], X.shape[2]
    start_time = time.perf_counter()
    

    # Integration grid
    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)
    dt = float((b - a) / max(1, steps - 1)) if steps > 1 else float(b - a)
    # Accumulator
    attr = torch.zeros(L, device=device, dtype=X.dtype)
    ones_L = torch.ones(L, device=device, dtype=X.dtype)
    direct = X.squeeze()
    pre_score = None
    target_score = None
    EXPLORATION = 1000 # Higher for more exploration
    attrs = []
    db_scores = []
    
    out = model(
        inputs_embeds=X,
        attention_mask=attention_mask,
        **extra_kwargs
    )
    logits = out.logits[0]  # (num_labels,)
    probs = F.softmax(logits, dim=-1)
    target_prelogit = logits[pred_id]
    target_prob = probs[pred_id]
    ts = []
    n_base = L*3
    ps = 0.9
    p_sample = torch.full((L,1), ps).to(device)
    prev_lb_score = None
    prev_lg_score = None
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id

    mask_token_tensor = torch.tensor([[mask_token_id]], device=device)
    mask_embedding = model.get_input_embeddings()(mask_token_tensor.clone().contiguous())
    X_RefMask = mask_embedding.repeat(1,L,1)
    def func(x):
        iterpolated = x.tile((1,d))
        padding_mask = torch.full((L,1), 1).to(device)
        padding_mask[0] = 0
        padding_mask[-1] = 0

        iterpolated[0,:] = 1
        iterpolated[-1,:] = 1

        X_Ref = X_RefMask

        X_inter = X * iterpolated  + X_Ref * (1-iterpolated)
        eps = torch.zeros((1,L,1), device=device, dtype=X.dtype).requires_grad_(True)
        X_inter = X_inter + eps * padding_mask
    
        out = model(
            inputs_embeds=X_inter,
            attention_mask=attention_mask,
            **extra_kwargs
        )
        logits = out.logits[0]
        probs = F.softmax(logits, dim=-1)
        logit_score = logits[pred_id]
        label_score = probs[pred_id]
        return logit_score
    start_time = time.perf_counter()
    for m in range(1):
        sum_dlg = 0
        for i in range(len(t_vals)):
            # ε(t) = t * 1_L  -> (L,)
            ones_L_rand = torch.ones(L).to(device)
            t = t_vals[i]
            ts.append(t.item())
    
            inteprolate_v = t * ones_L_rand
            itepolated_o = inteprolate_v.view(L,1)
            ex = torch.zeros((L,1), device=device, dtype=X.dtype).requires_grad_(True)
            itepolated_o = itepolated_o + ex
            iterpolated = itepolated_o.tile((1,d))


      
            padding_mask = torch.full((L,1), 1).to(device)
            padding_mask[0] = 0
            padding_mask[-1] = 0

            iterpolated[0,:] = 1
            iterpolated[-1,:] = 1


            X_Ref = X_RefMask

            X_inter = X * iterpolated  + X_Ref * (1-iterpolated)
            eps = torch.zeros((1,L,1), device=device, dtype=X.dtype).requires_grad_(True)
            X_inter = X_inter + eps * padding_mask
        
            out = model(
                inputs_embeds=X_inter,
                attention_mask=attention_mask,
                **extra_kwargs
            )
            logits = out.logits[0]  # (num_labels,)

            probs = F.softmax(logits, dim=-1)
            logit_score = logits[pred_id]
            label_score = probs[pred_id]
            dscore = target_prob - label_score
            if i == 0:
                prev_label_score = label_score
                prev_lg_score = logit_score
            dlogit = logit_score - prev_lg_score    
            dlb = label_score - prev_label_score
            prev_label_score = label_score
            prev_lg_score = logit_score
            # ∂score/∂ε  (L,)
            (grad_eps,) = torch.autograd.grad(logit_score, itepolated_o, retain_graph=False, create_graph=False)
            grad_eps_n = grad_eps
            grad_eps_n = grad_eps / (torch.sum(grad_eps) + 1e-10)
            grad_eps_n = grad_eps_n.squeeze()

            db_scores.append((label_score.detach().cpu().numpy(), logit_score.detach().cpu().numpy(), dlb.detach().cpu().numpy(), t.item()))
            # accumulate ∫ grad_ε · dε, with dε = 1_L * dt
            attri = grad_eps_n * dlogit
            sum_dlg += dlb
            attrs.append(attri)
            attr += attri
    end_time = time.perf_counter()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    base_token_emb = get_base_token_emb(model, tokenizer, device)
    inp = get_inputs(model, tokenizer, sentence, device)
    input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
    log_odd, pred = calculate_log_odds(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)
    comp = calculate_comprehensiveness(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)
    suff = calculate_sufficiency(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)
    if not show_special_tokens:
        special_ids = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids]
        tokens = [tokens[i] for i in keep_idx]
        attr = attr[keep_idx]
    return {
        "tokens": tokens, 
        "attributions": attr.detach().cpu(), 
        "attributions_steps": attrs, 
        "db_scores": db_scores, 
        "ts": ts,
        "time": end_time - start_time,
        "log_odd": log_odd,
        "comp": comp,
        "suff": suff,
        "predicted_label": pred_id
    }  