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
    model, tokenizer = get_model_tokenizer(model_name, device, type="qa")

    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
    )
    input_ids       = enc["input_ids"].to(device)
    attention_mask  = enc["attention_mask"].to(device)
    token_type_ids  = enc.get("token_type_ids", None)
    special_tokens_mask = enc.get("special_tokens_mask", torch.zeros_like(input_ids)).to(device)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)                    # (1, L, d)
        outputs0 = model(inputs_embeds=X, attention_mask=attention_mask, **extra_kwargs)
        start_logits0 = outputs0.start_logits[0]   # (L,)
        end_logits0   = outputs0.end_logits[0]     # (L,)

    L, d = X.shape[1], X.shape[2]
    start_idx = int(start_logits0.argmax().item())
    end_idx   = int(end_logits0.argmax().item())
    if end_idx < start_idx:
        end_idx = start_idx

    start_prob = F.softmax(start_logits0, dim=0)[start_idx]
    end_prob   = F.softmax(end_logits0,   dim=0)[end_idx]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

    # --- Baseline ---
    mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id
    with torch.no_grad():
        mask_emb  = embed(torch.tensor([[mask_token_id]], device=device))  # (1,1,d)
    X_baseline = mask_emb.expand(1, L, d)   # (1, L, d)

    # Fixed positions (CLS, SEP, PAD) — keep coef=1
    ids = input_ids[0]
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    is_special = special_tokens_mask[0].bool()
    is_pad     = (attention_mask[0] == 0)
    is_cls     = (ids == cls_id) if cls_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    is_sep     = (ids == sep_id) if sep_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    fixed = (is_special | is_pad | is_cls | is_sep)  # (L,)

    # === BATCHED integration ===
    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)  # (steps,)

    # Build coef tensor: (steps, L), fixed positions always 1
    coefs_base = t_vals.unsqueeze(1).expand(steps, L).clone()
    coefs_base[:, fixed] = 1.0
    coefs = coefs_base.detach().requires_grad_(True)   # (steps, L) — leaf

    # Interpolated embeddings: (steps, L, d)
    coefs_exp = coefs.unsqueeze(-1).expand(steps, L, d)
    X_inter   = X.squeeze(0) * coefs_exp + X_baseline.squeeze(0) * (1 - coefs_exp)

    # Expand mask/token_type for the batch
    attn_batch = attention_mask.expand(steps, -1)          # (steps, L)
    extra_batch = {}
    if "token_type_ids" in extra_kwargs:
        extra_batch["token_type_ids"] = extra_kwargs["token_type_ids"].expand(steps, -1)

    start_time = time.perf_counter()

    # One forward pass for all steps
    out = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_batch)
    start_scores = out.start_logits[:, start_idx]  # (steps,)
    end_scores   = out.end_logits[:,   end_idx]    # (steps,)

    # PACE deltas: delta[i] = score[i] - score[i-1], delta[0] = 0
    delta_start = start_scores - torch.cat([start_scores[:1], start_scores[:-1]])  # (steps,)
    delta_end   = end_scores   - torch.cat([end_scores[:1],   end_scores[:-1]])    # (steps,)

    # Gradients: ∂(Σ_i start_score_i)/∂coefs — shape (steps, L)
    # Cross-terms are zero: start_score[i] only depends on coefs[i,:]
    grad_start, grad_end = torch.autograd.grad(
        [start_scores.sum(), end_scores.sum()],
        [coefs, coefs],
        retain_graph=False,
    )
    # grad_start and grad_end are both (steps, L)
    # Note: autograd accumulates both into grad_start since coefs is shared;
    # we need separate grads, so use two separate backward passes:

    end_time_tmp = time.perf_counter()  # placeholder, recompute below

    # --- Recompute with separate grads (coefs is shared leaf, need two passes) ---
    coefs2 = coefs_base.detach().requires_grad_(True)
    coefs2_exp = coefs2.unsqueeze(-1).expand(steps, L, d)
    X_inter2 = X.squeeze(0) * coefs2_exp + X_baseline.squeeze(0) * (1 - coefs2_exp)

    start_time = time.perf_counter()

    out2 = model(inputs_embeds=X_inter2, attention_mask=attn_batch, **extra_batch)
    start_scores2 = out2.start_logits[:, start_idx]
    end_scores2   = out2.end_logits[:,   end_idx]

    delta_start2 = start_scores2 - torch.cat([start_scores2[:1], start_scores2[:-1]])
    delta_end2   = end_scores2   - torch.cat([end_scores2[:1],   end_scores2[:-1]])

    (grad_start2,) = torch.autograd.grad(start_scores2.sum(), coefs2, retain_graph=True)
    (grad_end2,)   = torch.autograd.grad(end_scores2.sum(),   coefs2, retain_graph=False)
    # (steps, L) each — cross-step independence guarantees these are correct

    end_time = time.perf_counter()

    # Normalize and accumulate
    grad_start_n = grad_start2 / (grad_start2.sum(dim=1, keepdim=True) + 1e-10)
    grad_end_n   = grad_end2   / (grad_end2.sum(  dim=1, keepdim=True) + 1e-10)

    attr_start = (grad_start_n * delta_start2.unsqueeze(1)).sum(dim=0)  # (L,)
    attr_end   = (grad_end_n   * delta_end2.unsqueeze(1)  ).sum(dim=0)  # (L,)

    # --- Build outputs ---
    base_token_emb     = mask_emb.squeeze(0)          # (1, d) for metrics
    special_tokens_mask_out = fixed                    # (L,) boolean

    tokens_out     = tokens.copy()
    attr_start_out = attr_start.clone()
    attr_end_out   = attr_end.clone()

    if not show_special_tokens:
        special_ids_set = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids_set]
        tokens_out     = [tokens[i] for i in keep_idx]
        attr_start_out = attr_start[keep_idx]
        attr_end_out   = attr_end[keep_idx]

    return {
        "tokens":             tokens_out,
        "attributions_start": attr_start_out,
        "attributions_end":   attr_end_out,
        "time":               end_time - start_time,
        "predicted_answer":   pred_answer,
        "start_idx":          start_idx,
        "end_idx":            end_idx,
        "start_logit":        float(start_logits0[start_idx].item()),
        "end_logit":          float(end_logits0[end_idx].item()),
        # Raw tensors for metrics (unfiltered)
        "model":              model,
        "input_embed":        X,
        "attention_mask":     attention_mask,
        "token_type_ids":     token_type_ids,
        "base_token_emb":     base_token_emb,
        "special_tokens_mask": special_tokens_mask_out,
        "start_prob":         start_prob,
        "end_prob":           end_prob,
    }
# def pace_gradient_qa(
#     question: str,
#     context: str,
#     a: float = 0.0,
#     b: float = 1.0,
#     steps: int = 101,
#     model_name: str = "deepset/bert-base-cased-squad2",
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
#     show_special_tokens: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Compute PACE (Prediction-Aware Consistency-Enhanced Gated Gradients) attributions for Question Answering.
    
#     Uses Riemann-sum integration of gradients along the path ε(t) = t * 1
#     from baseline (t=a) to original (t=b) embeddings.
    
#     Computes SEPARATE attributions for start and end logits, allowing 
#     analysis of which tokens contribute to predicting the answer start
#     vs. the answer end position.
    
#     Args:
#         question: The question string
#         context: The context/passage containing the answer
#         a: Start of interpolation range (0 = baseline)
#         b: End of interpolation range (1 = original)
#         steps: Number of Riemann sum steps
#         model_name: HuggingFace QA model name
#         device: Computation device
#         show_special_tokens: Whether to include [CLS], [SEP] in output
    
#     Returns:
#         Dictionary containing:
#         - tokens: List of token strings
#         - attributions_start: Token attribution scores for start logit (L,)
#         - attributions_end: Token attribution scores for end logit (L,)
#         - predicted_answer: The model's predicted answer string
#         - start_idx, end_idx: Answer span indices
#         - time: Computation time in seconds
#     """
#     model, tokenizer = get_model_tokenizer(model_name, device, type="qa")

#     enc = tokenizer(
#         question,
#         context,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#         return_special_tokens_mask=True,
#         return_offsets_mapping=True,
#     )
#     input_ids = enc["input_ids"].to(device)           # (1, L)
#     attention_mask = enc["attention_mask"].to(device) # (1, L)
#     token_type_ids = enc.get("token_type_ids", None)  # (1, L) - 0 for question, 1 for context
#     special_tokens_mask = enc.get("special_tokens_mask", torch.zeros_like(input_ids)).to(device)
#     offset_mapping = enc.get("offset_mapping", None)  # For extracting answer text
    
#     if token_type_ids is not None:
#         token_type_ids = token_type_ids.to(device)

#     # Only pass token_type_ids if model accepts it
#     fwd_params = inspect.signature(model.forward).parameters
#     extra_kwargs = {}
#     if "token_type_ids" in fwd_params and token_type_ids is not None:
#         extra_kwargs["token_type_ids"] = token_type_ids

#     embed = model.get_input_embeddings()
#     with torch.no_grad():
#         X = embed(input_ids)  # (1, L, d) - original input embeddings
        
#         # Forward pass to get start/end logits for answer span
#         outputs = model(inputs_embeds=X, attention_mask=attention_mask, **extra_kwargs)
#         start_logits = outputs.start_logits[0]  # (L,)
#         end_logits = outputs.end_logits[0]      # (L,)
    
#     L, d = X.shape[1], X.shape[2]
    
#     start_idx = int(start_logits.argmax().item())
#     end_idx = int(end_logits.argmax().item())
#     start_prob = F.softmax(start_logits, dim=0)[start_idx]
#     end_prob = F.softmax(end_logits, dim=0)[end_idx]
    
#     # Ensure valid span (end >= start)
#     if end_idx < start_idx:
#         end_idx = start_idx
    
#     # Compute target scores (the logits we want to attribute)
#     target_start_logit = start_logits[start_idx]
#     target_end_logit = end_logits[end_idx]
    
#     # Extract predicted answer text
#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#     pred_answer_tokens = tokens[start_idx:end_idx + 1]
#     pred_answer = tokenizer.convert_tokens_to_string(pred_answer_tokens)

#     mask_token_id = tokenizer.mask_token_id
#     if mask_token_id is None:
#         # Fallback to PAD token if MASK not available
#         mask_token_id = tokenizer.pad_token_id
    
#     mask_token_tensor = torch.tensor([[mask_token_id]], device=device)
#     with torch.no_grad():
#         mask_embedding = embed(mask_token_tensor)  # (1, 1, d)
#     X_baseline = mask_embedding.repeat(1, L, 1)    # (1, L, d) - baseline for all positions

#     ids = input_ids[0]
#     is_special = special_tokens_mask[0].bool()
#     is_pad = (attention_mask[0] == 0)
#     cls_id = tokenizer.cls_token_id
#     sep_id = tokenizer.sep_token_id
#     is_cls = (ids == cls_id) if cls_id is not None else torch.zeros_like(ids, dtype=torch.bool)
#     is_sep = (ids == sep_id) if sep_id is not None else torch.zeros_like(ids, dtype=torch.bool)
#     fixed_mask = (is_special | is_pad | is_cls | is_sep).view(L, 1)  # (L, 1)

#     t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)
    
#     attr_start = torch.zeros(L, device=device, dtype=X.dtype)
#     attr_end = torch.zeros(L, device=device, dtype=X.dtype)

#     start_time = time.perf_counter()
    
#     # Track previous scores for computing differences (separate for start/end)
#     prev_start_score = None
#     prev_end_score = None

#     for i in range(len(t_vals)):
#         t = t_vals[i]
#         ones_L = torch.ones(L, device=device, dtype=X.dtype)
#         interpolate_v = t * ones_L
#         interpolate_coef = interpolate_v.view(L, 1).requires_grad_(True)
        
#         # Expand to embedding dimension: (L, 1) -> (L, d)
#         interpolate_expanded = interpolate_coef.tile((1, d))
        
#         padding_mask = torch.ones((L, 1), device=device, dtype=X.dtype)
#         padding_mask[fixed_mask] = 0
#         interpolate_expanded[fixed_mask.expand(-1, d)] = 1  # Keep original for fixed

#         X_inter = X * interpolate_expanded + X_baseline * (1 - interpolate_expanded)

#         outputs = model(
#             inputs_embeds=X_inter,
#             attention_mask=attention_mask,
#             **extra_kwargs
#         )
#         start_logits_t = outputs.start_logits[0]
#         end_logits_t = outputs.end_logits[0]
        
#         # Get the start and end logits at the predicted positions
#         start_score = start_logits_t[start_idx]
#         end_score = end_logits_t[end_idx]
        
#         if i == 0:
#             prev_start_score = start_score.detach()
#             prev_end_score = end_score.detach()
#             # continue  # Skip first step (no delta yet)
        
#         delta_start = start_score - prev_start_score
#         delta_end = end_score - prev_end_score
#         prev_start_score = start_score.detach()
#         prev_end_score = end_score.detach()
        
#         (grad_start,) = torch.autograd.grad(
#             start_score, 
#             interpolate_coef, 
#             retain_graph=True,  # Need to retain for end gradient
#             create_graph=False
#         )
#         (grad_end,) = torch.autograd.grad(
#             end_score, 
#             interpolate_coef, 
#             retain_graph=False, 
#             create_graph=False
#         )
        
#         grad_start_normalized = grad_start / (torch.sum(grad_start) + 1e-10)
#         grad_start_normalized = grad_start_normalized.squeeze()  # (L,)
        
#         grad_end_normalized = grad_end / (torch.sum(grad_end) + 1e-10)
#         grad_end_normalized = grad_end_normalized.squeeze()  # (L,)
        
#         attri_start = grad_start_normalized * delta_start
#         attri_end = grad_end_normalized * delta_end

#         attr_start += attri_start
#         attr_end += attri_end

#     end_time = time.perf_counter()

#     base_token_emb = mask_embedding.squeeze(0)  # (1, d)
#     special_tokens_mask = fixed_mask.squeeze()  # (L,) boolean tensor

#     tokens_output = tokens.copy()
#     attr_start_output = attr_start.clone()
#     attr_end_output = attr_end.clone()
    
#     if not show_special_tokens:
#         special_ids = set(tokenizer.all_special_ids)
#         keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids]
#         tokens_output = [tokens[i] for i in keep_idx]
#         attr_start_output = attr_start[keep_idx]
#         attr_end_output = attr_end[keep_idx]

#     return {
#         # Token-level outputs (filtered)
#         "tokens": tokens_output,
#         "attributions_start": attr_start_output,
#         "attributions_end": attr_end_output,
#         "time": end_time - start_time,
#         # QA-specific outputs
#         "predicted_answer": pred_answer,
#         "start_idx": start_idx,
#         "end_idx": end_idx,
#         "start_logit": float(target_start_logit.item()),
#         "end_logit": float(target_end_logit.item()),
#         # Raw data for metrics calculation (unfiltered, on device)
#         "model": model,
#         "input_embed": X,
#         "attention_mask": attention_mask,
#         "token_type_ids": token_type_ids,
#         "base_token_emb": base_token_emb,
#         "special_tokens_mask": special_tokens_mask,
#         "start_prob": start_prob,
#         "end_prob": end_prob
#     }

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
    model     = cache[model_name]["model"]
    model.eval()

    enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                    return_special_tokens_mask=True)
    enc           = {k: v.to(device) for k, v in enc.items()}
    input_ids     = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)  # (1, L, d)

    L, d = X.shape[1], X.shape[2]

    # Baseline — identical to original
    mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id
    mask_tensor   = torch.tensor([[mask_token_id]], device=device)
    with torch.no_grad():
        mask_embedding = embed(mask_tensor)        # (1, 1, d)
    X_RefMask = mask_embedding.repeat(1, L, 1)    # (1, L, d)

    # Predicted label
    with torch.no_grad():
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask,
                        **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())
    target_prob = F.softmax(logits0, dim=-1)[pred_id]

    # Integration grid
    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)  # (steps,)

    # ── Build batched interpolation, replicating the original's `ex` trick ──
    #
    # Original per step:
    #   itepolated_o = t * ones_L  →  shape (L, 1)
    #   ex = zeros((L,1), requires_grad=True)
    #   itepolated_o = itepolated_o + ex          ← ex is the grad leaf
    #   iterpolated  = itepolated_o.tile((1, d))  ← actual copies, not view
    #   fixed[0] = fixed[-1] = 1
    #   X_inter = X * iterpolated + X_Ref * (1 - iterpolated)
    #   grad wrt itepolated_o (= wrt ex since ex is the leaf and grad flows through +)

    # Batched equivalent:
    #   coefs_base: (steps, L) — t broadcast over tokens, fixed positions = 1
    #   ex:         (steps, L, 1) — zero leaf, replaces the per-step ex
    #   itepolated_o = coefs_base.unsqueeze(-1) + ex     (steps, L, 1)
    #   iterpolated  = itepolated_o.tile((1, 1, d))      (steps, L, d) — actual copies
    #   X_inter = X * iterpolated + X_Ref * (1 - iterpolated)
    #   grad wrt ex → (steps, L, 1) → squeeze → (steps, L)

    coefs_base = t_vals.unsqueeze(1).expand(steps, L).clone()  # (steps, L)
    # Fix first and last token (CLS / SEP) — identical to original's hardcoded [0] and [-1]
    coefs_base[:, 0]  = 1.0
    coefs_base[:, -1] = 1.0

    ex = torch.zeros(steps, L, 1, device=device, dtype=X.dtype, requires_grad=True)  # leaf

    itepolated_o = coefs_base.unsqueeze(-1) + ex          # (steps, L, 1)  — mirrors original
    iterpolated  = itepolated_o.tile((1, 1, d))           # (steps, L, d)  — actual copies

    X_inter = (X.squeeze(0) * iterpolated
               + X_RefMask.squeeze(0) * (1 - iterpolated))  # (steps, L, d)

    attn_batch = attention_mask.expand(steps, -1)           # (steps, L)
    extra_batch = {}
    if "token_type_ids" in extra_kwargs:
        extra_batch["token_type_ids"] = extra_kwargs["token_type_ids"].expand(steps, -1)

    start_time = time.perf_counter()

    out          = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_batch)
    logits_batch = out.logits[:, pred_id]                   # (steps,)

    # delta[i] = logit[i] - logit[i-1],  delta[0] = 0  (mirrors original's if i==0 branch)
    delta = logits_batch - torch.cat([logits_batch[:1], logits_batch[:-1]])  # (steps,)
    delta[0] = 0.0   # explicit, matches original: attri[0] = grad * 0

    # Grad w.r.t. ex: (steps, L, 1) — cross-step independence holds because
    # ex[i] only appears in X_inter[i], so ∂logit[i]/∂ex[j] = 0 for j≠i
    (grad_ex,) = torch.autograd.grad(logits_batch.sum(), ex)  # (steps, L, 1)
    grad_ex = grad_ex.squeeze(-1)                              # (steps, L)

    # Normalize per step — identical to original's grad_eps_n
    grad_norm = grad_ex / (grad_ex.sum(dim=1, keepdim=True) + 1e-10)  # (steps, L)

    # Weighted sum — mirrors original's `attr += grad_eps_n * dlogit`
    attr = (grad_norm * delta.unsqueeze(1)).sum(dim=0)  # (L,)

    end_time = time.perf_counter()

    # Metrics — unchanged from original
    base_token_emb = get_base_token_emb(model, tokenizer, device)
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

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
    special_ids_set = set(tokenizer.all_special_ids)
    if not show_special_tokens:
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist())
                    if tid not in special_ids_set]
        tokens = [tokens[i] for i in keep_idx]
        attr   = attr[keep_idx]

    return {
        "tokens":          tokens,
        "attributions":    attr.detach().cpu(),
        "time":            end_time - start_time,
        "log_odd":         log_odd,
        "comp":            comp,
        "suff":            suff,
        "predicted_label": pred_id,
    }