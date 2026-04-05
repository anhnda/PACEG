"""
reagent_gpt2.py
===============
ReAGent adapted for decoder-only GPT-2 on the TellMeWhy dataset.

Original ReAGent (Zhao & Shan, AAAI 2024) was designed for causal LMs on
text generation, using RoBERTa-MLM as a token-replacement oracle to estimate
per-token importance scores.  This module is a clean port of that algorithm
to the GPT-2 + TellMeWhy setting, producing outputs that are **drop-in
compatible** with the existing xai_metrics_gpt2.py and run_eval_pg_gpt2.py.

─────────────────────────────────────────────────────────────────────────────
Comparison: original ReAGent  vs.  this module
─────────────────────────────────────────────────────────────────────────────
| Dimension            | Original ReAGent            | This module             |
|----------------------|-----------------------------|-------------------------|
| Target model         | GPT-2 / OPT (causal LM)     | GPT-2 (causal LM)       |
| Task                 | Open-ended text generation  | TellMeWhy why-QA        |
| Attribution target   | P(next token | context)     | P(answer token | [Q|A]) |
| Divergence measure   | Hellinger on vocab dist     | same (unchanged)        |
| Token oracle         | RoBERTa-MLM top-k replace   | same (unchanged)        |
| Context positions    | Q tokens only               | Q tokens only           |
| Aggregation          | mean over replacements      | same (unchanged)        |
| Stopping condition   | convergence on imp. dist.   | fixed steps (simpler)   |
| Output keys          | tokens, attributions, ...   | same as paceg_gpt2.py   |

─────────────────────────────────────────────────────────────────────────────
Algorithm (per sample)
─────────────────────────────────────────────────────────────────────────────
Given Q = narrative + why-question,  A = generated (or gold) answer:

1.  Tokenise Q, generate A greedy (or use gold_answer), form [Q | A].
2.  For each token position i in Q (non-special):
    a.  Replace token i with RoBERTa-MLM top-k predictions → {x̃_i^j}.
    b.  For each replacement j, feed [Q̃_j | A] to GPT-2 and measure
            Δ_j = Hellinger( P_full(A), P_replaced_j(A) )
        where P(A) = softmax over *answer-token logits*, aggregated across
        all answer positions (mean of per-position distributions).
    c.  importance[i] = mean_j Δ_j
3.  Q positions not in the oracle input (special tokens) get score 0.
4.  A positions also get score 0 (we only explain the question context).

─────────────────────────────────────────────────────────────────────────────
Why Hellinger on aggregated answer-token distributions?
─────────────────────────────────────────────────────────────────────────────
Original ReAGent measures Hellinger(P_orig_t, P_replaced_t) for each
*predicted* token t independently.  For TellMeWhy the "prediction" is the
full answer span, so we aggregate:

    P_answer(x) = mean_{t in answer_positions} softmax(logits_t(x))

This collapses the answer span to a single distribution over the vocabulary,
to which Hellinger can be applied once per candidate replacement — exactly
the same formula as the original, applied to the same-shaped distribution.

─────────────────────────────────────────────────────────────────────────────
Return keys  (identical to paceg_gpt2.py for drop-in compatibility)
─────────────────────────────────────────────────────────────────────────────
    tokens           : list[str]     -- all tokens in [Q | A]
    q_len            : int           -- number of Q tokens
    answer_positions : list[int]     -- indices of A tokens in full sequence
    answer_ids       : Tensor [La]   -- answer token ids (CPU)
    attributions     : Tensor [T]    -- ReAGent importance per token (CPU)
    input_embed      : Tensor[1,T,D] -- original embedding (CPU)
    base_embed       : Tensor[1,T,D] -- zero baseline (CPU)  [kept for API]
    logits_full      : Tensor[T,V]   -- reference logits (CPU)
    predicted_answer : str
    model            : GPT2LMHeadModel
    tokenizer        : GPT2TokenizerFast
    time             : float
"""

import time
import math
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
)

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_GPT2_CACHE: dict = {}    # (model_name, device) -> (model, tokenizer)
_MLM_CACHE:  dict = {}    # (mlm_name,  device) -> (model, tokenizer)

# Default MLM oracle — same as original ReAGent paper
DEFAULT_MLM = "roberta-base"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def get_model_tokenizer(
    model_name: str = "gpt2",
    device: str = "cpu",
):
    """Return (GPT2LMHeadModel, GPT2TokenizerFast), loading once per process."""
    key = (model_name, device)
    if key not in _GPT2_CACHE:
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        mdl = GPT2LMHeadModel.from_pretrained(model_name)
        mdl.eval().to(device)
        _GPT2_CACHE[key] = (mdl, tok)
    return _GPT2_CACHE[key]


def _get_mlm(mlm_name: str, device: str):
    """Return (RobertaForMaskedLM, RobertaTokenizerFast), loading once."""
    key = (mlm_name, device)
    if key not in _MLM_CACHE:
        tok = RobertaTokenizerFast.from_pretrained(mlm_name)
        mdl = RobertaForMaskedLM.from_pretrained(mlm_name)
        mdl.eval().to(device)
        _MLM_CACHE[key] = (mdl, tok)
    return _MLM_CACHE[key]


# ---------------------------------------------------------------------------
# Hellinger distance (matches xai_metrics_gpt2._hellinger exactly)
# ---------------------------------------------------------------------------

def _hellinger(P: torch.Tensor, Q: torch.Tensor) -> float:
    """
    H(P,Q) = (1/√2) * sqrt( sum_v (sqrt(p_v) - sqrt(q_v))^2 )
    Range [0,1], symmetric.  Applied to CPU tensors.
    """
    P = P.float().clamp(min=0.0)
    Q = Q.float().clamp(min=0.0)
    return (
        (1.0 / math.sqrt(2))
        * ((P.sqrt() - Q.sqrt()).pow(2).sum()).sqrt()
    ).item()


# ---------------------------------------------------------------------------
# Aggregated answer distribution  P_answer(x)
# ---------------------------------------------------------------------------

def _answer_dist(
    model: GPT2LMHeadModel,
    full_ids: torch.Tensor,             # [1, T] on device
    answer_positions: list,
    device: str,
) -> torch.Tensor:
    """
    Compute the mean softmax distribution over the vocabulary, averaged
    across all answer-token positions.

    P_answer = mean_{t in answer_positions} softmax( logits_t )

    This collapses the whole answer span into one [V]-vector so that a
    single Hellinger call can compare two sequences.

    Returns Tensor [V] on CPU.
    """
    with torch.no_grad():
        logits = model(input_ids=full_ids.to(device)).logits  # [1, T, V]

    # Stack softmax distributions at answer positions → [La, V]
    dists = torch.stack([
        F.softmax(logits[0, pos, :], dim=-1)
        for pos in answer_positions
    ], dim=0)                                                  # [La, V]

    return dists.mean(dim=0).cpu()                             # [V]


def _answer_dist_from_embed(
    model: GPT2LMHeadModel,
    full_embed: torch.Tensor,           # [1, T, D] on CPU
    answer_positions: list,
    device: str,
) -> torch.Tensor:
    """Same as _answer_dist but accepts pre-computed embeddings (for input_ids path)."""
    with torch.no_grad():
        logits = model(
            inputs_embeds=full_embed.to(device)
        ).logits                                               # [1, T, V]

    dists = torch.stack([
        F.softmax(logits[0, pos, :], dim=-1)
        for pos in answer_positions
    ], dim=0)                                                  # [La, V]

    return dists.mean(dim=0).cpu()                             # [V]


# ---------------------------------------------------------------------------
# MLM oracle: top-k replacement token ids
# ---------------------------------------------------------------------------

def _get_top_k_replacements(
    gpt2_tokenizer: GPT2TokenizerFast,
    mlm_tokenizer:  RobertaTokenizerFast,
    mlm_model:      RobertaForMaskedLM,
    full_ids:       torch.Tensor,     # [1, T]  GPT-2 token ids (CPU)
    position:       int,              # which Q-token to replace
    gpt2_tokens:    list,             # decoded token strings (length T)
    top_k:          int,
    device:         str,
) -> list:
    """
    Query the RoBERTa-MLM oracle for the top-k replacement candidates
    (in the GPT-2 vocabulary) at sequence position `position`.

    Steps — same as reagent_classification._get_top_k_replacements:
      1. Build masked text: convert GPT-2 tokens to a string, replacing
         position with RoBERTa's <mask> token.
      2. Re-encode with the MLM tokenizer.
      3. Run RoBERTa, collect top-k*3 MLM-vocab token strings.
      4. Re-encode each string with the GPT-2 tokenizer; keep only
         single-subword results so sequence length stays constant.

    Returns a list of GPT-2 token ids (ints), length <= top_k.
    Falls back to the original token if nothing survives filtering.
    """
    gpt2_special = set(gpt2_tokenizer.all_special_tokens)
    mlm_special  = set(mlm_tokenizer.all_special_tokens)

    # Build token list, inserting <mask> at the target position.
    # Skip GPT-2 special tokens so they don't pollute the MLM input.
    masked_tokens = []
    for i, tok in enumerate(gpt2_tokens):
        if tok in gpt2_special:
            continue
        masked_tokens.append(
            mlm_tokenizer.mask_token if i == position else tok
        )

    # Decode to plain text using the MLM tokenizer's converter
    # (handles Ġ / ## subword prefixes correctly)
    masked_text = mlm_tokenizer.convert_tokens_to_string(masked_tokens)

    mlm_enc = mlm_tokenizer(
        masked_text, return_tensors="pt", truncation=True
    ).to(device)

    # Locate the <mask> position in the re-encoded sequence
    mask_id  = mlm_tokenizer.mask_token_id
    mask_pos = (mlm_enc["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]

    if len(mask_pos) == 0:
        # Mask disappeared after re-tokenisation — keep original
        return [full_ids[0, position].item()]

    with torch.no_grad():
        mlm_logits = mlm_model(**mlm_enc).logits[0]   # [L_mlm, V_mlm]

    top_mlm_ids = mlm_logits[mask_pos[0].item()].topk(top_k * 3).indices

    replacement_ids = []
    for mlm_id in top_mlm_ids.tolist():
        tok_str = mlm_tokenizer.decode([mlm_id]).strip()
        if not tok_str or tok_str in mlm_special:
            continue
        # Re-encode with GPT-2 tokenizer — keep only single-token results
        gpt2_ids = gpt2_tokenizer.encode(tok_str, add_special_tokens=False)
        if len(gpt2_ids) == 1:
            replacement_ids.append(gpt2_ids[0])
        if len(replacement_ids) >= top_k:
            break

    if not replacement_ids:
        replacement_ids = [full_ids[0, position].item()]

    return replacement_ids[:top_k]


# ---------------------------------------------------------------------------
# Per-token importance computation
# ---------------------------------------------------------------------------

def _compute_importance(
    gpt2_model:      GPT2LMHeadModel,
    gpt2_tokenizer:  GPT2TokenizerFast,
    mlm_model:       RobertaForMaskedLM,
    mlm_tokenizer:   RobertaTokenizerFast,
    full_ids:        torch.Tensor,    # [1, T] on CPU
    gpt2_tokens:     list,            # decoded strings, length T
    q_len:           int,
    answer_positions: list,
    p_orig:          torch.Tensor,   # [V] reference answer distribution
    top_k:           int,
    device:          str,
) -> torch.Tensor:
    """
    Compute per-token ReAGent importance scores.

    For each Q-token position i (non-special, i < q_len):
        importance[i] = mean_j  Hellinger( P_answer_orig, P_answer_replaced_j )

    A-token positions and special tokens get score 0.

    Returns Tensor [T] on CPU.
    """
    T = full_ids.shape[1]
    importance = torch.zeros(T)

    special_ids = set(gpt2_tokenizer.all_special_ids)

    for i in range(q_len):                       # only Q positions
        tok_id = full_ids[0, i].item()
        if tok_id in special_ids:
            importance[i] = 0.0
            continue

        # Get top-k MLM replacements
        rep_ids = _get_top_k_replacements(
            gpt2_tokenizer, mlm_tokenizer, mlm_model,
            full_ids, i, gpt2_tokens, top_k, device,
        )

        # Measure Hellinger distance for each replacement
        scores = []
        for rep_id in rep_ids:
            full_ids_rep = full_ids.clone()
            full_ids_rep[0, i] = rep_id

            p_rep = _answer_dist(
                gpt2_model, full_ids_rep, answer_positions, device
            )
            scores.append(_hellinger(p_orig, p_rep))

        importance[i] = float(sum(scores) / len(scores)) if scores else 0.0

    return importance                             # [T] CPU


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reagent_gpt2(
    question: str,
    model_name: str = "gpt2",
    mlm_name:   str = DEFAULT_MLM,
    device:     str = "cpu",
    top_k:      int = 3,
    max_new_tokens: int = 30,
    gold_answer: Optional[str] = None,
) -> dict:
    """
    Run ReAGent attribution for one TellMeWhy (Q, A) pair on GPT-2.

    Parameters
    ----------
    question       : Full prompt — "narrative  Why did ...?"
    model_name     : GPT-2 local path or HuggingFace id.
    mlm_name       : RoBERTa MLM oracle id (default: roberta-base).
    device         : 'cpu' or 'cuda'.
    top_k          : MLM replacement candidates per token (paper default 3).
    max_new_tokens : Tokens to generate when gold_answer is None.
    gold_answer    : If provided, skip generation.

    Returns
    -------
    dict with the same keys as paceg_gpt2.pace_gradient_gpt2(), so it is
    a drop-in replacement in run_eval_pg_gpt2.py / run_eval_reagent_gpt2.py.
    """
    t0 = time.time()

    gpt2_model, gpt2_tokenizer = get_model_tokenizer(model_name, device)
    mlm_model,  mlm_tokenizer  = _get_mlm(mlm_name, device)

    # ------------------------------------------------------------------
    # 1. Tokenise Q  (with explicit attention_mask to silence the warning)
    # ------------------------------------------------------------------
    q_enc       = gpt2_tokenizer(question, return_tensors="pt",
                                 add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)         # [1, Lq]
    q_attn_mask = q_enc["attention_mask"].to(device)    # [1, Lq]
    q_len       = q_ids.shape[1]

    # ------------------------------------------------------------------
    # 2. Get answer token ids — generate or use gold
    # ------------------------------------------------------------------
    if gold_answer is not None:
        a_enc    = gpt2_tokenizer(gold_answer, return_tensors="pt",
                                  add_special_tokens=False)
        a_ids_d  = a_enc["input_ids"].to(device)        # [1, La]
        full_ids = torch.cat([q_ids, a_ids_d], dim=1)   # [1, T]
    else:
        with torch.no_grad():
            full_ids = gpt2_model.generate(
                q_ids,
                attention_mask=q_attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=gpt2_tokenizer.eos_token_id,
            )                                            # [1, T]

    T  = full_ids.shape[1]
    La = T - q_len
    if La == 0:
        raise ValueError(
            "Answer is empty -- increase max_new_tokens or supply gold_answer."
        )

    full_ids         = full_ids.cpu()
    answer_positions = list(range(q_len, T))
    answer_ids       = full_ids[0, q_len:]               # [La]

    # ------------------------------------------------------------------
    # 3. Reference answer distribution  P_answer_orig
    # ------------------------------------------------------------------
    p_orig = _answer_dist(gpt2_model, full_ids, answer_positions, device)

    # ------------------------------------------------------------------
    # 4. Token strings for MLM oracle
    # ------------------------------------------------------------------
    gpt2_tokens = gpt2_tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    # ------------------------------------------------------------------
    # 5. ReAGent importance scores
    # ------------------------------------------------------------------
    attributions = _compute_importance(
        gpt2_model, gpt2_tokenizer,
        mlm_model,  mlm_tokenizer,
        full_ids, gpt2_tokens,
        q_len, answer_positions,
        p_orig, top_k, device,
    )                                                    # [T] CPU

    # ------------------------------------------------------------------
    # 6. Embeddings  (kept for API compatibility with xai_metrics_gpt2)
    # ------------------------------------------------------------------
    embed_layer = gpt2_model.transformer.wte
    with torch.no_grad():
        input_embed = embed_layer(full_ids.to(device)).detach().cpu()  # [1,T,D]
    base_embed  = torch.zeros_like(input_embed)

    # Reference logits
    with torch.no_grad():
        logits_full = gpt2_model(
            inputs_embeds=input_embed.to(device)
        ).logits[0].detach().cpu()                       # [T, V]

    return {
        "tokens":           gpt2_tokens,
        "q_len":            q_len,
        "answer_positions": answer_positions,
        "answer_ids":       answer_ids,                  # [La] CPU
        "attributions":     attributions,                # [T]  CPU
        "input_embed":      input_embed,                 # [1,T,D] CPU
        "base_embed":       base_embed,                  # [1,T,D] CPU
        "logits_full":      logits_full,                 # [T,V]  CPU
        "predicted_answer": gpt2_tokenizer.decode(
                                answer_ids.tolist(),
                                skip_special_tokens=True),
        "model":            gpt2_model,
        "tokenizer":        gpt2_tokenizer,
        "time":             time.time() - t0,
    }