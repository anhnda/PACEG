"""
reagent_gpt2.py
===============
ReAGent-style occlusion attribution for decoder-only GPT-2 on TellMeWhy.

NO external MLM oracle.  GPT-2 is a causal LM — it cannot fill [MASK]
tokens.  Instead we follow the core ReAGent intuition directly:

    "Replacing an important token should cause a larger change in the
     model's confidence in predicting the target."

We implement this via **token occlusion**: for each Q-token position i,
we replace token i with the GPT-2 EOS/pad token id (a neutral, low-
information substitute) and measure how much the answer distribution
shifts using Hellinger distance.

─────────────────────────────────────────────────────────────────────────────
Comparison: original ReAGent  vs.  this module
─────────────────────────────────────────────────────────────────────────────
| Dimension            | Original ReAGent             | This module             |
|----------------------|------------------------------|-------------------------|
| Target model         | GPT-2 / OPT (causal LM)      | GPT-2 (causal LM)       |
| Task                 | Open-ended generation        | TellMeWhy why-QA        |
| Attribution target   | P(next token | context)      | P(answer tokens | [Q|A])|
| Divergence measure   | Hellinger on vocab dist      | same (unchanged)        |
| Token replacement    | RoBERTa-MLM top-k tokens     | EOS token occlusion     |
| Why different oracle | RoBERTa is a MLM model       | GPT-2 is causal LM —   |
|                      |                              | no [MASK] prediction    |
| Aggregation          | mean over k replacements     | single occlusion        |
| Output keys          | same as paceg_gpt2.py        | same as paceg_gpt2.py   |

─────────────────────────────────────────────────────────────────────────────
Algorithm (per sample)
─────────────────────────────────────────────────────────────────────────────
Given Q = narrative + why-question,  A = generated (or gold) answer:

1.  Tokenise Q, get A (generate or gold), form full = [Q | A].
2.  Compute reference answer distribution:
        P_orig = mean_{t in answer_positions} softmax( logits_t(full) )   [V]
3.  For each non-special token position i in Q:
        full_occ = full with token i replaced by EOS id
        P_occ    = mean_{t in answer_positions} softmax( logits_t(full_occ) )
        importance[i] = Hellinger( P_orig, P_occ )
4.  Special tokens and A positions get score 0.

─────────────────────────────────────────────────────────────────────────────
Return keys  (identical to paceg_gpt2.py — drop-in compatible)
─────────────────────────────────────────────────────────────────────────────
    tokens           : list[str]     -- all tokens in [Q | A]
    q_len            : int
    answer_positions : list[int]
    answer_ids       : Tensor [La]
    attributions     : Tensor [T]    -- Hellinger occlusion score per token
    input_embed      : Tensor[1,T,D]
    base_embed       : Tensor[1,T,D]
    logits_full      : Tensor[T,V]
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
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def get_model_tokenizer(model_name: str = "gpt2", device: str = "cpu"):
    """Return (GPT2LMHeadModel, GPT2TokenizerFast), loading once per process."""
    key = (model_name, device)
    if key not in _CACHE:
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        mdl = GPT2LMHeadModel.from_pretrained(model_name)
        mdl.eval().to(device)
        _CACHE[key] = (mdl, tok)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Hellinger distance
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
# Answer distribution helper
# ---------------------------------------------------------------------------

def _answer_dist(
    model: GPT2LMHeadModel,
    full_ids: torch.Tensor,        # [1, T]  on device
    answer_positions: list,
    device: str,
) -> torch.Tensor:
    """
    Mean softmax distribution over answer-token positions.

        P_answer = mean_{t in answer_positions} softmax( logits_t )

    Collapses the answer span to one [V]-vector so a single Hellinger
    call captures the full answer-level distributional shift.

    Returns Tensor [V] on CPU.
    """
    with torch.no_grad():
        logits = model(input_ids=full_ids.to(device)).logits  # [1, T, V]

    dists = torch.stack([
        F.softmax(logits[0, pos, :], dim=-1)
        for pos in answer_positions
    ], dim=0)                                                  # [La, V]

    return dists.mean(dim=0).cpu()                             # [V]


# ---------------------------------------------------------------------------
# Per-token importance via occlusion
# ---------------------------------------------------------------------------

def _compute_importance(
    model:            GPT2LMHeadModel,
    tokenizer:        GPT2TokenizerFast,
    full_ids:         torch.Tensor,   # [1, T] on CPU
    q_len:            int,
    answer_positions: list,
    p_orig:           torch.Tensor,   # [V] reference distribution
    device:           str,
) -> torch.Tensor:
    """
    For each non-special Q-token i:
        occlude token i with EOS id
        importance[i] = Hellinger( P_orig, P_occluded )

    A-token positions and special tokens → score 0.

    Returns Tensor [T] on CPU.
    """
    T           = full_ids.shape[1]
    importance  = torch.zeros(T)
    special_ids = set(tokenizer.all_special_ids)
    occ_id      = tokenizer.eos_token_id       # neutral replacement token

    for i in range(q_len):
        tok_id = full_ids[0, i].item()
        if tok_id in special_ids:
            importance[i] = 0.0
            continue

        # Replace token i with EOS
        full_ids_occ      = full_ids.clone()
        full_ids_occ[0, i] = occ_id

        p_occ = _answer_dist(model, full_ids_occ, answer_positions, device)
        importance[i] = _hellinger(p_orig, p_occ)

    return importance                          # [T] CPU


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reagent_gpt2(
    question:       str,
    model_name:     str = "gpt2",
    device:         str = "cpu",
    max_new_tokens: int = 30,
    gold_answer:    Optional[str] = None,
) -> dict:
    """
    Run ReAGent-style occlusion attribution for one TellMeWhy sample.

    Parameters
    ----------
    question       : Full prompt — "narrative  Why did ...?"
    model_name     : GPT-2 local path or HuggingFace id.
    device         : 'cpu' or 'cuda'.
    max_new_tokens : Tokens to generate when gold_answer is None.
    gold_answer    : If provided, skip generation.

    Returns
    -------
    dict with the same keys as paceg_gpt2.pace_gradient_gpt2() —
    drop-in compatible with xai_metrics_gpt2 and run_eval_reagent_gpt2.
    """
    t0 = time.time()
    model, tokenizer = get_model_tokenizer(model_name, device)

    # ------------------------------------------------------------------
    # 1. Tokenise Q  (explicit attention_mask — GPT-2 pad == eos warning)
    # ------------------------------------------------------------------
    q_enc       = tokenizer(question, return_tensors="pt",
                            add_special_tokens=True)
    q_ids       = q_enc["input_ids"].to(device)         # [1, Lq]
    q_attn_mask = q_enc["attention_mask"].to(device)    # [1, Lq]
    q_len       = q_ids.shape[1]

    # ------------------------------------------------------------------
    # 2. Get answer token ids — generate or use gold
    # ------------------------------------------------------------------
    if gold_answer is not None:
        a_enc    = tokenizer(gold_answer, return_tensors="pt",
                             add_special_tokens=False)
        a_ids_d  = a_enc["input_ids"].to(device)        # [1, La]
        full_ids = torch.cat([q_ids, a_ids_d], dim=1)   # [1, T]
    else:
        with torch.no_grad():
            full_ids = model.generate(
                q_ids,
                attention_mask=q_attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
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
    # 3. Reference answer distribution
    # ------------------------------------------------------------------
    p_orig = _answer_dist(model, full_ids, answer_positions, device)

    # ------------------------------------------------------------------
    # 4. ReAGent occlusion importance scores
    # ------------------------------------------------------------------
    attributions = _compute_importance(
        model, tokenizer,
        full_ids, q_len, answer_positions,
        p_orig, device,
    )                                                    # [T] CPU

    # ------------------------------------------------------------------
    # 5. Embeddings  (for API compatibility with xai_metrics_gpt2)
    # ------------------------------------------------------------------
    embed_layer = model.transformer.wte
    with torch.no_grad():
        input_embed = embed_layer(
            full_ids.to(device)
        ).detach().cpu()                                 # [1,T,D]
    base_embed = torch.zeros_like(input_embed)

    with torch.no_grad():
        logits_full = model(
            inputs_embeds=input_embed.to(device)
        ).logits[0].detach().cpu()                       # [T,V]

    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    return {
        "tokens":           tokens,
        "q_len":            q_len,
        "answer_positions": answer_positions,
        "answer_ids":       answer_ids,
        "attributions":     attributions,
        "input_embed":      input_embed,
        "base_embed":       base_embed,
        "logits_full":      logits_full,
        "predicted_answer": tokenizer.decode(
                                answer_ids.tolist(),
                                skip_special_tokens=True),
        "model":            model,
        "tokenizer":        tokenizer,
        "time":             time.time() - t0,
    }