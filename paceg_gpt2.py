"""
paceg_gpt2.py

PACE Gradient Attribution for Decoder-Only Models (GPT-2) on TellMeWhy.

Pipeline:
    1. Load GPT-2 model + tokenizer
    2. For each TellMeWhy sample: tokenize Q, generate answer A greedily
    3. Concatenate [Q | A] as the full input sequence
    4. Compute PACE (Path-Averaged Counterfactual Explanation) gradients:
         - Interpolate token embeddings between a zero baseline and the input
         - Integrate gradients w.r.t. the sum of answer-token logits (causal,
           so Q tokens cannot attend to A tokens — naturally enforced by GPT-2)
    5. Return per-token attribution scores over the full [Q|A] sequence

Key design decisions vs. BERT-QA version:
    - No token_type_ids, no [CLS]/[SEP]
    - Causal mask is applied internally by GPT-2 — no manual mask needed
    - Baseline: zero embedding (before positional encoding is added inside model)
    - Target: sum of generated answer-token logits at their respective positions
    - Attribution norm: L2 over embedding dim  →  scalar per token
"""

import time
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Optional

# ── module-level cache so we don't reload on every sample ──────────────────
_MODEL_CACHE: dict = {}


# ───────────────────────────────────────────────────────────────────────────
# Model loading
# ───────────────────────────────────────────────────────────────────────────

def get_model_tokenizer(
    model_name: str = "gpt2",
    device: str = "cpu",
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Load (and cache) a GPT-2 model and tokenizer.

    Args:
        model_name: HuggingFace model identifier, e.g. 'gpt2', 'gpt2-medium'.
        device:     'cuda' or 'cpu'.

    Returns:
        (model, tokenizer) — model is in eval mode on the requested device.
    """
    cache_key = (model_name, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


# ───────────────────────────────────────────────────────────────────────────
# Answer generation
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_answer(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    question_ids: torch.Tensor,          # [1, q_len]
    max_new_tokens: int = 30,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Greedily generate answer tokens given question token IDs.

    Returns:
        answer_ids   : [a_len]   — generated answer token IDs
        full_ids     : [1, T]    — concatenated [Q | A] token IDs
    """
    gen_ids = model.generate(
        question_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy — deterministic, matches ReAGent
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # gen_ids = [1, q_len + a_len]
    answer_ids = gen_ids[0, question_ids.shape[1]:]   # [a_len]
    return answer_ids, gen_ids                         # full_ids = gen_ids


# ───────────────────────────────────────────────────────────────────────────
# PACE gradient core
# ───────────────────────────────────────────────────────────────────────────

def _answer_score(
    model: GPT2LMHeadModel,
    inputs_embeds: torch.Tensor,          # [1, T, D]  requires_grad
    answer_ids: torch.Tensor,             # [a_len]
    answer_positions: list[int],
) -> torch.Tensor:
    """
    Scalar target: sum of logits for each generated answer token
    at its corresponding causal position.

    For position p in the sequence, GPT-2 predicts token at p+1 using
    the hidden state at p.  So to get the logit that generated answer_ids[i],
    we look at logits[0, answer_positions[i] - 1, answer_ids[i]].

    The first answer token is predicted from the last question token, so
    the logit for answer_ids[0] lives at logits[0, q_len - 1, answer_ids[0]].
    """
    outputs = model(inputs_embeds=inputs_embeds)
    logits = outputs.logits  # [1, T, V]

    score = torch.tensor(0.0, device=inputs_embeds.device)
    for i, pos in enumerate(answer_positions):
        # logits at pos-1 produced the token at pos
        score = score + logits[0, pos - 1, answer_ids[i]]
    return score


def pace_gradient_gpt2(
    question: str,
    model_name: str = "gpt2",
    device: str = "cpu",
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 100,
    max_new_tokens: int = 30,
    baseline: str = "zero",              # "zero" | "pad"
    gold_answer: Optional[str] = None,   # if provided, use instead of generating
) -> dict:
    """
    Run PACE gradient attribution on a single TellMeWhy question.

    Args:
        question       : Full question string (narrative + "Why did …?")
        model_name     : GPT-2 variant identifier.
        device         : Computation device.
        a, b           : Integration interval endpoints (default 0→1).
        steps          : Number of Riemann-sum steps.
        max_new_tokens : Max tokens to generate for the answer.
        baseline       : Embedding baseline type.
        gold_answer    : If given, tokenize and use as A instead of generating.

    Returns:
        dict with keys:
            tokens            – list of decoded token strings, length T
            q_len             – number of question tokens
            answer_positions  – list of answer token positions in [Q|A]
            answer_ids        – tensor of answer token IDs
            attributions      – [T] tensor of per-token attribution scores
            predicted_answer  – decoded answer string
            time              – wall-clock seconds
    """
    t0 = time.time()

    model, tokenizer = get_model_tokenizer(model_name, device)

    # ── 1. Tokenise question ────────────────────────────────────────────────
    q_enc = tokenizer(question, return_tensors="pt").to(device)
    question_ids = q_enc["input_ids"]   # [1, q_len]
    q_len = question_ids.shape[1]

    # ── 2. Obtain answer tokens ─────────────────────────────────────────────
    if gold_answer is not None:
        # Prepend a space so GPT-2 tokenises cleanly after question text
        a_enc = tokenizer(" " + gold_answer.strip(), return_tensors="pt").to(device)
        answer_ids = a_enc["input_ids"][0]                        # [a_len]
        full_ids = torch.cat([question_ids, a_enc["input_ids"]], dim=1)
    else:
        answer_ids, full_ids = generate_answer(
            model, tokenizer, question_ids,
            max_new_tokens=max_new_tokens, device=device,
        )

    a_len = answer_ids.shape[0]
    T = full_ids.shape[1]

    if a_len == 0:
        raise ValueError("Model generated an empty answer — increase max_new_tokens.")

    # Positions of answer tokens in the full sequence
    answer_positions = list(range(q_len, q_len + a_len))   # [q_len, …, T-1]

    # ── 3. Token embeddings (no positional encoding yet — added inside GPT-2) ──
    embed_layer = model.transformer.wte                     # word token embeddings
    with torch.no_grad():
        input_embed = embed_layer(full_ids).detach()        # [1, T, D]

    # Baseline embedding
    if baseline == "zero":
        base_embed = torch.zeros_like(input_embed)
    elif baseline == "pad":
        pad_id = torch.tensor([[tokenizer.eos_token_id]], device=device)
        base_embed = embed_layer(pad_id).expand_as(input_embed).detach()
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}")

    delta = input_embed - base_embed                        # [1, T, D]

    # ── 4. Riemann-sum integration ──────────────────────────────────────────
    alphas = torch.linspace(a, b, steps, device=device)
    accumulated_grads = torch.zeros_like(input_embed)       # [1, T, D]

    for alpha in alphas:
        interp = (base_embed + alpha * delta).detach().requires_grad_(True)

        score = _answer_score(model, interp, answer_ids, answer_positions)
        score.backward()

        accumulated_grads += interp.grad.detach()
        model.zero_grad()

    # ── 5. Integrated gradients × (input − baseline) ───────────────────────
    integrated = (accumulated_grads / steps) * delta        # [1, T, D]
    attributions = integrated.norm(dim=-1).squeeze(0)       # [T]

    # ── 6. Decode tokens for readability ────────────────────────────────────
    tokens = [tokenizer.decode([tid]) for tid in full_ids[0].tolist()]
    predicted_answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    elapsed = time.time() - t0

    return {
        "tokens": tokens,
        "q_len": q_len,
        "answer_positions": answer_positions,
        "answer_ids": answer_ids,
        "attributions": attributions,
        "predicted_answer": predicted_answer,
        "model": model,
        "tokenizer": tokenizer,
        "input_embed": input_embed,
        "base_embed": base_embed,
        "full_ids": full_ids,
        "time": elapsed,
    }