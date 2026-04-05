"""
xai_metrics_gpt2.py

Faithfulness metrics for PACE-Gradient on decoder-only GPT-2 models,
following the ReAGent paper (Zhao & Shan, 2024):

    • Soft-NC  (Soft Normalised Comprehensiveness)
    • Soft-NS  (Soft Normalised Sufficiency)

Both metrics use Hellinger distance over the full vocabulary distribution
instead of scalar probability differences (which are numerically unstable
for high-dimensional softmax outputs).

Perturbation strategy:
    Soft Bernoulli masking:  x'_i = x_i ⊙ e_i,  e_i ~ Ber(q_i)
        q_i = s_i       (sufficiency   — retain important tokens)
        q_i = 1 - s_i   (comprehensiveness — drop important tokens)
    where s_i ∈ [0,1] is the normalised attribution score for token i.

Evaluation scope (sequence-level, TellMeWhy):
    Metrics are computed at every answer-token step t, then averaged,
    matching the "sequence-level" evaluation in the ReAGent paper.

Reference:
    Zhao & Shan (2024). ReAGent: A Model-agnostic Feature Attribution
    Method for Generative Language Models. AAAI 2024.
    https://arxiv.org/abs/2402.00794
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


# ───────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ───────────────────────────────────────────────────────────────────────────

def _hellinger(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Hellinger distance between two probability distributions.

    H(P, Q) = (1/√2) * ||√P − √Q||₂

    Args:
        p, q : [V] tensors of probabilities (must sum to ~1).

    Returns:
        Scalar tensor in [0, 1].
    """
    return (1.0 / math.sqrt(2.0)) * (p.sqrt() - q.sqrt()).pow(2).sum().sqrt()


def _vocab_dist(
    model,
    inputs_embeds: torch.Tensor,      # [1, T, D]
    position: int,                    # predict the token *at* this position
) -> torch.Tensor:
    """
    Return the softmax probability distribution over the vocabulary
    that GPT-2 assigns to the token at `position`, given `inputs_embeds`.

    GPT-2 causal logic:  hidden_state[t] → predicts token[t+1].
    So to get the distribution for the token at `position`, we read
    logits at index `position - 1`.

    Args:
        model         : GPT2LMHeadModel in eval mode.
        inputs_embeds : [1, T, D] float tensor.
        position      : Index of the token we want the distribution for.

    Returns:
        [V] probability tensor (float32, sums to 1).
    """
    with torch.no_grad():
        logits = model(inputs_embeds=inputs_embeds).logits   # [1, T, V]
    return F.softmax(logits[0, position - 1, :], dim=-1)    # [V]


@torch.no_grad()
def _soft_perturb(
    input_embed: torch.Tensor,        # [1, T, D]
    base_embed: torch.Tensor,         # [1, T, D]
    norm_scores: torch.Tensor,        # [T]   attribution scores in [0,1]
    mode: str,                        # "comprehensiveness" | "sufficiency"
    n_samples: int = 10,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a soft-perturbed embedding by stochastic Bernoulli masking.

    For each token i, independently sample a binary mask e_i ~ Ber(q_i):
        comprehensiveness:  q_i = 1 − s_i  (likely to zero-out important tokens)
        sufficiency:        q_i = s_i      (likely to keep important tokens)

    When e_i = 0, token i is replaced by the baseline embedding.
    We average over `n_samples` Monte-Carlo draws for stability.

    Args:
        input_embed  : Original embedding [1, T, D].
        base_embed   : Baseline (zero) embedding [1, T, D].
        norm_scores  : Normalised attribution per token [T], in [0, 1].
        mode         : Which metric to compute.
        n_samples    : Number of Bernoulli samples to average.
        seed         : Optional RNG seed for reproducibility.

    Returns:
        Averaged perturbed embedding [1, T, D].
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = input_embed.device
    T = norm_scores.shape[0]

    if mode == "comprehensiveness":
        q = 1.0 - norm_scores            # [T]  — drop important tokens
    elif mode == "sufficiency":
        q = norm_scores                  # [T]  — keep important tokens
    else:
        raise ValueError(f"mode must be 'comprehensiveness' or 'sufficiency', got {mode!r}")

    # q: [T] → [n_samples, T, 1] Bernoulli probabilities
    q_expanded = q.unsqueeze(0).unsqueeze(-1).expand(n_samples, T, 1)

    # Sample binary masks: shape [n_samples, T, 1]
    masks = torch.bernoulli(q_expanded).to(device)

    # Apply mask: keep original if mask=1, replace with baseline if mask=0
    # input_embed : [1, T, D] → [n_samples, T, D]
    inp = input_embed.expand(n_samples, T, -1)
    bas = base_embed.expand(n_samples, T, -1)

    perturbed = masks * inp + (1.0 - masks) * bas   # [n_samples, T, D]
    return perturbed.mean(dim=0, keepdim=True)       # [1, T, D]


# ───────────────────────────────────────────────────────────────────────────
# Zero-input baseline distance  ΔP_{0,t}
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _zero_baseline_distance(
    model,
    input_embed: torch.Tensor,        # [1, T, D]  original
    base_embed: torch.Tensor,         # [1, T, D]  zero baseline
    answer_positions: list[int],
) -> torch.Tensor:
    """
    Compute ΔP_{0,t}: Hellinger distance between the vocabulary distribution
    under the zero-baseline input and the full (original) input, averaged
    over all answer token positions.

    This serves as the normalisation denominator in Soft-NS and Soft-NC.

    Returns:
        Scalar tensor — mean Hellinger distance across answer positions.
    """
    distances = []
    P_full_list = [
        _vocab_dist(model, input_embed, pos) for pos in answer_positions
    ]
    P_zero_list = [
        _vocab_dist(model, base_embed, pos)  for pos in answer_positions
    ]

    for p_full, p_zero in zip(P_full_list, P_zero_list):
        distances.append(_hellinger(p_full, p_zero))

    return torch.stack(distances).mean()


# ───────────────────────────────────────────────────────────────────────────
# Normalise attribution scores to [0, 1]
# ───────────────────────────────────────────────────────────────────────────

def _normalise(attributions: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalise attribution scores to [0, 1].
    Handles the degenerate case where all scores are equal.
    """
    a_min = attributions.min()
    a_max = attributions.max()
    if (a_max - a_min).abs() < 1e-8:
        return torch.ones_like(attributions) * 0.5
    return (attributions - a_min) / (a_max - a_min)


# ───────────────────────────────────────────────────────────────────────────
# Public metrics API
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calculate_soft_nc_gpt2(
    model,
    input_embed: torch.Tensor,         # [1, T, D]
    base_embed: torch.Tensor,          # [1, T, D]
    attributions: torch.Tensor,        # [T]
    answer_positions: list[int],
    n_samples: int = 10,
    seed: int = 42,
) -> torch.Tensor:
    """
    Soft Normalised Comprehensiveness (Soft-NC) for a decoder-only model.

    Soft-NC = ΔP_{X'\\R, t} / ΔP_{0,t}

    A higher value means removing the important tokens (those with high
    attribution scores) causes a larger shift in the model's output
    distribution — i.e., those tokens are truly important.

    Args:
        model            : GPT2LMHeadModel.
        input_embed      : Full [Q|A] embedding [1, T, D].
        base_embed       : Zero baseline embedding [1, T, D].
        attributions     : Per-token attribution scores [T] (unnormalised).
        answer_positions : List of answer token positions in [Q|A].
        n_samples        : Bernoulli samples for soft perturbation.
        seed             : RNG seed.

    Returns:
        Scalar Soft-NC value.
    """
    norm_scores = _normalise(attributions)   # [T] in [0, 1]

    # Perturbed embedding: important tokens are likely removed
    perturbed_embed = _soft_perturb(
        input_embed, base_embed, norm_scores,
        mode="comprehensiveness", n_samples=n_samples, seed=seed,
    )

    # ΔP_{X'\\R, t}: Hellinger distance between full and perturbed
    delta_perturbed = []
    for pos in answer_positions:
        p_full = _vocab_dist(model, input_embed,   pos)
        p_pert = _vocab_dist(model, perturbed_embed, pos)
        delta_perturbed.append(_hellinger(p_full, p_pert))
    delta_perturbed = torch.stack(delta_perturbed).mean()

    # ΔP_{0,t}: normalisation anchor
    delta_zero = _zero_baseline_distance(
        model, input_embed, base_embed, answer_positions
    )

    if delta_zero < 1e-8:
        return torch.tensor(0.0)

    return delta_perturbed / delta_zero


@torch.no_grad()
def calculate_soft_ns_gpt2(
    model,
    input_embed: torch.Tensor,         # [1, T, D]
    base_embed: torch.Tensor,          # [1, T, D]
    attributions: torch.Tensor,        # [T]
    answer_positions: list[int],
    n_samples: int = 10,
    seed: int = 42,
) -> torch.Tensor:
    """
    Soft Normalised Sufficiency (Soft-NS) for a decoder-only model.

    Soft-NS = max(0, ΔP_{0,t} − ΔP_{X', t}) / ΔP_{0,t}

    A higher value means keeping only the important tokens is sufficient
    to recover the original model behaviour — those tokens explain the output.

    Args:
        model            : GPT2LMHeadModel.
        input_embed      : Full [Q|A] embedding [1, T, D].
        base_embed       : Zero baseline embedding [1, T, D].
        attributions     : Per-token attribution scores [T] (unnormalised).
        answer_positions : List of answer token positions in [Q|A].
        n_samples        : Bernoulli samples for soft perturbation.
        seed             : RNG seed.

    Returns:
        Scalar Soft-NS value.
    """
    norm_scores = _normalise(attributions)

    # Perturbed embedding: only important tokens are likely kept
    perturbed_embed = _soft_perturb(
        input_embed, base_embed, norm_scores,
        mode="sufficiency", n_samples=n_samples, seed=seed,
    )

    # ΔP_{X', t}: Hellinger distance between full and sufficiency-perturbed
    delta_perturbed = []
    for pos in answer_positions:
        p_full = _vocab_dist(model, input_embed,    pos)
        p_pert = _vocab_dist(model, perturbed_embed, pos)
        delta_perturbed.append(_hellinger(p_full, p_pert))
    delta_perturbed = torch.stack(delta_perturbed).mean()

    # ΔP_{0,t}: normalisation anchor
    delta_zero = _zero_baseline_distance(
        model, input_embed, base_embed, answer_positions
    )

    if delta_zero < 1e-8:
        return torch.tensor(0.0)

    soft_ns = torch.clamp(delta_zero - delta_perturbed, min=0.0) / delta_zero
    return soft_ns


@torch.no_grad()
def calculate_log_odds_gpt2(
    model,
    input_embed: torch.Tensor,         # [1, T, D]
    base_embed: torch.Tensor,          # [1, T, D]
    attributions: torch.Tensor,        # [T]
    answer_ids: torch.Tensor,          # [a_len]
    answer_positions: list[int],
    topk: int = 20,                    # percentage (0–100) of tokens to mask
) -> torch.Tensor:
    """
    Log-odds metric: how much do the top-k attributed question tokens
    contribute to predicting the answer tokens?

    log_odds = mean_t [ log p(a_t | full) − log p(a_t | masked) ]

    Only question tokens are candidates for masking (attributions over
    answer positions are excluded since those are the target).

    Args:
        model            : GPT2LMHeadModel.
        input_embed      : [1, T, D].
        base_embed       : [1, T, D].
        attributions     : [T] per-token scores.
        answer_ids       : [a_len] generated answer token IDs.
        answer_positions : List of answer positions in [Q|A].
        topk             : Percentage of Q-tokens to mask.

    Returns:
        Scalar log-odds value (positive = important tokens matter).
    """
    q_len = answer_positions[0]          # question spans [0, q_len)
    q_attr = attributions[:q_len]        # attributions over Q only

    # Select top-k% question tokens by attribution score
    k = max(1, int(len(q_attr) * topk / 100))
    topk_indices = q_attr.topk(k).indices   # [k] indices into Q

    # Build masked embedding: zero out top-k Q tokens
    masked_embed = input_embed.clone()
    for idx in topk_indices:
        masked_embed[0, idx, :] = base_embed[0, idx, :]

    log_odds_per_step = []
    for i, pos in enumerate(answer_positions):
        token_id = answer_ids[i].item()

        p_full = _vocab_dist(model, input_embed,  pos)[token_id]
        p_mask = _vocab_dist(model, masked_embed, pos)[token_id]

        # Clamp to avoid log(0)
        p_full = p_full.clamp(min=1e-10)
        p_mask = p_mask.clamp(min=1e-10)

        log_odds_per_step.append(p_full.log() - p_mask.log())

    return torch.stack(log_odds_per_step).mean()


# ───────────────────────────────────────────────────────────────────────────
# Convenience wrapper: compute all three metrics at once
# ───────────────────────────────────────────────────────────────────────────

def calculate_all_metrics_gpt2(
    model,
    input_embed: torch.Tensor,
    base_embed: torch.Tensor,
    attributions: torch.Tensor,
    answer_ids: torch.Tensor,
    answer_positions: list[int],
    topk: int = 20,
    n_samples: int = 10,
    seed: int = 42,
) -> dict:
    """
    Compute Soft-NC, Soft-NS, and Log-odds for a single sample.

    Returns:
        dict with keys: 'soft_nc', 'soft_ns', 'log_odds'  (scalar tensors)
    """
    soft_nc = calculate_soft_nc_gpt2(
        model, input_embed, base_embed, attributions,
        answer_positions, n_samples=n_samples, seed=seed,
    )
    soft_ns = calculate_soft_ns_gpt2(
        model, input_embed, base_embed, attributions,
        answer_positions, n_samples=n_samples, seed=seed,
    )
    log_odds = calculate_log_odds_gpt2(
        model, input_embed, base_embed, attributions,
        answer_ids, answer_positions, topk=topk,
    )
    return {
        "soft_nc":  soft_nc,
        "soft_ns":  soft_ns,
        "log_odds": log_odds,
    }