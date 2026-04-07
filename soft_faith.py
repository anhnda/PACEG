"""
soft_faith.py
=============
Soft Normalized Sufficiency (Soft-NS) and Soft Normalized Comprehensiveness
(Soft-NC) metrics from:

  Zhao & Aletras, "Incorporating Attribution Importance for Improving
  Faithfulness Metrics", ACL 2023.
  https://github.com/casszhao/SoftFaith

The core idea: instead of *hard* erasure (entirely removing / retaining
top-k tokens), we apply a per-token Bernoulli dropout to the token
embeddings, with dropout probability proportional to the FA importance
score.  This preserves the full ranking and avoids out-of-distribution
corruptions caused by hard masking.

API
---
All public functions accept the same forward-function / model / embedding
tensors that are produced by `pace_gradient_classification` in
`pace_gradients.py`.

    soft_input_perturbation(embeddings, attr_scores, mode)
        -> perturbed embedding tensor

    calculate_soft_sufficiency(nn_forward_func, model,
                               input_embed, position_embed, type_embed,
                               attention_mask, attr_full,
                               n_samples=10)
        -> scalar float (Soft-NS score for one instance)

    calculate_soft_comprehensiveness(nn_forward_func, model,
                                     input_embed, position_embed, type_embed,
                                     attention_mask, attr_full,
                                     n_samples=10)
        -> scalar float (Soft-NC score for one instance)

    calculate_soft_log_odds(nn_forward_func, model,
                            input_embed, position_embed, type_embed,
                            attention_mask, attr_full,
                            base_token_emb, n_samples=10)
        -> scalar float (Soft log-odds variant for one instance)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_scores(attr: torch.Tensor) -> torch.Tensor:
    """Map attribution scores to [0, 1] via min-max normalisation.

    Handles both positive and negative attributions by shifting so that the
    minimum maps to 0 and the maximum to 1.  A flat (all-equal) vector is
    returned as all-0.5 so that every element gets 50 % retention.
    """
    a_min = attr.min()
    a_max = attr.max()
    if (a_max - a_min).abs() < 1e-9:
        return torch.full_like(attr, 0.5)
    return (attr - a_min) / (a_max - a_min)


def _get_predicted_prob(
    nn_forward_func: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
) -> float:
    """Return the probability of the *predicted* class on the *full* input."""
    with torch.no_grad():
        logits = nn_forward_func(
            model, input_embed, position_embed, type_embed, attention_mask
        )
        probs = F.softmax(logits, dim=-1)
        pred_class = probs.argmax(dim=-1)
        return probs[0, pred_class].item()


def _get_prob_from_embed(
    nn_forward_func: Callable,
    model: torch.nn.Module,
    perturbed_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    pred_class: int,
) -> float:
    """Return the probability of `pred_class` given a perturbed embedding."""
    with torch.no_grad():
        logits = nn_forward_func(
            model, perturbed_embed, position_embed, type_embed, attention_mask
        )
        probs = F.softmax(logits, dim=-1)
        return probs[0, pred_class].item()


# ---------------------------------------------------------------------------
# Soft perturbation  (Eq. 3 in the paper)
# ---------------------------------------------------------------------------

def soft_input_perturbation(
    token_embeddings: torch.Tensor,
    attr_scores: torch.Tensor,
    mode: str = "sufficiency",
) -> torch.Tensor:
    """Apply soft Bernoulli dropout to token embeddings.

    Parameters
    ----------
    token_embeddings : Tensor, shape (1, seq_len, hidden_dim)
        Raw token embeddings for one example.
    attr_scores : Tensor, shape (seq_len,)
        Normalised attribution scores in [0, 1].
    mode : {"sufficiency", "comprehensiveness"}
        - "sufficiency"      : retain elements ∝ importance  (q = a_i)
        - "comprehensiveness": remove elements ∝ importance  (q = 1 - a_i)

    Returns
    -------
    Tensor, shape (1, seq_len, hidden_dim)
        Soft-perturbed embeddings (gradients detached).
    """
    assert mode in ("sufficiency", "comprehensiveness"), \
        f"mode must be 'sufficiency' or 'comprehensiveness', got {mode!r}"

    scores = _normalize_scores(attr_scores)          # (seq_len,)

    if mode == "sufficiency":
        q = scores                                   # retain ∝ importance
    else:
        q = 1.0 - scores                             # retain ∝ (1 - importance)

    # Bernoulli mask: shape (seq_len,) -> broadcast over hidden dim
    device = token_embeddings.device
    mask = torch.bernoulli(q.to(device))             # (seq_len,)
    mask = mask.unsqueeze(0).unsqueeze(-1)           # (1, seq_len, 1)

    perturbed = token_embeddings.detach() * mask
    return perturbed


# ---------------------------------------------------------------------------
# Baseline probability  (S(X, ŷ, 0)  in Eq. 1-2)
# ---------------------------------------------------------------------------

def _baseline_prob(
    nn_forward_func: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    base_token_emb: Optional[torch.Tensor],
    pred_class: int,
) -> float:
    """Probability on a zeroed-out (baseline) sequence.

    If `base_token_emb` is provided the sequence is filled with that
    single baseline vector; otherwise the token embeddings are zeroed.
    """
    seq_len = input_embed.shape[1]
    if base_token_emb is not None:
        # base_token_emb : (1, d)  ->  (1, seq_len, d)
        zero_embed = base_token_emb.unsqueeze(0).expand(
            1, seq_len, -1
        ).to(input_embed.device)
    else:
        zero_embed = torch.zeros_like(input_embed)

    return _get_prob_from_embed(
        nn_forward_func, model,
        zero_embed, position_embed, type_embed, attention_mask,
        pred_class,
    )


# ---------------------------------------------------------------------------
# Soft-NS
# ---------------------------------------------------------------------------

def calculate_soft_sufficiency(
    nn_forward_func: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    attr_full: torch.Tensor,
    base_token_emb: Optional[torch.Tensor] = None,
    n_samples: int = 10,
) -> float:
    """Compute Soft Normalised Sufficiency (Soft-NS) for one instance.

    Following Eq. 4 of the paper:

        Soft-S  = 1 - max(0, p(ŷ|X) - p(ŷ|X'))
        Soft-NS = (Soft-S - S(X, ŷ, 0)) / (1 - S(X, ŷ, 0))

    The perturbation X' is stochastic, so we average over `n_samples`.

    Parameters
    ----------
    nn_forward_func, model, input_embed, position_embed, type_embed,
    attention_mask
        Standard inputs from pace_gradient_classification output.
    attr_full : Tensor, shape (seq_len,)
        Per-token attribution scores (signed or unsigned).
    base_token_emb : Tensor or None
        Baseline token embedding for normalisation denominator.
    n_samples : int
        Number of stochastic soft-perturbation samples to average.

    Returns
    -------
    float
        Soft-NS score (higher = more sufficient rationale).
    """
    # Full-input probability
    with torch.no_grad():
        logits_full = nn_forward_func(
            model, input_embed, position_embed, type_embed, attention_mask
        )
        probs_full = F.softmax(logits_full, dim=-1)
        pred_class = int(probs_full.argmax(dim=-1).item())
        p_full = probs_full[0, pred_class].item()

    # Baseline probability for normalisation
    p_base = _baseline_prob(
        nn_forward_func, model,
        input_embed, position_embed, type_embed, attention_mask,
        base_token_emb, pred_class,
    )
    s_base = 1.0 - max(0.0, p_full - p_base)

    # Average Soft-S over n_samples
    soft_s_values = []
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(
            input_embed, attr_full, mode="sufficiency"
        )
        p_prime = _get_prob_from_embed(
            nn_forward_func, model,
            x_prime, position_embed, type_embed, attention_mask,
            pred_class,
        )
        soft_s_values.append(1.0 - max(0.0, p_full - p_prime))

    soft_s = float(np.mean(soft_s_values))

    denom = 1.0 - s_base
    if abs(denom) < 1e-9:
        return 0.0
    return (soft_s - s_base) / denom


# ---------------------------------------------------------------------------
# Soft-NC
# ---------------------------------------------------------------------------

def calculate_soft_comprehensiveness(
    nn_forward_func: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    attr_full: torch.Tensor,
    base_token_emb: Optional[torch.Tensor] = None,
    n_samples: int = 10,
) -> float:
    """Compute Soft Normalised Comprehensiveness (Soft-NC) for one instance.

    Following Eq. 5 of the paper:

        Soft-C  = max(0, p(ŷ|X) - p(ŷ|X'))
        Soft-NC = Soft-C / (1 - S(X, ŷ, 0))

    Parameters
    ----------
    (same as calculate_soft_sufficiency)

    Returns
    -------
    float
        Soft-NC score (higher = more comprehensive rationale).
    """
    with torch.no_grad():
        logits_full = nn_forward_func(
            model, input_embed, position_embed, type_embed, attention_mask
        )
        probs_full = F.softmax(logits_full, dim=-1)
        pred_class = int(probs_full.argmax(dim=-1).item())
        p_full = probs_full[0, pred_class].item()

    p_base = _baseline_prob(
        nn_forward_func, model,
        input_embed, position_embed, type_embed, attention_mask,
        base_token_emb, pred_class,
    )
    s_base = 1.0 - max(0.0, p_full - p_base)
    denom = 1.0 - s_base
    if abs(denom) < 1e-9:
        return 0.0

    soft_c_values = []
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(
            input_embed, attr_full, mode="comprehensiveness"
        )
        p_prime = _get_prob_from_embed(
            nn_forward_func, model,
            x_prime, position_embed, type_embed, attention_mask,
            pred_class,
        )
        soft_c_values.append(max(0.0, p_full - p_prime))

    soft_c = float(np.mean(soft_c_values))
    return soft_c / denom


# ---------------------------------------------------------------------------
# Soft log-odds (analogous to the hard log-odds used in xai_metrics.py)
# ---------------------------------------------------------------------------

def calculate_soft_log_odds(
    nn_forward_func: Callable,
    model: torch.nn.Module,
    input_embed: torch.Tensor,
    position_embed: torch.Tensor,
    type_embed: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    attr_full: torch.Tensor,
    base_token_emb: Optional[torch.Tensor] = None,
    n_samples: int = 10,
) -> float:
    """Soft log-odds: expected change in log-probability after soft erasure.

    Uses comprehensiveness-mode perturbation (removes important tokens).
    Higher values indicate that important tokens carry more predictive signal.

    Returns
    -------
    float
        Mean log-odds drop across `n_samples`.
    """
    with torch.no_grad():
        logits_full = nn_forward_func(
            model, input_embed, position_embed, type_embed, attention_mask
        )
        probs_full = F.softmax(logits_full, dim=-1)
        pred_class = int(probs_full.argmax(dim=-1).item())
        p_full = float(probs_full[0, pred_class].item())

    log_odds_vals = []
    eps = 1e-9
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(
            input_embed, attr_full, mode="comprehensiveness"
        )
        p_prime = _get_prob_from_embed(
            nn_forward_func, model,
            x_prime, position_embed, type_embed, attention_mask,
            pred_class,
        )
        lo = np.log((p_full + eps) / (1 - p_full + eps)) \
           - np.log((p_prime + eps) / (1 - p_prime + eps))
        log_odds_vals.append(lo)

    return float(np.mean(log_odds_vals))