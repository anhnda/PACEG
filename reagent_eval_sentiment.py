"""
reagent_classification.py

ReAGent adapted for BERT-based sequence classification
(DistilBERT / BERT / RoBERTa + classification head).

Original ReAGent (Zhao & Shan, AAAI 2024) was designed for decoder-only
causal LMs on text generation. This module ports the core algorithm to the
encoder-only classification setting, matching the interface of
pace_gradient_classification() so it drops into the existing PACE eval
harness (evaluate_slalom.py / the PACE main script) with zero changes.

─────────────────────────────────────────────────────────────────────────────
What changes vs. the original ReAGent
─────────────────────────────────────────────────────────────────────────────
| Dimension          | Original ReAGent              | This adaptation        |
|--------------------|-------------------------------|------------------------|
| Target model       | Causal LM (GPT-2, OPT, …)    | Encoder classifier     |
| Attribution target | P(next token | context)       | P(class | full input)  |
| Divergence measure | Hellinger on vocab dist       | Hellinger on label dist|
| Token oracle       | RoBERTa MLM top-k replacements| same (unchanged)       |
| Aggregation        | mean over replacement samples | same (unchanged)       |
| Stopping cond.     | top-n tokens explain target   | same logic, adapted    |
| Output             | per-token importance scores   | same shape/interface   |

─────────────────────────────────────────────────────────────────────────────
Algorithm (per token position i)
─────────────────────────────────────────────────────────────────────────────
1. Obtain P_orig = softmax(classifier(x))               # original label dist

2. For each non-special token position i:
   a. Mask position i with [MASK] and query the MLM oracle to get the
      top-k most probable replacement tokens r_1 … r_k.
   b. For each r_j, replace x[i] with r_j → x̃, compute
      P_j = softmax(classifier(x̃)).
   c. importance[i] = mean_j  Hellinger(P_orig, P_j)
         where Hellinger(p,q) = (1/√2) ‖√p − √q‖₂
      A large distance means replacing token i strongly shifts the
      prediction → token i is important.

3. Special tokens ([CLS], [SEP], [PAD]) get importance = 0.

─────────────────────────────────────────────────────────────────────────────
Why Hellinger on label dist (not vocab dist)?
─────────────────────────────────────────────────────────────────────────────
The original uses Hellinger on the full vocabulary distribution because the
target is "did replacing this token change what the model predicts next?".
For classification the analogous question is "did replacing this token change
the label distribution?". The Hellinger distance is still a valid, bounded
[0,1] divergence between two probability distributions — we just apply it to
the (much smaller) C-dimensional label simplex instead of the V-dim vocab.

─────────────────────────────────────────────────────────────────────────────
Interface
─────────────────────────────────────────────────────────────────────────────
    res = reagent_classification(
        sentence    = "This film is great",
        model_name  = "distilbert-base-uncased-finetuned-sst-2-english",
        top_k       = 3,          # replacement candidates per position
        n_samples   = 1,          # repetitions per position (set >1 for MC avg)
        show_special_tokens = False,
    )
    # res["tokens"]       — list[str]
    # res["attributions"] — list[float]   (Hellinger importance per token)
    # res["log_odd"]      — float
    # res["comp"]         — float
    # res["suff"]         — float
    # res["time"]         — float

Usage as standalone eval script (matches PACE main script):
    python reagent_classification.py --model distilbert --dataset sst2
    python reagent_classification.py --model bert       --dataset imdb --top_k 5
"""

import time
import math
import random
import argparse
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Lazy model cache (avoid reloading on every call) ──────────────────────
_clf_cache:  dict = {}   # model_name → (tokenizer, model)
_mlm_cache:  dict = {}   # mlm_name   → (tokenizer, model)

MLM_ORACLE = "roberta-base"   # same oracle as original ReAGent

MODEL_NAMES = {
    ("distilbert", "sst2"):   "distilbert-base-uncased-finetuned-sst-2-english",
    ("distilbert", "imdb"):   "textattack/distilbert-base-uncased-imdb",
    ("distilbert", "rotten"): "textattack/distilbert-base-uncased-rotten-tomatoes",
    ("bert",       "sst2"):   "textattack/bert-base-uncased-SST-2",
    ("bert",       "imdb"):   "textattack/bert-base-uncased-imdb",
    ("bert",       "rotten"): "textattack/bert-base-uncased-rotten-tomatoes",
    ("roberta",    "sst2"):   "textattack/roberta-base-SST-2",
    ("roberta",    "imdb"):   "textattack/roberta-base-imdb",
    ("roberta",    "rotten"): "textattack/roberta-base-rotten-tomatoes",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _hellinger(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Hellinger distance between two probability vectors p and q.
    H(p,q) = (1/√2) * ‖√p − √q‖₂   ∈ [0, 1]

    Adaptation note: original ReAGent applies this to the full vocabulary
    distribution (dim = V ≈ 50k). Here we apply it to the label distribution
    (dim = C, typically 2). The formula and its properties are identical —
    only the dimensionality changes.
    """
    p = p.float().clamp(min=0.0)
    q = q.float().clamp(min=0.0)
    return (0.5 * ((p.sqrt() - q.sqrt()) ** 2).sum()).sqrt().item()


def _load_clf(model_name: str, device: str):
    if model_name not in _clf_cache:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        mdl.eval()
        _clf_cache[model_name] = (tok, mdl)
    return _clf_cache[model_name]


def _load_mlm(mlm_name: str, device: str):
    if mlm_name not in _mlm_cache:
        tok = AutoTokenizer.from_pretrained(mlm_name, use_fast=True)
        mdl = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device)
        mdl.eval()
        _mlm_cache[mlm_name] = (tok, mdl)
    return _mlm_cache[mlm_name]


def _get_label_dist(clf_model, clf_tokenizer, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, device: str) -> torch.Tensor:
    """Return softmax label distribution, shape (C,)."""
    with torch.no_grad():
        logits = clf_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits[0]
    return F.softmax(logits, dim=-1)


def _get_top_k_replacements(
    mlm_tokenizer,
    mlm_model,
    input_ids_clf: torch.Tensor,      # (1, L) — classifier tokenizer ids
    position: int,                     # which token to replace
    clf_tokens: list[str],             # decoded token strings
    top_k: int,
    device: str,
) -> list[int]:
    """
    Query the RoBERTa MLM oracle for the top-k replacement token ids
    (in the *classifier* vocabulary) for position `position`.

    Adaptation note:
      Original ReAGent feeds the causal-LM context directly to RoBERTa-MLM
      by inserting a [MASK] at the target position. We do the same here —
      except our "context" is the full sentence (not just left context),
      which is actually *better* suited to a bidirectional MLM oracle.

    Steps:
      1. Convert the classifier token sequence back to a string.
      2. Re-encode with the MLM tokenizer, inserting [MASK] at the
         corresponding word position.
      3. Run the MLM, collect top-k token strings.
      4. Re-encode those strings with the *classifier* tokenizer to get
         ids that can be substituted into input_ids_clf.
    """
    # Reconstruct context text from clf tokens (strips special tokens)
    clf_special = set(mlm_tokenizer.all_special_tokens)

    # Build a version of the token list with position masked
    masked_tokens = list(clf_tokens)
    masked_tokens[position] = mlm_tokenizer.mask_token   # "[MASK]" or "<mask>"

    # Re-join — subword tokens from BERT/RoBERTa need convert_tokens_to_string
    # to correctly handle the Ġ / ## prefixes.
    masked_text = mlm_tokenizer.convert_tokens_to_string(
        [t for t in masked_tokens if t not in clf_special]
    )

    mlm_enc = mlm_tokenizer(
        masked_text, return_tensors="pt", truncation=True
    ).to(device)

    # Find [MASK] position in MLM encoding
    mask_token_id = mlm_tokenizer.mask_token_id
    mask_positions = (mlm_enc["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        # Fallback: couldn't insert mask → return original token as only candidate
        return [input_ids_clf[0, position].item()]

    mask_pos = mask_positions[0].item()

    with torch.no_grad():
        mlm_logits = mlm_model(**mlm_enc).logits[0]   # (L_mlm, V_mlm)

    top_k_ids_mlm = mlm_logits[mask_pos].topk(top_k * 3).indices   # over-sample

    # Convert MLM token strings → classifier token ids
    clf_tok = _clf_cache.get(list(_clf_cache.keys())[0], (None, None))[0]
    # Re-fetch clf tokenizer safely
    clf_tokenizer_obj = None
    for v in _clf_cache.values():
        clf_tokenizer_obj = v[0]; break

    replacement_ids = []
    for mlm_id in top_k_ids_mlm.tolist():
        token_str = mlm_tokenizer.decode([mlm_id]).strip()
        if not token_str or token_str in clf_special:
            continue
        # Re-encode with classifier tokenizer (single subword only)
        clf_ids = clf_tokenizer_obj.encode(token_str, add_special_tokens=False)
        if len(clf_ids) == 1:          # keep only single-subword replacements
            replacement_ids.append(clf_ids[0])
        if len(replacement_ids) >= top_k:
            break

    if not replacement_ids:
        replacement_ids = [input_ids_clf[0, position].item()]   # fallback

    return replacement_ids[:top_k]


# ── Core ReAGent-classification importance computation ────────────────────

def _compute_importance_scores(
    clf_tokenizer,
    clf_model,
    mlm_tokenizer,
    mlm_model,
    input_ids: torch.Tensor,       # (1, L)
    attention_mask: torch.Tensor,  # (1, L)
    clf_tokens: list[str],         # token strings (length L, incl. specials)
    p_orig: torch.Tensor,          # original label distribution (C,)
    top_k: int,
    device: str,
) -> np.ndarray:
    """
    Compute per-token Hellinger importance scores.

    For each non-special token at position i:
      importance[i] = mean over top-k replacements of Hellinger(P_orig, P_replaced)

    Adaptation note:
      Original ReAGent accumulates scores *recursively* — it rationalizes
      tokens one by one, updating which tokens have been "explained". Here we
      use the simpler independent-perturbation variant (one position at a time,
      no recursive update). This matches how the scores are consumed by the
      PACE eval metrics (which treat attributions as an independent ranking).
    """
    L = input_ids.shape[1]
    special_ids = set(clf_tokenizer.all_special_ids)
    importance = np.zeros(L, dtype=np.float32)

    for i in range(L):
        tok_id = input_ids[0, i].item()
        if tok_id in special_ids:
            importance[i] = 0.0
            continue

        # Get top-k replacement candidates from MLM oracle
        replacement_ids = _get_top_k_replacements(
            mlm_tokenizer, mlm_model,
            input_ids, i, clf_tokens, top_k, device,
        )

        # Average Hellinger over replacements
        scores = []
        for rep_id in replacement_ids:
            x_rep = input_ids.clone()
            x_rep[0, i] = rep_id
            p_rep = _get_label_dist(clf_model, clf_tokenizer,
                                    x_rep, attention_mask, device)
            scores.append(_hellinger(p_orig, p_rep))

        importance[i] = float(np.mean(scores)) if scores else 0.0

    return importance


# ── Faithfulness metrics (log-odds / comp / suff) ─────────────────────────
# Identical logic to evaluate_slalom.py — token-level masking with baseline.

def _get_base_emb(clf_model, clf_tokenizer, device: str) -> torch.Tensor:
    mask_id = clf_tokenizer.mask_token_id or clf_tokenizer.pad_token_id
    with torch.no_grad():
        return clf_model.get_input_embeddings()(
            torch.tensor([[mask_id]], device=device)
        ).squeeze(0)


def _forward_prob_emb(clf_model, embed_input, attention_mask,
                      pred_id: int, device: str) -> torch.Tensor:
    with torch.no_grad():
        logits = clf_model(
            inputs_embeds=embed_input.to(device),
            attention_mask=attention_mask.to(device),
        ).logits[0]
    return F.softmax(logits, dim=-1)[pred_id]


def _compute_faithfulness_metrics(
    clf_model,
    clf_tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    importance: np.ndarray,
    device: str,
    topk_pct: int = 20,
) -> tuple[float, float, float, int]:
    """
    Compute log-odds, comprehensiveness, sufficiency using embedding replacement.
    Returns (log_odd, comp, suff, pred_id).
    """
    embed = clf_model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids.to(device))                       # (1, L, D)
        logits0 = clf_model(
            inputs_embeds=X, attention_mask=attention_mask.to(device)
        ).logits[0]
    pred_id = int(logits0.argmax().item())
    prob_orig = F.softmax(logits0, dim=-1)[pred_id]

    base_emb = _get_base_emb(clf_model, clf_tokenizer, device)
    L = X.shape[1]

    special_ids = set(clf_tokenizer.all_special_ids)
    fixed = torch.tensor(
        [tid in special_ids for tid in input_ids[0].tolist()],
        device=device, dtype=torch.bool,
    )

    attr_rank = torch.tensor(importance, device=device, dtype=torch.float32)
    attr_rank[fixed] = -float("inf")
    k = max(1, int((~fixed).sum().item() * topk_pct / 100))
    topk_idx = torch.topk(attr_rank, k, sorted=False).indices

    # log-odds
    X_lo = X.clone(); X_lo[0, topk_idx] = base_emb
    prob_lo = _forward_prob_emb(clf_model, X_lo, attention_mask, pred_id, device)
    log_odd = (torch.log(prob_lo + 1e-10) - torch.log(prob_orig + 1e-10)).item()

    # comprehensiveness
    X_comp = X.clone(); X_comp[0, topk_idx] = base_emb
    prob_comp = _forward_prob_emb(clf_model, X_comp, attention_mask, pred_id, device)
    comp = (prob_orig - prob_comp).item()

    # sufficiency
    keep = torch.zeros(L, dtype=torch.bool, device=device)
    keep[topk_idx] = True; keep[fixed] = True
    X_suff = X.clone(); X_suff[0, ~keep] = base_emb
    prob_suff = _forward_prob_emb(clf_model, X_suff, attention_mask, pred_id, device)
    suff = (prob_orig - prob_suff).item()

    return log_odd, comp, suff, pred_id


# ── Public API ─────────────────────────────────────────────────────────────

def reagent_classification(
    sentence: str,
    model_name: str,
    top_k: int = 3,
    topk_pct: int = 20,
    mlm_name: str = MLM_ORACLE,
    show_special_tokens: bool = False,
    device: str | None = None,
) -> dict:
    """
    Run ReAGent-style feature attribution on a BERT-based classifier.

    Args:
        sentence:            Input text to explain.
        model_name:          HuggingFace classifier model name/path.
        top_k:               Number of MLM replacement candidates per token.
                             Original paper uses top_k=3 (config: top3_replace0.1).
        topk_pct:            Percentage of tokens to ablate for eval metrics.
        mlm_name:            MLM oracle for generating replacements.
                             Defaults to "roberta-base" (same as original).
        show_special_tokens: Whether to include [CLS]/[SEP] in output.
        device:              Torch device; auto-detected if None.

    Returns dict with keys:
        tokens       — list[str]
        attributions — list[float]  Hellinger importance per token
        predicted_label — int
        log_odd, comp, suff — float  faithfulness metrics
        time         — float  wall-clock seconds
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()

    # ── Load models ──
    clf_tokenizer, clf_model = _load_clf(model_name, device)
    mlm_tokenizer, mlm_model = _load_mlm(mlm_name, device)

    # ── Tokenize input ──
    enc = clf_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        return_special_tokens_mask=True,
    )
    input_ids      = enc["input_ids"]        # (1, L)
    attention_mask = enc["attention_mask"]   # (1, L)

    # Decoded token strings (used by MLM oracle and output)
    clf_tokens = clf_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # ── Original label distribution ──
    p_orig = _get_label_dist(clf_model, clf_tokenizer,
                             input_ids, attention_mask, device)

    # ── Per-token Hellinger importance ──
    importance = _compute_importance_scores(
        clf_tokenizer, clf_model,
        mlm_tokenizer, mlm_model,
        input_ids, attention_mask, clf_tokens,
        p_orig, top_k, device,
    )

    # ── Faithfulness metrics ──
    log_odd, comp, suff, pred_id = _compute_faithfulness_metrics(
        clf_model, clf_tokenizer,
        input_ids, attention_mask,
        importance, device, topk_pct,
    )

    t1 = time.perf_counter()

    # ── Build output (optionally filter specials) ──
    special_ids = set(clf_tokenizer.all_special_ids)
    out_tokens = []
    out_attr   = []
    for tok_str, tok_id, imp in zip(
        clf_tokens, input_ids[0].tolist(), importance.tolist()
    ):
        if not show_special_tokens and tok_id in special_ids:
            continue
        out_tokens.append(tok_str)
        out_attr.append(imp)

    return {
        "tokens":          out_tokens,
        "attributions":    out_attr,
        "predicted_label": pred_id,
        "log_odd":         log_odd,
        "comp":            comp,
        "suff":            suff,
        "time":            t1 - t0,
    }


# ── Benchmark loop (matches PACE main script structure) ───────────────────

def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAMES[(args.model, args.dataset)]
    print(f"Device    : {device}")
    print(f"Classifier: {model_name}")
    print(f"MLM oracle: {MLM_ORACLE}  (top_k={args.top_k})")
    print(f"Dataset   : {args.dataset}")

    # Pre-load both models
    _load_clf(model_name, device)
    _load_mlm(MLM_ORACLE, device)

    if args.dataset == "imdb":
        dataset = load_dataset("imdb")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))
        data    = random.sample(data, min(args.num_samples * 2, len(data)))
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")["validation"]
        data    = list(zip(dataset["sentence"], dataset["label"]))
    elif args.dataset == "rotten":
        dataset = load_dataset("rotten_tomatoes")["test"]
        data    = list(zip(dataset["text"], dataset["label"]))

    if len(data) > args.num_samples:
        data = random.sample(data, args.num_samples)
    print(f"Samples   : {len(data)}")

    log_odds = comps = suffs = total_time = 0.0
    count = errors = 0
    print_step = 100

    for row in tqdm(data):
        text = row[0]
        try:
            res = reagent_classification(
                sentence=text,
                model_name=model_name,
                top_k=args.top_k,
                topk_pct=args.topk_pct,
                device=device,
                show_special_tokens=False,
            )
            log_odds   += res["log_odd"]
            comps      += res["comp"]
            suffs      += res["suff"]
            total_time += res["time"]
            count      += 1

            if count % print_step == 0:
                print(
                    f"\n[{count}/{len(data)}]"
                    f"  log-odds={log_odds/count:.4f}"
                    f"  comp={comps/count:.4f}"
                    f"  suff={suffs/count:.4f}"
                    f"  time={total_time/count:.4f}s"
                )
        except Exception:
            errors += 1
            if errors <= 5:
                import traceback; traceback.print_exc()

    if count > 0:
        print(f"\n{'─'*52}")
        print(f"ReAGent-clf (top_k={args.top_k})  |  {args.model} / {args.dataset}")
        print(f"  Log-odds         : {log_odds/count:.6f}")
        print(f"  Comprehensiveness: {comps/count:.6f}")
        print(f"  Sufficiency      : {suffs/count:.6f}")
        print(f"  Avg time/sample  : {total_time/count:.4f}s")
        print(f"  Evaluated        : {count}  |  Errors: {errors}")
        print(f"{'─'*52}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["distilbert", "bert", "roberta"],
                        default="distilbert")
    parser.add_argument("--dataset", choices=["sst2", "imdb", "rotten"],
                        default="sst2")
    parser.add_argument("--top_k",    type=int, default=3,
                        help="MLM replacement candidates per token (paper default: 3)")
    parser.add_argument("--topk_pct", type=int, default=20,
                        help="Top-%% tokens to ablate for eval metrics")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    run_benchmark(args)