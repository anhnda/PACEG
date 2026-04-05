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
import random
import argparse
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

# Match AttCAT / PACE SDP settings so numerics are identical when run standalone
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ── Shared classifier cache ────────────────────────────────────────────────
# AttCAT uses a module-level `cache` dict keyed by model_name → {"model":..., "tokenizer":...}
# We import and reuse it so both methods share one loaded classifier instance,
# saving GPU memory and avoiding a redundant download/reload.
try:
    from attcat_eval_sentiment import cache as _attcat_cache
except ImportError:
    _attcat_cache = {}   # running standalone — use our own empty dict

_clf_cache = _attcat_cache   # same object, not a copy

# ── Separate cache only for the MLM oracle ────────────────────────────────
# MLM cache: keyed by model_name (same checkpoint as the classifier).
# AutoModelForMaskedLM reads from the same local cache — no extra download.
_mlm_cache: dict = {}

# MLM oracle = the classifier checkpoint itself.
# AutoModelForMaskedLM reads the same cached files as AutoModelForSequenceClassification
# and simply ignores the classification head — zero extra download,
# identical to how AttCAT/PACE reuse the same model_name.

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


def _resolve_mlm_name(model_family: str, clf_model_name: str) -> str:
    # Always the classifier checkpoint itself — no separate download ever.
    return clf_model_name


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
    """
    Load classifier, reusing AttCAT's shared cache dict format:
        _clf_cache[model_name] = {"model": ..., "tokenizer": ...}
    If AttCAT already loaded this model we get it for free — no re-download.
    """
    if model_name not in _clf_cache:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        mdl.eval()
        _clf_cache[model_name] = {"model": mdl, "tokenizer": tok}
    entry = _clf_cache[model_name]
    return entry["tokenizer"], entry["model"]


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
    clf_tokenizer,                     # BUG FIX: passed explicitly, no cache hack
    mlm_tokenizer,
    mlm_model,
    input_ids_clf: torch.Tensor,      # (1, L) — classifier tokenizer ids
    position: int,                     # which token to replace
    clf_tokens: list[str],             # decoded token strings (clf vocab)
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
      1. Convert the classifier token sequence back to a plain string,
         inserting the MLM [MASK] token at `position`.
      2. Re-encode with the MLM tokenizer.
      3. Run the MLM, collect top-k token strings from the MLM vocabulary.
      4. Re-encode those strings with the *classifier* tokenizer to get
         ids that can be substituted into input_ids_clf.
    """
    clf_special_tokens = set(clf_tokenizer.all_special_tokens)
    mlm_special_tokens = set(mlm_tokenizer.all_special_tokens)

    # Build masked token list using the CLF token strings, substituting
    # the MLM's own mask token at `position`.
    masked_tokens = []
    for i, tok in enumerate(clf_tokens):
        if tok in clf_special_tokens:
            continue                              # skip [CLS]/[SEP]/[PAD]
        if i == position:
            masked_tokens.append(mlm_tokenizer.mask_token)
        else:
            masked_tokens.append(tok)

    # Convert to a plain string — handles Ġ (RoBERTa) and ## (BERT) prefixes
    masked_text = mlm_tokenizer.convert_tokens_to_string(masked_tokens)

    mlm_enc = mlm_tokenizer(
        masked_text, return_tensors="pt", truncation=True
    ).to(device)

    # Locate the [MASK] position in the re-encoded MLM input
    mask_token_id = mlm_tokenizer.mask_token_id
    mask_positions = (mlm_enc["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        # Fallback: mask disappeared after re-tokenisation → keep original token
        return [input_ids_clf[0, position].item()]

    mask_pos = mask_positions[0].item()

    with torch.no_grad():
        mlm_logits = mlm_model(**mlm_enc).logits[0]   # (L_mlm, V_mlm)

    # Over-sample top-k*3 to account for filtering below
    top_k_ids_mlm = mlm_logits[mask_pos].topk(top_k * 3).indices

    replacement_ids = []
    for mlm_id in top_k_ids_mlm.tolist():
        token_str = mlm_tokenizer.decode([mlm_id]).strip()
        if not token_str or token_str in mlm_special_tokens:
            continue
        # Re-encode with the classifier's tokenizer.
        # Keep only single-subword results so sequence length stays constant.
        clf_ids = clf_tokenizer.encode(token_str, add_special_tokens=False)
        if len(clf_ids) == 1:
            replacement_ids.append(clf_ids[0])
        if len(replacement_ids) >= top_k:
            break

    if not replacement_ids:
        # Nothing survived filtering — fall back to the original token
        replacement_ids = [input_ids_clf[0, position].item()]

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
            clf_tokenizer, mlm_tokenizer, mlm_model,
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


# ── Faithfulness metrics ───────────────────────────────────────────────────
# Uses the same xai_metrics + *_helper.py path as attcat_eval_sentiment.py
# and the PACE main script, so results are directly comparable.

from xai_metrics import (
    calculate_log_odds,
    calculate_comprehensiveness,
    calculate_sufficiency,
)


def _get_helper_fns(model_name: str):
    """
    Return (get_inputs, get_base_token_emb, nn_forward_func) from the
    appropriate *_helper module, matching the pattern used in attcat_eval
    and pace_gradients.
    """
    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, get_base_token_emb, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, get_base_token_emb, nn_forward_func
    else:
        raise NotImplementedError(f"No helper module for model: {model_name}")
    return get_inputs, get_base_token_emb, nn_forward_func


def _compute_faithfulness_metrics(
    clf_model,
    clf_tokenizer,
    model_name: str,
    sentence: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    importance: np.ndarray,
    device: str,
    topk_pct: int = 20,
) -> tuple[float, float, float, int]:
    """
    Compute log-odds, comprehensiveness, sufficiency via xai_metrics helpers.
    This matches the exact call pattern of attcat_eval_sentiment.py and
    pace_gradients.py, ensuring metrics are comparable across all methods.

    Returns (log_odd, comp, suff, pred_id).
    """
    get_inputs, get_base_token_emb, nn_forward_func = _get_helper_fns(model_name)

    # Get embedding matrix input (same as other eval scripts)
    embed = clf_model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids.to(device))                        # (1, L, D)
        logits0 = clf_model(
            inputs_embeds=X, attention_mask=attention_mask.to(device)
        ).logits[0]
    pred_id = int(logits0.argmax().item())

    base_token_emb = get_base_token_emb(clf_model, clf_tokenizer, device)
    inp = get_inputs(clf_model, clf_tokenizer, sentence, device)
    # inp unpacking: (input_ids, ref_ids, token_type_ids, ref_type_ids,
    #                 position_emb, ref_position_emb, type_emb, ref_type_emb, attention_mask)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    # Guarantee every tensor is on `device` (CUDA) — the model weights live there,
    # and nn_forward_func passes these directly into LayerNorm / attention.
    # get_inputs() and get_base_token_emb() may or may not honour `device`
    # depending on the helper implementation, so we enforce it explicitly.
    X              = X.to(device)
    position_embed = position_embed.to(device) if position_embed is not None else None
    type_embed     = type_embed.to(device)     if type_embed     is not None else None
    base_token_emb = base_token_emb.to(device)

    # attr_full drives topk() inside xai_metrics → produces index tensor on `device`.
    # attention_mask[0][mask] then requires attention_mask on the same device.
    attr_full        = torch.tensor(importance, dtype=torch.float32, device=device)
    attention_mask_d = attention_mask.to(device)

    log_odd, _ = calculate_log_odds(
        nn_forward_func, clf_model, X, position_embed, type_embed,
        attention_mask_d, base_token_emb, attr_full, topk=topk_pct,
    )
    comp = calculate_comprehensiveness(
        nn_forward_func, clf_model, X, position_embed, type_embed,
        attention_mask_d, base_token_emb, attr_full, topk=topk_pct,
    )
    suff = calculate_sufficiency(
        nn_forward_func, clf_model, X, position_embed, type_embed,
        attention_mask_d, base_token_emb, attr_full, topk=topk_pct,
    )

    return log_odd, comp, suff, pred_id


# ── Public API ─────────────────────────────────────────────────────────────

def reagent_classification(
    sentence: str,
    model_name: str,
    top_k: int = 3,
    topk_pct: int = 20,
    mlm_name: str | None = None,   # None → auto-resolved from model_name
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
                             Defaults to the classifier checkpoint itself (no extra download).
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

    # ── MLM oracle = classifier checkpoint itself (zero extra download) ──
    if mlm_name is None:
        mlm_name = model_name

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

    # ── Faithfulness metrics (via xai_metrics, same path as AttCAT/PACE) ──
    log_odd, comp, suff, pred_id = _compute_faithfulness_metrics(
        clf_model, clf_tokenizer, model_name, sentence,
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
    resolved_mlm = _resolve_mlm_name(args.model, model_name)
    print(f"MLM oracle: {resolved_mlm}  (top_k={args.top_k})")
    print(f"Dataset   : {args.dataset}")

    # Pre-load both models
    _load_clf(model_name, device)
    _load_mlm(resolved_mlm, device)

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