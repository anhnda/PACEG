"""
attcat_eval_sentiment.py

Evaluation of AttCAT (Attentive Class Activation Tokens) attributions on
sentiment classification datasets, following the structure of the PACE
gradient evaluation script.

AttCAT paper: "AttCAT: Explaining Transformers via Attentive Class Activation
Tokens", Qiang et al., NeurIPS 2022.
https://github.com/qiangyao1988/AttCAT

AttCAT Algorithm (per token i, summed over all L layers):
  1. CAT_i^l  = grad(h_i^l) ⊙ h_i^l           (Hadamard product, no ReLU)
  2. AttCAT_i^l = mean_over_heads( alpha_i^l @ CAT_i^l )
                                                (attention-weighted CAT)
  3. score_i  = sum_l  sum_d  AttCAT_i^l        (scalar per token)

where:
  h_i^l    : hidden state of token i at layer l  [d]
  grad(h)  : gradient of predicted-class logit w.r.t. h
  alpha_i^l: attention weights of token i at layer l  [n_heads, n_tokens]
"""

import json
import time
import tqdm
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from xai_metrics import *   # log_odds, comprehensiveness, sufficiency helpers

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ---------------------------------------------------------------------------
# Model / tokenizer cache
# ---------------------------------------------------------------------------
_model_cache: Dict[str, Tuple] = {}


def _load_model(model_name: str, device: str):
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
        ).to(device)
        model.eval()
        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Core AttCAT computation
# ---------------------------------------------------------------------------

def attcat_classification(
    sentence: str,
    model_name: str,
    show_special_tokens: bool = False,
    device: str = "cpu",
) -> Dict:
    """
    Compute AttCAT token attributions for a single sentence.

    Returns
    -------
    dict with keys:
        tokens        : list[str]   – tokens (special tokens optionally excluded)
        attributions  : Tensor      – per-token AttCAT scalar score
        pred_class    : int         – predicted class index
        log_odd       : float       – log-odds faithfulness metric
        comp          : float       – comprehensiveness
        suff          : float       – sufficiency
        time          : float       – wall-clock seconds
    """
    t0 = time.time()
    tokenizer, model = _load_model(model_name, device)

    # ------------------------------------------------------------------ encode
    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = encoding["input_ids"].to(device)           # [1, seq]
    attention_mask = encoding["attention_mask"].to(device)

    # ------------------------------------------------------- forward + hooks
    # We need hidden states at every encoder layer AND attention weights.
    # We collect them via register_forward_hook so we can call backward later.

    hidden_states_list: List[torch.Tensor] = []   # one per layer, [1, seq, d]
    attn_weights_list: List[torch.Tensor] = []    # one per layer, [1, H, seq, seq]

    hooks = []

    # Detect architecture-specific layer attribute
    if hasattr(model, "bert"):
        encoder_layers = model.bert.encoder.layer
        layer_attr = "output"           # BertLayer output attribute name
    elif hasattr(model, "distilbert"):
        encoder_layers = model.distilbert.transformer.layer
        layer_attr = "output"
    elif hasattr(model, "roberta"):
        encoder_layers = model.roberta.encoder.layer
        layer_attr = "output"
    else:
        # Generic fallback: rely solely on model output hidden_states
        encoder_layers = []
        layer_attr = None

    # ----------------------------------- register hooks on each encoder layer
    def make_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            # out is a tuple; first element is the hidden state tensor
            h = out[0] if isinstance(out, tuple) else out
            h = h.detach().clone().requires_grad_(True)
            hidden_states_list.append(h)
        return hook_fn

    def make_attn_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            # attention module output: (context, attn_weights) or just context
            if isinstance(out, tuple) and len(out) >= 2:
                attn_w = out[1]          # [1, H, seq, seq]
                if attn_w is not None:
                    attn_weights_list.append(attn_w.detach().clone())
        return hook_fn

    # --------------------------------------------------------- hook attachment
    # We hook the full transformer layer output to get hidden states, and
    # the self-attention sub-module to get attention weights.
    for idx, layer in enumerate(encoder_layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))
        # Locate the self-attention sub-module
        if hasattr(layer, "attention"):
            attn_module = layer.attention
        elif hasattr(layer, "self_attn"):
            attn_module = layer.self_attn
        elif hasattr(layer, "multihead_attn"):
            attn_module = layer.multihead_attn
        else:
            attn_module = None
        if attn_module is not None:
            hooks.append(attn_module.register_forward_hook(make_attn_hook(idx)))

    # ------------------------------------------------------------- forward pass
    with torch.enable_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    for h in hooks:
        h.remove()

    logits = outputs.logits            # [1, num_classes]
    pred_class = logits.argmax(dim=-1).item()

    # --------------------------------------------------- fallback hidden states
    # If hooks captured nothing (unusual arch), use model output hidden_states
    if len(hidden_states_list) == 0 and outputs.hidden_states is not None:
        # hidden_states[0] is embedding layer, [1:] are transformer layers
        for hs in outputs.hidden_states[1:]:
            h_req = hs.detach().clone().requires_grad_(True)
            hidden_states_list.append(h_req)

    # ---------------------------------------------------- fallback attention
    if len(attn_weights_list) == 0 and outputs.attentions is not None:
        for aw in outputs.attentions:
            if aw is not None:
                attn_weights_list.append(aw.detach().clone())

    n_layers = len(hidden_states_list)
    seq_len  = input_ids.shape[1]

    # ------------------------------------------------ compute gradients per layer
    # For each layer hidden state h^l, compute grad(logit_c) w.r.t. h^l.
    # We perform separate backward passes (retain_graph) for efficiency.
    # AttCAT_i^l = mean_H( alpha_i^l  @  (grad_h^l ⊙ h^l) )  summed over d

    attcat_scores = torch.zeros(seq_len, device=device)  # [seq]

    target_logit = logits[0, pred_class]   # scalar

    for l_idx in range(n_layers):
        h_l = hidden_states_list[l_idx]   # [1, seq, d]
        h_l.requires_grad_(True)

        # Re-run only the classification head with this layer's hidden state
        # to get a clean gradient signal. We use the [CLS] token output.
        # For most BERT-style classifiers: logit = classifier(h_l[:, 0, :])
        # We approximate: gradient of original logit w.r.t. h_l.
        # Since h_l is a leaf (detached), we need to route through the head.
        try:
            # Try to pass through the classifier head
            if hasattr(model, "classifier"):
                cls_out = h_l[:, 0, :]         # [1, d]  CLS token
                # Handle dropout gracefully
                model.eval()
                with torch.enable_grad():
                    logit_l = model.classifier(cls_out)   # [1, num_classes]
                    score_l = logit_l[0, pred_class]
                    score_l.backward(retain_graph=(l_idx < n_layers - 1))
            elif hasattr(model, "pre_classifier"):
                # DistilBERT-style: pre_classifier → relu → dropout → classifier
                cls_out = h_l[:, 0, :]
                model.eval()
                with torch.enable_grad():
                    x = model.pre_classifier(cls_out)
                    x = torch.relu(x)
                    logit_l = model.classifier(x)
                    score_l = logit_l[0, pred_class]
                    score_l.backward(retain_graph=(l_idx < n_layers - 1))
            else:
                # Fallback: use the top-level logit gradient approximation
                target_logit.backward(retain_graph=True)
                if h_l.grad is None:
                    attcat_scores += 0.0
                    continue
        except Exception:
            # If anything fails, skip this layer
            continue

        grad_h_l = h_l.grad.clone()   # [1, seq, d]

        # CAT^l_i = grad ⊙ h  (no ReLU – we want directionality)
        cat_l = grad_h_l * h_l.detach()   # [1, seq, d]

        # AttCAT^l_i = mean_H( alpha^l_i · CAT^l )
        # alpha^l has shape [1, H, seq, seq]
        # alpha^l_i is row i of attention: [1, H, seq]
        # We weight each token j's CAT by alpha_{i,j}, sum over j → [1, H, d]
        # then mean over H → [1, d], sum over d → scalar per token i

        if l_idx < len(attn_weights_list):
            alpha_l = attn_weights_list[l_idx]   # [1, H, seq, seq]
            # alpha_l[:, :, i, :] is the attention of token i over all tokens
            # Weighted sum: sum_j alpha_{i,j} * cat_j^l
            # = einsum('bhij, bjd -> bhid') ... sum over j
            # cat_l: [1, seq, d] → [1, 1, seq, d]
            cat_l_exp = cat_l.unsqueeze(1)                   # [1, 1, seq, d]
            alpha_l_exp = alpha_l.unsqueeze(-1)              # [1, H, seq, seq, 1]
            # weighted_j = alpha[i,j] * cat[j]: sum over j
            # weighted_i: [1, H, seq, d]
            weighted = (alpha_l_exp * cat_l_exp.unsqueeze(2)).sum(dim=3)
            # mean over heads → [1, seq, d]
            attcat_l = weighted.mean(dim=1)
        else:
            # No attention weights available – fall back to plain CAT
            attcat_l = cat_l   # [1, seq, d]

        # Sum over hidden dimension → [seq]
        attcat_scores += attcat_l[0].sum(dim=-1)

        # Zero gradient for next iteration
        if h_l.grad is not None:
            h_l.grad.zero_()

    # ------------------------------------------- token decoding
    tokens_raw = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    if show_special_tokens:
        tokens = tokens_raw
        scores = attcat_scores
    else:
        special = {tokenizer.cls_token, tokenizer.sep_token,
                   tokenizer.pad_token, "<s>", "</s>", "<pad>"}
        mask = [tok not in special for tok in tokens_raw]
        tokens = [t for t, m in zip(tokens_raw, mask) if m]
        scores = attcat_scores[[m for m in mask]]

    # ------------------------------------------- faithfulness metrics
    # Re-use the helpers from xai_metrics (same interface as PACE eval)
    try:
        _attr_np = scores.detach().cpu().numpy()
        log_odd_val = log_odds(
            sentence, _attr_np, tokens, pred_class, model, tokenizer, device
        )
        comp_val = comprehensiveness(
            sentence, _attr_np, tokens, pred_class, model, tokenizer, device
        )
        suff_val = sufficiency(
            sentence, _attr_np, tokens, pred_class, model, tokenizer, device
        )
    except Exception:
        log_odd_val = comp_val = suff_val = 0.0

    elapsed = time.time() - t0

    return {
        "tokens":       tokens,
        "attributions": scores,
        "pred_class":   pred_class,
        "log_odd":      log_odd_val,
        "comp":         comp_val,
        "suff":         suff_val,
        "time":         elapsed,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AttCAT attributions on sentiment datasets."
    )
    parser.add_argument("--model",   type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"],
                        help="Backbone model family")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sst2", "imdb", "rotten"],
                        help="Evaluation dataset")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of random samples (for imdb/rotten)")
    parser.add_argument("--print_step", type=int, default=100,
                        help="Print running averages every N samples")
    args = parser.parse_args()

    # -------------------------------------------------- model name resolution
    model_key     = args.model
    dataset_name  = args.dataset

    MODEL_MAP = {
        "distilbert": {
            "sst2":   "distilbert-base-uncased-finetuned-sst-2-english",
            "imdb":   "textattack/distilbert-base-uncased-imdb",
            "rotten": "textattack/distilbert-base-uncased-rotten-tomatoes",
        },
        "bert": {
            "sst2":   "textattack/bert-base-uncased-SST-2",
            "imdb":   "textattack/bert-base-uncased-imdb",
            "rotten": "textattack/bert-base-uncased-rotten-tomatoes",
        },
        "roberta": {
            "sst2":   "textattack/roberta-base-SST-2",
            "imdb":   "textattack/roberta-base-imdb",
            "rotten": "textattack/roberta-base-rotten-tomatoes",
        },
    }
    model_name = MODEL_MAP[model_key][dataset_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model    : {model_name}")
    print(f"Dataset  : {dataset_name}")
    print(f"Device   : {device}")

    # -------------------------------------------------- quick sanity check
    demo_text = (
        "This is a really bad movie, although it has a promising start, "
        "it ended on a very low note."
    )
    print("\n--- AttCAT demo attribution ---")
    res_demo = attcat_classification(
        demo_text, model_name=model_name,
        show_special_tokens=False, device=device,
    )
    for tok, val in zip(res_demo["tokens"], res_demo["attributions"]):
        print(f"  {tok:>15s} : {val.item():+.6f}")

    # -------------------------------------------------- load dataset
    print("\nLoading dataset ...")
    if dataset_name == "imdb":
        ds   = load_dataset("imdb")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(args.n_samples, len(data)))

    elif dataset_name == "sst2":
        ds   = load_dataset("glue", "sst2")["validation"]   # test has no labels
        data = list(zip(ds["sentence"], ds["label"], ds["idx"]))

    elif dataset_name == "rotten":
        ds   = load_dataset("rotten_tomatoes")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(args.n_samples, len(data)))

    # -------------------------------------------------- evaluation loop
    print(f"Evaluating {len(data)} samples with AttCAT ...\n")

    log_odds_sum = comps_sum = suffs_sum = total_time_sum = 0.0
    count = 0
    print_step = args.print_step

    for row in tqdm.tqdm(data):
        text = row[0]
        try:
            res = attcat_classification(
                text, model_name=model_name,
                show_special_tokens=False, device=device,
            )
            log_odds_sum  += res["log_odd"]
            comps_sum     += res["comp"]
            suffs_sum     += res["suff"]
            total_time_sum += res["time"]
            count += 1
        except Exception as e:
            # Skip problematic samples silently
            continue

        if count % print_step == 0:
            print(
                f"[{count:>5d}]  "
                f"Log-odds: {log_odds_sum / count:.4f}  "
                f"Comp: {comps_sum / count:.4f}  "
                f"Suff: {suffs_sum / count:.4f}  "
                f"Time/sample: {total_time_sum / count:.4f}s"
            )

    print("\n=== Final Results ===")
    print(
        f"Log-odds      : {log_odds_sum / max(count,1):.4f}\n"
        f"Comprehensiveness: {comps_sum / max(count,1):.4f}\n"
        f"Sufficiency   : {suffs_sum / max(count,1):.4f}\n"
        f"Time/sample   : {total_time_sum / max(count,1):.4f}s\n"
        f"Total samples : {count}"
    )