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

Key implementation note
-----------------------
h^l must stay IN the computation graph (no .detach()) so that
torch.autograd.grad(logit_c, h_l) actually returns non-zero gradients.
Attention weights can be safely detached (they are data, not differentiated).
"""

import time
import tqdm
import torch
import random
import argparse
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from xai_metrics import *   # log_odds, comprehensiveness, sufficiency

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
# Architecture helpers
# ---------------------------------------------------------------------------

def _get_encoder_layers(model):
    """Return the list of transformer encoder layer modules."""
    if hasattr(model, "bert"):
        return list(model.bert.encoder.layer)
    if hasattr(model, "distilbert"):
        return list(model.distilbert.transformer.layer)
    if hasattr(model, "roberta"):
        return list(model.roberta.encoder.layer)
    raise RuntimeError("Unsupported model architecture.")


def _get_attn_submodule(layer):
    """Return the self-attention sub-module of a transformer layer."""
    if hasattr(layer, "attention"):
        return layer.attention
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    return None


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

    Strategy
    --------
    1. Register forward hooks that capture:
       - h^l : the encoder-layer output TENSOR kept IN the computation graph
               (absolutely no .detach() here — that was the bug)
       - alpha^l : attention weights (detached, used only as scalar weights)
    2. Single forward pass with torch.enable_grad().
    3. For each layer l:
         grad_h_l = torch.autograd.grad(logit_c, h_l, retain_graph=True)[0]
         cat_l    = grad_h_l * h_l.detach()       # [1, seq, d]
         attcat_l = einsum(alpha_l, cat_l)         # [1, seq, d]
         scores  += attcat_l[0].sum(-1)            # accumulate [seq]

    Returns
    -------
    dict with keys: tokens, attributions, pred_class, log_odd, comp, suff, time
    """
    t0 = time.time()
    tokenizer, model = _load_model(model_name, device)

    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    seq_len        = input_ids.shape[1]

    # ---------------------------------------------------------------- hooks
    hidden_states_list: List[torch.Tensor] = []   # in-graph [1, seq, d]
    attn_weights_list:  List[torch.Tensor] = []   # detached  [1, H, seq, seq]
    hooks = []

    encoder_layers = _get_encoder_layers(model)

    def make_layer_hook(idx: int):
        def fn(module, inp, out):
            # DistilBERT TransformerBlock returns a tuple whose elements can be:
            #   (attn_weights [1,H,seq,seq],  ffn_output [1,seq,d])
            # We want the 3-D hidden-state tensor, not the 4-D attention matrix.
            # Strategy: pick the last element that is 3-D [B, seq, d].
            if isinstance(out, tuple):
                h = None
                for t in reversed(out):
                    if isinstance(t, torch.Tensor) and t.dim() == 3:
                        h = t
                        break
                if h is None:          # fallback: first element
                    h = out[0]
            else:
                h = out
            # *** Keep h in the computation graph — NO detach ***
            hidden_states_list.append(h)
        return fn

    def make_attn_hook(idx: int):
        def fn(module, inp, out):
            # Attention module returns (context, attn_weights, ...) for
            # BERT/RoBERTa when output_attentions=True.
            # DistilBERT MultiHeadSelfAttention returns (context, attn_weights).
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_weights_list.append(out[1].detach())  # [1, H, seq, seq]
        return fn

    for idx, layer in enumerate(encoder_layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
        attn_mod = _get_attn_submodule(layer)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(make_attn_hook(idx)))

    # -------------------------------------------------------------- forward
    with torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    logits     = outputs.logits
    pred_class = int(logits.argmax(dim=-1).item())
    target     = logits[0, pred_class]   # scalar, still in graph

    # Fallback hidden states (e.g. architecture not covered above)
    if len(hidden_states_list) == 0:
        if outputs.hidden_states is not None:
            hidden_states_list = list(outputs.hidden_states[1:])
        else:
            raise RuntimeError("No hidden states captured.")

    # Fallback attention weights
    if len(attn_weights_list) == 0 and outputs.attentions is not None:
        attn_weights_list = [
            a.detach() for a in outputs.attentions if a is not None
        ]

    n_layers = len(hidden_states_list)

    # --------------------------------------------------------- AttCAT scores
    attcat_scores = torch.zeros(seq_len, device=device)

    for l_idx in range(n_layers):
        h_l = hidden_states_list[l_idx]   # [1, seq, d] — IN GRAPH

        try:
            (grad_h_l,) = torch.autograd.grad(
                target, h_l,
                retain_graph=True,   # keep graph alive for remaining layers
                create_graph=False,
                allow_unused=False,
            )
        except RuntimeError:
            continue   # h_l not differentiable from target for this layer

        if grad_h_l is None:
            continue

        # CAT^l = grad ⊙ h  (no ReLU — preserve directionality per paper)
        # Squeeze batch dim (always 1) to get clean [seq, d] tensors
        cat_l = (grad_h_l * h_l.detach()).view(-1, grad_h_l.shape[-1])  # [seq, d]

        # AttCAT^l_i = mean_H( sum_j alpha_{i,j} * cat_j^l )
        if l_idx < len(attn_weights_list):
            alpha_l = attn_weights_list[l_idx].squeeze(0)  # [H, seq_q, seq_k]
            # 'hij,jd->hid': for each head h, query i attends over keys j
            attcat_l = torch.einsum(
                "hij,jd->hid", alpha_l, cat_l
            ).mean(dim=0)              # mean over heads -> [seq, d]
        else:
            attcat_l = cat_l           # plain CAT fallback

        attcat_scores = attcat_scores + attcat_l.sum(dim=-1)   # [seq]

    # ----------------------------------------------------------- token filter
    tokens_raw = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    if show_special_tokens:
        tokens = tokens_raw
        scores = attcat_scores
    else:
        special = {
            tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
            "<s>", "</s>", "<pad>", None,
        }
        keep   = [t not in special for t in tokens_raw]
        tokens = [t for t, k in zip(tokens_raw, keep) if k]
        scores = attcat_scores[torch.tensor(keep, device=device).bool()]

    # ------------------------------------------------------- faithfulness metrics
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

    return {
        "tokens":       tokens,
        "attributions": scores,
        "pred_class":   pred_class,
        "log_odd":      log_odd_val,
        "comp":         comp_val,
        "suff":         suff_val,
        "time":         time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AttCAT attributions on sentiment datasets."
    )
    parser.add_argument("--model", type=str, default="distilbert",
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sst2", "imdb", "rotten"])
    parser.add_argument("--n_samples",  type=int, default=2000)
    parser.add_argument("--print_step", type=int, default=100)
    args = parser.parse_args()

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
    model_name = MODEL_MAP[args.model][args.dataset]
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model  : {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Device : {device}")

    # ---------------------------------------------------------------- demo
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

    # --------------------------------------------------------------- dataset
    print("\nLoading dataset ...")
    if args.dataset == "imdb":
        ds   = load_dataset("imdb")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(args.n_samples, len(data)))
    elif args.dataset == "sst2":
        ds   = load_dataset("glue", "sst2")["validation"]
        data = list(zip(ds["sentence"], ds["label"], ds["idx"]))
    elif args.dataset == "rotten":
        ds   = load_dataset("rotten_tomatoes")["test"]
        data = list(zip(ds["text"], ds["label"]))
        data = random.sample(data, min(args.n_samples, len(data)))

    print(f"Evaluating {len(data)} samples with AttCAT ...\n")

    log_odds_sum = comps_sum = suffs_sum = total_time_sum = 0.0
    count = 0

    for row in tqdm.tqdm(data):
        text = row[0]
        try:
            res = attcat_classification(
                text, model_name=model_name,
                show_special_tokens=False, device=device,
            )
            log_odds_sum   += res["log_odd"]
            comps_sum      += res["comp"]
            suffs_sum      += res["suff"]
            total_time_sum += res["time"]
            count += 1
        except Exception:
            continue

        if count % args.print_step == 0:
            print(
                f"[{count:>5d}]  "
                f"Log-odds: {log_odds_sum / count:.4f}  "
                f"Comp: {comps_sum / count:.4f}  "
                f"Suff: {suffs_sum / count:.4f}  "
                f"Time/sample: {total_time_sum / count:.4f}s"
            )

    print("\n=== Final Results ===")
    n = max(count, 1)
    print(
        f"Log-odds         : {log_odds_sum / n:.4f}\n"
        f"Comprehensiveness: {comps_sum / n:.4f}\n"
        f"Sufficiency      : {suffs_sum / n:.4f}\n"
        f"Time/sample      : {total_time_sum / n:.4f}s\n"
        f"Total samples    : {count}"
    )