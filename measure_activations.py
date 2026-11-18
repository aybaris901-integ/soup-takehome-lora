"""
Check the activation term of memory_budget.py against what autograd really saves.

Builds randomly initialised Qwen2 blocks with the REAL Qwen2.5-1.5B widths, wraps them
in LoRA exactly like the training run, and counts the bytes of every tensor autograd
saves for backward (saved_tensors_hooks, unique storages). Running it with 2 and 4
layers isolates the per-layer cost from the embedding + logits cost.

Runs on CPU (bf16, 2-byte activations like fp16) or on the Colab GPU (fp16).
    python measure_activations.py [--tokens 64] [--batch 2]
"""
import argparse
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

import memory_budget as mb

SHAPES = None  # set to a list to record what was saved


def saved_bytes(model, ids, labels):
    # parameter storages are saved by reference for backward but are not NEW memory
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen, total = set(params), [0]

    def pack(t):
        try:
            ptr, nbytes = t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()
        except Exception:
            return t
        if ptr not in seen:
            seen.add(ptr)
            total[0] += nbytes
            if SHAPES is not None:
                SHAPES.append((tuple(t.shape), str(t.dtype), nbytes))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(input_ids=ids, labels=labels)
    return total[0], out.loss


def build(layers, dtype, device, dropout):
    c = dict(mb.QWEN25_15B)
    cfg = Qwen2Config(hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
                      num_hidden_layers=layers, num_attention_heads=c["num_attention_heads"],
                      num_key_value_heads=c["num_key_value_heads"], vocab_size=c["vocab_size"],
                      tie_word_embeddings=True, max_position_embeddings=512)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).to(dtype=dtype, device=device)
    lc = LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout, target_modules=mb.TARGETS, task_type="CAUSAL_LM")
    m = get_peft_model(m, lc)
    for n, p in m.named_parameters():          # trainable LoRA params in fp32, like the Trainer
        if p.requires_grad:
            p.data = p.data.float()
    m.train()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.05)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    ids = torch.randint(0, 151936, (a.batch, a.tokens), device=device)
    labels = ids.clone()
    labels[:, : a.tokens // 2] = -100          # prompt masked, as in SFT

    res = {}
    for L in (2, 4):
        m = build(L, dtype, device, a.dropout)
        with torch.autocast(device_type=device, dtype=dtype):
            b, _ = saved_bytes(m, ids, labels)
        res[L] = b
        del m
    ntok = a.batch * a.tokens
    per_layer_tok = (res[4] - res[2]) / 2 / ntok
    rest_tok = (res[2] - 2 * per_layer_tok * ntok) / ntok
    pred = mb.activation_bytes_per_token_layer(mb.QWEN25_15B, 16, mb.TARGETS, a.dropout)
    out = {"device": device, "dtype": str(dtype), "tokens": ntok,
           "measured_bytes_per_token_per_layer": per_layer_tok,
           "predicted_bytes_per_token_per_layer": pred,
           "ratio_measured/predicted": per_layer_tok / pred,
           "measured_non_layer_bytes_per_token (embedding + logits + loss)": rest_tok,
           "predicted_saved_logit_bytes_per_token (fp32 log-softmax)": mb.QWEN25_15B["vocab_size"] * 4}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
