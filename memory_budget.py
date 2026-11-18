"""
Part 1 - VRAM budget, written BEFORE training.

Itemised estimate for LoRA SFT, fp16 merge and bnb-nf4 quantization of
Qwen2.5-1.5B-Instruct on a T4 (16 GB, no bf16). Every line is a formula over
the model config, so the same script re-runs with the real config.json and
the real token lengths on Colab:

    python memory_budget.py                                  # built-in Qwen2.5-1.5B config
    python memory_budget.py --config /path/config.json --tokens tokens.json --batch 8

--tokens is a JSON list of per-row token counts (the notebook writes it from
the real tokenizer + chat template). Without it a conservative 96 tokens/row
is assumed.

Units: GiB = 2**30 bytes. torch.cuda.max_memory_allocated() also reports
bytes, so compare like with like; nvidia-smi additionally shows the CUDA
context and the caching allocator's reserved-but-free pool.
"""
import argparse
import json

GiB = 2 ** 30
MiB = 2 ** 20

QWEN25_15B = {  # Qwen/Qwen2.5-1.5B-Instruct config.json (published values)
    "hidden_size": 1536, "intermediate_size": 8960, "num_hidden_layers": 28,
    "num_attention_heads": 12, "num_key_value_heads": 2, "vocab_size": 151936,
    "tie_word_embeddings": True,
}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def shapes(c):
    """(in, out) of every linear layer in one decoder block."""
    h, ff = c["hidden_size"], c["intermediate_size"]
    kv = h // c["num_attention_heads"] * c["num_key_value_heads"]
    return {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
            "gate_proj": (h, ff), "up_proj": (h, ff), "down_proj": (ff, h)}, kv


def param_counts(c):
    sh, _ = shapes(c)
    h, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    linear = sum(i * o for i, o in sh.values()) * L
    # q/k/v carry biases in Qwen2; 2 RMSNorm vectors per layer + final norm
    bias = (h + 2 * shapes(c)[1]) * L
    norms = 2 * h * L + h
    emb = V * h
    head = 0 if c.get("tie_word_embeddings") else V * h
    return {"linear": linear, "bias": bias, "norms": norms, "embedding": emb, "lm_head": head,
            "total": linear + bias + norms + emb + head}


def lora_params(c, r, targets):
    sh, _ = shapes(c)
    return sum(r * (sh[t][0] + sh[t][1]) for t in targets) * c["num_hidden_layers"]


def activation_bytes_per_token_layer(c, r, targets, lora_dropout, act_bytes=2):
    """Tensors autograd keeps for backward in ONE decoder block, per token, without
    gradient checkpointing. Frozen base weights still need their INPUTS saved,
    because the gradient must flow through them to earlier LoRA layers."""
    sh, kv = shapes(c)
    h, ff = c["hidden_size"], c["intermediate_size"]
    e = 0
    e += 2 * h            # input_layernorm: saved input + normalized output
    e += 2 * h            # post_attention_layernorm: same
    e += h + 2 * kv       # q, k, v outputs before RoPE
    e += h + 2 * kv       # q, k after RoPE (+ v passed through) fed to attention
    e += h                # attention output (input to o_proj)
    e += 2 * h            # two residual stream copies
    e += 4 * ff           # gate out, up out, silu(gate), silu(gate)*up (input to down_proj)
    # LoRA branch: PEFT casts the adapter input to the adapter's dtype (fp32, because the
    # trainable params are kept in fp32) BEFORE dropout, so with p > 0 every target module
    # keeps its own fp32 copy of its input. Found by measure_activations.py: v0 of this
    # estimate counted these at 2 bytes and came out 19% low.
    if lora_dropout > 0:
        lora_copies = sum(sh[t][0] for t in targets) * 4
        mask = sum(sh[t][0] for t in targets)   # bool dropout masks, 1 byte each
    else:
        lora_copies, mask = 0, 0
    e += r * len(targets)  # A x, the rank-r intermediate of every adapter
    return e * act_bytes + lora_copies + mask


def training_budget(c, tokens, batch, r, targets, lora_dropout, grad_ckpt):
    p = param_counts(c)
    n_lora = lora_params(c, r, targets)
    L, h, V = c["num_hidden_layers"], c["hidden_size"], c["vocab_size"]
    T = sum(tokens) / len(tokens)                 # mean tokens per row
    Tmax = max(tokens)                            # batches are padded to their longest row
    pad_T = min(Tmax, T * 1.35)                   # expected padded length (calibrated by notebook)
    btok = batch * pad_T                          # tokens materialised per micro-batch

    out = {}
    out["base weights fp16 (frozen)"] = p["total"] * 2
    out["LoRA weights fp32"] = n_lora * 4
    out["LoRA grads fp32"] = n_lora * 4
    out["AdamW states fp32 (m, v)"] = n_lora * 8
    per_tl = activation_bytes_per_token_layer(c, r, targets, lora_dropout)
    if grad_ckpt:
        out["activations (ckpt: block inputs + 1 recomputed block)"] = btok * (L * h * 2 + per_tl)
    else:
        out["activations (all 28 blocks)"] = btok * L * per_tl
    # Logits for EVERY position (prompt included, the mask only zeroes the loss):
    # fp16 matmul output (2 B) + fp32 upcast (4 B) + fp32 log-softmax saved for
    # backward (4 B, measured) -> ~10 B per token per vocab entry at the forward
    # peak; backward holds the saved 4 B plus an fp32 grad (4 B) and an fp16 grad (2 B).
    out["logits (fp16 out + fp32 upcast + fp32 log-softmax)"] = btok * V * (2 + 4 + 4)
    out["embedding-out grad / misc workspace (cuBLAS etc.)"] = 256 * MiB
    total = sum(out.values())
    return out, total, {"mean_tokens": T, "max_tokens": Tmax, "padded_tokens_per_row": pad_T,
                        "tokens_per_microbatch": btok, "lora_params": n_lora, **p}


def merge_budget(c, on_gpu=True):
    p = param_counts(c)
    out = {"base weights fp16": p["total"] * 2 if on_gpu else 0,
           "LoRA A,B fp16 (loaded)": 0,  # filled by caller
           "largest temp (B@A for down/gate, fp32)": 0}
    return out, p


def quant_budget(c):
    p = param_counts(c)
    h, V = c["hidden_size"], c["vocab_size"]
    out = {}
    out["4-bit linear weights (0.5 B/param)"] = p["linear"] * 0.5
    # absmax per 64-weight block; with double quant: 8-bit absmax + fp32 per 256 blocks
    out["nf4 absmax (double-quant)"] = p["linear"] / 64 * 1 + p["linear"] / 64 / 256 * 4
    out["embedding fp16 (not quantized, tied head)"] = p["embedding"] * 2
    out["biases + norms fp16"] = (p["bias"] + p["norms"]) * 2
    steady = sum(out.values())
    # while loading, each fp16 weight arrives on GPU before being packed; the
    # largest single tensor in flight is the embedding (V x h, fp16)
    transient = V * h * 2
    return out, steady, transient


def fmt(b):
    return f"{b / GiB:7.3f} GiB" if b >= 0.1 * GiB else f"{b / MiB:7.1f} MiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--tokens", help="JSON list of per-row token counts")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    c = json.load(open(a.config)) if a.config else QWEN25_15B
    tokens = json.load(open(a.tokens)) if a.tokens else [96]
    rep = {}

    print("=" * 72)
    print("TRAINING  (LoRA, fp16 frozen base, fp16 autocast, AdamW)")
    items, total, info = training_budget(c, tokens, a.batch, a.r, TARGETS, a.lora_dropout, a.grad_ckpt)
    print(f"params total {info['total']/1e9:.3f} B  (linear {info['linear']/1e9:.3f} B, "
          f"embedding {info['embedding']/1e6:.0f} M, tied head: {c.get('tie_word_embeddings')})")
    print(f"LoRA r={a.r} on {len(TARGETS)} modules x {c['num_hidden_layers']} layers = "
          f"{info['lora_params']/1e6:.2f} M trainable params")
    print(f"tokens/row mean {info['mean_tokens']:.1f}, max {info['max_tokens']}, "
          f"~padded {info['padded_tokens_per_row']:.1f}; micro-batch {a.batch} -> "
          f"{info['tokens_per_microbatch']:.0f} tokens")
    for k, v in items.items():
        print(f"  {k:58s} {fmt(v)}")
    ctx = 0.35 * GiB
    print(f"  {'= predicted torch peak allocated':58s} {fmt(total)}")
    print(f"  {'+ CUDA context (in nvidia-smi only, not in torch)':58s} {fmt(ctx)}")
    rep["train"] = {"items": items, "total_alloc": total, "nvidia_smi_estimate": total + ctx, "info": info}

    print("=" * 72)
    print("MERGE  (soup merge: base loaded with dtype=float16, device_map='cpu')")
    print("  GPU: 0 by construction - Soup merges on the CPU. The peak is in HOST RAM:")
    p = param_counts(c)
    n_lora = lora_params(c, a.r, TARGETS)
    sh, _ = shapes(c)
    biggest = max(sh[t][0] * sh[t][1] for t in TARGETS)
    m = {"base weights fp16 (hub bf16 cast to fp16 on load)": p["total"] * 2,
         "adapter fp32 as saved": n_lora * 4,
         "largest delta B@A in flight (fp32 on CPU)": biggest * 4,
         "safetensors write buffer (one shard, ~= model)": p["total"] * 2}
    for k, v in m.items():
        print(f"  {k:58s} {fmt(v)}")
    mt = sum(m.values())
    print(f"  {'= predicted host-RAM peak (excl. Python/torch ~1 GiB)':58s} {fmt(mt)}")
    rep["merge"] = {"items": m, "total_alloc": mt}

    print("=" * 72)
    print("QUANTIZE  (bitsandbytes NF4, double quant, fp16 compute)")
    q, steady, transient = quant_budget(c)
    for k, v in q.items():
        print(f"  {k:58s} {fmt(v)}")
    print(f"  {'= steady-state 4-bit model':58s} {fmt(steady)}")
    print(f"  {'+ transient fp16 tensor while packing (embedding)':58s} {fmt(transient)}")
    print(f"  {'= predicted peak during quantize/load':58s} {fmt(steady + transient)}")
    rep["quant"] = {"items": q, "steady": steady, "peak": steady + transient}

    if a.json_out:
        json.dump(rep, open(a.json_out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
