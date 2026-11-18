"""
Part 2 - did the model actually train, and did the training survive merge and 4-bit?

A falling loss is evidence about the training process, not about the artifact a user
gets. This script inspects the artifacts themselves, at two levels:

WEIGHTS (no GPU needed)
  W1  exactly the LoRA target matrices changed between base and merged; nothing else
  W2  merged - base equals the delta the adapter predicts, (alpha/r) * B @ A, per matrix
  W3  every adapter layer has a non-zero B (Soup only refuses the all-zero case)
  W4  how much of the delta survives NF4 rounding (cosine of the 4-bit delta with B@A)

BEHAVIOUR (held-out data the model never saw, greedy decoding, each artifact's OWN
tokenizer, chat template and generation_config, exactly as a user would load it)
  B1  exact match on eval_real / eval_novel / val vs base, paired bootstrap 95% CI
  B2  teacher-forced log-likelihood of the gold answer (+ end-of-turn token)
  B3  stop rate (generation ended with EOS before the token budget) and grammar validity
  B4  merged == base+adapter (unmerged PEFT): agreement and NLL gap
  B5  4-bit vs merged: agreement, exact-match delta, KL of next-token distributions

A control artifact (same pipeline, a run that should NOT have learned) is pushed through
the same checks; the script is only trustworthy if the control FAILS.

Usage (Colab):
  python verify.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/main/adapter \
      --merged runs/main/merged --quant4 runs/main/merged-nf4 \
      --control-adapter runs/control/adapter --control-merged runs/control/merged \
      --out results/verify.json
"""
import argparse
import gc
import glob
import json
import math
import os
import random
import re
import sys
import time

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# --------------------------------------------------------------------------- grammar
ROOM = r"(kitchen|bedroom|living_room|bathroom|hall|office|garage)"
DELAY = r"( delay=\d+[smh])?"
CALL = re.compile("|".join([
    rf"light (on|off)( room={ROOM})?{DELAY}",
    rf"fan (on|off|speed=[1-3]){DELAY}",
    rf"relay [1-8] (on|off){DELAY}",
    r"led [a-z]+",
    r"grip( force=\d{1,3}| adjust=[+-]20)?",
    r"release", r"arm (home|cell=[1-9])",
    r"timer \d+[smh]( label=[a-z]+)?",
    r"sensor (temp|humidity|motion|light)",
    rf"status( (light|fan|arm))?( room={ROOM})?",
    r"cloud",
]))


def valid(out):
    return all(CALL.fullmatch(c) for c in out.split("|")) and out != ""


# --------------------------------------------------------------------------- utils
def log(*a):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)


def load_rows(path, n=None):
    rows = [json.loads(l) for l in open(path)]
    pairs = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    return pairs[:n] if n else pairs


def paired_bootstrap(a, b, iters=5000, seed=SEED):
    """95% CI of mean(b) - mean(a) for paired 0/1 vectors."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def free():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def local_dir(path_or_id):
    if os.path.isdir(path_or_id):
        return path_or_id
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


# --------------------------------------------------------------------------- loading
def load(kind, path, base=None):
    """kind: hf (fp16) | peft (base fp16 + unmerged adapter) | nf4 (saved bnb dir, or fp16 dir quantized on load)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    if kind == "peft":
        from peft import PeftModel
        try:
            tok = AutoTokenizer.from_pretrained(path)
        except (OSError, ValueError):   # adapter dir saved without a tokenizer
            tok = AutoTokenizer.from_pretrained(base)
        m = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(DEV)
        m = PeftModel.from_pretrained(m, path)
    elif kind == "nf4":
        tok = AutoTokenizer.from_pretrained(path)
        cfg = json.load(open(os.path.join(local_dir(path), "config.json")))
        if "quantization_config" in cfg:      # already a saved 4-bit artifact
            m = AutoModelForCausalLM.from_pretrained(path, device_map={"": 0})
        else:
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
            m = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, device_map={"": 0})
    else:
        tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(DEV)
    m.eval()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEV == "cuda" else float("nan")
    log(f"loaded {kind:4s} {path}  in {time.time()-t0:.1f}s  peak alloc {peak:.2f} GiB")
    return m, tok


def eos_ids(model, tok):
    g = getattr(model, "generation_config", None)
    e = getattr(g, "eos_token_id", None) if g is not None else None
    if e is None:
        e = tok.eos_token_id
    return set(e if isinstance(e, list) else [e])


def render(tok, text, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# Pure greedy decoding. Qwen2.5-Instruct ships generation_config with do_sample=True,
# temperature 0.7, top_p 0.8, top_k 20 and repetition_penalty 1.05-1.1; a bare
# do_sample=False would still apply the repetition penalty. The main metric switches it
# off; silent_failures.py P9 measures what the shipped generation_config does.
GREEDY = {"do_sample": False, "repetition_penalty": 1.0, "temperature": None, "top_p": None, "top_k": None}


@torch.no_grad()
def generate(model, tok, texts, system=None, bs=32, max_new=64, gen_kwargs=None):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stops = eos_ids(model, tok)
    outs, stopped = [], []
    for i in range(0, len(texts), bs):
        prompts = [render(tok, t, system) for t in texts[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        kw = GREEDY if gen_kwargs is None else gen_kwargs
        g = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
        for row in g[:, enc["input_ids"].shape[1]:].tolist():
            hit = next((j for j, t in enumerate(row) if t in stops), None)
            stopped.append(hit is not None)
            outs.append(tok.decode(row[:hit] if hit is not None else row, skip_special_tokens=True).strip())
    return outs, stopped


@torch.no_grad()
def gold_logprobs(model, tok, pairs, keep_dist=0, bs=16):
    """Per-example mean log p(gold token) over answer tokens + the end-of-turn token.
    For the first `keep_dist` examples also return full log-softmax rows (for KL)."""
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    end = tok.convert_tokens_to_ids("<|im_end|>")
    res, dists = [], []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        seqs, spans = [], []
        for text, gold in batch:
            p = tok(render(tok, text), add_special_tokens=False)["input_ids"]
            a = tok(gold, add_special_tokens=False)["input_ids"] + [end]
            seqs.append(p + a)
            spans.append((len(p), len(p) + len(a)))
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits.float()
        lp = torch.log_softmax(logits, -1)
        for j, (s, e) in enumerate(spans):
            tgt = ids[j, s:e].to(DEV)
            rows = lp[j, s - 1:e - 1]
            res.append(rows.gather(1, tgt[:, None]).mean().item())
            if len(dists) < keep_dist:
                dists.append(rows.half().cpu())
        del logits, lp
    return res, dists


def kl_mean(p_rows, q_rows):
    """mean over positions of KL(P || Q), rows are log-probs"""
    tot, n = 0.0, 0
    for p, q in zip(p_rows, q_rows):
        p, q = p.float(), q.float()
        tot += (p.exp() * (p - q)).sum(-1).sum().item()
        n += p.shape[0]
    return tot / max(n, 1)


# --------------------------------------------------------------------------- weight checks
def tensors(d):
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h = safe_open(f, "pt")
        for k in h.keys():
            out[k] = (f, k)
    return out


def get(idx, k, raw=False):
    from safetensors import safe_open
    f, key = idx[k]
    with safe_open(f, "pt") as h:
        t = h.get_tensor(key)
    return t if raw else t.float()


def adapter_deltas(adapter_dir):
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    idx = tensors(adapter_dir)
    mods = {}
    for k in idx:
        m = re.match(r"base_model\.model\.(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = k
    extra = [k for k in idx if "lora_" not in k]
    return cfg, scale, idx, mods, extra


def weight_checks(base, merged, adapter, quant4_model=None):
    log("weight checks: base vs merged vs adapter-predicted delta")
    bidx, midx = tensors(local_dir(base)), tensors(merged)
    cfg, scale, aidx, mods, extra = adapter_deltas(adapter)
    rep = {"lora_r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "scale": scale,
           "use_rslora": bool(cfg.get("use_rslora")), "target_modules": cfg.get("target_modules"),
           "adapter_non_lora_tensors": extra,
           "keys_only_in_base": sorted(set(bidx) - set(midx))[:20],
           "keys_only_in_merged": sorted(set(midx) - set(bidx))[:20]}

    changed, unchanged_targets, per = [], [], {}
    zero_b, max_rel_err = [], 0.0
    targets = {m + ".weight" for m in mods}
    # dtype conversion (e.g. hub bf16 -> merged fp16) changes EVERY tensor a little.
    # Compare merged against base cast to the merged dtype, and report the cast itself.
    cast = {"base_dtypes": set(), "merged_dtypes": set(), "elements": 0, "changed_by_cast": 0,
            "underflow_to_zero": 0, "overflow_to_inf": 0, "max_rel_cast_err": 0.0}
    lost, nonzero_pred = 0, 0
    for k in sorted(set(bidx) & set(midx)):
        wb_raw, wm_raw = get(bidx, k, raw=True), get(midx, k, raw=True)
        if wb_raw.shape != wm_raw.shape:
            rep.setdefault("shape_mismatch", []).append(k)
            continue
        cast["base_dtypes"].add(str(wb_raw.dtype)); cast["merged_dtypes"].add(str(wm_raw.dtype))
        wb_exact = wb_raw.float()
        wb = wb_raw.to(wm_raw.dtype).float()
        wm = wm_raw.float()
        if wb_raw.dtype != wm_raw.dtype:
            cast["elements"] += wb.numel()
            cast["changed_by_cast"] += int((wb != wb_exact).sum())
            cast["underflow_to_zero"] += int(((wb_exact != 0) & (wb == 0)).sum())
            cast["overflow_to_inf"] += int(torch.isinf(wb).sum() - torch.isinf(wb_exact).sum())
            cast["max_rel_cast_err"] = max(cast["max_rel_cast_err"],
                                           ((wb - wb_exact).norm() / wb_exact.norm().clamp_min(1e-12)).item())
        d = (wm - wb)
        dn = d.norm().item()
        if dn > 0:
            changed.append(k)
        if k in targets:
            mod = k[:-len(".weight")]
            A, B = get(aidx, mods[mod]["A"]), get(aidx, mods[mod]["B"])
            pred = scale * (B @ A)
            pn = pred.norm().item()
            if B.abs().max().item() == 0:
                zero_b.append(mod)
            if dn == 0:
                unchanged_targets.append(k)
            rel_err = (d - pred).norm().item() / max(pn, 1e-12)
            nz = pred != 0
            nonzero_pred += int(nz.sum())
            lost += int((nz & (d == 0)).sum())   # delta smaller than half an ulp: erased by the merge
            max_rel_err = max(max_rel_err, rel_err)
            per[k] = {"delta_norm": dn, "pred_norm": pn, "rel_err_vs_pred": rel_err,
                      "rel_change": dn / wb.norm().item()}
    off_target = [k for k in changed if k not in targets]
    cast["base_dtypes"], cast["merged_dtypes"] = sorted(cast["base_dtypes"]), sorted(cast["merged_dtypes"])
    rep["dtype_cast"] = cast
    rep["delta_elements_erased_by_rounding"] = lost / max(nonzero_pred, 1)
    rep.update({"n_target_matrices": len(targets), "n_changed": len(changed),
                "off_target_changed": off_target[:20], "n_off_target_changed": len(off_target),
                "targets_unchanged": unchanged_targets[:20], "n_targets_unchanged": len(unchanged_targets),
                "adapter_layers_with_zero_B": len(zero_b), "max_rel_err_merged_vs_pred": max_rel_err,
                "median_rel_change": sorted(v["rel_change"] for v in per.values())[len(per) // 2] if per else 0})
    if quant4_model is not None:
        rep["nf4"] = nf4_survival(quant4_model, bidx, midx, aidx, mods, scale)
    rep["per_matrix_sample"] = dict(list(per.items())[:6])
    return rep


def nf4_survival(qmodel, bidx, midx, aidx, mods, scale, n=21):
    """Dequantize a few 4-bit matrices: is the LoRA delta still there under the rounding noise?"""
    import bitsandbytes.functional as bnbF
    out = []
    names = sorted(mods)[:: max(1, len(mods) // n)]
    params = dict(qmodel.named_modules())
    for mod in names:
        layer = params.get(mod)
        if layer is None or not hasattr(layer.weight, "quant_state"):
            continue
        w4 = bnbF.dequantize_4bit(layer.weight.data, layer.weight.quant_state).float().cpu()
        wb, wm = get(bidx, mod + ".weight"), get(midx, mod + ".weight")
        A, B = get(aidx, mods[mod]["A"]), get(aidx, mods[mod]["B"])
        pred = (scale * (B @ A)).flatten()
        d4 = (w4 - wb).flatten()
        qnoise = (w4 - wm).norm().item()
        out.append({"module": mod,
                    "cos(4bit-base, BA)": torch.nn.functional.cosine_similarity(d4, pred, 0).item(),
                    "delta_norm/quant_noise_norm": pred.norm().item() / max(qnoise, 1e-12)})
    return out


# --------------------------------------------------------------------------- behaviour
FEWSHOT_SYS = None


def fewshot_system(train_path, k=12):
    """Give the BASE model its best shot: the grammar plus k training examples."""
    rows = load_rows(train_path)
    rng = random.Random(SEED)
    ex = "\n".join(f"{t} -> {g}" for t, g in rng.sample(rows, k))
    return ("Convert the user's device command into a tool call. Output only the call, nothing else.\n"
            "Calls: light on|off [room=<room>] [delay=<n><s|m|h>]; fan on|off|speed=<1-3> [delay=..]; "
            "relay <1-8> on|off [delay=..]; led <color>|off; grip [force=<0-100>|adjust=+20|adjust=-20]; "
            "release; arm home|cell=<1-9>; timer <n><s|m|h> [label=<word>]; "
            "sensor temp|humidity|motion|light; status [light|fan|arm] [room=<room>]; "
            "cloud (anything else). Join several calls with |.\nExamples:\n" + ex)


def behaviour(name, model, tok, sets, system=None, keep_dist=0):
    r = {"name": name, "sets": {}}
    for sname, pairs in sets.items():
        texts = [t for t, _ in pairs]
        t0 = time.time()
        outs, stopped = generate(model, tok, texts, system)
        gen_s = time.time() - t0
        em = [int(o == g) for o, (_, g) in zip(outs, pairs)]
        entry = {"n": len(pairs), "exact_match": sum(em) / len(em), "em_vec": em, "outputs": outs,
                 "stop_rate": sum(stopped) / len(stopped),
                 "grammar_valid": sum(valid(o) for o in outs) / len(outs), "gen_seconds": gen_s}
        if system is None:
            lp, d = gold_logprobs(model, tok, pairs, keep_dist=keep_dist if sname == "eval_real" else 0)
            entry["gold_logprob_mean"] = sum(lp) / len(lp)
            entry["lp_vec"] = lp
            if d:
                r["dists"] = d
        r["sets"][sname] = entry
        log(f"  {name:14s} {sname:10s} EM {entry['exact_match']:.3f}  stop {entry['stop_rate']:.3f}  "
            f"valid {entry['grammar_valid']:.3f}  gold-logp {entry.get('gold_logprob_mean', float('nan')):.3f}  "
            f"({gen_s:.0f}s)")
    return r


def compare(a, b, sname):
    ea, eb = a["sets"][sname]["em_vec"], b["sets"][sname]["em_vec"]
    lo, hi = paired_bootstrap(ea, eb)
    agree = sum(x == y for x, y in zip(a["sets"][sname]["outputs"], b["sets"][sname]["outputs"])) / len(ea)
    out = {"em_a": sum(ea) / len(ea), "em_b": sum(eb) / len(eb), "delta": (sum(eb) - sum(ea)) / len(ea),
           "ci95": [lo, hi], "output_agreement": agree}
    if "lp_vec" in a["sets"][sname] and "lp_vec" in b["sets"][sname]:
        la, lb = a["sets"][sname]["lp_vec"], b["sets"][sname]["lp_vec"]
        out["gold_logprob_delta"] = (sum(lb) - sum(la)) / len(la)
    diffs = [(t, g, x, y) for (t, g), x, y in zip(SETS[sname], a["sets"][sname]["outputs"],
                                                   b["sets"][sname]["outputs"]) if x != y][:8]
    out["examples_where_outputs_differ"] = [{"in": t, "gold": g, a["name"]: x, b["name"]: y} for t, g, x, y in diffs]
    return out


SETS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--merged")
    ap.add_argument("--quant4", help="saved 4-bit dir, or an fp16 dir to quantize to NF4 on load")
    ap.add_argument("--control-adapter")
    ap.add_argument("--control-merged")
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, help="rows per eval set (smoke tests)")
    ap.add_argument("--out", default="results/verify.json")
    ap.add_argument("--skip-weights", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.manual_seed(SEED)

    for s in ("eval_real", "eval_novel", "val"):
        SETS[s] = load_rows(os.path.join(a.data, f"{s}.jsonl"), a.limit)
    log(f"device {DEV}; eval sets " + ", ".join(f"{k}={len(v)}" for k, v in SETS.items()))
    report = {"base": a.base, "artifacts": {}, "comparisons": {}, "checks": {}}
    R = report["artifacts"]
    KD = 48  # examples whose full next-token distributions are kept for KL

    # ---- behaviour, one artifact at a time (keeps peak VRAM at one model) ----
    m, t = load("hf", a.base)
    R["base"] = behaviour("base", m, t, SETS, keep_dist=KD)
    R["base_fewshot"] = behaviour("base_fewshot", m, t, {"eval_real": SETS["eval_real"]},
                                  system=fewshot_system(os.path.join(a.data, "train.jsonl")))
    del m; free()
    if a.adapter:
        m, t = load("peft", a.adapter, base=a.base)
        R["adapter"] = behaviour("adapter", m, t, SETS)
        del m; free()
    if a.merged:
        m, t = load("hf", a.merged)
        R["merged"] = behaviour("merged", m, t, SETS, keep_dist=KD)
        del m; free()
    qmodel = None
    if a.quant4 and DEV == "cuda":
        qmodel, t = load("nf4", a.quant4)
        R["nf4"] = behaviour("nf4", qmodel, t, SETS, keep_dist=KD)
    if a.control_merged:
        m, t = load("hf", a.control_merged)
        R["control"] = behaviour("control", m, t, SETS)
        del m; free()

    # ---- comparisons ----
    C = report["comparisons"]
    for s in SETS:
        if "merged" in R:
            C[f"base->merged/{s}"] = compare(R["base"], R["merged"], s)
        if "adapter" in R and "merged" in R:
            C[f"adapter->merged/{s}"] = compare(R["adapter"], R["merged"], s)
        if "nf4" in R:
            C[f"merged->nf4/{s}"] = compare(R["merged"], R["nf4"], s)
        if "control" in R:
            C[f"base->control/{s}"] = compare(R["base"], R["control"], s)
    if "merged" in R:
        C["base_fewshot->merged/eval_real"] = compare(R["base_fewshot"], R["merged"], "eval_real")
    kl = {}
    for x in ("base", "nf4"):
        if x in R and "merged" in R and "dists" in R[x]:
            kl[f"KL(merged||{x})"] = kl_mean(R["merged"]["dists"], R[x]["dists"])
    C["kl_on_gold_positions"] = kl

    # ---- weights ----
    W = {}
    if not a.skip_weights and a.merged and a.adapter:
        W["main"] = weight_checks(a.base, a.merged, a.adapter, qmodel)
    if not a.skip_weights and a.control_merged and a.control_adapter:
        W["control"] = weight_checks(a.base, a.control_merged, a.control_adapter)
    report["weights"] = W

    # ---- verdicts ----
    K = report["checks"]

    def trained(tag, art):
        w = W.get(tag, {})
        c_real, c_nov = C.get(f"base->{art}/eval_real"), C.get(f"base->{art}/eval_novel")
        res = {
            "W1 only LoRA targets changed": bool(w) and w["n_off_target_changed"] == 0 and w["n_targets_unchanged"] == 0,
            "W2 merged-base == (alpha/r)BA": bool(w) and w["max_rel_err_merged_vs_pred"] < 0.25,
            "W3 no all-zero B layer": bool(w) and w["adapter_layers_with_zero_B"] == 0,
            "B1 EM gain CI>+10pt (real)": bool(c_real) and c_real["ci95"][0] > 0.10,
            "B1 EM gain CI>+10pt (novel)": bool(c_nov) and c_nov["ci95"][0] > 0.10,
            "B2 gold log-prob up >0.5 nat/token": bool(c_real) and c_real.get("gold_logprob_delta", 0) > 0.5,
            "B3 stop rate >= 0.98 (real)": R[art]["sets"]["eval_real"]["stop_rate"] >= 0.98,
        }
        res["TRAINED"] = all(res.values())
        return res

    if "merged" in R:
        K["main"] = trained("main", "merged")
    if "adapter" in R and "merged" in R:
        c = C["adapter->merged/eval_real"]
        K["B4 merge faithful"] = {"agreement>=0.99": c["output_agreement"] >= 0.99,
                                  "|logp gap|<0.02": abs(c.get("gold_logprob_delta", 1)) < 0.02}
    if "nf4" in R:
        c = C["merged->nf4/eval_real"]
        K["B5 4-bit faithful"] = {"agreement>=0.95": c["output_agreement"] >= 0.95,
                                  "EM drop CI lower > -0.05": c["ci95"][0] > -0.05,
                                  "KL<0.05": kl.get("KL(merged||nf4)", 1) < 0.05}
    if "control" in R:
        K["control"] = trained("control", "control")
        K["control"]["CHECK DISCRIMINATES (control must NOT be TRAINED)"] = not K["control"]["TRAINED"]

    for r in R.values():
        r.pop("dists", None)
    json.dump(report, open(a.out, "w"), indent=1, default=str)

    print("\n" + "=" * 78)
    for s in SETS:
        line = f"{s:10s} " + "  ".join(f"{k}={R[k]['sets'][s]['exact_match']:.3f}"
                                         for k in ("base", "adapter", "merged", "nf4", "control")
                                         if k in R and s in R[k]["sets"])
        print(line)
    print("base few-shot on eval_real:", f"{R['base_fewshot']['sets']['eval_real']['exact_match']:.3f}")
    for k, v in kl.items():
        print(f"{k} = {v:.4f} nats/token")
    print(json.dumps(K, indent=1))
    for tag, w in W.items():
        print(f"weights[{tag}]: changed {w['n_changed']} / targets {w['n_target_matrices']}, "
              f"off-target {w['n_off_target_changed']}, max rel err vs BA {w['max_rel_err_merged_vs_pred']:.3f}, "
              f"median rel change {w['median_rel_change']:.2e}, zero-B layers {w['adapter_layers_with_zero_B']}, "
              f"delta erased by rounding {w['delta_elements_erased_by_rounding']:.1%}")
        print(f"   dtype cast: {w['dtype_cast']}")
        if w["off_target_changed"]:
            print("   off-target changed:", w["off_target_changed"][:5])
        for row in w.get("nf4", [])[:6]:
            print("   nf4", row)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
