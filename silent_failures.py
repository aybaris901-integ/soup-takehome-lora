"""
Part 3 - probes for failures that raise no error.

Each probe prints one line and writes results/silent_failures.json. They reuse
verify.py's loading/decoding so every artifact is used exactly as a user would load it.

  P1 tokenizer / chat template / EOS / pad identical across base, adapter, merged, 4-bit
  P2 loss mask of the ACTUAL run: Soup's flight recorder (rewind.jsonl) logs how many
     tokens each row trained on; compare with len(answer tokens) + end-of-turn
  P3 system prompt at serve time differs from training (no system -> Qwen default)
  P4 input casing / punctuation: the generator lowercased every training prompt
  P5 Soup's 'exact' scorer is case-insensitive; the device parser is not
  P6 batch / left-padding invariance (bs=1 vs bs=32 give the same greedy output?)
  P7 sampling at serve time (temperature 0.7) vs greedy eval
  P8 digits that change between merged and 4-bit (from verify.json)
  P9 the generation_config shipped with the artifact (sampling + repetition penalty)

    python silent_failures.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/main \
        --merged runs/main-merged --nf4 runs/main-merged-nf4 --verify results/verify.json
"""
import argparse
import hashlib
import json
import os
import re
import statistics

import torch

import verify as V


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16] if os.path.exists(path) else None


def p1_tokenizers(paths):
    from transformers import AutoTokenizer, GenerationConfig
    rows = {}
    for name, p in paths.items():
        if not p:
            continue
        d = V.local_dir(p)
        try:
            tok = AutoTokenizer.from_pretrained(p)
        except (OSError, ValueError) as e:
            rows[name] = {"error": f"no tokenizer: {e.__class__.__name__}"}
            continue
        try:
            gen = GenerationConfig.from_pretrained(p).eos_token_id
        except OSError:
            gen = None
        vocab = sorted(tok.get_vocab().items())
        content = json.dumps([vocab, sorted(map(str, tok.all_special_tokens))])
        rows[name] = {"vocab+special_sha": hashlib.sha256(content.encode()).hexdigest()[:16],
                      "tokenizer.json_bytes_sha (info only)": file_sha(os.path.join(d, "tokenizer.json")),
                      "chat_template_sha": hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:16],
                      "eos": tok.eos_token, "pad": tok.pad_token, "pad_is_eos": tok.pad_token == tok.eos_token,
                      "generation_eos_ids": gen, "vocab": len(tok)}
    ref = next(iter(rows.values()))
    keys = ("vocab+special_sha", "chat_template_sha", "eos", "pad", "vocab")
    drift = {n: [k for k in keys if r.get(k) != ref.get(k)] for n, r in rows.items()}
    return {"artifacts": rows, "drift_vs_first": drift,
            "ok": all(not v for v in drift.values()) and not any(r.get("pad_is_eos") for r in rows.values())}


def p2_loss_mask(run_dir, train_path, base):
    """Compare the trained-token counts Soup actually used with what an answer-only mask implies."""
    from transformers import AutoTokenizer
    path = os.path.join(run_dir, "rewind.jsonl")
    if not os.path.exists(path):
        return {"ok": None, "note": "no rewind.jsonl in run dir"}
    seen = []
    for l in open(path):
        r = json.loads(l)
        if r.get("kind") == "batch":
            seen += r["row_tokens"]
    tok = AutoTokenizer.from_pretrained(base)
    exp, full = [], []
    n = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
    header = n("<|im_start|>assistant\n")          # role prefix of the assistant turn
    for l in open(train_path):
        m = json.loads(l)["messages"]
        exp.append(n(m[1]["content"]))
        full.append(n(tok.apply_chat_template(m, tokenize=False)))
    off = statistics.mean(seen) - statistics.mean(exp)
    res = {"rows_logged": len(seen), "mean_trained_tokens_per_row": statistics.mean(seen),
           "mean_answer_tokens": statistics.mean(exp), "mean_full_sequence_tokens": statistics.mean(full),
           "extra_tokens_trained_per_row (end-of-turn etc.)": off,
           "fraction_of_sequence_trained": statistics.mean(seen) / statistics.mean(full),
           "min_max_trained": [min(seen), max(seen)]}
    # Which mask explains the counts?  answer+<|im_end|>(+\n): offset 1-2.
    # role header too (Soup's fallback when the template has no {% generation %} markers):
    # offset header+1 .. header+2. Prompt trained: offset ~ full - answer.
    res["role_header_tokens"] = header
    res["explained_by"] = ("answer + end-of-turn" if 0.5 <= off <= 2.5 else
                           "role header + answer + end-of-turn" if header + 0.5 <= off <= header + 2.5 else
                           "PROMPT TOKENS IN LOSS or other mismatch")
    res["constant_tokens_share_of_trained"] = (off) / statistics.mean(seen)
    res["ok"] = res["explained_by"] != "PROMPT TOKENS IN LOSS or other mismatch"
    # Soup's own masking (data doctor uses the same dispatch as training): decode 3 trained spans
    try:
        from soup_cli.utils.data_doctor import _build_row_labels
        spans = []
        for l in list(open(train_path))[:3]:
            m = json.loads(l)["messages"]
            b = _build_row_labels(tok, m, max_length=256, train_on_responses_only=True,
                                  train_on_messages_with_train_field=False, include_eot=False)
            ids = [i for i, lab in zip(b["input_ids"], b["labels"]) if lab != -100]
            spans.append(tok.decode(ids))
        res["trained_text_examples"] = spans
    except Exception as e:  # private API, may move between Soup versions
        res["trained_text_examples"] = f"unavailable: {e}"
    return res


def em(model, tok, pairs, system=None, transform=None, bs=32, gen_kwargs=None):
    texts = [transform(t) if transform else t for t, _ in pairs]
    outs, _ = V.generate(model, tok, texts, system, bs=bs, gen_kwargs=gen_kwargs)
    return sum(o == g for o, (_, g) in zip(outs, pairs)) / len(pairs), outs


def probes_on_model(name, model, tok, pairs):
    r = {}
    base_em, base_out = em(model, tok, pairs)
    r["P3 system prompt"] = {
        "none (training condition)": base_em,
        "explicit Qwen default": em(model, tok, pairs, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")[0],
        "'You are a helpful assistant.'": em(model, tok, pairs, "You are a helpful assistant.")[0],
        "task instruction": em(model, tok, pairs, "Convert the command to a device tool call.")[0],
    }
    r["P4 input surface form"] = {
        "as written": base_em,
        "Sentence case.": em(model, tok, pairs, transform=lambda t: t[:1].upper() + t[1:] + ".")[0],
        "Title Case": em(model, tok, pairs, transform=lambda t: t.title())[0],
        "UPPER": em(model, tok, pairs, transform=str.upper)[0],
        "trailing '?' or '!'": em(model, tok, pairs, transform=lambda t: t + "!")[0],
    }
    one, out1 = em(model, tok, pairs[:64], bs=1)
    r["P6 batch invariance"] = {"em_bs1": one, "em_bs32": sum(o == g for o, (_, g) in zip(base_out[:64], pairs[:64])) / 64,
                                "outputs_identical_bs1_vs_bs32": sum(a == b for a, b in zip(out1, base_out[:64])) / 64}
    samp = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        tok.padding_side = "left"
        texts = [t for t, _ in pairs]
        outs = []
        for i in range(0, len(texts), 32):
            enc = tok([V.render(tok, t) for t in texts[i:i + 32]], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(V.DEV)
            g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=64,
                               pad_token_id=tok.pad_token_id)
            outs += [tok.decode(x, skip_special_tokens=True).strip() for x in g[:, enc["input_ids"].shape[1]:]]
        samp.append(sum(o == g for o, (_, g) in zip(outs, pairs)) / len(pairs))
    r["P7 sampling T=0.7"] = {"greedy": base_em, "sampled_mean": sum(samp) / 3, "sampled_runs": samp}
    # P9: the artifact's OWN generation_config, untouched (what model.generate() / most servers do)
    gc = model.generation_config
    r["P9 shipped generation_config"] = {
        "config": {k: getattr(gc, k, None) for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty")},
        "pure greedy (main metric)": base_em,
        "greedy + shipped repetition_penalty": em(model, tok, pairs, gen_kwargs={"do_sample": False})[0],
    }
    runs = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        runs.append(em(model, tok, pairs, gen_kwargs={})[0])
    r["P9 shipped generation_config"]["defaults untouched (3 seeds)"] = runs
    V.log(f"  {name}: " + json.dumps({k: v for k, v in r.items()}, default=lambda x: round(x, 3)))
    return r


def p5_soup_scorer():
    from soup_cli.eval.custom import score_exact
    cases = [("LIGHT ON", "light on"), ("  light on \n", "light on"), ("Light On Room=Kitchen", "light on room=kitchen")]
    return {"cases": [{"output": o, "expected": e, "soup_exact": score_exact(o, e), "strict": o == e} for o, e in cases],
            "ok": all(score_exact(o, e) == (o == e) for o, e in cases)}


def p8_digits(verify_path):
    if not os.path.exists(verify_path):
        return {"note": "run verify.py first"}
    r = json.load(open(verify_path))
    A = r["artifacts"]
    if "merged" not in A or "nf4" not in A:
        return {"note": "needs merged and nf4 in verify.json"}
    flips, diffs = [], 0
    for s in A["merged"]["sets"]:
        for a, b in zip(A["merged"]["sets"][s]["outputs"], A["nf4"]["sets"][s]["outputs"]):
            if a != b:
                diffs += 1
                if re.sub(r"\d+", "#", a) == re.sub(r"\d+", "#", b):
                    flips.append({"merged": a, "nf4": b})
    return {"outputs_differ": diffs, "differ_only_in_numbers": len(flips), "examples": flips[:10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--merged")
    ap.add_argument("--nf4")
    ap.add_argument("--data", default="data")
    ap.add_argument("--verify", default="results/verify.json")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="results/silent_failures.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    R = {}
    R["P1 tokenizer/template/eos"] = p1_tokenizers({"base": a.base, "adapter": a.adapter, "merged": a.merged, "nf4": a.nf4})
    V.log("P1", json.dumps(R["P1 tokenizer/template/eos"]["drift_vs_first"]), "ok:", R["P1 tokenizer/template/eos"]["ok"])
    if a.adapter:
        R["P2 loss mask (actual run)"] = p2_loss_mask(a.adapter, os.path.join(a.data, "train.jsonl"), a.base)
        V.log("P2", json.dumps({k: v for k, v in R["P2 loss mask (actual run)"].items() if k != "trained_text_examples"}))
    R["P5 soup exact scorer"] = p5_soup_scorer()
    V.log("P5", json.dumps(R["P5 soup exact scorer"]))
    pairs = V.load_rows(os.path.join(a.data, "eval_real.jsonl"), a.limit)
    for name, kind, p in (("merged", "hf", a.merged), ("nf4", "nf4", a.nf4)):
        if not p or (kind == "nf4" and V.DEV != "cuda"):
            continue
        m, t = V.load(kind, p)
        R[f"model probes: {name}"] = probes_on_model(name, m, t, pairs)
        del m
        V.free()
    R["P8 digit flips merged->nf4"] = p8_digits(a.verify)
    V.log("P8", json.dumps(R["P8 digit flips merged->nf4"])[:600])
    json.dump(R, open(a.out, "w"), indent=1, default=str)
    V.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
