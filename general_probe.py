"""
What does the model actually answer on Soup's general suites?

`soup ship` reported 7 of 8 general suites collapsing but saves no per-item outputs.
This re-runs Soup's OWN bundled suites through Soup's OWN scorers
(`score_bundled_suite`) and generator (`make_generator`, same 256-token greedy
budget), but records every prompt/output pair, in fp16 (the numerics we ship;
`soup ship` hard-codes bf16 on CUDA), for three prompt conditions:

  plain      the item exactly as Soup sends it           (what leg 2 measured)
  directed   "Answer this general question; it is not a device command." + item
             (is the general ability still THERE, just routed away by default?)

    python general_probe.py --base Qwen/Qwen2.5-1.5B-Instruct \
        --tuned runs/main-merged --nf4 runs/main-merged-nf4 --out results/general_probe.json
"""
import argparse
import collections
import json
import os
import time

from soup_cli.eval.custom import score_exact
from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS, DEFAULT_GENERAL_SUITE, score_bundled_suite
from soup_cli.utils import live_eval

DIRECTED = ("Answer this general question normally; it is not a device command.\n\n{}")


def recorder(gen, transform=None):
    log = []

    def g(prompt):
        p = transform(prompt) if transform else prompt
        out = gen(p)
        log.append({"prompt": prompt, "output": out})
        return out
    return g, log


def run(name, path, device, ship_eval, conds):
    t0 = time.time()
    gen = live_eval.make_generator(path, device=device, max_new_tokens=BEHAVIOURAL_MAX_NEW_TOKENS, dtype="float16")
    res = {"path": path, "suites": {}}
    for cond, tf in [c for c in (("plain", None), ("directed", DIRECTED.format)) if c[0] in conds]:
        for suite in DEFAULT_GENERAL_SUITE:
            g, log = recorder(gen, tf)
            score = score_bundled_suite(suite, g)
            outs = collections.Counter(x["output"].strip() for x in log)
            res["suites"].setdefault(suite, {})[cond] = {
                "score": score, "n": len(log),
                "share_answered_cloud": sum(x["output"].strip() == "cloud" for x in log) / max(len(log), 1),
                "most_common_outputs": outs.most_common(3), "items": log}
            print(f"[{name:6s}] {suite:18s} {cond:8s} score {score:.3f}  "
                  f"cloud {res['suites'][suite][cond]['share_answered_cloud']:.0%}  top: {outs.most_common(1)}",
                  flush=True)
    # leg 1 again, fp16, with Soup's own (case-insensitive) scorer and a strict one
    rows = [json.loads(l) for l in open(ship_eval)]
    outs = [gen(r["prompt"]) for r in rows]
    res["task"] = {"soup_exact": sum(score_exact(o, r["expected"]) for o, r in zip(outs, rows)) / len(rows),
                   "strict_exact": sum(o.strip() == r["expected"] for o, r in zip(outs, rows)) / len(rows)}
    print(f"[{name:6s}] task (ship_eval, fp16): {res['task']}  {time.time()-t0:.0f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--nf4")
    ap.add_argument("--ship-eval", default="data/ship_eval.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/general_probe.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    R = {}
    for name, path in (("base", a.base), ("merged", a.tuned), ("nf4", a.nf4)):
        if path:
            # the base model needs no "directed" condition: it was never routed away
            conds = ("plain",) if name == "base" else ("plain", "directed")
            R[name] = run(name, path, a.device, a.ship_eval, conds)
            json.dump(R, open(a.out, "w"), indent=1)
    print("\nsuite              " + "  ".join(f"{n}:plain/directed" for n in R))
    for suite in DEFAULT_GENERAL_SUITE:
        print(f"{suite:18s} " + "  ".join(
            f"{R[n]['suites'][suite]['plain']['score']:.2f}/"
            f"{R[n]['suites'][suite].get('directed', {}).get('score', float('nan')):.2f}"
            .ljust(len(n) + 16) for n in R))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
