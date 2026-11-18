"""
Item 4 of v2: what does the deployed system do once outputs pass a grammar guard?

Deployment rule from the verdict: an output the device grammar rejects is replaced
by `cloud` (hand the command to the cloud LLM). That turns some errors into safe
deferrals; it cannot touch errors that are grammatical. This script re-scores the
outputs verify.py already produced (no GPU), so the guard is measured, not assumed.

Every output lands in one bucket:
  correct            exact match (durations compared in seconds: 120m == 2h)
  deferred           the guarded output is `cloud` but the gold is a device action
  wrong: opposite    same device, on <-> off (or grip <-> release) flipped
  wrong: value       same calls, a number / unit / room / label differs
  wrong: other       different device or intent, or a call missing / extra
  wrong: answered    gold is `cloud` but the model emitted a device action

    python guarded_eval.py results/verify.json --out results/guarded_eval.json
"""
import argparse
import collections
import json
import re

import verify as V

OPPOSITE = {("on", "off"), ("off", "on"), ("grip", "release"), ("release", "grip")}


def secs(s):
    return re.sub(r"(\d+)([smh])\b", lambda m: f"{int(m.group(1)) * {'s': 1, 'm': 60, 'h': 3600}[m.group(2)]}s", s)


def guard(out):
    return out if V.valid(out) else "cloud"


def head(call):
    """device + action of one call, e.g. 'relay 3 on delay=5m' -> ('relay', 'on')."""
    w = call.split()
    if w[0] in ("grip", "release"):
        return ("gripper", w[0])
    state = next((x for x in w[1:] if x in ("on", "off")), w[1] if len(w) > 1 else "")
    return (w[0], state.split("=")[0])


def bucket(out, gold):
    if secs(out) == secs(gold):
        return "correct"
    if out == "cloud":
        return "deferred"
    if gold == "cloud":
        return "wrong: answered"
    oc, gc = out.split("|"), gold.split("|")
    if len(oc) != len(gc):
        return "wrong: other"
    kinds = set()
    for o, g in zip(oc, gc):
        if secs(o) == secs(g):
            continue
        ho, hg = head(o), head(g)
        if ho[0] == hg[0] and (ho[1], hg[1]) in OPPOSITE:
            kinds.add("wrong: opposite")
        elif ho == hg:
            kinds.add("wrong: value")
        else:
            kinds.add("wrong: other")
    # the most dangerous kind wins
    for k in ("wrong: opposite", "wrong: other", "wrong: value"):
        if k in kinds:
            return k
    return "wrong: other"


ORDER = ["correct", "deferred", "wrong: value", "wrong: other", "wrong: opposite", "wrong: answered"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify_json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/guarded_eval.json")
    a = ap.parse_args()
    r = json.load(open(a.verify_json))
    res = {}
    print(f"{'artifact':10s} {'set':10s} {'guard':5s} " + " ".join(f"{k:>15s}" for k in ORDER) + f" {'silent wrong':>13s}")
    for art in ("merged", "nf4"):
        if art not in r["artifacts"]:
            continue
        for s, entry in r["artifacts"][art]["sets"].items():
            gold = [g for _, g in V.load_rows(f"{a.data}/{s}.jsonl")]
            for g_on in (False, True):
                outs = [guard(o) if g_on else o for o in entry["outputs"]]
                c = collections.Counter(bucket(o, g) for o, g in zip(outs, gold))
                n = len(gold)
                if not g_on:   # an invalid output counts as wrong when unguarded
                    c = collections.Counter(
                        bucket(o, g) if V.valid(o) or o == "cloud" else "wrong: other"
                        for o, g in zip(outs, gold))
                silent = sum(v for k, v in c.items() if k.startswith("wrong")) / n
                key = f"{art}/{s}/{'guarded' if g_on else 'raw'}"
                res[key] = {"n": n, **{k: c.get(k, 0) for k in ORDER}, "silent_wrong_rate": silent,
                            "examples": [{"gold": g, "out": o, "bucket": bucket(o, g)}
                                         for o, g in zip(outs, gold) if bucket(o, g) not in ("correct",)][:12]}
                print(f"{art:10s} {s:10s} {'on' if g_on else 'off':5s} " +
                      " ".join(f"{c.get(k, 0):>15d}" for k in ORDER) + f" {silent:>12.1%}")
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
