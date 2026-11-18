"""
Build the fine-tuning data for the Soup task from the command->tool-call generator.

Outputs (all in data/):
  train.jsonl        1500 rows, chatml {"messages": [...]} for `soup train`
  val.jsonl           200 rows, same distribution as train, disjoint texts
  eval_real.jsonl     131 hand-written commands (never generated, never trained)
  eval_novel.jsonl    300 rows whose key words are banned from train
  ship_eval.jsonl     eval_real in Soup's task-eval schema (prompt/expected/scoring)
  manifest.json       counts, sha256 of every file, leakage-check results

Prompt format decision: the user turn is the raw command, no system prompt.
Reason: `soup ship` live eval sends only a user message, and Qwen2.5's chat
template inserts its default system prompt when none is given. Training with
a custom system prompt would make Soup evaluate a different prompt than the
one we trained on, silently.
"""
import collections
import hashlib
import json
import random

import gen_v2

SEED = 20261110
N_TRAIN, N_VAL, N_NOVEL = 1500, 200, 300


def chat(text, target):
    return {"messages": [{"role": "user", "content": text},
                         {"role": "assistant", "content": target}]}


def intent(target):
    """Coarse intent of a target, used for the class-balance report."""
    first = target.split("|")[0].split()[0]
    return first + ("+chain" if "|" in target else "")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main():
    forbid = gen_v2.load_tests()                      # normalized test_novel + test_real texts
    pool, _ = gen_v2.build(20000, SEED, forbid)       # same generator as the ESP32 v2 model
    rng = random.Random(SEED)

    # dedupe by input text: a duplicate prompt with the same target only adds weight
    seen, uniq = set(), []
    for t, g in pool:
        if t not in seen:
            seen.add(t)
            uniq.append((t, g))
    rng.shuffle(uniq)
    train, val = uniq[:N_TRAIN], uniq[N_TRAIN:N_TRAIN + N_VAL]

    real = [tuple(l.rstrip("\n").split("\t")) for l in open("data/test_real.tsv") if l.strip()]
    novel_all = [json.loads(l) for l in open("data/test_novel.jsonl")]
    novel = [(r["in"], r["out"]) for r in random.Random(SEED).sample(novel_all, N_NOVEL)]

    # ---------- leakage checks ----------
    norm = gen_v2.normalize
    train_txt = {norm(t) for t, _ in train}
    checks = {
        "val_in_train": sum(norm(t) in train_txt for t, _ in val),
        "real_in_train": sum(norm(t) in train_txt for t, _ in real),
        "novel_in_train": sum(norm(t) in train_txt for t, _ in novel),
        "train_rows_with_banned_words": sum(bool(gen_v2.BANNED & set(t.split())) for t, _ in train),
    }
    assert all(v == 0 for v in checks.values()), checks

    write("data/train.jsonl", [chat(t, g) for t, g in train])
    write("data/val.jsonl", [chat(t, g) for t, g in val])
    write("data/eval_real.jsonl", [chat(t, g) for t, g in real])
    write("data/eval_novel.jsonl", [chat(t, g) for t, g in novel])
    write("data/ship_eval.jsonl", [{"prompt": t, "expected": g, "scoring": "exact", "category": intent(g)}
                                   for t, g in real])

    files = ["train", "val", "eval_real", "eval_novel", "ship_eval"]
    manifest = {
        "seed": SEED,
        "generator": "gen_v2.py (template paraphrases + typo/word-dropout augmentation)",
        "counts": {"train": len(train), "val": len(val), "eval_real": len(real), "eval_novel": len(novel)},
        "leakage_checks": checks,
        "train_intents": dict(collections.Counter(intent(g) for _, g in train).most_common()),
        "eval_real_intents": dict(collections.Counter(intent(g) for _, g in real).most_common()),
        "sha256": {f: sha(f"data/{f}.jsonl") for f in files},
    }
    json.dump(manifest, open("data/manifest.json", "w"), indent=2)
    print(json.dumps({k: manifest[k] for k in ("counts", "leakage_checks", "train_intents")}, indent=1))


if __name__ == "__main__":
    main()
