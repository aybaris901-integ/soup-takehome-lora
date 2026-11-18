"""
Join the nvidia-smi trace with the phase markers written by the notebook and compare
the measured peaks with the BEFORE-training prediction (results/memory_budget.json).

    python phase_report.py
Writes results/memory_actual.json and prints one row per phase.
"""
import csv
import datetime as dt
import json
import os
import re

GiB = 2 ** 30


def parse_smi_time(s):
    return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def main():
    phases = {}
    for row in csv.reader(open("logs/phases.csv")):
        kind, name, t = row[0], row[1], float(row[2])
        phases.setdefault(name, {})[kind] = t
    trace = []
    with open("logs/nvidia_smi_trace.csv") as f:
        r = csv.reader(f)
        next(r, None)   # empty trace (no GPU) -> no rows
        for row in r:
            try:
                trace.append((parse_smi_time(row[0]), float(row[1]), float(row[3])))
            except (ValueError, IndexError):
                continue
    # torch-side peaks printed by Soup / our scripts
    soup_peak = {}
    for name in ("train_main", "train_control"):
        p = f"logs/{name}.log"
        if os.path.exists(p):
            # Soup prints max_memory_allocated()/1024**3 labelled "GB" (it is GiB), e.g. "8.2/14.7 GB"
            vals = [float(x) for x in re.findall(r"(\d+\.\d)/\d+\.\d GB", open(p).read())]
            soup_peak[name] = max(vals) if vals else None
    rss = {}
    for name in ("merge_main", "merge_control"):
        p = f"logs/{name}.log"
        if os.path.exists(p):
            m = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", open(p).read())
            rss[name] = int(m[-1]) * 1024 / GiB if m else None   # last attempt = the one that ran
    pred = json.load(open("results/memory_budget.json")) if os.path.exists("results/memory_budget.json") else {}
    out = {}
    print(f"{'phase':22s} {'seconds':>8s} {'smi peak MiB':>13s} {'smi peak GiB':>13s} {'util% mean':>10s}")
    for name, p in phases.items():
        if "start" not in p or "end" not in p:
            continue
        pts = [(m, u) for t, m, u in trace if p["start"] <= t <= p["end"]]
        peak = max((m for m, _ in pts), default=0)
        util = sum(u for _, u in pts) / len(pts) if pts else 0
        out[name] = {"seconds": p["end"] - p["start"], "nvidia_smi_peak_MiB": peak,
                     "nvidia_smi_peak_GiB": peak * 2 ** 20 / GiB, "mean_util_pct": util, "samples": len(pts)}
        print(f"{name:22s} {out[name]['seconds']:8.0f} {peak:13.0f} {out[name]['nvidia_smi_peak_GiB']:13.2f} {util:10.0f}")
    out["soup_reported_peak_GB"] = soup_peak
    # v2: the caching allocator's own counters, read in the training process itself
    for name in ("train_main", "train_control"):
        p = f"results/{name}_cuda_stats.json"
        if os.path.exists(p):
            out[f"{name}_cuda_stats"] = json.load(open(p))
            print(f"{name} allocator:", json.dumps(out[f"{name}_cuda_stats"]))
    if os.path.exists("results/cuda_context.json"):
        out["cuda_context"] = json.load(open("results/cuda_context.json"))
        print("CUDA context:", out["cuda_context"])
    out["merge_host_rss_GiB"] = rss
    if pred:
        out["predicted"] = {"train_torch_alloc_GiB": pred["train"]["total_alloc"] / GiB,
                            "train_nvidia_smi_GiB": pred["train"]["nvidia_smi_estimate"] / GiB,
                            "merge_host_GiB": pred["merge"]["total_alloc"] / GiB,
                            "quant_peak_GiB": pred["quant"]["peak"] / GiB}
        print("predicted:", json.dumps(out["predicted"], indent=1))
    print("soup-reported torch peak (GB, from logs):", soup_peak)
    print("merge host-RAM max RSS (GiB):", rss)
    json.dump(out, open("results/memory_actual.json", "w"), indent=1)


if __name__ == "__main__":
    main()
