"""
Run `soup train` IN THIS PROCESS (single GPU: Soup does not re-exec), then read the
CUDA caching allocator's own counters, which the CLI never prints. Part 1 needs them
to split nvidia-smi's peak into: allocated tensors, reserved-but-free cache, CUDA context.

    python train_with_stats.py configs/main.yaml results/train_main_cuda_stats.json

Identical training to `soup --no-telemetry train -c <cfg> --yes --allow-oom-attempt`.
"""
import json
import subprocess
import sys
import time

import torch
import typer

from soup_cli.cli import app

cfg, out = sys.argv[1], sys.argv[2]
args = ["--no-telemetry", "train", "-c", cfg, "--yes", "--allow-oom-attempt"]
t0 = time.time()
rc = 0
try:
    typer.main.get_command(app)(args, standalone_mode=False)
except SystemExit as e:          # click/typer exit paths
    rc = e.code or 0
except typer.Exit as e:
    rc = e.exit_code or 0

stats = {"config": cfg, "exit_code": rc, "seconds": round(time.time() - t0, 1)}
if torch.cuda.is_available():
    s = torch.cuda.memory_stats()
    G = 2 ** 30
    stats.update({
        "peak_allocated_GiB": s.get("allocated_bytes.all.peak", 0) / G,
        "peak_reserved_GiB": s.get("reserved_bytes.all.peak", 0) / G,
        "peak_inactive_split_GiB (fragmentation)": s.get("inactive_split_bytes.all.peak", 0) / G,
        "num_alloc_retries": s.get("num_alloc_retries", 0),
        "num_ooms": s.get("num_ooms", 0),
        "device_total_GiB": torch.cuda.get_device_properties(0).total_memory / G,
    })
    # what nvidia-smi says this process holds right now (reserved cache + context)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    stats["nvidia_smi_used_now_MiB"] = float(smi.splitlines()[0]) if smi else None
    stats["reserved_now_GiB"] = torch.cuda.memory_reserved() / G
print(json.dumps(stats, indent=1))
json.dump(stats, open(out, "w"), indent=1)
sys.exit(rc)
