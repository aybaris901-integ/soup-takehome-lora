"""Generates soup_task.ipynb (kept as code so the notebook is reviewable in diffs)."""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Soup take-home: LoRA → merge → 4-bit → SHIP / DON'T SHIP

**Task:** Qwen2.5-1.5B-Instruct learns to turn English device commands into compact tool calls
(`turn off the kitchen light in 10 min` → `light off room=kitchen delay=10m`). Outputs are checkable by
exact match, so "did it learn?" has an objective answer.

**How to run:** Runtime → Change runtime type → **T4 GPU**. Upload `soup_task.zip` to the Colab file panel,
then Runtime → Run all (~60–80 min). The last cell downloads `deliverables.zip`.

Every command runs through `sh(phase, cmd)`: each output line gets a UTC timestamp and is written to
`logs/<phase>.log`; phase start/end times go to `logs/phases.csv`, and a background `nvidia-smi` trace
(every 200 ms) plus raw `nvidia-smi` snapshots (every 30 s) go to `logs/`. Nothing is edited or hidden:
failed commands stay in the logs.
""")

code(r'''
import os, subprocess, time, datetime, json, sys
os.chdir("/content")
if os.path.exists("soup_task.zip") and not os.path.exists("soup_task/verify.py"):
    subprocess.run("unzip -oq soup_task.zip -d soup_task", shell=True, check=True)
os.chdir("/content/soup_task")
os.makedirs("logs", exist_ok=True); os.makedirs("results", exist_ok=True)

def sh(phase, cmd, check=False):
    """Run cmd, timestamp every output line, tee to logs/<phase>.log, mark phase start/end."""
    with open("logs/phases.csv", "a") as f: f.write(f"start,{phase},{time.time()}\n")
    t0 = time.time()
    with open(f"logs/{phase}.log", "a") as log:
        log.write(f"### {datetime.datetime.utcnow().isoformat()}Z $ {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            stamped = f"[{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z] {line.rstrip()}"
            print(stamped); log.write(stamped + "\n")
        rc = p.wait()
        log.write(f"### exit code {rc} after {time.time()-t0:.1f}s\n")
    with open("logs/phases.csv", "a") as f: f.write(f"end,{phase},{time.time()}\n")
    print(f"--- {phase}: exit {rc}, {time.time()-t0:.0f}s")
    if check and rc != 0: raise RuntimeError(f"{phase} failed with exit code {rc}")
    return rc

# background GPU telemetry: fine-grained CSV trace + raw nvidia-smi snapshots
subprocess.Popen("nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw "
                 "--format=csv,nounits -lms 200 > logs/nvidia_smi_trace.csv", shell=True)
subprocess.Popen("while true; do echo \"===== $(date -u +%FT%TZ)\"; nvidia-smi; sleep 30; done > logs/nvidia_smi_raw.log 2>&1",
                 shell=True)
sh("env_gpu", "nvidia-smi; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv")
''')

md("## 0. Environment (pinned to the Soup version the scripts were written against)")
code(r'''
sh("env_install", 'pip install -q "soup-cli[train]==0.77.2" 2>&1 | tail -3; [ -x /usr/bin/time ] || apt-get -qq install -y time > /dev/null', check=True)
sh("env_versions", "python -c \"import torch,transformers,peft,trl,bitsandbytes as b;print('torch',torch.__version__,'cuda',torch.version.cuda,'bf16_supported',torch.cuda.is_bf16_supported());print('transformers',transformers.__version__,'peft',peft.__version__,'trl',trl.__version__,'bnb',b.__version__)\"; soup --version; pip freeze > logs/pip_freeze.txt")
sh("env_soup_doctor", "soup --no-telemetry doctor")
''')

md("""
## 1. Data
Rebuild the data from the generator and check the bytes match the shipped manifest (reproducibility),
then run Soup's data doctor with the **real** Qwen tokenizer and record per-row token counts for the memory budget.
""")
code(r'''
shipped = json.load(open("data/manifest.json"))["sha256"]
sh("data_build", "python build_data.py", check=True)
rebuilt = json.load(open("data/manifest.json"))["sha256"]
print("data reproducible:", shipped == rebuilt)
sh("data_doctor", "soup --no-telemetry data doctor data/train.jsonl -m Qwen/Qwen2.5-1.5B-Instruct --max-length 256 --sample 1500 --show-mask 3 -o results/data_doctor.json")
sh("data_tokens", """python - <<'EOF'
import json, statistics
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
n = [len(tok(tok.apply_chat_template(json.loads(l)["messages"], tokenize=False), add_special_tokens=False)["input_ids"])
     for l in open("data/train.jsonl")]
json.dump(n, open("data/tokens.json", "w"))
print("rows", len(n), "mean", statistics.mean(n), "p95", sorted(n)[int(.95*len(n))], "max", max(n))
EOF""", check=True)
''')

md("""
## 2. Part 1 — memory budget, **before** training
Our itemised estimate (`memory_budget.py`, formulas over the real config and real token counts), Soup's own
pre-flight estimate (`soup profile`), and a direct measurement of the activation term on this GPU.
""")
code(r'''
sh("budget_before", "python memory_budget.py --config $(python -c \"from huggingface_hub import hf_hub_download as d;print(d('Qwen/Qwen2.5-1.5B-Instruct','config.json'))\") --tokens data/tokens.json --batch 8 --json-out results/memory_budget.json", check=True)
sh("budget_soup_profile", "soup --no-telemetry profile -c configs/main.yaml --json > results/soup_profile.json; cat results/soup_profile.json")
sh("budget_measure_activations", "python measure_activations.py --tokens 96 --batch 4 | tee results/activation_measurement.json")
''')

md("## 3. Train: main run and the control run (identical except lr = 1e-7)")
code(r'''
sh("train_main", "soup --no-telemetry train -c configs/main.yaml --yes", check=True)
sh("snapshot_after_train_main", "nvidia-smi")
sh("train_control", "soup --no-telemetry train -c configs/control.yaml --yes", check=True)
''')

md("## 4. Merge (Soup, fp16, runs on CPU — host-RAM peak via /usr/bin/time) and quantize to NF4")
code(r'''
sh("merge_main", "/usr/bin/time -v soup --no-telemetry merge --adapter runs/main --output runs/main-merged --dtype float16", check=True)
sh("merge_control", "/usr/bin/time -v soup --no-telemetry merge --adapter runs/control --output runs/control-merged --dtype float16", check=True)
sh("quantize_main", "python quantize_nf4.py runs/main-merged runs/main-merged-nf4", check=True)
sh("artifact_sizes", "du -sh runs/main runs/main-merged runs/main-merged-nf4 runs/control runs/control-merged")
''')

md("## 5. Part 2 — did it train? base vs adapter vs merged vs 4-bit, plus the control")
code(r'''
sh("verify", "python verify.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/main --merged runs/main-merged "
             "--quant4 runs/main-merged-nf4 --control-adapter runs/control --control-merged runs/control-merged "
             "--out results/verify.json", check=True)
''')

md("## 6. Part 3 — silent-failure probes")
code(r'''
sh("silent_failures", "python silent_failures.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/main "
                      "--merged runs/main-merged --nf4 runs/main-merged-nf4 --verify results/verify.json")
''')

md("""
## 7. What do Soup's own gates say?
`soup ship` on the main adapter, on the **control** adapter (does Soup catch a run that learned nothing but
has a non-zero adapter?), and on the 4-bit artifact we would actually ship.
""")
code(r'''
B = "--base Qwen/Qwen2.5-1.5B-Instruct --task-eval data/ship_eval.jsonl --device cuda"
sh("ship_main", f"soup --no-telemetry ship {B} --adapter runs/main -o results/ship_main.json")
sh("ship_control", f"soup --no-telemetry ship {B} --adapter runs/control -o results/ship_control.json")
sh("ship_nf4", f"soup --no-telemetry ship {B} --tuned runs/main-merged-nf4 -o results/ship_nf4.json")
''')

md("## 8. Memory: predicted vs measured, and the reported throughput")
code(r'''
sh("memory_report", "python phase_report.py")
sh("throughput", """python - <<'EOF'
import json, re
tok = json.load(open("data/tokens.json"))
log = open("logs/train_main.log").read()
rt = float(re.findall(r"'train_runtime': '([0-9.]+)'", log)[-1])
rows = 1350; epochs = 2
real = sum(tok) * rows / len(tok) * epochs
trained = sum(sum(json.loads(l)["row_tokens"]) for l in open("runs/main/rewind.jsonl") if '"batch"' in l)
print(f"train_runtime {rt:.1f}s | real tokens/s {real/rt:.0f} | loss-bearing tokens/s {trained/rt:.0f} | "
      f"loss-bearing share {trained/real:.1%}")
EOF""")
''')

md("## 9. Package deliverables")
code(r'''
sh("package", "zip -qr deliverables.zip logs results configs data/manifest.json data/*.jsonl *.py soup_task.ipynb "
              "runs/main/soup_run.json runs/main/rewind.jsonl runs/control/rewind.jsonl runs/main-merged-nf4/quantize_stats.json; ls -la deliverables.zip")
from google.colab import files
files.download("deliverables.zip")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("soup_task.ipynb", "w"), indent=1)
print("wrote soup_task.ipynb with", len(cells), "cells")
