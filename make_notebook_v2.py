"""Generates soup_task_v2.ipynb: one clean, guarded pass that also closes v1's open items."""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Soup take-home — v2: one clean run + the four open items from v1

**Run:** Runtime → Change runtime type → **T4 GPU** → upload `soup_task_v2.zip` → **Runtime → Run all** (~2 h).
The last cell downloads `deliverables_v2.zip`. **If any cell stops with a red error, stop and send it** — every cell
checks the previous one's output and refuses to continue on missing or mismatched artifacts.

What changed vs v1:
1. **Clean single pass.** torchao is removed before anything imports peft. Training refuses to run twice (an existing
   adapter is reused, never silently overwritten), and every adapter / merged / 4-bit file is SHA-256 hashed at birth
   and re-checked by every later step, so all results provably judge the same bytes.
2. **What the model says on Soup's general suites:** `general_probe.py` re-runs Soup's own suites and scorers in fp16
   and saves every prompt/output pair, plain and with a one-line "this is a general question" instruction.
3. **Memory, completed:** the allocator's own peak reserved/allocated counters (training runs in-process), the CUDA
   context measured on its own, and the activation measurement on the GPU.
4. **The grammar guard, measured:** `guarded_eval.py` re-scores outputs with invalid → `cloud` and splits the
   remaining errors into value / other / **opposite action** (on↔off).

Also: the 4-bit gate in `verify.py` now checks every eval set (v1 checked eval_real only, the flaw v1 found).
""")

code(r'''
import os, subprocess, time, datetime, json, hashlib, glob
UTC = datetime.timezone.utc
os.chdir("/content")
if os.path.exists("soup_task_v2.zip") and not os.path.exists("soup_task/verify.py"):
    subprocess.run("unzip -oq soup_task_v2.zip -d soup_task", shell=True, check=True)
os.chdir("/content/soup_task")
os.makedirs("logs", exist_ok=True); os.makedirs("results", exist_ok=True)

def now(): return datetime.datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

def sh(phase, cmd, check=True):
    """Run cmd, timestamp every line, tee to logs/<phase>.log, mark phase start/end. Fails loudly by default."""
    with open("logs/phases.csv", "a") as f: f.write(f"start,{phase},{time.time()}\n")
    t0 = time.time()
    with open(f"logs/{phase}.log", "a") as log:
        log.write(f"### {now()} $ {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            stamped = f"[{now()}] {line.rstrip()}"
            print(stamped); log.write(stamped + "\n")
        rc = p.wait()
        log.write(f"### exit code {rc} after {time.time()-t0:.1f}s\n")
    with open("logs/phases.csv", "a") as f: f.write(f"end,{phase},{time.time()}\n")
    print(f"--- {phase}: exit {rc}, {time.time()-t0:.0f}s")
    if check and rc != 0: raise RuntimeError(f"STOP: {phase} failed (exit {rc}). Send the lines above; do not run further cells.")
    return rc

def need(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing: raise RuntimeError(f"STOP: missing {missing} - an earlier cell did not finish. Do not continue.")

def hashes(d):
    return {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
            for f in sorted(glob.glob(f"{d}/*.safetensors"))}

PROV = "results/provenance.json"
def record(name, d):
    p = json.load(open(PROV)) if os.path.exists(PROV) else {}
    p[name] = {"hashes": hashes(d), "recorded_at": now()}
    json.dump(p, open(PROV, "w"), indent=1); print("provenance", name, p[name])

def check_same(name, d):
    """Refuse to evaluate an artifact whose bytes differ from the ones recorded when it was made."""
    rec = json.load(open(PROV))[name]["hashes"]
    cur = hashes(d)
    if cur != rec: raise RuntimeError(f"STOP: {d} changed since it was recorded ({rec} -> {cur}).")
    print(f"provenance ok: {name} {cur}")

subprocess.Popen("nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw "
                 "--format=csv,nounits -lms 200 > logs/nvidia_smi_trace.csv", shell=True)
subprocess.Popen("while true; do echo \"===== $(date -u +%FT%TZ)\"; nvidia-smi; sleep 30; done > logs/nvidia_smi_raw.log 2>&1", shell=True)
sh("env_gpu", "nvidia-smi; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv")
''')

md("## 0. Environment — torchao removed BEFORE anything imports peft (v1's failure)")
code(r'''
sh("env_install", 'pip install -q "soup-cli[train]==0.77.2" 2>&1 | tail -3; pip uninstall -y -q torchao; '
                  '[ -x /usr/bin/time ] || apt-get -qq install -y time > /dev/null')
sh("env_versions", "python -c \"import importlib.util as u; assert u.find_spec('torchao') is None, 'torchao still installed'; "
                   "import torch,transformers,peft,trl,bitsandbytes as b;print('torch',torch.__version__,'cuda',torch.version.cuda,"
                   "'bf16_supported',torch.cuda.is_bf16_supported());print('transformers',transformers.__version__,'peft',peft.__version__,"
                   "'trl',trl.__version__,'bnb',b.__version__)\"; soup --version; pip freeze > logs/pip_freeze.txt")
''')

md("## 1. Data — must rebuild byte-identical to the shipped manifest")
code(r'''
shipped = json.load(open("data/manifest.json"))["sha256"]
sh("data_build", "python build_data.py")
assert json.load(open("data/manifest.json"))["sha256"] == shipped, "STOP: data did not rebuild identically"
print("data reproducible: True")
sh("data_doctor", "soup --no-telemetry data doctor data/train.jsonl -m Qwen/Qwen2.5-1.5B-Instruct --max-length 256 --sample 1500 --show-mask 3 -o results/data_doctor.json", check=False)
sh("data_tokens", """python - <<'EOF'
import json, statistics
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
n = [len(tok(tok.apply_chat_template(json.loads(l)["messages"], tokenize=False), add_special_tokens=False)["input_ids"]) for l in open("data/train.jsonl")]
json.dump(n, open("data/tokens.json", "w"))
print("rows", len(n), "mean", statistics.mean(n), "p95", sorted(n)[int(.95*len(n))], "max", max(n))
EOF""")
''')

md("## 2. Part 1 — memory budget before training, CUDA context, activations on the GPU")
code(r'''
sh("budget_before", "python memory_budget.py --config $(python -c \"from huggingface_hub import hf_hub_download as d;print(d('Qwen/Qwen2.5-1.5B-Instruct','config.json'))\") --tokens data/tokens.json --batch 8 --json-out results/memory_budget.json")
sh("budget_soup_profile", "soup --no-telemetry profile -c configs/main.yaml --json > results/soup_profile.json; cat results/soup_profile.json", check=False)
sh("budget_cuda_context", """python - <<'EOF'
import json, subprocess, torch
torch.zeros(1, device="cuda"); torch.cuda.synchronize()
used = float(subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.split()[0])
ctx = {"nvidia_smi_MiB_with_1_tensor": used, "torch_reserved_MiB": torch.cuda.memory_reserved()/2**20}
ctx["context_MiB"] = used - ctx["torch_reserved_MiB"]
print(json.dumps(ctx)); json.dump(ctx, open("results/cuda_context.json","w"))
EOF""")
sh("budget_measure_activations", "python measure_activations.py --tokens 64 --batch 4 > results/activation_measurement.json; cat results/activation_measurement.json")
''')

md("## 3. Train — main and control, in-process so the allocator counters are readable. Never overwrites an adapter.")
code(r'''
for name, cfg in (("main", "configs/main.yaml"), ("control", "configs/control.yaml")):
    if os.path.exists(f"runs/{name}/adapter_model.safetensors"):
        print(f"SKIP train_{name}: runs/{name} already has an adapter (delete the folder yourself to retrain)")
    else:
        sh(f"train_{name}", f"python train_with_stats.py {cfg} results/train_{name}_cuda_stats.json")
    need(f"runs/{name}/adapter_model.safetensors", f"runs/{name}/rewind.jsonl")
    p = json.load(open(PROV)) if os.path.exists(PROV) else {}
    if f"adapter_{name}" not in p: record(f"adapter_{name}", f"runs/{name}")
    else: check_same(f"adapter_{name}", f"runs/{name}")
''')

md("## 4. Merge (fp16, CPU) and quantize (NF4)")
code(r'''
for name in ("main", "control"):
    check_same(f"adapter_{name}", f"runs/{name}")
    if not os.path.exists(f"runs/{name}-merged/config.json"):
        TIME = "/usr/bin/time -v " if os.path.exists("/usr/bin/time") else ""   # host-RAM peak of the merge
        sh(f"merge_{name}", f"{TIME}soup --no-telemetry merge --adapter runs/{name} --output runs/{name}-merged --dtype float16")
    need(f"runs/{name}-merged/config.json")
    if f"merged_{name}" not in json.load(open(PROV)): record(f"merged_{name}", f"runs/{name}-merged")
if not os.path.exists("runs/main-merged-nf4/config.json"):
    sh("quantize_main", "python quantize_nf4.py runs/main-merged runs/main-merged-nf4")
need("runs/main-merged-nf4/config.json")
if "nf4_main" not in json.load(open(PROV)): record("nf4_main", "runs/main-merged-nf4")
sh("artifact_sizes", "du -sh runs/main runs/main-merged runs/main-merged-nf4 runs/control runs/control-merged")
''')

md("## 5. Part 2 — verify (provenance checked first) and the guarded re-score")
code(r'''
for n, d in (("adapter_main","runs/main"),("merged_main","runs/main-merged"),("nf4_main","runs/main-merged-nf4"),
             ("adapter_control","runs/control"),("merged_control","runs/control-merged")):
    check_same(n, d)
sh("verify", "python verify.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/main --merged runs/main-merged "
             "--quant4 runs/main-merged-nf4 --control-adapter runs/control --control-merged runs/control-merged "
             "--out results/verify.json")
sh("guarded_eval", "python guarded_eval.py results/verify.json --out results/guarded_eval.json")
''')

md("## 6. Part 3 — silent-failure probes")
code(r'''
check_same("adapter_main", "runs/main"); check_same("nf4_main", "runs/main-merged-nf4")
sh("silent_failures", "python silent_failures.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/main "
                      "--merged runs/main-merged --nf4 runs/main-merged-nf4 --verify results/verify.json")
''')

md("## 7. Soup's general suites with every answer recorded (fp16), then Soup's own `ship` gate (bf16)")
code(r'''
check_same("merged_main", "runs/main-merged"); check_same("nf4_main", "runs/main-merged-nf4")
sh("general_probe", "python general_probe.py --base Qwen/Qwen2.5-1.5B-Instruct --tuned runs/main-merged "
                    "--nf4 runs/main-merged-nf4 --out results/general_probe.json")
''')
code(r'''
B = "--base Qwen/Qwen2.5-1.5B-Instruct --task-eval data/ship_eval.jsonl --device cuda"
check_same("adapter_main", "runs/main")
sh("ship_main", f"soup --no-telemetry ship {B} --adapter runs/main -o results/ship_main.json", check=False)       # exit 2 = DON'T SHIP, expected
check_same("adapter_control", "runs/control")
sh("ship_control", f"soup --no-telemetry ship {B} --adapter runs/control -o results/ship_control.json", check=False)
check_same("nf4_main", "runs/main-merged-nf4")
sh("ship_nf4", f"soup --no-telemetry ship {B} --tuned runs/main-merged-nf4 -o results/ship_nf4.json", check=False)
for f in ("results/ship_main.json", "results/ship_control.json", "results/ship_nf4.json"):
    need(f); print(f, json.load(open(f))["decision"])
''')

md("## 8. Memory report, throughput, final provenance check, package")
code(r'''
sh("memory_report", "python phase_report.py")
sh("throughput", """python - <<'EOF'
import json, re
tok = json.load(open("data/tokens.json"))
log = open("logs/train_main.log").read()
rt = float(re.findall(r"'train_runtime': '([0-9.]+)'", log)[-1])
real = sum(tok) * 1350 / len(tok) * 2
trained = sum(sum(json.loads(l)["row_tokens"]) for l in open("runs/main/rewind.jsonl") if '"batch"' in l)
print(f"train_runtime {rt:.1f}s | real tokens/s {real/rt:.0f} | loss-bearing tokens/s {trained/rt:.0f} | loss-bearing share {trained/real:.1%}")
EOF""")
for n, d in (("adapter_main","runs/main"),("merged_main","runs/main-merged"),("nf4_main","runs/main-merged-nf4"),
             ("adapter_control","runs/control"),("merged_control","runs/control-merged")):
    check_same(n, d)
print("ALL ARTIFACTS UNCHANGED FROM BIRTH TO LAST RESULT")
sh("package", "zip -qr deliverables_v2.zip logs results configs data/manifest.json data/*.jsonl *.py soup_task_v2.ipynb "
              "runs/main/soup_run.json runs/main/rewind.jsonl runs/control/rewind.jsonl runs/main-merged-nf4/quantize_stats.json; ls -la deliverables_v2.zip")
from google.colab import files
files.download("deliverables_v2.zip")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("soup_task_v2.ipynb", "w"), indent=1)
print("wrote soup_task_v2.ipynb with", len(cells), "cells")
