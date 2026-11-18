"""
Generates soup_task_kaggle.ipynb from make_notebook_v2.py by replacing ONLY the
Colab-specific cells. Every other cell is byte-identical to the v2 notebook, so the
experiment (data, configs, seeds, training, evaluation) is unchanged.

Replaced: the header (run instructions), cell 1 (paths, project discovery, GPU
pinning, telemetry), cell 2 (dependencies), the last cell (packaging/download).
Added: a pre-flight cell (GPU model, internet, disk) right after cell 1.
"""
import json
import re

src = open("make_notebook_v2.py").read()
src = src.replace('json.dump(nb, open("soup_task_v2.ipynb", "w"), indent=1)',
                  'json.dump(nb, open("soup_task_kaggle.ipynb", "w"), indent=1)')
src = src.replace('print("wrote soup_task_v2.ipynb with", len(cells), "cells")',
                  'print("wrote soup_task_kaggle.ipynb with", len(cells), "cells")')
src = src.replace('"colab": {"provenance": []},', '"kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},')


def swap(old_start, old_end, new):
    """Replace the generator text from old_start up to (not including) old_end."""
    global src
    i = src.index(old_start)
    j = src.index(old_end, i)
    src = src[:i] + new + src[j:]


# ---------------------------------------------------------------- header
swap('md("""\n# Soup take-home', 'code(r\'\'\'\nimport os, subprocess', '''md("""
# Soup take-home — v2 on **Kaggle**: one clean run + the four open items from v1

**Before running (Kaggle editor, right-hand panel → Settings):**
1. **Accelerator: GPU T4 x2.** The notebook pins ONE T4 (see cell 1). A P100 is refused, because v1 was measured on a T4.
2. **Internet: On.** It is needed to install Soup and download Qwen2.5-1.5B-Instruct (Kaggle asks for phone verification once).
3. **Add Input → your dataset `SOuptsk`.** It must hold the v2 project: either `soup_task_v2.zip` or its unpacked files.
   Cell 1 searches `/kaggle/input` for it, so the exact folder layout does not matter.

**Run:** *Save Version → Save & Run All (Commit)* runs it headless, with no idle timeout, in ~2 h.
`/kaggle/working/deliverables_v2.zip` then appears under the version's **Output**. *Run All* in the editor works too,
but keep the tab active. **If a cell stops with `STOP:`, send that output; every cell refuses to continue on
missing or mismatched artifacts.**

What v2 changes vs v1: (1) a clean single pass, with torchao removed first, no retraining, and SHA-256 provenance
re-checked on every step; (2) Soup's general suites with every answer recorded, in fp16; (3) the allocator's peak
reserved memory, the CUDA context and the GPU activation measurement; (4) the grammar guard measured, with
opposite-action errors counted. The 4-bit gate in `verify.py` now covers every eval set.

Kaggle-only changes (nothing in the experiment): paths `/content` → `/kaggle/working`; the project is read from
`/kaggle/input`; GPU 0 only; nvidia-smi traces GPU 0; no `google.colab`; the result is a file in `/kaggle/working`.
""")

''')

# ---------------------------------------------------------------- cell 1: setup
swap("code(r'''\nimport os, subprocess", "def now():", r'''code(r\'\'\'
import os, subprocess, time, datetime, json, hashlib, glob, zipfile, shutil

# --- ONE GPU. Kaggle's "T4 x2" exposes two GPUs, and a single-process HF Trainer that sees two
# wraps the model in nn.DataParallel. That would double the batch (8 -> 8 per GPU x 2) and change
# the training. It must be set before anything initialises CUDA; every subprocess inherits it.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
UTC = datetime.timezone.utc

INPUT, WORK = "/kaggle/input", "/kaggle/working"
PROJECT = f"{WORK}/soup_task"                 # was /content/soup_task on Colab
REQUIRED = ["build_data.py", "verify.py", "train_with_stats.py", "general_probe.py", "guarded_eval.py",
            "silent_failures.py", "quantize_nf4.py", "memory_budget.py", "measure_activations.py",
            "phase_report.py", "configs/main.yaml", "configs/control.yaml", "data/manifest.json",
            "data/test_real.tsv", "data/test_novel.jsonl"]

def is_project(d):
    return all(os.path.exists(os.path.join(d, r)) for r in REQUIRED)

def tree(root, depth=4):
    for dirpath, dirs, files in os.walk(root, followlinks=True):
        lvl = dirpath[len(root):].count(os.sep)
        if lvl >= depth: dirs[:] = []; continue
        print("  " * lvl + os.path.basename(dirpath) + "/")
        for f in sorted(files)[:12]: print("  " * (lvl + 1) + f)
        if len(files) > 12: print("  " * (lvl + 1) + f"... {len(files) - 12} more")

def locate_project():
    """Find the v2 project in /kaggle/input WITHOUT assuming its layout: an unpacked folder (Kaggle unpacks an
    uploaded zip), or a zip somewhere inside the dataset. /kaggle/input is read-only, so it is copied to
    /kaggle/working first."""
    if is_project(PROJECT):
        return f"already present in {PROJECT}"
    print("Attached inputs under /kaggle/input:"); tree(INPUT)
    # prefer a dataset whose path mentions SOuptsk (case-insensitive), then anything else
    found = []
    for dirpath, dirs, files in os.walk(INPUT, followlinks=True):
        if is_project(dirpath): found.append(("dir", dirpath))
        for f in files:
            if f.lower().endswith(".zip"): found.append(("zip", os.path.join(dirpath, f)))
    found.sort(key=lambda x: ("souptsk" not in x[1].lower(), x[0] != "dir", x[1]))
    for kind, path in found:
        if kind == "dir":
            shutil.copytree(path, PROJECT)
            return f"copied unpacked project {path} -> {PROJECT}"
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if not any(n.endswith("verify.py") for n in names): continue
            tmp = f"{WORK}/_unzip"; shutil.rmtree(tmp, ignore_errors=True)
            z.extractall(tmp)                                   # zipfile drops absolute / '..' parts
        for dirpath, dirs, files in os.walk(tmp):
            if is_project(dirpath):
                shutil.move(dirpath, PROJECT); shutil.rmtree(tmp, ignore_errors=True)
                return f"extracted {path} -> {PROJECT}"
        raise RuntimeError(f"STOP: {path} contains verify.py but not the full v2 project "
                           f"(needs {REQUIRED}). Is it the v1 soup_task.zip? Upload soup_task_v2.zip.")
    raise RuntimeError("STOP: no v2 project found under /kaggle/input (tree above). In the editor: Add Input -> "
                       "your dataset SOuptsk, which must contain soup_task_v2.zip or its unpacked files.")

print(locate_project())
os.chdir(PROJECT)
for d in ("logs", "results", "runs"):                     # checkpoints and models are written under runs/
    os.makedirs(d, exist_ok=True)
print("working directory:", os.getcwd())

''')

# telemetry: trace GPU 0 only (two rows per timestamp would mix GPUs in phase_report)
src = src.replace('''subprocess.Popen("nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw "''',
                  '''subprocess.Popen("nvidia-smi -i 0 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw "''')
src = src.replace('''done > logs/nvidia_smi_raw.log 2>&1", shell=True)
sh("env_gpu", "nvidia-smi; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv")''',
                  '''done > logs/nvidia_smi_raw.log 2>&1", shell=True)
sh("env_gpu", "nvidia-smi; nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv")''')

# ---------------------------------------------------------------- pre-flight + dependencies
swap('md("## 0. Environment', 'md("## 1. Data', r'''md("## 0. Pre-flight (Kaggle): exactly one T4 visible, internet on, enough disk")
code(r\'\'\'
q = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                   shell=True, capture_output=True, text=True).stdout.strip().splitlines()
print("GPUs on the machine:", q)
if not q:
    raise RuntimeError("STOP: no GPU. Settings -> Accelerator -> GPU T4 x2.")
if "T4" not in q[0]:
    raise RuntimeError(f"STOP: GPU 0 is {q[0]}. v1 was measured on a Tesla T4, so memory, speed and fp16/NF4 "
                       "numerics are only comparable on a T4. Settings -> Accelerator -> GPU T4 x2.")
vis = subprocess.run("python -c \"import torch;print(torch.cuda.device_count(), torch.cuda.get_device_name(0))\"",
                     shell=True, capture_output=True, text=True)
print("visible to PyTorch:", vis.stdout.strip() or vis.stderr[-300:])
if not vis.stdout.startswith("1 "):
    raise RuntimeError("STOP: PyTorch must see exactly 1 GPU (CUDA_VISIBLE_DEVICES=0 is set in cell 1). "
                       "Restart the session and run from the top.")
import urllib.request
for url in ("https://pypi.org/simple/soup-cli/", "https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct"):
    try:
        urllib.request.urlopen(url, timeout=15); print("reachable:", url)
    except Exception as e:
        raise RuntimeError(f"STOP: {url} unreachable ({e}). Settings -> Internet -> On.")
for p in (WORK, os.path.expanduser("~")):
    free = shutil.disk_usage(p).free / 2**30
    print(f"free disk {p}: {free:.1f} GiB")
    if free < 12: raise RuntimeError(f"STOP: {p} has {free:.1f} GiB free; the run writes about 8 GiB of models.")
\'\'\')

md("## 0b. Environment: torchao removed before anything imports peft (v1's failure)")
code(r\'\'\'
# Kaggle preinstalls torch/transformers/peft/accelerate. pip only replaces what Soup 0.77.2's pins require
# (transformers>=5.16.1,<6, peft>=0.20,<1, trl>=0.29,<1, torch>=2.6). Torch is recorded before and after so a
# swap is visible in the log. The long pip output goes to logs/pip_install.txt; its last lines are shown.
sh("env_torch_before", "python -c \"import torch;print('torch before', torch.__version__, torch.version.cuda)\"", check=False)
sh("env_install", 'pip install -q "soup-cli[train]==0.77.2" > logs/pip_install.txt 2>&1; rc=$?; tail -5 logs/pip_install.txt; '
                  'pip uninstall -y -q torchao 2>/dev/null; '
                  '[ -x /usr/bin/time ] || (apt-get -qq update > /dev/null && apt-get -qq install -y time > /dev/null) || true; exit $rc')
sh("env_versions", "which soup && python -c \"import importlib.util as u; assert u.find_spec('torchao') is None, 'torchao still installed'; "
                   "import torch,transformers,peft,trl,bitsandbytes as b;print('torch',torch.__version__,'cuda',torch.version.cuda,"
                   "'gpus',torch.cuda.device_count(),'bf16_supported',torch.cuda.is_bf16_supported());"
                   "print('transformers',transformers.__version__,'peft',peft.__version__,"
                   "'trl',trl.__version__,'bnb',b.__version__)\"; soup --version; pip freeze > logs/pip_freeze.txt")
sh("env_bnb_smoke", "python -c \"import torch,bitsandbytes.functional as F; w=torch.randn(64,64,device='cuda',dtype=torch.float16); "
                    "q,s=F.quantize_4bit(w,quant_type='nf4'); print('bnb nf4 ok, max err', (F.dequantize_4bit(q,s)-w).abs().max().item())\"")
\'\'\')

''')

# ---------------------------------------------------------------- packaging
old_pkg_start = 'sh("package", "zip -qr deliverables_v2.zip'
old_pkg_end = "''')\n\nnb = {"
swap(old_pkg_start, old_pkg_end, r'''# package: Python zipfile, so a missing optional file is listed, not fatal
OUT = f"{WORK}/deliverables_v2.zip"
items = (["logs", "results", "configs", "data/manifest.json"] + sorted(glob.glob("data/*.jsonl")) + sorted(glob.glob("*.py"))
         + sorted(glob.glob("*.ipynb")) + ["runs/main/soup_run.json", "runs/main/rewind.jsonl", "runs/control/rewind.jsonl",
         "runs/main-merged-nf4/quantize_stats.json"])
missing = [i for i in items if not os.path.exists(i)]
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for i in items:
        if os.path.isdir(i):
            for f in sorted(glob.glob(f"{i}/**/*", recursive=True)):
                if os.path.isfile(f): z.write(f)
        elif os.path.exists(i):
            z.write(i)
print(f"wrote {OUT} ({os.path.getsize(OUT)/2**20:.1f} MiB); not found (listed, not packaged): {missing}")
from IPython.display import FileLink, display
display(FileLink(os.path.relpath(OUT, os.getcwd())))     # editor download link; after a commit it is in Output
''')

open("_make_kaggle_tmp.py", "w").write(src)
exec(compile(src, "make_notebook_kaggle(v2-derived)", "exec"))
