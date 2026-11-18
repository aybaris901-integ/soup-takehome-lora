"""
Quantize the merged fp16 model to 4-bit NF4 (bitsandbytes, double quant) and save it,
recording the GPU peak so it can be compared with memory_budget.py.

    python quantize_nf4.py runs/main-merged runs/main-merged-nf4

Why NF4 via bitsandbytes and not `soup quantize` (GGUF): the 4-bit artifact must be
loadable by the same transformers harness as base and merged, so verify.py can compare
all three with one tokenizer path, one decoding loop and teacher-forced log-probs.
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

src, dst = sys.argv[1], sys.argv[2]
assert torch.cuda.is_available(), "bitsandbytes 4-bit needs the GPU"
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(src, quantization_config=q, device_map={"": 0})
load_peak = torch.cuda.max_memory_allocated()
steady = torch.cuda.memory_allocated()
tok = AutoTokenizer.from_pretrained(src)
os.makedirs(dst, exist_ok=True)
model.save_pretrained(dst)
tok.save_pretrained(dst)

n4 = sum(1 for m in model.modules() if m.__class__.__name__ == "Linear4bit")
not_q = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m.__class__.__name__ != "Linear4bit"]
stats = {"src": src, "dst": dst, "seconds": round(time.time() - t0, 1),
         "gpu_peak_alloc_GiB": load_peak / 2 ** 30, "gpu_steady_alloc_GiB": steady / 2 ** 30,
         "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2 ** 30,
         "linear4bit_modules": n4, "linear_layers_left_unquantized": not_q,
         "footprint_GiB(get_memory_footprint)": model.get_memory_footprint() / 2 ** 30,
         "saved_bytes": sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))}
print(json.dumps(stats, indent=1))
json.dump(stats, open(os.path.join(dst, "quantize_stats.json"), "w"), indent=1)
