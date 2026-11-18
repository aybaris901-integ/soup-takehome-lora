# Did the fine-tune actually work? LoRA → merge → 4-bit → SHIP / DON'T SHIP

My solution to an AI-engineer take-home for [Soup](https://trysoup.dev): fine-tune an open model of up to 3B parameters with LoRA on a free Colab T4, merge it, quantize it to 4-bit, prove that it really trained, find the failures that raise no error, and make a ship decision.

**Verdict: DON'T SHIP for unattended device control; pilot with confirmation.** The full reasoning is in [`REPORT.md`](REPORT.md) (two pages, also as [`REPORT.pdf`](REPORT.pdf)), with everything else in [`APPENDIX.md`](APPENDIX.md).

## What I built

Qwen2.5-1.5B-Instruct learns to turn English device commands into compact tool calls (`turn off the kitchen light in 10 min` → `light off room=kitchen delay=10m`), and to map anything else to `cloud`. Every answer is right or wrong by exact match, so "did it learn?" has an objective answer.

| | eval_real (131 hand-written) | eval_novel (300, unseen words) |
|---|---|---|
| Base model | 0.0% | 0.0% |
| Base + grammar and 12 examples in the prompt | 62.6% | — |
| Merged fp16 | 99.2% | 85.0% |
| **4-bit NF4 (the shipped artifact)** | **98.5%** | **81.3%** |
| Control run (lr 1e-7, learned nothing) | 0.0% | 0.0% |

The training worked. The verdict is still DON'T SHIP, because behind a grammar check the 4-bit model gives valid-but-wrong actions on about 7% of noisy or unfamiliar inputs, sometimes on↔off swaps, and 2.6% of general questions trigger a device action.

## Things worth looking at

- **Proving training without trusting the loss** ([`verify.py`](run2_kaggle/verify.py)). It compares base, unmerged adapter, merged and 4-bit models on weights and behavior. It checks that merged − base equals (α/r)·B·A for every matrix, and runs a control that must fail. The tool printed "Training Complete! Loss: 5.77 → 5.74" for a control run that learned nothing.
- **Memory predicted before training, then explained.** Allocated 6.27 GiB, reserved 11.71 GiB (2.44 GiB of it fragmentation), plus a 0.12 GiB CUDA context, adds up to nvidia-smi's 11.84 GiB.
- **The 4-bit model works, but the training is invisible in its weights:** the LoRA update is 4–12% the size of the NF4 rounding noise.
- **Nine silent failures**, with what the tool did or did not catch ([`silent_failures.py`](run2_kaggle/silent_failures.py), [`general_probe.py`](run2_kaggle/general_probe.py), [`guarded_eval.py`](run2_kaggle/guarded_eval.py)).
- **Provenance:** in run 2, every weight file was hashed at creation and re-checked before every later step. Run 1 lacked this, and an accidental retrain made two of its results judge different models. That is why run 2 exists.
- **The verdict changed between runs**, on criteria written down in advance.

## Layout

```
REPORT.md / REPORT.pdf    two-page report
APPENDIX.md               full tables, confidence intervals, verdict reasoning, run notes, AI-tool use
run2_kaggle/              the run the numbers come from (Kaggle T4, one clean pass)
  *.py                    data generator, memory budget, training with allocator stats, verify,
                          guarded eval, general-suite probe, silent-failure probes, notebook generators
  soup_task_kaggle.ipynb  the notebook that ran (soup_task_v2.ipynb is the Colab version)
  configs/ data/          Soup configs (main + control) and the datasets
  logs/                   every command with UTC timestamps, nvidia-smi traces, pip freeze
  results/                verify, guarded eval, general probe, silent failures, soup ship, memory, provenance
run1_colab/               the first run (Colab T4), failed attempts included
FILES.md                  index of the submission files
```

Model weights are not included (3+ GB). They are rebuilt by running the notebook, and `run2_kaggle/results/provenance.json` holds the SHA-256 of each one.

## Reproduce

On Kaggle (GPU T4 x2, internet on): attach this folder as a dataset, import `run2_kaggle/soup_task_kaggle.ipynb`, then Save & Run All. It takes about 2 hours. The notebook pins one GPU, refuses to retrain an existing adapter, and stops if any artifact's hash changes.

## Tools

Soup 0.77.2, PyTorch, Transformers, PEFT, TRL, bitsandbytes. I used Claude (Anthropic) extensively for the scripts, the analysis and drafting. The full list of what it did, and the numbers I verified myself, are in [`APPENDIX.md`](APPENDIX.md), section 7.
