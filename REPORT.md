# Soup take-home: tool-call fine-tune of Qwen2.5-1.5B-Instruct

**Author:** Aybaris Zhenis · **Date:** 10 November 2026 · Details, full tables and logs index: `APPENDIX.md`

## Verdict: DON'T SHIP for unattended device control; pilot with confirmation

The 4-bit (NF4) model must not drive devices on its own. It can go into a pilot where each action is confirmed by the user.

The fine-tune is real: 98.5% exact match on 131 hand-written commands, against 0% for the base model and 62.6% for the best few-shot prompt. But run 2 measured three failures a grammar check cannot remove:

- **Valid but wrong actions:** 7.0% on noisy in-distribution input and 7.3% on unfamiliar wording, with the grammar check on.
- **Opposite actions:** 10 of 300 on unfamiliar wording (`put it down` → `grip`).
- **False actuation:** 8 of 304 general prompts (2.6%) produced a valid device action.

`soup ship` also says DON'T SHIP (7 of 8 general suites regressed). After run 1 I recommended a scoped SHIP with stop conditions of 2% valid-but-wrong and no on↔off swaps. Run 2 broke both, so I reversed the verdict.

## Task and data

The model turns English device commands into compact tool calls (`turn off the kitchen light in 10 min` → `light off room=kitchen delay=10m`); anything else must map to `cloud`. I used a synthetic template dataset (1,500 rows) instead of an open instruction dataset. Exact match makes "did it learn" objective, and I control leakage: held-out words are banned from training, with 0 overlaps. The cost is that it is narrower than real language, so every score here is an upper bound on real use.

Setup: Soup 0.77.2, LoRA r=16 on all 7 linear layers of an fp16 base (not QLoRA), 2 epochs, seed 42; `soup merge` to fp16, then bitsandbytes NF4. A control run (lr 1e-7) goes through the same checks. Numbers are from run 2 (Kaggle T4); run 1 (Colab T4) is a replicate.

## Part 1: Memory

| Phase | Predicted before | Soup pre-flight | torch allocated | torch reserved | nvidia-smi |
| --- | --- | --- | --- | --- | --- |
| Training | 6.47 GiB | 15.25 GiB (16.4 GB) | 6.27 GiB | 11.71 GiB | 11.84 GiB |
| Merge (CPU) | 5.9 GiB + ~1 runtime | — | 6.63 GiB RSS | — | 0.17 GiB |
| NF4 quantize | 1.50 GiB peak | — | 1.10 GiB | 1.12 GiB | 1.24 GiB |

- My estimate was 3.2% above the allocated peak, partly by luck: measured on the T4, activations are 77% of my formula, so about 0.36 GiB is unmodelled elsewhere.
- Soup's pre-flight was 2.4× too high and refused to launch. It prices every row at 256 tokens; real rows average 44.
- **Allocated vs reserved vs nvidia-smi:** live tensors peaked at 6.27 GiB, but PyTorch's caching allocator held 11.71 GiB (2.44 GiB of it fragmentation). The CUDA context adds 0.12 GiB, which gives nvidia-smi's 11.84. Soup's "GPU peak: 6.3/14.6 GB" is the allocated figure in GiB, labelled GB, so it understates what the card held by 5.4 GiB.

## Part 2: Did it train?

Yes. Exact match, greedy decoding, run 2 (run 1 in brackets):

| Artifact | eval_real (131) | eval_novel (300) | val (200) |
| --- | --- | --- | --- |
| Base | 0.0% | 0.0% | 0.0% |
| Base + grammar and 12 examples | 62.6% | — | — |
| Merged fp16 | 99.2% (100) | 85.0% (84.7) | 92.0% (93.0) |
| **NF4 4-bit** | **98.5%** (99.2) | **81.3%** (80.0) | **91.0%** (91.5) |
| Control (lr 1e-7) | 0.0% | 0.0% | 0.0% |

- Merged vs few-shot: +36.6 pt, 95% CI [28.2, 45.0]. NF4 vs merged on eval_novel: −3.7 pt [−6.3, −1.0] over all 300 rows; on the 250 unique texts −4.0 pt [−6.8, −1.2].
- Weights: exactly the 196 target matrices changed. Merged − base matches (α/r)·B·A to within 6.5% (largest relative Frobenius-norm error over the 196).
- Loss is not proof: Soup printed "Training Complete! Loss: 5.77 → 5.74" for the control, which scores 0%.
- In the 4-bit weights the training is invisible: the LoRA delta is 4–12% of the NF4 rounding noise (22 sampled matrices). Only behavior shows it.

NF4 behind a grammar check (invalid output → `cloud`):

| Set | Correct | Deferred | Valid but wrong | Opposite action |
| --- | --- | --- | --- | --- |
| eval_real (131) | 129 | 2 | 0 | 0 |
| val (200) | 182 | 4 | 14 (7.0%) | 1 |
| eval_novel (300) | 245 | 33 | 22 (7.3%) | 10 |

The check cuts wrong actions on eval_novel from 17.0% to 7.3%. "Wrong" excludes deferrals to `cloud` and compares durations in seconds, so it differs from 100 − exact match (18.7%).

## Part 3: Silent failures (top 5 of 9)

| Silently wrong | Evidence | Caught by Soup? |
| --- | --- | --- |
| General text triggers device actions | 304 general prompts: 2.6% valid actions (`hotwire a car` → `relay 1 off\|relay 2 on`) | Partly: `soup ship` flags the suites as regressed, not the actions |
| 4-bit gives wrong but valid actions | 7.0–7.3% after the grammar check | No: `soup ship` scored the NF4 task leg "won" |
| Loss includes fixed tokens | 10.2 loss tokens per row vs 5.2 answer tokens; 49% are template | Data doctor: yes, rated MINOR |
| "Training Complete" on a no-op run | Control: loss 5.77 → 5.74, val loss 6.0 | Not at training; `soup ship` later: yes ("no win") |
| System prompt at serve time | Merged: 99.2% → 93.1% with `You are a helpful assistant.` | No |

## What would change the verdict

1. Retrain with command-like negatives and the phrase families behind the swaps, then test on a **new** held-out set; eval_novel is spent.
2. Ship the fp16 merge where 3 GiB is available.
3. Reach at most 2% valid-but-wrong and 0 opposite actions on that set, keeping confirmation for irreversible actions.

## What surprised me and what still concerns me

Two things surprised me. The 4-bit model still works although the trained change is an order of magnitude smaller than the quantization noise in the weights. And my own run-1 verdict was wrong: I had explained the general-suite collapse as "routing to `cloud`" without seeing the answers, and saving them showed 10% is not routing. What still concerns me is that every eval set is synthetic or written by me, so I do not know the error rate on real users, and that the false-actuation rate comes from only 304 prompts.

**AI use:** I used Claude extensively, for the scripts, the analysis and drafting this report. The full list is in `APPENDIX.md`, section 7, with the eight headline numbers I checked against the source files myself.
