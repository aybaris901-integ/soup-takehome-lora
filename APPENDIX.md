# Appendix: Soup take-home, tool-call fine-tune of Qwen2.5-1.5B-Instruct

**Author:** Aybaris Zhenis · **Date:** 10 November 2026 · Main report: `REPORT.md`

All numbers come from run 2 (`run2_kaggle/`) unless marked run 1 (`run1_colab/`). Source files are named in brackets.

## 1. Setup

The model learns to turn English device commands into compact tool calls: `turn off the kitchen light in 10 min` → `light off room=kitchen delay=10m`. Anything that is not a device command must map to `cloud`. Every answer is right or wrong by exact match, and the base model has never seen this format.

- **Data:** 1,500 training rows from a template generator with typos and dropped words (Soup holds out 150 for its validation loss). Three held-out sets: `eval_real`, 131 commands written by hand; `eval_novel`, 300 rows (250 unique texts) whose key words are banned from training (`lounge`, `deactivate`, `cooler`, `cyan`); `val`, 200 rows from the training distribution. Leakage check: 0 overlaps. The data rebuilt byte-identically on both platforms [`data/manifest.json`].
- **Why not an open instruction dataset:** the brief suggests one. I chose a synthetic template dataset because exact match gives an objective answer to "did it learn", and because I can control leakage by banning held-out words from training. The trade-off is that template text is narrower than real language, so scores on it overstate real-world accuracy. The 131 hand-written commands reduce that gap but do not close it.
- **Training:** Soup 0.77.2 SFT on Qwen2.5-1.5B-Instruct. LoRA r=16, α=32, on all 7 linear layers (18,464,768 parameters, 1.18%), on an **fp16 base, not QLoRA**, so the adapter is merged into the weights it was trained against. 2 epochs, lr 2e-4, batch 8 × accumulation 2, seed 42, 170 steps, 112.5 s [`logs/train_main.log`].
- **Artifacts:** `soup merge` to fp16, then bitsandbytes NF4 with double quantization: 1.07 GiB, 196 block linears 4-bit, `lm_head` left unquantized [`quantize_stats.json`]. The 4-bit model is the one that would ship.
- **Control run:** the same config with lr 1e-7. Its adapter is non-zero but has learned nothing, so any check that cannot reject it is worthless.
- **Two runs.** Run 1: Colab T4, torch 2.14.1, transformers 5.22.1. Run 2: Kaggle T4 (one of two GPUs pinned), torch 2.11.0, transformers 5.20.2 [`logs/pip_freeze.txt`], with four additions planned after run 1: SHA-256 provenance on every artifact, Soup's general suites with all answers saved, allocator memory counters, and a grammar-guard re-score. No training data or config changed between runs.

## 2. Part 1: Memory, predicted before training vs measured

My itemised estimate (`memory_budget.py`, run before training with the real token counts) was 3.2% above the training peak PyTorch allocated (6.470 vs 6.272 GiB). Soup's pre-flight check predicted 2.4× the measured peak and refused to launch; I overrode it with `--allow-oom-attempt`.

| Phase | My prediction | Soup pre-flight | Measured: torch allocated | Measured: torch reserved | Measured: nvidia-smi |
| --- | --- | --- | --- | --- | --- |
| LoRA training (GPU) | 6.47 GiB | 15.25 GiB (printed as 16.4 GB) | 6.27 GiB | 11.71 GiB | 11.84 GiB |
| Merge (`soup merge` runs on CPU) | 5.9 GiB RAM + ~1 GiB runtime | — | 6.63 GiB max RSS | — | 0.17 GiB |
| NF4 quantize (GPU) | 1.06 GiB steady, 1.50 peak | — | 1.07 steady, 1.10 peak | 1.12 GiB | 1.24 GiB |

Sources: `results/memory_budget.json`, `soup_profile.json`, `train_main_cuda_stats.json`, `memory_actual.json`, `cuda_context.json`, `logs/merge_main.log`.

- **Units.** Soup's pre-flight reports decimal GB (16,379,518,485 bytes = 16.38 GB = 15.25 GiB). Its training panel reports GiB labelled "GB" (6.3/14.6; the device total is 14.56 GiB). In one unit the pre-flight is 15.25 / 6.27 = 2.4× the measured allocated peak. An earlier draft said 2.6×, which divided GB by GiB.
- **The 5.6 GiB between "allocated" and nvidia-smi is measured.** PyTorch's caching allocator reserved 11.71 GiB at peak while 6.27 GiB was in live tensors; 2.44 GiB of the difference is `inactive_split` (fragmentation). I attribute the fragmentation to batches of different lengths; I did not isolate the cause. The CUDA context adds 0.12 GiB (121 MiB). 11.71 + 0.12 = 11.82 GiB, against nvidia-smi's 11.84.
- **My total was right partly by luck.** The estimate is weights 2.875 + LoRA weights, grads and Adam 0.276 + activations 2.395 + logits 0.675 + workspace 0.25 GiB. Measured on the T4 in fp16, the bytes autograd saves per token per layer are 77% of my formula (147,464 vs 192,480) [`activation_measurement.json`], so activations are about 0.56 GiB lower than predicted. The total still landed 0.2 GiB high, so roughly 0.36 GiB is unmodelled elsewhere. The same formula matched within 1% on CPU in bf16 (pre-run check), so the saved-tensor set depends on device and dtype.
- **Soup's pre-flight** prices every row at `max_length` = 256 tokens (`seq_len: 256`). Real rows average 44.2 tokens (p95 57, max 75). Its activation term is 7.46 GiB (8.01 GB) against my 2.40 GiB, and its logits term 3.48 GiB (3.73 GB) against my 0.68 GiB.
- **Quantize:** I over-predicted the peak by 0.4 GiB. I assumed the 0.44 GiB embedding passes through as a temporary fp16 tensor, but it is never quantized, so there is nothing to pack.

**Can Soup's VRAM number be trusted?** Only for what it measures. "GPU peak: 6.3/14.6 GB" is `max_memory_allocated` in GiB, labelled GB. The process held 11.7 GiB of the card. For deciding whether a second job fits on the same GPU, Soup's number is 5.4 GiB too low; its pre-flight is 2.4× too high.

**Can tok/s be trusted?** Yes, as a count. TRL's `num_tokens` (59.6k per epoch) equals 1,350 rows × 44.2 real tokens, so padding is not counted. That is 1,060 real tokens/s, but only 23.1% of those tokens carry loss (245/s) [`logs/throughput.log`], and 48.9% of those are fixed template tokens (section 4). About 12% of processed tokens carry task signal.

## 3. Part 2: Did it actually train?

Yes. The artifact that ships, not just the training process, changed in exactly the places LoRA should change it, and on held-out data its behavior moved from 0% to 98–99%. The control run went through the same checks and failed every behavioral one. A falling loss would not have shown this: Soup reported the control as "Training Complete! Loss: 5.7728 → 5.7405", but its validation loss stayed at 6.0 (6.008 → 5.988) and it learned nothing.

Exact match, greedy decoding, each artifact loaded with its own tokenizer and chat template. Run 2, with run 1 in brackets [`results/verify.json`]:

| Artifact | eval_real (131) | eval_novel (300) | val (200) | gold log-prob / token (eval_real) |
| --- | --- | --- | --- | --- |
| Base, no prompt | 0.0% | 0.0% | 0.0% | −7.73 |
| Base, grammar + 12 examples in prompt | 62.6% (82/131) | — | — | — |
| Adapter (unmerged) | 99.2% (100) | 85.0% (84.7) | 92.0% (93.0) | −0.006 |
| Merged fp16 | 99.2% (100) | 85.0% (84.7) | 92.0% (93.0) | −0.006 |
| **NF4 4-bit (shipped)** | **98.5%** (99.2) | **81.3%** (80.0) | **91.0%** (91.5) | −0.009 |
| Control (lr 1e-7) | 0.0% | 0.0% | 0.0% | −7.71 |

The two runs used the same data, config and seed on different software stacks and agree within 1.3 points everywhere. They are not identical: one eval_real example differs. A fixed seed did not make the run reproducible across platforms.

**Confidence intervals.** Paired bootstrap, 5,000 resamples, 95%, run 2. The bootstrap resamples **rows**. That is exact for eval_real and val (all texts unique). eval_novel has 300 rows but 250 unique texts, so its row-level intervals are slightly too narrow; I recomputed them on unique texts (first occurrence; duplicate texts always received identical outputs).

| Comparison | All rows | Unique texts |
| --- | --- | --- |
| Merged vs few-shot, eval_real | +36.6 pt [28.2, 45.0] | same (131 unique) |
| NF4 vs merged, eval_real | −0.8 pt [−2.3, 0.0] | same |
| NF4 vs merged, val | −1.0 pt [−3.0, +1.0] | same (200 unique) |
| NF4 vs merged, eval_novel | −3.7 pt [−6.3, −1.0] (n=300) | −4.0 pt [−6.8, −1.2] (n=250) |
| Base vs merged, eval_novel | +85.0 pt [81.0, 88.7] | +86.0 pt [81.6, 90.0] |

Run 1, NF4 vs merged on eval_novel (rows): −4.7 pt [−7.7, −2.0]. The 4-bit loss on unfamiliar wording appears in both runs and survives de-duplication. The adapter and the merged model give the same output on 630 of 631 prompts; KL(merged ‖ NF4) = 0.0015 nats/token. My 4-bit gate now covers every eval set, and it **fails**: on eval_novel only 92% of NF4 outputs match the merged model.

**The 4-bit model behind a grammar check** (invalid output → `cloud`) [`results/guarded_eval_rescored.json`]:

| NF4, guarded | Correct | Deferred to cloud | Valid but wrong | of which on↔off or grip↔release |
| --- | --- | --- | --- | --- |
| eval_real (131) | 129 | 2 | 0 | 0 |
| val (200), typos and chains | 182 | 4 | 14 (7.0%) | 1 |
| eval_novel (300), unseen words | 245 | 33 | 22 (7.3%) | 10 (6 unique texts) |
| eval_novel, 250 unique texts | 206 | 26 | 18 (7.2%) | 6 |

Without the check, wrong actions on eval_novel are 51 of 300 (17.0%); with it, 22 (7.3%). This "wrong" rate is not 100 − exact match (18.7%, 56 rows): it excludes the 4 outputs that were already `cloud`, and it compares durations in seconds, which accepts 1 row (`timer 120m` for `timer 2h`). The check cannot touch a grammatical mistake: `shut the fan up` → `fan on`, `put it down` → `grip`, `i ened the desk fan` → `fan off`.

**Weight checks, from the files:**
- Exactly the 196 target matrices changed (28 layers × 7), and 0 other tensors.
- **Merge matches the adapter.** For each target matrix I compute ‖(W_merged − W_base cast to fp16) − (α/r)·B·A‖_F / ‖(α/r)·B·A‖_F, a relative Frobenius-norm error. The largest of the 196 is 0.065. For the control it is 0.98, because its delta is mostly below fp16 resolution. I attribute the 6.5% to fp16 rounding of the merged weights; that cause is consistent with the numbers but not isolated. No adapter layer has an all-zero B.
- **The bf16 → fp16 cast** in `soup merge` changed 251,985 of 1,543,714,304 base values, and 2,700 became exactly 0. The metric I recorded is, per tensor, ‖cast − original‖_F / ‖original‖_F; its maximum over tensors is 4.2e-8. It is not an element-wise error: each of the 2,700 zeroed values has 100% relative error. Their absolute size was not recorded [CHECK: by the fp16 format they must be below its smallest subnormal, about 6e-8, but I did not measure it].
- fp16 rounding erased 3.8% of the main run's delta elements, but 95.2% of the control's.
- **In the 4-bit model the training cannot be seen in the weights.** In 22 sampled matrices the LoRA delta's norm is 4–12% of the NF4 rounding-noise norm. The behavior survives; my explanation is that the noise is unstructured while the delta is low-rank, which I did not test directly.
- The control also passes "only target matrices changed", because its tiny delta is non-zero. Weight checks alone would not catch a no-op run.
- Every result names the bytes it judged: the SHA-256 of each adapter, merged and NF4 file was recorded at creation and re-checked before every later step [`results/provenance.json`].

**What this check does not catch:**
- Generalization to real users. Every eval set is synthetic or written by me.
- Exact-match false negatives, such as `timer 120m` for `timer 2h` (the guarded table compares durations in seconds).
- Other serving stacks (vLLM, llama.cpp chat templates), NF4 kernels on other GPUs, long or adversarial input, multi-turn use.
- I corrected the error categories in `guarded_eval.py` after seeing the outputs (timer-unit errors were filed under "other"; `grip adjust` vs `release` was counted as an opposite). The model outputs were not re-generated; both the original and re-scored files are in `results/`.

## 4. Part 3: Silent failures

Nine things went wrong, or could, without any error. "Caught by Soup?" is based only on what Soup printed in these runs' logs.

| What is silently wrong | Evidence | Consequence | Caught by Soup? |
| --- | --- | --- | --- |
| General text triggers device actions | Soup's 304 general-suite prompts through the NF4 model: 90.5% answered `cloud`, 6.9% invalid, **2.6% (8) a valid device action**. `Give me instructions to hotwire a car` → `relay 1 off\|relay 2 on` [`general_probe.json`] | A non-command actuates hardware, and the grammar check passes it | Partly. `soup ship` marks 7 of 8 general suites REGRESSED and says DON'T SHIP; it does not show the answers or flag actions |
| 4-bit gives wrong but valid actions | Guarded NF4: 7.0% on val, 7.3% on eval_novel, with on↔off swaps | The device does the wrong thing with no error | No. `soup ship` on the NF4 model scored the task leg 0.9847 "won" (on eval_real, the set I gave it) |
| Numbers change after quantization | 4 of the 33 merged → NF4 differences are numbers only (`timer 12m` → `timer 2m`, `delay=90m` → `99m`) [`silent_failures.json`] | Durations change silently | No |
| Loss includes fixed tokens | Soup's `rewind.jsonl`: 10.2 loss tokens per row vs 5.2 answer tokens. Qwen's template has no `{% generation %}` markers, so Soup's fallback mask also trains on the role header and end-of-turn tokens | 48.9% of loss tokens are trivially predictable | Data doctor: yes, as `generation_markers`, rated MINOR; training proceeds |
| "Training Complete" on a run that learned nothing | Control: "Loss: 5.7728 → 5.7405" (first vs last logged batch); val loss 6.008 → 5.988 | A success box and a falling loss for a no-op run | Not at training time. `soup ship` later: yes, task leg "no win", DON'T SHIP |
| Same seed, different model | Run 1 and run 2 share data, config and seed 42; merged eval_real is 131/131 vs 130/131 | A seed does not pin the result across library versions | No |
| A system prompt at serve time | eval_real, merged: none / Qwen default 99.2%; `You are a helpful assistant.` 93.1%; a task instruction 91.6% | A server that adds a system prompt costs 6–8 pt | No. Data doctor's `system_role` check reports "no rows use a system message" as OK |
| Training text was lowercased | The generator normalizes case; `Title Case` input scores 96.2% | Real typing loses about 3 pt | No |
| Soup's `exact` scorer ignores case and whitespace | `LIGHT ON` scores correct against `light on` | Soup's task metric would pass outputs a strict parser rejects | No (it is Soup's own scorer) |

Soup's pre-flight caught none of these. Its one intervention was a false alarm: it refused a run that peaked at 6.27 GiB allocated.

Checked and clean: tokenizer, chat template and EOS are identical across base, adapter, merged and NF4, and pad ≠ eos. Batch size 1 vs 32 gives identical outputs. Every row trains its end-of-turn token; the stop rate is 100%. 0 of 1,500 rows are truncated. The shipped `generation_config` (sampling at T=0.7, repetition penalty 1.1) costs 0–0.8 pt.

One run-1 finding did not survive. Run 1 suggested Soup's bf16 judge disagreed with fp16 (99.24% vs 100%). In run 2, Soup's bf16 score and my fp16 score are identical (99.24%), so the run-1 gap came from the adapter being retrained between the two measurements, not from numerics.

## 5. Verdict reasoning in full

Do not ship the NF4 model as a parser whose output drives devices directly. It can go into a pilot where every action is shown to the user and confirmed before it runs.

**This reverses my run-1 verdict, on the criteria I wrote then.** After run 1 I recommended a scoped SHIP with stop conditions: more than 2% valid-but-wrong actions, or any on↔off swap. Run 2 measured both for the first time, with the grammar check applied:

- **Valid but wrong: 7.0% on val and 7.3% on eval_novel**, against a 2% limit. On the 131 hand-written commands it is 0, so the failures come from typos, chained commands and unfamiliar words.
- **Opposite actions: 10 of 300 on eval_novel and 1 of 200 on val.** `put it down` → `grip` matters when a gripper is holding something.
- **False actuation: 8 of 304 general prompts (2.6%) produced a valid device action.** The run-1 report called the collapse on Soup's general suites "the specification working". The saved answers show that is 90% true: the model says `cloud`. Of the other 29 answers, 8 would move hardware.

**Soup's DON'T SHIP stands, for a reason Soup did not state.** Its gate failed the model for forgetting: 7 of 8 general suites regressed, six of them to 0%. For a single-purpose parser that would be acceptable if every non-command became `cloud`. It does not, and adding "this is a general question, answer normally" to the prompt restores nothing (still 0% on six suites).

**What is solid.** The fine-tune is real and replicated on two platforms. It beats the best prompt-only baseline by 36.6 points [28.2, 45.0], the merge reproduces the adapter's outputs on 630 of 631 prompts, and on clean hand-written commands the guarded 4-bit model made 0 wrong actions in 131.

**What would change the verdict:**
1. Retrain with the missing cases: negatives that look like commands, and the three phrase families behind the swaps (`shut … up`, `put it down`, `get … going`). Then test on a **new** held-out set written before retraining; eval_novel is now spent.
2. Ship the fp16 merge where 3 GiB is available. NF4 costs 3.7 points on unfamiliar wording in run 2 and 4.7 in run 1.
3. Keep the grammar check and serve with no system prompt and lowercased input, as trained.
4. Accept at most 2% valid-but-wrong and 0 opposite actions on that new set, with confirmation kept for irreversible actions.

## 6. What I would change in Soup

1. **Make `soup ship`'s task gate statistical.** Today it ships on any point-estimate win, so one flipped example in 131 (+0.8 pt) counts. Require a paired-bootstrap CI or a minimum effect size, and write per-item outputs to the JSON so a verdict can be audited.
2. **Let a run declare its scope.** For a single-purpose model, show the general suite as information rather than a veto, or run it with the system prompt the deployment uses. Today a router that answers "cloud" and one that has lost its abilities get the same "REGRESSED" row. I could only tell them apart by saving the answers myself, and the difference decided my verdict.
3. **Price the pre-flight from real token lengths.** Data doctor already knows p95 = 57 tokens; the pre-flight assumed 256 and refused a run that peaked at 6.27 GiB.
4. **Judge the "Training Complete" box on validation loss against a step-0 baseline.** Comparing the first and last logged batch made a no-op run look like progress.
5. **Report VRAM honestly.** Label GiB as GiB, use one unit in pre-flight and panel, and show reserved memory next to `max_memory_allocated`.
6. **Use a stronger no-op guard.** Replace "all lora_B == 0" with a behavioral delta on the task eval, or a ‖ΔW‖/‖W‖ threshold that also counts the part of the delta erased by fp16 rounding at merge (95% for the control).
7. **Raise missing `{% generation %}` markers from MINOR to WARNING.** For known templates such as ChatML, mask the role header by default.
8. **Offer a strict scorer and state the judge's numerics.** bf16 and fp16 agreed on this task, but the gate should still be able to judge in the dtype that ships. Add a case-sensitive `exact` scorer.

## 7. Run log notes and AI-tool use

Both runs' logs are submitted with UTC timestamps per line; nothing was removed.

**Run 1 (Colab T4, 10 Nov, 09:01–11:00 UTC):**
- Colab's preinstalled torchao 0.10 broke peft. The first training attempts failed, as did the GPU activation measurement. Uninstalling torchao fixed it.
- Soup's pre-flight refused to launch (16.4 GB predicted). I ran with `--allow-oom-attempt`.
- I ran cells past a failed training once; merge and verify then failed on the missing adapter.
- After verify, the training cells ran a second time and overwrote both adapters, so `soup ship` judged a different adapter than verify did. This is why run 2 exists.

**Run 2 (Kaggle T4, 11 Nov, 03:59–05:37 UTC):**
- I had used up my Colab GPU quota, so the notebook was ported to Kaggle: paths, project discovery under `/kaggle/input`, one of the two T4s pinned so the batch size stays 8, and no `google.colab`. Cells from data build to `soup ship` are byte-identical to the Colab v2 notebook.
- The first Kaggle attempt had no GPU attached (`nvidia-smi: not found`); the accelerator needed phone verification.
- After that, one pass from top to bottom, every phase exit 0 (`soup ship` exits 2 for DON'T SHIP), and the final cell confirmed all five weight files unchanged since creation.
- Changes made after seeing run-1 results, so they are not blind: the 4-bit gate now covers all eval sets, and the guarded re-score and general-suite probe were added. No training data or config changed.

**AI-tool use.** I used Claude (Anthropic) to:
- choose the task framing
- write the data generator, all scripts and both notebooks, and port the notebook to Kaggle
- read Soup's source to find how it masks the loss, prices memory, merges and scores
- smoke-test each pipeline on CPU with a tiny random model before I spent GPU time
- debug the Colab and Kaggle failures
- analyse `results/*.json` and draft this report, including the change of verdict after run 2
- check this report's numbers against the source files and split it into `REPORT.md` and `APPENDIX.md`

I ran every step on Colab and Kaggle myself.

**What I verified myself.** I opened each source file listed below and confirmed that the value in this report matches it. All eight match.

- [x] Cast: 251,985 values changed, 2,700 became 0, max per-tensor relative Frobenius error 4.2e-8 → `results/verify.json` → `weights.main.dtype_cast`
- [x] Pre-flight 16,379,518,485 bytes (15.25 GiB) vs 6.27 GiB allocated = 2.4× → `results/soup_profile.json` → `total_memory_bytes`; `results/train_main_cuda_stats.json` → `peak_allocated_GiB`
- [x] Guard effect on eval_novel: 51/300 (17.0%) → 22/300 (7.3%); exact match 244/300 → `results/guarded_eval_rescored.json` → `nf4/eval_novel/raw` and `/guarded`
- [x] Prediction 6.470 GiB vs 6.272 GiB allocated = 3.2% above → `results/memory_budget.json` → `train.total_alloc` (bytes ÷ 2^30)
- [x] eval_novel: 300 rows, 250 unique texts; NF4 vs merged −3.7 pt [−6.3, −1.0] on rows → `results/verify.json` → `comparisons["merged->nf4/eval_novel"]`. The unique-text interval (−4.0 pt [−6.8, −1.2]) is not in a results file; it was recomputed from `em_vec` in `verify.json` with the same bootstrap function.
- [x] Merge error 0.065 (largest relative Frobenius-norm error over 196 matrices) → `results/verify.json` → `weights.main.max_rel_err_merged_vs_pred`
- [x] Reserved 11.71 GiB, fragmentation 2.44 GiB, context 121 MiB → `results/train_main_cuda_stats.json`, `results/cuda_context.json`
- [x] False actuation 8 of 304 → `results/general_probe.json` (count NF4 "plain" outputs that are not `cloud` and pass the grammar)
