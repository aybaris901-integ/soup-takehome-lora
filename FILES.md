# Soup take-home: submission files

- `REPORT.pdf` / `REPORT.md`: the 2-page report. `APPENDIX.md`: full tables, verdict reasoning, run logs, AI use.

- `run2_kaggle/`: the run the report's numbers come from (Kaggle T4, one clean pass, 11 Nov 03:59-05:37 UTC).
  - `logs/`: every command with UTC timestamps, `nvidia_smi_trace.csv` (200 ms), `nvidia_smi_raw.log` (30 s), `pip_freeze.txt`
  - `results/`: `verify.json`, `guarded_eval.json` (as run), `guarded_eval_rescored.json` (error categories corrected
    afterwards by `guarded_eval_rescore.py`; model outputs unchanged), `general_probe.json` (every answer on Soup's
    general suites), `silent_failures.json`, `ship_*.json`, memory files, `provenance.json` (SHA-256 of each weight file)
  - scripts, configs, data and `soup_task_v2.ipynb` (the Kaggle port is `soup_task_kaggle.ipynb` if present)
- `run1_colab/`: the first run (Colab T4, 10 Nov 09:01-11:00 UTC), including its failed attempts. Shown in the report as a replicate.
