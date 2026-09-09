# Archived Experiment Results

These CSVs contain the exact metrics printed by the final submitted notebook.

They are stored separately from newly generated outputs so that the historical result record is not silently overwritten by future reruns.

- `submitted_holdout_metrics.csv` — chronological 80/20 real-market holdout.
- `submitted_cv_metrics.csv` — all four strategies for each of five walk-forward folds.
- `submitted_cv_summary.csv` — mean fold performance.
- `submitted_synthetic_metrics.csv` — synthetic OU sanity-check results.

The submitted wealth curves were gross of realized transaction costs. Turnover affected PPO's reward but was not deducted from reported portfolio value.

New runs from `scripts/` are written under `outputs/`, which is git-ignored by default.
