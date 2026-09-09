# Reproducibility Notes

## What is directly archived

The CSV files in `results/` preserve the numerical outputs printed by the final experiment. The SVG figures in `figures/` are regenerated directly from those archived metrics, so the core findings can be inspected without requiring a multi-hour PPO retraining job.

## What is reproducible from code

The public code contains:

- Yahoo Finance data acquisition;
- explicit asset ordering;
- the complete 101-feature engineering pipeline;
- the 116-dimensional portfolio environment;
- PPO construction with the fixed final hyperparameters;
- the synthetic OU market;
- equal-weight, MVO, and random-allocation policies;
- chronological holdout evaluation;
- five-fold expanding-window validation;
- performance metrics;
- portfolio-cost accounting modes;
- explainability helpers.

A user with internet access can rerun the real-data scripts from raw market data.

## Why exact numerical equality is not guaranteed

Deep RL experiments are sensitive to library versions, hardware, random-number implementations, market-data revisions, and stochastic optimization. Yahoo's adjusted historical data may also be revised after corporate actions.

For that reason the repository distinguishes between:

1. **submitted outputs** — the exact saved numerical results from the original experiment; and
2. **refactored code** — the audited implementation intended for clean reruns and extensions.

## Hyperparameter tuning

The final hyperparameter values used in the submitted project are preserved in `src/rl_portfolio/config.py`.

The separate Optuna tuning notebook used during the course project is no longer available. The repository therefore does **not** claim to reproduce the historical search trials, objective values, or parameter-importance analysis. The final configuration is treated as fixed input.

This is also why the original risk-aversion sweep is not promoted as a core result. It was executed in a stateful notebook after cross-validation and depended on variables left by earlier cells.

## Transaction costs

The submitted plots and result tables use gross wealth. The turnover penalty was part of PPO's reward but was not deducted from portfolio value.

`PortfolioEnv` exposes two modes:

- `reward_only` — reproduces that submitted design;
- `net` — deducts the proportional turnover-cost proxy from wealth.

When `net` is selected in `scripts/train_holdout.py`, the same target-weight turnover convention is also applied to the reference strategies.

## Asset ordering

The original notebook downloaded multiple tickers from Yahoo Finance and later labelled action components using the manually specified ticker list. Yahoo returned the price columns in a different order in the saved run.

The public refactor fixes this at the data boundary:

```python
prices = prices.loc[:, configured_assets]
```

and propagates the same order through returns, the environment, saved weights, and SHAP targets.

As a consequence, newly trained policies from the refactored implementation should be treated as new experimental runs rather than bit-for-bit reproductions of the saved model.

## Computational cost

The default 200,000-timestep holdout training is moderately expensive on CPU. Five-fold validation trains five independent PPO agents and is substantially more expensive.

For code verification:

```bash
python scripts/train_holdout.py --timesteps 10000
```

For the experiment budget:

```bash
python scripts/train_holdout.py --timesteps 200000
python scripts/cross_validate.py --timesteps 200000
```

## MVO convention

The refactor makes the original PyPortfolioOpt convention explicit: MVO uses a 2% annual risk-free rate inside the maximum-Sharpe optimization. The reported evaluation Sharpe metric itself uses a zero risk-free rate, as in the submitted notebook. This distinction is preserved rather than hidden behind a library default.
