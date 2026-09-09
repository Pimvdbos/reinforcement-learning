from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import shap
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_portfolio.config import ExperimentConfig
from rl_portfolio.environment import PortfolioEnv
from rl_portfolio.explainability import make_weight_predict_fn, observation_feature_names
from rl_portfolio.pipeline import chronological_split, load_real_market


def collect_states(model, features, returns, cfg):
    env = PortfolioEnv(
        features,
        returns,
        initial_balance=cfg.initial_balance,
        risk_aversion=cfg.risk_aversion,
        transaction_cost=cfg.transaction_cost,
        cost_accounting=cfg.cost_accounting,
    )
    obs, _ = env.reset(seed=cfg.seed)
    states = [obs.copy()]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if not done:
            states.append(obs.copy())
    return np.asarray(states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "holdout" / "ppo_model.zip")
    parser.add_argument("--asset", default="TSLA")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--global-samples", type=int, default=60)
    parser.add_argument("--kernel-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "explainability")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    market_data = load_real_market(cfg)
    split = chronological_split(market_data.features, market_data.returns, cfg.train_fraction)

    model = PPO.load(args.model)
    train_states = collect_states(model, split.train_scaled, split.train_returns, cfg)
    test_states = collect_states(model, split.test_scaled, split.test_returns, cfg)

    predict_weight = make_weight_predict_fn(model, list(cfg.assets), args.asset)
    feature_names = observation_feature_names(
        list(split.train_scaled.columns),
        list(cfg.assets),
    )

    background = shap.kmeans(train_states, 50)
    explainer = shap.KernelExplainer(predict_weight, background, link="identity")

    target_idx = args.time_index if args.time_index >= 0 else len(test_states) - 1
    target_state = test_states[target_idx]
    local_values = explainer.shap_values(
        target_state,
        nsamples=args.kernel_samples,
        l1_reg=False,
        silent=True,
    )

    explanation = shap.Explanation(
        values=local_values,
        base_values=explainer.expected_value,
        data=target_state,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP waterfall — PPO weight in {args.asset}")
    plt.tight_layout()
    plt.savefig(args.output_dir / f"{args.asset.lower()}_waterfall.png", dpi=170, bbox_inches="tight")
    plt.close()

    n = min(args.global_samples, len(test_states))
    idx = np.linspace(0, len(test_states) - 1, n).astype(int)
    matrix = explainer.shap_values(
        test_states[idx],
        nsamples=args.kernel_samples,
        l1_reg=False,
        silent=True,
    )
    mean_abs = np.abs(matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [feature_names[i] for i in order][::-1]
    values = mean_abs[order][::-1]
    ax.barh(labels, values)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global feature importance — PPO weight in {args.asset}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / f"{args.asset.lower()}_global_shap.png", dpi=170, bbox_inches="tight")
    plt.close()

    pred = float(predict_weight(target_state.reshape(1, -1))[0])
    print(f"Explained {args.asset} allocation at test state {target_idx}: {pred:.4f}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
