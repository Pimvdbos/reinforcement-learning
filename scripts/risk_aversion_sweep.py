from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_portfolio.agent import build_ppo
from rl_portfolio.config import ExperimentConfig
from rl_portfolio.environment import PortfolioEnv
from rl_portfolio.evaluation import evaluate_wealth
from rl_portfolio.pipeline import chronological_split, load_real_market


def evaluate(model, features, returns, cfg, risk_aversion):
    env = PortfolioEnv(
        features,
        returns,
        initial_balance=cfg.initial_balance,
        risk_aversion=risk_aversion,
        transaction_cost=cfg.transaction_cost,
        cost_accounting=cfg.cost_accounting,
    )
    obs, _ = env.reset(seed=cfg.seed)
    done = False
    wealth = [cfg.initial_balance]
    weights = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        wealth.append(info["portfolio_value"])
        weights.append(info["weights"])
    metrics = evaluate_wealth(wealth, cfg.initial_balance)
    W = np.asarray(weights)
    metrics["mean_target_turnover"] = (
        float(np.abs(np.diff(W, axis=0)).sum(axis=1).mean()) if len(W) > 1 else 0.0
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk-aversion",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.1344558062397565, 0.25, 0.40, 0.60, 0.80],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1986])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "risk_aversion_sweep.csv")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    market_data = load_real_market(cfg)
    split = chronological_split(market_data.features, market_data.returns, cfg.train_fraction)

    rows = []
    for ra in args.risk_aversion:
        for seed in args.seeds:
            def make_env(ra=ra, seed=seed):
                env = PortfolioEnv(
                    split.train_scaled,
                    split.train_returns,
                    initial_balance=cfg.initial_balance,
                    risk_aversion=ra,
                    transaction_cost=cfg.transaction_cost,
                    cost_accounting=cfg.cost_accounting,
                )
                env.reset(seed=seed)
                return Monitor(env)

            model = build_ppo(DummyVecEnv([make_env]), seed=seed, verbose=0, device="auto")
            model.learn(total_timesteps=args.timesteps)
            metrics = evaluate(model, split.test_scaled, split.test_returns, cfg, ra)
            rows.append({"risk_aversion": ra, "seed": seed, **metrics})
            print(
                f"risk_aversion={ra:.5f} seed={seed} "
                f"return={metrics['total_return']:.2%} "
                f"sharpe={metrics['sharpe']:.2f}"
            )

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print("\nGrouped means:")
    print(
        out.groupby("risk_aversion")[
            ["total_return", "annualized_return", "sharpe", "max_drawdown", "mean_target_turnover"]
        ].mean().round(4)
    )


if __name__ == "__main__":
    main()
