from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_portfolio.agent import build_ppo
from rl_portfolio.benchmarks import (
    benchmark_metrics,
    mvo_weights,
    random_allocation_returns,
    target_weight_strategy_returns,
)
from rl_portfolio.config import ExperimentConfig
from rl_portfolio.environment import PortfolioEnv
from rl_portfolio.pipeline import load_real_market, scale_feature_pair
from rl_portfolio.evaluation import evaluate_wealth


def evaluate_policy(model, features, returns, cfg):
    env = PortfolioEnv(
        features,
        returns,
        initial_balance=cfg.initial_balance,
        risk_aversion=cfg.risk_aversion,
        transaction_cost=cfg.transaction_cost,
        cost_accounting=cfg.cost_accounting,
    )
    obs, _ = env.reset(seed=cfg.seed)
    done = False
    wealth = [cfg.initial_balance]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        wealth.append(info["portfolio_value"])
    return evaluate_wealth(wealth, cfg.initial_balance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument(
        "--cost-accounting",
        choices=["reward_only", "net"],
        default="reward_only",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cross_validation.csv")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        train_timesteps=args.timesteps,
        n_cv_splits=args.splits,
        cost_accounting=args.cost_accounting,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    market_data = load_real_market(cfg)
    prices, features, returns = market_data.prices, market_data.features, market_data.returns

    rows = []
    splitter = TimeSeriesSplit(n_splits=cfg.n_cv_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features), start=1):
        train_f, test_f = features.iloc[train_idx], features.iloc[test_idx]
        train_r, test_r = returns.iloc[train_idx], returns.iloc[test_idx]

        train_scaled, test_scaled, _ = scale_feature_pair(train_f, test_f)

        def make_train_env():
            return Monitor(
                PortfolioEnv(
                    train_scaled,
                    train_r,
                    initial_balance=cfg.initial_balance,
                    risk_aversion=cfg.risk_aversion,
                    transaction_cost=cfg.transaction_cost,
                    cost_accounting=cfg.cost_accounting,
                )
            )

        model = build_ppo(DummyVecEnv([make_train_env]), seed=cfg.seed, verbose=0, device="auto")
        model.learn(total_timesteps=cfg.train_timesteps)
        rows.append({"fold": fold, "strategy": "PPO", **evaluate_policy(model, test_scaled, test_r, cfg)})

        benchmark_r = test_r.iloc[1:]
        equal_w = np.repeat(1.0 / len(cfg.assets), len(cfg.assets))
        train_prices = prices.loc[train_r.index, list(cfg.assets)]
        w = mvo_weights(train_prices)
        _, ra_w = random_allocation_returns(benchmark_r, seed=cfg.seed + fold)
        tc = cfg.transaction_cost if cfg.cost_accounting == "net" else 0.0

        rows.append({
            "fold": fold,
            "strategy": "Equal Weight",
            **benchmark_metrics(target_weight_strategy_returns(
                benchmark_r, equal_w, transaction_cost_rate=tc, name="Equal Weight"
            )),
        })
        rows.append({
            "fold": fold,
            "strategy": "MVO",
            **benchmark_metrics(target_weight_strategy_returns(
                benchmark_r, w, transaction_cost_rate=tc, name="MVO"
            )),
        })
        rows.append({
            "fold": fold,
            "strategy": "Random Allocation",
            **benchmark_metrics(target_weight_strategy_returns(
                benchmark_r, ra_w, transaction_cost_rate=tc, name="Random Allocation"
            )),
        })
        print(f"Completed fold {fold}/{cfg.n_cv_splits}")

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print("\nMean across folds:")
    print(
        out.groupby("strategy")[["total_return", "annualized_return", "sharpe", "max_drawdown"]]
        .mean()
        .round(4)
    )


if __name__ == "__main__":
    main()
