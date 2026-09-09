from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_portfolio.agent import build_ppo
from rl_portfolio.config import ExperimentConfig
from rl_portfolio.environment import PortfolioEnv
from rl_portfolio.evaluation import evaluate_wealth
from rl_portfolio.features import build_synthetic_features
from rl_portfolio.synthetic import generate_ou_prices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "synthetic_metrics.csv")
    args = parser.parse_args()

    cfg = ExperimentConfig(train_timesteps=args.timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    prices = generate_ou_prices(cfg.assets, seed=cfg.seed)
    features, returns = build_synthetic_features(prices, cfg.assets)
    split = int(len(features) * cfg.train_fraction)

    train_f, test_f = features.iloc[:split], features.iloc[split:]
    train_r, test_r = returns.iloc[:split], returns.iloc[split:]
    scaler = RobustScaler()
    train_s = pd.DataFrame(scaler.fit_transform(train_f), index=train_f.index, columns=train_f.columns)
    test_s = pd.DataFrame(scaler.transform(test_f), index=test_f.index, columns=test_f.columns)

    def make_env():
        return Monitor(
            PortfolioEnv(
                train_s,
                train_r,
                initial_balance=cfg.initial_balance,
                risk_aversion=cfg.risk_aversion,
                transaction_cost=cfg.transaction_cost,
                cost_accounting=cfg.cost_accounting,
            )
        )

    model = build_ppo(DummyVecEnv([make_env]), seed=cfg.seed, verbose=1)
    model.learn(total_timesteps=cfg.train_timesteps)

    env = PortfolioEnv(
        test_s,
        test_r,
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

    metrics = evaluate_wealth(wealth, cfg.initial_balance)
    pd.DataFrame([{"strategy": "PPO", **metrics}]).to_csv(args.output, index=False)
    print(pd.Series(metrics).round(4))


if __name__ == "__main__":
    main()
