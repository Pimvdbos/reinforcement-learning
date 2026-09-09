from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_portfolio.agent import EpisodeLoggerCallback, build_ppo
from rl_portfolio.benchmarks import (
    benchmark_metrics,
    mvo_weights,
    random_allocation_returns,
    target_weight_strategy_returns,
)
from rl_portfolio.config import ExperimentConfig
from rl_portfolio.environment import PortfolioEnv
from rl_portfolio.pipeline import load_real_market, chronological_split
from rl_portfolio.evaluation import evaluate_wealth, wealth_from_returns


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
    weights, turnover = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        wealth.append(info["portfolio_value"])
        weights.append(info["weights"])
        turnover.append(info["turnover"])
    return np.asarray(wealth), np.asarray(weights), np.asarray(turnover)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cost-accounting",
        choices=["reward_only", "net"],
        default="reward_only",
        help="reward_only reproduces the submitted experiment; net deducts turnover costs from wealth.",
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "holdout")
    args = parser.parse_args()

    cfg = ExperimentConfig(cost_accounting=args.cost_accounting, train_timesteps=args.timesteps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    market_data = load_real_market(cfg)
    prices = market_data.prices
    split = chronological_split(market_data.features, market_data.returns, cfg.train_fraction)
    train_features, test_features = split.train_features, split.test_features
    train_returns, test_returns = split.train_returns, split.test_returns
    train_scaled, test_scaled = split.train_scaled, split.test_scaled

    def make_train_env():
        return Monitor(
            PortfolioEnv(
                train_scaled,
                train_returns,
                initial_balance=cfg.initial_balance,
                risk_aversion=cfg.risk_aversion,
                transaction_cost=cfg.transaction_cost,
                cost_accounting=cfg.cost_accounting,
            )
        )

    vec_env = DummyVecEnv([make_train_env])
    model = build_ppo(vec_env, seed=cfg.seed, verbose=1, device="auto")
    callback = EpisodeLoggerCallback()
    model.learn(total_timesteps=cfg.train_timesteps, callback=callback)

    ppo_wealth, ppo_weights, ppo_turnover = evaluate_policy(model, test_scaled, test_returns, cfg)
    ppo_metrics = evaluate_wealth(ppo_wealth, cfg.initial_balance)

    # PortfolioEnv observes the first test state and realizes returns from the
    # following row onward. Benchmarks use the identical realized horizon.
    benchmark_returns = test_returns.iloc[1:]

    equal_w = np.repeat(1.0 / len(cfg.assets), len(cfg.assets))
    train_prices = prices.loc[train_returns.index, list(cfg.assets)]
    mvo_w = mvo_weights(train_prices)
    _, ra_w = random_allocation_returns(benchmark_returns, seed=cfg.seed)

    tc = cfg.transaction_cost if cfg.cost_accounting == "net" else 0.0
    ew_r = target_weight_strategy_returns(
        benchmark_returns, equal_w, transaction_cost_rate=tc, name="Equal Weight"
    )
    mvo_r = target_weight_strategy_returns(
        benchmark_returns, mvo_w, transaction_cost_rate=tc, name="MVO"
    )
    ra_r = target_weight_strategy_returns(
        benchmark_returns, ra_w, transaction_cost_rate=tc, name="Random Allocation"
    )
    ew_metrics = benchmark_metrics(ew_r, cfg.initial_balance)
    mvo_metrics = benchmark_metrics(mvo_r, cfg.initial_balance)
    ra_metrics = benchmark_metrics(ra_r, cfg.initial_balance)

    metrics = []
    for strategy, m in [
        ("PPO", ppo_metrics),
        ("Equal Weight", ew_metrics),
        ("MVO", mvo_metrics),
        ("Random Allocation", ra_metrics),
    ]:
        metrics.append({"strategy": strategy, **m})
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)

    pd.DataFrame(
        ppo_weights,
        index=test_returns.index[1 : 1 + len(ppo_weights)],
        columns=list(cfg.assets),
    ).to_csv(args.output_dir / "ppo_weights.csv")
    pd.Series(ppo_turnover, name="turnover").to_csv(args.output_dir / "ppo_turnover.csv", index=False)

    wealth = pd.DataFrame(index=benchmark_returns.index)
    wealth["PPO"] = pd.Series(ppo_wealth[1:], index=benchmark_returns.index[: len(ppo_wealth) - 1])
    wealth["Equal Weight"] = wealth_from_returns(ew_r, cfg.initial_balance)
    wealth["MVO"] = wealth_from_returns(mvo_r, cfg.initial_balance)
    wealth["Random Allocation"] = wealth_from_returns(ra_r, cfg.initial_balance)
    wealth.to_csv(args.output_dir / "wealth.csv")

    ax = wealth.plot(figsize=(12, 6), linewidth=1.5)
    ax.axhline(cfg.initial_balance, linestyle="--", alpha=0.5)
    ax.set_title("Out-of-sample portfolio value")
    ax.set_ylabel("Portfolio value")
    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(args.output_dir / "portfolio_value.png", dpi=160)
    plt.close()

    model.save(args.output_dir / "ppo_model")
    print(pd.DataFrame(metrics).set_index("strategy").round(4))


if __name__ == "__main__":
    main()
