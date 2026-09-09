"""Experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


ASSETS = [
    "XOM", "DAL", "NFLX", "MCD", "CMG", "BKNG", "DG",
    "LULU", "NVDA", "INTC", "TSLA", "CVX", "UAL", "JPM",
]
MARKET_TICKER = "^GSPC"

# Fixed values used in the submitted experiment. The original Optuna notebook
# is no longer available, so the search process itself is not reproduced.
PPO_PARAMS = {
    "learning_rate": 8.27494264609064e-05,
    "n_steps": 512,
    "batch_size": 32,
    "gamma": 0.9896501306459291,
    "gae_lambda": 0.9501835963152316,
    "ent_coef": 0.11274745595800702,
}

OPTIMIZED_RISK_AVERSION = 0.1344558062397565
MVO_RISK_FREE_RATE = 0.02


@dataclass(frozen=True)
class ExperimentConfig:
    start_date: str = "2010-01-01"
    end_date: str = "2025-01-01"
    market_ticker: str = MARKET_TICKER
    assets: tuple[str, ...] = field(default_factory=lambda: tuple(ASSETS))
    initial_balance: float = 100_000.0
    risk_aversion: float = OPTIMIZED_RISK_AVERSION
    transaction_cost: float = 0.001
    cost_accounting: str = "reward_only"
    seed: int = 1986
    train_fraction: float = 0.80
    train_timesteps: int = 200_000
    n_cv_splits: int = 5
