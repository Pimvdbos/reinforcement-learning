"""Gymnasium environment for dynamic portfolio allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .core import portfolio_drawdown, portfolio_step


class PortfolioEnv(gym.Env):
    """Daily long-only target-weight portfolio environment.

    Observation
    -----------
    101 engineered market features + 14 previous asset weights + one
    cash-fraction compatibility feature = 116 dimensions for the default
    14-asset universe.

    Action
    ------
    One non-negative component per asset. Components are normalized to a
    simplex before the t+1 return is applied.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        *,
        initial_balance: float = 100_000.0,
        risk_aversion: float = 0.1344558062397565,
        transaction_cost: float = 0.001,
        cost_accounting: str = "reward_only",
    ):
        super().__init__()
        if not features.index.equals(returns.index):
            raise ValueError("features and returns must have identical indexes")
        if "drawdown" not in features.columns:
            raise ValueError("features must contain a drawdown column")
        if len(features) < 2:
            raise ValueError("environment requires at least two observations")

        self.feature_columns = list(features.columns)
        self.asset_names = list(returns.columns)
        self.features = features.to_numpy(dtype=float)
        self.returns = returns.to_numpy(dtype=float)
        self.n_features = self.features.shape[1]
        self.n_assets = self.returns.shape[1]
        self.drawdown_idx = self.feature_columns.index("drawdown")
        self.initial_balance = float(initial_balance)
        self.risk_aversion = float(risk_aversion)
        self.transaction_cost = float(transaction_cost)
        self.cost_accounting = cost_accounting

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + self.n_assets + 1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_history: list[float] = []
        self.prev_weights = np.repeat(1.0 / self.n_assets, self.n_assets)

    def _get_obs(self) -> np.ndarray:
        features = self.features[self.current_step].copy()
        features[self.drawdown_idx] = portfolio_drawdown(self.portfolio_history)
        cash_fraction = max(0.0, 1.0 - float(self.prev_weights.sum()))
        return np.concatenate([features, self.prev_weights, [cash_fraction]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        self.prev_weights = np.repeat(1.0 / self.n_assets, self.n_assets)
        return self._get_obs(), {}

    def step(self, action):
        next_step = self.current_step + 1
        result = portfolio_step(
            action,
            self.returns[next_step],
            self.prev_weights,
            self.portfolio_value,
            risk_aversion=self.risk_aversion,
            transaction_cost_rate=self.transaction_cost,
            cost_accounting=self.cost_accounting,
        )
        self.portfolio_value = result.next_value
        self.portfolio_history.append(self.portfolio_value)
        self.prev_weights = result.weights
        self.current_step = next_step

        terminated = self.current_step >= len(self.features) - 1
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": result.weights.copy(),
            "gross_return": result.gross_return,
            "net_return": result.net_return,
            "turnover": result.turnover,
            "transaction_cost": result.transaction_cost,
        }
        return self._get_obs(), result.reward, terminated, False, info
