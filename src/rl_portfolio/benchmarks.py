"""Reference allocation policies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import evaluate_wealth, wealth_from_returns


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    w = np.repeat(1.0 / returns.shape[1], returns.shape[1])
    return pd.Series(returns.to_numpy() @ w, index=returns.index, name="Equal Weight")


def random_allocation_returns(returns: pd.DataFrame, *, seed: int = 1986) -> tuple[pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(returns.shape[1]), size=len(returns))
    r = np.einsum("ij,ij->i", returns.to_numpy(), weights)
    return pd.Series(r, index=returns.index, name="Random Allocation"), weights


def mvo_weights(train_prices: pd.DataFrame, *, risk_free_rate: float = 0.02) -> np.ndarray:
    """Long-only maximum-Sharpe MVO using the same library as the notebook."""
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    mu = expected_returns.mean_historical_return(train_prices, frequency=252)
    cov = risk_models.sample_cov(train_prices, frequency=252)
    ef = EfficientFrontier(mu, cov)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned = ef.clean_weights()
    # Preserve the exact train_prices column order.
    return np.asarray([cleaned[c] for c in train_prices.columns], dtype=float)


def fixed_weight_returns(returns: pd.DataFrame, weights: np.ndarray, name: str = "MVO") -> pd.Series:
    weights = np.asarray(weights, dtype=float)
    if len(weights) != returns.shape[1]:
        raise ValueError("weights do not match return columns")
    return pd.Series(returns.to_numpy() @ weights, index=returns.index, name=name)


def target_weight_strategy_returns(
    returns: pd.DataFrame,
    target_weights: np.ndarray,
    *,
    transaction_cost_rate: float = 0.0,
    initial_weights: np.ndarray | None = None,
    name: str = "Strategy",
) -> pd.Series:
    """Simulate daily target weights with the same turnover convention as PPO.

    Turnover is measured as the L1 change between consecutive target-weight
    vectors. This mirrors the submitted environment's transaction-cost proxy;
    it does not model post-return weight drift within the day.
    """
    R = returns.to_numpy(dtype=float)
    W = np.asarray(target_weights, dtype=float)
    if W.ndim == 1:
        if len(W) != returns.shape[1]:
            raise ValueError("weights do not match return columns")
        W = np.tile(W, (len(returns), 1))
    if W.shape != R.shape:
        raise ValueError("target_weights must be a vector or match returns.shape")

    if initial_weights is None:
        prev = np.repeat(1.0 / returns.shape[1], returns.shape[1])
    else:
        prev = np.asarray(initial_weights, dtype=float)

    out = np.empty(len(returns), dtype=float)
    for i, (r, w) in enumerate(zip(R, W)):
        turnover = float(np.abs(w - prev).sum())
        out[i] = float(w @ r) - transaction_cost_rate * turnover
        prev = w
    return pd.Series(out, index=returns.index, name=name)


def benchmark_metrics(return_series: pd.Series, initial_balance: float = 100_000.0) -> dict[str, float]:
    wealth = wealth_from_returns(return_series, initial_balance)
    return evaluate_wealth(np.r_[initial_balance, wealth.to_numpy()], initial_balance)
