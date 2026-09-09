"""Pure portfolio mechanics shared by the environment and tests."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


EPS = 1e-8


def normalize_action(action: np.ndarray, n_assets: int | None = None) -> np.ndarray:
    """Map a raw Box action to a long-only, fully invested weight vector.

    Stable-Baselines3 clips Box actions before passing them to the environment,
    but the environment also applies clipping defensively. A degenerate
    all-zero action falls back to equal weight so the full-investment
    constraint remains well defined.
    """
    x = np.asarray(action, dtype=float).reshape(-1)
    if n_assets is not None and len(x) != n_assets:
        raise ValueError(f"Expected {n_assets} action components, got {len(x)}")
    x = np.clip(x, 0.0, 1.0)
    total = float(x.sum())
    if total <= EPS:
        return np.repeat(1.0 / len(x), len(x))
    return x / total


def portfolio_drawdown(history: list[float] | np.ndarray) -> float:
    """Current drawdown relative to the running portfolio peak."""
    values = np.asarray(history, dtype=float)
    if values.size == 0:
        return 0.0
    peak = float(values.max())
    if peak <= 0:
        return 0.0
    return float(values[-1] / peak - 1.0)


@dataclass(frozen=True)
class StepResult:
    weights: np.ndarray
    gross_return: float
    turnover: float
    transaction_cost: float
    net_return: float
    reward: float
    next_value: float


def portfolio_step(
    action: np.ndarray,
    next_asset_returns: np.ndarray,
    previous_weights: np.ndarray,
    portfolio_value: float,
    *,
    risk_aversion: float,
    transaction_cost_rate: float,
    cost_accounting: str = "reward_only",
) -> StepResult:
    """Apply one target-allocation decision to the next-period asset returns.

    Parameters
    ----------
    cost_accounting:
        ``"reward_only"`` reproduces the submitted experiment: transaction
        costs regularize the RL reward but are not deducted from wealth.
        ``"net"`` deducts proportional turnover costs from both wealth and
        reward and is provided for follow-up experiments.
    """
    weights = normalize_action(action, len(previous_weights))
    returns = np.asarray(next_asset_returns, dtype=float)
    if returns.shape != weights.shape:
        raise ValueError("Return vector and weight vector must have identical shape")

    gross = float(weights @ returns)
    turnover = float(np.abs(weights - previous_weights).sum())
    cost = float(transaction_cost_rate * turnover)

    if cost_accounting == "reward_only":
        net = gross
        log_term = np.log1p(max(gross, -1.0 + EPS))
        reward = float(log_term - risk_aversion * gross**2 - cost)
    elif cost_accounting == "net":
        net = gross - cost
        log_term = np.log1p(max(net, -1.0 + EPS))
        reward = float(log_term - risk_aversion * net**2)
    else:
        raise ValueError("cost_accounting must be 'reward_only' or 'net'")

    next_value = float(portfolio_value * (1.0 + net))
    return StepResult(weights, gross, turnover, cost, net, reward, next_value)
