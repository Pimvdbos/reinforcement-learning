"""Portfolio evaluation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def evaluate_wealth(
    wealth: np.ndarray | list[float] | pd.Series,
    initial_balance: float | None = None,
) -> dict[str, float]:
    """Compute the metrics used in the submitted experiment."""
    values = np.asarray(wealth, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("wealth must contain at least two observations")
    start = float(values[0] if initial_balance is None else initial_balance)
    daily = np.diff(values) / values[:-1]
    total_return = float(values[-1] / start - 1.0)
    annualized_return = float((1.0 + total_return) ** (TRADING_DAYS / len(daily)) - 1.0)
    sharpe = float(daily.mean() / (daily.std(ddof=0) + 1e-8) * np.sqrt(TRADING_DAYS))
    peaks = np.maximum.accumulate(values)
    max_drawdown = float(np.min(values / peaks - 1.0))
    return {
        "final_value": float(values[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def wealth_from_returns(returns: pd.Series, initial_balance: float = 100_000.0) -> pd.Series:
    return initial_balance * (1.0 + returns).cumprod()
