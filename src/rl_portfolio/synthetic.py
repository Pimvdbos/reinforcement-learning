"""Controlled Ornstein-Uhlenbeck market used as an RL sanity check."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "XOM": (0.05, 100.0, 0.015),
    "DAL": (0.05, 50.0, 0.020),
    "NFLX": (0.03, 300.0, 0.030),
    "MCD": (0.04, 200.0, 0.012),
    "CMG": (0.04, 150.0, 0.025),
    "BKNG": (0.03, 2000.0, 0.022),
    "DG": (0.05, 100.0, 0.013),
    "LULU": (0.04, 300.0, 0.028),
    "NVDA": (0.03, 400.0, 0.035),
    "INTC": (0.05, 50.0, 0.018),
    "TSLA": (0.02, 200.0, 0.040),
    "CVX": (0.05, 120.0, 0.015),
    "UAL": (0.05, 45.0, 0.022),
    "JPM": (0.04, 130.0, 0.016),
}


def generate_ou_prices(
    assets: Sequence[str],
    *,
    n_days: int = 3770,
    dt: float = 1 / 252,
    seed: int = 1986,
) -> pd.DataFrame:
    assets = list(assets)
    if any(a not in DEFAULT_PARAMS for a in assets):
        raise ValueError("Synthetic parameters are not defined for every requested asset")

    n = len(assets)
    corr = np.eye(n)
    # Reproduce the submitted experiment's pairwise -0.6 shock structure.
    for i in range(0, n - 1, 2):
        corr[i, i + 1] = corr[i + 1, i] = -0.6
    L = np.linalg.cholesky(corr)

    rng = np.random.default_rng(seed)
    prices = np.zeros((n_days, n), dtype=float)
    for j, a in enumerate(assets):
        prices[0, j] = DEFAULT_PARAMS[a][1]

    for t in range(1, n_days):
        shocks = L @ rng.normal(size=n)
        for j, a in enumerate(assets):
            theta, mu, sigma = DEFAULT_PARAMS[a]
            p = prices[t - 1, j]
            prices[t, j] = max(
                1.0,
                p + theta * (mu - p) * dt + sigma * p * np.sqrt(dt) * shocks[j],
            )

    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=assets)
