"""Feature engineering for the portfolio MDP."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd


def _rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100.0 - 100.0 / (1.0 + rs)


def build_features(
    prices: pd.DataFrame,
    market: pd.Series,
    assets: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the 101 engineered features and aligned one-day simple returns.

    All feature values at date t depend only on prices observed at or before t.
    The environment then applies the return at t+1 after an action is selected.
    """
    assets = list(assets)
    prices = prices.loc[:, assets]
    simple_returns = prices.pct_change()

    feature_map: dict[str, pd.Series | float] = {}
    for ticker in assets:
        r = simple_returns[ticker]
        p = prices[ticker]
        feature_map[f"{ticker}_log_return_1d"] = np.log1p(r)
        feature_map[f"{ticker}_log_return_5d"] = np.log(p / p.shift(5))
        feature_map[f"{ticker}_rolling_vol_20d"] = r.rolling(20).std()
        feature_map[f"{ticker}_momentum_20d"] = p.pct_change(20)
        feature_map[f"{ticker}_rsi_14"] = _rsi(p, 14)

        mid = p.rolling(20).mean()
        std = p.rolling(20).std()
        upper = mid + 2.0 * std
        lower = mid - 2.0 * std
        feature_map[f"{ticker}_bb_width"] = (upper - lower) / (mid + 1e-8)
        feature_map[f"{ticker}_bb_position"] = (p - lower) / (upper - lower + 1e-8)

    market = market.reindex(prices.index).ffill()
    market_log_return = np.log1p(market.pct_change())
    feature_map["market_return"] = market_log_return
    feature_map["market_volatility"] = market_log_return.rolling(20).std()

    f = pd.DataFrame(feature_map, index=prices.index)

    # Dynamic value is injected by PortfolioEnv._get_obs.
    f["drawdown"] = 0.0

    f = f.dropna()
    returns = simple_returns.loc[f.index, assets]
    if f.shape[1] != 101:
        raise AssertionError(f"Expected 101 engineered features, got {f.shape[1]}")
    return f, returns


def build_synthetic_features(
    prices: pd.DataFrame,
    assets: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same asset features to synthetic data.

    The cross-sectional average return is used as a synthetic market proxy,
    matching the design of the submitted notebook.
    """
    assets = list(assets)
    simple_returns = prices.loc[:, assets].pct_change()
    market_proxy = (1.0 + simple_returns.mean(axis=1).fillna(0.0)).cumprod()
    return build_features(prices.loc[:, assets], market_proxy, assets)
