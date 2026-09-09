"""Reusable chronological data-preparation helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import ExperimentConfig
from .data import download_yahoo_prices
from .features import build_features


@dataclass
class RealMarketData:
    prices: pd.DataFrame
    market: pd.Series
    features: pd.DataFrame
    returns: pd.DataFrame


@dataclass
class ChronologicalSplit:
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    train_returns: pd.DataFrame
    test_returns: pd.DataFrame
    train_scaled: pd.DataFrame
    test_scaled: pd.DataFrame
    scaler: RobustScaler


def load_real_market(cfg: ExperimentConfig) -> RealMarketData:
    prices, market = download_yahoo_prices(
        cfg.assets, cfg.market_ticker, cfg.start_date, cfg.end_date
    )
    features, returns = build_features(prices, market, cfg.assets)
    return RealMarketData(prices, market, features, returns)


def scale_feature_pair(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    scaler = RobustScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        index=train_features.index,
        columns=train_features.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_features),
        index=test_features.index,
        columns=test_features.columns,
    )
    return train_scaled, test_scaled, scaler


def chronological_split(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    train_fraction: float,
) -> ChronologicalSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    if not features.index.equals(returns.index):
        raise ValueError("features and returns must share the same index")

    split = int(len(features) * train_fraction)
    train_f, test_f = features.iloc[:split], features.iloc[split:]
    train_r, test_r = returns.iloc[:split], returns.iloc[split:]

    train_s, test_s, scaler = scale_feature_pair(train_f, test_f)
    return ChronologicalSplit(train_f, test_f, train_r, test_r, train_s, test_s, scaler)
