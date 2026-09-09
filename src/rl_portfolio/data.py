"""Market-data download and alignment utilities."""
from __future__ import annotations

from collections.abc import Sequence
import pandas as pd


def align_asset_columns(prices: pd.DataFrame, assets: Sequence[str]) -> pd.DataFrame:
    """Return prices in one explicit asset order and fail on missing assets."""
    assets = list(assets)
    missing = [a for a in assets if a not in prices.columns]
    if missing:
        raise ValueError(f"Missing assets: {missing}")
    return prices.loc[:, assets].copy()


def download_yahoo_prices(
    assets: Sequence[str],
    market_ticker: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download adjusted closes from Yahoo Finance via yfinance.

    The explicit reindex is important: yfinance can return multi-ticker columns
    in an order different from the input list. The original notebook did not
    enforce this ordering, which made later allocation labels ambiguous.
    """
    import yfinance as yf

    raw = yf.download(list(assets), start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    prices = prices.ffill().dropna(how="any")
    prices = align_asset_columns(prices, assets)

    market_raw = yf.download(market_ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(market_raw.columns, pd.MultiIndex):
        market = market_raw["Close"].iloc[:, 0]
    elif "Close" in market_raw:
        market = market_raw["Close"]
    else:
        market = market_raw.squeeze()
    market = market.rename(market_ticker).ffill().dropna()

    common = prices.index.intersection(market.index)
    return prices.loc[common], market.loc[common]
