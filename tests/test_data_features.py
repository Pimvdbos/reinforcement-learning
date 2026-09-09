import numpy as np
import pandas as pd

from rl_portfolio.data import align_asset_columns
from rl_portfolio.features import build_features


def test_asset_order_is_explicit_not_source_dependent():
    prices = pd.DataFrame({"B": [1, 2], "A": [3, 4]})
    aligned = align_asset_columns(prices, ["A", "B"])
    assert list(aligned.columns) == ["A", "B"]


def test_feature_count_and_current_return_alignment():
    assets = [f"A{i}" for i in range(14)]
    idx = pd.bdate_range("2020-01-01", periods=80)
    base = np.arange(80, dtype=float)
    prices = pd.DataFrame(
        {a: 100 + (j + 1) * 0.05 * base + 0.001 * base**2 for j, a in enumerate(assets)},
        index=idx,
    )
    market = pd.Series(3000 + 0.5 * base + 0.002 * base**2, index=idx)
    features, returns = build_features(prices, market, assets)

    assert features.shape[1] == 101
    assert features.index.equals(returns.index)

    t = features.index[-1]
    prev = prices.index[prices.index.get_loc(t) - 1]
    expected = np.log(prices.loc[t, assets[0]] / prices.loc[prev, assets[0]])
    assert np.isclose(features.loc[t, f"{assets[0]}_log_return_1d"], expected)
