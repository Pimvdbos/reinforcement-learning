import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from rl_portfolio.environment import PortfolioEnv


def test_environment_uses_next_row_return_after_current_state():
    idx = pd.date_range("2024-01-01", periods=3)
    features = pd.DataFrame(
        {
            "signal": [10.0, 20.0, 30.0],
            "drawdown": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    returns = pd.DataFrame(
        {
            "A": [0.90, 0.10, -0.20],
            "B": [0.90, 0.00, 0.00],
        },
        index=idx,
    )

    env = PortfolioEnv(features, returns, initial_balance=100.0, risk_aversion=0.0, transaction_cost=0.0)
    obs, _ = env.reset()
    assert obs[0] == 10.0

    # Allocate fully to A. The first realized return must be returns.iloc[1]
    # (= +10%), not returns.iloc[0] (= +90%).
    obs, _, done, _, info = env.step(np.array([1.0, 0.0]))
    assert np.isclose(info["portfolio_value"], 110.0)
    assert not done


def test_default_observation_dimension_formula():
    idx = pd.date_range("2024-01-01", periods=3)
    f = pd.DataFrame(np.zeros((3, 101)), index=idx, columns=[f"f{i}" for i in range(101)])
    f = f.rename(columns={"f100": "drawdown"})
    r = pd.DataFrame(np.zeros((3, 14)), index=idx, columns=[f"A{i}" for i in range(14)])
    env = PortfolioEnv(f, r)
    obs, _ = env.reset()
    assert obs.shape == (116,)
