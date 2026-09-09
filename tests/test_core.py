import numpy as np

from rl_portfolio.core import normalize_action, portfolio_step


def test_normalize_action_enforces_long_only_simplex():
    w = normalize_action(np.array([-0.3, 0.2, 0.8]))
    assert np.all(w >= 0.0)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, [0.0, 0.2, 0.8])


def test_zero_action_falls_back_to_equal_weight():
    w = normalize_action(np.zeros(4))
    assert np.allclose(w, np.repeat(0.25, 4))


def test_reward_only_matches_submitted_wealth_accounting():
    result = portfolio_step(
        action=np.array([0.75, 0.25]),
        next_asset_returns=np.array([0.02, -0.01]),
        previous_weights=np.array([0.50, 0.50]),
        portfolio_value=100_000.0,
        risk_aversion=0.1344558062397565,
        transaction_cost_rate=0.001,
        cost_accounting="reward_only",
    )
    expected_gross = 0.75 * 0.02 + 0.25 * -0.01
    assert np.isclose(result.gross_return, expected_gross)
    assert np.isclose(result.turnover, 0.50)
    assert np.isclose(result.transaction_cost, 0.0005)
    # Submitted experiment: cost affects reward, not reported wealth.
    assert np.isclose(result.net_return, expected_gross)
    assert np.isclose(result.next_value, 100_000 * (1 + expected_gross))


def test_net_cost_mode_deducts_turnover_cost():
    result = portfolio_step(
        action=np.array([0.75, 0.25]),
        next_asset_returns=np.array([0.02, -0.01]),
        previous_weights=np.array([0.50, 0.50]),
        portfolio_value=100_000.0,
        risk_aversion=0.1,
        transaction_cost_rate=0.001,
        cost_accounting="net",
    )
    assert np.isclose(result.net_return, result.gross_return - result.transaction_cost)
