import numpy as np

from rl_portfolio.evaluation import evaluate_wealth


def test_evaluate_wealth_basic_properties():
    wealth = np.array([100.0, 101.0, 100.0, 110.0])
    metrics = evaluate_wealth(wealth)
    assert np.isclose(metrics["total_return"], 0.10)
    assert metrics["max_drawdown"] < 0
    assert np.isfinite(metrics["sharpe"])
