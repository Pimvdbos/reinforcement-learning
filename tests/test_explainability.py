import numpy as np

from rl_portfolio.explainability import make_weight_predict_fn


class MockModel:
    def predict(self, state, deterministic=True):
        # Already in [0, 1], normalized by helper to [0.1, 0.2, 0.7].
        return np.array([0.1, 0.2, 0.7]), None


def test_target_asset_maps_to_explicit_asset_order():
    fn = make_weight_predict_fn(MockModel(), ["B", "A", "C"], "A")
    out = fn(np.zeros((2, 5)))
    assert np.allclose(out, 0.2)
