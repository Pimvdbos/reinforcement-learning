"""Policy explainability helpers.

SHAP is intentionally separated from the training code because Kernel SHAP is
computationally expensive. The asset index is always resolved from the
environment's explicit asset order, preventing the label mismatch present in
the original notebook.
"""
from __future__ import annotations

import numpy as np

from .core import normalize_action


def make_weight_predict_fn(model, asset_names: list[str], target_asset: str):
    if target_asset not in asset_names:
        raise ValueError(f"Unknown target asset: {target_asset}")
    asset_idx = asset_names.index(target_asset)

    def predict_fn(X):
        X = np.atleast_2d(X)
        out = np.zeros(len(X), dtype=float)
        for i, state in enumerate(X):
            action, _ = model.predict(state, deterministic=True)
            out[i] = normalize_action(action, len(asset_names))[asset_idx]
        return out

    return predict_fn


def observation_feature_names(engineered_features: list[str], asset_names: list[str]) -> list[str]:
    return engineered_features + [f"weight_{a}" for a in asset_names] + ["cash_fraction"]
