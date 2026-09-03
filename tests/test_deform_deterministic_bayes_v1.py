from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "audit_deform_deterministic_bayes_v1.py"
)


def _load_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "audit_deform_deterministic_bayes_v1", MODULE_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_deterministic_point_mean_ignores_covariance() -> None:
    module = _load_module()
    rng = np.random.default_rng(4)
    batch, horizon, nodes, features_count = 2, 3, 8, 5
    internal = nodes - 4
    features = rng.normal(size=(batch, horizon, internal, features_count))
    frames = np.broadcast_to(np.eye(3), (batch, 3, 3)).copy()
    baseline = rng.normal(size=(batch, horizon, nodes, 3))
    location = rng.normal(size=(internal, features_count))
    scale = rng.uniform(0.5, 2.0, size=(internal, features_count))
    coefficients = rng.normal(size=(internal, features_count + 1, 3))
    shrinkage = 0.25

    actual = module._apply_deterministic_point_mean(
        features,
        frames,
        baseline,
        feature_location=location,
        feature_scale=scale,
        coefficients=coefficients,
        shrinkage=shrinkage,
    )

    expected = baseline.copy()
    for node in range(internal):
        standardized = (features[:, :, node] - location[node]) / scale[node]
        design = np.concatenate(
            [np.ones((*standardized.shape[:2], 1)), standardized], axis=2
        )
        expected[:, :, node + 2] += shrinkage * (design @ coefficients[node])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(actual[:, :, :2], baseline[:, :, :2])
    np.testing.assert_array_equal(actual[:, :, -2:], baseline[:, :, -2:])


def test_case_and_horizon_metrics_use_complete_trajectories() -> None:
    module = _load_module()
    truth = np.zeros((2, 6, 5, 3), dtype=np.float64)
    prediction = np.zeros_like(truth)
    prediction[0] = 1.0
    prediction[1, :2] = 2.0

    cases = module._case_l1(prediction, truth)
    np.testing.assert_allclose(cases, np.asarray([1.0, 2.0 / 3.0]))
    horizon = module._horizon_l1(prediction, truth)
    assert horizon == {"early": 1.5, "middle": 0.5, "late": 0.5}
