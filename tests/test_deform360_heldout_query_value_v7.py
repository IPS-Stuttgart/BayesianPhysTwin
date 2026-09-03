from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "run_deform360_heldout_query_value_v7.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "deform360_heldout_query_value_v7_test_target", RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HeldOutQueryValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_runner()

    def test_query_bank_and_split_are_deterministic(self) -> None:
        module = self.module
        dimension = 4 * module.FIELD_DIMENSION_PER_SENSOR
        first = module.build_query_bank(dimension, 64)
        second = module.build_query_bank(dimension, 64)
        self.assertEqual(list(first), list(second))
        calibration, evaluation = module.split_query_names(first, 32, 260903)
        self.assertEqual(len(calibration), 32)
        self.assertEqual(len(evaluation), 32)
        self.assertFalse(set(calibration) & set(evaluation))
        self.assertEqual(set(calibration) | set(evaluation), set(first))
        for name in first:
            np.testing.assert_array_equal(first[name].weight, second[name].weight)
            self.assertTrue(
                math.isclose(
                    float(np.sum(np.abs(first[name].weight))),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )

    def test_controls_preserve_registered_marginals(self) -> None:
        module = self.module
        rng = np.random.default_rng(11)
        dimension = 4 * module.FIELD_DIMENSION_PER_SENSOR
        covariance = module._DummyCovarianceModel(
            mean_error=np.zeros(dimension),
            diagonal=np.full(dimension, 0.04),
            factor=rng.normal(scale=0.01, size=(dimension, 5)),
            multiplier=1.3,
            marginal_z=1.64,
            source_marginal_coverage=0.9,
            source_joint_nanees=1.0,
        )
        models = module.build_covariance_models(
            module._DummyBase, covariance, seed=17
        )
        reference = module.marginal_variance(models["full_low_rank"])
        for arm in module.SHARED_ARMS:
            np.testing.assert_allclose(
                module.marginal_variance(models[arm]),
                reference,
                rtol=1e-12,
                atol=1e-12,
            )
        self.assertTrue(
            np.all(
                module.marginal_variance(models["local_diagonal_only"])
                <= reference
            )
        )

    def test_source_only_width_match_is_exact(self) -> None:
        module = self.module
        rng = np.random.default_rng(23)
        dimension = 4 * module.FIELD_DIMENSION_PER_SENSOR
        bank = module.build_query_bank(dimension, 64)
        calibration, _ = module.split_query_names(bank, 32, 260903)
        covariance = module._DummyCovarianceModel(
            mean_error=np.zeros(dimension),
            diagonal=np.full(dimension, 0.02),
            factor=rng.normal(scale=0.015, size=(dimension, 6)),
            multiplier=1.0,
            marginal_z=1.64,
            source_marginal_coverage=0.9,
            source_joint_nanees=1.0,
        )
        models = module.build_covariance_models(
            module._DummyBase, covariance, seed=29
        )
        source_errors = rng.normal(scale=0.2, size=(40, dimension))
        calibrations, width_error = module.build_calibrations(
            source_errors,
            bank,
            calibration,
            models,
            probability=0.9,
            minimum_scale=1e-8,
        )
        self.assertLessEqual(width_error, 1e-12)
        full = calibrations["arm_calibrated"]["full_low_rank"]
        matched = calibrations["arm_calibrated"]["diagonal_width_matched"]
        self.assertTrue(
            math.isclose(
                full["mean_calibration_query_width"],
                matched["mean_calibration_query_width"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
