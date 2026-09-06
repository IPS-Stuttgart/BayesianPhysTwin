"""Focused numerical and information-boundary tests for the GP pilot."""

import unittest

import numpy as np

from bayesian_phystwin_experiments.deform_dlo_local_residual import deform_causal_inputs
from experiments.deform_gp_residual_dev_v1.run import (
    basis,
    fit_gp_basis,
    kernel,
    predict_gp,
    solve_gp,
    transform,
    world_prediction,
)


class GaussianProcessTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(41)
        self.x = self.rng.normal(size=(24, 9))
        self.y = self.rng.normal(size=(24, 3))

    def test_kernel_psd(self):
        for material in (False, True):
            gram = kernel(self.x, self.x, 0.5, material)
            np.testing.assert_allclose(gram, gram.T, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(gram).min(), -1e-10)
            np.testing.assert_allclose(np.diag(gram), 1, atol=1e-12)

    def test_gp_kernel_ridge_mean_identity(self):
        model = solve_gp(fit_gp_basis(self.x, self.y, 12, 0.5, False), 0.2)
        phi = basis(transform(self.x, model), model)
        gram = phi @ phi.T
        expected = gram @ np.linalg.solve(gram + 0.2 * np.eye(len(self.x)), self.y)
        actual, variance = predict_gp(self.x, model, True)
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        self.assertTrue(np.isfinite(variance).all())
        self.assertTrue((variance > 0).all())

    def test_all_rows_enter_likelihood(self):
        first = fit_gp_basis(self.x, self.y, 6, 0.5, False)
        changed = self.y.copy()
        unselected = sorted(set(range(len(self.x))) - set(first["inducing_indices"]))[0]
        changed[unselected] += 10
        second = fit_gp_basis(self.x, changed, 6, 0.5, False)
        np.testing.assert_array_equal(
            first["inducing_indices"], second["inducing_indices"]
        )
        a, _ = predict_gp(self.x, solve_gp(first, 0.2))
        b, _ = predict_gp(self.x, solve_gp(second, 0.2))
        self.assertGreater(float(np.linalg.norm(a - b)), 1e-4)

    def test_query_does_not_change_normalizer(self):
        model = fit_gp_basis(self.x, self.y, 6, 1.5, True)
        before = model["location"].copy()
        predict_gp(self.x * 100, solve_gp(model, 0.5))
        np.testing.assert_array_equal(before, model["location"])

    def test_clamped_exact_and_frame_rotation(self):
        baseline = np.zeros((2, 5, 8, 3))
        local = np.ones((2, 5, 4, 3))
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        frames = np.stack((np.eye(3), rotation))
        result = world_prediction(baseline, local, frames, 0.5)
        np.testing.assert_array_equal(result[:, :, [0, 1, -2, -1]], 0)
        np.testing.assert_allclose(result[1, 0, 2], [-0.5, 0.5, 0.5])
        np.testing.assert_array_equal(
            world_prediction(baseline, local, frames, 0), baseline
        )

    def test_future_free_node_truth_not_in_causal_inputs(self):
        trajectory = self.rng.normal(size=(2, 7, 8, 3))
        initial, action = deform_causal_inputs(trajectory)
        trajectory[:, 2:, 2:-2] += 999
        other_initial, other_action = deform_causal_inputs(trajectory)
        np.testing.assert_array_equal(initial, other_initial)
        np.testing.assert_array_equal(action, other_action)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            fit_gp_basis(self.x * np.nan, self.y, 6, 0.5, False)
        with self.assertRaises(ValueError):
            kernel(self.x, self.x, -1)
        with self.assertRaises(ValueError):
            solve_gp(fit_gp_basis(self.x, self.y, 6, 0.5, False), 0)


if __name__ == "__main__":
    unittest.main()
