"""Numerical/causal checks for issue #946; synthetic tests are not real evidence."""

import unittest

import numpy as np

from bayesian_phystwin_experiments.deform_gp_residual_v1 import (
    add_residual,
    balanced_anchor_indices,
    fit_gp_bank,
    matern_material_kernel,
)


class GPTests(unittest.TestCase):
    def test_kernel_psd_and_material_dependence(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(20, 4))
        a = np.linspace(-1, 1, 20)
        k = matern_material_kernel(x, x, 2, a, a)
        np.testing.assert_allclose(k, k.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(k).min(), 0)
        np.testing.assert_allclose(np.diag(k), 1)
        self.assertFalse(np.allclose(k, matern_material_kernel(x, x, 2)))

    def test_gp_matches_finite_kernel_ridge(self):
        x = np.linspace(-1, 1, 15)[:, None]
        y = np.sin(x * 2)
        gp = fit_gp_bank(
            x,
            y,
            arc=None,
            anchor_indices=np.array([0, 4, 8, 14]),
            length_multiplier=1,
            noises=[0.1],
            row_weight=1,
        )[0.1]
        kxu = matern_material_kernel(
            (x - gp.location) / gp.scale, gp.anchors, gp.length
        )
        phi = np.linalg.solve(gp.factor, kxu.T).T
        kernel = phi @ phi.T
        expected = kernel @ np.linalg.solve(kernel + 0.1 * np.eye(len(x)), y)
        np.testing.assert_allclose(gp.predict(x), expected, rtol=1e-9, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(gp.weight_covariance).min(), 0)

    def test_query_batch_does_not_change_fit(self):
        x = np.linspace(-1, 1, 20)[:, None]
        gp = fit_gp_bank(
            x,
            x**2,
            arc=None,
            anchor_indices=np.arange(0, 20, 2),
            length_multiplier=1,
            noises=[0.1],
            row_weight=0.1,
        )[0.1]
        query = np.array([[0.2], [0.5]])
        alone = gp.predict(query)
        extended = gp.predict(np.concatenate((query, np.array([[1000.0]]))))[:2]
        np.testing.assert_allclose(alone, extended)

    def test_boundary_and_exact_fallback(self):
        baseline = np.arange(2 * 5 * 7 * 3, dtype=float).reshape(2, 5, 7, 3)
        correction = np.ones((2, 5, 3, 3))
        frames = np.repeat(np.eye(3)[None], 2, axis=0)
        np.testing.assert_array_equal(
            add_residual(baseline, correction, frames, 0), baseline
        )
        changed = add_residual(baseline, correction, frames, 0.5)
        np.testing.assert_array_equal(
            changed[:, :, [0, 1, -2, -1]], baseline[:, :, [0, 1, -2, -1]]
        )
        np.testing.assert_allclose(changed[:, :, 2:-2], baseline[:, :, 2:-2] + 0.5)

    def test_balanced_anchors(self):
        indices = balanced_anchor_indices(40, 498, 9, 240)
        self.assertEqual(len(set(indices.tolist())), 240)
        np.testing.assert_array_equal(
            np.bincount(indices // (498 * 9)), np.repeat(6, 40)
        )

    def test_bad_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            matern_material_kernel(np.ones((2, 3)), np.ones((2, 3)), 0)
        with self.assertRaises(ValueError):
            matern_material_kernel(np.array([[np.nan]]), np.ones((2, 1)), 1)
        with self.assertRaises(ValueError):
            balanced_anchor_indices(40, 20, 9, 20)


if __name__ == "__main__":
    unittest.main()
