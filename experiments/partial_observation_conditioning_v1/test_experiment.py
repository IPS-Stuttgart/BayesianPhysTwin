"""Run with unittest discovery; synthetic arrays test code, not real-data claims."""
import unittest

import numpy as np

from run import (ANCHORS, MASKS, covariance, gaussian_map, information_map,
                 indices, interpolation_map, metrics, moments, windows)


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260906)

    def test_gaussian_equals_information_form_map(self):
        a = self.rng.normal(size=(24, 24))
        c = a @ a.T * 1e-4 + np.eye(24) * 1e-5
        obs, hidden, _ = indices([0, 7], 0)
        gain, post = gaussian_map(c, obs, hidden, 1e-6)
        other = information_map(c, obs, hidden, 1e-6)
        np.testing.assert_allclose(gain, other, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(post).min(), 0)

    def test_diagonal_cannot_correct_unobserved_nodes(self):
        c = np.diag(np.linspace(1e-5, 1e-3, 48))
        obs, hidden, _ = indices([0, 2, 5, 7], 30)
        gain, _ = gaussian_map(c, obs, hidden, 1e-6)
        np.testing.assert_array_equal(gain, np.zeros_like(gain))

    def test_masks_disjoint_and_future_observations_not_used(self):
        for nodes in MASKS.values():
            for h in [0, 10, 30]:
                obs, hidden, hn = indices(nodes, h)
                self.assertTrue(set(nodes).isdisjoint(hn))
                self.assertEqual(set(nodes) | set(hn), set(range(8)))
                self.assertLess(obs.max(), 24)
                self.assertEqual(len(hidden), 3 * (8 - len(nodes)))
                if h:
                    self.assertGreaterEqual(hidden.min(), 24)

    def test_covariance_marginals_match_and_are_positive_definite(self):
        for h in [0, 30]:
            d = 24 if h == 0 else 48
            _, sample = moments(self.rng.normal(size=(5, 18, d)) * .01)
            for q in [(.25, 10, 50, 1e-6), (1., 1, 10, 1e-8), (.3, 0, 50, 1e-4)]:
                c = covariance(sample, h, q)
                np.testing.assert_allclose(np.diag(c), np.diag(sample), atol=1e-14)
                self.assertGreater(np.linalg.eigvalsh(c).min(), 0)

    def test_correct_dependence_has_positive_control_value(self):
        c = np.kron(np.ones((8, 8)), np.eye(3)) * 1e-4 + np.eye(24) * 1e-8
        truth = self.rng.multivariate_normal(np.zeros(24), c, size=1000)
        obs, hidden, _ = indices([0, 7], 0)
        gain, _ = gaussian_map(c, obs, hidden, 1e-8)
        before = np.mean(truth[:, hidden] ** 2)
        after = np.mean((truth[:, hidden] - truth[:, obs] @ gain.T) ** 2)
        self.assertLess(after, .01 * before)

    def test_hidden_target_poison_does_not_change_conditioning(self):
        a = self.rng.normal(size=(48, 48))
        c = a @ a.T + np.eye(48)
        obs, hidden, _ = indices([0, 7], 30)
        x = self.rng.normal(size=(20, 48))
        gain, _ = gaussian_map(c, obs, hidden, .01)
        original = x[:, obs] @ gain.T
        x[:, hidden] = 1e9
        np.testing.assert_array_equal(original, x[:, obs] @ gain.T)

    def test_interpolation_reproduces_affine_shape(self):
        nodes = [0, 7]
        hidden = list(range(1, 7))
        gain = interpolation_map(nodes, hidden)
        values = np.array([[n, 2*n+1, -n] for n in nodes]).reshape(-1)
        expected = np.array([[n, 2*n+1, -n] for n in hidden]).reshape(-1)
        np.testing.assert_allclose(gain @ values, expected)

    def test_window_index_and_clamped_node_exclusion(self):
        p = np.zeros((2, 498, 12, 3))
        t = np.broadcast_to(np.arange(498)[None, :, None, None], p.shape).copy()
        e, _, _ = windows(p, t, 30)
        self.assertEqual(e.shape, (2, 18, 48))
        np.testing.assert_array_equal(e[0, :, 0], ANCHORS)
        np.testing.assert_array_equal(e[0, :, 24], ANCHORS+30)

    def test_translation_free_metric_removes_only_translation(self):
        e = np.full((2, 18, 12), .01)
        m = metrics(e)
        np.testing.assert_allclose(m['rmse_mm'], np.sqrt(3) * 10)
        np.testing.assert_allclose(m['translation_free_rmse_mm'], 0, atol=1e-12)

    def test_zero_error_metrics_finite(self):
        m = metrics(np.zeros((2, 18, 12)), np.eye(12) * 1e-4)
        self.assertEqual(m['coverage90'].tolist(), [1., 1.])
        self.assertTrue(all(np.isfinite(v).all() for v in m.values()))


if __name__ == '__main__':
    unittest.main()
