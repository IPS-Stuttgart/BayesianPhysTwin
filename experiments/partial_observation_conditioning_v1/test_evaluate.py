"""Numerical and information-boundary tests; fixtures are not empirical evidence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate as m


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(49)
        u = rng.normal(size=(m.D, 4))*.01
        q = u @ u.T + np.eye(m.D)*1e-5
        self.model = {'A': .9*np.eye(m.D), 'B': np.zeros((m.D, 24)),
                      'c': np.zeros(m.D), 'init': np.zeros((m.D, 12)),
                      'init_bias': np.zeros(m.D), 'P0': q*10, 'Q': q}
        self.boundary = np.zeros((30, 12))
        self.obs = np.where(m.masks('center_hidden', 30), rng.normal(size=(30, m.D))*.01, np.nan)

    def test_cross_covariance_updates_hidden(self):
        p = np.eye(m.D)
        p[0, 3] = p[3, 0] = .8
        mean, post = m.condition_gaussian(np.zeros(m.D), p, np.array([0]), np.array([2.]), 1.)
        self.assertAlmostEqual(mean[3], .8)
        self.assertGreater(np.linalg.eigvalsh(post).min(), 0)

    def test_diagonal_does_not_instantly_update_hidden(self):
        ix = np.array([0, 1, 2])
        result, _ = m.condition_gaussian(np.zeros(m.D), np.eye(m.D), ix, np.ones(3), .01)
        np.testing.assert_array_equal(result[3:], 0)

    def test_gaussian_map_mean_is_identical(self):
        p = self.model['P0']
        ix = np.array([0, 1, 2, 9, 10, 11])
        h = np.eye(m.D)[ix]
        values = np.arange(6)*.01
        mean, post = m.condition_gaussian(np.zeros(m.D), p, ix, values, .0001)
        precision = np.linalg.inv(p) + h.T @ h/.0001
        deterministic = np.linalg.solve(precision, h.T @ values/.0001)
        np.testing.assert_allclose(mean, deterministic, atol=1e-12)
        np.testing.assert_allclose(post, np.linalg.inv(precision), atol=1e-12)

    def test_future_observations_cannot_change_earlier_estimates_or_forecasts(self):
        changed = self.obs.copy()
        changed[15:] = np.where(np.isfinite(changed[15:]), 900., np.nan)
        a = m.replay(self.model, self.boundary, self.obs, 'full', .5)
        b = m.replay(self.model, self.boundary, changed, 'full', .5)
        np.testing.assert_array_equal(a[0][:15], b[0][:15])
        for h in m.CONFIG['horizons_steps']:
            np.testing.assert_array_equal(a[1][h][:15], b[1][h][:15])

    def test_no_observations_equals_shared_open_loop(self):
        obs = np.full_like(self.obs, np.nan)
        a = m.replay(self.model, self.boundary, obs, 'full', .5)[0]
        b = m.replay(self.model, self.boundary, obs, 'model_only', 1.)[0]
        np.testing.assert_array_equal(a, b)

    def test_audit_has_exact_same_full_mean(self):
        result = m.replay(self.model, self.boundary, self.obs, 'full', .5, audit=True)
        np.testing.assert_allclose(result[0], result[3]['full'], atol=1e-14)
        visible = np.isfinite(self.obs)
        np.testing.assert_array_equal(result[3]['prior'][~visible], result[3]['diagonal'][~visible])

    def test_source_split_disjoint_deterministic(self):
        names = [f'{n}.pkl' for n in range(56)]
        a = m.split_names(names, 'DLO4')
        b = m.split_names(list(reversed(names)), 'DLO4')
        self.assertEqual(a, b)
        self.assertEqual([len(a[k]) for k in ('fit', 'calibration', 'source_test')], [39, 9, 8])
        self.assertEqual(len(set(sum(a.values(), []))), 56)

    def test_reserved_evaluation_path_rejected_before_open(self):
        with self.assertRaises(ValueError):
            m.load_train(Path('/does-not-exist/DLO4/eval/example.pkl'))

    def test_masks_are_nodewise_and_prespecified(self):
        for condition in m.CONDITIONS:
            visible = m.masks(condition, 100).reshape(100, 8, 3)
            np.testing.assert_array_equal(visible[:, :, 0], visible[:, :, 1])
            np.testing.assert_array_equal(visible[:, :, 1], visible[:, :, 2])
        self.assertFalse(m.masks('gap10', 100)[30:40].any())

    def test_all_visible_overwrite_exact(self):
        rng = np.random.default_rng(9)
        obs = rng.normal(size=(30, m.D))
        result = m.replay(self.model, self.boundary, obs, 'overwrite', 1.)[0]
        np.testing.assert_array_equal(result, obs)

    def test_rmse_is_euclidean_per_node_not_coordinate(self):
        self.assertAlmostEqual(m.rmse_mm(np.ones((2, m.D))*.001, np.zeros((2, m.D)), np.ones((2, m.D), bool)), np.sqrt(3))

    def test_dataset_loader_rigid_transform_preserves_shape(self):
        import pickle
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'DLO4'/'train'/'one.pkl'
            path.parent.mkdir(parents=True)
            raw = np.zeros((500, 3, 12))
            raw[:, 0] = np.arange(12)*.01
            raw[:, 2] = -.1
            with path.open('wb') as f:
                pickle.dump(raw, f)
            boundary, residual = m.load_train(path)
            self.assertEqual(boundary.shape, (100, 12))
            np.testing.assert_allclose(residual, 0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
