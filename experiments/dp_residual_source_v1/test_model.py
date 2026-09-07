"""Numerical, probabilistic, temporal-boundary, and reproducibility checks."""

import itertools
import unittest

import numpy as np
from model import (
    Draw,
    design,
    fit,
    forecast,
    forward,
    sample_forecast,
    transition_counts,
    update_beta,
)
from scipy.special import logsumexp


class ModelTests(unittest.TestCase):
    def test_design_alignment(self):
        d = np.arange(20.0).reshape(2, 5, 2)
        c = np.arange(10.0).reshape(2, 5, 1)
        x, y = design(d, c)
        np.testing.assert_array_equal(x[0, 0], np.r_[d[0, 1], d[0, 0], c[0, 2], 1])
        np.testing.assert_array_equal(y, d[:, 2:])

    def test_design_rejects_bad_data(self):
        with self.assertRaises(ValueError):
            design(np.zeros((2, 2, 1)), np.zeros((2, 2, 1)))
        with self.assertRaises(ValueError):
            design(np.full((2, 3, 1), np.nan), np.zeros((2, 3, 1)))

    def test_forward_exact_enumeration(self):
        p = np.array([[0.8, 0.2], [0.3, 0.7]])
        initial = np.array([0.4, 0.6])
        e = np.log(np.array([[[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]]]))
        f, ll = forward(e, p, initial)
        masses = []
        terminal = np.zeros(2)
        for z in itertools.product(range(2), repeat=3):
            v = initial[z[0]] * np.exp(e[0, 0, z[0]])
            for t in (1, 2):
                v *= p[z[t - 1], z[t]] * np.exp(e[0, t, z[t]])
            masses.append(v)
            terminal[z[-1]] += v
        self.assertAlmostEqual(ll[0], np.log(sum(masses)), places=12)
        np.testing.assert_allclose(
            np.exp(f[0, -1]), terminal / terminal.sum(), atol=1e-12
        )
        np.testing.assert_allclose(logsumexp(f, axis=-1), 0, atol=1e-12)

    def test_no_cross_recording_transitions(self):
        z = np.array([[0, 0, 1], [1, 0, 0]])
        c = transition_counts(z, 2)
        self.assertEqual(c.sum(), 4)
        np.testing.assert_array_equal(c, [[2, 1], [1, 0]])

    def test_sticky_tables_do_not_count_as_global_tables(self):
        rng = np.random.default_rng(7)
        counts = np.zeros((3, 3), int)
        counts[0, 0] = 500
        beta = np.ones(3) / 3
        draws = np.array(
            [update_beta(counts, beta, 2, 1e12, 1, rng) for _ in range(300)]
        )
        np.testing.assert_allclose(draws.sum(axis=1), 1, atol=1e-12)
        self.assertLess(abs(draws[:, 0].mean() - 1 / 3), 0.06)

    def fixture(self, k=1):
        coef = np.zeros((k, 4, 1))
        coef[:, 0, 0] = 0.8
        coef[:, 1, 0] = 0.1
        coef[:, 2, 0] = 0.3
        return Draw(
            coef,
            np.ones((k, 1, 1)) * 0.002,
            np.eye(k),
            np.ones(k) / k,
            np.ones(k) / k,
            k,
            0.0,
        )

    def test_single_regime_exact_recursion(self):
        dr = self.fixture()
        prefix = np.array([[[0.1], [0.2], [0.3], [0.25]]])
        past = np.zeros((1, 4, 1))
        future = np.ones((1, 8, 1))
        actual = forecast([dr], prefix, past, future)
        a, b = 0.3, 0.25
        expected = []
        for _ in range(8):
            a, b = b, 0.8 * b + 0.1 * a + 0.3
            expected.append(b)
        np.testing.assert_allclose(actual.ravel(), expected, atol=1e-12)

    def test_predictive_sampling_matches_moments(self):
        dr = self.fixture(2)
        dr.coef[1, 0, 0] = -0.6
        dr.transition = np.array([[0.9, 0.1], [0.05, 0.95]])
        prefix = np.zeros((1, 4, 1))
        past = np.zeros((1, 4, 1))
        future = np.ones((1, 5, 1))
        exact = forecast([dr], prefix, past, future)
        draws = sample_forecast([dr], prefix, past, future, samples=1600, seed=3)
        np.testing.assert_allclose(draws.mean(axis=0), exact, atol=0.025)

    def test_forecast_does_not_mutate_prefix(self):
        prefix = np.arange(6.0).reshape(1, 6, 1)
        original = prefix.copy()
        forecast([self.fixture()], prefix, np.zeros((1, 6, 1)), np.zeros((1, 5, 1)))
        np.testing.assert_array_equal(prefix, original)

    def test_sampler_reproducible_and_positive(self):
        rng = np.random.default_rng(42)
        d = rng.normal(size=(4, 25, 2)).cumsum(axis=1) * 0.05
        c = np.zeros((4, 25, 1))
        a, trace = fit(d, c, k=3, hdp=True, seed=2, iterations=8, burn=4, thin=2)
        b, _ = fit(d, c, k=3, hdp=True, seed=2, iterations=8, burn=4, thin=2)
        self.assertEqual(len(a), 2)
        for u, v in zip(a, b, strict=False):
            np.testing.assert_array_equal(u.coef, v.coef)
            self.assertTrue((np.linalg.eigvalsh(u.covariance) > 0).all())
            np.testing.assert_allclose(u.transition.sum(axis=1), 1)
        self.assertFalse(trace["mcmc_convergence_established"])

    def test_finite_control_global_weights_are_fixed(self):
        rng = np.random.default_rng(2)
        d = rng.normal(size=(3, 20, 1))
        c = np.zeros((3, 20, 1))
        a, _ = fit(d, c, k=2, hdp=False, seed=5, iterations=6, burn=3, thin=1)
        for dr in a:
            np.testing.assert_array_equal(dr.beta, [0.5, 0.5])

    def test_parameter_draw_averaging_is_label_invariant(self):
        dr = self.fixture(2)
        dr.coef[1, 0, 0] = -0.4
        dr.transition = np.array([[0.9, 0.1], [0.2, 0.8]])
        perm = np.array([1, 0])
        swapped = Draw(
            dr.coef[perm],
            dr.covariance[perm],
            dr.transition[np.ix_(perm, perm)],
            dr.initial[perm],
            dr.beta[perm],
            2,
            0.0,
        )
        prefix = np.zeros((1, 4, 1))
        past = np.zeros((1, 4, 1))
        future = np.ones((1, 5, 1))
        np.testing.assert_allclose(
            forecast([dr], prefix, past, future),
            forecast([swapped], prefix, past, future),
            atol=1e-12,
        )

    def test_future_observations_are_not_forecast_arguments(self):
        import inspect

        self.assertEqual(
            list(inspect.signature(forecast).parameters),
            ["draws", "prefix", "past_controls", "future_controls", "hard"],
        )


if __name__ == "__main__":
    unittest.main()
