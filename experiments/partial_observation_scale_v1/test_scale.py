"""Analytic and information-boundary checks; no real outcomes used."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy import integrate, stats

spec = importlib.util.spec_from_file_location("scale_study", Path(__file__).with_name("run.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class ScaleTests(unittest.TestCase):
    def test_conjugate_formula(self):
        q, m, nu, scale = np.array([0., 7., 40.]), 6, 5., 1.3
        df, mult = s.law(q, m, {"family": "bayes", "nu": nu, "scale": scale})
        self.assertEqual(df, 11.)
        alpha = (nu + m) / 2
        beta = (scale * (nu - 2) + q) / 2
        np.testing.assert_allclose(mult, beta / alpha)

    def test_exact_moment_match(self):
        q = np.linspace(0, 50, 100)
        cfg = {"family": "bayes", "nu": 3., "scale": 2.}
        df, tscale = s.law(q, 12, cfg)
        gdf, variance = s.law(q, 12, {"family": "moment", "scale": 1.}, cfg)
        self.assertIsNone(gdf)
        np.testing.assert_allclose(variance, tscale * df / (df - 2))

    def test_gaussian_limit(self):
        q = np.linspace(0, 100, 50)
        cfg = {"family": "bayes", "nu": 1e9, "scale": 1.7}
        df, v = s.law(q, 6, cfg)
        np.testing.assert_allclose(v, 1.7, rtol=1e-7)

    def test_direct_family_can_reproduce_bayesian_conditional_law(self):
        q = np.linspace(0, 60, 30)
        for m in (6, 12):
            nu, prior_scale = 5., 1.4
            df, ts = s.law(q, m, {"family": "bayes", "nu": nu, "scale": prior_scale})
            a = prior_scale * (nu - 2) / (nu + m - 2)
            b = m / (nu + m - 2)
            direct_df, direct_scale = s.law(q, m, {"family": "direct", "nu": df,
                                                    "scale": a + b, "beta": b / (a + b)})
            self.assertEqual(df, direct_df)
            np.testing.assert_allclose(ts, direct_scale)

    def test_multivariate_density_against_scipy(self):
        rng = np.random.default_rng(3)
        for df in (None, 9.):
            cov = np.array([[1., .3], [.3, .7]])
            errors = rng.normal(size=(3, 5, 2))
            mult = np.full((3, 5), 1.6)
            packet = {"dimension": 2, "logdet": np.linalg.slogdet(cov)[1],
                      "mahal": np.einsum('...i,ij,...j->...', errors, np.linalg.inv(cov), errors)}
            actual = s.log_score(packet, df, mult)
            dist = stats.multivariate_normal(cov=cov * 1.6) if df is None else stats.multivariate_t(shape=cov * 1.6, df=df)
            expected = -dist.logpdf(errors) / 2
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_crps_numerical_integral(self):
        for df in (None, 3., 9., 30.):
            for y in (-2., 0., 1.):
                dist = stats.norm(scale=1.3) if df is None else stats.t(df, scale=1.3)
                expected = (integrate.quad(lambda x: dist.cdf(x) ** 2, -np.inf, y)[0]
                            + integrate.quad(lambda x: dist.sf(x) ** 2, y, np.inf)[0])
                actual = float(s.crps_values(np.array(y), np.array(1.3), df))
                self.assertAlmostEqual(actual, expected, places=7)

    def test_hidden_and_future_poison_does_not_change_prediction(self):
        rng = np.random.default_rng(9)
        a = rng.normal(size=(48, 48))
        cov = (a @ a.T + np.eye(48)) * .0001
        mu = rng.normal(size=48) * .001
        e = rng.normal(size=(2, 18, 48)) * .01
        for nodes in s.MASKS.values():
            obs, hidden = s.indices(nodes)
            before = s.visible_prediction(e[..., obs], mu, cov, nodes, 1e-6)
            poisoned = e.copy()
            unobserved = [j for j in range(48) if j not in obs]
            poisoned[..., unobserved] = 1e8
            after = s.visible_prediction(poisoned[..., obs], mu, cov, nodes, 1e-6)
            for x, y in zip(before, after):
                np.testing.assert_array_equal(x, y)
            self.assertFalse(set(obs) & set(hidden))
            self.assertTrue(np.all(obs < 24))
            self.assertTrue(np.all(hidden >= 24))

    def test_larger_visible_surprise_increases_posterior_variance(self):
        df, scales = s.law(np.array([0., 1., 20.]), 6, {"family": "bayes", "nu": 5., "scale": 1.})
        self.assertTrue(np.all(np.diff(scales) > 0))
        self.assertGreater(df, 2)

    def test_partition_unique_and_order_invariant(self):
        names = [f"{j}.pkl" for j in range(56)]
        split = s.split_names(names)
        reverse = names[::-1]
        other = s.split_names(reverse)
        self.assertEqual([len(x) for x in split.values()], [32, 12, 12])
        for k in split:
            self.assertEqual([names[i] for i in split[k]], [reverse[i] for i in other[k]])
        self.assertEqual(len(set(sum(split.values(), []))), 56)

    def test_correct_shared_scale_positive_control(self):
        rng = np.random.default_rng(55)
        n, m, k, nu = 12000, 6, 6, 5.
        latent = (nu - 2) / rng.chisquare(nu, size=n)
        obs = rng.normal(size=(n, m)) * np.sqrt(latent[:, None])
        hid = rng.normal(size=(n, k)) * np.sqrt(latent[:, None])
        q = np.sum(obs ** 2, axis=-1)
        p = {"q": q, "mahal": np.sum(hid ** 2, axis=-1), "dimension": k, "logdet": 0.}
        b = {"family": "bayes", "nu": nu, "scale": 1.}
        d = {"family": "direct", "nu": nu, "scale": 1., "beta": 0.}
        active = s.log_score(p, *s.law(q, m, b)).mean()
        static = s.log_score(p, *s.law(q, m, d)).mean()
        self.assertLess(active, static - .01)

    def test_independent_hidden_scale_negative_control(self):
        rng = np.random.default_rng(11)
        n, m, k, nu = 12000, 6, 6, 5.
        lo = (nu - 2) / rng.chisquare(nu, size=n)
        lh = (nu - 2) / rng.chisquare(nu, size=n)
        q = np.sum((rng.normal(size=(n, m)) * np.sqrt(lo[:, None])) ** 2, axis=-1)
        hid = rng.normal(size=(n, k)) * np.sqrt(lh[:, None])
        p = {"mahal": np.sum(hid ** 2, axis=-1), "dimension": k, "logdet": 0.}
        active = s.log_score(p, *s.law(q, m, {"family": "bayes", "nu": nu, "scale": 1.})).mean()
        static = s.log_score(p, *s.law(q, m, {"family": "direct", "nu": nu, "scale": 1., "beta": 0.})).mean()
        self.assertGreater(active, static + .01)

    def test_nested_direct_dispatch(self):
        cfg = {"family": "direct", "nu": 9., "scale": 1.2, "beta": .3}
        nested = {"family": "by_dimension", "configs": {"6": cfg}}
        expected = s.law(np.array([2., 8.]), 6, cfg)
        actual = s.law(np.array([2., 8.]), 6, nested)
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    def test_invalid_observation_rejected(self):
        with self.assertRaises(ValueError):
            s.visible_prediction(np.ones((1, 3)), np.zeros(48), np.eye(48), [0, 7], 1e-6)
        with self.assertRaises(ValueError):
            s.law(np.array([1.]), 6, {"family": "bayes", "nu": 2., "scale": 1.})


if __name__ == "__main__":
    unittest.main()
