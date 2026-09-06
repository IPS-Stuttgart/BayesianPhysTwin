"""Fast numerical tests, not scientific evidence."""
import unittest
import numpy as np
from scipy.special import ndtr
from model import ConditionalMixture, Specification

class TestConditionalMixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(7)
        cls.x=rng.normal(size=(160,3))
        cls.y=cls.x @ rng.normal(scale=.003,size=(3,2))+rng.normal(scale=.001,size=(160,2))
        cls.fit=ConditionalMixture(Specification('single',1)).fit(cls.x,cls.y)
    def test_normalized_probabilities_and_finite_density(self):
        w,m=self.fit.components(self.x)
        np.testing.assert_allclose(w.sum(1),1)
        self.assertTrue(np.isfinite(self.fit.logpdf(self.x,self.y)).all())
    def test_prediction_permutation(self):
        idx=np.arange(len(self.x))[::-1]
        np.testing.assert_allclose(self.fit.predict(self.x[idx]),self.fit.predict(self.x)[idx])
    def test_single_component_matches_hard_and_gaussian_quantiles(self):
        np.testing.assert_allclose(self.fit.predict(self.x),self.fit.predict(self.x,hard=True))
        lo,hi=self.fit.intervals(self.x)
        mu=self.fit.predict(self.x)
        sd=np.sqrt(np.diag(self.fit.conditionals[0]))
        np.testing.assert_allclose(lo,mu-1.6448536269514722*sd,atol=1e-12)
        np.testing.assert_allclose(hi,mu+1.6448536269514722*sd,atol=1e-12)
    def test_one_component_dp_equals_finite(self):
        model=ConditionalMixture(Specification('dp',1)).fit(self.x,self.y)
        np.testing.assert_allclose(model.predict(self.x),self.fit.predict(self.x),atol=1e-10)
    def test_invalid_input_rejected(self):
        for arr in (np.array([[np.nan]*3]), np.zeros((0,3))):
            with self.assertRaises(ValueError): self.fit.predict(arr)
    def test_log_density_matches_independent_calculation(self):
        from scipy.stats import multivariate_normal
        predicted=self.fit.predict(self.x)
        expected=np.array([multivariate_normal.logpdf(y,mean=m,cov=self.fit.conditionals[0]) for y,m in zip(self.y,predicted)])
        np.testing.assert_allclose(expected,self.fit.logpdf(self.x,self.y),atol=1e-10)
    def test_covariance_positive_definite(self):
        for covariance in self.fit.conditionals: self.assertGreater(np.linalg.eigvalsh(covariance).min(),0)
    def test_output_dimension_preserved(self):
        self.assertEqual(self.fit.predict(self.x).shape,self.y.shape)

if __name__=='__main__': unittest.main()
