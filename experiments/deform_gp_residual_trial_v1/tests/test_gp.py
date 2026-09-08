"""Numerical checks only. These are NOT DEFORM empirical evidence."""
from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gp import GPConfig, balanced_anchor_indices, fit_gp, matern32


def sample(seed=13):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(80, 5))
    y = rng.normal(size=(80, 3))
    groups = np.repeat(np.arange(10), 8)
    return x, y, groups


def test_zero_nonlinearity_matches_legacy_ridge_mean():
    x, y, groups = sample()
    gp = fit_gp(x, y, groups, GPConfig(amplitude=0))
    design = np.c_[np.ones(len(x)), (x-x.mean(0))/x.std(0)]
    penalty = np.diag([0]+[1]*x.shape[1])
    weights = np.linalg.solve(design.T@design + penalty, design.T@y)
    assert_allclose(gp.weights, weights, rtol=1e-12, atol=1e-13)
    assert_allclose(gp.predict(x), design@weights, rtol=1e-12, atol=1e-13)


def test_linear_reference_cluster_covariance():
    x, y, groups = sample()
    gp = fit_gp(x, y, groups, GPConfig(amplitude=0))
    design = gp.design(x)
    bread = np.linalg.inv(design.T@design+np.diag([0]+[1]*x.shape[1]))
    error = y-design@gp.weights
    for c in range(y.shape[1]):
        score = np.stack([design[groups==g].T@error[groups==g,c] for g in np.unique(groups)])
        expected = bread@(score.T@score*10/9)@bread
        assert_allclose(gp.cluster_covariance[c], expected, atol=1e-12)


def test_nystrom_feature_kernel_identity():
    x, y, groups = sample()
    gp = fit_gp(x,y,groups,GPConfig(max_anchors=18))
    z = (x-gp.location)/gp.scale
    cross = matern32(z,gp.anchors,np.sqrt(5.0))
    kzz = matern32(gp.anchors,gp.anchors,np.sqrt(5.0))+1e-9*np.eye(18)
    nonlinear = gp.design(x)[:,6:]
    assert_allclose(nonlinear@nonlinear.T,9*cross@np.linalg.solve(kzz,cross.T),atol=1e-10)


def test_mean_matches_direct_kernel_universal_kriging():
    x,y,groups = sample()
    gp=fit_gp(x,y,groups,GPConfig(max_anchors=20,ridge=1.7))
    phi=gp.design(x)[:,1:]
    c=phi@phi.T+1.7*np.eye(len(x))
    ci_y=np.linalg.solve(c,y)
    ci_1=np.linalg.solve(c,np.ones(len(x)))
    intercept=(ci_1@y)/ci_1.sum()
    alpha=ci_y-ci_1[:,None]*intercept
    expected=intercept+(phi@phi.T)@alpha
    assert_allclose(gp.predict(x),expected,atol=1e-11)


@pytest.mark.parametrize('robust',[False,True])
def test_covariance_psd_and_marginal_agreement(robust):
    x,y,groups=sample()
    gp=fit_gp(x,y,groups,GPConfig(max_anchors=20))
    cov=gp.joint_covariance(x[:17],robust=robust,observation_noise=True)
    marginal=gp.marginal_variance(x[:17],robust=robust)
    assert np.linalg.eigvalsh(cov).min()>-1e-10
    assert_allclose(np.diagonal(cov,axis1=1,axis2=2).T,marginal,atol=1e-11)


def test_anchor_selection_balanced():
    groups=np.repeat(np.arange(10),8)
    ix=balanced_anchor_indices(groups,25,12)
    counts=np.bincount(groups[ix],minlength=10)
    assert counts.max()-counts.min()==1
    assert len(np.unique(ix))==25


def test_basis_does_not_depend_on_targets():
    x,y,groups=sample()
    a=fit_gp(x,y,groups,GPConfig(max_anchors=20))
    b=fit_gp(x,-100*y,groups,GPConfig(max_anchors=20))
    assert_allclose(a.anchors,b.anchors,atol=0,rtol=0)
    assert_allclose(a.anchor_cholesky,b.anchor_cholesky,atol=0,rtol=0)


def test_every_training_target_is_used():
    x,y,groups=sample()
    a=fit_gp(x,y,groups,GPConfig(max_anchors=10))
    j=next(i for i in range(len(x)) if i not in a.anchor_indices)
    altered=y.copy(); altered[j]+=100
    b=fit_gp(x,altered,groups,GPConfig(max_anchors=10))
    assert not np.allclose(a.predict(x),b.predict(x))


def test_output_offset_equivariance():
    x,y,groups=sample()
    a=fit_gp(x,y,groups,GPConfig(max_anchors=20))
    b=fit_gp(x,y+np.array([3,-10,5]),groups,GPConfig(max_anchors=20))
    assert_allclose(b.predict(x),a.predict(x)+[3,-10,5],atol=1e-10)


def test_zero_residual():
    x,y,groups=sample()
    gp=fit_gp(x,np.zeros_like(y),groups,GPConfig(max_anchors=10))
    assert_allclose(gp.predict(x),0,atol=0)


def test_constant_input_features():
    x,y,groups=sample(); x[:,2]=4
    gp=fit_gp(x,y,groups,GPConfig(max_anchors=20))
    assert np.isfinite(gp.predict(x)).all()
    assert gp.scale[2]==1


@pytest.mark.parametrize('change',[{'ridge':0},{'amplitude':-1},{'length_multiplier':0},
                                    {'max_anchors':0},{'kernel_jitter':0},{'seed':-1}])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        replace(GPConfig(),**change).validate()


def test_nonfinite_data_rejected():
    x,y,groups=sample(); x[0,0]=np.nan
    with pytest.raises(ValueError): fit_gp(x,y,groups)


def test_one_cluster_rejected():
    x,y,groups=sample()
    with pytest.raises(ValueError): fit_gp(x,y,np.zeros_like(groups))


def test_input_arrays_unchanged():
    x,y,groups=sample(); copies=(x.copy(),y.copy(),groups.copy())
    fit_gp(x,y,groups,GPConfig(max_anchors=20))
    for before,after in zip(copies,(x,y,groups)):
        assert np.array_equal(before,after)


def test_pickle_free_arrays(tmp_path):
    x,y,groups=sample()
    gp=fit_gp(x,y,groups,GPConfig(max_anchors=10))
    path=tmp_path/'model.npz'; np.savez_compressed(path,**gp.arrays())
    with np.load(path,allow_pickle=False) as bundle:
        assert set(bundle.files)==set(gp.arrays())
        for key in bundle.files: assert bundle[key].dtype.kind!='O'


def test_nonlinear_synthetic_mechanism():
    """A deliberately nonlinear software fixture, not real-world evidence."""
    rng=np.random.default_rng(891)
    x=rng.uniform(-1,1,(240,2)); groups=np.repeat(np.arange(24),10)
    y=(np.sin(3*x[:,0])+.1*x[:,1])[:,None]
    q=rng.uniform(-.9,.9,(150,2)); truth=(np.sin(3*q[:,0])+.1*q[:,1])[:,None]
    linear=fit_gp(x,y,groups,GPConfig(amplitude=0))
    gp=fit_gp(x,y,groups,GPConfig(max_anchors=64,length_multiplier=.5))
    rmse_linear=np.sqrt(np.mean((linear.predict(q)-truth)**2))
    rmse_gp=np.sqrt(np.mean((gp.predict(q)-truth)**2))
    assert rmse_gp < rmse_linear
