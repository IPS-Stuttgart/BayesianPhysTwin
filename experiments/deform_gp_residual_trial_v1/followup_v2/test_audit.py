"""Numerical fixtures only; these tests do not constitute real-data evidence."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import (fit_fast, condition, fit_bias, fold_indices, probabilistic_scores,
                   full_model_prediction, world_prediction, calibrate_covariance, OBS)
from gp import GPConfig, fit_gp

@pytest.mark.parametrize('amplitude',[0.,3.])
def test_fast_fit_preserves_original_mean_precision_and_noise(amplitude):
    r=np.random.default_rng(1);x=r.normal(size=(100,5));y=r.normal(size=(100,3));g=np.repeat(np.arange(10),10)
    cfg=GPConfig(amplitude=amplitude,max_anchors=12)
    a=fit_gp(x,y,g,cfg);b=fit_fast(x,y,g,cfg)
    for name in ['weights','precision_inverse','residual_variance','location','scale','anchors','anchor_cholesky']:
        np.testing.assert_allclose(getattr(a,name),getattr(b,name),atol=2e-12,rtol=2e-12)


def test_condition_matches_block_gaussian():
    r=np.random.default_rng(2);f=r.normal(size=(17,6));k=f@f.T+.3*np.eye(17)
    obs=np.arange(4);q=np.arange(4,17);innovation=r.normal(size=(4,3))
    delta,var=condition(k[np.ix_(q,obs)],k[np.ix_(obs,obs)],np.diag(k)[q],innovation)
    np.testing.assert_allclose(delta,k[np.ix_(q,obs)]@np.linalg.solve(k[np.ix_(obs,obs)],innovation),atol=1e-12)
    truth=k[np.ix_(q,q)]-k[np.ix_(q,obs)]@np.linalg.solve(k[np.ix_(obs,obs)],k[np.ix_(obs,q)])
    np.testing.assert_allclose(var,np.diag(truth),atol=1e-12)
    assert np.all(var<=np.diag(k)[q]+1e-12)


def test_zero_cross_covariance_no_update():
    delta,var=condition(np.zeros((8,2)),np.eye(2),np.ones(8),np.ones((2,3))*99)
    assert np.array_equal(delta,np.zeros((8,3)))
    assert np.array_equal(var,np.ones(8))


def test_fold_partition_is_trajectory_disjoint_and_complete():
    folds=fold_indices(40)
    assert np.array_equal(np.sort(np.concatenate(folds)),np.arange(40))
    assert all(len(x)==8 for x in folds)
    for a,b in zip(folds,fold_indices(40)): assert np.array_equal(a,b)


def test_bias_fit_constant_and_unhelpful():
    e=np.ones((40,498,8,3))
    np.testing.assert_allclose(fit_bias(e),1.)
    e[:,50:]*=-1
    np.testing.assert_allclose(fit_bias(e),0.)


def test_probabilistic_score_gaussian_fixture():
    r=np.random.default_rng(4);e=r.normal(scale=.01,size=(40,498,8,3))
    s=probabilistic_scores(e,np.ones((1,1,8,3))*1e-4)
    assert abs(s['coverage95_percent'].mean()-95)<.2
    assert abs(s['normalized_squared_error'].mean()-1)<.02
    assert abs(s['crps_mm'].mean()-10/np.sqrt(np.pi))<.05


def test_world_rotation_and_clamps():
    base=np.zeros((2,498,12,3));local=np.ones((2,498,8,3));frames=np.repeat(np.eye(3)[None],2,axis=0)
    frames[1]=[[0,-1,0],[1,0,0],[0,0,1]]
    p=world_prediction(base,local,frames)
    assert np.array_equal(p[:,:,[0,1,-2,-1]],base[:,:,[0,1,-2,-1]])
    np.testing.assert_allclose(p[1,:,2:-2,0],-1)
    assert not base.any()


def test_adaptation_accepts_only_prefix_and_is_deterministic():
    r=np.random.default_rng(5);x=r.normal(size=(60,3));y=r.normal(size=(60,3));g=np.repeat(np.arange(6),10)
    m=fit_fast(x,y,g,GPConfig(amplitude=0));features=r.normal(size=(2,498,1,3))
    p=[{'a':1.,'b':1.,'normalizer':1.,'q_m2':[1.,1.,1.]}]
    prefix=r.normal(size=(2,len(OBS),1,3))
    a=full_model_prediction(features,[m],p,prefix)
    b=full_model_prediction(features,[m],p,prefix.copy())
    for aa,bb in zip(a,b): np.testing.assert_array_equal(aa,bb)
    with pytest.raises(ValueError):full_model_prediction(features,[m],p,np.zeros((2,498,1,3)))


def test_covariance_calibration_converges_on_known_correlated_fixture():
    r=np.random.default_rng(6);n=40;grid=20
    # Use the same covariance at every trajectory, with nondegenerate eigenvalues.
    k=.7*np.ones((grid,grid))+.3*np.eye(grid)
    c=np.broadcast_to(k,(n,1,grid,grid)).copy()
    e=r.normal(size=(n,498,1,3))
    from audit import GRID
    samples=np.einsum('ij,njc->nic',np.linalg.cholesky(k),r.normal(size=(n,grid,3)))
    e[:,GRID,0]=samples
    p=calibrate_covariance(e,c)[0]
    assert p['optimizer_success'] and p['a']>.2 and p['b']<1
