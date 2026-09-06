import itertools

import numpy as np
import pytest
from model import emissions, ffbs, fit, forecast, normalize_log, sample_global_weights, transition_counts


def test_normalization_extreme_logs():
    p = normalize_log(np.array([[-100000., -100001.], [2., 3.]]))
    np.testing.assert_allclose(p.sum(-1), 1.)
    np.testing.assert_allclose(p[0], p[1][::-1])


def test_transition_counts_do_not_join_recordings():
    z = np.array([[0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(transition_counts(z, 2), [[2, 1], [1, 0]])


def test_ffbs_agrees_with_enumerated_posterior():
    pi = np.array([[.9, .1], [.2, .8]])
    beta = np.array([.6, .4])
    likelihood = np.array([[.8, .2], [.3, .7], [.6, .4]])
    exact = np.zeros((3, 2))
    total = 0.
    for z in itertools.product(range(2), repeat=3):
        p = beta[z[0]] * likelihood[0,z[0]]
        for t in range(1,3):
            p *= pi[z[t-1],z[t]] * likelihood[t,z[t]]
        total += p
        for t in range(3):
            exact[t,z[t]] += p
    exact /= total
    states = ffbs(np.broadcast_to(np.log(likelihood), (20000,3,2)), pi, beta, np.random.default_rng(2))
    np.testing.assert_allclose(np.stack([(states==k).mean(0) for k in range(2)],axis=-1),exact,atol=.012)


def test_forecast_matches_exact_future_path_enumeration():
    b = np.array([[[.1],[.2],[.8]],[[-.1],[.4],[.5]]])
    q = np.array([[[.2]],[[.3]]])
    pi = np.array([[.85,.15],[.25,.75]])
    beta = np.array([.7,.3])
    prefix_y = np.array([[.4],[.3]])
    prefix_phi = np.array([[.1],[.2]])
    future_phi = np.array([[.2],[.4],[.3]])
    loglik = emissions(np.array([1.,.2,.4])[None,None],np.array([.3])[None,None],b,q)[0,0]
    w = normalize_log(np.log(beta@pi)+loglik)
    result = np.zeros((3,1))
    for start in range(2):
        for modes in itertools.product(range(2),repeat=3):
            p=w[start]
            previous=start
            for k in modes:
                p *= pi[previous,k]
                previous=k
            a=.3
            for h,k in enumerate(modes):
                a=(np.array([1.,future_phi[h,0],a])@b[k])[0]
                result[h,0] += p*a
    actual=forecast(dict(b=b,q=q,pi=pi,beta=beta),prefix_y,prefix_phi,future_phi)
    np.testing.assert_allclose(actual,result,rtol=1e-12,atol=1e-12)


def test_single_expert_exact_autoregression():
    draw=dict(b=np.array([[[0.],[0.],[.8]]]),q=np.array([[[.1]]]),pi=np.ones((1,1)),beta=np.ones(1))
    got=forecast(draw,np.array([[.5],[1.]]),np.zeros((2,1)),np.zeros((5,1)))
    np.testing.assert_allclose(got[:,0],.8**np.arange(1,6))


def test_invalid_prefix_rejected():
    draw=dict(b=np.zeros((1,3,1)),q=np.ones((1,1,1)),pi=np.ones((1,1)),beta=np.ones(1))
    with pytest.raises(ValueError):
        forecast(draw,np.zeros((1,1)),np.zeros((1,1)),np.zeros((3,1)))


def test_weak_limit_weights_are_probability_distribution():
    rng=np.random.default_rng(3)
    w=sample_global_weights(np.array([[40,2],[3,20]]),np.array([0,1,0]),np.array([.5,.5]),2.,20.,1.,rng)
    assert np.isfinite(w).all() and (w>=0).all()
    np.testing.assert_allclose(w.sum(),1.)


def test_sampler_smoke_and_reproducibility():
    rng=np.random.default_rng(9)
    y=rng.normal(size=(3,16,2))
    x=np.concatenate((np.ones((3,15,1)),rng.normal(size=(3,15,1)),y[:,:-1]),axis=-1)
    settings=dict(seed=42,alpha=2.,kappa=20.,gamma=1.,gibbs_sweeps=4,burn_in=2,retained_every=1)
    for hdp in (False,True):
        a,ta=fit(x,y[:,1:],k=2,hdp=hdp,settings=settings)
        b,tb=fit(x,y[:,1:],k=2,hdp=hdp,settings=settings)
        assert ta==tb and len(a)==2
        for aa,bb in zip(a,b):
            for key in aa:
                np.testing.assert_array_equal(aa[key],bb[key])
            assert (np.linalg.eigvalsh(aa['q'])>0).all()
            np.testing.assert_allclose(aa['pi'].sum(-1),1.)
            assert forecast(aa,y[0,:8],x[0,:8,1:2],x[0,8:12,1:2]).shape==(4,2)
