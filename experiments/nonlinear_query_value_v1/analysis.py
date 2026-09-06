"""Same-mean quadratic geometric readouts; NumPy only, no simulator refitting."""
from __future__ import annotations
import numpy as np


def queries():
    weights, info = [], []
    for i in range(8):
        for j in range(i + 2, 8):
            w = np.zeros(8); w[i] = 1; w[j] = -1
            weights.append(w)
            info.append(dict(name=f'distance_{i+2}_{j+2}', family='squared_distance', heldout=j-i >= 4))
    for center in range(1, 7):
        w = np.zeros(8); w[center-1:center+2] = [1, -2, 1]
        weights.append(w)
        info.append(dict(name=f'bend_{center+2}', family='bending_statistic', heldout=(center+2) % 2 == 1))
    return np.asarray(weights), info


def geometric(x, w):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-2:] != (8, 3) or not np.isfinite(x).all():
        raise ValueError('Expected finite free-marker coordinates (...,8,3)')
    return np.sum(np.einsum('qv,...vc->...qc', w, x, optimize=True)**2, axis=-1)


def matrix_power(a, exponent, floor=1e-14):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all() or not np.allclose(a, a.swapaxes(-1,-2), rtol=1e-8, atol=1e-12):
        raise ValueError('Covariance is not finite and symmetric')
    val, vec = np.linalg.eigh(a)
    if np.min(val) < -1e-10:
        raise ValueError('Covariance is not positive semidefinite')
    return np.einsum('...ia,...a,...ja->...ij', vec, np.maximum(val, floor)**exponent, vec, optimize=True)


def fit_coupling(errors, covariance):
    invroot = matrix_power(covariance, -0.5)
    z = np.einsum('...vab,...vb->...va', invroot, errors, optimize=True).reshape(-1, 24)
    s = z.T @ z / len(z)
    transform = np.zeros((24,24))
    for v in range(8):
        sl = slice(3*v,3*v+3)
        transform[sl,sl] = matrix_power(s[sl,sl], -0.5)
    r = transform @ s @ transform.T
    r = 0.5*r + 0.5*np.eye(24)  # Fixed shrinkage, never selected on targets.
    if min(np.linalg.eigvalsh(r)) < -1e-10:
        raise ValueError('Source coupling not PSD')
    for v in range(8):
        if not np.allclose(r[3*v:3*v+3,3*v:3*v+3], np.eye(3), atol=1e-6):
            raise ValueError('Source coupling failed native marginal preservation')
    return r


def corrections(covariance, w, coupling=None):
    covariance = np.asarray(covariance, dtype=np.float64)
    root = matrix_power(covariance, 0.5)
    trace = np.trace(covariance, axis1=-2, axis2=-1)
    native = np.einsum('qv,...v->...q', w*w, trace, optimize=True)
    if coupling is None:
        return native
    values = []
    for row in w:
        k = np.einsum('v,...vab->...avb', row, root).reshape(*root.shape[:-3],3,24)
        values.append(np.einsum('...ad,de,...ae->...', k, coupling, k, optimize=True))
    coupled = np.stack(values, axis=-1)
    if np.min(coupled) < -1e-10:
        raise ValueError('Negative quadratic variance')
    return np.maximum(coupled,0)


def features(g, initial, scale):
    n,h,q = g.shape
    t = np.broadcast_to(np.linspace(0,1,h)[None,:,None],(n,h,q))
    a = g/scale
    b = np.broadcast_to(initial[:,None,:],g.shape)/scale
    return np.stack([a,b,t,t*t,a*t],axis=-1)


def fit_ridge(x, y):
    x = np.asarray(x).reshape(-1,x.shape[-1]); y=np.asarray(y).reshape(-1)
    loc=x.mean(0); scale=np.maximum(x.std(0),1e-8)
    design=np.column_stack([np.ones(len(x)),(x-loc)/scale])
    penalty=np.eye(design.shape[1]); penalty[0,0]=0
    # Effective fit weight is eight independent trajectories, not all frames.
    normal=8*(design.T@design)/len(x)+penalty
    beta=np.linalg.solve(normal,8*(design.T@y)/len(x))
    return loc,scale,beta


def predict_ridge(fit,x):
    loc,scale,beta=fit
    return beta[0]+np.einsum('...d,d->...', (x-loc)/scale,beta[1:])


def build_readouts(source_mean, source_truth, source_cov, source_initial,
                   target_mean, target_cov, target_initial, *, heldout_mode=False):
    """No target truth argument: prediction construction cannot use it."""
    w,info=queries()
    gs=geometric(source_mean,w); ys=geometric(source_truth,w)
    gt=geometric(target_mean,w)
    si=geometric(source_initial,w); ti=geometric(target_initial,w)
    error=source_truth-source_mean
    projected=np.einsum('qv,ntvc->ntqc',w,error,optimize=True)
    centered=projected-projected.mean(axis=(0,1),keepdims=True)
    global_variance=(centered**2).sum(-1).mean(axis=(0,1))
    centered_h=projected-projected.mean(axis=0,keepdims=True)
    horizon_variance=(centered_h**2).sum(-1).mean(axis=0)
    symmetric_moment=(projected**2).sum(-1).mean(axis=(0,1))
    r=fit_coupling(error,source_cov)
    vn=corrections(target_cov,w)
    vc=corrections(target_cov,w,r)
    arms={'point_plugin':gt.copy(),
          'native_block_posterior':gt+vn,
          'source_coupled_posterior_extension':gt+vc,
          'empirical_global_centered':gt+global_variance,
          'empirical_horizon_centered':gt+horizon_variance,
          'empirical_symmetric_residual':gt+symmetric_moment}
    scales=np.empty(len(info)); bias=np.empty(len(info)); hbias=np.empty_like(gs[0]); ridge=np.empty_like(gt)
    fits={}
    for family in ('squared_distance','bending_statistic'):
        ids=np.array([i for i,v in enumerate(info) if v['family']==family])
        train=np.array([i for i in ids if not info[i]['heldout']]) if heldout_mode else ids
        scale=max(float(np.sqrt(np.mean(ys[:,:,train]**2))),1e-12)
        scales[ids]=scale
        residual=ys-gs
        if heldout_mode:
            bias[ids]=residual[:,:,train].mean()
            hbias[:,ids]=residual[:,:,train].mean(axis=(0,2))[:,None]
        else:
            bias[ids]=residual[:,:,ids].mean(axis=(0,1))
            hbias[:,ids]=residual[:,:,ids].mean(axis=0)
        xs=features(gs[:,:,train],si[:,train],scale)
        xt=features(gt[:,:,ids],ti[:,ids],scale)
        fit=fit_ridge(xs,residual[:,:,train]/scale)
        ridge[:,:,ids]=gt[:,:,ids]+scale*predict_ridge(fit,xt)
        fits[family]={'location':fit[0].tolist(),'scale':fit[1].tolist(),'beta':fit[2].tolist(),'query_scale_m2':scale}
    arms['source_query_bias']=gt+bias
    arms['source_query_horizon_bias']=gt+hbias
    arms['source_query_ridge']=ridge
    if any(not np.isfinite(a).all() for a in arms.values()):
        raise ValueError('Non-finite readout')
    return arms,scales,dict(coupling=r.tolist(),ridge=fits,query_bias=bias.tolist(),native_v_min=float(vn.min()),native_v_max=float(vn.max()),coupled_v_min=float(vc.min()),coupled_v_max=float(vc.max()))


def paired_bootstrap(left,right,seed=20260906,reps=10000):
    """left/right shape (objects, trajectories); objects fixed, not iid frames."""
    left=np.asarray(left); right=np.asarray(right)
    if left.shape!=right.shape or left.ndim!=2:
        raise ValueError('Expected matched object by trajectory arrays')
    diff=left-right
    rng=np.random.default_rng(seed)
    estimates=np.zeros(reps)
    for row in diff:
        ix=rng.integers(len(row),size=(reps,len(row)))
        estimates+=row[ix].mean(axis=1)/len(diff)
    return dict(mean_mse_difference=float(diff.mean()),
                bootstrap95=np.quantile(estimates,[0.025,0.975]).tolist(),
                wins=int(np.sum(diff<0)),ties=int(np.sum(diff==0)),losses=int(np.sum(diff>0)))
