"""Finite / weak-limit sticky-HDP autoregressive residual experts.

Exploratory, existing-recording experiment; not exact infinite-dimensional
inference and not a physical-regime-identification or calibration result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import invwishart


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_log(logp):
    logp = np.asarray(logp, dtype=float)
    norm = logsumexp(logp, axis=-1, keepdims=True)
    if not np.isfinite(norm).all():
        raise ValueError('No finite likelihood support')
    return np.exp(logp - norm)


def categorical(p, rng):
    p = p / p.sum(axis=-1, keepdims=True)
    cdf = p.cumsum(axis=-1)
    cdf[..., -1] = 1.0  # exact floating-point CDF endpoint
    return (rng.random(p.shape[:-1])[..., None] > cdf).sum(axis=-1)


def log_probability(p):
    result = np.full_like(p, -np.inf, dtype=float)
    np.log(p, out=result, where=p > 0)
    return result


def ffbs(emission, transition, initial, rng):
    n, t, k = emission.shape
    filtered = np.empty_like(emission)
    filtered[:, 0] = normalize_log(emission[:, 0] + log_probability(initial))
    for j in range(1, t):
        filtered[:, j] = normalize_log(emission[:, j] + log_probability(filtered[:, j-1] @ transition))
    z = np.empty((n, t), dtype=int)
    z[:, -1] = categorical(filtered[:, -1], rng)
    for j in range(t-2, -1, -1):
        z[:, j] = categorical(filtered[:, j] * transition[:, z[:, j+1]].T, rng)
    return z


def transition_counts(z, k):
    return np.bincount((z[:, :-1] * k + z[:, 1:]).ravel(), minlength=k*k).reshape(k, k)


def sample_global_weights(counts, z0, beta, alpha, kappa, gamma, rng):
    k = len(beta)
    tables = np.zeros((k, k), dtype=int)
    for j in range(k):
        for l in range(k):
            concentration = alpha * beta[l] + (kappa if j == l else 0.0)
            count = int(counts[j, l])
            if count:
                tables[j, l] = np.sum(rng.random(count) < concentration / (concentration + np.arange(count)))
        if tables[j, j]:
            tables[j, j] -= rng.binomial(tables[j, j], kappa / (alpha * beta[j] + kappa))
    return rng.dirichlet(gamma / k + tables.sum(axis=0) + np.bincount(z0, minlength=k))


def emissions(x, y, coefficients, covariance):
    result = np.empty(y.shape[:-1] + (len(coefficients),))
    d = y.shape[-1]
    for k, (b, q) in enumerate(zip(coefficients, covariance)):
        chol = np.linalg.cholesky(q)
        residual = y - x @ b
        whitened = solve_triangular(chol, residual.reshape(-1, d).T, lower=True).T
        result[..., k] = -.5 * (np.sum(whitened**2, axis=-1).reshape(y.shape[:-1]) + d*np.log(2*np.pi) + 2*np.log(np.diag(chol)).sum())
    return result


def sample_experts(x, y, z, k, rng):
    n, p, d = len(x), x.shape[-1], y.shape[-1]
    b0 = np.zeros((p, d))
    b0[-d:] = .98 * np.eye(d)
    k0 = np.eye(p)
    s0 = .01 * np.eye(d)
    nu0 = d + 2
    coefficients, covariances = [], []
    for j in range(k):
        select = z == j
        xx, yy = x[select], y[select]
        kn = k0 + xx.T @ xx
        chol = np.linalg.cholesky(kn)
        mn = np.linalg.solve(kn, k0 @ b0 + xx.T @ yy)
        residual = yy - xx @ mn
        prior_residual = mn - b0
        sn = s0 + residual.T @ residual + prior_residual.T @ k0 @ prior_residual
        q = np.atleast_2d(invwishart.rvs(df=nu0+len(xx), scale=sn, random_state=rng))
        b = mn + solve_triangular(chol.T, rng.normal(size=(p, d)), lower=False) @ np.linalg.cholesky(q).T
        coefficients.append(b)
        covariances.append(q)
    return np.asarray(coefficients), np.asarray(covariances)


def fit(x, y, *, k, hdp, settings):
    rng = np.random.default_rng(settings['seed'])
    n, t, d = y.shape
    xf, yf = x.reshape(-1, x.shape[-1]), y.reshape(-1, d)
    if k == 1:
        z = np.zeros((n, t), dtype=int)
    else:
        _, labels = kmeans2(yf, k, iter=15, minit='++', seed=settings['seed'])
        z = labels.reshape(n, t)
    beta = np.full(k, 1/k)
    saved, trace = [], []
    alpha, kappa, gamma = (settings[v] for v in ('alpha', 'kappa', 'gamma'))
    start = time.monotonic()
    for sweep in range(settings['gibbs_sweeps']):
        b, q = sample_experts(xf, yf, z.ravel(), k, rng)
        counts = transition_counts(z, k)
        if hdp:
            beta = sample_global_weights(counts, z[:, 0], beta, alpha, kappa, gamma, rng)
        transition = np.stack([rng.dirichlet(counts[j]+alpha*beta+kappa*np.eye(k)[j]) for j in range(k)])
        if k > 1:
            z = ffbs(emissions(x, y, b, q), transition, beta, rng)
        occupancy = np.bincount(z.ravel(), minlength=k)
        trace.append({'sweep': sweep+1, 'occupancy': occupancy.tolist(), 'beta': beta.tolist()})
        if sweep >= settings['burn_in'] and (sweep-settings['burn_in']) % settings['retained_every'] == 0:
            saved.append({'b': b.copy(), 'q': q.copy(), 'pi': transition.copy(), 'beta': beta.copy()})
        if (sweep+1) % 10 == 0:
            print(json.dumps({'k':k,'hdp':hdp,'sweep':sweep+1,'seconds':round(time.monotonic()-start,2),'occupancy':occupancy.tolist()}), flush=True)
    if not saved:
        raise ValueError('No posterior draws retained')
    return saved, trace


def forecast(draw, prefix_y, prefix_features, future_features):
    """Only observed prefix responses are accepted; no future-response argument.

    Conditional on each posterior parameter draw, unobserved future mode paths
    are marginalized exactly for the first moment of linear autoregressions.
    """
    b, q, transition, beta = (draw[v] for v in ('b','q','pi','beta'))
    prefix_y, prefix_features, future_features = map(np.asarray, (prefix_y,prefix_features,future_features))
    if prefix_y.ndim != 2 or len(prefix_y)<2 or len(prefix_y)!=len(prefix_features):
        raise ValueError('Invalid observed prefix')
    d, k = prefix_y.shape[-1], len(beta)
    weights = beta.copy()
    for j in range(1, len(prefix_y)):
        x = np.concatenate(([1.], prefix_features[j], prefix_y[j-1]))
        likelihood = emissions(x[None,None], prefix_y[j][None,None], b, q)[0,0]
        weights = normalize_log(log_probability(weights @ transition) + likelihood)
    weighted_means = weights[:,None] * prefix_y[-1]
    predictions = []
    for phi in future_features:
        next_weights = weights @ transition
        transferred_means = transition.T @ weighted_means
        offset = np.einsum('p,kpd->kd', np.r_[1.,phi], b[:,:-d])
        weighted_means = np.einsum('kd,kde->ke', transferred_means, b[:,-d:]) + next_weights[:,None]*offset
        weights = next_weights
        predictions.append(weighted_means.sum(axis=0))
    result = np.asarray(predictions)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite unguarded forecast; no silent clipping')
    return result


class Representation:
    def __init__(self, source, rank=6, feature_rank=4):
        f = source['features']
        self.floc = f.mean(axis=(0,1))
        fs = f.std(axis=(0,1))
        self.fscale = np.where(fs>1e-10, fs, 1.)
        ff = ((f-self.floc)/self.fscale).reshape(-1, f.shape[-1])
        _, _, v = np.linalg.svd(ff, full_matrices=False)
        self.fb = v[:feature_rank]
        self.fs = (ff @ self.fb.T).std(axis=0)
        self.fs = np.where(self.fs>1e-10, self.fs, 1.)
        residual = self.canonical_residual(source)
        self.mean = residual.mean(axis=(0,1))
        _, _, v = np.linalg.svd((residual-self.mean).reshape(-1,residual.shape[-1]), full_matrices=False)
        self.basis = v[:rank]
        self.scale = ((residual-self.mean) @ self.basis.T).std(axis=(0,1))
        self.scale = np.where(self.scale>1e-10,self.scale,1.)

    @staticmethod
    def canonical_residual(data):
        error = (data['truth']-data['local'])[:,:,2:-2]
        return np.einsum('ntvi,nij->ntvj',error,data['frames']).reshape(error.shape[0],error.shape[1],-1)

    def features(self,data):
        return ((data['features']-self.floc)/self.fscale) @ self.fb.T / self.fs

    def responses(self,data):
        return (self.canonical_residual(data)-self.mean) @ self.basis.T / self.scale

    def restore(self, coeff, last_residual, frame):
        last_reconstruction = ((last_residual-self.mean) @ self.basis.T) @ self.basis + self.mean
        complement = last_residual-last_reconstruction
        canonical = (coeff*self.scale) @ self.basis + self.mean + complement
        return canonical.reshape(len(coeff),-1,3) @ frame.T

    def save(self,path):
        np.savez_compressed(path,**self.__dict__)


def load_split(root, key):
    with np.load(root/(key+'.npz'), allow_pickle=False) as f:
        data = {k:f[k] for k in f.files}
    for k in ('features','truth','local','hybrid','frames'):
        if not np.isfinite(data[k]).all():
            raise ValueError('Nonfinite input '+k)
    return data


def evaluate(data, rep, models, settings):
    phi = rep.features(data)
    y = rep.responses(data)
    raw_residual = rep.canonical_residual(data)
    horizons = settings['horizons']
    hmax = max(horizons)
    metrics = {name:{str(h):[] for h in horizons} for name in ['hybrid','local','last_residual','local_last_residual',*models]}
    for i in range(len(y)):
        case = {name:{str(h):[] for h in horizons} for name in metrics}
        for t in settings['forecast_origins']:
            if t+hmax>len(y[i]):
                raise ValueError('Forecast window exceeds recording')
            truth = data['truth'][i,t:t+hmax]
            baseline = data['hybrid'][i,t:t+hmax]
            local = data['local'][i,t:t+hmax]
            predictions = {'hybrid':baseline,'local':local}
            predictions['last_residual'] = baseline.copy()
            predictions['last_residual'][:,2:-2] += (data['truth']-data['hybrid'])[i,t-1,2:-2]
            predictions['local_last_residual'] = local.copy()
            predictions['local_last_residual'][:,2:-2] += (data['truth']-data['local'])[i,t-1,2:-2]
            start = t-settings['prefix_frames']
            for name, draws in models.items():
                coefficient = np.mean([forecast(draw,y[i,start:t],phi[i,start:t],phi[i,t:t+hmax]) for draw in draws],axis=0)
                candidate = local.copy()
                candidate[:,2:-2] += rep.restore(coefficient,raw_residual[i,t-1],data['frames'][i])
                assert np.array_equal(candidate[:,[0,1,-2,-1]],local[:,[0,1,-2,-1]])
                predictions[name] = candidate
            for name, pred in predictions.items():
                for h in horizons:
                    case[name][str(h)].append(float(np.mean(np.abs(pred[:h]-truth[:h])))*1000)
        for name in metrics:
            for h in horizons:
                metrics[name][str(h)].append(float(np.mean(case[name][str(h)])))
    return metrics


def summarize(metrics):
    return {name:{h:float(np.mean(v)) for h,v in cells.items()} for name,cells in metrics.items()}


def bootstrap(delta):
    delta = np.asarray(delta)
    rng = np.random.default_rng(951)
    samples = delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    return {'mean_delta_mm':float(delta.mean()),'bootstrap_95_percent_mm':np.quantile(samples,[.025,.975]).tolist(),'wins':int(np.sum(delta<0)),'ties':int(np.sum(delta==0)),'n':len(delta)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    request=json.loads(args.request.read_text())
    settings=request['residual_model']
    (out/'protocol.json').write_text(json.dumps(request,indent=2)+'\n')
    source=load_split(args.input,'fit')
    validation=load_split(args.input,'validation')
    rep=Representation(source,settings['representation_rank'],settings['exogenous_feature_rank'])
    rep.save(out/'representation.npz')
    y=rep.responses(source)
    phi=rep.features(source)
    x=np.concatenate((np.ones((*y[:,1:].shape[:2],1)),phi[:,1:],y[:,:-1]),axis=-1)
    models={}
    for hdp, ks in ((False,settings['fixed_k']),(True,settings['hdp_weak_limit_k'])):
        for k in ks:
            name=('hdp' if hdp else 'finite')+'_k'+str(k)
            draws,trace=fit(x,y[:,1:],k=k,hdp=hdp,settings=settings)
            models[name]=draws
            np.savez_compressed(out/(name+'.npz'),**{f'{s}_{key}':a for s,draw in enumerate(draws) for key,a in draw.items()})
            (out/(name+'_trace.json')).write_text(json.dumps(trace,indent=2)+'\n')
    val=evaluate(validation,rep,models,settings)
    means=summarize(val)
    select=lambda names:min(names,key=lambda n:(means[n]['50'],n))
    selection={'best_baseline':select(['hybrid','local','last_residual','local_last_residual']),
        'finite':select([n for n in models if n.startswith('finite')]),
        'hdp':select([n for n in models if n.startswith('hdp')]),'validation_mean_l1_mm':means,
        'source_test_loaded_for_new_model_selection':False,
        'artifacts':{p.name:sha(p) for p in sorted(out.glob('*.npz'))}}
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    # The candidate family, settings, posterior draws and validation choice are
    # frozen before loading this already-open retrospective scoring partition.
    test=load_split(args.input,'source_test')
    metrics=evaluate(test,rep,models,settings)
    test_means=summarize(metrics)
    comparisons={}
    for h in settings['horizons']:
        cell=str(h)
        for a,b in ((selection['hdp'],selection['best_baseline']),(selection['finite'],selection['best_baseline']),(selection['hdp'],selection['finite'])):
            comparisons[f'{a}-minus-{b}/h{h}']=bootstrap(np.asarray(metrics[a][cell])-metrics[b][cell])
    report={'schema':'recorded-residual-regimes-result-v1','evidence_class':'retrospective-development-only',
        'names':test['names'].tolist(),'selection':selection,'test_mean_l1_mm':test_means,'test_per_trajectory_l1_mm':metrics,
        'paired_comparisons':comparisons,'settings':settings,'implementation_sha256':sha(__file__),
        'input_sha256':{p.name:sha(p) for p in sorted(args.input.glob('*.npz'))},
        'official_eval_read':False,'hardware_acquisition':False,
        'limitations':['single-chain exploratory MCMC; convergence not established','weak-limit truncation is 8, not exact infinite HDP','eight previously opened DLO1 trajectories, not fresh confirmation','known future clamped actions; passive observation prefixes','mean forecasts only; no calibration claim']}
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'selection':selection,'test_mean_l1_mm':test_means,'paired_comparisons':comparisons},indent=2),flush=True)


if __name__=='__main__':
    main()
