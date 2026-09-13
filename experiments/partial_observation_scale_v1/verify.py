"""Artifact verifier independent of the experiment implementation (no run import)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    a = p.parse_args()
    root = a.output
    result = json.loads((root / 'result.json').read_text())
    selection = json.loads((root / 'scale_selection.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    rows = json.loads((root / 'per_case.json').read_text())['records']
    assert len(rows) == 1176 == result['record_count']
    keys = {(r['dlo'], r['trajectory'], r['mask'], r['arm']) for r in rows}
    assert len(keys) == len(rows)
    indexed = {(r['dlo'], r['trajectory'], r['mask'], r['arm']): r for r in rows}
    max_density_error, max_residual_error, mean_error, interval_error = 0., 0., 0., 0.
    for dlo in ('DLO4', 'DLO5'):
        cfg = selection['objects'][dlo]
        parts = cfg['partitions']
        assert [len(parts[k]) for k in ('fit','select','calibrate')] == [32,12,12]
        assert len(set(sum(parts.values(), []))) == 56
        model_path = root / (dlo + '_scale_model.npz')
        assert digest(model_path) == cfg['model_sha256']
        with np.load(model_path, allow_pickle=False) as model:
            mean, cov = model['mean'], model['covariance']
        data_path = root / (dlo + '_scoring_inputs.npz')
        assert digest(data_path) == result['scoring_inputs_sha256'][dlo]
        with np.load(a.scratch / (dlo + '_target.npz'), allow_pickle=False) as raw:
            names = list(raw['names'].astype(str))
            residual = (raw['truth'][:,:,2:10].astype(float) - raw['prediction'][:,:,2:10].astype(float)).reshape(14,498,24)
        assert not set(names) & set(sum(parts.values(), []))
        joint = np.concatenate((residual[:,protocol['anchors']], residual[:,np.array(protocol['anchors'])+30]), axis=-1)
        with np.load(data_path, allow_pickle=False) as data:
            assert names == list(data['names'].astype(str))
            for mask, nodes in protocol['masks'].items():
                obs = [3*n+c for n in nodes for c in range(3)]
                hid = [24+3*n+c for n in range(8) if n not in nodes for c in range(3)]
                oo = cov[np.ix_(obs,obs)] + np.eye(len(obs))*cfg['base']['nugget']
                ho = cov[np.ix_(hid,obs)]
                gain = np.linalg.solve(oo, ho.T).T
                schur = cov[np.ix_(hid,hid)] - gain @ ho.T
                schur = (schur + schur.T)/2
                observed = joint[...,obs] - mean[obs]
                error = joint[...,hid] - mean[hid] - observed @ gain.T
                max_residual_error = max(max_residual_error, float(np.max(np.abs(error-data[mask+'__error']))))
                flat = observed.reshape(-1,len(obs))
                energy = np.einsum('ni,in->n', flat, np.linalg.solve(oo,flat.T)).reshape(14,18)
                bayes = cfg['configs']['bayesian_scale_t']
                alpha = (bayes['nu'] + len(obs))/2
                beta = (bayes['scale']*(bayes['nu']-2)+energy)/2
                for arm, conf in cfg['configs'].items():
                    if conf['family'] == 'by_dimension':
                        conf = conf['configs'][str(len(obs))]
                    if conf['family'] == 'bayes':
                        df, multiplier = 2*alpha, beta/alpha
                    elif conf['family'] == 'moment':
                        df, multiplier = None, conf['scale']*beta/(alpha-1)
                    else:
                        multiplier = conf['scale']*((1-conf['beta'])+conf['beta']*energy/len(obs))
                        multiplier = np.maximum(multiplier,1e-12)
                        df = conf.get('nu')
                        if df is not None:
                            multiplier *= (df-2)/df
                    standardized = error/np.sqrt(multiplier[...,None])
                    distribution = (stats.multivariate_normal(cov=schur) if df is None else stats.multivariate_t(shape=schur,df=df))
                    nll = (-distribution.logpdf(standardized) + .5*len(hid)*np.log(multiplier))/len(hid)
                    for i,name in enumerate(names):
                        expected = indexed[(dlo,name,mask,arm)]['nll']
                        max_density_error = max(max_density_error, abs(float(nll[i].mean())-expected))
    for arm, metrics in result['means'].items():
        for metric, value in metrics.items():
            reproduced = np.mean([r[metric] for r in rows if r['arm']==arm])
            mean_error = max(mean_error, abs(float(reproduced)-value))
    for arm, comparisons in result['comparisons'].items():
        for metric, stored in comparisons.items():
            diffs=[]
            for dlo in ('DLO4','DLO5'):
                names=sorted({r['trajectory'] for r in rows if r['dlo']==dlo})
                values=[]
                for name in names:
                    av=lambda aa: np.mean([r[metric] for r in rows if r['dlo']==dlo and r['trajectory']==name and r['arm']==aa])
                    values.append(av('bayesian_scale_t')-av(arm))
                diffs.append(np.array(values))
            rng=np.random.default_rng(20260906)
            boot=np.mean([d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1) for d in diffs],axis=0)
            interval_error=max(interval_error,float(np.max(np.abs(np.quantile(boot,[.025,.975])-stored['interval95']))))
    assert max_residual_error < 1e-12
    assert max_density_error < 1e-8
    assert mean_error < 1e-10
    assert interval_error < 1e-10
    receipt={'status':'verified','record_count':len(rows),'max_residual_error_m':max_residual_error,
             'max_scipy_density_error':max_density_error,'max_aggregate_error':mean_error,
             'max_paired_interval_error':interval_error,'result_sha256':digest(root/'result.json'),
             'scope':'independent residual reconstruction, SciPy densities, scalar aggregation and paired intervals; no simulator rerun or refit'}
    (root/'verification.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT_VERIFICATION '+json.dumps(receipt,sort_keys=True),flush=True)
    print('FULL_RESULT_JSON '+json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    main()
