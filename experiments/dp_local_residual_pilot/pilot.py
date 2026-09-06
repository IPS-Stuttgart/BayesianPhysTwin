"""Development-only DP Gaussian-mixture regression; no target-conditioned gates.

Inference uses plug-in variational mixture moments, not a fully integrated
posterior predictive. No calibrated uncertainty claim is made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
import numpy as np
import scipy
import sklearn
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SETTINGS = dict(seed=20260907, pca_dimensions=8, temporal_stride=8,
                covariance_floor=1e-3, max_iter=150, n_init=2,
                shrinkages=[0.0, 0.25, 0.5, 1.0], dp_truncation=8,
                dp_alphas=[0.1, 1.0, 10.0], finite_counts=[1, 2, 4, 8],
                claim='historical-development-only; no official-eval access')


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def conditional_mean(x: np.ndarray, weights: np.ndarray, means: np.ndarray,
                     covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Condition a joint Gaussian mixture on x only, with stable input gating."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('predictor inputs must be finite and two-dimensional')
    p = x.shape[1]
    log_gate, predictions = [], []
    for weight, mu, cov in zip(weights, means, covariances):
        cxx = cov[:p, :p]
        chol = np.linalg.cholesky(cxx)
        delta = x - mu[:p]
        white = np.linalg.solve(chol, delta.T)
        log_gate.append(np.log(max(float(weight), 1e-300))
                        - np.log(np.diag(chol)).sum()
                        - 0.5*np.sum(white*white, axis=0))
        beta = np.linalg.solve(cxx, cov[:p, p:])
        predictions.append(mu[p:] + delta @ beta)
    log_gate = np.stack(log_gate, axis=1)
    gates = np.exp(log_gate - logsumexp(log_gate, axis=1, keepdims=True))
    return np.einsum('nk,nkd->nd', gates, np.stack(predictions, axis=1)), gates


class JointRegressor:
    def __init__(self, family: str, k: int, alpha: float = 1.0):
        self.family, self.k, self.alpha = family, k, alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'JointRegressor':
        self.x_scaler = StandardScaler().fit(x)
        self.pca = PCA(n_components=min(SETTINGS['pca_dimensions'], x.shape[1]),
                       whiten=True, svd_solver='full')
        q = self.pca.fit_transform(self.x_scaler.transform(x))
        self.y_scaler = StandardScaler().fit(y)
        joint = np.c_[q, self.y_scaler.transform(y)]
        kw = dict(n_components=self.k, covariance_type='full',
                  reg_covar=SETTINGS['covariance_floor'],
                  max_iter=SETTINGS['max_iter'], n_init=SETTINGS['n_init'],
                  tol=1e-3, random_state=SETTINGS['seed'])
        if self.family == 'finite':
            self.mix = GaussianMixture(**kw)
        else:
            self.mix = BayesianGaussianMixture(
                **kw, weight_concentration_prior_type=(
                    'dirichlet_process' if self.family == 'dp' else 'dirichlet_distribution'),
                weight_concentration_prior=(self.alpha if self.family == 'dp'
                                            else self.alpha / self.k),
                mean_precision_prior=0.1, degrees_of_freedom_prior=joint.shape[1]+2,
                covariance_prior=np.eye(joint.shape[1]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.mix.fit(joint)
        self.diagnostics = dict(converged=bool(self.mix.converged_),
                                iterations=int(self.mix.n_iter_),
                                weights=self.mix.weights_.tolist(),
                                occupied_weight_above_001=int((self.mix.weights_ > 0.01).sum()),
                                warnings=[str(w.message) for w in caught])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        q = self.pca.transform(self.x_scaler.transform(x))
        prediction, _ = conditional_mean(q, self.mix.weights_, self.mix.means_,
                                        self.mix.covariances_)
        return self.y_scaler.inverse_transform(prediction)


def self_test() -> dict:
    x = np.array([[-2.], [0.], [3.]])
    cov = np.array([[[1., 2.], [2., 5.]]])
    pred, gates = conditional_mean(x, np.ones(1), np.zeros((1, 2)), cov)
    np.testing.assert_allclose(pred[:, 0], 2*x[:, 0], atol=1e-12)
    np.testing.assert_allclose(gates.sum(axis=1), 1.)
    _, gates = conditional_mean(np.array([[1e6]]), np.array([0.4, 0.6]),
                               np.zeros((2, 2)), np.repeat(cov, 2, axis=0))
    np.testing.assert_allclose(gates.sum(axis=1), 1., atol=1e-3)
    # Constructed positive control, not deformable-data evidence.
    rng = np.random.default_rng(12)
    q = np.r_[rng.normal(-2, .3, (300, 1)), rng.normal(2, .3, (300, 1))]
    y = np.where(q < 0, 2*q+4, -2*q+4) + rng.normal(0, .05, q.shape)
    idx = rng.permutation(len(q)); tr, te = idx[:450], idx[450:]
    model = JointRegressor('dp', 8, 1.).fit(q[tr], y[tr])
    prediction = model.predict(q[te])
    beta = np.linalg.lstsq(np.c_[np.ones(len(tr)), q[tr]], y[tr], rcond=None)[0]
    linear = np.c_[np.ones(len(te)), q[te]] @ beta
    rmse = lambda a: float(np.sqrt(np.mean((a-y[te])**2)))
    assert rmse(prediction) < rmse(linear), 'DP failed constructed positive control'
    return dict(analytic_conditioning=True, finite_input_gates=True,
                positive_control_dp_rmse=rmse(prediction),
                positive_control_linear_rmse=rmse(linear),
                scope='synthetic implementation check only')


def fit_predict(features, residual, train, query, specification):
    """Fit independently per internal material node; equal time sampling per group."""
    family, k, alpha = specification
    predictions, diagnostics = [], []
    for node in range(features.shape[2]):
        x = features[train, ::SETTINGS['temporal_stride'], node].reshape(-1, features.shape[-1])
        y = residual[train, ::SETTINGS['temporal_stride'], node].reshape(-1, 3)
        q = features[query, :, node].reshape(-1, features.shape[-1])
        if family == 'trees':
            model = ExtraTreesRegressor(n_estimators=80, max_depth=12,
                                       min_samples_leaf=20, max_features=1.,
                                       random_state=SETTINGS['seed'], n_jobs=1).fit(x, y)
            diagnostics.append(dict(family='extra-trees'))
        else:
            model = JointRegressor(family, k, alpha).fit(x, y)
            diagnostics.append(model.diagnostics)
        predictions.append(model.predict(q).reshape(len(query), features.shape[1], 3))
    return np.stack(predictions, axis=2), diagnostics


def corrected(base, local, frames, shrink):
    result = base.copy()
    if shrink:
        result[:, :, 2:-2] += shrink*np.einsum('ntvj,nij->ntvi', local, frames)
    np.testing.assert_array_equal(result[:, :, [0, 1, -2, -1]], base[:, :, [0, 1, -2, -1]])
    return result


def errors(prediction, targets):
    return np.mean(np.abs(prediction-targets), axis=(1, 2, 3))*1000.


def run(input_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', SETTINGS)
    write_json(output/'self_test.json', self_test())
    data = np.load(input_path, allow_pickle=False)
    f, y, b, t, frames = (data[k] for k in ('features', 'residual', 'baseline', 'targets', 'frames'))
    names, roles = data['names'].astype(str), data['roles'].astype(str)
    groups = data['query_hashes'].astype(str)
    fit_indices = np.flatnonzero(roles == 'fit')
    val_indices = np.flatnonzero(roles == 'validation')
    if len(fit_indices) != 40 or len(val_indices) != 8:
        raise ValueError('pilot requires the frozen 40-fit/8-validation DLO2 split')
    unique = sorted(set(groups[fit_indices]))
    if len(unique) < 10:
        raise ValueError('insufficient independent causal-query groups for inner split')
    representatives = []
    y = y.copy()
    t = t.copy()
    for group in unique:
        members = fit_indices[groups[fit_indices] == group]
        representatives.append(int(members[0]))
        y[members[0]] = y[members].mean(axis=0)
        t[members[0]] = t[members].mean(axis=0)
    fit = np.asarray(representatives)
    rng = np.random.default_rng(SETTINGS['seed'])
    shuffled = rng.permutation(fit)
    n_inner = max(2, int(round(.2*len(fit))))
    inner_val, inner_fit = shuffled[:n_inner], shuffled[n_inner:]
    boundary = dict(fit_cases=40, fit_unique_query_groups=len(fit), validation_cases=8,
                    inner_fit_cases=names[inner_fit].tolist(),
                    inner_validation_cases=names[inner_val].tolist(),
                    fit_validation_identical_queries=len(set(groups[fit]) & set(groups[val_indices])),
                    official_eval_read=False, source_test_read=False,
                    fresh_confirmation=False)
    write_json(output/'data_boundary.json', boundary)
    bank = [('finite', k, 1.) for k in SETTINGS['finite_counts']]
    bank += [('bayes_finite', k, 1.) for k in [2, 4, 8]]
    bank += [('dp', 8, a) for a in SETTINGS['dp_alphas']]
    bank += [('trees', 0, 0.)]
    selections, inner_records = {}, []
    for specification in bank:
        started = time.monotonic()
        prediction, diagnostics = fit_predict(f, y, inner_fit, inner_val, specification)
        scores = []
        for shrink in SETTINGS['shrinkages']:
            point = corrected(b[inner_val], prediction, frames[inner_val], shrink)
            scores.append((float(errors(point, t[inner_val]).mean()), shrink))
        loss, shrink = min(scores)
        record = dict(specification=list(specification), inner_l1_mm=loss, shrinkage=shrink,
                      grid=[dict(l1_mm=e, shrinkage=s) for e, s in scores],
                      fit_diagnostics=diagnostics, seconds=time.monotonic()-started)
        inner_records.append(record)
        print('INNER', json.dumps({k:v for k,v in record.items() if k != 'fit_diagnostics'}), flush=True)
        write_json(output/'inner_scores.json', inner_records)
        family = specification[0]
        if family not in selections or loss < selections[family]['inner_l1_mm']:
            selections[family] = record
    # Freeze choices before validation scoring; targets are loaded but never used for selection.
    write_json(output/'selection.json', selections)
    predictions = dict(baseline=b[val_indices].copy(), current_ridge=data['current_predictions'][val_indices].copy())
    final_diagnostics = {}
    refit_cache = {}
    for family, selected in selections.items():
        spec = tuple(selected['specification'])
        local, diag = fit_predict(f, y, fit, val_indices, spec)
        refit_cache[spec] = (local, diag)
        predictions[family] = corrected(b[val_indices], local, frames[val_indices], selected['shrinkage'])
        final_diagnostics[family] = diag
    # Predeclared reference, independent of inner selection or validation outcomes.
    spec = ('dp', 8, 1.)
    if spec not in refit_cache:
        refit_cache[spec] = fit_predict(f, y, fit, val_indices, spec)
    local, diag = refit_cache[spec]
    predictions['dp_fixed_alpha1_shrink025'] = corrected(b[val_indices], local, frames[val_indices], .25)
    final_diagnostics['dp_fixed_alpha1_shrink025'] = diag
    np.savez_compressed(output/'validation_predictions.npz', names=names[val_indices], **predictions)
    per_case = {key: errors(pred, t[val_indices]).tolist() for key, pred in predictions.items()}
    reference = np.array(per_case['current_ridge'])
    rows = {}
    for key, value in per_case.items():
        a = np.array(value)
        rows[key] = dict(mean_l1_mm=float(a.mean()),
                         relative_improvement_vs_current_pct=float(100*(1-a.mean()/reference.mean())),
                         wins_vs_current=int((a < reference-1e-9).sum()),
                         ties_vs_current=int(np.isclose(a, reference, atol=1e-9, rtol=0).sum()),
                         worst_case_ratio_vs_current=float(np.max(a/reference)), per_case_l1_mm=value)
    result = dict(settings=SETTINGS, boundary=boundary, selection=selections,
                  validation_names=names[val_indices].tolist(), results=rows,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__,
                               scipy=scipy.__version__, sklearn=sklearn.__version__),
                  input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  covariance_calibration_claim=False, posterior_predictive='plug-in mixture conditional mean')
    write_json(output/'result.json', result)
    write_json(output/'final_fit_diagnostics.json', final_diagnostics)
    print('FINAL', json.dumps(rows, sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.self_test:
            print(json.dumps(self_test(), indent=2))
        elif args.input is None or args.output is None:
            parser.error('--input and --output are required unless --self-test is used')
        else:
            run(args.input, args.output)
