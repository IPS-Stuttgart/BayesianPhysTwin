"""Source-only sticky weak-limit HDP-AR(2) prototype.

Blocked Gibbs: matrix-normal/inverse-Wishart emissions, FFBS state paths,
Chinese-restaurant auxiliary tables, and sticky diagonal-table thinning.
The finite controls use independent symmetric Dirichlet transition priors.
A finite truncation and short chains are NOT exact infinite-HDP inference.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.stats import invwishart


@dataclass
class Draw:
    coef: np.ndarray             # [K, 2*d+c+1, d]
    covariance: np.ndarray       # [K, d, d]
    transition: np.ndarray       # [K, K]
    initial: np.ndarray          # [K]
    beta: np.ndarray             # [K]
    occupied: int
    log_likelihood: float


def _finite(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f'{name} contains non-finite values')
    return a


def design(d: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d, control = _finite(d, 'states'), _finite(control, 'controls')
    if d.ndim != 3 or control.ndim != 3 or d.shape[:2] != control.shape[:2] or d.shape[1] < 3:
        raise ValueError('Expected aligned [sequence,time,dimension] arrays, time >= 3')
    x = np.concatenate((d[:, 1:-1], d[:, :-2], control[:, 2:], np.ones((*d[:, 2:].shape[:2], 1))), axis=-1)
    return x, d[:, 2:]


def log_emissions(x: np.ndarray, y: np.ndarray, draw: Draw) -> np.ndarray:
    out = np.empty((*y.shape[:2], len(draw.coef)))
    for k, (w, q) in enumerate(zip(draw.coef, draw.covariance)):
        error = y - x @ w
        inv = np.linalg.inv(q)
        out[..., k] = -0.5 * (y.shape[-1] * np.log(2*np.pi) + np.linalg.slogdet(q)[1]
                              + np.einsum('...i,ij,...j->...', error, inv, error))
    return out


def forward(loge: np.ndarray, transition: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized log filtering probabilities and per-sequence log evidence."""
    logp = np.log(np.maximum(transition, 1e-300))
    f = np.empty_like(loge)
    a = loge[:, 0] + np.log(np.maximum(initial, 1e-300))
    norm = logsumexp(a, axis=-1)
    f[:, 0] = a - norm[:, None]
    evidence = norm.copy()
    for t in range(1, loge.shape[1]):
        a = loge[:, t] + logsumexp(f[:, t-1, :, None] + logp[None], axis=1)
        norm = logsumexp(a, axis=-1)
        f[:, t] = a - norm[:, None]
        evidence += norm
    return f, evidence


def categorical(prob: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    prob = prob / prob.sum(axis=-1, keepdims=True)
    u = rng.random(prob.shape[:-1])[..., None]
    return np.minimum((u > np.cumsum(prob, axis=-1)).sum(axis=-1), prob.shape[-1]-1)


def sample_paths(loge: np.ndarray, transition: np.ndarray, initial: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, float]:
    f, evidence = forward(loge, transition, initial)
    z = np.empty(loge.shape[:2], dtype=np.int64)
    z[:, -1] = categorical(np.exp(f[:, -1]), rng)
    for t in range(z.shape[1]-2, -1, -1):
        a = f[:, t] + np.log(np.maximum(transition[:, z[:, t+1]].T, 1e-300))
        z[:, t] = categorical(np.exp(a-logsumexp(a, axis=-1, keepdims=True)), rng)
    return z, float(evidence.sum())


def transition_counts(z: np.ndarray, k: int) -> np.ndarray:
    return np.bincount((z[:, :-1]*k+z[:, 1:]).ravel(), minlength=k*k).reshape(k, k)


def update_beta(counts: np.ndarray, beta: np.ndarray, alpha: float, sticky: float,
                gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Exact auxiliary update for the finite weak-limit hierarchical prior."""
    k = len(beta)
    global_tables = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            n = int(counts[i, j])
            a = alpha*beta[j] + (sticky if i == j else 0.0)
            tables = int(np.sum(rng.random(n) < a/(a+np.arange(n))))
            # Sticky self-transition tables must not inflate global beta.
            global_tables[i, j] = (rng.binomial(tables, alpha*beta[j]/a)
                                   if i == j else tables)
    return rng.dirichlet(gamma/k + global_tables.sum(axis=0))


def fit(d: np.ndarray, control: np.ndarray, *, k: int, hdp: bool, seed: int,
        iterations: int = 60, burn: int = 30, thin: int = 5,
        alpha: float = 5.0, sticky: float = 25.0, gamma: float = 1.0,
        ridge: float = 5.0) -> tuple[list[Draw], dict]:
    if k < 1 or not 0 <= burn < iterations or thin < 1 or min(alpha, gamma, ridge) <= 0 or sticky < 0:
        raise ValueError('Invalid sampler configuration')
    x, y = design(d, control)
    n, t, dim = y.shape
    p = x.shape[-1]
    rng = np.random.default_rng(seed)
    xf, yf = x.reshape(-1, p), y.reshape(-1, dim)
    # K-means initialization on TRAINING residual states, not evaluation outcomes.
    features = np.concatenate((d[:, 1:-1], 5*(y-d[:, 1:-1])), axis=-1).reshape(n*t, -1)
    centers = features[rng.choice(len(features), k, replace=False)].copy()
    for _ in range(12):
        zflat = ((features[:, None]-centers[None])**2).sum(axis=-1).argmin(axis=1)
        for j in range(k):
            points = features[zflat == j]
            centers[j] = points.mean(axis=0) if len(points) else features[rng.integers(len(features))]
    z = zflat.reshape(n, t)
    beta = np.full(k, 1/k)
    v0 = ridge*np.eye(p)
    m0 = np.zeros((p, dim))
    m0[:dim] = 1.8*np.eye(dim)
    m0[dim:2*dim] = -0.8*np.eye(dim)
    s0 = 0.02*np.eye(dim)
    samples, trace = [], []
    for iteration in range(iterations):
        w, cov = [], []
        for j in range(k):
            xx, yy = xf[z.ravel() == j], yf[z.ravel() == j]
            precision = v0 + xx.T @ xx
            mean = np.linalg.solve(precision, v0 @ m0 + xx.T @ yy)
            residual = yy - xx @ mean
            dm = mean-m0
            scale = s0 + residual.T @ residual + dm.T @ v0 @ dm
            scale = (scale + scale.T)/2
            q = np.atleast_2d(invwishart.rvs(df=dim+2+len(xx), scale=scale, random_state=rng))
            row_chol = np.linalg.cholesky(np.linalg.inv(precision))
            w.append(mean + row_chol @ rng.standard_normal((p, dim)) @ np.linalg.cholesky(q).T)
            cov.append(q)
        counts = transition_counts(z, k)
        if hdp:
            beta = update_beta(counts, beta, alpha, sticky, gamma, rng)
        trans = np.stack([rng.dirichlet(counts[j]+alpha*beta+sticky*np.eye(k)[j]) for j in range(k)])
        initial = rng.dirichlet(np.ones(k)/k + np.bincount(z[:, 0], minlength=k))
        draw = Draw(np.asarray(w), np.asarray(cov), trans, initial, beta.copy(), 0, 0.0)
        z, likelihood = sample_paths(log_emissions(x, y, draw), trans, initial, rng)
        draw.occupied = int(np.unique(z).size)
        draw.log_likelihood = likelihood
        trace.append({'iteration': iteration, 'occupied': draw.occupied, 'log_likelihood': likelihood})
        if iteration >= burn and (iteration-burn) % thin == 0:
            samples.append(draw)
    return samples, {'seed': seed, 'k': k, 'hdp': hdp, 'trace': trace,
                     'posterior_draws': len(samples), 'mcmc_convergence_established': False}


def forecast(draws: list[Draw], prefix: np.ndarray, past_controls: np.ndarray,
             future_controls: np.ndarray, *, hard: bool = False) -> np.ndarray:
    """Posterior mean via exact conditional switching-linear first moments.

Only a PREFIX of observed residuals is accepted; future controls are known
baseline/control features. Parameters are averaged without aligning labels.
"""
    x, y = design(prefix, past_controls)
    future_controls = _finite(future_controls, 'future controls')
    if not draws or future_controls.ndim != 3 or future_controls.shape[0] != prefix.shape[0]:
        raise ValueError('Invalid forecast arguments')
    dim = prefix.shape[-1]
    result = []
    for draw in draws:
        f, _ = forward(log_emissions(x, y, draw), draw.transition, draw.initial)
        prob = np.exp(f[:, -1])
        if hard:
            prob = np.eye(len(draw.coef))[prob.argmax(axis=1)]
        state = np.concatenate((prefix[:, -1], prefix[:, -2]), axis=-1)
        mass_state = prob[..., None]*state[:, None]
        predictions = []
        for t in range(future_controls.shape[1]):
            transition = draw.transition
            if hard:
                transition = np.eye(len(draw.coef))[transition.argmax(axis=1)]
            next_prob = prob @ transition
            incoming = np.einsum('nki,kj->nji', mass_state, transition)
            new = np.einsum('nkp,kpd->nkd', incoming, draw.coef[:, :2*dim])
            inputs = np.concatenate((future_controls[:, t], np.ones((len(prefix), 1))), axis=-1)
            new += next_prob[..., None]*np.einsum('np,kpd->nkd', inputs, draw.coef[:, 2*dim:])
            mass_state = np.concatenate((new, incoming[..., :dim]), axis=-1)
            prob = next_prob
            predictions.append(new.sum(axis=1))
        result.append(np.stack(predictions, axis=1))
    return _finite(np.mean(result, axis=0), 'forecast')


def sample_forecast(draws: list[Draw], prefix: np.ndarray, past_controls: np.ndarray,
                    future_controls: np.ndarray, *, samples: int, seed: int) -> np.ndarray:
    """Coherent state/parameter/process-uncertainty trajectory samples."""
    rng = np.random.default_rng(seed)
    x, y = design(prefix, past_controls)
    filters = [np.exp(forward(log_emissions(x, y, dr), dr.transition, dr.initial)[0][:, -1]) for dr in draws]
    n, horizon, _ = future_controls.shape
    out = []
    for _ in range(samples):
        j = int(rng.integers(len(draws)))
        dr = draws[j]
        z = categorical(filters[j], rng)
        prev, cur = prefix[:, -2].copy(), prefix[:, -1].copy()
        path = []
        for t in range(horizon):
            z = categorical(dr.transition[z], rng)
            xx = np.concatenate((cur, prev, future_controls[:, t], np.ones((n, 1))), axis=-1)
            mean = np.einsum('np,npd->nd', xx, dr.coef[z])
            noise = np.einsum('nij,nj->ni', np.linalg.cholesky(dr.covariance[z]), rng.standard_normal(cur.shape))
            prev, cur = cur, mean+noise
            path.append(cur.copy())
        out.append(np.stack(path, axis=1))
    return _finite(np.asarray(out), 'predictive samples')
