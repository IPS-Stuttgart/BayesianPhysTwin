"""Source-only DP residual regression, not the native DEFORM hybrid benchmark.

The archive contains normalized residuals relative to a kinematic predictor.
At prediction time expert weights use FEATURES ONLY. Source response values
may inform training assignments, never prediction-time assignments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ARCHIVE_SHA = "fdb35f680e4cf685303f841b8974c16eb9301ad52cbdc8f5af0d87a9dfc358ee"
RESULT_SHA = "77332c323ddd09d945f65f57e3b83a12deedd9ea94509e43c13c9ca87f3cc353"
MODEL_SHA = "a43aed43cd563ee47358e48cab84829dc7eebc77d97725721a11b228f3b6b7f0"
METRIC = "nrmse_percent_span"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def validate_xy(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("unaligned feature/response matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite data")


@dataclass
class Reduction:
    scaler: StandardScaler
    xpca: PCA
    ypca: PCA
    yscale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, rank: int = 12):
        validate_xy(x, y)
        scaler = StandardScaler().fit(x)
        xpca = PCA(
            n_components=min(rank, x.shape[1], len(x) - 1),
            whiten=True,
            svd_solver="full",
        )
        xpca.fit(scaler.transform(x))
        ypca = PCA(
            n_components=min(rank, y.shape[1], len(y) - 1),
            svd_solver="full",
        ).fit(y)
        scale = np.maximum(np.sqrt(ypca.explained_variance_), 1e-8)
        return cls(scaler, xpca, ypca, scale)

    def x(self, x):
        return self.xpca.transform(self.scaler.transform(x))

    def y(self, y):
        return self.ypca.transform(y) / self.yscale

    def decode(self, y):
        return self.ypca.inverse_transform(y * self.yscale)


@dataclass
class ConditionalMixture:
    reduction: Reduction
    estimator: BayesianGaussianMixture
    diagnostics: dict

    @classmethod
    def fit(cls, x, y, reduction, family, components, concentration, seed=1729):
        if family not in ("finite", "dp"):
            raise ValueError("unknown mixture family")
        joint = np.concatenate((reduction.x(x), reduction.y(y)), axis=1)
        prior = "dirichlet_process" if family == "dp" else "dirichlet_distribution"
        # Match total concentration, not finite per-component concentration.
        alpha = concentration if family == "dp" else concentration / components
        estimator = BayesianGaussianMixture(
            n_components=components,
            covariance_type="full",
            weight_concentration_prior_type=prior,
            weight_concentration_prior=alpha,
            mean_precision_prior=0.1,
            reg_covar=1e-4,
            max_iter=500,
            tol=1e-4,
            n_init=1,
            random_state=seed,
        )
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(joint)
        diagnostics = {
            "family": family,
            "components_cap": components,
            "concentration": concentration,
            "converged": bool(estimator.converged_),
            "iterations": int(estimator.n_iter_),
            "weights": estimator.weights_.tolist(),
            "occupied_gt_0p01": int(np.sum(estimator.weights_ > 0.01)),
            "warnings": [str(w.message) for w in caught],
            "fit_seconds": time.perf_counter() - start,
        }
        return cls(reduction, estimator, diagnostics)

    def feature_weights(self, x: np.ndarray) -> np.ndarray:
        """Use the feature marginal, never a future response or residual."""
        z = self.reduction.x(x)
        d = z.shape[1]
        logits = []
        components = zip(
            self.estimator.weights_,
            self.estimator.means_,
            self.estimator.covariances_,
            strict=True,
        )
        for weight, mu, cov in components:
            factor = cho_factor(cov[:d, :d], lower=True, check_finite=False)
            delta = z - mu[:d]
            inv_delta = cho_solve(factor, delta.T, check_finite=False).T
            logdet = 2 * np.log(np.diag(factor[0])).sum()
            quadratic = np.sum(delta * inv_delta, axis=1)
            log_density = -0.5 * (quadratic + logdet + d * np.log(2 * np.pi))
            logits.append(np.log(max(weight, 1e-300)) + log_density)
        logs = np.stack(logits, axis=1)
        return np.exp(logs - logsumexp(logs, axis=1, keepdims=True))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Conditional mixture mean using fitted covariance point estimates."""
        z = self.reduction.x(x)
        d = z.shape[1]
        conditional = []
        for mu, cov in zip(
            self.estimator.means_, self.estimator.covariances_, strict=True
        ):
            factor = cho_factor(cov[:d, :d], lower=True, check_finite=False)
            inv_delta = cho_solve(factor, (z - mu[:d]).T, check_finite=False).T
            conditional.append(mu[d:] + inv_delta @ cov[:d, d:])
        means = np.stack(conditional, axis=1)
        latent = np.einsum("nk,nkd->nd", self.feature_weights(x), means)
        return self.reduction.decode(latent)


@dataclass
class FullFeatureExperts:
    gate: ConditionalMixture
    scaler: StandardScaler
    global_model: Ridge
    deltas: list

    @classmethod
    def fit(cls, gate, x, y, scaler, global_model, alpha):
        """Global ridge plus regularized expert-specific deviations.

        SOURCE responses inform training assignments. Prediction gates only
        receive features. Regressions retain all 81 features and 600 outputs.
        Coefficients and gate parameters are plug-in estimates, not a complete
        joint Bayesian posterior or a sticky HDP switching model.
        """
        joint = np.concatenate((gate.reduction.x(x), gate.reduction.y(y)), axis=1)
        assignments = gate.estimator.predict_proba(joint)
        design = scaler.transform(x)
        residual = y - global_model.predict(design)
        deltas = []
        for k in range(assignments.shape[1]):
            model = Ridge(alpha=alpha).fit(
                design, residual, sample_weight=assignments[:, k]
            )
            deltas.append(model)
        return cls(gate, scaler, global_model, deltas)

    def predict(self, x):
        design = self.scaler.transform(x)
        delta = np.stack([m.predict(design) for m in self.deltas], axis=1)
        correction = np.einsum("nk,nkd->nd", self.gate.feature_weights(x), delta)
        return self.global_model.predict(design) + correction


def score(y, pred, groups):
    diff = y - pred
    rows = []
    for group in sorted(set(groups.tolist())):
        error = diff[groups == group]
        rows.append(
            {
                "trajectory": group,
                METRIC: float(100 * np.sqrt(np.mean(error**2))),
                "nl1_percent_span": float(100 * np.mean(np.abs(error))),
            }
        )
    return {
        METRIC: float(np.mean([r[METRIC] for r in rows])),
        "nl1_percent_span": float(np.mean([r["nl1_percent_span"] for r in rows])),
        "pooled_nrmse_percent_span": float(100 * np.sqrt(np.mean(diff**2))),
        "per_trajectory": rows,
    }


def nearest_prediction(xfit, yfit, xquery, neighbors, temperature):
    d = pairwise_distances(xquery, xfit, metric="sqeuclidean") / xfit.shape[1]
    order = np.argsort(d, axis=1, kind="stable")[:, : min(neighbors, len(xfit))]
    ds = np.take_along_axis(d, order, axis=1)
    bandwidth = np.maximum(np.median(ds, axis=1) * temperature, 1e-12)
    logits = -(ds - ds[:, 0:1]) / bandwidth[:, None]
    w = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return np.einsum("nk,nkd->nd", w, yfit[order])


def load_source(source: Path):
    model = source / "source_model.npz"
    if digest(model) != MODEL_SHA:
        raise ValueError("source model SHA-256 mismatch")
    metadata = source / "source_result.json"
    if digest(metadata) != RESULT_SHA:
        raise ValueError("source metadata SHA-256 mismatch")
    result = json.loads(metadata.read_text())
    panels = {}
    with np.load(model, allow_pickle=False) as archive:
        for dlo in ("DLO4", "DLO5"):
            names = sorted(result["train_manifest"][dlo])
            if len(names) != 56:
                raise ValueError("expected 56 complete source trajectories")
            x = archive[dlo.lower() + "_features"]
            y = archive[dlo.lower() + "_residuals"]
            if x.shape != (1064, 81) or y.shape != (1064, 600):
                raise ValueError("unexpected source representation")
            validate_xy(x, y)
            panels[dlo] = (
                x.copy(),
                y.copy(),
                np.repeat(names, 19),
                result["dlos"][dlo]["partition"],
            )
    return panels


def partitions(names, historical, folds):
    if folds == 1:
        yield (
            "historical-39-9-8",
            historical["fit"],
            historical["calibration"],
            historical["source_test"],
        )
        return
    unique = np.array(sorted(set(names)))
    rng = np.random.default_rng(27091)
    batches = np.array_split(rng.permutation(unique), folds)
    for i, test in enumerate(batches):
        remaining = np.setdiff1d(unique, test)
        rng = np.random.default_rng(9127 + i)
        remaining = rng.permutation(remaining)
        yield (
            f"grouped-{i}",
            remaining[9:].tolist(),
            remaining[:9].tolist(),
            test.tolist(),
        )


def run_panel(dlo, x, y, groups, split, output):
    fold, fit_names, val_names, test_names = split
    a, b, c = map(set, (fit_names, val_names, test_names))
    if a & b or a & c or b & c or a | b | c != set(groups.tolist()):
        raise ValueError("incomplete or overlapping trajectory partition")
    fit, validation, test = [np.isin(groups, list(s)) for s in (a, b, c)]
    xf, yf = x[fit], y[fit]
    xv, yv = x[validation], y[validation]
    xq = x[test]
    scaler = StandardScaler().fit(xf)
    zf = scaler.transform(xf)
    red = Reduction.fit(xf, yf)
    contenders = {}
    validation_table = []

    def add(family, settings, predictor, diagnostics=None):
        metrics = score(yv, predictor(xv), groups[validation])
        entry = {
            "family": family,
            "settings": settings,
            "validation_nrmse_percent_span": metrics[METRIC],
        }
        if diagnostics is not None:
            entry["fit"] = diagnostics
        validation_table.append(entry)
        previous = contenders.get(family)
        key = "validation_nrmse_percent_span"
        if previous is None or entry[key] < previous[0][key]:
            contenders[family] = (entry, predictor)

    add("kinematic", {}, lambda q: np.zeros((len(q), y.shape[1])))
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha).fit(zf, yf)
        add(
            "ridge",
            {"alpha": alpha},
            lambda q, m=model: m.predict(scaler.transform(q)),
        )
    for k in (4, 8, 16, 32):
        for temp in (0.5, 1.0, 2.0):
            add(
                "kernel_analog",
                {"neighbors": k, "temperature": temp},
                lambda q, k=k, temp=temp: nearest_prediction(
                    zf, yf, scaler.transform(q), k, temp
                ),
            )
    dfit = pairwise_distances(zf, zf, metric="sqeuclidean") / zf.shape[1]
    for gamma in (0.1, 1.0, 10.0):
        kernel = np.exp(-gamma * dfit)
        for alpha in (0.1, 1.0, 10.0):
            dual = cho_solve(
                cho_factor(kernel + alpha * np.eye(len(kernel)), lower=True), yf
            )

            def rbf_predict(q, gamma=gamma, dual=dual):
                dist = (
                    pairwise_distances(scaler.transform(q), zf, metric="sqeuclidean")
                    / zf.shape[1]
                )
                return np.exp(-gamma * dist) @ dual

            add("rbf_ridge", {"gamma": gamma, "alpha": alpha}, rbf_predict)
    best_alpha = contenders["ridge"][0]["settings"]["alpha"]
    global_ridge = Ridge(alpha=best_alpha).fit(zf, yf)
    single = ConditionalMixture.fit(xf, yf, red, "finite", 1, 1.0)
    add("single_gaussian", {"components": 1}, single.predict, single.diagnostics)
    for family in ("finite", "dp"):
        for k in (2, 4, 8):
            for alpha in (0.1, 1.0, 10.0):
                m = ConditionalMixture.fit(xf, yf, red, family, k, alpha)
                settings = {"components_cap": k, "concentration": alpha}
                add(family, settings, m.predict, m.diagnostics)
                for expert_ridge in (10.0, 100.0):
                    experts = FullFeatureExperts.fit(
                        m, xf, yf, scaler, global_ridge, expert_ridge
                    )
                    full_settings = {
                        **settings,
                        "expert_ridge": expert_ridge,
                        "global_ridge": best_alpha,
                    }
                    add(
                        family + "_full",
                        full_settings,
                        experts.predict,
                        m.diagnostics,
                    )
        print(dlo, fold, family, contenders[family][0], flush=True)
    seal = {
        "dlo": dlo,
        "fold": fold,
        "fit": sorted(a),
        "validation": sorted(b),
        "test": sorted(c),
        "selection_metric": "mean trajectory RMSE, percent endpoint span",
        "selected": {k: v[0] for k, v in contenders.items()},
        "validation_table": validation_table,
        "test_used_for_selection": False,
    }
    out = output / f"{dlo}-{fold}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "selection.json", seal)
    # Seal predictions before using test residuals for scoring.
    predictions = {name: predictor(xq) for name, (_, predictor) in contenders.items()}
    np.savez_compressed(out / "predictions.npz", groups=groups[test], **predictions)
    results = {
        name: score(y[test], pred, groups[test]) for name, pred in predictions.items()
    }
    record = {
        "dlo": dlo,
        "fold": fold,
        "selection_sha256": digest(out / "selection.json"),
        "predictions_sha256": digest(out / "predictions.npz"),
        "metrics": results,
    }
    write_json(out / "result.json", record)
    return record


def summarize(records, seed=773):
    totals = {}
    for method in records[0]["metrics"]:
        rows = []
        for r in records:
            for row in r["metrics"][method]["per_trajectory"]:
                rows.append({"id": r["dlo"] + "/" + row["trajectory"], **row})
        if len(set(row["id"] for row in rows)) != len(rows):
            raise ValueError("duplicate evaluated trajectory")
        totals[method] = {
            METRIC: float(np.mean([v[METRIC] for v in rows])),
            "nl1_percent_span": float(np.mean([v["nl1_percent_span"] for v in rows])),
            "trajectory_count": len(rows),
            "per_trajectory": rows,
        }
    contrasts = {}
    dp = totals["dp_full"]
    for other in (
        "finite_full",
        "finite",
        "dp",
        "ridge",
        "kernel_analog",
        "rbf_ridge",
        "kinematic",
    ):
        paired = totals[other]["per_trajectory"]
        if [r["id"] for r in dp["per_trajectory"]] != [r["id"] for r in paired]:
            raise ValueError("paired trajectory IDs do not align")
        diff = np.array(
            [
                a[METRIC] - b[METRIC]
                for a, b in zip(dp["per_trajectory"], paired, strict=True)
            ]
        )
        rng = np.random.default_rng(seed)
        intervals = []
        for _ in range(5000):
            sample = []
            for dlo in ("DLO4", "DLO5"):
                idx = np.array(
                    [
                        i
                        for i, row in enumerate(dp["per_trajectory"])
                        if row["id"].startswith(dlo + "/")
                    ]
                )
                sample.extend(diff[rng.choice(idx, len(idx), replace=True)].tolist())
            intervals.append(float(np.mean(sample)))
        contrasts[other] = {
            "dp_minus_comparator_nrmse_percentage_points": float(diff.mean()),
            "descriptive_trajectory_bootstrap_95": np.quantile(
                intervals, [0.025, 0.975]
            ).tolist(),
            "dp_wins": int((diff < -1e-9).sum()),
            "ties": int((np.abs(diff) <= 1e-9).sum()),
            "dp_relative_improvement_percent": 100
            * (1 - dp[METRIC] / totals[other][METRIC]),
        }
    return {"methods": totals, "contrast_method": "dp_full", "dp_contrasts": contrasts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, choices=(1, 5), default=5)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("output directory must be new")
    a.output.mkdir(parents=True)
    started = time.perf_counter()
    panels = load_source(a.source_dir)
    protocol = {
        "contract": "dp-residual-source-screen-v1",
        "folds": a.folds,
        "source_model_sha256": MODEL_SHA,
        "source_result_sha256": RESULT_SHA,
        "script_sha256": digest(Path(__file__)),
        "units": "percent of current endpoint span, not millimetres",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "mixture_grid": {
            "families": ["finite", "dp"],
            "full_feature_expert_ridge": [10.0, 100.0],
            "caps": [2, 4, 8],
            "total_concentrations": [0.1, 1.0, 10.0],
            "rank_x": 12,
            "rank_y": 12,
        },
        "official_evaluation_read": False,
        "is_native_deform_hybrid": False,
        "historical_source_reanalysis": True,
        "persistent_hdp_implemented": False,
    }
    write_json(a.output / "protocol.json", protocol)
    records = []
    with threadpool_limits(limits=1):
        for dlo, (x, y, g, historical) in panels.items():
            for split in partitions(g, historical, a.folds):
                records.append(run_panel(dlo, x, y, g, split, a.output))
    result = {
        "protocol": protocol,
        "records": records,
        **summarize(records),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Normalized errors are percentages of endpoint span, not millimetres. "
            "Retrospective source-only real DEFORM data, kinematic baseline, "
            "25-frame predictions with five-frame prefixes. Not the native "
            "trained DEFORM hybrid or official evaluation, not a full sticky HDP. "
            "Bootstrap intervals are descriptive."
        ),
    }
    write_json(a.output / "summary.json", result)
    compact = {
        method: {k: v for k, v in values.items() if k != "per_trajectory"}
        for method, values in result["methods"].items()
    }
    print(json.dumps(compact, indent=2), flush=True)
    print(json.dumps(result["dp_contrasts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
