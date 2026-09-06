#!/usr/bin/env python3
"""Development-only finite-rank GP mean comparison; never opens test partitions.

The finite-rank GP uses a whitened Nystrom Matern-3/2 basis with a Gaussian
weight prior. Its mean is also kernel ridge regression; this pilot makes no
claim about uniquely Bayesian accuracy or calibrated posterior uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist


def kernel(x: np.ndarray, z: np.ndarray, length: float) -> np.ndarray:
    distance = cdist(x, z) / (length * math.sqrt(x.shape[1]))
    a = math.sqrt(3.0) * distance
    return (1.0 + a) * np.exp(-a)


def gp_mean(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    *,
    length: float,
    noise: float,
    inducing: int = 128,
    seed: int = 20260907,
) -> np.ndarray:
    """Fit finite-rank GP on all fit rows, with fit-only inducing locations."""
    if length <= 0 or noise <= 0 or inducing < 1:
        raise ValueError("GP parameters must be positive")
    if x.ndim != 2 or y.ndim != 2 or q.ndim != 2 or len(x) != len(y):
        raise ValueError("Invalid training/query shape")
    if x.shape[1] != q.shape[1] or not all(np.isfinite(v).all() for v in [x, y, q]):
        raise ValueError("Nonfinite or incompatible GP data")
    location, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    x, q = (x - location) / scale, (q - location) / scale
    center = y.mean(0)
    y = y - center
    rng = np.random.default_rng(seed)
    z = x[rng.choice(len(x), min(inducing, len(x)), replace=False)]
    kmm = kernel(z, z, length)
    chol = np.linalg.cholesky(kmm + np.eye(len(z)) * 1e-7)
    normal = noise * np.eye(len(z))
    rhs = np.zeros((len(z), y.shape[1]))
    for start in range(0, len(x), 2048):
        basis = solve_triangular(
            chol, kernel(x[start : start + 2048], z, length).T, lower=True
        ).T
        normal += basis.T @ basis
        rhs += basis.T @ y[start : start + 2048]
    weights = cho_solve(cho_factor(normal, lower=True), rhs)
    result = np.empty((len(q), y.shape[1]))
    for start in range(0, len(q), 2048):
        basis = solve_triangular(
            chol, kernel(q[start : start + 2048], z, length).T, lower=True
        ).T
        result[start : start + 2048] = center + basis @ weights
    return result


def unit_tests() -> dict[str, bool]:
    rng = np.random.default_rng(7)
    x, q = rng.normal(size=(50, 3)), rng.normal(size=(17, 3))
    y = np.column_stack([np.sin(x[:, 0]), x[:, 1] ** 2])
    result = gp_mean(x, y, q, length=1.0, noise=0.1, inducing=len(x))
    location, scale = x.mean(0), x.std(0)
    xx, qq = (x - location) / scale, (q - location) / scale
    direct = y.mean(0) + kernel(qq, xx, 1.0) @ np.linalg.solve(
        kernel(xx, xx, 1.0) + np.eye(len(x)) * 0.1, y - y.mean(0)
    )
    assert np.max(np.abs(result - direct)) < 2e-5
    assert np.array_equal(
        result, gp_mean(x, y, q, length=1.0, noise=0.1, inducing=len(x))
    )
    assert np.allclose(
        gp_mean(x, np.ones((50, 2)) * 3.0, q, length=1.0, noise=0.1), 3.0
    )
    return {
        "dense_gp_mean_parity": True,
        "deterministic": True,
        "constant_response": True,
    }


def load_partition(path: Path, names: list[str]) -> dict:
    """Read one explicitly named development NPZ with verified identity order."""
    if path.stem not in {"fit", "validation", "fit_rollout", "validation_rollout"}:
        raise ValueError("Only explicit fit/validation cache files are allowed")
    aliases = {
        "initial": ["initial", "initial_states"],
        "action": ["action", "actions", "clamped_action"],
        "baseline": ["baseline", "baseline_predictions", "predictions"],
        "targets": ["targets", "target", "truth"],
    }
    result = {}
    with np.load(path, allow_pickle=False) as data:
        identity_key = next((k for k in ["names", "ordered_names"] if k in data), None)
        if identity_key is None:
            raise ValueError("Cache must contain explicit trajectory identities")
        result["names"] = data[identity_key].astype(str).tolist()
        if result["names"] != names:
            raise ValueError("Cache identities do not match frozen development order")
        for dest, choices in aliases.items():
            key = next((k for k in choices if k in data), None)
            if key is None:
                raise ValueError(f"{path.name}: missing {dest}; available {data.files}")
            result[dest] = np.asarray(data[key], dtype=np.float64)
    if result["baseline"].shape != result["targets"].shape:
        raise ValueError("Baseline/target shapes differ")
    if not all(np.isfinite(result[k]).all() for k in aliases):
        raise ValueError("Nonfinite development arrays")
    result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("gp_pilot_result.json"))
    args = parser.parse_args()
    tests = unit_tests()
    if args.self_test:
        print(json.dumps(tests))
        return 0
    if args.cache_root is None or args.manifest is None:
        parser.error("--cache-root and --manifest are required")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        _collapse_duplicate_queries,
        build_deform_local_residual_features,
        fit_deform_local_residual,
        predict_deform_local_residual,
    )

    manifest = json.loads(args.manifest.read_text())
    fit_names, val_names = manifest["split"]["fit"], manifest["split"]["validation"]
    if len(fit_names) != 32 or len(val_names) != 12 or set(fit_names) & set(val_names):
        raise ValueError("Expected frozen disjoint DLO2 32/12 development split")
    if (set(fit_names) | set(val_names)) & set(manifest["split"]["source_test"]):
        raise ValueError("Development/source-test overlap")
    fit = load_partition(args.cache_root / "fit.npz", fit_names)
    val = load_partition(args.cache_root / "validation.npz", val_names)
    started = time.perf_counter()
    fitted = fit_deform_local_residual(
        fit["initial"],
        fit["action"],
        fit["baseline"],
        fit["targets"],
        fit_names,
        ridge=1.0,
        variance_floor_m2=1e-10,
    )
    ridge = predict_deform_local_residual(
        fitted, val["initial"], val["action"], val["baseline"], shrinkage=0.25
    )["predictions"]
    initial, action, baseline, target, groups = _collapse_duplicate_queries(
        fit["initial"], fit["action"], fit["baseline"], fit["targets"], fit_names
    )
    xf, frames = build_deform_local_residual_features(initial, action, baseline)
    xv, vframes = build_deform_local_residual_features(
        val["initial"], val["action"], val["baseline"]
    )
    y = np.einsum("ntvi,nij->ntvj", target - baseline, frames)[:, :, 2:-2]
    results = []

    def score(name: str, pred: np.ndarray, settings: dict | None = None) -> None:
        assert np.array_equal(
            pred[:, :, [0, 1, -2, -1]], val["baseline"][:, :, [0, 1, -2, -1]]
        )
        errors = np.mean(np.abs(pred - val["targets"]), axis=(1, 2, 3)) * 1000
        ref = np.mean(np.abs(ridge - val["targets"]), axis=(1, 2, 3)) * 1000
        record = {
            "model": name,
            "mean_l1_mm": float(errors.mean()),
            "case_l1_mm": dict(zip(val_names, errors.tolist(), strict=True)),
            "wins_vs_ridge": int(np.sum(errors < ref - 1e-10)),
            "losses_vs_ridge": int(np.sum(errors > ref + 1e-10)),
            "settings": settings or {},
        }
        results.append(record)
        print("GP_PILOT_ARM", json.dumps(record, sort_keys=True), flush=True)

    score("unchanged_deform_hybrid", val["baseline"])
    score("incumbent_ridge1_shrink0.25", ridge)
    for shrink in [0.1, 0.5, 1.0]:
        pred = predict_deform_local_residual(
            fitted, val["initial"], val["action"], val["baseline"], shrinkage=shrink
        )["predictions"]
        score(f"ridge1_shrink{shrink}", pred, {"shrinkage": shrink})
    for length in [0.5, 1.0, 2.0]:
        for noise in [0.1, 1.0]:
            canonical = np.zeros_like(y[:1]).repeat(len(val_names), axis=0)
            for node in range(xf.shape[2]):
                canonical[:, :, node] = gp_mean(
                    xf[:, :, node].reshape(-1, xf.shape[-1]),
                    y[:, :, node].reshape(-1, 3),
                    xv[:, :, node].reshape(-1, xv.shape[-1]),
                    length=length,
                    noise=noise,
                ).reshape(len(val_names), xv.shape[1], 3)
            correction = np.einsum("ntvj,nij->ntvi", canonical, vframes)
            for shrink in [0.1, 0.25, 0.5, 1.0]:
                pred = val["baseline"].copy()
                pred[:, :, 2:-2] += shrink * correction
                score(
                    f"gp_l{length}_n{noise}_s{shrink}",
                    pred,
                    {
                        "length": length,
                        "noise": noise,
                        "shrinkage": shrink,
                        "inducing_per_node": 128,
                    },
                )
    report = {
        "schema_version": 1,
        "contract": "deform-gp-development-pilot-v1",
        "scope": "exploratory DLO2 fit/validation only; selection-biased development scores",
        "source_test_read": False,
        "official_eval_read": False,
        "new_hardware_data": False,
        "unique_bayesian_accuracy_claim": False,
        "fit_names": fit_names,
        "validation_names": val_names,
        "collapsed_fit_groups": [list(g) for g in groups],
        "cache_sha256": {"fit": fit["sha256"], "validation": val["sha256"]},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tests": tests,
        "elapsed_seconds": time.perf_counter() - started,
        "source_commit": os.environ.get("GITHUB_SHA"),
        "arms": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("GP_PILOT_RESULT", json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
