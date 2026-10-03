"""Fit the frozen shared forecaster and execute only the validation headroom gate."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    build_deform_local_residual_features,
)

from .core import (
    PANELS,
    PrefixProvider,
    context_from_gp,
    headroom_summary,
    pairs,
    prefix_times,
)
from .gp import GPConfig, fit_gp
from .prepare import HERE, digest, revision, write_json


def load_cache(cache: Path, partition: str, receipt: dict) -> dict:
    if partition not in ("fit", "calibration"):
        raise PermissionError("headroom runner cannot open source-test")
    path = cache / f"{partition}.npz"
    if digest(path) != receipt["files"][partition]["sha256"]:
        raise ValueError("cache checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        result = {
            key: archive[key]
            for key in ("names", "initial", "action", "baseline", "target")
        }
    names = result["names"].tolist()
    if names != receipt["partitions"][partition]:
        raise ValueError("cache recording order differs")
    n = len(names)
    for key, shape in {
        "initial": (n, 2, 12, 3),
        "action": (n, 498, 4, 3),
        "baseline": (n, 498, 12, 3),
        "target": (n, 498, 12, 3),
    }.items():
        result[key] = np.asarray(result[key], dtype=np.float64)
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError(f"invalid cache array: {key}")
    return result


def calibrate_noise(
    model, features, frames, data, protocol
) -> tuple[float, list[dict]]:
    rows = []
    for inflation in protocol["noise_inflations"]:
        losses = []
        for i in range(len(data["names"])):
            for endpoint in protocol["endpoints"]:
                context = context_from_gp(
                    model,
                    features[i],
                    data["baseline"][i],
                    frames[i],
                    endpoint=endpoint,
                    horizon=100,
                    panel=0,
                    inflation=inflation,
                )
                times = prefix_times(endpoint)
                design = context.design[times].reshape(-1, context.design.shape[-1])
                residual = (
                    (data["target"][i, times, 2:-2] - context.open_loop[times, 2:-2])
                    @ frames[i]
                ).reshape(-1, 3)
                loss = 0.0
                for axis in range(3):
                    cov = design @ context.coefficient_covariance[axis] @ design.T
                    cov += np.eye(len(design)) * context.noise_m2[axis]
                    factor = cho_factor((cov + cov.T) / 2, lower=True)
                    error = residual[:, axis]
                    loss += float(error @ cho_solve(factor, error))
                    loss += 2 * float(np.log(np.diag(factor[0])).sum())
                    loss += len(error) * np.log(2 * np.pi)
                losses.append(loss / (2 * residual.size))
        rows.append({"inflation": inflation, "mean_prefix_nll": float(np.mean(losses))})
    selected = min(rows, key=lambda r: (r["mean_prefix_nll"], r["inflation"]))
    return float(selected["inflation"]), rows


def run(cache: Path, output: Path, implementation_commit: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    receipt = json.loads((cache / "manifest.json").read_text())
    if (
        receipt["contract"] != protocol["contract"]
        or receipt["protocol_sha256"] != digest(protocol_path)
        or receipt["source_test_payload_read"] is not False
        or receipt["checkpoint_sha256"] != protocol["checkpoint"]["sha256"]
    ):
        raise ValueError("export provenance differs")
    started = time.perf_counter()
    fit = load_cache(cache, "fit", receipt)
    validation = load_cache(cache, "calibration", receipt)
    if len(fit["names"]) != 39 or len(validation["names"]) != 9:
        raise ValueError("locked panel incomplete")
    if set(fit["names"]) & set(validation["names"]):
        raise ValueError("fit/validation overlap")
    features, frames = build_deform_local_residual_features(
        fit["initial"], fit["action"], fit["baseline"]
    )
    residual = np.einsum("ntvi,nij->ntvj", fit["target"] - fit["baseline"], frames)
    groups = np.repeat(np.arange(39), 498 * 8)
    print("fitting pooled GP: 39 recordings, all 155376 free-node rows", flush=True)
    model = fit_gp(
        features.reshape(-1, features.shape[-1]),
        residual[:, :, 2:-2].reshape(-1, 3),
        groups,
        GPConfig(**protocol["gp"]),
    )
    model_path = output / "model.npz"
    np.savez_compressed(model_path, **model.arrays())
    fit_seconds = time.perf_counter() - started
    vf, vr = build_deform_local_residual_features(
        validation["initial"], validation["action"], validation["baseline"]
    )
    inflation, calibration = calibrate_noise(model, vf, vr, validation, protocol)
    write_json(
        output / "calibration.json",
        {
            "method": protocol["noise_selection"],
            "selected_inflation": inflation,
            "candidates": calibration,
            "source_test_payload_read": False,
            "distribution_free_coverage_claim": False,
        },
    )
    frozen = {
        "contract": protocol["contract"],
        "protocol_sha256": digest(protocol_path),
        "model_sha256": digest(model_path),
        "cache_manifest_sha256": digest(cache / "manifest.json"),
        "calibration_sha256": digest(output / "calibration.json"),
        "source_files": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
        "source_test_payload_read": False,
        "protected_cohort_access": False,
        "implementation_commit": revision(implementation_commit),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "fit_seconds": fit_seconds,
        },
    }
    write_json(output / "method_seal.json", frozen)
    predictions, metadata, technical = [], [], []
    for i, name in enumerate(validation["names"].tolist()):
        for endpoint in protocol["endpoints"]:
            for panel in (0, 1):
                context = context_from_gp(
                    model,
                    vf[i],
                    validation["baseline"][i],
                    vr[i],
                    endpoint=endpoint,
                    horizon=100,
                    panel=panel,
                    inflation=inflation,
                )
                times = prefix_times(endpoint)
                provider = PrefixProvider(
                    times,
                    PANELS[panel],
                    validation["target"][i][np.ix_(times, PANELS[panel])],
                    endpoint=endpoint,
                )
                # Every counterfactual pair is sealed before any future scoring.
                for selected in pairs(panel):
                    score = context.score(selected)
                    forecast = context.forecast(selected, provider.reveal(selected))
                    if not score.valid or not forecast.valid:
                        technical.append(
                            {
                                "name": name,
                                "endpoint": endpoint,
                                "panel": panel,
                                "nodes": list(map(int, selected)),
                                "failure": score.failure or forecast.failure,
                            }
                        )
                    predictions.append(forecast.positions[:, context.scoring_nodes])
                    metadata.append(
                        {
                            "name": name,
                            "recording": i,
                            "endpoint": endpoint,
                            "panel": panel,
                            "nodes": list(map(int, selected)),
                            "predicted_mse_reduction_m2": score.expected_mse_reduction_m2,
                            "valid": score.valid and forecast.valid,
                        }
                    )
        print(f"sealed-candidate-ready validation recording {i + 1}/9", flush=True)
    prediction_path = output / "validation_predictions.npz"
    np.savez_compressed(prediction_path, predictions=np.stack(predictions))
    write_json(output / "query_log.json", metadata)
    write_json(
        output / "prediction_seal.json",
        {
            **frozen,
            "predictions_sha256": digest(prediction_path),
            "query_log_sha256": digest(output / "query_log.json"),
            "candidate_count": len(metadata),
            "future_scoring_started": False,
            "technical_failures": technical,
        },
    )
    if technical:
        result = {
            **frozen,
            "decision": "technical_failure_no_advancement",
            "passed": False,
            "technical_failures": technical,
            "source_test_authorized": False,
        }
    else:
        scores = []
        for prediction, row in zip(predictions, metadata, strict=True):
            truth = validation["target"][
                row["recording"], row["endpoint"] : row["endpoint"] + 100
            ][:, PANELS[1 - row["panel"]]]
            scores.append(float(np.mean((prediction - truth) ** 2)))
        mse = np.asarray(scores).reshape(9, 3, 2, 6)
        result = {
            **frozen,
            **headroom_summary(mse, validation["names"].tolist()),
            "technical_failures": [],
            "pair_mse_m2": mse.tolist(),
            "source_test_authorized": False,
            "next_stage": "freeze_all_controls_before_source_test",
        }
        if not result["passed"]:
            result["next_stage"] = "terminal_stop_no_selector_development"
    result["wall_seconds"] = time.perf_counter() - started
    result["prediction_seal_sha256"] = digest(output / "prediction_seal.json")
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "decision",
                    "best_fixed_rmse_mm",
                    "oracle_rmse_mm",
                    "relative_rmse_headroom",
                    "wall_seconds",
                    "source_test_payload_read",
                )
            }
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", type=revision, required=True)
    args = parser.parse_args()
    try:
        run(args.cache.resolve(), args.output.resolve(), args.revision)
    except Exception as exc:
        if args.output.is_dir():
            write_json(
                args.output / "technical_failure.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retry_authorized": False,
                    "source_test_authorized": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
