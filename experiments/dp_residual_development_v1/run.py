"""Run one retrospective DLO1 development comparison, never official evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from model import ResidualExperts, Spec, self_test

ROOT = Path(__file__).resolve().parents[2]
SOURCE_BASE = Path("/home/florianpfaff/source-only/deform-bayesian-v1")
UPSTREAM = SOURCE_BASE / "DEFORM-b73b8b8"
LONGRUN = Path(
    "/home/florianpfaff/source-only/deform-bayesian-v2/runs/longrun-v2-2060c71/longrun_result.json"
)
SHRINKAGES = (0.0, 0.25, 0.5, 1.0)
SPECS = [Spec("ridge"), Spec("rff")]
SPECS += [Spec("finite", k) for k in (2, 4, 8)]
SPECS += [Spec("finite_bayes", k) for k in (2, 4, 8)]
SPECS += [Spec("dp", 8, a) for a in (0.1, 1.0, 10.0)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def case_errors(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid predicted trajectory")
    return np.mean(np.abs(pred - target), axis=(1, 2, 3))


def corrected(
    base: np.ndarray, canonical: np.ndarray, frames: np.ndarray, shrinkage: float
) -> np.ndarray:
    pred = base.copy().astype(np.float64)
    if shrinkage != 0:
        pred[:, :, 2:-2] += shrinkage * np.einsum("ntvj,nij->ntvi", canonical, frames)
    if not np.array_equal(pred[:, :, [0, 1, -2, -1]], base[:, :, [0, 1, -2, -1]]):
        raise AssertionError("Clamped nodes were modified")
    return pred


def save_model(path: Path, model: ResidualExperts) -> None:
    arrays = {
        key: getattr(model, key)
        for key in ("location", "scale", "coef", "rff_w", "rff_b", "component_mass")
    }
    arrays["spec_json"] = np.asarray(json.dumps(asdict(model.spec), sort_keys=True))
    if model.gate is not None:
        for key in ("gx_center", "gx_scale", "gy_center", "gy_scale"):
            arrays[key] = getattr(model, key)
        for key in ("means_", "covariances_", "weights_"):
            arrays["gate_" + key] = getattr(model.gate, key)
        for kind in ("x", "y"):
            for key in ("components_", "mean_", "explained_variance_"):
                arrays[kind + "_pca_" + key] = getattr(
                    getattr(model, kind + "_pca"), key
                )
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise RuntimeError("Refusing to overwrite a completed experiment")
    started = time.perf_counter()
    write(output / "controlled_test.json", self_test())
    print("CONTROLLED_TEST_PASSED", flush=True)

    config = json.loads(
        (ROOT / "configs/sota/deform_dlo_local_residual_v4.json").read_text()
    )
    manifest_path = ROOT / config["source_manifest"]["repository_path"]
    if digest(manifest_path) != config["source_manifest"]["sha256"]:
        raise ValueError("Source manifest hash changed")
    if digest(LONGRUN) != config["longrun_result"]["sha256"]:
        raise ValueError("Checkpoint source record hash changed")
    manifest = json.loads(manifest_path.read_text())
    longrun = json.loads(LONGRUN.read_text())
    if (
        longrun["selected_checkpoint"]["checkpoint"]["sha256"]
        != config["baseline"]["checkpoint_sha256"]
    ):
        raise ValueError("Unexpected backbone")
    lp = ROOT / config["longrun_protocol"]["repository_path"]
    if digest(lp) != config["longrun_protocol"]["sha256"]:
        raise ValueError("Long-run protocol hash changed")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = json.loads(lp.read_text())["training"][
        "cublas_workspace_config"
    ]
    import run_deform_dlo_longrun_posterior as posterior
    import run_deform_dlo_source as source
    import torch

    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features,
        deform_causal_inputs,
        fit_deform_local_residual,
        predict_deform_local_residual,
    )

    source._assert_upstream(UPSTREAM, longrun["upstream"]["commit"])
    for dlo in ("DLO2", "DLO3", "DLO4", "DLO5"):
        source._install_eval_read_guard(UPSTREAM / "data_set" / dlo)
    source._install_eval_read_guard(UPSTREAM / "data_set/DLO1/eval")
    blocked = {
        str(Path(manifest["trajectories"][name]["path"]).resolve())
        for name in manifest["split"]["source_test"]
    }

    def protect_source_test(event, values):
        if event == "open" and isinstance(values[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(values[0])
            if os.path.realpath(path) in blocked:
                raise PermissionError("This experiment may not open DLO1 source_test")

    sys.addaudithook(protect_source_test)
    fit_names = manifest["split"]["fit"]
    val_names = manifest["split"]["validation"]
    names = fit_names + val_names
    assert (
        len(fit_names) == 40
        and len(val_names) == 8
        and not set(fit_names) & set(val_names)
    )
    inner_val = np.arange(0, 40, 5)
    inner_train = np.setdiff1d(np.arange(40), inner_val)
    protocol = {
        "contract": "dp-residual-dlo1-development-v1",
        "evidence_status": "retrospective-development-only",
        "code_sha": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "script_sha256": digest(Path(__file__)),
        "model_sha256": digest(Path(__file__).with_name("model.py")),
        "source_manifest_sha256": digest(manifest_path),
        "backbone_checkpoint": longrun["selected_checkpoint"]["checkpoint"],
        "backbone_semantics": "DEFORM rod physics plus trained GCN correction",
        "fit_names": fit_names,
        "validation_names": val_names,
        "inner_fit_names": [fit_names[i] for i in inner_train],
        "inner_validation_names": [fit_names[i] for i in inner_val],
        "candidate_specs": [asdict(s) for s in SPECS],
        "shrinkages": list(SHRINKAGES),
        "ridge": 1.0,
        "feature_rank": 6,
        "response_rank": 4,
        "gate_stride": 10,
        "selection": "minimum inner-validation trajectory-balanced full-coordinate L1, then shrinkage, then name",
        "query_information": "two observed states, known future clamped inputs, frozen backbone rollout",
        "source_test_read": False,
        "official_eval_read": False,
        "other_dlo_read": False,
        "prob4d_used": False,
        "new_physical_data": False,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "backbone checkpoint previously selected on this development validation panel",
            "inner residual split is not independent of backbone training",
            "two-stage plug-in DP gate; no full DP regression posterior",
            "no temporal regime prior or correlated observation likelihood",
            "no covariance calibration or latent physical-state correction claim",
        ],
    }
    write(output / "protocol.json", protocol)
    source._seed_everything(torch, 42)
    modules = source._load_upstream(UPSTREAM)
    state = posterior._checkpoint_states(longrun, {6400}, torch=torch)[6400]
    trajectories = source._load_named_trajectories(
        manifest, names, frame_count=500, node_count=13
    )
    full = np.stack([trajectories[name] for name in names])
    print("FROZEN_BACKBONE_ROLLOUT_START", len(names), flush=True)
    rollout = posterior._evaluate_state(
        state,
        trajectories,
        modules=modules,
        torch=torch,
        device="cuda:0",
        dlo_type="DLO1",
        node_count=13,
    )
    assert list(rollout["names"]) == names
    base = np.asarray(rollout["predictions"], dtype=np.float64)
    target = np.asarray(rollout["targets"], dtype=np.float64)
    initial, action = deform_causal_inputs(full)
    x, frames = build_deform_local_residual_features(initial, action, base)
    y = np.einsum("ntvi,nij->ntvj", target - base, frames)[:, :, 2:-2]
    modified = full.copy()
    modified[:, 2:, 2:-2] += 123.0
    initial2, action2 = deform_causal_inputs(modified)
    x2, frames2 = build_deform_local_residual_features(initial2, action2, base)
    assert np.array_equal(x, x2) and np.array_equal(frames, frames2)
    del modified, x2
    query_hashes = [
        hashlib.sha256(
            initial[i].tobytes() + action[i].tobytes() + base[i].tobytes()
        ).hexdigest()
        for i in range(len(names))
    ]
    if len(set(query_hashes)) != len(names):
        raise ValueError(
            "Duplicate causal queries: grouped split is required before fitting"
        )
    np.savez_compressed(
        output / "development_arrays.npz",
        initial=initial,
        action=action,
        base=base,
        target=target,
        names=np.asarray(names),
    )
    print("FROZEN_BACKBONE_ROLLOUT_DONE", base.shape, flush=True)

    selection = {}
    tuning = []
    for spec in SPECS:
        t0 = time.perf_counter()
        model = ResidualExperts(spec).fit(x[inner_train], y[inner_train])
        residual = model.predict(x[inner_val])
        options = []
        for shrink in SHRINKAGES:
            pred = corrected(base[inner_val], residual, frames[inner_val], shrink)
            options.append((float(case_errors(pred, target[inner_val]).mean()), shrink))
        error, shrink = min(options)
        record = {
            "spec": asdict(spec),
            "name": spec.name,
            "inner_l1_m": error,
            "shrinkage": shrink,
            "active_components": model.active_components,
            "fit_seconds": time.perf_counter() - t0,
            "warnings": model.warnings,
            "converged": True if model.gate is None else bool(model.gate.converged_),
        }
        tuning.append(record)
        old = selection.get(spec.kind)
        if old is None or (error, shrink, spec.name) < (
            old["inner_l1_m"],
            old["shrinkage"],
            old["name"],
        ):
            selection[spec.kind] = record
        print("INNER", json.dumps(record), flush=True)
        write(output / "inner_selection_progress.json", {"candidates": tuning})
    write(
        output / "selection_seal.json",
        {
            "selection": selection,
            "all_candidates": tuning,
            "outer_validation_scored_by_new_method": False,
        },
    )
    del model

    predictions = {"backbone": base[40:].copy()}
    details = {}
    for kind, choice in selection.items():
        spec = Spec(**choice["spec"])
        t0 = time.perf_counter()
        model = ResidualExperts(spec).fit(x[:40], y[:40])
        residual = model.predict(x[40:])
        predictions[kind] = corrected(
            base[40:], residual, frames[40:], choice["shrinkage"]
        )
        save_model(output / f"model_{kind}.npz", model)
        details[kind] = {
            "selection": choice,
            "active_components": model.active_components,
            "refit_seconds": time.perf_counter() - t0,
            "warnings": model.warnings,
            "converged": True if model.gate is None else bool(model.gate.converged_),
            "component_weights": [1.0]
            if model.gate is None
            else model.gate.weights_.tolist(),
        }
        if kind == "ridge":
            predictions["ridge_fixed_s0p5"] = corrected(
                base[40:], residual, frames[40:], 0.5
            )
        print("REFIT", kind, "components", model.active_components, flush=True)
    reference = fit_deform_local_residual(
        initial[:40],
        action[:40],
        base[:40],
        target[:40],
        fit_names,
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    reference_pred = predict_deform_local_residual(
        reference, initial[40:], action[40:], base[40:], shrinkage=0.5
    )["predictions"]
    parity = float(np.max(np.abs(reference_pred - predictions["ridge_fixed_s0p5"])))
    if parity > 1e-7:
        raise AssertionError(f"Current ridge mean reproduction failed: {parity}")
    np.savez_compressed(output / "sealed_predictions.npz", **predictions)
    write(
        output / "prediction_seal.json",
        {
            "prediction_sha256": digest(output / "sealed_predictions.npz"),
            "selection_sha256": digest(output / "selection_seal.json"),
            "ridge_max_abs_parity_m": parity,
            "future_free_truth_invariance_passed": True,
        },
    )
    expected = float(config["baseline"]["validation_l1_m"])
    actual = float(case_errors(predictions["backbone"], target[40:]).mean())
    if abs(actual - expected) > float(config["baseline"]["reproduction_tolerance_m"]):
        raise AssertionError(f"Backbone reproduction drift: {actual} versus {expected}")
    errors = {
        kind: case_errors(pred, target[40:]) for kind, pred in predictions.items()
    }
    metrics = {}
    for kind, e in errors.items():
        pred = predictions[kind]
        metrics[kind] = {
            "mean_coordinate_l1_mm": float(e.mean() * 1000),
            "coordinate_rmse_mm": float(
                np.sqrt(np.mean((pred - target[40:]) ** 2)) * 1000
            ),
            "per_trajectory_l1_mm": dict(
                zip(val_names, (e * 1000).tolist(), strict=False)
            ),
            "wins_vs_fixed_ridge": int(np.sum(e < errors["ridge_fixed_s0p5"] - 1e-12)),
            "per_horizon_third_l1_mm": [
                float(np.mean(np.abs(a - b)) * 1000)
                for a, b in zip(
                    np.array_split(pred, 3, axis=1),
                    np.array_split(target[40:], 3, axis=1),
                    strict=False,
                )
            ],
        }
    rng = np.random.default_rng(991)
    draw = rng.integers(0, len(val_names), size=(10000, len(val_names)))
    contrasts = {}
    for kind in (
        "backbone",
        "ridge_fixed_s0p5",
        "ridge",
        "rff",
        "finite",
        "finite_bayes",
    ):
        delta = (errors["dp"] - errors[kind]) * 1000
        contrasts[kind] = {
            "dp_minus_comparator_mm": float(delta.mean()),
            "descriptive_trajectory_bootstrap_95": np.quantile(
                delta[draw].mean(1), [0.025, 0.975]
            ).tolist(),
            "dp_wins": int(np.sum(delta < -1e-9)),
            "relative_improvement_percent": float(
                100 * (1 - errors["dp"].mean() / errors[kind].mean())
            ),
        }
    result = {
        "contract": protocol["contract"],
        "status": "completed",
        "evidence_status": "retrospective-development-only",
        "code_sha": protocol["code_sha"],
        "run_id": protocol["run_id"],
        "source_test_read": False,
        "official_eval_read": False,
        "n_fit": 40,
        "n_validation": 8,
        "n_forecast_frames": int(base.shape[1]),
        "metrics": metrics,
        "selected_models": details,
        "dp_contrasts": contrasts,
        "ridge_max_abs_parity_m": parity,
        "backbone_reproduction_error_m": abs(actual - expected),
        "elapsed_seconds": time.perf_counter() - started,
        "paper_claim_authorized": False,
        "limitations": protocol["limitations"],
    }
    write(output / "result.json", result)
    print("RESULT_JSON", json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
