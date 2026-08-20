#!/usr/bin/env python3
"""Validate or explicitly complete DLO3 source runs from immutable method seals."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deserialize_deform_local_residual_model,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    build_deform_bayesian_covariance_ablation_v1,
    deform_bayesian_covariance_archive_key,
    deform_local_feature_indices,
    evaluate_deform_dlo3_source_gate,
    evaluate_deform_predictive_distribution,
    load_deform_dlo3_method_seal_recovery_v1,
    load_deform_dlo_robustness_v1_protocol,
    predict_deform_local_residual_variant,
    validate_deform_dlo3_source_manifest,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

Array = np.ndarray[Any, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "complete"), required=True)
    parser.add_argument("--recovery-lock", type=Path, required=True)
    parser.add_argument("--failure-receipt", type=Path, required=True)
    parser.add_argument("--calibration-smoke", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--failed-root", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--implementation-archive", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked recovery output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _identity(path: Path, *, update: int | None = None) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if update is not None:
        identity["update"] = update
    return identity


def _verified_file(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(cast(Any, identity.get("size_bytes", -1)))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity changed")
    return path


def _load_full_model(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        model = deserialize_deform_local_residual_model(archive)
        required = {"coefficient_covariance_full", "residual_covariance_full"}
        if not required.issubset(archive.files):
            raise ValueError("recovery full covariance model is incomplete")
        model["coefficient_covariance_full"] = np.asarray(
            archive["coefficient_covariance_full"]
        ).copy()
        model["residual_covariance_full"] = np.asarray(
            archive["residual_covariance_full"]
        ).copy()
    return model


def _load_local_model(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return deserialize_deform_local_residual_model(archive)


def _load_variant(path: Path, *, expected_arm: str) -> dict[str, object]:
    coordinate_frame = (
        "action-centered-global"
        if expected_arm == "full-global"
        else "initial-action-local"
    )
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "arm",
            "coordinate_frame",
            "node_count",
            "prediction_horizon",
            "feature_indices",
            "feature_location",
            "feature_scale",
            "coefficients",
            "ridge",
        }
        if set(archive.files) != required:
            raise ValueError("recovery mechanism model fields differ")
        arm_values = tuple(str(value) for value in np.asarray(archive["arm"]))
        frame_values = tuple(
            str(value) for value in np.asarray(archive["coordinate_frame"])
        )
        node_values = tuple(int(value) for value in np.asarray(archive["node_count"]))
        horizon_values = tuple(
            int(value) for value in np.asarray(archive["prediction_horizon"])
        )
        ridge_values = tuple(float(value) for value in np.asarray(archive["ridge"]))
        feature_indices = tuple(
            int(value) for value in np.asarray(archive["feature_indices"])
        )
        location = np.asarray(archive["feature_location"], dtype=np.float64).copy()
        scale = np.asarray(archive["feature_scale"], dtype=np.float64).copy()
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64).copy()
    if (
        arm_values != (expected_arm,)
        or frame_values != (coordinate_frame,)
        or node_values != (12,)
        or horizon_values != (498,)
        or ridge_values != (1.0,)
        or feature_indices != deform_local_feature_indices(expected_arm)
        or location.shape != (8, len(feature_indices))
        or scale.shape != location.shape
        or coefficients.shape != (8, len(feature_indices) + 1, 3)
        or not np.isfinite(location).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(coefficients).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("recovery mechanism model content differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo-local-residual-variant-v1",
        "arm": expected_arm,
        "coordinate_frame": coordinate_frame,
        "node_count": 12,
        "prediction_horizon": 498,
        "full_feature_count": 92,
        "feature_indices": feature_indices,
        "feature_location": location,
        "feature_scale": scale,
        "coefficients": coefficients,
        "ridge": 1.0,
    }


def _mean_l1(prediction: Array, target: Array) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def _validate_parent_artifacts(
    *,
    lock: Mapping[str, object],
    protocol: Mapping[str, object],
    protocol_path: Path,
    manifest: Mapping[str, object],
    manifest_path: Path,
    failure_receipt_path: Path,
    calibration_smoke_path: Path,
    failed_root: Path,
    failure_log_path: Path,
    seed: int,
) -> dict[str, Any]:
    parent = _mapping(lock.get("failed_execution"), label="failed execution")
    seed_record = _mapping(
        _mapping(lock.get("seed_artifacts"), label="seed artifacts").get(str(seed)),
        label=f"seed {seed} artifacts",
    )
    if seed not in (42, 43):
        raise ValueError("recovery seed is outside the exact failed pair")
    if sha256_file(protocol_path) != parent.get("protocol_sha256") or sha256_file(
        manifest_path
    ) != parent.get("source_manifest_sha256"):
        raise ValueError("recovery protocol or source manifest identity changed")
    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=False,
    )

    failure_identity = _mapping(parent.get("failure_receipt"), label="failure receipt")
    smoke_identity = _mapping(
        parent.get("calibration_mechanism_preflight"),
        label="calibration mechanism preflight",
    )
    if (
        sha256_file(failure_receipt_path) != failure_identity.get("sha256")
        or sha256_file(calibration_smoke_path) != smoke_identity.get("sha256")
        or sha256_file(failure_log_path) != seed_record.get("failure_log_sha256")
    ):
        raise ValueError("recovery failure or smoke evidence identity changed")
    failure = _read_json(failure_receipt_path)
    smoke = _read_json(calibration_smoke_path)
    if (
        failure.get("contract") != "deform-dlo3-robustness-preseal-runtime-failure-v1"
        or failure.get("replacement_execution_authorized") is not False
        or failure.get("target_accessed") is not False
        or failure.get("official_eval_read") is not False
        or smoke.get("contract")
        != "deform-dlo3-calibration-mechanism-preflight-smoke-v1"
        or smoke.get("passed") is not True
        or smoke.get("source_test_opened") is not False
        or smoke.get("source_test_scores_read") is not False
        or _mapping(smoke.get("method_seal_sha256s"), label="smoke method seals").get(
            str(seed)
        )
        != seed_record.get("method_seal_sha256")
    ):
        raise ValueError("recovery failure or smoke evidence content changed")

    for name in (
        "prediction_seal.json",
        "source_predictions.npz",
        "source_result.json",
    ):
        if (failed_root / name).exists():
            raise RuntimeError(
                "failed root already contains a source completion artifact"
            )
    method_path = failed_root / "method_seal.json"
    compute_match_path = failed_root / "compute_match.json"
    if (
        not method_path.is_file()
        or sha256_file(method_path) != seed_record.get("method_seal_sha256")
        or not compute_match_path.is_file()
        or sha256_file(compute_match_path) != seed_record.get("compute_match_sha256")
    ):
        raise ValueError("recovery parent method or compute-match identity changed")
    method = _read_json(method_path)
    residual = _mapping(protocol.get("local_residual"), label="local residual")
    if (
        method.get("contract") != "deform-dlo3-robustness-source-method-seal-v1"
        or int(cast(Any, method.get("seed", -1))) != seed
        or _mapping(method.get("protocol"), label="method protocol").get("sha256")
        != sha256_file(protocol_path)
        or _mapping(method.get("source_manifest"), label="method manifest").get(
            "sha256"
        )
        != sha256_file(manifest_path)
        or float(cast(Any, method.get("ridge", math.nan)))
        != float(cast(Any, residual["ridge"]))
        or float(cast(Any, method.get("shrinkage", math.nan)))
        != float(cast(Any, residual["shrinkage"]))
        or method.get("source_test_opened") is not False
        or method.get("official_eval_read") is not False
        or method.get("target_selection") is not False
    ):
        raise ValueError("recovery method seal content differs")

    physical_path = _verified_file(
        method.get("physical_checkpoint"), label="recovery physical checkpoint"
    )
    compute_path = _verified_file(
        method.get("compute_matched_checkpoint"),
        label="recovery compute-matched checkpoint",
    )
    schedule_path = _verified_file(
        method.get("window_schedule"), label="recovery window schedule"
    )
    local_model_path = _verified_file(
        method.get("local_residual_model"), label="recovery local residual model"
    )
    full_model_path = _verified_file(
        method.get("full_covariance_model"), label="recovery full covariance model"
    )
    calibration_path = _verified_file(
        method.get("covariance_calibration"), label="recovery covariance calibration"
    )

    compute_match = _read_json(compute_match_path)
    compute_contract = _mapping(
        protocol.get("compute_matched_control"), label="compute-matched control"
    )
    local_seconds = float(
        cast(Any, compute_match.get("local_residual_wall_seconds", math.nan))
    )
    update_seconds = float(
        cast(Any, compute_match.get("median_update_seconds_6301_6400", math.nan))
    )
    additional_updates = int(cast(Any, compute_match.get("additional_updates", -1)))
    expected_additional = (
        int(math.ceil(local_seconds / update_seconds))
        if math.isfinite(local_seconds)
        and local_seconds > 0.0
        and math.isfinite(update_seconds)
        and update_seconds > 0.0
        else -1
    )
    if (
        compute_match.get("contract") != "deform-dlo3-compute-match-v1"
        or int(cast(Any, compute_match.get("seed", -1))) != seed
        or additional_updates != expected_additional
        or additional_updates
        < int(cast(Any, compute_contract["minimum_additional_updates"]))
        or additional_updates
        > int(cast(Any, compute_contract["maximum_additional_updates"]))
        or int(cast(Any, compute_match.get("start_update", -1))) != 6400
        or int(cast(Any, compute_match.get("end_update", -1)))
        != 6400 + additional_updates
        or compute_match.get("source_test_opened") is not False
        or compute_match.get("official_eval_read") is not False
        or int(
            cast(
                Any,
                _mapping(method.get("physical_checkpoint"), label="physical").get(
                    "update", -1
                ),
            )
        )
        != 6400
        or int(
            cast(
                Any,
                _mapping(method.get("compute_matched_checkpoint"), label="compute").get(
                    "update", -1
                ),
            )
        )
        != 6400 + additional_updates
    ):
        raise ValueError("recovery compute-match record differs")

    training = _mapping(protocol.get("physical_training"), label="physical training")
    with np.load(schedule_path, allow_pickle=False) as schedule:
        fit_names = tuple(str(value) for value in np.asarray(schedule["fit_names"]))
        trajectory_indices = np.asarray(schedule["trajectory_indices"])
        start_indices = np.asarray(schedule["start_indices"])
    expected_fit_names = tuple(partitions["fit"])
    expected_trajectory_indices, expected_start_indices = source_runtime._make_schedule(
        fit_names=list(expected_fit_names),
        updates=int(cast(Any, training["total_updates"]))
        + int(cast(Any, compute_contract["maximum_additional_updates"])),
        batch_size=int(cast(Any, training["batch_size"])),
        frame_count=int(
            cast(Any, _mapping(protocol.get("data"), label="data")["frame_count"])
        ),
        horizon=int(cast(Any, training["unroll_horizon_frames"])),
        seed=seed,
    )
    if (
        fit_names != expected_fit_names
        or not np.array_equal(trajectory_indices, expected_trajectory_indices)
        or not np.array_equal(start_indices, expected_start_indices)
    ):
        raise ValueError("recovery training schedule differs")

    local_model = _load_local_model(local_model_path)
    full_model = _load_full_model(full_model_path)
    for key in ("feature_location", "feature_scale", "coefficients"):
        if not np.array_equal(
            np.asarray(local_model[key]), np.asarray(full_model[key])
        ):
            raise ValueError("recovery local and full model means differ")
    calibration = _read_json(calibration_path)
    if (
        calibration.get("contract") != "deform-dlo-full-covariance-calibration-v1"
        or int(cast(Any, calibration.get("rank", -1))) != 9
        or calibration.get("order_statistic") != "maximum-of-nine"
        or calibration.get("source_test_opened") is not False
        or calibration.get("official_eval_read") is not False
    ):
        raise ValueError("recovery covariance calibration differs")

    expected_models = {
        "persistence-plus-full-local": "full-local",
        "physical-plus-intercept-only": "intercept-only",
        "physical-plus-full-no-action": "full-no-action",
        "physical-plus-full-global-frame": "full-global",
    }
    model_identities = _mapping(
        method.get("mechanism_models"), label="recovery mechanism models"
    )
    if set(str(label) for label in model_identities) != set(expected_models):
        raise ValueError("recovery mechanism model set differs")
    mechanism_models = {
        label: _load_variant(
            _verified_file(
                model_identities.get(label), label=f"recovery mechanism {label}"
            ),
            expected_arm=arm,
        )
        for label, arm in expected_models.items()
    }
    return {
        "partitions": partitions,
        "method": method,
        "method_path": method_path,
        "compute_match": compute_match,
        "compute_match_path": compute_match_path,
        "physical_path": physical_path,
        "compute_path": compute_path,
        "schedule_path": schedule_path,
        "local_model": local_model,
        "full_model": full_model,
        "calibration": calibration,
        "calibration_path": calibration_path,
        "mechanism_models": mechanism_models,
    }


def main() -> int:
    args = _parse_args()
    lock_path = args.recovery_lock.resolve()
    protocol_path = args.protocol.resolve()
    manifest_path = args.source_manifest.resolve()
    failure_receipt_path = args.failure_receipt.resolve()
    calibration_smoke_path = args.calibration_smoke.resolve()
    failed_root = args.failed_root.resolve()
    failure_log_path = args.failure_log.resolve()
    output_root = args.output_root.resolve()
    lock = load_deform_dlo3_method_seal_recovery_v1(lock_path)
    decision = _mapping(lock.get("decision"), label="recovery decision")
    implementation_archive_path: Path | None = None
    if args.mode == "complete":
        if decision.get("source_completion_authorized") is not True:
            raise PermissionError(
                "DLO3 method-seal source completion is not authorized"
            )
        policy = _mapping(lock.get("recovery_policy"), label="recovery policy")
        expected_output_name = _mapping(
            policy.get("completion_output_names"), label="recovery output names"
        ).get(str(args.seed))
        if output_root.name != expected_output_name:
            raise ValueError("recovery completion output root name differs")
        if args.upstream_root is None:
            raise ValueError("recovery completion requires the frozen upstream root")
        if args.implementation_archive is None:
            raise ValueError(
                "recovery completion requires its authorized source archive"
            )
        implementation_archive_path = args.implementation_archive.resolve()
        if not implementation_archive_path.is_file() or sha256_file(
            implementation_archive_path
        ) != decision.get("implementation_archive_sha256"):
            raise ValueError("recovery implementation archive identity differs")
    if output_root == failed_root or failed_root in output_root.parents:
        raise ValueError(
            "recovery output root must not modify the immutable failed root"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"recovery output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    manifest = _read_json(manifest_path)
    artifacts = _validate_parent_artifacts(
        lock=lock,
        protocol=protocol,
        protocol_path=protocol_path,
        manifest=manifest,
        manifest_path=manifest_path,
        failure_receipt_path=failure_receipt_path,
        calibration_smoke_path=calibration_smoke_path,
        failed_root=failed_root,
        failure_log_path=failure_log_path,
        seed=args.seed,
    )
    validation = {
        "schema_version": 1,
        "contract": "deform-dlo3-method-seal-recovery-validation-v1",
        "mode": args.mode,
        "seed": args.seed,
        "recovery_lock": _identity(lock_path),
        "failure_receipt": _identity(failure_receipt_path),
        "calibration_smoke": _identity(calibration_smoke_path),
        "failure_log": _identity(failure_log_path),
        "parent_method_seal": _identity(cast(Path, artifacts["method_path"])),
        "parent_compute_match": _identity(cast(Path, artifacts["compute_match_path"])),
        "implementation_source_revision": decision.get(
            "implementation_source_revision"
        ),
        "implementation_archive": (
            _identity(implementation_archive_path)
            if implementation_archive_path is not None
            else None
        ),
        "source_payload_deserialized": False,
        "source_test_opened": False,
        "source_test_scored": False,
        "official_eval_read": False,
        "dlo4_dlo5_reserve_access": False,
        "held_v8_access": False,
        "retraining": False,
        "refitting": False,
        "verified": True,
    }
    validation_path = output_root / "validation.json"
    _write_json(validation_path, validation)
    if args.mode == "validate":
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=True,
    )
    upstream_protocol = _mapping(protocol.get("upstream"), label="upstream")
    upstream = source_runtime._assert_upstream(
        args.upstream_root, str(upstream_protocol["commit"])
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO4")
    source_runtime._install_eval_read_guard(data_root / "DLO5")

    training = _mapping(protocol.get("physical_training"), label="physical training")
    residual = _mapping(protocol.get("local_residual"), label="local residual")
    data = _mapping(protocol.get("data"), label="data")
    cublas_config = str(training["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(torch, args.seed)
    physical_path = cast(Path, artifacts["physical_path"])
    compute_path = cast(Path, artifacts["compute_path"])
    schedule_path = cast(Path, artifacts["schedule_path"])
    physical_bundle = torch.load(physical_path, map_location="cpu", weights_only=True)
    compute_bundle = torch.load(compute_path, map_location="cpu", weights_only=True)
    compute_match = cast(dict[str, object], artifacts["compute_match"])
    for bundle, update, label in (
        (physical_bundle, 6400, "physical"),
        (
            compute_bundle,
            int(cast(Any, compute_match["end_update"])),
            "compute-matched",
        ),
    ):
        if (
            bundle.get("update") != update
            or bundle.get("seed") != args.seed
            or bundle.get("protocol_sha256") != sha256_file(protocol_path)
            or bundle.get("source_manifest_sha256") != sha256_file(manifest_path)
            or bundle.get("schedule_sha256") != sha256_file(schedule_path)
            or bundle.get("official_eval_read") is not False
        ):
            raise ValueError(f"recovery {label} checkpoint lineage differs")

    frame_count = int(cast(Any, data["frame_count"]))
    node_count = int(cast(Any, data["node_count"]))
    calibration_names = list(partitions["calibration"])
    calibration_trajectories = source_runtime._load_named_trajectories(
        manifest,
        calibration_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    calibration_rollout = posterior_runtime._evaluate_state(
        physical_bundle["model_state_dict"],
        calibration_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO3",
        node_count=node_count,
    )
    calibration_initial, calibration_action = local_runtime._causal_inputs(
        calibration_trajectories, calibration_names
    )
    mechanism_models = cast(
        dict[str, Mapping[str, object]], artifacts["mechanism_models"]
    )
    calibration_baselines = {
        "persistence-plus-full-local": np.asarray(calibration_rollout["persistence"]),
        "physical-plus-intercept-only": np.asarray(calibration_rollout["predictions"]),
        "physical-plus-full-no-action": np.asarray(calibration_rollout["predictions"]),
        "physical-plus-full-global-frame": np.asarray(
            calibration_rollout["predictions"]
        ),
    }
    for label, model in mechanism_models.items():
        baseline = calibration_baselines[label]
        prediction = predict_deform_local_residual_variant(
            model,
            calibration_initial,
            calibration_action,
            baseline,
            shrinkage=float(cast(Any, residual["shrinkage"])),
        )["predictions"]
        if prediction.shape != baseline.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"recovery calibration preflight failed for {label}")
    full_model = cast(Mapping[str, object], artifacts["full_model"])
    calibration = cast(dict[str, object], artifacts["calibration"])
    calibration_bayesian = build_deform_bayesian_covariance_ablation_v1(
        full_model,
        calibration_initial,
        calibration_action,
        np.asarray(calibration_rollout["predictions"]),
        shrinkage=float(cast(Any, residual["shrinkage"])),
        variance_scale=float(cast(Any, calibration["variance_scale"])),
    )
    calibration_means = [
        np.asarray(values["predictions"]) for values in calibration_bayesian.values()
    ]
    if any(
        not np.array_equal(calibration_means[0], values)
        for values in calibration_means[1:]
    ):
        raise RuntimeError("recovery Bayesian calibration point means differ")

    recovery_method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-method-seal-recovery-method-seal-v1",
        "seed": args.seed,
        "recovery_lock": _identity(lock_path),
        "validation": _identity(validation_path),
        "parent_method_seal": _identity(cast(Path, artifacts["method_path"])),
        "parent_compute_match": _identity(cast(Path, artifacts["compute_match_path"])),
        "failure_receipt": _identity(failure_receipt_path),
        "calibration_smoke": _identity(calibration_smoke_path),
        "implementation_source_revision": decision["implementation_source_revision"],
        "implementation_archive": _identity(cast(Path, implementation_archive_path)),
        "upstream": upstream,
        "mechanism_preflight_count": len(mechanism_models),
        "bayesian_distribution_preflight_count": len(calibration_bayesian),
        "source_test_opened": False,
        "source_test_scored": False,
        "official_eval_read": False,
        "dlo4_dlo5_reserve_access": False,
        "held_v8_access": False,
        "retraining": False,
        "refitting": False,
        "point_method_changed": False,
    }
    recovery_method_seal_path = output_root / "recovery_method_seal.json"
    _write_json(recovery_method_seal_path, recovery_method_seal)

    started = time.perf_counter()
    source_test_names = list(partitions["source_test"])
    source_test_trajectories = source_runtime._load_named_trajectories(
        manifest,
        source_test_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    source_rollout = posterior_runtime._evaluate_state(
        physical_bundle["model_state_dict"],
        source_test_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO3",
        node_count=node_count,
    )
    compute_rollout = posterior_runtime._evaluate_state(
        compute_bundle["model_state_dict"],
        source_test_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO3",
        node_count=node_count,
    )
    source_initial, source_action = local_runtime._causal_inputs(
        source_test_trajectories, source_test_names
    )
    shrinkage = float(cast(Any, residual["shrinkage"]))
    bayesian_predictions = build_deform_bayesian_covariance_ablation_v1(
        full_model,
        source_initial,
        source_action,
        np.asarray(source_rollout["predictions"]),
        shrinkage=shrinkage,
        variance_scale=float(cast(Any, calibration["variance_scale"])),
    )
    source_prediction = bayesian_predictions[
        "trajectory-clustered-full-coordinate-covariance-v1"
    ]
    calibrated_covariance = np.asarray(
        bayesian_predictions["calibrated-full-coordinate-covariance-v1"][
            "coordinate_covariance_m2"
        ]
    )
    mechanism_predictions = {
        "physical-only": np.asarray(source_rollout["predictions"]),
        "persistence-plus-full-local": predict_deform_local_residual_variant(
            mechanism_models["persistence-plus-full-local"],
            source_initial,
            source_action,
            np.asarray(source_rollout["persistence"]),
            shrinkage=shrinkage,
        )["predictions"],
        "physical-plus-intercept-only": predict_deform_local_residual_variant(
            mechanism_models["physical-plus-intercept-only"],
            source_initial,
            source_action,
            np.asarray(source_rollout["predictions"]),
            shrinkage=shrinkage,
        )["predictions"],
        "physical-plus-full-no-action": predict_deform_local_residual_variant(
            mechanism_models["physical-plus-full-no-action"],
            source_initial,
            source_action,
            np.asarray(source_rollout["predictions"]),
            shrinkage=shrinkage,
        )["predictions"],
        "physical-plus-full-global-frame": predict_deform_local_residual_variant(
            mechanism_models["physical-plus-full-global-frame"],
            source_initial,
            source_action,
            np.asarray(source_rollout["predictions"]),
            shrinkage=shrinkage,
        )["predictions"],
        "physical-plus-full-local-unshrunk": predict_deform_local_residual(
            cast(Mapping[str, object], artifacts["local_model"]),
            source_initial,
            source_action,
            np.asarray(source_rollout["predictions"]),
            shrinkage=1.0,
        )["predictions"],
        "physical-plus-full-local-fixed": source_prediction["predictions"],
    }
    predictions_path = output_root / "source_predictions.npz"
    prediction_payload: dict[str, Array] = {
        "names": np.asarray(source_test_names),
        "physical": np.asarray(source_rollout["predictions"]),
        "compute_matched_physical": np.asarray(compute_rollout["predictions"]),
        "candidate": np.asarray(source_prediction["predictions"]),
        "coordinate_covariance_m2": np.asarray(
            source_prediction["coordinate_covariance_m2"]
        ),
        "calibrated_coordinate_covariance_m2": calibrated_covariance,
    }
    prediction_payload.update(
        {
            f"mechanism_{label}": np.asarray(values)
            for label, values in mechanism_predictions.items()
        }
    )
    prediction_payload.update(
        {
            deform_bayesian_covariance_archive_key(label): np.asarray(
                prediction["coordinate_covariance_m2"]
            )
            for label, prediction in bayesian_predictions.items()
        }
    )
    np.savez_compressed(predictions_path, **cast(dict[str, Any], prediction_payload))
    prediction_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-prediction-seal-v1",
        "seed": args.seed,
        "method_seal": _identity(cast(Path, artifacts["method_path"])),
        "recovery_method_seal": _identity(recovery_method_seal_path),
        "recovery_lock": _identity(lock_path),
        "predictions": _identity(predictions_path),
        "source_test_case_count": len(source_test_names),
        "bayesian_ablation_distributions": list(
            DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        ),
        "bayesian_covariance_archive_keys": {
            label: deform_bayesian_covariance_archive_key(label)
            for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        },
        "bayesian_point_means_identical": True,
        "source_outcomes_scored": False,
        "official_eval_read": False,
    }
    prediction_seal_path = output_root / "prediction_seal.json"
    _write_json(prediction_seal_path, prediction_seal)

    targets = np.asarray(source_rollout["targets"])
    baseline_predictions = np.asarray(source_rollout["predictions"])
    primary_gate = evaluate_deform_dlo3_source_gate(
        np.asarray(source_prediction["predictions"]),
        baseline_predictions,
        targets,
        source_test_names,
        protocol,
    )
    mechanism_results = {
        label: evaluate_deform_dlo3_source_gate(
            np.asarray(values),
            baseline_predictions,
            targets,
            source_test_names,
            protocol,
        )
        for label, values in mechanism_predictions.items()
    }
    bayesian_distributions = {
        label: evaluate_deform_predictive_distribution(
            np.asarray(prediction["predictions"]),
            targets,
            np.asarray(prediction["coordinate_covariance_m2"]),
        )
        for label, prediction in bayesian_predictions.items()
    }
    result = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-seed-result-v1",
        "claim_boundary": "DLO3 train source panel only; official evaluation unopened.",
        "seed": args.seed,
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "method_seal": _identity(cast(Path, artifacts["method_path"])),
        "recovery_method_seal": _identity(recovery_method_seal_path),
        "prediction_seal": _identity(prediction_seal_path),
        "physical_checkpoint": dict(
            _mapping(
                cast(Mapping[str, object], artifacts["method"]).get(
                    "physical_checkpoint"
                ),
                label="physical checkpoint",
            )
        ),
        "compute_match": {
            **compute_match,
            "checkpoint": dict(
                _mapping(
                    cast(Mapping[str, object], artifacts["method"]).get(
                        "compute_matched_checkpoint"
                    ),
                    label="compute checkpoint",
                )
            ),
            "source_mean_l1_m": _mean_l1(
                np.asarray(compute_rollout["predictions"]), targets
            ),
        },
        "primary_source_gate": primary_gate,
        "mechanism_ablation": mechanism_results,
        "bayesian_audit": {
            "calibration": calibration,
            "uncalibrated": bayesian_distributions[
                "trajectory-clustered-full-coordinate-covariance-v1"
            ],
            "calibrated": bayesian_distributions[
                "calibrated-full-coordinate-covariance-v1"
            ],
            "distributions": bayesian_distributions,
            "point_mean_unchanged": True,
            "distribution_selection": "none",
            "source_test_outcomes_used_for_covariance_construction": False,
        },
        "recovery": {
            "contract": "deform-dlo3-method-seal-source-completion-v1",
            "authorization": _identity(lock_path),
            "implementation_source_revision": decision[
                "implementation_source_revision"
            ],
            "implementation_archive": _identity(
                cast(Path, implementation_archive_path)
            ),
            "failure_receipt": _identity(failure_receipt_path),
            "parent_method_seal": _identity(cast(Path, artifacts["method_path"])),
            "retraining": False,
            "refitting": False,
            "checkpoint_continuation": False,
            "source_reselection": False,
            "case_replacement": False,
            "completion_count": 1,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "training_record": {
            "status": "reused-immutable-method-seal",
            "completed_updates": 6400,
            "failure_log": _identity(failure_log_path),
        },
        "checkpoints": [
            _identity(physical_path, update=6400),
            _identity(
                compute_path,
                update=int(cast(Any, compute_match["end_update"])),
            ),
        ],
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    _write_json(output_root / "source_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
