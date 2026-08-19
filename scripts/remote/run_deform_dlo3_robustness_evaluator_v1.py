#!/usr/bin/env python3
"""Dry-run or execute the one-shot frozen DLO3 robustness evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import run_deform_dlo3_pyelastica_source_v1 as backend_runtime
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deserialize_deform_local_residual_model,
)
from bayesian_phystwin_experiments.deform_dlo_pyelastica import (
    PyElasticaParameters,
    deform_pyelastica_parameter_bank,
    simulate_deform_pyelastica,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    build_deform_bayesian_covariance_ablation_v1,
    deform_bayesian_covariance_archive_key,
    evaluate_deform_backend_portability_report,
    evaluate_deform_compute_matched_report,
    evaluate_deform_dlo3_target_gate,
    evaluate_deform_predictive_distribution,
    load_deform_dlo_robustness_v1_protocol,
    predict_deform_local_residual_full_covariance,
    scale_deform_coordinate_covariance,
    validate_deform_bayesian_audit_v1,
    validate_deform_dlo3_alltrain_compute_match_v1,
    validate_deform_dlo3_source_manifest,
    verify_deform_dlo3_backend_artifacts_v1,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "official"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--alltrain-result", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--readiness-attestation", type=Path)
    parser.add_argument("--pyelastica-root", type=Path)
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


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked evaluator output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _load_final_method(
    result_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    result = _read_json(result_path)
    if (
        result.get("contract") != "deform-dlo3-robustness-alltrain-result-v1"
        or result.get("primary_eval_read") is not False
        or result.get("target_authorized") is not False
        or result.get("retry_authorized") is not False
        or result.get("bayesian_audit_complete") is not True
        or int(cast(Any, result.get("bayesian_distribution_count", -1)))
        != len(DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS)
    ):
        raise ValueError("DLO3 all-train result custody differs")
    method_path = _verified_file(result.get("final_method"), label="final method")
    method = _read_json(method_path)
    result_compute = dict(
        _mapping(result.get("compute_match"), label="alltrain compute match")
    )
    method_compute = dict(
        _mapping(method.get("compute_match"), label="method compute match")
    )
    result_compute_verification = dict(
        _mapping(
            result.get("compute_match_verification"),
            label="alltrain compute verification",
        )
    )
    method_compute_verification = dict(
        _mapping(
            method.get("compute_match_verification"),
            label="method compute verification",
        )
    )
    if (
        method.get("contract") != "deform-dlo3-robustness-alltrain-final-method-v1"
        or method.get("primary_eval_read") is not False
        or method.get("target_selection") is not False
        or method.get("target_calibration") is not False
        or method.get("target_retries") is not False
        or method.get("source_bayesian_audit_complete") is not True
        or method.get("source_diagnostics_verified") is not True
        or tuple(
            str(value)
            for value in cast(
                list[object], method.get("bayesian_ablation_distributions", [])
            )
        )
        != DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        or method.get("distribution_selection") != "none"
        or result_compute != method_compute
        or result_compute_verification != method_compute_verification
    ):
        raise ValueError("DLO3 final method custody differs")
    return result, method, method_path


def _load_backend_target_arm(
    alltrain: Mapping[str, object],
    method: Mapping[str, object],
    protocol: Mapping[str, object],
    pyelastica_root: Path | None,
) -> dict[str, object]:
    authorization = _mapping(
        alltrain.get("authorization"), label="alltrain authorization"
    )
    result_path = _verified_file(
        authorization.get("backend_result"), label="backend source result"
    )
    result = _read_json(result_path)
    artifacts = verify_deform_dlo3_backend_artifacts_v1(result, protocol)
    final_record = dict(
        _mapping(method.get("backend_target_arm"), label="final backend target arm")
    )
    if final_record != artifacts:
        raise ValueError("DLO3 final backend target arm differs")
    authorized = bool(artifacts["backend_target_arm_authorized"])
    if not authorized:
        return {
            "authorized": False,
            "source_result": _identity(result_path),
            "artifacts": artifacts,
        }
    if pyelastica_root is None:
        raise ValueError(
            "authorized PyElastica target arm requires its pinned checkout"
        )
    backend = _mapping(protocol.get("backend_portability"), label="backend")
    elastica = backend_runtime._load_pyelastica(
        pyelastica_root,
        str(backend["commit"]),
        str(backend["version"]),
    )
    model_path = _verified_file(
        artifacts.get("full_covariance_model"), label="backend full covariance model"
    )
    with np.load(model_path, allow_pickle=False) as archive:
        model = deserialize_deform_local_residual_model(archive)
        model["coefficient_covariance_full"] = np.asarray(
            archive["coefficient_covariance_full"]
        )
        model["residual_covariance_full"] = np.asarray(
            archive["residual_covariance_full"]
        )
    raw_parameters = _mapping(
        artifacts.get("selected_parameters"), label="backend selected parameters"
    )
    parameters = PyElasticaParameters(
        youngs_modulus_pa=float(cast(Any, raw_parameters["youngs_modulus_pa"])),
        density_kg_m3=float(cast(Any, raw_parameters["density_kg_m3"])),
        damping_constant=float(cast(Any, raw_parameters["damping_constant"])),
        integration_substeps=int(cast(Any, raw_parameters["integration_substeps"])),
    )
    if parameters not in deform_pyelastica_parameter_bank(protocol):
        raise ValueError("authorized PyElastica target parameters differ")
    raw_bank = _mapping(backend.get("parameter_bank"), label="backend parameter bank")
    return {
        "authorized": True,
        "source_result": _identity(result_path),
        "artifacts": artifacts,
        "elastica": elastica,
        "model": model,
        "parameters": parameters,
        "poisson_ratio": float(cast(Any, raw_bank["poisson_ratio"])),
        "radius_ratio": float(cast(Any, raw_bank["radius_to_mean_edge_ratio"])),
        "variance_scale": float(cast(Any, artifacts["variance_scale"])),
    }


def _build_eval_manifest(
    eval_root: Path,
    *,
    protocol_path: Path,
    alltrain_result_path: Path,
) -> dict[str, object]:
    paths = tuple(sorted(eval_root.glob("*.pkl"), key=lambda path: path.name))
    if len(paths) != 14:
        raise ValueError(
            f"official DLO3 evaluation expected 14 trajectories, got {len(paths)}"
        )
    identities = {
        path.name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    }
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-eval-manifest-v1",
        "partition": "DLO3/eval",
        "trajectory_policy": "all-fourteen-sorted-once-no-replacement",
        "trajectories": identities,
        "ordered_names": list(identities),
        "protocol": _identity(protocol_path),
        "alltrain_result": _identity(alltrain_result_path),
        "official_eval_read": True,
        "outcomes_scored": False,
    }


def _failure(stage: str, error: BaseException) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-official-failure-v1",
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
        "official_eval_read": True,
        "retry_authorized": False,
        "case_replacement": False,
    }


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    alltrain_path = args.alltrain_result.resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    alltrain, method, method_path = _load_final_method(alltrain_path)
    protocol_identity = _mapping(
        _mapping(alltrain.get("authorization"), label="alltrain authorization").get(
            "protocol"
        ),
        label="alltrain protocol",
    )
    if protocol_identity.get("sha256") != sha256_file(protocol_path):
        raise ValueError("DLO3 evaluator protocol lineage differs")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"evaluator output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    readiness_identity: dict[str, object] | None = None
    readiness: dict[str, object] | None = None
    if args.mode == "official":
        if args.readiness_attestation is None or args.source_manifest is not None:
            raise ValueError("official mode requires only a readiness attestation")
        readiness_path = args.readiness_attestation.resolve()
        readiness = _read_json(readiness_path)
        if (
            readiness.get("contract") != "deform-dlo3-robustness-readiness-v1"
            or readiness.get("target_authorized") is not True
            or readiness.get("official_eval_read") is not False
            or readiness.get("bayesian_audit_complete") is not True
            or readiness.get("bayesian_artifacts_verified") is not True
            or readiness.get("compute_matched_control_verified") is not True
            or _mapping(readiness.get("protocol"), label="readiness protocol").get(
                "sha256"
            )
            != sha256_file(protocol_path)
            or _mapping(
                readiness.get("alltrain_result"), label="readiness alltrain"
            ).get("sha256")
            != sha256_file(alltrain_path)
        ):
            raise ValueError("DLO3 readiness attestation differs")
        readiness_identity = _identity(readiness_path)
    elif args.source_manifest is None or args.readiness_attestation is not None:
        raise ValueError("dry-run mode requires only the source manifest")

    backend_arm = _load_backend_target_arm(
        alltrain,
        method,
        protocol,
        args.pyelastica_root.resolve() if args.pyelastica_root is not None else None,
    )
    backend_authorized = bool(backend_arm["authorized"])
    if readiness is not None and (
        readiness.get("backend_target_arm_authorized") is not backend_authorized
        or readiness.get("backend_artifacts") != backend_arm["artifacts"]
    ):
        raise ValueError("DLO3 readiness backend target arm differs")

    runtime = _mapping(method.get("runtime"), label="method runtime")
    training = _mapping(protocol.get("physical_training"), label="training")
    cublas_config = str(training["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    if torch.__version__ != runtime.get("torch") or torch.version.cuda != runtime.get(
        "cuda"
    ):
        raise RuntimeError("DLO3 evaluator runtime differs from all-train method")
    checkpoint_path = _verified_file(
        method.get("physical_checkpoint"), label="physical checkpoint"
    )
    compute_checkpoint_path = _verified_file(
        method.get("compute_matched_checkpoint"),
        label="compute-matched checkpoint",
    )
    compute_match_path = _verified_file(
        method.get("compute_match"), label="compute-matched record"
    )
    compute_match = _read_json(compute_match_path)
    schedule_path = _verified_file(method.get("window_schedule"), label="schedule")
    method_spec_path = _verified_file(method.get("method_spec"), label="method spec")
    method_spec = _read_json(method_spec_path)
    local_model_path = _verified_file(
        method.get("full_covariance_model"), label="full covariance model"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    compute_checkpoint = torch.load(
        compute_checkpoint_path, map_location="cpu", weights_only=True
    )
    compute_contract = _mapping(
        protocol.get("compute_matched_control"), label="compute-matched control"
    )
    compute_verification = validate_deform_dlo3_alltrain_compute_match_v1(
        compute_match, protocol
    )
    compute_updates = int(cast(Any, compute_verification["additional_updates"]))
    registered_updates = int(cast(Any, training["total_updates"]))
    primary_seed = int(cast(Any, training["primary_seed"]))
    if (
        method_spec.get("contract") != "deform-dlo3-robustness-alltrain-method-spec-v1"
        or method_spec.get("compute_matched_control")
        != "frozen-wall-time-equivalent-DEFORM-continuation-v1"
        or method_spec.get("official_eval_read") is not False
        or method.get("compute_match_verification") != compute_verification
        or alltrain.get("compute_match_verification") != compute_verification
        or dict(_mapping(alltrain.get("compute_match"), label="alltrain compute"))
        != dict(_mapping(method.get("compute_match"), label="method compute"))
    ):
        raise ValueError("DLO3 compute-matched method record differs")
    with np.load(schedule_path, allow_pickle=False) as schedule:
        fit_names = [str(value) for value in np.asarray(schedule["fit_names"])]
        trajectory_indices = np.asarray(schedule["trajectory_indices"])
        start_indices = np.asarray(schedule["start_indices"])
    alltrain_authorization = _mapping(
        alltrain.get("authorization"), label="alltrain authorization"
    )
    source_manifest_path = _verified_file(
        alltrain_authorization.get("source_manifest"), label="source manifest"
    )
    source_manifest = _read_json(source_manifest_path)
    source_partitions = validate_deform_dlo3_source_manifest(
        source_manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=False,
    )
    expected_fit_names = sorted(
        name for values in source_partitions.values() for name in values
    )
    maximum_extra = int(cast(Any, compute_contract["maximum_additional_updates"]))
    expected_trajectory_indices, expected_start_indices = source_runtime._make_schedule(
        fit_names=fit_names,
        updates=registered_updates + maximum_extra,
        batch_size=int(cast(Any, training["batch_size"])),
        frame_count=int(
            cast(Any, _mapping(protocol["data"], label="data")["frame_count"])
        ),
        horizon=int(cast(Any, training["unroll_horizon_frames"])),
        seed=primary_seed,
    )
    if (
        fit_names != expected_fit_names
        or len(fit_names) != 56
        or len(set(fit_names)) != 56
        or not np.array_equal(trajectory_indices, expected_trajectory_indices)
        or not np.array_equal(start_indices, expected_start_indices)
    ):
        raise ValueError("DLO3 compute-matched schedule continuation differs")
    checkpoint_identities = tuple(
        dict(_mapping(value, label="alltrain checkpoint"))
        for value in cast(list[object], alltrain.get("checkpoints", []))
    )
    physical_identity = dict(
        _mapping(method.get("physical_checkpoint"), label="physical checkpoint")
    )
    compute_identity = dict(
        _mapping(
            method.get("compute_matched_checkpoint"),
            label="compute-matched checkpoint",
        )
    )
    if (
        checkpoint.get("update") != registered_updates
        or checkpoint.get("seed") != primary_seed
        or checkpoint.get("protocol_sha256") != sha256_file(protocol_path)
        or checkpoint.get("schedule_sha256") != sha256_file(schedule_path)
        or checkpoint.get("method_spec_sha256") != sha256_file(method_spec_path)
        or checkpoint.get("official_eval_read") is not False
    ):
        raise ValueError("DLO3 evaluator checkpoint lineage differs")
    if (
        compute_checkpoint.get("update") != registered_updates + compute_updates
        or compute_checkpoint.get("seed") != primary_seed
        or compute_checkpoint.get("protocol_sha256") != sha256_file(protocol_path)
        or compute_checkpoint.get("schedule_sha256") != sha256_file(schedule_path)
        or compute_checkpoint.get("method_spec_sha256") != sha256_file(method_spec_path)
        or compute_checkpoint.get("official_eval_read") is not False
        or physical_identity not in checkpoint_identities
        or compute_identity not in checkpoint_identities
    ):
        raise ValueError("DLO3 compute-matched checkpoint lineage differs")
    if readiness is not None and (
        readiness.get("physical_checkpoint") != _identity(checkpoint_path)
        or readiness.get("compute_matched_checkpoint")
        != _identity(compute_checkpoint_path)
        or readiness.get("compute_matched_record") != _identity(compute_match_path)
        or readiness.get("compute_match_verification") != compute_verification
        or _mapping(
            readiness.get("compute_matched_dry_run"),
            label="readiness compute-matched dry run",
        ).get("status")
        != "scored"
    ):
        raise ValueError("DLO3 readiness compute-matched lineage differs")
    with np.load(local_model_path, allow_pickle=False) as archive:
        full_model = deserialize_deform_local_residual_model(archive)
        full_model["coefficient_covariance_full"] = np.asarray(
            archive["coefficient_covariance_full"]
        )
        full_model["residual_covariance_full"] = np.asarray(
            archive["residual_covariance_full"]
        )

    authorization = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-evaluator-authorization-v1",
        "mode": args.mode,
        "protocol": _identity(protocol_path),
        "alltrain_result": _identity(alltrain_path),
        "final_method": _identity(method_path),
        "readiness_attestation": readiness_identity,
        "backend_source_result": backend_arm["source_result"],
        "backend_target_arm_authorized": backend_authorized,
        "compute_matched_checkpoint": _identity(compute_checkpoint_path),
        "compute_matched_record": _identity(compute_match_path),
        "compute_match_verification": compute_verification,
        "one_shot_execution_authorized": args.mode == "official",
        "count_only_custody_deviation_acknowledged": True,
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "case_replacement": False,
        "official_eval_read": False,
    }
    authorization_path = output_root / "authorization.json"
    _write_json(authorization_path, authorization)

    source_runtime._assert_upstream(
        args.upstream_root,
        str(_mapping(protocol["upstream"], label="upstream")["commit"]),
    )
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(torch, 42)
    stage = "panel-authorization"
    started = time.perf_counter()
    try:
        if args.mode == "dry-run":
            data_root = args.upstream_root.resolve() / "data_set"
            source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")
            source_runtime._install_eval_read_guard(data_root / "DLO4")
            source_runtime._install_eval_read_guard(data_root / "DLO5")
            source_manifest_path = cast(Path, args.source_manifest).resolve()
            source_manifest = _read_json(source_manifest_path)
            partitions = validate_deform_dlo3_source_manifest(
                source_manifest,
                protocol,
                protocol_sha256=sha256_file(protocol_path),
                verify_files=True,
            )
            names = list(partitions["source_test"])
            panel_manifest = source_manifest
            panel_identity: dict[str, object] = _identity(source_manifest_path)
        else:
            stage = "target-manifest"
            eval_root = args.upstream_root.resolve() / "data_set" / "DLO3" / "eval"
            panel_manifest = _build_eval_manifest(
                eval_root,
                protocol_path=protocol_path,
                alltrain_result_path=alltrain_path,
            )
            manifest_path = output_root / "evaluation_manifest.json"
            _write_json(manifest_path, panel_manifest)
            names = list(cast(list[str], panel_manifest["ordered_names"]))
            panel_identity = _identity(manifest_path)
        stage = "panel-load"
        trajectories = source_runtime._load_named_trajectories(
            panel_manifest,
            names,
            frame_count=500,
            node_count=12,
        )
        stage = "physical-rollout"
        baseline_rollout = posterior_runtime._evaluate_state(
            checkpoint["model_state_dict"],
            trajectories,
            modules=modules,
            torch=torch,
            device=args.device,
            dlo_type="DLO3",
            node_count=12,
        )
        stage = "compute-matched-rollout"
        try:
            compute_rollout = posterior_runtime._evaluate_state(
                compute_checkpoint["model_state_dict"],
                trajectories,
                modules=modules,
                torch=torch,
                device=args.device,
                dlo_type="DLO3",
                node_count=12,
            )
            compute_values = np.asarray(compute_rollout["predictions"])
            if (
                compute_values.shape
                != np.asarray(baseline_rollout["predictions"]).shape
                or not np.isfinite(compute_values).all()
            ):
                raise ValueError("compute-matched rollout is invalid")
            compute_prediction_record: dict[str, object] = {
                "status": "sealed",
                "checkpoint": _identity(compute_checkpoint_path),
                "compute_match": _identity(compute_match_path),
                "compute_match_verification": compute_verification,
                "selection_effect": "none",
                "retry_authorized": False,
            }
        except Exception as error:
            if args.mode == "dry-run":
                raise
            compute_values = None
            compute_prediction_record = {
                "status": "technical-failure",
                "stage": stage,
                "exception_type": type(error).__name__,
                "message": str(error),
                "checkpoint": _identity(compute_checkpoint_path),
                "compute_match": _identity(compute_match_path),
                "compute_match_verification": compute_verification,
                "selection_effect": "none",
                "retry_authorized": False,
            }
        initial, action = local_runtime._causal_inputs(trajectories, names)
        bayesian_predictions = build_deform_bayesian_covariance_ablation_v1(
            full_model,
            initial,
            action,
            np.asarray(baseline_rollout["predictions"]),
            shrinkage=float(cast(Any, method["shrinkage"])),
            variance_scale=float(cast(Any, method["variance_scale"])),
        )
        prediction = bayesian_predictions[
            "trajectory-clustered-full-coordinate-covariance-v1"
        ]
        calibrated_covariance = np.asarray(
            bayesian_predictions["calibrated-full-coordinate-covariance-v1"][
                "coordinate_covariance_m2"
            ]
        )
        backend_values: dict[str, np.ndarray] | None = None
        if not backend_authorized:
            backend_prediction_record: dict[str, object] = {
                "status": "not-authorized",
                "source_gate_authorized": False,
                "selection_effect": "none",
                "retry_authorized": False,
            }
        else:
            stage = "backend-portability-rollout"
            try:
                backend_baseline = np.stack(
                    [
                        simulate_deform_pyelastica(
                            trajectories[name],
                            cast(PyElasticaParameters, backend_arm["parameters"]),
                            elastica=backend_arm["elastica"],
                            poisson_ratio=float(
                                cast(Any, backend_arm["poisson_ratio"])
                            ),
                            radius_to_mean_edge_ratio=float(
                                cast(Any, backend_arm["radius_ratio"])
                            ),
                        )
                        for name in names
                    ]
                )
                residual = _mapping(
                    protocol.get("local_residual"), label="local residual"
                )
                backend_raw = predict_deform_local_residual_full_covariance(
                    cast(Mapping[str, object], backend_arm["model"]),
                    initial,
                    action,
                    backend_baseline,
                    shrinkage=float(cast(Any, residual["shrinkage"])),
                )
                backend_calibrated_covariance = scale_deform_coordinate_covariance(
                    backend_raw["coordinate_covariance_m2"],
                    float(cast(Any, backend_arm["variance_scale"])),
                )
                backend_values = {
                    "baseline": np.asarray(backend_baseline),
                    "candidate": np.asarray(backend_raw["predictions"]),
                    "coordinate_covariance_m2": np.asarray(
                        backend_raw["coordinate_covariance_m2"]
                    ),
                    "calibrated_coordinate_covariance_m2": np.asarray(
                        backend_calibrated_covariance
                    ),
                }
                backend_prediction_record = {
                    "status": "sealed",
                    "source_gate_authorized": True,
                    "selection_effect": "none",
                    "retry_authorized": False,
                }
            except Exception as error:
                if args.mode == "dry-run":
                    raise
                backend_prediction_record = {
                    "status": "technical-failure",
                    "source_gate_authorized": True,
                    "stage": stage,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "selection_effect": "none",
                    "retry_authorized": False,
                }
        predictions_path = output_root / "predictions.npz"
        prediction_payload: dict[str, Any] = {
            "names": np.asarray(names),
            "baseline": baseline_rollout["predictions"],
            "candidate": prediction["predictions"],
            "calibrated_coordinate_covariance_m2": calibrated_covariance,
        }
        if compute_values is not None:
            prediction_payload["compute_matched_physical"] = compute_values
        prediction_payload.update(
            {
                deform_bayesian_covariance_archive_key(label): np.asarray(
                    values["coordinate_covariance_m2"]
                )
                for label, values in bayesian_predictions.items()
            }
        )
        if backend_values is not None:
            prediction_payload.update(
                {
                    f"backend_pyelastica_{key}": values
                    for key, values in backend_values.items()
                }
            )
        np.savez_compressed(predictions_path, **prediction_payload)
        prediction_seal = {
            "schema_version": 1,
            "contract": "deform-dlo3-robustness-evaluator-prediction-seal-v1",
            "mode": args.mode,
            "authorization": _identity(authorization_path),
            "panel_manifest": panel_identity,
            "predictions": _identity(predictions_path),
            "bayesian_ablation_distributions": list(
                DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
            ),
            "bayesian_covariance_archive_keys": {
                label: deform_bayesian_covariance_archive_key(label)
                for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
            },
            "bayesian_point_means_identical": True,
            "backend_portability": backend_prediction_record,
            "compute_matched_control": compute_prediction_record,
            "outcomes_scored": False,
            "target_retries": False,
        }
        prediction_seal_path = output_root / "prediction_seal.json"
        _write_json(prediction_seal_path, prediction_seal)

        targets = np.asarray(baseline_rollout["targets"])
        if compute_values is None:
            compute_matched_control = compute_prediction_record
        else:
            compute_matched_control = {
                "status": "scored",
                "checkpoint": _identity(compute_checkpoint_path),
                "compute_match": _identity(compute_match_path),
                "compute_match_verification": compute_verification,
                "selection_effect": "none",
                "retry_authorized": False,
                "report": evaluate_deform_compute_matched_report(
                    np.asarray(prediction["predictions"]),
                    np.asarray(baseline_rollout["predictions"]),
                    compute_values,
                    targets,
                    names,
                ),
            }
        if backend_values is None:
            backend_portability = backend_prediction_record
        else:
            backend_portability = {
                "status": "scored",
                "source_gate_authorized": True,
                "selection_effect": "none",
                "retry_authorized": False,
                "report": evaluate_deform_backend_portability_report(
                    backend_values["candidate"],
                    backend_values["baseline"],
                    targets,
                    names,
                ),
                "uncalibrated": evaluate_deform_predictive_distribution(
                    backend_values["candidate"],
                    targets,
                    backend_values["coordinate_covariance_m2"],
                ),
                "calibrated": evaluate_deform_predictive_distribution(
                    backend_values["candidate"],
                    targets,
                    backend_values["calibrated_coordinate_covariance_m2"],
                ),
                "point_mean_unchanged_by_calibration": True,
            }
        bayesian_distributions = {
            label: evaluate_deform_predictive_distribution(
                values["predictions"],
                targets,
                values["coordinate_covariance_m2"],
            )
            for label, values in bayesian_predictions.items()
        }
        distribution = bayesian_distributions[
            "calibrated-full-coordinate-covariance-v1"
        ]
        bayesian_audit = {
            "primary_distribution": "calibrated-full-coordinate-covariance-v1",
            "distributions": bayesian_distributions,
            "point_mean_unchanged": True,
            "distribution_selection": "none",
            "target_outcomes_used_for_distribution_construction": False,
            "target_outcomes_used_for_distribution_selection": False,
        }
        bayesian_audit_verification = validate_deform_bayesian_audit_v1(
            {"bayesian_audit": bayesian_audit}, context="evaluator"
        )
        if args.mode == "dry-run":
            result = {
                "schema_version": 1,
                "contract": "deform-dlo3-robustness-evaluator-dry-run-v1",
                "claim_boundary": "Already-open DLO3 source-test only; nonconfirmatory.",
                "authorization": _identity(authorization_path),
                "prediction_seal": _identity(prediction_seal_path),
                "pipeline_passed": True,
                "distribution": distribution,
                "bayesian_audit": bayesian_audit,
                "bayesian_audit_verification": bayesian_audit_verification,
                "backend_portability": backend_portability,
                "compute_matched_control": compute_matched_control,
                "runtime": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": args.device,
                },
                "primary_eval_enumerated_by_this_runner": False,
                "primary_eval_read": False,
                "target_authorized": False,
                "retry_authorized": False,
                "held_v8_access": False,
            }
            result_path = output_root / "dry_run_result.json"
        else:
            gate = evaluate_deform_dlo3_target_gate(
                prediction["predictions"],
                np.asarray(baseline_rollout["predictions"]),
                targets,
                names,
                protocol,
            )
            result = {
                "schema_version": 1,
                "contract": "deform-dlo3-robustness-official-result-v1",
                "claim_boundary": "One-shot DLO3 official evaluation with disclosed count-only metadata deviation.",
                "authorization": _identity(authorization_path),
                "evaluation_manifest": panel_identity,
                "prediction_seal": _identity(prediction_seal_path),
                "target_gate": gate,
                "distribution": distribution,
                "bayesian_audit": bayesian_audit,
                "bayesian_audit_verification": bayesian_audit_verification,
                "backend_portability": backend_portability,
                "compute_matched_control": compute_matched_control,
                "runtime": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": args.device,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                "official_eval_read": True,
                "retry_authorized": False,
                "case_replacement": False,
                "prob4d_used": False,
                "held_v8_access": False,
            }
            result_path = output_root / "official_result.json"
        _write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        if args.mode == "official":
            _write_json(output_root / "failure.json", _failure(stage, error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
