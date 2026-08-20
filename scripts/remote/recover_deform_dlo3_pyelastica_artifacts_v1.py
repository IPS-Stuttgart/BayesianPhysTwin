#!/usr/bin/env python3
"""Persist the full covariance already used by the sealed DLO3 PyElastica arm."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import run_deform_dlo3_pyelastica_source_v1 as backend_runtime
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deserialize_deform_local_residual_model,
    serialize_deform_local_residual_model,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    augment_deform_local_residual_full_covariance,
    load_deform_dlo_robustness_v1_protocol,
    predict_deform_local_residual_full_covariance,
    scale_deform_coordinate_covariance,
    validate_deform_dlo3_backend_result_v1,
    validate_deform_dlo3_source_manifest,
    verify_deform_dlo3_backend_artifacts_v1,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

Array = np.ndarray[Any, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--pyelastica-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "recover"), default="recover")
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
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verified_path(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if (
        identity.get("sha256") != sha256_file(path)
        or identity.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(f"{label} identity differs")
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked recovery output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _copy_exact(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"recovery copy differs: {destination}")


def _locked_digest(
    lock: Mapping[str, object], key: str, path: Path, *, label: str
) -> None:
    digests = _mapping(
        lock.get("original_artifact_sha256s"), label="recovery artifact digests"
    )
    if digests.get(key) != sha256_file(path):
        raise ValueError(f"{label} differs from the recovery lock")


def _assert_array_equivalent(
    actual: Array,
    expected: Array,
    *,
    reduction_terms: int,
    label: str,
) -> dict[str, object]:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    epsilon = float(np.finfo(np.float64).eps)
    if reduction_terms < 1 or reduction_terms * epsilon >= 1.0:
        raise ValueError(f"{label} reduction count is invalid")
    gamma = reduction_terms * epsilon / (1.0 - reduction_terms * epsilon)
    scale = max(1.0, float(np.max(np.abs(right))))
    tolerance = gamma * scale
    maximum_delta = (
        float(np.max(np.abs(left - right))) if left.shape == right.shape else math.inf
    )
    if left.shape != right.shape or not np.allclose(
        left, right, rtol=0.0, atol=tolerance
    ):
        raise ValueError(f"{label} exceeds its floating-point replay bound")
    return {
        "label": label,
        "reduction_terms": reduction_terms,
        "machine_epsilon": epsilon,
        "gamma_n": gamma,
        "reference_scale": scale,
        "absolute_tolerance": tolerance,
        "maximum_absolute_delta": maximum_delta,
        "numerically_equivalent": True,
    }


def _selected_parameter(
    protocol: Mapping[str, object], selection: Mapping[str, object]
) -> Any:
    selected = dict(
        _mapping(selection.get("selected_parameters"), label="selected parameters")
    )
    bank = backend_runtime.deform_pyelastica_parameter_bank(protocol)
    matches = [member for member in bank if member.to_record() == selected]
    if len(matches) != 1:
        raise ValueError("sealed PyElastica parameters differ from the frozen bank")
    selected_index = int(cast(Any, selection.get("selected_index", -1)))
    members = cast(Sequence[object], selection.get("members", ()))
    if (
        selection.get("contract") != "deform-dlo3-pyelastica-fit-selection-v1"
        or selection.get("selection_partition") != "fit-only"
        or selection.get("tie_break") != "first-frozen-bank-index"
        or selected_index < 0
        or selected_index >= len(members)
        or selected_index >= len(bank)
        or bank[selected_index].to_record() != selected
        or selection.get("source_test_opened") is not False
        or selection.get("primary_eval_read") is not False
    ):
        raise ValueError("sealed PyElastica fit selection differs")
    return matches[0]


def main() -> int:
    args = _parse_args()
    lock_path = args.recovery_lock.resolve()
    protocol_path = args.protocol.resolve()
    manifest_path = args.source_manifest.resolve()
    original_result_path = args.original_result.resolve()
    lock = _read_json(lock_path)
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    manifest = _read_json(manifest_path)
    if (
        lock.get("contract") != "deform-dlo3-pyelastica-artifact-recovery-v2"
        or lock.get("permitted_operation")
        != "persist-already-computed-full-covariance-and-reseal-byte-identical-predictions"
        or lock.get("protocol_sha256") != sha256_file(protocol_path)
        or lock.get("source_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("PyElastica recovery lock differs")
    custody = _mapping(lock.get("custody"), label="recovery custody")
    if any(value is not False for value in custody.values()):
        raise ValueError("PyElastica recovery custody must remain closed")
    required_replay = _mapping(
        lock.get("required_replay"), label="required recovery replay"
    )
    if (
        any(
            required_replay.get(key) is not True
            for key in (
                "selected_fit_error_exact",
                "point_predictions_numerically_equivalent",
                "coordinate_covariance_numerically_equivalent",
                "calibrated_coordinate_covariance_numerically_equivalent",
                "source_predictions_file_byte_identical",
            )
        )
        or required_replay.get("source_gate_recomputed") is not False
    ):
        raise ValueError("PyElastica recovery replay contract differs")
    equivalence = _mapping(
        lock.get("floating_point_equivalence"),
        label="floating-point replay equivalence",
    )
    epsilon = float(np.finfo(np.float64).eps)
    point_reduction_terms = int(cast(Any, equivalence.get("point_reduction_terms", -1)))
    covariance_reduction_terms = int(
        cast(Any, equivalence.get("covariance_reduction_terms", -1))
    )
    if (
        equivalence.get("dtype") != "float64"
        or equivalence.get("formula") != "gamma_n_times_max_one_reference_scale"
        or float(cast(Any, equivalence.get("machine_epsilon", math.nan))) != epsilon
        or float(cast(Any, equivalence.get("relative_tolerance", math.nan))) != 0.0
        or point_reduction_terms != 93
        or covariance_reduction_terms != 39 * 498
    ):
        raise ValueError("PyElastica floating-point replay contract differs")
    _locked_digest(
        lock, "source_result", original_result_path, label="original source result"
    )
    original = _read_json(original_result_path)
    validate_deform_dlo3_backend_result_v1(original, protocol)
    original_protocol = _mapping(original.get("protocol"), label="original protocol")
    if (
        original_protocol.get("sha256") != sha256_file(protocol_path)
        or original.get("primary_eval_read") is not False
        or original.get("retry_authorized") is not False
        or original.get("held_v8_access") is not False
    ):
        raise ValueError("original PyElastica result custody differs")

    method_path = _verified_path(original.get("method_seal"), label="original method")
    prediction_seal_path = _verified_path(
        original.get("prediction_seal"), label="original prediction seal"
    )
    _locked_digest(lock, "method_seal", method_path, label="original method")
    _locked_digest(
        lock,
        "prediction_seal",
        prediction_seal_path,
        label="original prediction seal",
    )
    method = _read_json(method_path)
    prediction_seal = _read_json(prediction_seal_path)
    fit_selection_path = _verified_path(
        method.get("fit_selection"), label="original fit selection"
    )
    local_model_path = _verified_path(
        method.get("local_residual_model"), label="original local model"
    )
    calibration_path = _verified_path(
        method.get("covariance_calibration"), label="original calibration"
    )
    predictions_path = _verified_path(
        prediction_seal.get("predictions"), label="original predictions"
    )
    for key, path, label in (
        ("fit_selection", fit_selection_path, "fit selection"),
        ("local_residual_model", local_model_path, "local model"),
        ("covariance_calibration", calibration_path, "calibration"),
        ("source_predictions", predictions_path, "source predictions"),
    ):
        _locked_digest(lock, key, path, label=label)
    if (
        method.get("contract") != "deform-dlo3-pyelastica-source-method-seal-v1"
        or "full_covariance_model" in method
        or method.get("source_test_opened") is not False
        or method.get("primary_eval_read") is not False
        or method.get("selection_effect_after_fit") != "none"
        or prediction_seal.get("contract")
        != "deform-dlo3-pyelastica-source-prediction-seal-v1"
        or _mapping(
            prediction_seal.get("method_seal"), label="sealed original method"
        ).get("sha256")
        != sha256_file(method_path)
        or prediction_seal.get("source_outcomes_scored") is not False
        or prediction_seal.get("primary_eval_read") is not False
    ):
        raise ValueError("original PyElastica seal lineage differs")

    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=True,
    )
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"PyElastica recovery output root is not empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    upstream = source_runtime._assert_upstream(
        args.upstream_root,
        str(_mapping(protocol["upstream"], label="upstream")["commit"]),
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO4")
    source_runtime._install_eval_read_guard(data_root / "DLO5")
    backend = _mapping(protocol.get("backend_portability"), label="backend")
    elastica = backend_runtime._load_pyelastica(
        args.pyelastica_root,
        str(backend["commit"]),
        str(backend["version"]),
    )
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-artifact-recovery-preflight-v2",
        "mode": args.mode,
        "recovery_lock": _identity(lock_path),
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "original_source_result": _identity(original_result_path),
        "original_method_seal": _identity(method_path),
        "original_prediction_seal": _identity(prediction_seal_path),
        "original_predictions": _identity(predictions_path),
        "upstream": upstream,
        "fit_parameter_reselection": False,
        "source_score_recomputation": False,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    selection = _read_json(fit_selection_path)
    selected = _selected_parameter(protocol, selection)
    fit_names = list(partitions["fit"])
    fit = source_runtime._load_named_trajectories(
        manifest, fit_names, frame_count=500, node_count=12
    )
    raw_bank = _mapping(backend.get("parameter_bank"), label="parameter bank")
    fit_predictions = backend_runtime._simulate_panel(
        fit,
        fit_names,
        selected,
        elastica=elastica,
        poisson_ratio=float(cast(Any, raw_bank["poisson_ratio"])),
        radius_ratio=float(cast(Any, raw_bank["radius_to_mean_edge_ratio"])),
    )
    fit_targets = backend_runtime._stack_targets(fit, fit_names)
    fit_error = float(np.mean(np.abs(fit_predictions - fit_targets)))
    if fit_error != float(cast(Any, selection["selected_fit_mean_l1_m"])):
        raise ValueError("selected PyElastica fit error does not replay exactly")
    fit_initial, fit_action = local_runtime._causal_inputs(fit, fit_names)
    with np.load(local_model_path, allow_pickle=False) as archive:
        local_model = deserialize_deform_local_residual_model(archive)
    full_model = augment_deform_local_residual_full_covariance(
        local_model,
        fit_initial,
        fit_action,
        fit_predictions,
        fit_targets,
        fit_names,
    )
    full_model_path = output_root / "full_covariance_model.npz"
    full_payload = serialize_deform_local_residual_model(full_model)
    full_payload["coefficient_covariance_full"] = np.asarray(
        full_model["coefficient_covariance_full"]
    )
    full_payload["residual_covariance_full"] = np.asarray(
        full_model["residual_covariance_full"]
    )
    np.savez_compressed(full_model_path, **full_payload)

    with np.load(predictions_path, allow_pickle=False) as archive:
        source_names = [str(value) for value in np.asarray(archive["names"])]
        sealed_backend = np.asarray(archive["backend"])
        sealed_candidate = np.asarray(archive["candidate"])
        sealed_covariance = np.asarray(archive["coordinate_covariance_m2"])
        sealed_calibrated = np.asarray(archive["calibrated_coordinate_covariance_m2"])
    if source_names != list(partitions["source_test"]):
        raise ValueError("sealed PyElastica source names differ")
    source_panel = source_runtime._load_named_trajectories(
        manifest, source_names, frame_count=500, node_count=12
    )
    source_initial, source_action = local_runtime._causal_inputs(
        source_panel, source_names
    )
    replay = predict_deform_local_residual_full_covariance(
        full_model,
        source_initial,
        source_action,
        sealed_backend,
        shrinkage=float(cast(Any, method["shrinkage"])),
    )
    calibration = _read_json(calibration_path)
    replay_calibrated = scale_deform_coordinate_covariance(
        replay["coordinate_covariance_m2"],
        float(cast(Any, calibration["variance_scale"])),
    )
    point_equivalence = _assert_array_equivalent(
        np.asarray(replay["predictions"]),
        sealed_candidate,
        reduction_terms=point_reduction_terms,
        label="point prediction",
    )
    covariance_equivalence = _assert_array_equivalent(
        np.asarray(replay["coordinate_covariance_m2"]),
        sealed_covariance,
        reduction_terms=covariance_reduction_terms,
        label="coordinate covariance",
    )
    calibrated_equivalence = _assert_array_equivalent(
        replay_calibrated,
        sealed_calibrated,
        reduction_terms=covariance_reduction_terms,
        label="calibrated coordinate covariance",
    )

    copied_selection = output_root / "fit_selection.json"
    copied_calibration = output_root / "covariance_calibration.json"
    copied_predictions = output_root / "source_predictions.npz"
    _copy_exact(fit_selection_path, copied_selection)
    _copy_exact(calibration_path, copied_calibration)
    _copy_exact(predictions_path, copied_predictions)
    recovery_lineage = {
        "contract": "deform-dlo3-pyelastica-artifact-recovery-replay-v2",
        "recovery_lock": _identity(lock_path),
        "original_source_result": _identity(original_result_path),
        "original_method_seal": _identity(method_path),
        "original_prediction_seal": _identity(prediction_seal_path),
        "selected_fit_error_exact": True,
        "point_prediction_equivalence": point_equivalence,
        "coordinate_covariance_equivalence": covariance_equivalence,
        "calibrated_coordinate_covariance_equivalence": calibrated_equivalence,
        "source_predictions_file_byte_identical": True,
        "source_gate_recomputed": False,
        "fit_parameter_reselection": False,
        "primary_eval_read": False,
    }
    recovered_method = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-method-seal-v1",
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "fit_selection": _identity(copied_selection),
        "full_covariance_model": _identity(full_model_path),
        "covariance_calibration": _identity(copied_calibration),
        "selected_parameters": dict(
            _mapping(method["selected_parameters"], label="selected parameters")
        ),
        "ridge": float(cast(Any, method["ridge"])),
        "shrinkage": float(cast(Any, method["shrinkage"])),
        "artifact_recovery": recovery_lineage,
        "source_test_opened": False,
        "primary_eval_read": False,
        "selection_effect_after_fit": "none",
    }
    recovered_method_path = output_root / "method_seal.json"
    _write_json(recovered_method_path, recovered_method)
    recovered_prediction_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-prediction-seal-v1",
        "method_seal": _identity(recovered_method_path),
        "predictions": _identity(copied_predictions),
        "artifact_recovery": recovery_lineage,
        "source_outcomes_scored": False,
        "primary_eval_read": False,
    }
    recovered_prediction_seal_path = output_root / "prediction_seal.json"
    _write_json(recovered_prediction_seal_path, recovered_prediction_seal)
    recovered_result = {
        **original,
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "method_seal": _identity(recovered_method_path),
        "prediction_seal": _identity(recovered_prediction_seal_path),
        "artifact_recovery": recovery_lineage,
    }
    recovered_result_path = output_root / "source_result.json"
    _write_json(recovered_result_path, recovered_result)
    verification = verify_deform_dlo3_backend_artifacts_v1(recovered_result, protocol)
    receipt = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-artifact-recovery-receipt-v2",
        "recovery": recovery_lineage,
        "recovered_source_result": _identity(recovered_result_path),
        "recovered_method_seal": _identity(recovered_method_path),
        "recovered_prediction_seal": _identity(recovered_prediction_seal_path),
        "recovered_full_covariance_model": _identity(full_model_path),
        "backend_artifact_verification": verification,
        "source_score_recomputation": False,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    _write_json(output_root / "recovery_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
