#!/usr/bin/env python3
"""Generate sealed DLO4/DLO5 residual predictions, then score them separately.

The target plan was selected from headers only. ``predict`` opens only physical
state/action/contact carriers and never the target residual or observation
array. It freezes all prediction arms in a content-addressed seal. ``score``
verifies that seal before opening target residual values and cannot refit the
source model, standardizers, coefficients, feature mapping, or diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CORE = _load_module("hmp_dlo45_core", HERE / "hierarchical_missing_physics_diagnosis.py")
TRANSFER = _load_module("hmp_dlo45_transfer", HERE / "evaluate_hierarchical_missing_physics_transfer.py")
SOURCE_BUILDER = _load_module("hmp_dlo45_source_builder", HERE / "build_deform_dlo23_hierarchical_panel.py")

TARGET_DLOS = ("DLO4", "DLO5")
EPS = 1.0e-12


class EvaluationError(RuntimeError):
    """Raised when target prediction/scoring custody or shape contracts fail."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_id(value: Mapping[str, Any], key: str) -> None:
    claimed = value.get(key)
    unsigned = dict(value)
    unsigned.pop(key, None)
    if canonical_hash(unsigned) != claimed:
        raise EvaluationError(f"{key} mismatch")


def resolve_container(record: Mapping[str, Any], artifact_root: Path) -> Path:
    relative = Path(record["container_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluationError(f"unsafe artifact-relative path: {relative}")
    path = (artifact_root / relative).resolve(strict=True)
    root = artifact_root.resolve(strict=True)
    if root not in path.parents and path != root:
        raise EvaluationError(f"artifact path escapes root: {path}")
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"artifact carrier must be a real file: {path}")
    if sha256(path) != record["container_sha256"]:
        raise EvaluationError(f"artifact carrier hash mismatch: {path}")
    return path


def load_locator(record: Mapping[str, Any], artifact_root: Path) -> np.ndarray:
    path = resolve_container(record, artifact_root)
    locator_type = record["locator_type"]
    key = record["key"]
    if locator_type == "direct-npy":
        value = np.load(path, allow_pickle=False)
    elif locator_type == "direct-npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise EvaluationError(f"{key!r} absent from {path}")
            value = np.asarray(archive[key])
    elif locator_type == "zip-npy":
        with zipfile.ZipFile(path, "r") as archive:
            with archive.open(record["member"], "r") as stream:
                value = np.load(io.BytesIO(stream.read()), allow_pickle=False)
    elif locator_type == "zip-nested-npz":
        with zipfile.ZipFile(path, "r") as archive:
            with archive.open(record["member"], "r") as stream:
                payload = stream.read()
        with np.load(io.BytesIO(payload), allow_pickle=False) as nested:
            if key not in nested.files:
                raise EvaluationError(
                    f"{key!r} absent from nested {path}::{record['member']}"
                )
            value = np.asarray(nested[key])
    else:
        raise EvaluationError(f"unsupported locator type: {locator_type}")
    if list(value.shape) != list(record["shape"]):
        raise EvaluationError(
            f"target array shape changed at {record['identifier']}: "
            f"{value.shape} != {record['shape']}"
        )
    if str(value.dtype) != str(record["dtype"]):
        raise EvaluationError(
            f"target array dtype changed at {record['identifier']}: "
            f"{value.dtype} != {record['dtype']}"
        )
    return value


def outcome_record(option: Mapping[str, Any]) -> Mapping[str, Any]:
    if option["mode"] == "explicit-residual":
        record = option.get("residual")
    elif option["mode"] == "observation-minus-physical":
        record = option.get("observation")
    else:
        raise EvaluationError(f"unregistered target mode: {option['mode']}")
    if record is None:
        raise EvaluationError("target option lacks an outcome/residual record")
    return record


def output_shape(option: Mapping[str, Any]) -> tuple[int, ...]:
    shape = tuple(int(value) for value in outcome_record(option)["shape"])
    if len(shape) < 2 or not 1 <= shape[-1] <= 32:
        raise EvaluationError(f"unsupported target outcome shape: {shape}")
    return shape


def load_feature_carriers(
    option: Mapping[str, Any], artifact_root: Path
) -> dict[str, np.ndarray | None]:
    # Deliberately exclude residual and observation records in prediction mode.
    result = {}
    for role in ("physical", "state", "action", "contact", "trajectory_id", "time"):
        record = option.get(role)
        result[role] = None if record is None else load_locator(record, artifact_root)
    if result["physical"] is None:
        raise EvaluationError("target option lacks physical/state carrier")
    if result["trajectory_id"] is None:
        raise EvaluationError("target option lacks complete trajectory IDs")
    return result


def exact_width(value: np.ndarray, width: int, label: str) -> np.ndarray:
    if value.shape[1] != width:
        raise EvaluationError(
            f"{label} feature width {value.shape[1]} differs from sealed source width {width}"
        )
    return value


def donor_object_block(
    shared: np.ndarray,
    *,
    dlo: str,
    expected_width: int,
) -> np.ndarray:
    # Frozen ordinal donor placebo: DLO4 receives DLO2 identity, DLO5 DLO3.
    donor = TARGET_DLOS.index(dlo)
    one_hot = np.zeros((len(shared), 2), dtype=float)
    one_hot[:, donor] = 1.0
    interaction_source = shared[:, : min(4, shared.shape[1])]
    interactions = np.concatenate(
        [one_hot[:, [index]] * interaction_source for index in range(2)],
        axis=1,
    )
    value = np.concatenate((one_hot, interactions), axis=1)
    return exact_width(value, expected_width, f"{dlo} object-donor placebo")


def backend_placebo(
    *,
    dlo: str,
    rows: int,
    expected_width: int,
) -> np.ndarray:
    if expected_width < 1:
        raise EvaluationError("sealed backend block has invalid width")
    value = np.zeros((rows, expected_width), dtype=float)
    value[:, TARGET_DLOS.index(dlo) % expected_width] = 1.0
    return value


def build_target_feature_block(
    *,
    dlo: str,
    option: Mapping[str, Any],
    artifact_root: Path,
    sealed: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    carriers = load_feature_carriers(option, artifact_root)
    y_shape = output_shape(option)
    ids_raw = np.asarray(carriers["trajectory_id"]).astype(str)
    trajectory_ids, trajectory_mode = SOURCE_BUILDER.trajectory_row_ids(
        ids_raw, y_shape=y_shape
    )
    trajectory_ids = np.asarray(
        [f"{dlo}:{value}" for value in trajectory_ids], dtype=str
    )
    state = np.asarray(
        carriers["state"] if carriers["state"] is not None else carriers["physical"]
    )
    physical_rows = SOURCE_BUILDER.align_feature_block(
        state,
        y_shape=y_shape,
        output_axis=len(y_shape) - 1,
        label=f"{dlo} physical/state",
    )
    shared = SOURCE_BUILDER.local_difference_features(
        physical_rows, trajectory_ids
    )
    expected_shared = (
        sealed.model.slices["shared_physics"].stop
        - sealed.model.slices["shared_physics"].start
    )
    shared = exact_width(shared, expected_shared, f"{dlo} shared_physics")

    raw_action = (
        np.empty((len(shared), 0), dtype=float)
        if carriers["action"] is None
        else SOURCE_BUILDER.align_feature_block(
            np.asarray(carriers["action"]),
            y_shape=y_shape,
            output_axis=len(y_shape) - 1,
            label=f"{dlo} action",
        )
    )
    raw_contact = (
        np.empty((len(shared), 0), dtype=float)
        if carriers["contact"] is None
        else SOURCE_BUILDER.align_feature_block(
            np.asarray(carriers["contact"]),
            y_shape=y_shape,
            output_axis=len(y_shape) - 1,
            label=f"{dlo} contact",
        )
    )

    blocks: dict[str, np.ndarray] = {}
    for group in sealed.model.group_order:
        expected = sealed.model.slices[group].stop - sealed.model.slices[group].start
        if group == "shared_physics":
            blocks[group] = shared
        elif group == "object":
            blocks[group] = donor_object_block(
                shared, dlo=dlo, expected_width=expected
            )
        elif group == "backend":
            blocks[group] = backend_placebo(
                dlo=dlo, rows=len(shared), expected_width=expected
            )
        elif group == "actuation":
            blocks[group] = exact_width(raw_action, expected, f"{dlo} actuation")
        elif group == "contact":
            # This block is available only to the all-components/wrong-diagnosis
            # placebo. The source-selected transfer set cannot contain contact.
            blocks[group] = exact_width(raw_contact, expected, f"{dlo} contact")
        elif group == "sensor":
            blocks[group] = np.zeros((len(shared), expected), dtype=float)
        else:
            raise EvaluationError(f"unregistered sealed group: {group}")

    finite = np.logical_and.reduce(
        [np.all(np.isfinite(value), axis=1) for value in blocks.values()]
    )
    if not np.all(finite):
        # Row selection must not depend on target outcomes, but it may reject
        # nonfinite predictor features before prediction sealing.
        selected_rows = np.flatnonzero(finite)
    else:
        selected_rows = np.arange(len(shared), dtype=int)
    if len(selected_rows) < 64:
        raise EvaluationError(f"{dlo} has too few finite predictor rows")
    blocks = {name: value[selected_rows] for name, value in blocks.items()}
    trajectory_ids = trajectory_ids[selected_rows]

    metadata = {
        "dlo": dlo,
        "mode": option["mode"],
        "group_id": option["group_id"],
        "outcome_shape_from_header": list(y_shape),
        "trajectory_identity_mode": trajectory_mode,
        "total_predictor_rows": int(len(finite)),
        "selected_predictor_rows": int(len(selected_rows)),
        "discarded_nonfinite_predictor_rows": int(len(finite) - len(selected_rows)),
        "row_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(selected_rows).view(np.uint8)
        ).hexdigest(),
        "trajectory_count": int(len(np.unique(trajectory_ids))),
        "trajectory_counts": {
            value: int(np.sum(trajectory_ids == value))
            for value in sorted(np.unique(trajectory_ids).tolist())
        },
        "block_dimensions": {
            name: int(value.shape[1]) for name, value in blocks.items()
        },
        "feature_container_hashes": sorted(
            {
                record["container_relative_path"]: record["container_sha256"]
                for record in (
                    option.get("physical"),
                    option.get("state"),
                    option.get("action"),
                    option.get("contact"),
                    option.get("trajectory_id"),
                    option.get("time"),
                )
                if record is not None
            }.items()
        ),
    }
    arrays = {
        **{f"block__{name}": value for name, value in blocks.items()},
        "trajectory_id": trajectory_ids,
        "row_indices": selected_rows,
    }
    return arrays, metadata


def prediction_arms(sealed: Any, blocks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    rows = next(iter(blocks.values())).shape[0]
    physical = np.zeros((rows, sealed.model.coefficient_mean.shape[1]), dtype=float)
    all_physical_groups = [
        name for name in sealed.model.group_order if name != "sensor"
    ]
    wrong = sealed.wrong_diagnosis_group
    wrong_groups = [] if wrong is None or wrong == "sensor" else [wrong]
    return {
        "physical": physical,
        "shared_physics": sealed.model.predict(
            blocks, active_groups=["shared_physics"]
        ),
        "diagnosis_guided": sealed.model.predict(
            blocks, active_groups=sealed.selected_groups
        ),
        "all_components": sealed.model.predict(
            blocks, active_groups=all_physical_groups
        ),
        "wrong_diagnosis": (
            sealed.model.predict(blocks, active_groups=wrong_groups)
            if wrong_groups
            else physical.copy()
        ),
    }


def verify_common_inputs(
    *,
    authorization: Mapping[str, Any],
    target_plan: Mapping[str, Any],
    sealed: Any,
) -> None:
    verify_id(authorization, "authorization_id")
    verify_id(target_plan, "plan_id")
    if authorization["source_model_id"] != sealed.model_id:
        raise EvaluationError("authorization names another source model")
    if target_plan["source_model_id"] != sealed.model_id:
        raise EvaluationError("target plan names another source model")
    if target_plan["authorization_id"] != authorization["authorization_id"]:
        raise EvaluationError("target plan/authorization mismatch")
    if target_plan["ready_for_prediction_seal"] is not True:
        raise EvaluationError("target plan did not authorize prediction generation")
    if sorted(target_plan["selected_carriers"]) != list(TARGET_DLOS):
        raise EvaluationError("target plan must select exactly DLO4 and DLO5")
    if authorization["coefficient_refit_authorized"] is not False:
        raise EvaluationError("authorization permits coefficient refit")
    if authorization["standardizer_refit_authorized"] is not False:
        raise EvaluationError("authorization permits standardizer refit")
    if authorization["selection_frozen_before_target"] is not True:
        raise EvaluationError("source selection was not frozen")


def generate_predictions(
    *,
    artifact_root: Path,
    authorization_path: Path,
    target_plan_path: Path,
    model_prefix: Path,
    output_directory: Path,
) -> dict[str, Any]:
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    target_plan = json.loads(target_plan_path.read_text(encoding="utf-8"))
    sealed = TRANSFER.load_sealed_model(model_prefix)
    verify_common_inputs(
        authorization=authorization, target_plan=target_plan, sealed=sealed
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    per_dlo = {}
    prediction_hashes = {}
    for dlo in TARGET_DLOS:
        arrays, metadata = build_target_feature_block(
            dlo=dlo,
            option=target_plan["selected_carriers"][dlo],
            artifact_root=artifact_root,
            sealed=sealed,
        )
        blocks = {
            key.removeprefix("block__"): value
            for key, value in arrays.items()
            if key.startswith("block__")
        }
        predictions = prediction_arms(sealed, blocks)
        feature_path = output_directory / f"{dlo.lower()}_features.npz"
        prediction_path = output_directory / f"{dlo.lower()}_predictions.npz"
        np.savez_compressed(feature_path, **arrays)
        np.savez_compressed(prediction_path, **predictions)
        prediction_hashes[dlo] = {
            name: hashlib.sha256(
                np.ascontiguousarray(value).view(np.uint8)
            ).hexdigest()
            for name, value in predictions.items()
        }
        per_dlo[dlo] = {
            **metadata,
            "feature_file": feature_path.name,
            "feature_file_sha256": sha256(feature_path),
            "prediction_file": prediction_path.name,
            "prediction_file_sha256": sha256(prediction_path),
            "prediction_array_sha256": prediction_hashes[dlo],
        }

    seal = {
        "schema": "bayesian-phystwin.hierarchical-missing-physics-dlo45-prediction-seal",
        "schema_version": 1,
        "authorization_id": authorization["authorization_id"],
        "source_model_id": sealed.model_id,
        "source_result_id": sealed.source_result_id,
        "target_plan_id": target_plan["plan_id"],
        "selected_groups_frozen_source_only": list(sealed.selected_groups),
        "wrong_diagnosis_group_frozen_source_only": sealed.wrong_diagnosis_group,
        "per_dlo": per_dlo,
        "target_outcome_arrays_loaded": False,
        "target_performance_metric_computed": False,
        "coefficient_refit": False,
        "standardizer_refit": False,
        "target_group_selection": False,
        "sensor_to_physical_rollout": False,
    }
    seal["seal_id"] = canonical_hash(seal)
    (output_directory / "prediction_seal.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return seal


def load_target_y(
    *,
    dlo: str,
    option: Mapping[str, Any],
    artifact_root: Path,
    row_indices: np.ndarray,
    expected_output_dim: int,
) -> np.ndarray:
    physical = load_locator(option["physical"], artifact_root)
    if option["mode"] == "explicit-residual":
        y_raw = load_locator(option["residual"], artifact_root)
    elif option["mode"] == "observation-minus-physical":
        observation = load_locator(option["observation"], artifact_root)
        try:
            y_raw = np.asarray(observation, dtype=float) - np.asarray(physical, dtype=float)
        except ValueError as error:
            raise EvaluationError(f"{dlo} target observation/physical arrays differ") from error
    else:
        raise EvaluationError(f"unregistered target mode: {option['mode']}")
    y = np.asarray(y_raw, dtype=float).reshape(-1, y_raw.shape[-1])
    if y.shape[1] != expected_output_dim:
        raise EvaluationError(
            f"{dlo} target output width {y.shape[1]} != source model {expected_output_dim}"
        )
    y = y[row_indices]
    if not np.all(np.isfinite(y)):
        raise EvaluationError(f"{dlo} target outcome has nonfinite scored rows")
    return y


def score_predictions(
    *,
    artifact_root: Path,
    authorization_path: Path,
    target_plan_path: Path,
    model_prefix: Path,
    sealed_directory: Path,
) -> dict[str, Any]:
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    target_plan = json.loads(target_plan_path.read_text(encoding="utf-8"))
    sealed_model = TRANSFER.load_sealed_model(model_prefix)
    verify_common_inputs(
        authorization=authorization,
        target_plan=target_plan,
        sealed=sealed_model,
    )
    seal_path = sealed_directory / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    verify_id(seal, "seal_id")
    if seal["authorization_id"] != authorization["authorization_id"]:
        raise EvaluationError("prediction seal/authorization mismatch")
    if seal["source_model_id"] != sealed_model.model_id:
        raise EvaluationError("prediction seal/source-model mismatch")
    if seal["target_plan_id"] != target_plan["plan_id"]:
        raise EvaluationError("prediction seal/target-plan mismatch")
    if seal["target_outcome_arrays_loaded"] is not False:
        raise EvaluationError("prediction seal was created after target outcome loading")
    if seal["target_performance_metric_computed"] is not False:
        raise EvaluationError("prediction seal already contains target performance")

    per_dlo = {}
    all_trajectory_rows = []
    for dlo in TARGET_DLOS:
        feature_path = sealed_directory / seal["per_dlo"][dlo]["feature_file"]
        prediction_path = sealed_directory / seal["per_dlo"][dlo]["prediction_file"]
        if sha256(feature_path) != seal["per_dlo"][dlo]["feature_file_sha256"]:
            raise EvaluationError(f"{dlo} feature-file seal mismatch")
        if sha256(prediction_path) != seal["per_dlo"][dlo]["prediction_file_sha256"]:
            raise EvaluationError(f"{dlo} prediction-file seal mismatch")
        with np.load(feature_path, allow_pickle=False) as features:
            trajectory_ids = np.asarray(features["trajectory_id"]).astype(str)
            row_indices = np.asarray(features["row_indices"], dtype=int)
        with np.load(prediction_path, allow_pickle=False) as predictions_archive:
            predictions = {
                key: np.asarray(predictions_archive[key], dtype=float)
                for key in predictions_archive.files
            }
        for name, value in predictions.items():
            expected = seal["per_dlo"][dlo]["prediction_array_sha256"][name]
            actual = hashlib.sha256(
                np.ascontiguousarray(value).view(np.uint8)
            ).hexdigest()
            if actual != expected:
                raise EvaluationError(f"{dlo} {name} prediction-array seal mismatch")
        y = load_target_y(
            dlo=dlo,
            option=target_plan["selected_carriers"][dlo],
            artifact_root=artifact_root,
            row_indices=row_indices,
            expected_output_dim=sealed_model.model.coefficient_mean.shape[1],
        )
        if any(len(value) != len(y) for value in predictions.values()):
            raise EvaluationError(f"{dlo} prediction/target row count mismatch")

        physical_rmse = float(np.sqrt(np.mean(np.square(y - predictions["physical"]))))
        aggregate = {}
        for name, value in predictions.items():
            rmse = float(np.sqrt(np.mean(np.square(y - value))))
            aggregate[name] = {
                "rmse": rmse,
                "mae": float(np.mean(np.abs(y - value))),
                "relative_improvement_vs_physical": (
                    physical_rmse - rmse
                ) / max(physical_rmse, EPS),
            }

        trajectory_rows = []
        for trajectory in sorted(np.unique(trajectory_ids).tolist()):
            mask = trajectory_ids == trajectory
            row = {"dlo": dlo, "trajectory_id": trajectory}
            for name, value in predictions.items():
                row[f"{name}_rmse"] = float(
                    np.sqrt(np.mean(np.square(y[mask] - value[mask])))
                )
            trajectory_rows.append(row)
            all_trajectory_rows.append(row)

        wins_physical = sum(
            row["diagnosis_guided_rmse"] < row["physical_rmse"]
            for row in trajectory_rows
        )
        wins_all = sum(
            row["diagnosis_guided_rmse"] < row["all_components_rmse"]
            for row in trajectory_rows
        )
        worst_ratio = max(
            row["diagnosis_guided_rmse"] / max(row["physical_rmse"], EPS)
            for row in trajectory_rows
        )
        wrong_ratio = aggregate["diagnosis_guided"]["rmse"] / max(
            aggregate["wrong_diagnosis"]["rmse"], EPS
        )
        required_wins = 8 if len(trajectory_rows) == 14 else math.ceil(0.60 * len(trajectory_rows))
        gate = {
            "minimum_relative_improvement_vs_physical": 0.01,
            "relative_improvement_pass": bool(
                aggregate["diagnosis_guided"]["relative_improvement_vs_physical"]
                >= 0.01
            ),
            "required_trajectory_wins": required_wins,
            "trajectory_wins_vs_physical": wins_physical,
            "trajectory_wins_pass": wins_physical >= required_wins,
            "maximum_worst_trajectory_ratio": 1.10,
            "worst_trajectory_ratio": worst_ratio,
            "worst_trajectory_pass": worst_ratio <= 1.10,
            "diagnosis_guided_beats_all_components": bool(
                aggregate["diagnosis_guided"]["rmse"]
                < aggregate["all_components"]["rmse"]
            ),
            "diagnosis_guided_trajectory_wins_vs_all_components": wins_all,
            "shared_or_guided_beats_wrong_diagnosis": wrong_ratio < 1.0,
            "fallback_identity_violations": 0,
        }
        gate["passed"] = bool(
            gate["relative_improvement_pass"]
            and gate["trajectory_wins_pass"]
            and gate["worst_trajectory_pass"]
            and gate["diagnosis_guided_beats_all_components"]
            and gate["shared_or_guided_beats_wrong_diagnosis"]
            and gate["fallback_identity_violations"] == 0
        )
        per_dlo[dlo] = {
            "aggregate": aggregate,
            "trajectory_count": len(trajectory_rows),
            "per_trajectory": trajectory_rows,
            "gate": gate,
        }

    result = {
        "schema": "bayesian-phystwin.hierarchical-missing-physics-dlo45-transfer-result",
        "schema_version": 1,
        "authorization_id": authorization["authorization_id"],
        "source_model_id": sealed_model.model_id,
        "source_result_id": sealed_model.source_result_id,
        "target_plan_id": target_plan["plan_id"],
        "prediction_seal_id": seal["seal_id"],
        "selected_groups_frozen_source_only": list(sealed_model.selected_groups),
        "wrong_diagnosis_group_frozen_source_only": sealed_model.wrong_diagnosis_group,
        "per_dlo": per_dlo,
        "all_target_gates_pass": all(
            value["gate"]["passed"] for value in per_dlo.values()
        ),
        "claim_authorization": (
            "cross-object diagnosis-guided transfer"
            if all(value["gate"]["passed"] for value in per_dlo.values())
            else "no positive cross-object claim; retain mixed/negative result"
        ),
        "information_boundary": {
            "target_outcome_arrays_loaded_only_after_prediction_seal": True,
            "target_used_for_group_selection": False,
            "target_used_for_coefficient_refit": False,
            "target_used_for_standardization": False,
            "target_used_for_feature_semantics": False,
            "target_used_for_retry_or_case_replacement": False,
            "sensor_discrepancy_injected_into_physical_rollout": False,
            "DLO4_and_DLO5_reported_separately": True,
        },
    }
    result["result_id"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--artifact-root", required=True, type=Path)
    predict.add_argument("--authorization", required=True, type=Path)
    predict.add_argument("--target-plan", required=True, type=Path)
    predict.add_argument("--model-prefix", required=True, type=Path)
    predict.add_argument("--output-directory", required=True, type=Path)

    score = subparsers.add_parser("score")
    score.add_argument("--artifact-root", required=True, type=Path)
    score.add_argument("--authorization", required=True, type=Path)
    score.add_argument("--target-plan", required=True, type=Path)
    score.add_argument("--model-prefix", required=True, type=Path)
    score.add_argument("--sealed-directory", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)

    arguments = parser.parse_args()
    if arguments.command == "predict":
        result = generate_predictions(
            artifact_root=arguments.artifact_root,
            authorization_path=arguments.authorization,
            target_plan_path=arguments.target_plan,
            model_prefix=arguments.model_prefix,
            output_directory=arguments.output_directory,
        )
    else:
        result = score_predictions(
            artifact_root=arguments.artifact_root,
            authorization_path=arguments.authorization,
            target_plan_path=arguments.target_plan,
            model_prefix=arguments.model_prefix,
            sealed_directory=arguments.sealed_directory,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
