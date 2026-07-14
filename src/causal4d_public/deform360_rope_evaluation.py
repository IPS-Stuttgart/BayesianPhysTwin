"""Leakage-safe held-out prediction seals and rope trajectory evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_contact import validate_contact_artifact


DEFORM360_ROPE_PREDICTION_SCHEMA_VERSION = 1
PRIMARY_METHODS = ("visual_only", "tactile_conditioned_z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def rope_prediction_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _trajectory(value: np.ndarray, name: str) -> np.ndarray:
    trajectory = np.asarray(value, dtype=np.float64)
    _require(
        trajectory.ndim == 3
        and trajectory.shape[2] == 3
        and trajectory.shape[0] >= 1
        and trajectory.shape[1] >= 2,
        f"{name} trajectory must have shape (T,N,3)",
    )
    _require(np.all(np.isfinite(trajectory)), f"{name} trajectory is non-finite")
    return trajectory


def seal_held_out_rope_predictions(
    archive_path: str | Path,
    predictions: Mapping[str, np.ndarray],
    *,
    protocol_id: str,
    contact_prediction_seal: Mapping[str, Any],
    shared_dynamics_fit_sha256: str,
    target_prefix_geometry_sha256: str,
    future_start_frame: int,
    rollout_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist both deployable target rollouts before any target future opens."""

    validate_contact_artifact(
        contact_prediction_seal,
        expected_kind="Deform360TargetContactPredictionSeal",
    )
    _require(
        contact_prediction_seal.get("protocol_id") == protocol_id,
        "contact prediction seal belongs to a different protocol",
    )
    _require(
        set(predictions) == set(PRIMARY_METHODS),
        "prediction seal requires exactly visual_only and tactile_conditioned_z",
    )
    _require(_valid_sha256(shared_dynamics_fit_sha256), "invalid shared-fit checksum")
    _require(_valid_sha256(target_prefix_geometry_sha256), "invalid prefix checksum")
    _require(future_start_frame >= 1, "future start frame must be positive")
    arrays = {name: _trajectory(predictions[name], name) for name in PRIMARY_METHODS}
    shapes = {array.shape for array in arrays.values()}
    _require(len(shapes) == 1, "held-out prediction shapes disagree")
    output = Path(archive_path).resolve()
    _require(output.suffix == ".npz", "prediction archive must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_ROPE_PREDICTION_SCHEMA_VERSION,
        "artifact_kind": "Deform360HeldOutRopePredictionSeal",
        "protocol_id": protocol_id,
        "target_episode_id": contact_prediction_seal["target_episode_id"],
        "contact_prediction_seal_sha256": contact_prediction_seal["result_sha256"],
        "shared_dynamics_fit_sha256": shared_dynamics_fit_sha256,
        "target_prefix_geometry_sha256": target_prefix_geometry_sha256,
        "future_start_frame": future_start_frame,
        "prediction_shape": list(next(iter(shapes))),
        "methods": {
            name: {
                "trajectory_sha256": _sha256_array(array),
                "contact_evidence": "robot opening only"
                if name == "visual_only"
                else "robot opening plus sealed six-frame tactile prefix",
            }
            for name, array in arrays.items()
        },
        "archive": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
        },
        "information_boundary": {
            "target_visual_prefix_read": True,
            "target_tactile_prefix_read": True,
            "target_future_geometry_read": False,
            "target_tactile_oracle_read": False,
            "target_prediction_metrics_computed": False,
        },
        "claim_boundary": (
            "Both deployable rollouts were immutable before opening target-future "
            "geometry or the full target tactile contact reference."
        ),
    }
    if rollout_configuration is not None:
        payload["rollout_configuration"] = dict(rollout_configuration)
    payload["result_sha256"] = rope_prediction_artifact_sha256(payload)
    return payload


def validate_held_out_rope_prediction_seal(
    payload: Mapping[str, Any], *, verify_archive: bool = True
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_ROPE_PREDICTION_SCHEMA_VERSION,
        "unsupported rope-prediction artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360HeldOutRopePredictionSeal",
        "unexpected rope-prediction artifact kind",
    )
    _require(
        payload.get("result_sha256") == rope_prediction_artifact_sha256(payload),
        "rope-prediction artifact checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("target_future_geometry_read") is False,
        "prediction seal used target-future geometry",
    )
    _require(
        boundary.get("target_tactile_oracle_read") is False,
        "prediction seal used the target tactile oracle",
    )
    _require(
        set(payload.get("methods", {})) == set(PRIMARY_METHODS),
        "prediction seal method set differs from the protocol",
    )
    if verify_archive:
        archive = Path(payload["archive"]["path"])
        _require(archive.is_file(), "held-out prediction archive is missing")
        _require(
            _sha256_file(archive) == payload["archive"]["sha256"],
            "held-out prediction archive checksum mismatch",
        )
        with np.load(archive, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(PRIMARY_METHODS), "archive method mismatch"
            )
            for name in PRIMARY_METHODS:
                array = _trajectory(stored[name], name)
                _require(
                    list(array.shape) == payload["prediction_shape"],
                    f"stored {name} trajectory shape mismatch",
                )
                _require(
                    _sha256_array(array)
                    == payload["methods"][name]["trajectory_sha256"],
                    f"stored {name} trajectory checksum mismatch",
                )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "future_geometry_unlock_authorized": True,
        "target_tactile_oracle_unlock_authorized": True,
    }


def load_held_out_rope_predictions(
    seal: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    validate_held_out_rope_prediction_seal(seal, verify_archive=True)
    with np.load(seal["archive"]["path"], allow_pickle=False) as stored:
        return {
            name: np.asarray(stored[name], dtype=np.float64) for name in PRIMARY_METHODS
        }


def build_oracle_tactile_rope_prediction(
    trajectory_m: np.ndarray,
    *,
    held_out_prediction_seal: Mapping[str, Any],
    target_contact_oracle: Mapping[str, Any],
    contact_schedule_sha256: str,
) -> dict[str, Any]:
    """Build the post-seal oracle-contact upper-bound rollout artifact."""

    validate_held_out_rope_prediction_seal(
        held_out_prediction_seal, verify_archive=True
    )
    validate_contact_artifact(
        target_contact_oracle,
        expected_kind="Deform360TargetContactOracleEvaluation",
    )
    _require(
        target_contact_oracle.get("held_out_prediction_seal_sha256")
        == held_out_prediction_seal["result_sha256"],
        "target contact oracle was opened under a different prediction seal",
    )
    _require(_valid_sha256(contact_schedule_sha256), "invalid oracle schedule checksum")
    trajectory = _trajectory(trajectory_m, "oracle_tactile")
    _require(
        list(trajectory.shape) == held_out_prediction_seal["prediction_shape"],
        "oracle trajectory shape differs from sealed deployable predictions",
    )
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_ROPE_PREDICTION_SCHEMA_VERSION,
        "artifact_kind": "Deform360OracleTactileRopePrediction",
        "protocol_id": held_out_prediction_seal["protocol_id"],
        "target_episode_id": held_out_prediction_seal["target_episode_id"],
        "held_out_prediction_seal_sha256": held_out_prediction_seal["result_sha256"],
        "target_contact_oracle_sha256": target_contact_oracle["result_sha256"],
        "contact_schedule_sha256": contact_schedule_sha256,
        "trajectory_m": trajectory.tolist(),
        "trajectory_sha256": _sha256_array(trajectory),
        "information_boundary": {
            "deployable_predictions_previously_sealed": True,
            "target_tactile_oracle_read": True,
            "target_future_geometry_used_for_rollout": False,
        },
        "claim_boundary": "Offline full-tactile contact upper bound, not a deployable method.",
    }
    payload["result_sha256"] = rope_prediction_artifact_sha256(payload)
    return payload


def _frame_metrics(
    reference: np.ndarray, prediction: np.ndarray
) -> dict[str, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("SciPy is required for rope evaluation") from error
    ordered = np.mean(np.linalg.norm(reference - prediction, axis=2), axis=1)
    chamfer = []
    for truth, guess in zip(reference, prediction, strict=True):
        truth_to_guess = cKDTree(guess).query(truth, k=1)[0]
        guess_to_truth = cKDTree(truth).query(guess, k=1)[0]
        chamfer.append(0.5 * (np.mean(truth_to_guess) + np.mean(guess_to_truth)))
    return {"track_error_m": ordered, "chamfer_distance_m": np.asarray(chamfer)}


def _metric_summary(values: np.ndarray) -> dict[str, Any]:
    thirds = np.array_split(values, 3)
    return {
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "final_m": float(values[-1]),
        "by_horizon_third_m": [float(np.mean(part)) for part in thirds],
        "per_frame_m": values.tolist(),
    }


def evaluate_held_out_rope_predictions(
    target_future_positions_m: np.ndarray,
    *,
    held_out_prediction_seal: Mapping[str, Any],
    target_future_geometry_sha256: str,
    oracle_prediction: Mapping[str, Any] | None = None,
    additional_predictions: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Evaluate only after the deployable prediction seal is immutable."""

    predictions = load_held_out_rope_predictions(held_out_prediction_seal)
    reference = _trajectory(target_future_positions_m, "target future")
    _require(
        _valid_sha256(target_future_geometry_sha256), "invalid target geometry hash"
    )
    _require(
        list(reference.shape) == held_out_prediction_seal["prediction_shape"],
        "target future shape differs from the sealed prediction shape",
    )
    methods = dict(predictions)
    additional_hashes = {}
    if additional_predictions is not None:
        overlap = set(methods).intersection(additional_predictions)
        _require(not overlap, f"additional prediction names overlap: {sorted(overlap)}")
        for name, values in additional_predictions.items():
            trajectory = _trajectory(values, name)
            _require(
                trajectory.shape == reference.shape,
                f"additional prediction shape differs for {name}",
            )
            methods[name] = trajectory
            additional_hashes[name] = _sha256_array(trajectory)
    if oracle_prediction is not None:
        _require(
            oracle_prediction.get("artifact_kind")
            == "Deform360OracleTactileRopePrediction",
            "unexpected oracle prediction artifact kind",
        )
        _require(
            oracle_prediction.get("result_sha256")
            == rope_prediction_artifact_sha256(oracle_prediction),
            "oracle prediction artifact checksum mismatch",
        )
        _require(
            oracle_prediction.get("held_out_prediction_seal_sha256")
            == held_out_prediction_seal["result_sha256"],
            "oracle prediction and deployable prediction seal differ",
        )
        methods["oracle_tactile"] = _trajectory(
            np.asarray(oracle_prediction["trajectory_m"]), "oracle_tactile"
        )
    results = {}
    for name, prediction in methods.items():
        metrics = _frame_metrics(reference, prediction)
        results[name] = {
            metric: _metric_summary(values) for metric, values in metrics.items()
        }
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_ROPE_PREDICTION_SCHEMA_VERSION,
        "artifact_kind": "Deform360HeldOutRopeEvaluation",
        "protocol_id": held_out_prediction_seal["protocol_id"],
        "target_episode_id": held_out_prediction_seal["target_episode_id"],
        "held_out_prediction_seal_sha256": held_out_prediction_seal["result_sha256"],
        "target_future_geometry_sha256": target_future_geometry_sha256,
        "target_future_trajectory_sha256": _sha256_array(reference),
        "methods": results,
        "additional_prediction_sha256": additional_hashes,
        "paired_primary_difference_m": {
            metric: float(
                results["tactile_conditioned_z"][metric]["mean_m"]
                - results["visual_only"][metric]["mean_m"]
            )
            for metric in ("track_error_m", "chamfer_distance_m")
        },
        "information_boundary": {
            "deployable_predictions_previously_sealed": True,
            "target_future_geometry_read_for_evaluation": True,
            "target_future_geometry_used_for_fitting": False,
        },
    }
    payload["result_sha256"] = rope_prediction_artifact_sha256(payload)
    return payload


__all__ = [
    "DEFORM360_ROPE_PREDICTION_SCHEMA_VERSION",
    "PRIMARY_METHODS",
    "build_oracle_tactile_rope_prediction",
    "evaluate_held_out_rope_predictions",
    "load_held_out_rope_predictions",
    "rope_prediction_artifact_sha256",
    "seal_held_out_rope_predictions",
    "validate_held_out_rope_prediction_seal",
]
