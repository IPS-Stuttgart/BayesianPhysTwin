"""Sealed contact-timing experiment for the public Deform360 rope cohort."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360 import (
    DEFORM360_OBJECT_ID,
    Deform360ProtocolConfig,
    protocol_config_sha256,
)


DEFORM360_CONTACT_SCHEMA_VERSION = 1
CONTACT_PATIENCE_FRAMES = 5
TACTILE_ROWS_USED = 12


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": str(array.dtype), "shape": list(array.shape)}
    )
    return _sha256_bytes(descriptor + b"\0" + array.tobytes())


def contact_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return _sha256_bytes(_canonical_bytes(canonical))


def _finish_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    payload["result_sha256"] = contact_artifact_sha256(payload)
    return payload


def validate_contact_artifact(
    payload: Mapping[str, Any], *, expected_kind: str
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_CONTACT_SCHEMA_VERSION,
        "unsupported Deform360 contact artifact schema",
    )
    _require(payload.get("artifact_kind") == expected_kind, "unexpected artifact kind")
    _require(
        payload.get("result_sha256") == contact_artifact_sha256(payload),
        "Deform360 contact artifact checksum mismatch",
    )
    return {
        "passed": True,
        "artifact_kind": expected_kind,
        "result_sha256": payload["result_sha256"],
    }


def load_contact_artifact(path: str | Path, *, expected_kind: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "contact artifact must be a JSON object")
    validate_contact_artifact(payload, expected_kind=expected_kind)
    return payload


def write_contact_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _episode_id(index: int) -> str:
    return f"{DEFORM360_OBJECT_ID}/episode_{index:04d}"


def _episode_dir(processed_root: Path, index: int) -> Path:
    path = processed_root / f"episode_{index:04d}"
    _require(path.is_dir(), f"processed episode is missing: {path}")
    return path


def _load_metadata(raw_object_dir: Path) -> tuple[dict[str, Any], str]:
    path = raw_object_dir / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("object") == DEFORM360_OBJECT_ID, "metadata object mismatch")
    sequences = payload.get("sequences")
    _require(isinstance(sequences, dict), "metadata sequences are missing")
    return sequences, _sha256_file(path)


def _load_robot(episode_dir: Path) -> dict[str, Any]:
    path = episode_dir / "robot" / "robot.npz"
    _require(path.is_file(), f"robot trajectory is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        openings = np.asarray(payload["openings"], dtype=np.float64)
        bimanual = bool(np.asarray(payload["bimanual"]).item())
    if openings.ndim == 1:
        openings = openings[:, None]
    _require(openings.ndim == 2, "robot openings must have shape (T,G)")
    _require(np.all(np.isfinite(openings)), "robot openings are non-finite")
    _require(
        openings.shape[1] == (2 if bimanual else 1),
        "robot opening/gripper count mismatch",
    )
    return {
        "openings": openings,
        "bimanual": bimanual,
        "sha256": _sha256_file(path),
    }


def _gripper_group(sensor_name: str) -> str:
    for suffix in ("_left", "_right"):
        if sensor_name.endswith(suffix):
            return sensor_name[: -len(suffix)]
    return sensor_name


def _sensor_paths(episode_dir: Path) -> list[Path]:
    return sorted(
        path / "synced_tactile.npy"
        for path in episode_dir.iterdir()
        if path.is_dir() and (path / "synced_tactile.npy").is_file()
    )


def _load_tactile_groups(
    episode_dir: Path,
    *,
    frame_slice: slice | None,
    hash_full_inputs: bool,
    threshold: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    groups: dict[str, np.ndarray] = {}
    inputs = []
    paths = _sensor_paths(episode_dir)
    _require(paths, f"no synchronized tactile streams in {episode_dir}")
    for path in paths:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        _require(values.ndim == 3, f"invalid tactile shape in {path}")
        selected = np.asarray(
            values if frame_slice is None else values[frame_slice], dtype=np.float32
        )
        active_taxels = np.count_nonzero(
            selected[:, :TACTILE_ROWS_USED, :] > threshold,
            axis=(1, 2),
        ).astype(np.int64)
        group = _gripper_group(path.parent.name)
        groups[group] = groups.get(group, np.zeros_like(active_taxels)) + active_taxels
        record: dict[str, Any] = {
            "sensor": path.parent.name,
            "selected_shape": list(selected.shape),
            "selected_sha256": _sha256_array(selected),
        }
        if hash_full_inputs:
            record["file_sha256"] = _sha256_file(path)
        inputs.append(record)
    lengths = {len(values) for values in groups.values()}
    _require(len(lengths) == 1, "tactile group lengths disagree")
    return groups, inputs


def _official_contact_window(
    active: np.ndarray,
) -> tuple[np.ndarray, int | None, int | None]:
    signal = np.asarray(active, dtype=bool)
    output = np.zeros_like(signal)
    start: int | None = None
    end: int | None = None
    missing = 0
    for frame, is_active in enumerate(signal):
        if start is None:
            if is_active:
                start = frame
            continue
        if is_active:
            missing = 0
        else:
            missing += 1
            if missing > CONTACT_PATIENCE_FRAMES:
                end = frame - missing
                break
    if start is None:
        return output, None, None
    if end is None:
        end = len(signal) - 1
    output[start : end + 1] = True
    return output, start, end


def _causal_confirmed(signal: np.ndarray, confirmation_frames: int) -> np.ndarray:
    raw = np.asarray(signal, dtype=bool)
    output = np.zeros_like(raw)
    state = False
    run = 0
    for index, value in enumerate(raw):
        if bool(value) == state:
            run = 0
        else:
            run += 1
            if run >= confirmation_frames:
                state = not state
                run = 0
        output[index] = state
    return output


def _binary_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(reference, dtype=bool)
    guess = np.asarray(prediction, dtype=bool)
    _require(truth.shape == guess.shape, "contact metric shape mismatch")
    tp = int(np.count_nonzero(truth & guess))
    tn = int(np.count_nonzero(~truth & ~guess))
    fp = int(np.count_nonzero(~truth & guess))
    fn = int(np.count_nonzero(truth & ~guess))
    positive_recall = tp / (tp + fn) if tp + fn else 1.0
    negative_recall = tn / (tn + fp) if tn + fp else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    truth_indices = np.flatnonzero(truth)
    guess_indices = np.flatnonzero(guess)
    return {
        "frame_count": int(len(truth)),
        "accuracy": (tp + tn) / len(truth) if len(truth) else 1.0,
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "reference_onset_frame": int(truth_indices[0]) if len(truth_indices) else None,
        "predicted_onset_frame": int(guess_indices[0]) if len(guess_indices) else None,
        "onset_error_frames": int(guess_indices[0] - truth_indices[0])
        if len(truth_indices) and len(guess_indices)
        else None,
        "reference_offset_frame": int(truth_indices[-1])
        if len(truth_indices)
        else None,
        "predicted_offset_frame": int(guess_indices[-1])
        if len(guess_indices)
        else None,
        "offset_error_frames": int(guess_indices[-1] - truth_indices[-1])
        if len(truth_indices) and len(guess_indices)
        else None,
    }


def _mono_event_group(groups: Mapping[str, np.ndarray]) -> tuple[str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    ranked = []
    for name, counts in sorted(groups.items()):
        active = counts > 1
        guard = max(10, int(round(0.1 * len(active))))
        initial_fraction = float(np.mean(active[:guard]))
        active_fraction = float(np.mean(active))
        salience = (1.0 - initial_fraction) * active_fraction
        diagnostics[name] = {
            "initial_active_fraction": initial_fraction,
            "active_fraction": active_fraction,
            "event_salience": salience,
        }
        ranked.append((-salience, name))
    _require(ranked, "no tactile groups available")
    selected = min(ranked)[1]
    return selected, diagnostics


def _episode_training_record(
    processed_root: Path,
    index: int,
    metadata: Mapping[str, Any],
    config: Deform360ProtocolConfig,
) -> dict[str, Any]:
    episode_dir = _episode_dir(processed_root, index)
    robot = _load_robot(episode_dir)
    groups, tactile_inputs = _load_tactile_groups(
        episode_dir,
        frame_slice=None,
        hash_full_inputs=True,
        threshold=config.tactile_contact_threshold,
    )
    _require(
        next(iter({len(value) for value in groups.values()}))
        == robot["openings"].shape[0],
        "robot/tactile frame count mismatch",
    )
    labels = {}
    group_windows = {}
    for name, counts in sorted(groups.items()):
        label, start, end = _official_contact_window(counts > 1)
        labels[name] = label
        group_windows[name] = {
            "contact_start_frame": start,
            "contact_end_frame": end,
            "active_taxel_peak": int(np.max(counts)),
        }
    sequence = metadata[str(index)]
    bimanual = sequence.get("bimanual") == "yes"
    nonprehensile = sequence.get("nonprehensile") == "yes"
    _require(bimanual == robot["bimanual"], "metadata/robot bimanual mismatch")
    mono_group = None
    sensor_diagnostics = None
    if not bimanual:
        mono_group, sensor_diagnostics = _mono_event_group(groups)
    return {
        "episode_index": index,
        "episode_id": _episode_id(index),
        "action": str(sequence["action"]),
        "bimanual": bimanual,
        "nonprehensile": nonprehensile,
        "openings": robot["openings"],
        "labels": labels,
        "group_windows": group_windows,
        "mono_event_group": mono_group,
        "sensor_diagnostics": sensor_diagnostics,
        "robot_sha256": robot["sha256"],
        "tactile_inputs": tactile_inputs,
    }


def _candidate_thresholds(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.concatenate(
        [
            np.asarray(record["openings"], dtype=np.float64).reshape(-1)
            for record in records
        ]
    )
    quantiles = np.linspace(0.01, 0.99, 199)
    return np.unique(np.quantile(values, quantiles))


def _mapping_candidates(records: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    group_names = sorted(
        {
            group
            for record in records
            if record["bimanual"] and not record["nonprehensile"]
            for group in record["labels"]
        }
    )
    _require(len(group_names) == 2, "expected two tactile gripper groups")
    return [
        dict(zip(group_names, permutation))
        for permutation in itertools.permutations((0, 1))
    ]


def _record_pairs(
    record: Mapping[str, Any], mapping: Mapping[str, int]
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    if record["bimanual"]:
        return [
            (
                np.asarray(record["openings"])[:, int(mapping[group])],
                np.asarray(record["labels"][group]),
                group,
            )
            for group in sorted(record["labels"])
        ]
    group = str(record["mono_event_group"])
    return [
        (
            np.asarray(record["openings"])[:, 0],
            np.asarray(record["labels"][group]),
            group,
        )
    ]


def _evaluate_opening_model(
    records: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, int],
    threshold_m: float,
    confirmation_frames: int,
) -> tuple[float, list[dict[str, Any]]]:
    episode_metrics = []
    scores = []
    for record in records:
        pair_metrics = []
        for openings, label, group in _record_pairs(record, mapping):
            prediction = _causal_confirmed(openings <= threshold_m, confirmation_frames)
            metrics = _binary_metrics(label, prediction)
            metrics["tactile_group"] = group
            pair_metrics.append(metrics)
            scores.append(float(metrics["balanced_accuracy"]))
        episode_metrics.append(
            {
                "episode_id": record["episode_id"],
                "action": record["action"],
                "pairs": pair_metrics,
                "mean_balanced_accuracy": float(
                    np.mean([item["balanced_accuracy"] for item in pair_metrics])
                ),
            }
        )
    return float(np.mean(scores)), episode_metrics


def fit_contact_model(
    raw_object_dir: str | Path,
    processed_root: str | Path,
    config: Deform360ProtocolConfig,
) -> dict[str, Any]:
    """Fit an opening-only contact model without touching target episode data."""

    raw_root = Path(raw_object_dir).resolve()
    processed = Path(processed_root).resolve()
    metadata, metadata_sha256 = _load_metadata(raw_root)
    source_records = [
        _episode_training_record(processed, index, metadata, config)
        for index in config.source_episode_ids
    ]
    calibration_records = [
        _episode_training_record(processed, index, metadata, config)
        for index in config.calibration_episode_ids
    ]
    fit_records = [record for record in source_records if not record["nonprehensile"]]
    _require(fit_records, "no prehensile source episodes are available")
    best: tuple[float, float, tuple[tuple[str, int], ...]] | None = None
    best_mapping: dict[str, int] | None = None
    best_threshold = 0.0
    for mapping in _mapping_candidates(fit_records):
        mapping_key = tuple(sorted(mapping.items()))
        for threshold in _candidate_thresholds(fit_records):
            score, _ = _evaluate_opening_model(
                fit_records,
                mapping,
                float(threshold),
                config.prefix_trigger_confirmation_frames,
            )
            candidate = (score, -float(threshold), mapping_key)
            if best is None or candidate > best:
                best = candidate
                best_mapping = mapping
                best_threshold = float(threshold)
    _require(best_mapping is not None, "contact-model search failed")
    source_score, source_metrics = _evaluate_opening_model(
        fit_records,
        best_mapping,
        best_threshold,
        config.prefix_trigger_confirmation_frames,
    )
    calibration_score, calibration_metrics = _evaluate_opening_model(
        [record for record in calibration_records if not record["nonprehensile"]],
        best_mapping,
        best_threshold,
        config.prefix_trigger_confirmation_frames,
    )

    def compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: record[key]
            for key in (
                "episode_id",
                "action",
                "bimanual",
                "nonprehensile",
                "group_windows",
                "mono_event_group",
                "sensor_diagnostics",
                "robot_sha256",
                "tactile_inputs",
            )
        }

    return _finish_artifact(
        {
            "schema_version": DEFORM360_CONTACT_SCHEMA_VERSION,
            "artifact_kind": "Deform360ContactModel",
            "protocol_id": config.protocol_id,
            "protocol_config_sha256": protocol_config_sha256(
                {"schema_version": 1, "config": config.as_dict()}
            ),
            "information_boundary": {
                "source_tactile_read": True,
                "calibration_tactile_read_for_validation_only": True,
                "target_robot_read": False,
                "target_tactile_read": False,
                "nonprehensile_source_used_for_threshold_fit": False,
            },
            "model": {
                "opening_contact_threshold_m": best_threshold,
                "confirmation_frames": config.prefix_trigger_confirmation_frames,
                "tactile_group_to_robot_axis": best_mapping,
                "target_trigger_aggregation": config.prefix_trigger_aggregation,
                "target_prefix_frame_count": config.prefix_frame_count,
                "tactile_contact_rule": (
                    f">1 active taxel over rows 0:{TACTILE_ROWS_USED}, release "
                    f"patience {CONTACT_PATIENCE_FRAMES} frames"
                ),
            },
            "fit": {
                "selection_metric": "episode/group balanced accuracy",
                "source_mean_balanced_accuracy": source_score,
                "source_episode_metrics": source_metrics,
                "calibration_mean_balanced_accuracy": calibration_score,
                "calibration_episode_metrics": calibration_metrics,
            },
            "inputs": {
                "metadata_sha256": metadata_sha256,
                "source": [compact_record(record) for record in source_records],
                "calibration": [
                    compact_record(record) for record in calibration_records
                ],
                "target_episode_ids_touched": [],
            },
            "claim_boundary": {
                "tactile_reference": (
                    "unitless normal-response contact reference, not calibrated force "
                    "or slip ground truth"
                ),
                "visual_model": (
                    "opening-only causal baseline; RGB contact prediction remains future work"
                ),
            },
        }
    )


def seal_target_contact_predictions(
    processed_root: str | Path,
    config: Deform360ProtocolConfig,
    contact_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal target visual and prefix-tactile contact estimates without oracle access."""

    validate_contact_artifact(contact_model, expected_kind="Deform360ContactModel")
    _require(
        contact_model["protocol_id"] == config.protocol_id,
        "contact model/protocol mismatch",
    )
    _require(len(config.target_episode_ids) == 1, "expected one target episode")
    index = config.target_episode_ids[0]
    episode_dir = _episode_dir(Path(processed_root).resolve(), index)
    robot = _load_robot(episode_dir)
    threshold = float(contact_model["model"]["opening_contact_threshold_m"])
    confirmation = int(contact_model["model"]["confirmation_frames"])
    visual_by_axis = [
        _causal_confirmed(robot["openings"][:, axis] <= threshold, confirmation)
        for axis in range(robot["openings"].shape[1])
    ]
    trigger_signal = np.logical_and.reduce(visual_by_axis)
    trigger_indices = np.flatnonzero(trigger_signal)
    _require(len(trigger_indices) > 0, "visual trigger never activated on target")
    prefix_start = int(trigger_indices[0])
    prefix_stop = prefix_start + config.prefix_frame_count
    _require(
        prefix_stop <= robot["openings"].shape[0],
        "target tactile prefix extends beyond the episode",
    )
    groups, tactile_inputs = _load_tactile_groups(
        episode_dir,
        frame_slice=slice(prefix_start, prefix_stop),
        hash_full_inputs=False,
        threshold=config.tactile_contact_threshold,
    )
    mapping = {
        str(group): int(axis)
        for group, axis in contact_model["model"]["tactile_group_to_robot_axis"].items()
    }
    prefix_evidence = []
    for group, axis in sorted(mapping.items(), key=lambda item: item[1]):
        _require(group in groups, f"target tactile group is missing: {group}")
        active = groups[group] > 1
        recent = active[-confirmation:]
        prefix_evidence.append(
            {
                "robot_axis": axis,
                "tactile_group": group,
                "active_frames_in_prefix": int(np.count_nonzero(active)),
                "active_fraction_in_prefix": float(np.mean(active)),
                "active_taxel_peak": int(np.max(groups[group])),
                "contact_at_prefix_end": bool(
                    np.count_nonzero(recent) >= (len(recent) + 1) // 2
                ),
            }
        )
    return _finish_artifact(
        {
            "schema_version": DEFORM360_CONTACT_SCHEMA_VERSION,
            "artifact_kind": "Deform360TargetContactPredictionSeal",
            "protocol_id": config.protocol_id,
            "contact_model_sha256": contact_model["result_sha256"],
            "target_episode_id": _episode_id(index),
            "information_boundary": {
                "target_robot_trajectory_read": True,
                "target_robot_role": "action-conditioning evidence",
                "target_tactile_prefix_read": True,
                "target_tactile_oracle_read": False,
                "prediction_metrics_computed": False,
                "target_tactile_used_to_choose_prefix": False,
            },
            "visual_only": {
                "opening_threshold_m": threshold,
                "confirmation_frames": confirmation,
                "state_by_robot_axis": [
                    prediction.astype(bool).tolist() for prediction in visual_by_axis
                ],
            },
            "target_prefix": {
                "selection_method": config.prefix_trigger_method,
                "aggregation": config.prefix_trigger_aggregation,
                "start_frame": prefix_start,
                "stop_frame_exclusive": prefix_stop,
                "frame_count": config.prefix_frame_count,
                "tactile_conditioned_z": prefix_evidence,
            },
            "inputs": {
                "target_robot_sha256": robot["sha256"],
                "target_tactile_prefix_slices": tactile_inputs,
                "target_tactile_full_file_hashes_computed": False,
            },
        }
    )


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def evaluate_target_contact_oracle(
    processed_root: str | Path,
    config: Deform360ProtocolConfig,
    contact_model: Mapping[str, Any],
    prediction_seal: Mapping[str, Any],
    *,
    held_out_prediction_seal_sha256: str,
) -> dict[str, Any]:
    """Open full target tactile only after held-out predictions are immutable."""

    validate_contact_artifact(contact_model, expected_kind="Deform360ContactModel")
    validate_contact_artifact(
        prediction_seal, expected_kind="Deform360TargetContactPredictionSeal"
    )
    _require(
        prediction_seal["contact_model_sha256"] == contact_model["result_sha256"],
        "prediction seal/contact model mismatch",
    )
    _require(
        _valid_sha256(held_out_prediction_seal_sha256),
        "a valid downstream held-out prediction seal is required",
    )
    index = config.target_episode_ids[0]
    episode_dir = _episode_dir(Path(processed_root).resolve(), index)
    groups, tactile_inputs = _load_tactile_groups(
        episode_dir,
        frame_slice=None,
        hash_full_inputs=True,
        threshold=config.tactile_contact_threshold,
    )
    confirmation = int(contact_model["model"]["confirmation_frames"])
    mapping = {
        str(group): int(axis)
        for group, axis in contact_model["model"]["tactile_group_to_robot_axis"].items()
    }
    visual_by_axis = [
        np.asarray(values, dtype=bool)
        for values in prediction_seal["visual_only"]["state_by_robot_axis"]
    ]
    per_gripper = []
    oracle_by_axis = []
    tactile_online_by_axis = []
    for group, axis in sorted(mapping.items(), key=lambda item: item[1]):
        _require(group in groups, f"target tactile group is missing: {group}")
        raw_active = groups[group] > 1
        oracle, start, end = _official_contact_window(raw_active)
        online = _causal_confirmed(raw_active, confirmation)
        visual_metrics = _binary_metrics(oracle, visual_by_axis[axis])
        tactile_metrics = _binary_metrics(oracle, online)
        per_gripper.append(
            {
                "robot_axis": axis,
                "tactile_group": group,
                "oracle_contact_start_frame": start,
                "oracle_contact_end_frame": end,
                "visual_only": visual_metrics,
                "tactile_conditioned_z": tactile_metrics,
                "oracle_tactile": _binary_metrics(oracle, oracle),
            }
        )
        oracle_by_axis.append(oracle)
        tactile_online_by_axis.append(online)
    oracle_union = np.logical_or.reduce(oracle_by_axis)
    visual_union = np.logical_or.reduce(visual_by_axis)
    tactile_union = np.logical_or.reduce(tactile_online_by_axis)
    return _finish_artifact(
        {
            "schema_version": DEFORM360_CONTACT_SCHEMA_VERSION,
            "artifact_kind": "Deform360TargetContactOracleEvaluation",
            "protocol_id": config.protocol_id,
            "contact_model_sha256": contact_model["result_sha256"],
            "contact_prediction_seal_sha256": prediction_seal["result_sha256"],
            "held_out_prediction_seal_sha256": held_out_prediction_seal_sha256,
            "target_episode_id": _episode_id(index),
            "information_boundary": {
                "target_contact_predictions_previously_sealed": True,
                "held_out_future_predictions_previously_sealed": True,
                "target_tactile_oracle_read": True,
                "target_tactile_used_for_fitting": False,
            },
            "per_gripper": per_gripper,
            "episode_union": {
                "visual_only": _binary_metrics(oracle_union, visual_union),
                "tactile_conditioned_z": _binary_metrics(oracle_union, tactile_union),
                "oracle_tactile": _binary_metrics(oracle_union, oracle_union),
            },
            "contact_state_by_robot_axis": {
                "visual_only": [
                    values.astype(bool).tolist() for values in visual_by_axis
                ],
                "tactile_conditioned_z": [
                    values.astype(bool).tolist() for values in tactile_online_by_axis
                ],
                "oracle_tactile": [
                    values.astype(bool).tolist() for values in oracle_by_axis
                ],
            },
            "inputs": {"target_tactile": tactile_inputs},
            "claim_boundary": (
                "Full tactile is an offline normal-response contact reference/upper "
                "bound, not calibrated force or slip ground truth."
            ),
        }
    )


__all__ = [
    "DEFORM360_CONTACT_SCHEMA_VERSION",
    "contact_artifact_sha256",
    "evaluate_target_contact_oracle",
    "fit_contact_model",
    "load_contact_artifact",
    "seal_target_contact_predictions",
    "validate_contact_artifact",
    "write_contact_artifact",
]
