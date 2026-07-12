"""Cross-action selection and transfer contracts for rest-geometry validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.real_protocol import load_protocol, validate_protocol


CANDIDATE_EVIDENCE_SCHEMA_VERSION = 1
FOLD_LOCK_SCHEMA_VERSION = 1
TRANSFER_PLAN_SCHEMA_VERSION = 1
PROTOCOL_RESULT_SCHEMA_VERSION = 1
SELECTION_RULE = (
    "one_standard_error_on_equal_execution_mean_log_validation_error_ratio"
)

_FRAME_MODES = ("none", "translation", "se3")
_CONTROLLER_REST_MODES = ("preserve", "recompute")
_HYPERPARAMETER_FIELDS = (
    "frame_mode",
    "frame_scale",
    "rest_geometry_scale",
    "controller_rest_mode",
    "graph_prior_strength",
    "rest_length_ratio_bound",
)
_EVIDENCE_BOUNDARY = {
    "evidence_frames": "pre_holdout_only",
    "holdout_frames_used": False,
    "manual_gt_track_used": False,
}
_RESULT_BOUNDARY = {
    "target_holdout_frames_used_for_inference": False,
    "target_outcomes_used_for_hyperparameter_selection": False,
    "manual_gt_track_used_for_hyperparameter_selection": False,
    "individual_counterfactual_ground_truth_claimed": False,
}
_RESULT_METHODS = (
    "released",
    "endpoint_restart",
    "output_frame_graph",
    "frame_state_original_rest",
    "graph_state_original_rest",
    "rest_geometry_only",
    "frame_rest_geometry",
    "frame_rest_geometry_reattached",
    "selected_frame_rest_geometry",
)
_RESULT_METRICS = (
    "future_chamfer_distance_m",
    "future_track_error_m",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def canonical_rest_geometry_hyperparameters(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize one shared correction candidate."""

    if set(value) != set(_HYPERPARAMETER_FIELDS):
        raise ValueError(
            "rest-geometry hyperparameters must contain exactly: "
            + ", ".join(_HYPERPARAMETER_FIELDS)
        )
    frame_mode = str(value["frame_mode"])
    controller_mode = str(value["controller_rest_mode"])
    if frame_mode not in _FRAME_MODES:
        raise ValueError(f"frame_mode must lie in {_FRAME_MODES}")
    if controller_mode not in _CONTROLLER_REST_MODES:
        raise ValueError(
            f"controller_rest_mode must lie in {_CONTROLLER_REST_MODES}"
        )
    frame_scale = _finite_number(value["frame_scale"], name="frame_scale")
    rest_scale = _finite_number(
        value["rest_geometry_scale"], name="rest_geometry_scale"
    )
    if not 0.0 <= frame_scale <= 1.0 or not 0.0 <= rest_scale <= 1.0:
        raise ValueError("frame and rest-geometry scales must lie in [0, 1]")
    graph_strength = _finite_number(
        value["graph_prior_strength"],
        name="graph_prior_strength",
        positive=True,
    )
    rest_bound = _finite_number(
        value["rest_length_ratio_bound"],
        name="rest_length_ratio_bound",
        positive=True,
    )
    if rest_bound <= 1.0:
        raise ValueError("rest_length_ratio_bound must exceed one")
    return {
        "frame_mode": frame_mode,
        "frame_scale": frame_scale,
        "rest_geometry_scale": rest_scale,
        "controller_rest_mode": controller_mode,
        "graph_prior_strength": graph_strength,
        "rest_length_ratio_bound": rest_bound,
    }


def rest_geometry_hyperparameter_id(value: Mapping[str, Any]) -> str:
    """Hash one canonical hyperparameter candidate."""

    return _sha256_json(canonical_rest_geometry_hyperparameters(value))


def identity_rest_geometry_hyperparameters(
    *,
    graph_prior_strength: float,
    rest_length_ratio_bound: float,
) -> dict[str, Any]:
    """Return the unique uncorrected candidate used to normalize executions."""

    return canonical_rest_geometry_hyperparameters(
        {
            "frame_mode": "none",
            "frame_scale": 0.0,
            "rest_geometry_scale": 0.0,
            "controller_rest_mode": "preserve",
            "graph_prior_strength": graph_prior_strength,
            "rest_length_ratio_bound": rest_length_ratio_bound,
        }
    )


def candidate_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    """Hash candidate evidence while excluding its self-digest field."""

    payload = deepcopy(dict(evidence))
    payload.pop("evidence_sha256", None)
    return _sha256_json(payload)


def build_rest_geometry_candidate_evidence(
    protocol: Mapping[str, Any],
    execution_id: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    fixed_config_sha256: str,
    source_artifacts_sha256: str,
) -> dict[str, Any]:
    """Build one execution's pre-holdout-only candidate score artifact."""

    validate_protocol(protocol)
    execution_ids = {
        execution["execution_id"] for execution in protocol["executions"]
    }
    if execution_id not in execution_ids:
        raise ValueError("candidate evidence execution is not in the protocol")
    if not _is_sha256(fixed_config_sha256) or not _is_sha256(
        source_artifacts_sha256
    ):
        raise ValueError("candidate evidence input digests must be SHA-256 values")
    records = []
    seen = set()
    for record in candidates:
        hyperparameters = canonical_rest_geometry_hyperparameters(
            record["hyperparameters"]
        )
        candidate_id = rest_geometry_hyperparameter_id(hyperparameters)
        if candidate_id in seen:
            raise ValueError("candidate evidence contains duplicate hyperparameters")
        seen.add(candidate_id)
        by_frame = np.asarray(
            record["validation_track_error_by_frame_m"], dtype=float
        )
        if (
            by_frame.ndim != 1
            or len(by_frame) < 1
            or not np.all(np.isfinite(by_frame))
            or np.any(by_frame <= 0.0)
        ):
            raise ValueError(
                "validation_track_error_by_frame_m must be positive and finite"
            )
        mean_error = _finite_number(
            record["validation_track_error_mean_m"],
            name="validation_track_error_mean_m",
            positive=True,
        )
        if not np.isclose(mean_error, float(np.mean(by_frame)), rtol=1e-10):
            raise ValueError("candidate mean does not match its frame errors")
        records.append(
            {
                "candidate_id": candidate_id,
                "hyperparameters": hyperparameters,
                "validation_track_error_by_frame_m": by_frame.tolist(),
                "validation_track_error_mean_m": mean_error,
                "validation_frame_count": len(by_frame),
            }
        )
    if len(records) < 2:
        raise ValueError("candidate evidence must compare at least two candidates")
    identity_count = sum(
        record["hyperparameters"]["frame_mode"] == "none"
        and record["hyperparameters"]["frame_scale"] == 0.0
        and record["hyperparameters"]["rest_geometry_scale"] == 0.0
        and record["hyperparameters"]["controller_rest_mode"] == "preserve"
        for record in records
    )
    if identity_count != 1:
        raise ValueError("candidate evidence must contain one canonical identity")
    records.sort(key=lambda record: record["candidate_id"])
    evidence = {
        "schema_version": CANDIDATE_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": "rest_geometry_candidate_evidence",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "execution_id": execution_id,
        "information_boundary": dict(_EVIDENCE_BOUNDARY),
        "fixed_config_sha256": fixed_config_sha256,
        "source_artifacts_sha256": source_artifacts_sha256,
        "candidates": records,
    }
    evidence["evidence_sha256"] = candidate_evidence_sha256(evidence)
    validate_rest_geometry_candidate_evidence(protocol, evidence)
    return evidence


def _case_summary_source_artifacts_sha256(summary: Mapping[str, Any]) -> str:
    inputs = summary.get("inputs", {})
    required = (
        "final_data",
        "baseline_trajectory",
        "optimal_params",
        "checkpoint",
    )
    digests = {}
    for name in required:
        descriptor = inputs.get(name, {})
        if not _is_sha256(descriptor.get("sha256")):
            raise ValueError(f"case summary input digest is invalid: {name}")
        digests[name] = descriptor["sha256"]
    track = inputs.get("gt_track_3d")
    if track is not None:
        if not _is_sha256(track.get("sha256")):
            raise ValueError("case summary gt_track_3d digest is invalid")
        digests["gt_track_3d"] = track["sha256"]
    official = inputs.get("official_repo", {})
    commit = official.get("commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError("case summary official PhysTwin commit is missing")
    digests["official_phystwin_commit"] = commit
    return _sha256_json(digests)


def _case_summary_fixed_config(summary: Mapping[str, Any]) -> dict[str, Any]:
    config = summary.get("config", {})
    fields = (
        "inner_validation_frames",
        "velocity_history_frames",
        "maximum_frame_rotation_rad",
        "maximum_frame_translation_m",
        "maximum_nonrigid_norm_m",
        "maximum_rest_log_ratio",
        "dt",
        "num_substeps",
        "self_collision",
        "deterministic_spring_forces",
    )
    missing = [field for field in fields if field not in config]
    if missing:
        raise ValueError(
            "case summary fixed configuration is missing: " + ", ".join(missing)
        )
    return {field: config[field] for field in fields}


def build_candidate_evidence_from_case_summaries(
    protocol: Mapping[str, Any],
    execution_id: str,
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract pre-holdout candidate evidence from per-frame-mode Warp summaries."""

    if not summaries:
        raise ValueError("at least one rest-geometry case summary is required")
    modes = []
    candidates = []
    source_digests = set()
    fixed_configs = []
    for summary in summaries:
        boundary = summary.get("information_boundary", {})
        if (
            boundary.get("holdout_frames_used_for_inference") is not False
            or boundary.get("holdout_frames_used_for_hyperparameter_selection")
            is not False
            or boundary.get("manual_gt_track_used_for_hyperparameter_selection")
            is not False
        ):
            raise ValueError("case summary crossed the candidate evidence boundary")
        config = summary.get("config", {})
        frame_mode = config.get("frame_mode")
        if frame_mode not in _FRAME_MODES or frame_mode in modes:
            raise ValueError("case summaries must use unique supported frame modes")
        modes.append(frame_mode)
        graph_strength = _finite_number(
            config.get("graph_prior_strength"),
            name="graph_prior_strength",
            positive=True,
        )
        maximum_rest_log_ratio = _finite_number(
            config.get("maximum_rest_log_ratio"),
            name="maximum_rest_log_ratio",
            positive=True,
        )
        selection = summary.get("selection", {})
        for candidate in selection.get("candidates", []):
            candidates.append(
                {
                    "hyperparameters": {
                        "frame_mode": frame_mode,
                        "frame_scale": candidate["frame_scale"],
                        "rest_geometry_scale": candidate[
                            "rest_geometry_scale"
                        ],
                        "controller_rest_mode": candidate[
                            "controller_rest_mode"
                        ],
                        "graph_prior_strength": graph_strength,
                        "rest_length_ratio_bound": float(
                            np.exp(maximum_rest_log_ratio)
                        ),
                    },
                    "validation_track_error_by_frame_m": candidate[
                        "track_error_by_frame_m"
                    ],
                    "validation_track_error_mean_m": candidate[
                        "track_error_mean_m"
                    ],
                }
            )
        source_digests.add(_case_summary_source_artifacts_sha256(summary))
        fixed_configs.append(_case_summary_fixed_config(summary))
    if "none" not in modes:
        raise ValueError("case summaries must include the canonical none frame mode")
    if len(source_digests) != 1:
        raise ValueError("case summaries use different source artifacts")
    canonical_fixed = _canonical_bytes(fixed_configs[0])
    if any(_canonical_bytes(value) != canonical_fixed for value in fixed_configs[1:]):
        raise ValueError("case summaries use different fixed configurations")
    return build_rest_geometry_candidate_evidence(
        protocol,
        execution_id,
        candidates,
        fixed_config_sha256=_sha256_json(fixed_configs[0]),
        source_artifacts_sha256=next(iter(source_digests)),
    )


def validate_rest_geometry_candidate_evidence(
    protocol: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one candidate artifact and its pre-holdout boundary."""

    validate_protocol(protocol)
    if evidence.get("schema_version") != CANDIDATE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported candidate evidence schema")
    if evidence.get("artifact_kind") != "rest_geometry_candidate_evidence":
        raise ValueError("unexpected candidate evidence kind")
    if evidence.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("candidate evidence protocol mismatch")
    if evidence.get("protocol_design_sha256") != protocol["design_sha256"]:
        raise ValueError("candidate evidence protocol digest mismatch")
    execution_ids = {
        execution["execution_id"] for execution in protocol["executions"]
    }
    if evidence.get("execution_id") not in execution_ids:
        raise ValueError("candidate evidence execution is not preregistered")
    if evidence.get("information_boundary") != _EVIDENCE_BOUNDARY:
        raise ValueError("candidate evidence crossed the holdout boundary")
    if not _is_sha256(evidence.get("fixed_config_sha256")) or not _is_sha256(
        evidence.get("source_artifacts_sha256")
    ):
        raise ValueError("candidate evidence input digest is invalid")
    if evidence.get("evidence_sha256") != candidate_evidence_sha256(evidence):
        raise ValueError("candidate evidence SHA-256 mismatch")
    records = list(evidence.get("candidates", []))
    if len(records) < 2:
        raise ValueError("candidate evidence has too few candidates")
    candidate_ids = []
    identity_count = 0
    for record in records:
        hyperparameters = canonical_rest_geometry_hyperparameters(
            record.get("hyperparameters", {})
        )
        candidate_id = rest_geometry_hyperparameter_id(hyperparameters)
        if record.get("candidate_id") != candidate_id:
            raise ValueError("candidate hyperparameter digest mismatch")
        candidate_ids.append(candidate_id)
        by_frame = np.asarray(
            record.get("validation_track_error_by_frame_m", []), dtype=float
        )
        if (
            by_frame.ndim != 1
            or len(by_frame) < 1
            or not np.all(np.isfinite(by_frame))
            or np.any(by_frame <= 0.0)
        ):
            raise ValueError("candidate frame errors are invalid")
        if record.get("validation_frame_count") != len(by_frame):
            raise ValueError("candidate validation frame count changed")
        mean_error = _finite_number(
            record.get("validation_track_error_mean_m"),
            name="validation_track_error_mean_m",
            positive=True,
        )
        if not np.isclose(mean_error, float(np.mean(by_frame)), rtol=1e-10):
            raise ValueError("candidate mean does not match frame errors")
        identity_count += int(
            hyperparameters["frame_mode"] == "none"
            and hyperparameters["frame_scale"] == 0.0
            and hyperparameters["rest_geometry_scale"] == 0.0
            and hyperparameters["controller_rest_mode"] == "preserve"
        )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate evidence contains duplicates")
    if candidate_ids != sorted(candidate_ids):
        raise ValueError("candidate evidence is not canonically ordered")
    if identity_count != 1:
        raise ValueError("candidate evidence identity is ambiguous")
    return {
        "execution_id": evidence["execution_id"],
        "candidate_count": len(records),
        "passed": True,
    }


def _candidate_complexity(hyperparameters: Mapping[str, Any]) -> tuple[Any, ...]:
    frame_rank = {"none": 0, "translation": 1, "se3": 2}
    return (
        hyperparameters["frame_scale"] + hyperparameters["rest_geometry_scale"],
        hyperparameters["rest_geometry_scale"],
        hyperparameters["frame_scale"],
        frame_rank[hyperparameters["frame_mode"]],
        hyperparameters["controller_rest_mode"] != "preserve",
        rest_geometry_hyperparameter_id(hyperparameters),
    )


def fold_lock_sha256(lock: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(lock))
    payload.pop("lock_sha256", None)
    return _sha256_json(payload)


def select_rest_geometry_fold_lock(
    protocol: Mapping[str, Any],
    fold_id: str,
    evidence_by_execution: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select shared hyperparameters using exactly one fold's fit executions."""

    validate_protocol(protocol)
    folds = {
        fold["fold_id"]: fold
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    }
    if fold_id not in folds:
        raise ValueError("fold is not preregistered")
    fold = folds[fold_id]
    fit_ids = sorted(fold["fit_execution_ids"])
    if sorted(evidence_by_execution) != fit_ids:
        raise ValueError("fold selection must receive exactly its fit executions")

    validated = {}
    candidate_sets = []
    fixed_config_digests = set()
    for execution_id in fit_ids:
        evidence = evidence_by_execution[execution_id]
        validate_rest_geometry_candidate_evidence(protocol, evidence)
        if evidence["execution_id"] != execution_id:
            raise ValueError("candidate evidence is indexed under the wrong execution")
        validated[execution_id] = evidence
        candidate_sets.append(
            tuple(record["candidate_id"] for record in evidence["candidates"])
        )
        fixed_config_digests.add(evidence["fixed_config_sha256"])
    if len(set(candidate_sets)) != 1:
        raise ValueError("fit executions do not share an identical candidate grid")
    if len(fixed_config_digests) != 1:
        raise ValueError("fit executions used different fixed configurations")

    candidate_ids = candidate_sets[0]
    records_by_execution = {
        execution_id: {
            record["candidate_id"]: record
            for record in evidence["candidates"]
        }
        for execution_id, evidence in validated.items()
    }
    identity_ids = []
    for candidate_id in candidate_ids:
        hyperparameters = records_by_execution[fit_ids[0]][candidate_id][
            "hyperparameters"
        ]
        if (
            hyperparameters["frame_mode"] == "none"
            and hyperparameters["frame_scale"] == 0.0
            and hyperparameters["rest_geometry_scale"] == 0.0
            and hyperparameters["controller_rest_mode"] == "preserve"
        ):
            identity_ids.append(candidate_id)
    if len(identity_ids) != 1:
        raise ValueError("shared candidate grid has no unique identity")
    identity_id = identity_ids[0]

    selection_table = []
    for candidate_id in candidate_ids:
        ratios = []
        for execution_id in fit_ids:
            candidate_error = records_by_execution[execution_id][candidate_id][
                "validation_track_error_mean_m"
            ]
            identity_error = records_by_execution[execution_id][identity_id][
                "validation_track_error_mean_m"
            ]
            ratios.append(candidate_error / identity_error)
        log_ratios = np.log(np.asarray(ratios, dtype=float))
        mean_log = float(np.mean(log_ratios))
        standard_error = float(
            np.std(log_ratios, ddof=1) / np.sqrt(len(log_ratios))
        )
        hyperparameters = records_by_execution[fit_ids[0]][candidate_id][
            "hyperparameters"
        ]
        selection_table.append(
            {
                "candidate_id": candidate_id,
                "hyperparameters": hyperparameters,
                "per_execution_error_ratio": {
                    execution_id: float(ratio)
                    for execution_id, ratio in zip(fit_ids, ratios)
                },
                "mean_log_error_ratio": mean_log,
                "standard_error_log_ratio": standard_error,
                "geometric_mean_error_ratio": float(np.exp(mean_log)),
                "geometric_mean_percent_change": 100.0 * (np.exp(mean_log) - 1.0),
            }
        )
    best = min(selection_table, key=lambda record: record["mean_log_error_ratio"])
    one_se_threshold = (
        best["mean_log_error_ratio"] + best["standard_error_log_ratio"]
    )
    eligible = [
        record
        for record in selection_table
        if record["mean_log_error_ratio"] <= one_se_threshold + 1e-15
    ]
    selected = min(
        eligible,
        key=lambda record: _candidate_complexity(record["hyperparameters"]),
    )
    selection_table.sort(
        key=lambda record: (
            record["mean_log_error_ratio"],
            _candidate_complexity(record["hyperparameters"]),
        )
    )
    lock = {
        "schema_version": FOLD_LOCK_SCHEMA_VERSION,
        "artifact_kind": "rest_geometry_fold_lock",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "fold_id": fold_id,
        "selection_rule": SELECTION_RULE,
        "hyperparameter_fit_execution_ids": fit_ids,
        "calibration_execution_ids": sorted(fold["calibration_execution_ids"]),
        "target_execution_ids": sorted(fold["target_execution_ids"]),
        "candidate_evidence_sha256": {
            execution_id: validated[execution_id]["evidence_sha256"]
            for execution_id in fit_ids
        },
        "fixed_config_sha256": next(iter(fixed_config_digests)),
        "identity_candidate_id": identity_id,
        "best_mean_candidate_id": best["candidate_id"],
        "one_standard_error_threshold_log_ratio": one_se_threshold,
        "selected_candidate_id": selected["candidate_id"],
        "selected_hyperparameters": selected["hyperparameters"],
        "selection_table": selection_table,
        "information_boundary": {
            "fit_execution_evidence_only": True,
            "calibration_execution_evidence_used": False,
            "target_execution_evidence_used": False,
            "target_outcomes_used": False,
        },
    }
    lock["lock_sha256"] = fold_lock_sha256(lock)
    validate_rest_geometry_fold_lock(protocol, lock)
    return lock


def validate_rest_geometry_fold_lock(
    protocol: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate fold roles, selected candidate, and lock digest."""

    validate_protocol(protocol)
    if lock.get("schema_version") != FOLD_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported rest-geometry fold lock schema")
    if lock.get("artifact_kind") != "rest_geometry_fold_lock":
        raise ValueError("unexpected rest-geometry fold lock kind")
    if lock.get("protocol_id") != protocol["protocol_id"] or lock.get(
        "protocol_design_sha256"
    ) != protocol["design_sha256"]:
        raise ValueError("rest-geometry fold lock protocol mismatch")
    folds = {
        fold["fold_id"]: fold
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    }
    fold_id = lock.get("fold_id")
    if fold_id not in folds:
        raise ValueError("rest-geometry fold lock is not preregistered")
    fold = folds[fold_id]
    roles = (
        ("hyperparameter_fit_execution_ids", "fit_execution_ids"),
        ("calibration_execution_ids", "calibration_execution_ids"),
        ("target_execution_ids", "target_execution_ids"),
    )
    for lock_field, fold_field in roles:
        if sorted(lock.get(lock_field, [])) != sorted(fold[fold_field]):
            raise ValueError(f"rest-geometry fold lock changed {lock_field}")
    if lock.get("selection_rule") != SELECTION_RULE:
        raise ValueError("rest-geometry fold selection rule changed")
    if lock.get("information_boundary") != {
        "fit_execution_evidence_only": True,
        "calibration_execution_evidence_used": False,
        "target_execution_evidence_used": False,
        "target_outcomes_used": False,
    }:
        raise ValueError("rest-geometry fold lock crossed its information boundary")
    fit_ids = sorted(fold["fit_execution_ids"])
    evidence_digests = lock.get("candidate_evidence_sha256", {})
    if sorted(evidence_digests) != fit_ids or not all(
        _is_sha256(value) for value in evidence_digests.values()
    ):
        raise ValueError("rest-geometry fold evidence digests are incomplete")
    if not _is_sha256(lock.get("fixed_config_sha256")):
        raise ValueError("rest-geometry fixed configuration digest is invalid")
    table = list(lock.get("selection_table", []))
    table_ids = [record.get("candidate_id") for record in table]
    if not table or len(table_ids) != len(set(table_ids)):
        raise ValueError("rest-geometry fold selection table is invalid")
    for record in table:
        hyperparameters = canonical_rest_geometry_hyperparameters(
            record.get("hyperparameters", {})
        )
        if record.get("candidate_id") != rest_geometry_hyperparameter_id(
            hyperparameters
        ):
            raise ValueError("rest-geometry fold table candidate digest changed")
        ratios = record.get("per_execution_error_ratio", {})
        if sorted(ratios) != fit_ids:
            raise ValueError("rest-geometry fold table execution ratios changed")
        ratio_values = np.asarray([ratios[key] for key in fit_ids], dtype=float)
        if not np.all(np.isfinite(ratio_values)) or np.any(ratio_values <= 0.0):
            raise ValueError("rest-geometry fold table error ratios are invalid")
        log_ratios = np.log(ratio_values)
        expected_mean = float(np.mean(log_ratios))
        expected_se = float(
            np.std(log_ratios, ddof=1) / np.sqrt(len(log_ratios))
        )
        if not np.isclose(record.get("mean_log_error_ratio"), expected_mean):
            raise ValueError("rest-geometry fold table mean changed")
        if not np.isclose(record.get("standard_error_log_ratio"), expected_se):
            raise ValueError("rest-geometry fold table standard error changed")
        if not np.isclose(
            record.get("geometric_mean_error_ratio"), np.exp(expected_mean)
        ):
            raise ValueError("rest-geometry fold table geometric mean changed")
        if not np.isclose(
            record.get("geometric_mean_percent_change"),
            100.0 * (np.exp(expected_mean) - 1.0),
        ):
            raise ValueError("rest-geometry fold table percent change changed")
    best = min(table, key=lambda record: record["mean_log_error_ratio"])
    expected_threshold = (
        best["mean_log_error_ratio"] + best["standard_error_log_ratio"]
    )
    if lock.get("best_mean_candidate_id") != best["candidate_id"] or not np.isclose(
        lock.get("one_standard_error_threshold_log_ratio"), expected_threshold
    ):
        raise ValueError("rest-geometry fold one-standard-error threshold changed")
    eligible = [
        record
        for record in table
        if record["mean_log_error_ratio"] <= expected_threshold + 1e-15
    ]
    expected_selected = min(
        eligible,
        key=lambda record: _candidate_complexity(record["hyperparameters"]),
    )
    selected = canonical_rest_geometry_hyperparameters(
        lock.get("selected_hyperparameters", {})
    )
    selected_id = rest_geometry_hyperparameter_id(selected)
    if (
        lock.get("selected_candidate_id") != selected_id
        or selected_id != expected_selected["candidate_id"]
    ):
        raise ValueError("rest-geometry selected candidate is inconsistent")
    if lock.get("lock_sha256") != fold_lock_sha256(lock):
        raise ValueError("rest-geometry fold lock SHA-256 mismatch")
    return {"fold_id": fold_id, "selected_candidate_id": selected_id, "passed": True}


def select_all_rest_geometry_fold_locks(
    protocol: Mapping[str, Any],
    evidence_by_execution: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select all 12 folds while enforcing fit-only evidence per fold."""

    validate_protocol(protocol)
    execution_ids = {
        execution["execution_id"] for execution in protocol["executions"]
    }
    if set(evidence_by_execution) != execution_ids:
        raise ValueError("all protocol executions need candidate evidence")
    return {
        fold["fold_id"]: select_rest_geometry_fold_lock(
            protocol,
            fold["fold_id"],
            {
                execution_id: evidence_by_execution[execution_id]
                for execution_id in fold["fit_execution_ids"]
            },
        )
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    }


def build_rest_geometry_transfer_plan(
    protocol: Mapping[str, Any],
    fold_locks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind frozen fold locks to factual and source-to-target evaluations."""

    validate_protocol(protocol)
    expected_fold_ids = {
        fold["fold_id"]
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    }
    if set(fold_locks) != expected_fold_ids:
        raise ValueError("transfer plan requires every preregistered fold lock")
    for fold_id, lock in fold_locks.items():
        validate_rest_geometry_fold_lock(protocol, lock)
        if lock["fold_id"] != fold_id:
            raise ValueError("fold lock is indexed under the wrong id")

    target_fold = {}
    for fold_id, lock in fold_locks.items():
        for execution_id in lock["target_execution_ids"]:
            if execution_id in target_fold:
                raise ValueError("execution is target in multiple fold locks")
            target_fold[execution_id] = fold_id
    execution_ids = {
        execution["execution_id"] for execution in protocol["executions"]
    }
    if set(target_fold) != execution_ids:
        raise ValueError("fold locks do not cover every execution target")

    def record(
        *,
        track: str,
        record_id: str,
        source_execution_id: str,
        target_execution_id: str,
        correction_evidence_policy: str,
        target_response_prefix_allowed: bool,
        contact_policy: str,
    ) -> dict[str, Any]:
        fold_id = target_fold[target_execution_id]
        lock = fold_locks[fold_id]
        return {
            "record_id": record_id,
            "evaluation_track": track,
            "source_execution_id": source_execution_id,
            "target_execution_id": target_execution_id,
            "fold_id": fold_id,
            "fold_lock_sha256": lock["lock_sha256"],
            "selected_candidate_id": lock["selected_candidate_id"],
            "correction_evidence_policy": correction_evidence_policy,
            "target_response_prefix_allowed": target_response_prefix_allowed,
            "contact_policy": contact_policy,
            "canonical_material_graph_required": True,
        }

    factual = [
        record(
            track="factual_continuation",
            record_id=f"factual::{item['execution_id']}",
            source_execution_id=item["execution_id"],
            target_execution_id=item["execution_id"],
            correction_evidence_policy="target_pre_holdout_only",
            target_response_prefix_allowed=True,
            contact_policy="factual_same_execution",
        )
        for item in protocol["splits"]["factual_continuation"]
    ]
    same_grasp = [
        record(
            track="same_grasp_intervention_prediction",
            record_id=(
                f"same-grasp::{item['source_execution_id']}"
                f"->{item['target_execution_id']}"
            ),
            source_execution_id=item["source_execution_id"],
            target_execution_id=item["target_execution_id"],
            correction_evidence_policy="source_pre_holdout_only",
            target_response_prefix_allowed=False,
            contact_policy="same_grasp",
        )
        for item in protocol["splits"]["same_grasp_intervention_prediction"]
    ]
    new_contact = [
        record(
            track="new_contact_intervention_prediction",
            record_id=(
                f"new-contact::{item['source_execution_id']}"
                f"->{item['target_execution_id']}"
            ),
            source_execution_id=item["source_execution_id"],
            target_execution_id=item["target_execution_id"],
            correction_evidence_policy="source_pre_holdout_only",
            target_response_prefix_allowed=False,
            contact_policy="new_contact",
        )
        for item in protocol["splits"]["new_contact_intervention_prediction"]
    ]
    plan = {
        "schema_version": TRANSFER_PLAN_SCHEMA_VERSION,
        "artifact_kind": "rest_geometry_transfer_plan",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "fold_lock_sha256": {
            fold_id: fold_locks[fold_id]["lock_sha256"]
            for fold_id in sorted(fold_locks)
        },
        "transfer_contract": {
            "persistent_components": [
                "frame_correction",
                "canonical_object_rest_lengths",
                "canonical_nonrigid_material_field",
            ],
            "target_specific_components": [
                "initial_dynamic_state",
                "controller_trajectory",
                "controller_attachment_springs",
            ],
            "same_grasp_attachment_policy": "reuse_registered_grasp",
            "new_contact_attachment_policy": "rebuild_on_registered_target_contact",
            "target_outcome_may_not_change_source_correction": True,
        },
        "factual_continuation": factual,
        "same_grasp_intervention_prediction": same_grasp,
        "new_contact_intervention_prediction": new_contact,
    }
    plan["plan_sha256"] = transfer_plan_sha256(plan)
    validate_rest_geometry_transfer_plan(protocol, fold_locks, plan)
    return plan


def transfer_plan_sha256(plan: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(plan))
    payload.pop("plan_sha256", None)
    return _sha256_json(payload)


def transfer_result_record_sha256(record: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(record))
    payload.pop("record_sha256", None)
    return _sha256_json(payload)


def _canonical_method_metrics(
    metrics_by_method: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, dict[str, list[float]]]:
    if set(metrics_by_method) != set(_RESULT_METHODS):
        raise ValueError(
            "transfer result methods must contain exactly: "
            + ", ".join(_RESULT_METHODS)
        )
    canonical = {}
    expected_frame_count = None
    for method in _RESULT_METHODS:
        metrics = metrics_by_method[method]
        if set(metrics) != set(_RESULT_METRICS):
            raise ValueError(
                f"transfer result metrics for {method} must contain exactly "
                + ", ".join(_RESULT_METRICS)
            )
        canonical[method] = {}
        for metric in _RESULT_METRICS:
            values = np.asarray(metrics[metric], dtype=float)
            if (
                values.ndim != 1
                or len(values) < 1
                or not np.all(np.isfinite(values))
                or np.any(values < 0.0)
            ):
                raise ValueError("transfer result metric arrays must be finite")
            if expected_frame_count is None:
                expected_frame_count = len(values)
            elif len(values) != expected_frame_count:
                raise ValueError("transfer result methods changed the frame window")
            canonical[method][metric] = values.tolist()
    return canonical


def build_rest_geometry_transfer_result_record(
    plan_record: Mapping[str, Any],
    metrics_by_method: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    canonical_material_graph_sha256: str,
    source_correction_sha256: str,
    target_rollout_bundle_sha256: str,
) -> dict[str, Any]:
    """Build one hash-addressed Warp result under a locked transfer record."""

    for name, value in (
        ("canonical_material_graph_sha256", canonical_material_graph_sha256),
        ("source_correction_sha256", source_correction_sha256),
        ("target_rollout_bundle_sha256", target_rollout_bundle_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a SHA-256 value")
    record = {
        "schema_version": PROTOCOL_RESULT_SCHEMA_VERSION,
        "artifact_kind": "rest_geometry_transfer_result_record",
        **{
            field: plan_record[field]
            for field in (
                "record_id",
                "evaluation_track",
                "source_execution_id",
                "target_execution_id",
                "fold_id",
                "fold_lock_sha256",
                "selected_candidate_id",
                "correction_evidence_policy",
                "target_response_prefix_allowed",
                "contact_policy",
                "canonical_material_graph_required",
            )
        },
        "shared_hyperparameters_frozen_before_target": True,
        "target_holdout_outcomes_used_for_correction": False,
        "canonical_material_graph_sha256": canonical_material_graph_sha256,
        "source_correction_sha256": source_correction_sha256,
        "target_rollout_bundle_sha256": target_rollout_bundle_sha256,
        "metrics_by_method": _canonical_method_metrics(metrics_by_method),
    }
    record["record_sha256"] = transfer_result_record_sha256(record)
    return record


def _plan_records_by_id(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for track in (
        "factual_continuation",
        "same_grasp_intervention_prediction",
        "new_contact_intervention_prediction",
    ):
        for record in plan[track]:
            record_id = record["record_id"]
            if record_id in records:
                raise ValueError("transfer plan record ids are not unique")
            records[record_id] = record
    return records


def validate_rest_geometry_transfer_result_record(
    plan_record: Mapping[str, Any],
    result_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one result against its immutable transfer-plan record."""

    if result_record.get("schema_version") != PROTOCOL_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported transfer result record schema")
    if result_record.get("artifact_kind") != "rest_geometry_transfer_result_record":
        raise ValueError("unexpected transfer result record kind")
    plan_fields = (
        "record_id",
        "evaluation_track",
        "source_execution_id",
        "target_execution_id",
        "fold_id",
        "fold_lock_sha256",
        "selected_candidate_id",
        "correction_evidence_policy",
        "target_response_prefix_allowed",
        "contact_policy",
        "canonical_material_graph_required",
    )
    if any(result_record.get(field) != plan_record.get(field) for field in plan_fields):
        raise ValueError("transfer result differs from its locked plan record")
    if result_record.get("shared_hyperparameters_frozen_before_target") is not True:
        raise ValueError("transfer result did not freeze shared hyperparameters")
    if result_record.get("target_holdout_outcomes_used_for_correction") is not False:
        raise ValueError("transfer result used target holdout outcomes for correction")
    for field in (
        "canonical_material_graph_sha256",
        "source_correction_sha256",
        "target_rollout_bundle_sha256",
    ):
        if not _is_sha256(result_record.get(field)):
            raise ValueError("transfer result artifact digest is invalid")
    if result_record.get("canonical_material_graph_required") is not True:
        raise ValueError("transfer result did not preserve canonical graph identity")
    _canonical_method_metrics(result_record.get("metrics_by_method", {}))
    if result_record.get("record_sha256") != transfer_result_record_sha256(
        result_record
    ):
        raise ValueError("transfer result record SHA-256 mismatch")
    return {"record_id": result_record["record_id"], "passed": True}


def _aggregate_transfer_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty transfer track")
    summary = {}
    for method in _RESULT_METHODS:
        summary[method] = {}
        for metric in _RESULT_METRICS:
            released_means = np.asarray(
                [
                    np.mean(record["metrics_by_method"]["released"][metric])
                    for record in records
                ],
                dtype=float,
            )
            candidate_means = np.asarray(
                [
                    np.mean(record["metrics_by_method"][method][metric])
                    for record in records
                ],
                dtype=float,
            )
            if np.any(released_means <= 0.0):
                raise ValueError("released transfer metric mean must be positive")
            per_record_percent = 100.0 * (
                candidate_means / released_means - 1.0
            )
            summary[method][metric] = {
                "record_count": len(records),
                "equal_record_macro_percent_change": float(
                    np.mean(per_record_percent)
                ),
                "per_record_percent_change": {
                    record["record_id"]: float(value)
                    for record, value in zip(records, per_record_percent)
                },
            }
    return summary


def protocol_result_sha256(result: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(result))
    payload.pop("result_sha256", None)
    return _sha256_json(payload)


def build_rest_geometry_protocol_result(
    protocol: Mapping[str, Any],
    fold_locks: Mapping[str, Mapping[str, Any]],
    transfer_plan: Mapping[str, Any],
    result_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate all locked factual and interventional Warp result records."""

    validate_rest_geometry_transfer_plan(protocol, fold_locks, transfer_plan)
    plan_records = _plan_records_by_id(transfer_plan)
    records_by_id = {}
    for result_record in result_records:
        record_id = result_record.get("record_id")
        if record_id not in plan_records or record_id in records_by_id:
            raise ValueError("protocol result contains an unexpected record")
        validate_rest_geometry_transfer_result_record(
            plan_records[record_id], result_record
        )
        records_by_id[record_id] = deepcopy(dict(result_record))
    if set(records_by_id) != set(plan_records):
        raise ValueError("protocol result does not cover every transfer record")
    ordered_records = [records_by_id[record_id] for record_id in sorted(records_by_id)]
    graph_digests = {
        record["canonical_material_graph_sha256"] for record in ordered_records
    }
    if len(graph_digests) != 1:
        raise ValueError("same-object records do not share one canonical graph")
    aggregate = {}
    for track in (
        "factual_continuation",
        "same_grasp_intervention_prediction",
        "new_contact_intervention_prediction",
    ):
        track_ids = {
            record["record_id"] for record in transfer_plan[track]
        }
        aggregate[track] = _aggregate_transfer_records(
            [record for record in ordered_records if record["record_id"] in track_ids]
        )
    result = {
        "schema_version": PROTOCOL_RESULT_SCHEMA_VERSION,
        "artifact_kind": "rest_geometry_same_object_protocol_result",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "transfer_plan_sha256": transfer_plan["plan_sha256"],
        "fold_lock_sha256": transfer_plan["fold_lock_sha256"],
        "canonical_material_graph_sha256": next(iter(graph_digests)),
        "information_boundary": dict(_RESULT_BOUNDARY),
        "record_count": len(ordered_records),
        "records": ordered_records,
        "aggregate": aggregate,
    }
    result["result_sha256"] = protocol_result_sha256(result)
    validate_rest_geometry_protocol_result(
        protocol, fold_locks, transfer_plan, result
    )
    return result


def validate_rest_geometry_protocol_result(
    protocol: Mapping[str, Any],
    fold_locks: Mapping[str, Mapping[str, Any]],
    transfer_plan: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute all result records and aggregate metrics from locked inputs."""

    validate_rest_geometry_transfer_plan(protocol, fold_locks, transfer_plan)
    if result.get("schema_version") != PROTOCOL_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported same-object result schema")
    if result.get("artifact_kind") != "rest_geometry_same_object_protocol_result":
        raise ValueError("unexpected same-object result kind")
    if result.get("protocol_id") != protocol["protocol_id"] or result.get(
        "protocol_design_sha256"
    ) != protocol["design_sha256"]:
        raise ValueError("same-object result protocol mismatch")
    if result.get("transfer_plan_sha256") != transfer_plan["plan_sha256"]:
        raise ValueError("same-object result transfer plan changed")
    if result.get("fold_lock_sha256") != transfer_plan["fold_lock_sha256"]:
        raise ValueError("same-object result fold locks changed")
    if not _is_sha256(result.get("canonical_material_graph_sha256")):
        raise ValueError("same-object result canonical graph digest is invalid")
    if result.get("information_boundary") != _RESULT_BOUNDARY:
        raise ValueError("same-object result crossed the information boundary")
    plan_records = _plan_records_by_id(transfer_plan)
    records = list(result.get("records", []))
    if result.get("record_count") != 66 or len(records) != 66:
        raise ValueError("same-object result must contain all 66 records")
    records_by_id = {}
    for record in records:
        record_id = record.get("record_id")
        if record_id not in plan_records or record_id in records_by_id:
            raise ValueError("same-object result record is unexpected")
        validate_rest_geometry_transfer_result_record(plan_records[record_id], record)
        records_by_id[record_id] = record
    if set(records_by_id) != set(plan_records):
        raise ValueError("same-object result records are incomplete")
    graph_digests = {
        record["canonical_material_graph_sha256"] for record in records
    }
    if graph_digests != {result["canonical_material_graph_sha256"]}:
        raise ValueError("same-object result mixed canonical material graphs")
    expected_aggregate = {}
    for track in (
        "factual_continuation",
        "same_grasp_intervention_prediction",
        "new_contact_intervention_prediction",
    ):
        track_ids = {
            record["record_id"] for record in transfer_plan[track]
        }
        expected_aggregate[track] = _aggregate_transfer_records(
            [records_by_id[record_id] for record_id in sorted(track_ids)]
        )
    if result.get("aggregate") != expected_aggregate:
        raise ValueError("same-object result aggregate does not match its records")
    if result.get("result_sha256") != protocol_result_sha256(result):
        raise ValueError("same-object result SHA-256 mismatch")
    return {
        "record_count": 66,
        "factual_count": 36,
        "same_grasp_count": 18,
        "new_contact_count": 12,
        "passed": True,
    }


def validate_rest_geometry_transfer_plan(
    protocol: Mapping[str, Any],
    fold_locks: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact transfer records and reject target-response leakage."""

    validate_protocol(protocol)
    expected_fold_ids = {
        fold["fold_id"]
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    }
    if set(fold_locks) != expected_fold_ids:
        raise ValueError("rest-geometry transfer plan fold locks are incomplete")
    for fold_id, lock in fold_locks.items():
        validate_rest_geometry_fold_lock(protocol, lock)
        if lock["fold_id"] != fold_id:
            raise ValueError("rest-geometry transfer plan fold lock is misindexed")
    if plan.get("schema_version") != TRANSFER_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported rest-geometry transfer plan schema")
    if plan.get("artifact_kind") != "rest_geometry_transfer_plan":
        raise ValueError("unexpected rest-geometry transfer plan kind")
    if plan.get("protocol_id") != protocol["protocol_id"] or plan.get(
        "protocol_design_sha256"
    ) != protocol["design_sha256"]:
        raise ValueError("rest-geometry transfer plan protocol mismatch")
    if plan.get("plan_sha256") != transfer_plan_sha256(plan):
        raise ValueError("rest-geometry transfer plan SHA-256 mismatch")
    expected_lock_digests = {
        fold_id: fold_locks[fold_id]["lock_sha256"]
        for fold_id in sorted(fold_locks)
    }
    if plan.get("fold_lock_sha256") != expected_lock_digests:
        raise ValueError("rest-geometry transfer plan changed its fold locks")
    if plan.get("transfer_contract") != {
        "persistent_components": [
            "frame_correction",
            "canonical_object_rest_lengths",
            "canonical_nonrigid_material_field",
        ],
        "target_specific_components": [
            "initial_dynamic_state",
            "controller_trajectory",
            "controller_attachment_springs",
        ],
        "same_grasp_attachment_policy": "reuse_registered_grasp",
        "new_contact_attachment_policy": "rebuild_on_registered_target_contact",
        "target_outcome_may_not_change_source_correction": True,
    }:
        raise ValueError("rest-geometry transfer contract changed")

    factual_expected = {
        f"factual::{item['execution_id']}": (
            item["execution_id"],
            item["execution_id"],
            "target_pre_holdout_only",
            True,
            "factual_same_execution",
        )
        for item in protocol["splits"]["factual_continuation"]
    }
    same_grasp_expected = {
        f"same-grasp::{item['source_execution_id']}->{item['target_execution_id']}": (
            item["source_execution_id"],
            item["target_execution_id"],
            "source_pre_holdout_only",
            False,
            "same_grasp",
        )
        for item in protocol["splits"]["same_grasp_intervention_prediction"]
    }
    new_contact_expected = {
        f"new-contact::{item['source_execution_id']}->{item['target_execution_id']}": (
            item["source_execution_id"],
            item["target_execution_id"],
            "source_pre_holdout_only",
            False,
            "new_contact",
        )
        for item in protocol["splits"]["new_contact_intervention_prediction"]
    }
    expected_by_track = {
        "factual_continuation": factual_expected,
        "same_grasp_intervention_prediction": same_grasp_expected,
        "new_contact_intervention_prediction": new_contact_expected,
    }
    seen = set()
    for track, expected_records in expected_by_track.items():
        records = list(plan.get(track, []))
        if len(records) != len(expected_records):
            raise ValueError(f"rest-geometry transfer plan changed {track} count")
        for record in records:
            record_id = record.get("record_id")
            if record_id not in expected_records or record_id in seen:
                raise ValueError("rest-geometry transfer record id is invalid")
            seen.add(record_id)
            expected = expected_records[record_id]
            actual = (
                record.get("source_execution_id"),
                record.get("target_execution_id"),
                record.get("correction_evidence_policy"),
                record.get("target_response_prefix_allowed"),
                record.get("contact_policy"),
            )
            if actual != expected:
                raise ValueError("rest-geometry transfer source/target policy changed")
            if record.get("evaluation_track") != track:
                raise ValueError("rest-geometry transfer record is in the wrong track")
            fold_id = record.get("fold_id")
            if fold_id not in fold_locks:
                raise ValueError("rest-geometry transfer record has no fold lock")
            lock = fold_locks[fold_id]
            if record.get("target_execution_id") not in lock["target_execution_ids"]:
                raise ValueError("rest-geometry transfer target is in the wrong fold")
            if record.get("fold_lock_sha256") != lock["lock_sha256"]:
                raise ValueError("rest-geometry transfer fold lock digest changed")
            if record.get("selected_candidate_id") != lock["selected_candidate_id"]:
                raise ValueError("rest-geometry transfer selected candidate changed")
            if record.get("canonical_material_graph_required") is not True:
                raise ValueError("rest-geometry transfer dropped canonical identity")
    return {
        "record_count": len(seen),
        "factual_count": 36,
        "same_grasp_count": 18,
        "new_contact_count": 12,
        "passed": True,
    }


def write_rest_geometry_cross_action_selection(
    protocol_path: str | Path,
    evidence_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Load 36 evidence files, select 12 locks, and write the transfer plan."""

    protocol = load_protocol(protocol_path)
    evidence_root = Path(evidence_dir)
    evidence_by_execution = {
        execution["execution_id"]: json.loads(
            (evidence_root / f"{execution['execution_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        for execution in protocol["executions"]
    }
    locks = select_all_rest_geometry_fold_locks(protocol, evidence_by_execution)
    plan = build_rest_geometry_transfer_plan(protocol, locks)
    output = Path(output_dir)
    locks_root = output / "fold_locks"
    locks_root.mkdir(parents=True, exist_ok=True)
    for fold_id, lock in locks.items():
        (locks_root / f"{fold_id}.json").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    plan_path = output / "rest_geometry_transfer_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "fold_lock_count": len(locks),
        "transfer_record_count": 66,
        "transfer_plan_sha256": plan["plan_sha256"],
        "output_dir": str(output.resolve()),
    }


def write_rest_geometry_protocol_result(
    protocol_path: str | Path,
    selection_dir: str | Path,
    result_record_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Aggregate 66 Warp records under the previously written fold locks."""

    protocol = load_protocol(protocol_path)
    selection_root = Path(selection_dir)
    locks = {
        fold["fold_id"]: json.loads(
            (
                selection_root
                / "fold_locks"
                / f"{fold['fold_id']}.json"
            ).read_text(encoding="utf-8")
        )
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    }
    transfer_plan = json.loads(
        (selection_root / "rest_geometry_transfer_plan.json").read_text(
            encoding="utf-8"
        )
    )
    records = []
    for path in sorted(Path(result_record_dir).glob("*.json")):
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate.get("artifact_kind") == "rest_geometry_transfer_result_record":
            records.append(candidate)
    result = build_rest_geometry_protocol_result(
        protocol,
        locks,
        transfer_plan,
        records,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "record_count": result["record_count"],
        "result_sha256": result["result_sha256"],
        "output": str(output.resolve()),
    }
