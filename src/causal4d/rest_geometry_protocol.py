"""Locked same-object protocol boundary for rest-geometry validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from causal4d.real_protocol import (
    load_protocol,
    validate_dataset,
    validate_protocol,
)
from causal4d.rest_geometry_transfer import load_canonical_material_graph


REST_GEOMETRY_ANALYSIS_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_rest_geometry_registration(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    canonical_graph_path: str | Path,
) -> dict[str, Any]:
    """Register the immutable material graph before confirmatory collection."""

    validate_protocol(protocol)
    root = Path(dataset_root).resolve()
    graph_path = Path(canonical_graph_path).resolve()
    try:
        relative_path = graph_path.relative_to(root)
    except ValueError as error:
        raise ValueError("canonical graph must live inside the dataset root") from error
    graph = load_canonical_material_graph(graph_path)
    return {
        "schema_version": 1,
        "artifact_kind": "rest_geometry_analysis_registration",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "registered_before_confirmatory_collection": True,
        "material_node_identity_fixed_across_executions": True,
        "controller_attachments_execution_specific": True,
        "canonical_material_graph": {
            "path": relative_path.as_posix(),
            "file_sha256": _sha256_file(graph_path),
            "canonical_material_graph_sha256": graph.sha256,
            "object_vertex_count": len(graph.vertices),
            "object_spring_count": len(graph.springs),
        },
    }


def validate_rest_geometry_registration(
    protocol: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
    verify_file: bool = False,
) -> dict[str, Any]:
    """Validate graph identity and the before-collection lock."""

    validate_protocol(protocol)
    if registration.get("schema_version") != 1 or registration.get(
        "artifact_kind"
    ) != "rest_geometry_analysis_registration":
        raise ValueError("unsupported rest-geometry registration")
    if registration.get("protocol_id") != protocol["protocol_id"] or registration.get(
        "protocol_design_sha256"
    ) != protocol["design_sha256"]:
        raise ValueError("rest-geometry registration protocol mismatch")
    required_flags = {
        "registered_before_confirmatory_collection": True,
        "material_node_identity_fixed_across_executions": True,
        "controller_attachments_execution_specific": True,
    }
    if any(registration.get(key) is not value for key, value in required_flags.items()):
        raise ValueError("rest-geometry registration changed its identity contract")
    descriptor = registration.get("canonical_material_graph", {})
    path_value = descriptor.get("path")
    if (
        not isinstance(path_value, str)
        or not path_value
        or Path(path_value).is_absolute()
        or ".." in Path(path_value).parts
    ):
        raise ValueError("rest-geometry canonical graph path is unsafe")
    if not _is_sha256(descriptor.get("file_sha256")) or not _is_sha256(
        descriptor.get("canonical_material_graph_sha256")
    ):
        raise ValueError("rest-geometry canonical graph digest is invalid")
    for field in ("object_vertex_count", "object_spring_count"):
        if not isinstance(descriptor.get(field), int) or descriptor[field] < 1:
            raise ValueError(f"rest-geometry canonical graph {field} is invalid")
    if verify_file:
        if dataset_root is None:
            raise ValueError("dataset_root is required to verify the canonical graph")
        graph_path = Path(dataset_root) / path_value
        if not graph_path.is_file() or _sha256_file(graph_path) != descriptor[
            "file_sha256"
        ]:
            raise ValueError("rest-geometry canonical graph file digest mismatch")
        graph = load_canonical_material_graph(graph_path)
        if graph.sha256 != descriptor["canonical_material_graph_sha256"]:
            raise ValueError("rest-geometry material graph digest mismatch")
        if len(graph.vertices) != descriptor["object_vertex_count"] or len(
            graph.springs
        ) != descriptor["object_spring_count"]:
            raise ValueError("rest-geometry canonical graph size changed")
    return {
        "canonical_material_graph_sha256": descriptor[
            "canonical_material_graph_sha256"
        ],
        "passed": True,
    }


def write_rest_geometry_registration(
    protocol_path: str | Path,
    dataset_root: str | Path,
    canonical_graph_path: str | Path,
) -> dict[str, Any]:
    """Write the graph registration into a not-yet-collected dataset root."""

    protocol = load_protocol(protocol_path)
    registration = build_rest_geometry_registration(
        protocol,
        dataset_root,
        canonical_graph_path,
    )
    output = Path(dataset_root) / "rest_geometry_registration.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != registration:
            raise FileExistsError("refusing to replace a different graph registration")
    else:
        output.write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    validate_rest_geometry_registration(
        protocol,
        registration,
        dataset_root=dataset_root,
        verify_file=True,
    )
    return {**registration, "registration_path": str(output.resolve())}


def build_rest_geometry_analysis_plan(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Map locked folds to leakage-safe rest-geometry fit and target roles."""

    validation = validate_protocol(protocol)
    folds = []
    for fold in protocol["splits"]["cross_action_contact_calibration_folds"]:
        folds.append(
            {
                "fold_id": fold["fold_id"],
                "hyperparameter_fit_execution_ids": list(
                    fold["fit_execution_ids"]
                ),
                "calibration_execution_ids": list(
                    fold["calibration_execution_ids"]
                ),
                "target_execution_ids": list(fold["target_execution_ids"]),
                "held_out_contact_region_id": fold[
                    "held_out_contact_region_id"
                ],
                "held_out_command_profile_id": fold[
                    "held_out_command_profile_id"
                ],
                "target_execution_inference": (
                    "factual continuation may use own pre-holdout observations; "
                    "same-grasp and new-contact transfer use only the paired "
                    "source execution, with shared hyperparameters already frozen"
                ),
                "shared_hyperparameters": [
                    "frame_mode",
                    "frame_scale",
                    "rest_geometry_scale",
                    "controller_rest_mode",
                    "graph_prior_strength",
                    "rest_length_ratio_bound",
                ],
            }
        )
    return {
        "schema_version": REST_GEOMETRY_ANALYSIS_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "status": "locked_before_target_evaluation",
        "method": "graph-regularized frame/rest-geometry PhysTwin injection",
        "canonical_material_graph_contract": {
            "one_object_graph_for_all_executions": True,
            "material_node_identity_fixed": True,
            "controller_attachment_springs_execution_specific": True,
            "registration_file": "rest_geometry_registration.json",
        },
        "shared_selection_rule": (
            "one_standard_error_on_equal_execution_mean_log_validation_error_ratio"
        ),
        "information_boundary": {
            "fit_actions_may_select_hyperparameters": True,
            "calibration_actions_may_fit_coverage_only": True,
            "target_actions_may_select_hyperparameters": False,
            "target_holdout_frames_may_infer_correction": False,
            "target_pre_holdout_frames_may_infer_execution_specific_field": True,
            "individual_counterfactual_ground_truth_claimed": False,
        },
        "required_method_ablations": [
            "released",
            "endpoint_restart",
            "output_frame_graph",
            "frame_state_original_rest",
            "graph_state_original_rest",
            "rest_geometry_only",
            "frame_rest_geometry",
            "frame_rest_geometry_reattached",
            "selected_frame_rest_geometry",
        ],
        "primary_metrics": [
            "future_chamfer_distance_m",
            "future_track_error_m",
        ],
        "reporting_groups": [
            "factual_continuation",
            "same_grasp_intervention_prediction",
            "new_contact_intervention_prediction",
            "forecast_horizon",
            "contact_region",
            "command_profile",
        ],
        "evaluation_tracks": {
            "factual_continuation": {
                "record_count": len(
                    protocol["splits"]["factual_continuation"]
                ),
                "correction_evidence": "target pre-holdout response allowed",
            },
            "same_grasp_intervention_prediction": {
                "record_count": len(
                    protocol["splits"]["same_grasp_intervention_prediction"]
                ),
                "correction_evidence": "paired source pre-holdout only",
                "target_response_prefix_allowed": False,
            },
            "new_contact_intervention_prediction": {
                "record_count": len(
                    protocol["splits"]["new_contact_intervention_prediction"]
                ),
                "correction_evidence": "paired source pre-holdout only",
                "target_response_prefix_allowed": False,
            },
        },
        "fold_count": len(folds),
        "execution_count": validation["executions"],
        "folds": folds,
    }


def audit_rest_geometry_dataset_readiness(
    dataset_root: str | Path,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Fail closed until every preregistered real execution is complete."""

    root = Path(dataset_root)
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        return {
            "schema_version": 1,
            "dataset_root": str(root.resolve()),
            "ready_for_confirmatory_analysis": False,
            "status": "missing_locked_protocol",
            "missing": ["protocol.json"],
        }
    protocol = load_protocol(protocol_path)
    plan = build_rest_geometry_analysis_plan(protocol)
    expected_ids = [
        execution["execution_id"] for execution in protocol["executions"]
    ]
    complete_ids = []
    template_ids = []
    missing_ids = []
    for execution_id in expected_ids:
        execution_root = root / "executions" / execution_id
        if (execution_root / "manifest.json").is_file():
            complete_ids.append(execution_id)
        elif (execution_root / "manifest.template.json").is_file():
            template_ids.append(execution_id)
        else:
            missing_ids.append(execution_id)
    required_top_level = (
        "acquisition_schedule.csv",
        "object_registration.json",
        "slip_pilot.json",
        "rest_geometry_registration.json",
    )
    missing_top_level = [
        name for name in required_top_level if not (root / name).is_file()
    ]
    rest_geometry_registration_validation = None
    registration_path = root / "rest_geometry_registration.json"
    if registration_path.is_file():
        rest_geometry_registration_validation = validate_rest_geometry_registration(
            protocol,
            json.loads(registration_path.read_text(encoding="utf-8")),
            dataset_root=root,
            verify_file=True,
        )
    ready = (
        len(complete_ids) == len(expected_ids)
        and not template_ids
        and not missing_ids
        and not missing_top_level
    )
    dataset_validation = None
    status = "awaiting_acquisition"
    if ready:
        dataset_validation = validate_dataset(
            protocol,
            root,
            verify_files=verify_files,
        )
        status = "ready_for_confirmatory_analysis"
    return {
        "schema_version": 1,
        "dataset_root": str(root.resolve()),
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "ready_for_confirmatory_analysis": ready,
        "status": status,
        "expected_execution_count": len(expected_ids),
        "complete_manifest_count": len(complete_ids),
        "template_manifest_count": len(template_ids),
        "missing_manifest_count": len(missing_ids),
        "missing_top_level_artifacts": missing_top_level,
        "template_execution_ids": sorted(template_ids),
        "missing_execution_ids": sorted(missing_ids),
        "analysis_plan": plan,
        "rest_geometry_registration_validation": (
            rest_geometry_registration_validation
        ),
        "dataset_validation": dataset_validation,
    }


def validate_rest_geometry_fold_result(
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a future fold result against the preregistered data boundary."""

    validate_protocol(protocol)
    if result.get("schema_version") != REST_GEOMETRY_ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported rest-geometry result schema")
    if result.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("rest-geometry result protocol mismatch")
    if result.get("protocol_design_sha256") != protocol["design_sha256"]:
        raise ValueError("rest-geometry result protocol digest mismatch")
    folds = {
        fold["fold_id"]: fold
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    }
    fold_id = result.get("fold_id")
    if fold_id not in folds:
        raise ValueError("rest-geometry result fold is not preregistered")
    fold = folds[fold_id]
    role_fields = (
        ("hyperparameter_fit_execution_ids", "fit_execution_ids"),
        ("calibration_execution_ids", "calibration_execution_ids"),
        ("target_execution_ids", "target_execution_ids"),
    )
    for result_field, fold_field in role_fields:
        if sorted(result.get(result_field, [])) != sorted(fold[fold_field]):
            raise ValueError(f"rest-geometry result changed {result_field}")
    boundary = result.get("information_boundary")
    required_boundary = {
        "target_holdout_frames_used_for_inference": False,
        "target_outcomes_used_for_hyperparameter_selection": False,
        "manual_gt_track_used_for_hyperparameter_selection": False,
        "individual_counterfactual_ground_truth_claimed": False,
    }
    if boundary != required_boundary:
        raise ValueError("rest-geometry result changed the information boundary")
    records = result.get("target_records", [])
    if sorted(record.get("execution_id") for record in records) != sorted(
        fold["target_execution_ids"]
    ):
        raise ValueError("rest-geometry target records are incomplete")
    for record in records:
        if record.get("correction_evidence") != "pre_holdout_only":
            raise ValueError("target correction used an invalid evidence window")
        if record.get("shared_hyperparameters_frozen_before_target") is not True:
            raise ValueError("target hyperparameters were not frozen")
    return {
        "fold_id": fold_id,
        "target_execution_count": len(records),
        "passed": True,
    }


def write_rest_geometry_protocol_artifacts(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Write the locked analysis plan and current acquisition readiness audit."""

    readiness = audit_rest_geometry_dataset_readiness(
        dataset_root,
        verify_files=verify_files,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "rest_geometry_analysis_plan.json"
    readiness_path = output / "rest_geometry_dataset_readiness.json"
    plan_path.write_text(
        json.dumps(readiness.get("analysis_plan"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readiness_path.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **readiness,
        "analysis_plan_path": str(plan_path.resolve()),
        "readiness_path": str(readiness_path.resolve()),
    }
