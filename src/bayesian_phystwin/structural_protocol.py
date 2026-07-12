"""Pre-acquisition amendment for hierarchical structural PhysTwin calibration."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from causal4d.real_protocol import load_protocol, validate_protocol


STRUCTURAL_AMENDMENT_SCHEMA_VERSION = 1
STRUCTURAL_AMENDMENT_ID = "hierarchical-structural-calibration-v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _locked_action_design(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "contact_regions": deepcopy(protocol["contact_regions"]),
        "command_profiles": deepcopy(protocol["command_profiles"]),
        "realization_conditions": deepcopy(protocol["realization_conditions"]),
        "sessions": deepcopy(protocol["sessions"]),
        "executions": deepcopy(protocol["executions"]),
        "splits": deepcopy(protocol["splits"]),
    }


def locked_action_design_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash every action, outcome, session, and split that an amendment may not change."""

    validate_protocol(protocol)
    return hashlib.sha256(_canonical_bytes(_locked_action_design(protocol))).hexdigest()


def structural_amendment_sha256(amendment: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(amendment))
    payload.pop("amendment_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_structural_protocol_amendment(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Add measurements and gates without changing the locked experiment design."""

    validate_protocol(protocol)
    amendment: dict[str, Any] = {
        "schema_version": STRUCTURAL_AMENDMENT_SCHEMA_VERSION,
        "artifact_kind": "structural_protocol_amendment",
        "amendment_id": STRUCTURAL_AMENDMENT_ID,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "locked_action_design_sha256": locked_action_design_sha256(protocol),
        "status": "locked_before_first_confirmatory_acquisition",
        "design_change": {
            "action_profiles_changed": False,
            "realization_conditions_changed": False,
            "execution_order_changed": False,
            "analysis_splits_changed": False,
            "outcomes_changed": False,
            "measurement_contract_only": True,
        },
        "structural_model_lock": {
            "model": "hierarchical_graph_regularized_structural_calibration",
            "object_persistent_variables": [
                "rest_geometry_coefficients",
                "stable_gravity_calibration",
                "stable_support_parameters",
            ],
            "session_variables": [
                "frame_rotation",
                "frame_translation",
                "settled_state_coefficients",
            ],
            "intervention_variables_unchanged": ["phi", "kappa"],
            "rank_candidates": [4, 8, 16],
            "first_stage_estimator": "MAP",
            "coefficient_covariance_stage": "deferred_until_mean_transfer_gate_passes",
            "rank_selection_evidence": "locked source-session O-minus only",
            "target_outcomes_may_select_rank_or_priors": False,
            "corrected_rest_lengths": "recomputed_from_corrected_embedded_geometry",
            "observed_pre_action_geometry_may_be_used_as_rest_shape": False,
            "equilibrium_requirement": (
                "simulate gravity/support equilibrium from corrected material rest "
                "geometry before adding session settled-state correction"
            ),
        },
        "measurement_amendment": {
            "minimum_static_o_minus_seconds_after_settling": 3.0,
            "minimum_static_o_minus_rgbd_frames": 90,
            "post_reset_static_scan_before_every_action": True,
            "required_session_artifacts": [
                "static_o_minus_rgbd_manifest",
                "support_plane_and_contact_geometry",
                "camera_world_calibration",
                "gravity_direction_calibration",
                "reset_procedure",
                "session_initial_state",
            ],
            "required_execution_sidecars": [
                "post_reset_static_scan",
                "support_contact_state",
                "gravity_direction_snapshot",
                "reset_realization",
            ],
            "existing_required_streams_reaffirmed": [
                "commanded_control_trajectory",
                "measured_end_effector_trajectory",
                "measured_gripper_state",
                "synchronized_rgbd_manifest",
            ],
            "force_or_gripper_state_when_available": True,
            "all_artifacts_require_path_sha256_and_bytes": True,
        },
        "slip_pilot_addendum": {
            "must_answer_before_confirmatory_collection": [
                "pre_action_settled_state_repeatability_across_resets",
                "object_persistent_rest_coefficient_stability_across_sessions",
                "fraction_of_reset_variability_explained_by_session_frame_and_state",
            ],
            "minimum_reset_repetitions": 5,
            "maximum_pre_action_settled_state_rmse_m": 0.003,
            "maximum_leave_one_session_rest_displacement_rmse_m": 0.002,
            "minimum_reset_variance_fraction_explained_by_session_terms": 0.70,
        },
        "acceptance_gates": {
            "cross_action_accuracy": {
                "baseline": "graph_persistence_readout",
                "metric": "paired held-out future track error",
                "maximum_mean_error_ratio": 0.95,
                "cluster_bootstrap_95_percent_upper_ratio": 1.0,
            },
            "late_horizon": {
                "nominal_coverage": 0.90,
                "minimum_coverage": 0.75,
                "minimum_absolute_improvement_over_current": 0.15,
                "current_diagnostic_coverage": 0.5311,
            },
            "far_graph": {
                "minimum_track_error_improvement_fraction": 0.05,
                "maximum_near_contact_degradation_fraction": 0.02,
            },
            "transfer": {
                "persistent_coefficients_fit_on_source_sessions_only": True,
                "maximum_unseen_action_contact_error_ratio": 0.95,
            },
            "stability": {
                "maximum_leave_one_session_rest_displacement_rmse_m": 0.002,
                "minimum_reset_variance_fraction_assigned_to_session_terms": 0.70,
            },
            "plausibility": {
                "maximum_absolute_edge_strain": 0.10,
                "maximum_99th_percentile_absolute_edge_strain": 0.05,
                "maximum_inverted_validity_cells": 0,
                "maximum_introduced_surface_self_intersections": 0,
                "support_anchors_must_remain_fixed": True,
            },
            "calibration": {
                "minimum_independent_executions_before_covariance_fit": 12,
                "minimum_contacts_before_covariance_fit": 3,
                "minimum_command_profiles_before_covariance_fit": 3,
                "aggregate_90_percent_coverage_interval": [0.85, 0.95],
                "minimum_worst_group_coverage": 0.80,
            },
        },
        "required_post_structural_audit": [
            "posterior",
            "current_bank_oracle",
            "expanded_bank_oracle",
            "structural_correction_ceiling",
            "variance_decomposition",
            "coverage_by_horizon",
            "coverage_by_graph_region",
        ],
        "frozen_directions": [
            "larger_intervention_bank",
            "additional_theta_particle_campaign",
            "transferred_global_affine_calibration",
            "MolmoMotion_tuning",
            "robot_execution",
        ],
    }
    amendment["amendment_sha256"] = structural_amendment_sha256(amendment)
    validate_structural_protocol_amendment(protocol, amendment)
    return amendment


def validate_structural_protocol_amendment(
    protocol: Mapping[str, Any], amendment: Mapping[str, Any]
) -> dict[str, Any]:
    validate_protocol(protocol)
    if amendment.get("schema_version") != STRUCTURAL_AMENDMENT_SCHEMA_VERSION:
        raise ValueError("unsupported structural protocol amendment schema")
    if amendment.get("artifact_kind") != "structural_protocol_amendment":
        raise ValueError("wrong structural protocol amendment kind")
    if amendment.get("amendment_id") != STRUCTURAL_AMENDMENT_ID:
        raise ValueError("unknown structural protocol amendment")
    if (
        amendment.get("protocol_id") != protocol["protocol_id"]
        or amendment.get("protocol_design_sha256") != protocol["design_sha256"]
    ):
        raise ValueError("structural amendment protocol identity mismatch")
    if amendment.get("locked_action_design_sha256") != locked_action_design_sha256(
        protocol
    ):
        raise ValueError("structural amendment changed the locked action design")
    if amendment.get("amendment_sha256") != structural_amendment_sha256(amendment):
        raise ValueError("structural amendment digest mismatch")
    design_change = amendment.get("design_change", {})
    required_change = {
        "action_profiles_changed": False,
        "realization_conditions_changed": False,
        "execution_order_changed": False,
        "analysis_splits_changed": False,
        "outcomes_changed": False,
        "measurement_contract_only": True,
    }
    if design_change != required_change:
        raise ValueError("structural amendment is not measurement-only")
    model = amendment.get("structural_model_lock", {})
    if model.get("rank_candidates") != [4, 8, 16]:
        raise ValueError("structural rank candidates changed")
    if model.get("first_stage_estimator") != "MAP":
        raise ValueError("structural first-stage estimator must remain MAP")
    if model.get("target_outcomes_may_select_rank_or_priors") is not False:
        raise ValueError("target outcomes may not select structural settings")
    measurement = amendment.get("measurement_amendment", {})
    if measurement.get("minimum_static_o_minus_seconds_after_settling", 0.0) < 3.0:
        raise ValueError("structural amendment needs at least three static seconds")
    gates = amendment.get("acceptance_gates", {})
    if gates.get("cross_action_accuracy", {}).get("baseline") != (
        "graph_persistence_readout"
    ):
        raise ValueError("structural accuracy gate must compare graph persistence")
    if gates.get("calibration", {}).get(
        "minimum_independent_executions_before_covariance_fit"
    ) < 12:
        raise ValueError("structural covariance gate has too few executions")
    return {
        "amendment_id": STRUCTURAL_AMENDMENT_ID,
        "amendment_sha256": amendment["amendment_sha256"],
        "locked_action_design_sha256": amendment["locked_action_design_sha256"],
        "passed": True,
    }


def _artifact_template(names: list[str]) -> dict[str, Any]:
    return {
        name: {"path": None, "sha256": None, "bytes": None} for name in names
    }


def structural_session_template(
    protocol: Mapping[str, Any], amendment: Mapping[str, Any], session_id: str
) -> dict[str, Any]:
    validate_structural_protocol_amendment(protocol, amendment)
    valid_ids = {value["session_id"] for value in protocol["sessions"]}
    if session_id not in valid_ids:
        raise KeyError(session_id)
    names = amendment["measurement_amendment"]["required_session_artifacts"]
    return {
        "schema_version": 1,
        "artifact_kind": "structural_session_measurements",
        "amendment_sha256": amendment["amendment_sha256"],
        "session_id": session_id,
        "acquisition_status": "template",
        "static_o_minus": {
            "settled_duration_s": None,
            "rgbd_frame_count": None,
            "settling_criterion": None,
        },
        "artifacts": _artifact_template(names),
        "derived_diagnostics": {
            "pre_action_settled_state_rmse_m": None,
            "frame_rotation_deg": None,
            "frame_translation_norm_m": None,
            "rest_coefficient_artifact_id": None,
            "session_state_coefficient_artifact_id": None,
        },
    }


def structural_execution_template(
    protocol: Mapping[str, Any], amendment: Mapping[str, Any], execution_id: str
) -> dict[str, Any]:
    validate_structural_protocol_amendment(protocol, amendment)
    execution_by_id = {
        value["execution_id"]: value for value in protocol["executions"]
    }
    if execution_id not in execution_by_id:
        raise KeyError(execution_id)
    execution = execution_by_id[execution_id]
    names = amendment["measurement_amendment"]["required_execution_sidecars"]
    return {
        "schema_version": 1,
        "artifact_kind": "structural_execution_measurements",
        "amendment_sha256": amendment["amendment_sha256"],
        "execution_id": execution_id,
        "session_id": execution["session_id"],
        "acquisition_status": "template",
        "artifacts": _artifact_template(names),
        "reset": {
            "procedure_id": None,
            "post_reset_static_scan_passed": None,
            "settled_state_rmse_m": None,
        },
    }


def structural_slip_pilot_addendum_template(
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "structural_slip_pilot_addendum",
        "amendment_sha256": amendment["amendment_sha256"],
        "pilot_execution_ids": [],
        "pre_action_settled_state_rmse_m": None,
        "leave_one_session_rest_displacement_rmse_m": None,
        "reset_variance_fraction_explained_by_session_terms": None,
        "passed": None,
        "decided_before_confirmatory_collection": None,
    }


def scaffold_structural_protocol_amendment(
    protocol_path: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Install non-overwriting measurement sidecars before any execution exists."""

    protocol = load_protocol(protocol_path)
    root = Path(dataset_root)
    completed = list(root.glob("executions/*/manifest.json"))
    if completed:
        raise RuntimeError("cannot add the structural amendment after acquisition began")
    amendment = build_structural_protocol_amendment(protocol)
    amendment_path = root / "structural_protocol_amendment.json"
    if amendment_path.exists():
        existing = json.loads(amendment_path.read_text(encoding="utf-8"))
        if existing != amendment:
            raise FileExistsError("refusing to replace a different structural amendment")
    else:
        amendment_path.write_text(
            json.dumps(amendment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    session_count = 0
    for session in protocol["sessions"]:
        path = (
            root
            / "structural_sessions"
            / session["session_id"]
            / "manifest.template.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                json.dumps(
                    structural_session_template(
                        protocol, amendment, session["session_id"]
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        session_count += 1
    execution_count = 0
    for execution in protocol["executions"]:
        path = (
            root
            / "executions"
            / execution["execution_id"]
            / "structural_measurements.template.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                json.dumps(
                    structural_execution_template(
                        protocol, amendment, execution["execution_id"]
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        execution_count += 1
    slip_path = root / "structural_slip_pilot_addendum.template.json"
    if not slip_path.exists():
        slip_path.write_text(
            json.dumps(
                structural_slip_pilot_addendum_template(amendment),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "amendment_path": str(amendment_path.resolve()),
        "amendment_sha256": amendment["amendment_sha256"],
        "session_template_count": session_count,
        "execution_template_count": execution_count,
        "completed_execution_count_at_amendment": 0,
        "locked_action_design_sha256": amendment["locked_action_design_sha256"],
    }


def audit_structural_protocol_readiness(
    protocol_path: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    root = Path(dataset_root)
    amendment_path = root / "structural_protocol_amendment.json"
    if not amendment_path.is_file():
        return {
            "ready": False,
            "status": "missing_structural_protocol_amendment",
        }
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    validation = validate_structural_protocol_amendment(protocol, amendment)
    session_templates = list(
        root.glob("structural_sessions/*/manifest.template.json")
    )
    execution_templates = list(
        root.glob("executions/*/structural_measurements.template.json")
    )
    completed = list(root.glob("executions/*/manifest.json"))
    expected_sessions = len(protocol["sessions"])
    expected_executions = len(protocol["executions"])
    scaffold_complete = (
        len(session_templates) == expected_sessions
        and len(execution_templates) == expected_executions
        and (root / "structural_slip_pilot_addendum.template.json").is_file()
    )
    return {
        "ready": scaffold_complete and not completed,
        "status": (
            "structural_measurements_locked_awaiting_acquisition"
            if scaffold_complete and not completed
            else "structural_scaffold_incomplete_or_acquisition_started"
        ),
        "validation": validation,
        "expected_session_count": expected_sessions,
        "session_template_count": len(session_templates),
        "expected_execution_count": expected_executions,
        "execution_template_count": len(execution_templates),
        "completed_execution_count": len(completed),
    }
