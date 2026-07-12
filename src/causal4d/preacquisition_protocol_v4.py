"""Prospective v4 addendum for mechanism controls and state-mode predictions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from causal4d.mechanism_gate_controls import mechanism_gate_control_sha256
from causal4d.preacquisition_protocol import (
    load_preacquisition_amendment,
    validate_preacquisition_amendment,
)
from causal4d.preacquisition_protocol_v3 import (
    load_preacquisition_v3,
    validate_preacquisition_v3,
)
from causal4d.real_protocol import load_protocol, validate_protocol


PREACQUISITION_V4_SCHEMA_VERSION = 1
PREACQUISITION_V4_PLAN_ID = "causal4d-sloth-preacquisition-v4"
_CANONICAL_V4_SHA256 = (
    "0e167538a7824e5ec053031d8359d4e9b4ff89ad61a85666400a86c2a88ac42f"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def preacquisition_v4_sha256(amendment: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(amendment))
    payload.pop("amendment_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_gate_control_evidence(evidence: Mapping[str, Any]) -> None:
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_kind") == "MechanismGateControlEvidence",
        "unexpected mechanism-gate control artifact",
    )
    _require(
        evidence.get("result_sha256") == mechanism_gate_control_sha256(dict(evidence)),
        "mechanism-gate control digest is invalid",
    )
    config = evidence["config"]
    _require(
        config["simulation_count"] >= 512,
        "mechanism-gate controls need at least 512 panels",
    )
    _require(
        config["minimum_shrinkage_fraction"] == 0.10
        and config["minimum_positive_sessions"] == 8,
        "mechanism-gate controls did not test the frozen v3 threshold",
    )
    checks = evidence["acceptance_checks"]
    _require(
        checks["placebo_null_full_gate_upper_below_5_percent"] is True,
        "placebo false-positive control failed",
    )
    _require(
        checks["positive_control_full_gate_lower_above_80_percent"] is True,
        "positive-control power gate failed",
    )
    _require(
        checks["wrong_family_on_positive_upper_below_5_percent"] is True,
        "wrong-family specificity control failed",
    )
    _require(
        evidence["frozen_v3_gate_supported_in_controlled_benchmark"] is True,
        "controlled benchmark did not support the frozen v3 gate",
    )


def build_preacquisition_v4(
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v3: Mapping[str, Any],
    gate_control_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build v4 without changing the locked physical acquisition design."""

    validate_protocol(protocol)
    validate_preacquisition_amendment(v2, protocol)
    validate_preacquisition_v3(v3, protocol, v2)
    _validate_gate_control_evidence(gate_control_evidence)
    arms = gate_control_evidence["arms"]
    amendment: dict[str, Any] = {
        "schema_version": PREACQUISITION_V4_SCHEMA_VERSION,
        "plan_id": PREACQUISITION_V4_PLAN_ID,
        "status": "supersedes_v3_before_any_physical_execution",
        "supersedes": {
            "plan_id": v3["plan_id"],
            "amendment_sha256": v3["amendment_sha256"],
            "git_tag": "causal4d-preacquisition-v3",
            "physical_executions_completed_before_supersession": 0,
        },
        "base_protocol": deepcopy(v3["base_protocol"]),
        "unchanged_acquisition_design": deepcopy(v3["unchanged_acquisition_design"]),
        "unchanged_v3_analysis": {
            "source_panel_crossfit": deepcopy(v3["source_panel_crossfit"]),
            "signature_eligibility_gates": deepcopy(v3["signature_eligibility_gates"]),
            "calibration_resolution": deepcopy(v3["calibration_resolution"]),
            "source_panel_role": deepcopy(v3["source_panel_role"]),
        },
        "mechanism_gate_control_lock": {
            "evidence_artifact": (
                "runs/causal4d_preacquisition_v4/mechanism_gate_controls.json"
            ),
            "evidence_sha256": gate_control_evidence["result_sha256"],
            "simulation_count_per_arm": gate_control_evidence["config"][
                "simulation_count"
            ],
            "frozen_threshold": {
                "minimum_geometric_mean_shrinkage_fraction": 0.10,
                "minimum_sessions_with_positive_shrinkage": 8,
                "session_count": 12,
            },
            "placebo": {
                "definition": gate_control_evidence["design"]["placebo"],
                "full_gate_pass_count": arms["placebo_null"][
                    "full_eligibility_pass_count"
                ],
                "full_gate_pass_rate": arms["placebo_null"][
                    "full_eligibility_pass_rate"
                ],
                "wilson_95": arms["placebo_null"]["full_eligibility_wilson_95"],
            },
            "positive_control": {
                "definition": gate_control_evidence["design"]["positive_control"],
                "full_gate_pass_count": arms["positive_control"][
                    "full_eligibility_pass_count"
                ],
                "full_gate_pass_rate": arms["positive_control"][
                    "full_eligibility_pass_rate"
                ],
                "wilson_95": arms["positive_control"]["full_eligibility_wilson_95"],
            },
            "threshold_changed_after_controls": False,
            "claim_boundary": gate_control_evidence["claim_boundary"],
            "real_world_false_positive_rate_claimed": False,
        },
        "state_propagation_interpretation_lock": {
            "empirical_description": "delta_x_T approximately Phi_a(T,t_p) delta_x_t_p",
            "linearization_boundary": (
                "Phi_a is an empirical secant or local linearization at the injected "
                "magnitude, not a globally valid state-transition matrix. Contact-mode "
                "switching can make propagation nonsmooth."
            ),
            "released_case_source": {
                "git_tag": "phystwin-discrepancy-localization-v1",
                "aggregate_artifact": (
                    "data/paper_evidence/phystwin_discrepancy_localization_v1/"
                    "state_correction_modes_aggregate.json"
                ),
                "aggregate_file_sha256": (
                    "97eafb9a64a51faac4fd92d8aff9fffb89b28143b4e5f1e91189ff6572b91df7"
                ),
            },
            "single_lift": {
                "readout_cd_gain_captured_fraction": 0.8267,
                "readout_track_gain_captured_fraction": 0.8738,
                "dominant_retained_mode": 0,
                "state_error_interpretation": "plausible_for_this_interaction",
            },
            "double_lift": {
                "final_outside_injected_rank4_fraction": 0.7356,
                "interpretation": "contraction_and_basis_leakage",
            },
            "double_stretch": {
                "near_middle_far_aligned_retention": [0.0555, -0.3638, 0.4798],
                "interpretation": (
                    "spatial redistribution or cancellation; contact switching remains "
                    "an alternative to a smooth rotation interpretation"
                ),
            },
            "cross_case_claim": (
                "A prefix state error may matter interaction by interaction, but one "
                "generic state reset is not a transferable persistent-discrepancy model."
            ),
            "attachment_correlations_inferential": False,
        },
        "prospective_mode0_reset_crosscheck": {
            "status": "preregistered_before_slip_reset_pilot",
            "hypothesis": (
                "If the single-lift mode-0 component is primarily initial pose, reset, "
                "or registration error, independently measured reset mode-0 variation "
                "should be commensurate with the released inferred correction."
            ),
            "released_reference": {
                "mode": 0,
                "initial_mode_energy_m2": 1.3009828632847338,
                "object_node_count": 6895,
                "per_node_vector_rms_m": 0.013736264750447176,
            },
            "pilot_statistic": (
                "95th percentile across fresh-reset sessions of per-node vector RMS in "
                "mode 0, expressed in the locked world frame before any per-reset best-fit "
                "alignment, plus the preregistered 95 percent registration uncertainty."
            ),
            "secondary_decomposition": [
                "locked-frame rigid translation",
                "best-fit SE3 registration component",
                "post-SE3 low-rank nonrigid component",
            ],
            "decision_rule": {
                "scale_compatible": (
                    "released mode-0 RMS is no more than twice the pilot statistic"
                ),
                "reset_scale_explanation_weakened": (
                    "released mode-0 RMS exceeds twice the pilot statistic"
                ),
                "compatibility_confirms_cause": False,
            },
            "action_outcomes_may_revise_mapping": False,
        },
        "mechanism_ladder_addition": {
            "name": "action_dependent_propagated_state_correction",
            "role": "source_panel_candidate_not_released_case_model_selection",
            "definition": (
                "Infer a prefix state update, propagate its basis through the frozen "
                "action/contact-conditioned simulator, and cross-fit every mechanism "
                "parameter on 8 source sessions before evaluation on 4 held-out sessions."
            ),
            "same_v3_shrinkage_and_prediction_gates_required": True,
            "implementation_may_use_target_sessions": False,
            "promoted_before_source_panel": False,
        },
        "contact_registration_contract": {
            "schema_version": 3,
            "artifact_kind": "PhysicalContactRegistration",
            "weighted_node_patch_required": True,
            "selected_and_rejected_candidates_required": True,
            "minimum_rejected_candidates_per_region": 1,
            "minimum_independent_reviews": 2,
            "minimum_calibrated_camera_views": 3,
            "se3_covariance_and_closure_required": True,
            "support_geometry_required": True,
            "approval_required_before_slip_pilot": True,
            "target_outcomes_may_revise_registration": False,
        },
        "collection_sequence": deepcopy(v3["collection_sequence"]),
        "collection_gate": {
            **deepcopy(v3["collection_gate"]),
            "v4_analysis_code_frozen": False,
            "first_confirmatory_execution_allowed": False,
        },
    }
    amendment["amendment_sha256"] = preacquisition_v4_sha256(amendment)
    validate_preacquisition_v4(
        amendment,
        protocol,
        v2,
        v3,
        gate_control_evidence,
    )
    return amendment


def validate_preacquisition_v4(
    amendment: Mapping[str, Any],
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v3: Mapping[str, Any],
    gate_control_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate v4 and its immutable evidence chain."""

    validate_protocol(protocol)
    validate_preacquisition_amendment(v2, protocol)
    validate_preacquisition_v3(v3, protocol, v2)
    _validate_gate_control_evidence(gate_control_evidence)
    _require(amendment.get("schema_version") == 1, "unsupported v4 schema")
    _require(amendment.get("plan_id") == PREACQUISITION_V4_PLAN_ID, "unexpected v4 id")
    _require(
        amendment.get("amendment_sha256") == preacquisition_v4_sha256(amendment),
        "v4 SHA-256 does not match its contents",
    )
    if _CANONICAL_V4_SHA256:
        _require(
            amendment["amendment_sha256"] == _CANONICAL_V4_SHA256,
            "v4 differs from the locked canonical design",
        )
    _require(
        amendment["supersedes"]["amendment_sha256"] == v3["amendment_sha256"],
        "v4 does not supersede the locked v3 artifact",
    )
    _require(
        amendment["base_protocol"] == v3["base_protocol"]
        and amendment["unchanged_acquisition_design"]
        == v3["unchanged_acquisition_design"],
        "v4 changed the physical acquisition design",
    )
    control = amendment["mechanism_gate_control_lock"]
    _require(
        control["evidence_sha256"] == gate_control_evidence["result_sha256"],
        "v4 references the wrong gate-control evidence",
    )
    _require(
        control["threshold_changed_after_controls"] is False
        and control["frozen_threshold"]["minimum_geometric_mean_shrinkage_fraction"]
        == v3["heldout_mechanism_eligibility"][
            "minimum_geometric_mean_shrinkage_fraction"
        ]
        and control["frozen_threshold"]["minimum_sessions_with_positive_shrinkage"]
        == v3["heldout_mechanism_eligibility"][
            "minimum_sessions_with_positive_shrinkage"
        ],
        "v4 changed the controlled v3 mechanism gate",
    )
    _require(
        amendment["prospective_mode0_reset_crosscheck"]["decision_rule"][
            "compatibility_confirms_cause"
        ]
        is False,
        "mode-0 scale compatibility cannot confirm a cause",
    )
    _require(
        amendment["state_propagation_interpretation_lock"][
            "attachment_correlations_inferential"
        ]
        is False,
        "three-case attachment correlations cannot be inferential",
    )
    _require(
        amendment["contact_registration_contract"]["schema_version"] == 3
        and amendment["contact_registration_contract"][
            "selected_and_rejected_candidates_required"
        ]
        is True,
        "v4 contact-registration provenance changed",
    )
    return {
        "passed": True,
        "plan_id": amendment["plan_id"],
        "amendment_sha256": amendment["amendment_sha256"],
        "physical_execution_count_changed": False,
        "mechanism_gate_threshold_changed": False,
        "contact_registration_schema": 3,
    }


def write_preacquisition_v4(path: str | Path, amendment: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(amendment), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_preacquisition_v4(
    path: str | Path,
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v3: Mapping[str, Any],
    gate_control_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    amendment = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_preacquisition_v4(
        amendment,
        protocol,
        v2,
        v3,
        gate_control_evidence,
    )
    return amendment


def load_v4_chain(
    protocol_path: str | Path,
    v2_path: str | Path,
    v3_path: str | Path,
    gate_control_path: str | Path,
    v4_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    v2 = load_preacquisition_amendment(v2_path, protocol)
    v3 = load_preacquisition_v3(v3_path, protocol, v2)
    gate_control = json.loads(Path(gate_control_path).read_text(encoding="utf-8"))
    v4 = load_preacquisition_v4(v4_path, protocol, v2, v3, gate_control)
    return protocol, v2, v3, v4
