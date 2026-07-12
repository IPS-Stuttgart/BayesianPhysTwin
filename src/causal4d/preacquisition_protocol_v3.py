"""Superseding analysis lock for the Causal4D pre-acquisition protocol."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from causal4d.preacquisition_protocol import (
    load_preacquisition_amendment,
    validate_preacquisition_amendment,
)
from causal4d.real_protocol import load_protocol, validate_protocol


PREACQUISITION_V3_SCHEMA_VERSION = 1
PREACQUISITION_V3_PLAN_ID = "causal4d-sloth-preacquisition-v3"
_CANONICAL_V3_SHA256 = (
    "5dd12d62242d672789e802ed6ed4365922e607932d15ef02a76b3208cdb9a1e2"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def preacquisition_v3_sha256(amendment: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(amendment))
    payload.pop("amendment_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source_crossfit_folds(v2: Mapping[str, Any]) -> list[dict[str, Any]]:
    executions = list(v2["preacquisition_signature_panel"]["executions"])
    folds = []
    for held_replicate in (1, 2, 3):
        heldout = [
            execution["execution_id"]
            for execution in executions
            if execution["replicate"] == held_replicate
        ]
        fit = [
            execution["execution_id"]
            for execution in executions
            if execution["replicate"] != held_replicate
        ]
        folds.append(
            {
                "fold_id": f"source-panel-hold-replicate-{held_replicate}",
                "mechanism_fit_execution_ids": fit,
                "heldout_shrinkage_execution_ids": heldout,
                "mechanism_fit_session_count": len(fit),
                "heldout_session_count": len(heldout),
                "readout_coefficients_refit_on_heldout_prefix": True,
                "heldout_future_used_for_coefficient_fit": False,
            }
        )
    return folds


def build_preacquisition_v3(
    protocol: Mapping[str, Any], v2: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a compact amendment that supersedes v2 before acquisition."""

    validate_protocol(protocol)
    validate_preacquisition_amendment(v2, protocol)
    source_panel = v2["preacquisition_signature_panel"]
    amendment: dict[str, Any] = {
        "schema_version": PREACQUISITION_V3_SCHEMA_VERSION,
        "plan_id": PREACQUISITION_V3_PLAN_ID,
        "status": "supersedes_v2_before_any_physical_execution",
        "supersedes": {
            "plan_id": v2["plan_id"],
            "amendment_sha256": v2["amendment_sha256"],
            "git_tag": "causal4d-preacquisition-v2",
            "physical_executions_completed_before_supersession": 0,
        },
        "base_protocol": deepcopy(v2["base_protocol"]),
        "unchanged_acquisition_design": {
            "source_panel_execution_ids": [
                execution["execution_id"] for execution in source_panel["executions"]
            ],
            "source_panel_execution_count": 12,
            "confirmatory_execution_count": 36,
            "confirmatory_target_ids_unchanged": True,
            "new_physical_executions_added_by_v3": 0,
        },
        "source_panel_role": {
            "role": "discovery_and_model_family_eligibility_only",
            "confirmatory_claim_allowed": False,
            "significance_test_allowed": False,
            "decision_rule": "predeclared_effect_direction_and_noise_scaled_magnitude",
            "replication_unit": "fresh_reset_session",
            "exact_repetitions_per_profile": 3,
        },
        "source_panel_crossfit": {
            "folds": _source_crossfit_folds(v2),
            "mechanism_fit_sessions_per_fold": 8,
            "heldout_shrinkage_sessions_per_fold": 4,
            "all_12_sessions_held_out_exactly_once": True,
            "mechanism_may_not_fit_heldout_session": True,
            "graph_correction_refit_boundary": (
                "Refit c_base and c_M independently from the permitted prefix of "
                "each held-out source execution, after mechanism parameters are frozen."
            ),
        },
        "repeatability_floor": {
            "name": "sigma_repeat",
            "definition": (
                "Square root of the equal-profile mean between-repeat variance of "
                "object-frame residual coordinates, computed separately by horizon "
                "or metric when used for standardization."
            ),
            "camera_specific_floor_reported_separately": True,
            "profiles_weighted_equally": True,
            "point_frames_are_not_replications": True,
        },
        "reversal_design": {
            "reset_separated": {
                "profiles": ["lift_high", "lower_high"],
                "purpose": "direction dependence from matched initial conditions",
                "pairing": "same replicate index after independent fresh resets",
                "sign_flip_statistic": (
                    "negative cosine between time-normalized action-axis residual "
                    "vectors in the common controller/world frame"
                ),
                "minimum_sign_flip_cosine": 0.50,
                "minimum_odd_component_rms_over_sigma_repeat": 1.50,
                "minimum_consistent_replicate_pairs": 2,
                "replicate_pair_count": 3,
            },
            "continuous": {
                "waveform": "minimum_jerk_out_hold_minimum_jerk_return",
                "purpose": "hysteresis and non-closure within one uninterrupted run",
                "nonclosure_statistic": (
                    "object-frame RMS of post-return-settle residual minus pre-action residual"
                ),
                "minimum_nonclosure_rms_over_sigma_repeat": 1.50,
                "minimum_persistence_frames": 3,
                "minimum_consistent_repetitions": 2,
                "repetition_count": 3,
            },
            "interpretation_lock": (
                "Reset-separated reversal diagnoses direction dependence; continuous "
                "out-and-return diagnoses path dependence. Neither substitutes for the other."
            ),
        },
        "signature_eligibility_gates": {
            "common": {
                "minimum_consistent_repetitions": 2,
                "repetition_count": 3,
                "p_values_or_null_rejection_allowed": False,
                "effect_below_threshold_interpretation": "not_eligible_for_confirmatory_mechanism",
            },
            "speed": {
                "profiles": ["lift_high", "lift_high_slow"],
                "minimum_effect_rms_over_sigma_repeat": 1.50,
                "required_measured_slow_to_fast_peak_speed_ratio": [0.35, 0.65],
                "matched": ["direction", "amplitude", "hold_duration", "contact"],
            },
            "hold_relaxation": {
                "profiles": ["lift_high", "lift_high_long_hold"],
                "minimum_relaxation_amplitude_over_sigma_repeat": 1.50,
                "minimum_exponential_log_r_squared": 0.80,
                "observable_time_constant_s": [0.0667, 0.50],
                "amplitude_definition": (
                    "difference between the first-three and last-three hold-frame "
                    "object-frame residual means"
                ),
            },
            "camera_observation": {
                "minimum_view_count": 3,
                "statistic": "leave-one-view-out correction transfer ratio",
                "required_direction": "held-view transfer worse than object-frame transfer",
                "magnitude_threshold": "at least 1.5 times camera-specific repeatability floor",
            },
            "reset_state": {
                "statistic": "residual covariance explained by measured reset deviation",
                "minimum_crossfit_effect_over_sigma_repeat": 1.50,
            },
        },
        "heldout_mechanism_eligibility": {
            "correction_metric": "graph_weighted_readout_correction_rms_m",
            "minimum_geometric_mean_shrinkage_fraction": 0.10,
            "minimum_sessions_with_positive_shrinkage": 8,
            "session_count": 12,
            "minimum_track_gain_over_metric_repeatability_sd": 1.0,
            "minimum_late_track_gain_over_metric_repeatability_sd": 1.0,
            "maximum_cd_degradation_over_metric_repeatability_sd": 0.5,
            "mechanism_parameters_frozen_before_heldout_prefix": True,
            "readout_correction_refit_on_heldout_prefix": True,
            "target_folds_used": False,
            "passing_meaning": (
                "Eligible for locked confirmatory evaluation, not a confirmed mechanism."
            ),
        },
        "calibration_resolution": {
            "calibration_unit": "one preregistered execution per independent session",
            "calibration_units_per_outer_fold": 9,
            "achievable_nominal_grid": [
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
            ],
            "selected_nominal_coverage": 0.90,
            "selected_order_statistic_rank_one_based": 9,
            "selected_threshold_is_maximum_calibration_score": True,
            "resolution": 0.10,
            "fragility_diagnostics": [
                "maximum_to_median_score_ratio",
                "largest_and_second_largest_scores",
                "leave_one_calibration_session_out_thresholds",
                "interval_width_by_outer_fold",
            ],
            "fragility_diagnostics_may_select_threshold": False,
            "pooled_coordinate_conformal_claim_allowed": False,
            "worst_group_coverage_guarantee_claimed": False,
            "claim": (
                "Finite marginal execution-block split-conformal coverage under "
                "session exchangeability; coarse rank-9-of-9 calibration."
            ),
        },
        "state_update_design_statement": {
            "role": "online_Bayesian_update_channel",
            "fast_transient_modes": "treated as effectively memoryless at 30 Hz",
            "slow_retained_modes": "retain posterior state uncertainty and propagate it",
            "mode_assignment_source": "frozen trajectory-only mode-retention audit",
            "material_time_constant_claimed": False,
        },
        "contact_registration_contract": {
            "schema_version": 2,
            "artifact_kind": "PhysicalContactRegistration",
            "weighted_node_patch_required": True,
            "minimum_independent_reviews": 2,
            "minimum_calibrated_camera_views": 3,
            "se3_covariance_and_closure_required": True,
            "support_geometry_required": True,
            "approval_required_before_slip_pilot": True,
            "target_outcomes_may_revise_registration": False,
        },
        "collection_sequence": deepcopy(v2["collection_sequence"]),
        "collection_gate": {
            **deepcopy(v2["collection_gate"]),
            "v3_analysis_code_frozen": False,
            "first_confirmatory_execution_allowed": False,
        },
    }
    amendment["amendment_sha256"] = preacquisition_v3_sha256(amendment)
    validate_preacquisition_v3(amendment, protocol, v2)
    return amendment


def validate_preacquisition_v3(
    amendment: Mapping[str, Any],
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate v3 without mutating the immutable v2 acquisition plan."""

    validate_protocol(protocol)
    validate_preacquisition_amendment(v2, protocol)
    _require(amendment.get("schema_version") == 1, "unsupported v3 schema")
    _require(amendment.get("plan_id") == PREACQUISITION_V3_PLAN_ID, "unexpected v3 id")
    _require(
        amendment.get("amendment_sha256") == preacquisition_v3_sha256(amendment),
        "v3 SHA-256 does not match its contents",
    )
    if _CANONICAL_V3_SHA256:
        _require(
            amendment["amendment_sha256"] == _CANONICAL_V3_SHA256,
            "v3 differs from the locked canonical design",
        )
    _require(
        amendment["supersedes"]["amendment_sha256"] == v2["amendment_sha256"],
        "v3 does not supersede the locked v2 artifact",
    )
    _require(
        amendment["base_protocol"] == v2["base_protocol"],
        "v3 changed the base confirmatory protocol",
    )
    v2_ids = [
        execution["execution_id"]
        for execution in v2["preacquisition_signature_panel"]["executions"]
    ]
    _require(
        amendment["unchanged_acquisition_design"]["source_panel_execution_ids"]
        == v2_ids,
        "v3 changed source-panel executions",
    )
    folds = amendment["source_panel_crossfit"]["folds"]
    _require(len(folds) == 3, "source cross-fit must contain three folds")
    heldout_counts = {identifier: 0 for identifier in v2_ids}
    for fold in folds:
        fit = set(fold["mechanism_fit_execution_ids"])
        heldout = set(fold["heldout_shrinkage_execution_ids"])
        _require(len(fit) == 8 and len(heldout) == 4, "cross-fit fold must be 8/4")
        _require(
            not fit & heldout and fit | heldout == set(v2_ids),
            "cross-fit boundary failed",
        )
        for identifier in heldout:
            heldout_counts[identifier] += 1
    _require(
        set(heldout_counts.values()) == {1},
        "every source execution must be held out once",
    )
    calibration = amendment["calibration_resolution"]
    _require(
        calibration["selected_order_statistic_rank_one_based"] == 9
        and calibration["selected_threshold_is_maximum_calibration_score"] is True,
        "90 percent calibration resolution is misstated",
    )
    _require(
        amendment["heldout_mechanism_eligibility"][
            "minimum_geometric_mean_shrinkage_fraction"
        ]
        > 0.0,
        "held-out shrinkage needs a positive magnitude threshold",
    )
    _require(
        amendment["source_panel_role"]["confirmatory_claim_allowed"] is False,
        "source panel cannot support a confirmatory claim",
    )
    _require(
        amendment["contact_registration_contract"]["schema_version"] == 2
        and amendment["contact_registration_contract"]["weighted_node_patch_required"]
        is True,
        "v3 contact registration contract changed",
    )
    return {
        "passed": True,
        "plan_id": amendment["plan_id"],
        "amendment_sha256": amendment["amendment_sha256"],
        "physical_execution_count_changed": False,
        "source_crossfit_fold_count": 3,
        "calibration_rank": 9,
    }


def write_preacquisition_v3(path: str | Path, amendment: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(amendment), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_preacquisition_v3(
    path: str | Path,
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
) -> dict[str, Any]:
    amendment = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_preacquisition_v3(amendment, protocol, v2)
    return amendment


def load_v3_chain(
    protocol_path: str | Path,
    v2_path: str | Path,
    v3_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    v2 = load_preacquisition_amendment(v2_path, protocol)
    v3 = load_preacquisition_v3(v3_path, protocol, v2)
    return protocol, v2, v3
