"""Versioned pre-acquisition amendment for the Causal4D real protocol."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from causal4d.preacquisition_analysis import conformal_rank_plan
from causal4d.real_protocol import load_protocol, validate_protocol


PREACQUISITION_SCHEMA_VERSION = 1
PREACQUISITION_PLAN_ID = "causal4d-sloth-preacquisition-v2"
_CANONICAL_AMENDMENT_SHA256 = (
    "57d9788c4de31ff3f103d487fcf7b2080523e69ead402e38961f47d0e749a719"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def amendment_sha256(amendment: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(amendment))
    payload.pop("amendment_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _signature_profiles(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    profiles = {
        profile["id"]: deepcopy(profile) for profile in protocol["command_profiles"]
    }
    reference = profiles["lift_high"]
    reverse = profiles["lower_high"]
    slow = {
        **deepcopy(reference),
        "id": "lift_high_slow",
        "outbound_duration_s": 1.50,
        "return_duration_s": 1.50,
    }
    long_hold = {
        **deepcopy(reference),
        "id": "lift_high_long_hold",
        "hold_duration_s": 1.00,
    }
    return [reference, reverse, slow, long_hold]


def _signature_panel(protocol: Mapping[str, Any]) -> dict[str, Any]:
    profiles = _signature_profiles(protocol)
    executions = []
    for profile in profiles:
        for replicate in range(1, 4):
            identifier = f"sloth-pre-v2-{profile['id']}-r{replicate}"
            executions.append(
                {
                    "execution_id": identifier,
                    "session_id": identifier,
                    "contact_region_id": "upper_torso",
                    "command_profile_id": profile["id"],
                    "realization_condition_id": "nominal",
                    "replicate": replicate,
                    "fresh_reset_and_fresh_grasp": True,
                    "confirmatory_fold_member": False,
                    "allowed_uses": [
                        "noise_floor_estimation",
                        "reset_repeatability",
                        "source_only_mechanism_signature_diagnosis",
                        "analysis_pipeline_testing",
                    ],
                    "forbidden_uses": [
                        "confirmatory_target_evaluation",
                        "target_threshold_selection",
                    ],
                }
            )
    return {
        "contact_region_id": "upper_torso",
        "independent_reset_per_execution": True,
        "profiles": profiles,
        "executions": executions,
        "execution_count": len(executions),
        "independent_session_count": len(executions),
        "exact_repetitions_per_profile": 3,
        "contrasts": [
            {
                "id": "direction_reversal",
                "reference_profile_id": "lift_high",
                "contrast_profile_id": "lower_high",
                "matched": [
                    "amplitude_m",
                    "outbound_duration_s",
                    "hold_duration_s",
                    "return_duration_s",
                ],
                "changed": ["direction_controller"],
                "signature": "tangential_or_directional_residual_sign_change",
            },
            {
                "id": "rate_dependence",
                "reference_profile_id": "lift_high",
                "contrast_profile_id": "lift_high_slow",
                "matched": [
                    "direction_controller",
                    "amplitude_m",
                    "hold_duration_s",
                ],
                "changed": ["outbound_duration_s", "return_duration_s"],
                "signature": "residual_changes_with_speed_at_fixed_amplitude",
            },
            {
                "id": "hold_relaxation",
                "reference_profile_id": "lift_high",
                "contrast_profile_id": "lift_high_long_hold",
                "matched": [
                    "direction_controller",
                    "amplitude_m",
                    "outbound_duration_s",
                    "return_duration_s",
                ],
                "changed": ["hold_duration_s"],
                "signature": "residual_relaxation_or_hysteresis_during_hold",
            },
        ],
        "camera_signature": {
            "minimum_synchronized_rgbd_views": 3,
            "analysis": "leave_one_camera_out_material_point_residual_transfer",
            "signature": "view_or_visibility_dependence_at_fixed_action",
        },
    }


def _balanced_calibration_executions(
    session_ids: list[str],
    *,
    session_by_id: Mapping[str, Mapping[str, Any]],
    execution_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Choose one independent execution per calibration session deterministically."""

    options = [
        list(session_by_id[session_id]["execution_ids"]) for session_id in session_ids
    ]
    best: tuple[tuple[Any, ...], list[str]] | None = None
    profile_ids = sorted(
        {execution["command_profile_id"] for execution in execution_by_id.values()}
    )
    for indices in itertools.product((0, 1), repeat=len(options)):
        selected = [
            choices[index] for choices, index in zip(options, indices, strict=True)
        ]
        profile_counts = Counter(
            execution_by_id[identifier]["command_profile_id"] for identifier in selected
        )
        pair_order_counts = Counter(
            execution_by_id[identifier]["pair_order"] for identifier in selected
        )
        profile_imbalance = sum(
            (profile_counts[profile] - len(selected) / len(profile_ids)) ** 2
            for profile in profile_ids
        )
        order_imbalance = abs(pair_order_counts[0] - pair_order_counts[1])
        acquisition_indices = tuple(
            execution_by_id[identifier]["acquisition_execution_index"]
            for identifier in selected
        )
        score = (profile_imbalance, order_imbalance, acquisition_indices)
        if best is None or score < best[0]:
            best = (score, selected)
    if best is None:
        raise ValueError("no calibration execution assignment exists")
    return best[1]


def _amended_calibration_folds(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    sessions = list(protocol["sessions"])
    executions = list(protocol["executions"])
    session_by_id = {session["session_id"]: session for session in sessions}
    execution_by_id = {execution["execution_id"]: execution for execution in executions}
    profile_ids_by_session = {
        session["session_id"]: {
            execution_by_id[identifier]["command_profile_id"]
            for identifier in session["execution_ids"]
        }
        for session in sessions
    }
    amended = []
    for original in protocol["splits"]["cross_action_contact_calibration_folds"]:
        target_execution_ids = list(original["target_execution_ids"])
        target_session_ids = sorted(
            {
                execution_by_id[identifier]["session_id"]
                for identifier in target_execution_ids
            }
        )
        held_contact = original["held_out_contact_region_id"]
        held_profile = original["held_out_command_profile_id"]
        fit_session_ids = sorted(
            session["session_id"]
            for session in sessions
            if session["session_id"] not in target_session_ids
            and session["contact_region_id"] != held_contact
            and held_profile not in profile_ids_by_session[session["session_id"]]
        )
        calibration_session_ids = sorted(
            session["session_id"]
            for session in sessions
            if session["session_id"] not in target_session_ids
            and session["session_id"] not in fit_session_ids
        )
        fit_execution_ids = [
            identifier
            for session_id in fit_session_ids
            for identifier in session_by_id[session_id]["execution_ids"]
        ]
        calibration_execution_ids = _balanced_calibration_executions(
            calibration_session_ids,
            session_by_id=session_by_id,
            execution_by_id=execution_by_id,
        )
        amended.append(
            {
                "fold_id": original["fold_id"],
                "held_out_contact_region_id": held_contact,
                "held_out_command_profile_id": held_profile,
                "fit_session_ids": fit_session_ids,
                "fit_execution_ids": fit_execution_ids,
                "calibration_session_ids": calibration_session_ids,
                "calibration_execution_ids": calibration_execution_ids,
                "target_session_ids": target_session_ids,
                "target_execution_ids": target_execution_ids,
                "unused_execution_ids_in_calibration_sessions": sorted(
                    {
                        identifier
                        for session_id in calibration_session_ids
                        for identifier in session_by_id[session_id]["execution_ids"]
                    }
                    - set(calibration_execution_ids)
                ),
                "unused_sibling_execution_ids_in_target_sessions": sorted(
                    {
                        identifier
                        for session_id in target_session_ids
                        for identifier in session_by_id[session_id]["execution_ids"]
                    }
                    - set(target_execution_ids)
                ),
                "calibration_plan": conformal_rank_plan(9, coverage=0.90),
            }
        )
    return amended


def build_preacquisition_amendment(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic amendment without changing the base 36-run grid."""

    validate_protocol(protocol)
    amendment: dict[str, Any] = {
        "schema_version": PREACQUISITION_SCHEMA_VERSION,
        "plan_id": PREACQUISITION_PLAN_ID,
        "status": "locked_before_first_physical_execution",
        "base_protocol": {
            "protocol_id": protocol["protocol_id"],
            "design_sha256": protocol["design_sha256"],
            "confirmatory_execution_count": len(protocol["executions"]),
            "target_execution_ids_unchanged": True,
            "action_contact_grid_unchanged": True,
        },
        "frozen_released_data_milestone": {
            "git_tag": "phystwin-discrepancy-localization-v1",
            "released_cases_available_for_further_model_selection": False,
            "allowed_followup": "post_hoc_state_correction_decay_diagnostic_only",
        },
        "preacquisition_signature_panel": _signature_panel(protocol),
        "amended_cross_action_calibration_folds": _amended_calibration_folds(protocol),
        "analysis_lock": {
            "replication_unit": "grasp_session",
            "paired_interval_method": "equal-session cluster bootstrap",
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 20_260_712,
            "confidence_level": 0.95,
            "regression_covariance": "CR1 session-clustered sandwich",
            "factorial_ladder": [
                "M0_nominal",
                "M1_measured_actuation_registered_contact",
                "M2_readout_persistence",
                "M3_hybrid_support_contact_regime",
                "M4_rate_dependent_material",
                "M5_self_contact_or_topology",
            ],
            "factorial_requirement": (
                "Fit every physical candidate both with and without the identical "
                "prefix-only graph readout persistence correction."
            ),
            "persistence_shrinkage_metric": "graph_readout_correction_rms_m",
            "persistence_shrinkage_gate": (
                "upper_95_percent_session_cluster_bootstrap_log_ratio_below_zero"
            ),
            "calibration": {
                "method": "execution-block split conformal",
                "score": (
                    "per-execution 0.90 quantile of absolute coordinate error "
                    "divided by raw predictive standard deviation"
                ),
                "calibration_unit": "one preregistered execution from each independent session",
                "calibration_units_per_fold": 9,
                "nominal_coverage": 0.90,
                "order_statistic_rank_one_based": 9,
                "fit_on": "amended calibration executions only",
                "target_adaptation_allowed": False,
            },
            "required_covariates": [
                "measured_reset_deviation",
                "action_direction",
                "measured_action_speed",
                "hold_duration",
                "contact_relative_tangential_motion",
                "support_proximity_and_normal",
                "camera_id_visibility_and_view_angle",
            ],
        },
        "actuator_realization_lock": {
            "simulator_primary_input": "measured_registered_actuator_trajectory",
            "commanded_trajectory_role": "intervention_command_and_diagnostic_reference",
            "hardware_timestamp_alignment_is_authoritative": True,
            "pyrecest": {
                "repository": "https://github.com/FlorianPfaff/PyRecEst",
                "locked_version": "2.4.1",
                "allowed_functions": [
                    "pyrecest.calibration.fit_time_offset",
                    "pyrecest.calibration.fit_sensor_bias_correction",
                    "pyrecest.metrics.nis",
                    "pyrecest.metrics.anees",
                ],
                "role": (
                    "source-only synchronization/bias diagnostics and consistency "
                    "statistics; never a replacement for hardware timestamps"
                ),
            },
            "maximum_rgbd_actuator_sync_error_ms": protocol["quality_gates"][
                "maximum_rgbd_actuator_sync_error_ms"
            ],
        },
        "collection_sequence": [
            "approve_contact_registration",
            "run_preregistered_slip_pilot",
            "collect_12_execution_signature_and_repeatability_panel",
            "validate_commanded_vs_measured_actuation",
            "validate_support_and_gravity_registration",
            "run_one_nonconfirmatory_end_to_end_dry_run",
            "seal_analysis_implementation_and_environment",
            "collect_36_unchanged_confirmatory_executions",
        ],
        "collection_gate": {
            "signature_panel_complete": False,
            "contact_registration_approved": False,
            "slip_pilot_passed_or_versioned_out": False,
            "actuator_sync_passed": False,
            "support_registration_passed": False,
            "end_to_end_dry_run_passed": False,
            "analysis_code_frozen": False,
            "first_confirmatory_execution_allowed": False,
        },
    }
    amendment["amendment_sha256"] = amendment_sha256(amendment)
    validate_preacquisition_amendment(amendment, protocol)
    return amendment


def validate_preacquisition_amendment(
    amendment: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the pre-acquisition design, analysis lock, and base boundary."""

    validate_protocol(protocol)
    _require(amendment.get("schema_version") == 1, "unsupported amendment schema")
    _require(amendment.get("plan_id") == PREACQUISITION_PLAN_ID, "unexpected plan id")
    _require(
        amendment.get("amendment_sha256") == amendment_sha256(amendment),
        "amendment SHA-256 does not match its contents",
    )
    if _CANONICAL_AMENDMENT_SHA256:
        _require(
            amendment["amendment_sha256"] == _CANONICAL_AMENDMENT_SHA256,
            "amendment differs from the locked canonical design",
        )
    base = amendment["base_protocol"]
    _require(base["protocol_id"] == protocol["protocol_id"], "base protocol changed")
    _require(
        base["design_sha256"] == protocol["design_sha256"],
        "base protocol hash changed",
    )
    _require(base["confirmatory_execution_count"] == 36, "base grid changed")

    panel = amendment["preacquisition_signature_panel"]
    profiles = {profile["id"]: profile for profile in panel["profiles"]}
    executions = list(panel["executions"])
    counts = Counter(execution["command_profile_id"] for execution in executions)
    _require(len(executions) == 12, "signature panel must contain 12 executions")
    _require(set(counts.values()) == {3}, "signature profiles need three repeats")
    _require(
        len({execution["session_id"] for execution in executions}) == 12,
        "signature repeats must use independent reset sessions",
    )
    _require(
        all(execution["confirmatory_fold_member"] is False for execution in executions),
        "signature execution entered a confirmatory fold",
    )
    reference = profiles["lift_high"]
    reverse = profiles["lower_high"]
    slow = profiles["lift_high_slow"]
    long_hold = profiles["lift_high_long_hold"]
    _require(
        reference["amplitude_m"] == reverse["amplitude_m"]
        and reference["outbound_duration_s"] == reverse["outbound_duration_s"]
        and reference["direction_controller"]
        == [-value for value in reverse["direction_controller"]],
        "direction-reversal contrast is not matched",
    )
    _require(
        reference["amplitude_m"] == slow["amplitude_m"]
        and reference["direction_controller"] == slow["direction_controller"]
        and reference["outbound_duration_s"] != slow["outbound_duration_s"],
        "speed contrast is not matched",
    )
    _require(
        reference["outbound_duration_s"] == long_hold["outbound_duration_s"]
        and reference["hold_duration_s"] != long_hold["hold_duration_s"],
        "hold contrast is not matched",
    )

    base_folds = {
        fold["fold_id"]: fold
        for fold in protocol["splits"]["cross_action_contact_calibration_folds"]
    }
    folds = list(amendment["amended_cross_action_calibration_folds"])
    _require(len(folds) == len(base_folds) == 12, "calibration folds changed")
    for fold in folds:
        original = base_folds[fold["fold_id"]]
        _require(
            fold["target_execution_ids"] == original["target_execution_ids"],
            "confirmatory target executions changed",
        )
        fit_sessions = set(fold["fit_session_ids"])
        calibration_sessions = set(fold["calibration_session_ids"])
        target_sessions = set(fold["target_session_ids"])
        _require(
            len(fit_sessions) == 6
            and len(calibration_sessions) == 9
            and len(target_sessions) == 3,
            "amended fold must partition all 18 sessions as 6/9/3",
        )
        _require(
            not (
                fit_sessions & calibration_sessions
                or fit_sessions & target_sessions
                or calibration_sessions & target_sessions
            ),
            "amended fold sessions overlap",
        )
        _require(len(fold["fit_execution_ids"]) == 12, "fit set must use 12 executions")
        _require(
            len(fold["calibration_execution_ids"]) == 9,
            "calibration must use one execution from each of nine sessions",
        )
        _require(
            fold["calibration_plan"]["finite_without_infinite_sentinel"] is True,
            "90 percent conformal calibration is not finite",
        )
    _require(
        amendment["analysis_lock"]["replication_unit"] == "grasp_session",
        "replication unit must remain the grasp session",
    )
    return {
        "passed": True,
        "plan_id": amendment["plan_id"],
        "amendment_sha256": amendment["amendment_sha256"],
        "signature_execution_count": len(executions),
        "confirmatory_execution_count": 36,
        "calibration_sessions_per_fold": 9,
    }


def write_preacquisition_amendment(
    path: str | Path, amendment: Mapping[str, Any]
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(amendment), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_preacquisition_amendment(
    path: str | Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    amendment = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_preacquisition_amendment(amendment, protocol)
    return amendment


def load_base_and_amendment(
    protocol_path: str | Path, amendment_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    amendment = load_preacquisition_amendment(amendment_path, protocol)
    return protocol, amendment
