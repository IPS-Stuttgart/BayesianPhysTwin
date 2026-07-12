"""Locked same-object, multi-action real protocol for Causal4D."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA_VERSION = 1
EXECUTION_MANIFEST_SCHEMA_VERSION = 1
PROTOCOL_ID = "causal4d-sloth-multi-action-v1"
_CANONICAL_DESIGN_SHA256 = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)

_CONTACT_IDS = ("left_forepaw", "right_forepaw", "upper_torso")
_PROFILE_IDS = (
    "lift_low",
    "lift_high",
    "lower_high",
    "lateral_low",
)
_CONDITION_IDS = (
    "nominal",
    "gain_low",
    "gain_high",
    "delay_2_frames",
    "frame_pitch_pos_3deg",
    "slip_low_force",
)

# The six edges of K4. The block assignment is a one-factorization: every
# profile appears exactly once in each of the three replicate blocks.
_PROFILE_PAIRS = (
    ("lift_low", "lift_high", 0),
    ("lift_low", "lower_high", 1),
    ("lift_low", "lateral_low", 2),
    ("lift_high", "lower_high", 2),
    ("lift_high", "lateral_low", 1),
    ("lower_high", "lateral_low", 0),
)

# Each contact receives all six K4 edges. Across contacts, every realization
# condition covers all four profiles with degree pattern (2, 2, 1, 1), and the
# one disjoint contact pair rotates evenly across conditions.
_PAIR_ASSIGNMENT_BY_CONTACT = (
    (0, 1, 2, 3, 4, 5),
    (1, 2, 3, 4, 5, 0),
    (5, 3, 1, 2, 0, 4),
)

_REQUIRED_ARTIFACTS = (
    "synchronized_rgbd_manifest",
    "commanded_control_trajectory",
    "measured_end_effector_trajectory",
    "measured_gripper_state",
    "camera_calibration",
    "controller_frame_calibration",
    "initial_object_state",
    "injected_intervention",
    "contact_region_annotation",
    "trial_reset_metadata",
    "drift_metadata",
)
_TIMESTAMPED_ARTIFACTS = (
    "synchronized_rgbd_manifest",
    "commanded_control_trajectory",
    "measured_end_effector_trajectory",
    "measured_gripper_state",
    "force_torque",
    "gripper_normal_force",
)
_ACQUISITION_SCHEDULE_FIELDS = (
    "acquisition_execution_index",
    "acquisition_session_index",
    "execution_id",
    "session_id",
    "pair_order",
    "contact_region_id",
    "command_profile_id",
    "realization_condition_id",
    "replicate_block",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def protocol_design_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash the canonical protocol while excluding its self-digest field."""

    payload = deepcopy(dict(protocol))
    payload.pop("design_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contact_regions() -> list[dict[str, Any]]:
    return [
        {
            "id": "left_forepaw",
            "label": "Anatomical left forepaw/wrist",
            "registration": "fixed canonical PhysTwin node set",
        },
        {
            "id": "right_forepaw",
            "label": "Anatomical right forepaw/wrist",
            "registration": "fixed canonical PhysTwin node set",
        },
        {
            "id": "upper_torso",
            "label": "Upper torso between the shoulders, below the neck seam",
            "registration": "fixed canonical PhysTwin node set",
        },
    ]


def _command_profiles() -> list[dict[str, Any]]:
    common = {
        "waveform": "minimum_jerk_out_hold_minimum_jerk_return",
        "outbound_duration_s": 0.75,
        "hold_duration_s": 0.25,
        "return_duration_s": 0.75,
        "post_return_settle_s": 1.5,
    }
    return [
        {
            "id": "lift_low",
            "direction_controller": [0.0, 0.0, 1.0],
            "amplitude_m": 0.04,
            **common,
        },
        {
            "id": "lift_high",
            "direction_controller": [0.0, 0.0, 1.0],
            "amplitude_m": 0.08,
            **common,
        },
        {
            "id": "lower_high",
            "direction_controller": [0.0, 0.0, -1.0],
            "amplitude_m": 0.08,
            **common,
        },
        {
            "id": "lateral_low",
            "direction_controller": [1.0, 0.0, 0.0],
            "amplitude_m": 0.04,
            **common,
        },
    ]


def _realization_conditions() -> list[dict[str, Any]]:
    nominal_phi = {
        "gain_multiplier": 1.0,
        "software_delay_frames": 0,
        "software_delay_s": 0.0,
        "frame_rotation_axis_controller": [0.0, 1.0, 0.0],
        "frame_rotation_deg": 0.0,
        "frame_translation_m": [0.0, 0.0, 0.0],
    }
    locked_kappa = {
        "gripper_force_scale": 1.0,
        "slip_mode": "locked_grip",
    }

    def condition(
        identifier: str,
        *,
        phi_updates: Mapping[str, Any] | None = None,
        kappa_updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        phi = {**nominal_phi, **dict(phi_updates or {})}
        kappa = {**locked_kappa, **dict(kappa_updates or {})}
        return {"id": identifier, "injection": {"phi": phi, "kappa": kappa}}

    return [
        condition("nominal"),
        condition("gain_low", phi_updates={"gain_multiplier": 0.85}),
        condition("gain_high", phi_updates={"gain_multiplier": 1.15}),
        condition(
            "delay_2_frames",
            phi_updates={
                "software_delay_frames": 2,
                "software_delay_s": 2.0 / 30.0,
            },
        ),
        condition(
            "frame_pitch_pos_3deg",
            phi_updates={"frame_rotation_deg": 3.0},
        ),
        condition(
            "slip_low_force",
            kappa_updates={
                "gripper_force_scale": 0.55,
                "slip_mode": "bounded_low_force",
                "target_slip_interval_m": [0.005, 0.015],
            },
        ),
    ]


def _session_order(sessions: list[dict[str, Any]]) -> list[str]:
    by_contact: dict[str, list[str]] = {}
    for contact_id in _CONTACT_IDS:
        identifiers = [
            session["session_id"]
            for session in sessions
            if session["contact_region_id"] == contact_id
        ]
        by_contact[contact_id] = sorted(
            identifiers,
            key=lambda value: hashlib.sha256(
                f"20260712:{value}".encode("ascii")
            ).hexdigest(),
        )
    ordered = []
    for round_index in range(6):
        for offset in range(3):
            contact_id = _CONTACT_IDS[(round_index + offset) % 3]
            ordered.append(by_contact[contact_id][round_index])
    return ordered


def _build_sessions_and_executions() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    sessions: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for contact_index, contact_id in enumerate(_CONTACT_IDS):
        for condition_index, condition_id in enumerate(_CONDITION_IDS):
            pair_index = _PAIR_ASSIGNMENT_BY_CONTACT[contact_index][condition_index]
            profile_a, profile_b, replicate_block = _PROFILE_PAIRS[pair_index]
            ordered_profiles = [profile_a, profile_b]
            if (contact_index + pair_index) % 2:
                ordered_profiles.reverse()
            session_id = f"sloth-v1-c{contact_index + 1}-s{condition_index + 1}"
            execution_ids = [f"{session_id}-e1", f"{session_id}-e2"]
            session = {
                "session_id": session_id,
                "contact_region_id": contact_id,
                "realization_condition_id": condition_id,
                "profile_pair_index": pair_index,
                "replicate_block": replicate_block,
                "same_grasp": True,
                "release_between_executions": False,
                "return_to_neutral_between_executions": True,
                "execution_ids": execution_ids,
            }
            sessions.append(session)
            for pair_order, (execution_id, profile_id) in enumerate(
                zip(execution_ids, ordered_profiles)
            ):
                executions.append(
                    {
                        "execution_id": execution_id,
                        "session_id": session_id,
                        "pair_order": pair_order,
                        "contact_region_id": contact_id,
                        "command_profile_id": profile_id,
                        "realization_condition_id": condition_id,
                        "replicate_block": replicate_block,
                    }
                )

    acquisition_order = _session_order(sessions)
    session_index = {
        session_id: index for index, session_id in enumerate(acquisition_order)
    }
    for session in sessions:
        session["acquisition_session_index"] = session_index[session["session_id"]]
    for execution in executions:
        execution["acquisition_execution_index"] = (
            2 * session_index[execution["session_id"]] + execution["pair_order"]
        )
    sessions.sort(key=lambda value: value["session_id"])
    executions.sort(key=lambda value: value["execution_id"])
    return sessions, executions


def _build_splits(
    sessions: list[dict[str, Any]],
    executions: list[dict[str, Any]],
) -> dict[str, Any]:
    session_by_id = {session["session_id"]: session for session in sessions}
    execution_by_id = {execution["execution_id"]: execution for execution in executions}
    lookup = {
        (
            execution["contact_region_id"],
            execution["realization_condition_id"],
            execution["command_profile_id"],
        ): execution["execution_id"]
        for execution in executions
    }

    factual = [
        {
            "execution_id": execution["execution_id"],
            "o_plus_prefix_frames": 6,
            "evaluation_window": "all untouched frames after the O+ prefix",
            "label_use": "evaluation_only",
        }
        for execution in executions
    ]

    same_grasp = []
    for session in sessions:
        first, second = session["execution_ids"]
        same_grasp.append(
            {
                "source_execution_id": first,
                "target_execution_id": second,
                "contact_region_id": session["contact_region_id"],
                "realization_condition_id": session["realization_condition_id"],
                "transfer_phi": True,
                "reuse_kappa": True,
                "contact_policy": "same_grasp",
                "target_semantics": (
                    "held_out_interventional_prediction_from_matched_initial_conditions"
                ),
            }
        )

    new_contact = []
    contact_transfers = ((0, 1), (1, 2), (0, 2))
    for condition_id in _CONDITION_IDS:
        for first_index, second_index in contact_transfers:
            first_contact = _CONTACT_IDS[first_index]
            second_contact = _CONTACT_IDS[second_index]
            first_profiles = {
                profile_id
                for contact_id, candidate_condition, profile_id in lookup
                if contact_id == first_contact and candidate_condition == condition_id
            }
            second_profiles = {
                profile_id
                for contact_id, candidate_condition, profile_id in lookup
                if contact_id == second_contact and candidate_condition == condition_id
            }
            shared_profiles = sorted(first_profiles & second_profiles)
            _require(
                len(shared_profiles) <= 1,
                "new-contact design has an ambiguous shared command profile",
            )
            if not shared_profiles:
                continue
            profile_id = shared_profiles[0]
            candidate_ids = [
                lookup[(first_contact, condition_id, profile_id)],
                lookup[(second_contact, condition_id, profile_id)],
            ]
            source_id, target_id = sorted(
                candidate_ids,
                key=lambda identifier: execution_by_id[identifier][
                    "acquisition_execution_index"
                ],
            )
            source = execution_by_id[source_id]
            target = execution_by_id[target_id]
            new_contact.append(
                {
                    "source_execution_id": source_id,
                    "target_execution_id": target_id,
                    "source_contact_region_id": source["contact_region_id"],
                    "target_contact_region_id": target["contact_region_id"],
                    "command_profile_id": profile_id,
                    "realization_condition_id": condition_id,
                    "transfer_phi": True,
                    "reuse_kappa": False,
                    "resample_kappa_cf": True,
                    "contact_policy": "new_contact",
                    "target_semantics": (
                        "held_out_interventional_prediction_from_matched_initial_conditions"
                    ),
                }
            )

    folds = []
    for held_contact in _CONTACT_IDS:
        for held_profile in _PROFILE_IDS:
            target_ids = sorted(
                execution["execution_id"]
                for execution in executions
                if execution["contact_region_id"] == held_contact
                and execution["command_profile_id"] == held_profile
            )
            eligible_sessions = [
                session
                for session in sessions
                if session["contact_region_id"] != held_contact
                and all(
                    execution_by_id[execution_id]["command_profile_id"] != held_profile
                    for execution_id in session["execution_ids"]
                )
            ]
            fit_session_ids = sorted(
                session["session_id"]
                for session in eligible_sessions
                if session["replicate_block"] in {0, 1}
            )
            calibration_session_ids = sorted(
                session["session_id"]
                for session in eligible_sessions
                if session["replicate_block"] == 2
            )

            def execution_ids(session_ids: list[str]) -> list[str]:
                return sorted(
                    execution_id
                    for session_id in session_ids
                    for execution_id in session_by_id[session_id]["execution_ids"]
                )

            folds.append(
                {
                    "fold_id": f"hold-{held_contact}-{held_profile}",
                    "held_out_contact_region_id": held_contact,
                    "held_out_command_profile_id": held_profile,
                    "fit_execution_ids": execution_ids(fit_session_ids),
                    "calibration_execution_ids": execution_ids(calibration_session_ids),
                    "target_execution_ids": target_ids,
                    "fit_session_ids": fit_session_ids,
                    "calibration_session_ids": calibration_session_ids,
                    "locked_hyperparameters": [
                        "likelihood_temperature",
                        "discrepancy_hyperparameters",
                        "coverage_transform",
                        "semantic_beta",
                    ],
                    "target_semantics": (
                        "held_out_interventional_prediction_from_matched_initial_conditions"
                    ),
                }
            )

    return {
        "factual_continuation": factual,
        "same_grasp_intervention_prediction": same_grasp,
        "new_contact_intervention_prediction": new_contact,
        "cross_action_contact_calibration_folds": folds,
    }


def build_same_object_real_protocol() -> dict[str, Any]:
    """Build the deterministic 36-execution preregistered protocol."""

    contacts = _contact_regions()
    profiles = _command_profiles()
    conditions = _realization_conditions()
    sessions, executions = _build_sessions_and_executions()
    acquisition_order = [
        session["session_id"]
        for session in sorted(
            sessions,
            key=lambda value: value["acquisition_session_index"],
        )
    ]
    protocol: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "preregistered_before_confirmatory_collection",
        "object": {
            "object_id": "sloth_plush_instance_1",
            "source_case": "single_lift_sloth",
            "object_instance_must_remain_fixed": True,
            "canonical_contact_registration_required": True,
        },
        "terminology": {
            "real_target_name": (
                "held_out_interventional_prediction_from_matched_initial_conditions"
            ),
            "individual_counterfactual_ground_truth_available": False,
            "individual_counterfactual_validation_domain": "controlled_simulator_only",
        },
        "sampling": {
            "rgbd_rate_hz": 30,
            "command_and_actuator_minimum_rate_hz": 100,
            "preferred_force_torque_rate_hz": 250,
            "shared_monotonic_clock_required": True,
        },
        "neutral_state": {
            "description": (
                "standardized suspended neutral pose with at least 0.10 m table "
                "clearance, reached without releasing the registered grasp"
            ),
            "pre_execution_stabilization_s": 2.0,
            "return_to_neutral_between_same_grasp_commands": True,
        },
        "contact_regions": contacts,
        "command_profiles": profiles,
        "realization_conditions": conditions,
        "sessions": sessions,
        "executions": executions,
        "acquisition_session_order": acquisition_order,
        "splits": _build_splits(sessions, executions),
        "recording_contract": {
            "required_artifacts": list(_REQUIRED_ARTIFACTS),
            "timestamped_artifacts": list(_TIMESTAMPED_ARTIFACTS),
            "artifact_descriptor_fields": [
                "path",
                "sha256",
                "bytes",
            ],
            "slip_condition_any_of": [
                "force_torque",
                "gripper_normal_force",
            ],
            "commanded_and_measured_actuator_trajectories_are_distinct": True,
        },
        "quality_gates": {
            "maximum_rgbd_actuator_sync_error_ms": 5.0,
            "maximum_initial_state_chamfer_m": 0.003,
            "maximum_end_effector_reset_error_m": 0.002,
            "maximum_contact_centroid_error_m": 0.005,
            "maximum_dropped_rgbd_frames": 0,
        },
        "slip_activation_gate": {
            "applies_to": "slip_low_force",
            "must_pass_before_confirmatory_collection": True,
            "minimum_pilot_executions": 5,
            "minimum_bounded_slip_successes": 4,
            "bounded_slip_interval_m": [0.005, 0.015],
            "maximum_slip_coefficient_of_variation": 0.35,
            "complete_release_allowed": False,
            "failure_action": (
                "stop collection and issue a new protocol version without the "
                "slip condition; do not substitute after seeing target outcomes"
            ),
        },
        "analysis_lock": {
            "split_unit": "grasp_session",
            "target_outcomes_may_not_select_hyperparameters": True,
            "no_session_shared_between_fit_and_calibration": True,
            "no_held_contact_or_profile_in_fold_source_sessions": True,
            "exclusions_locked_before_target_evaluation": True,
        },
    }
    protocol["design_sha256"] = protocol_design_sha256(protocol)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate factorial balance, causal split semantics, and the design hash."""

    _require(protocol.get("schema_version") == 1, "unsupported protocol schema")
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "unexpected protocol id")
    _require(
        protocol.get("design_sha256") == protocol_design_sha256(protocol),
        "protocol design SHA-256 does not match its contents",
    )
    sessions = list(protocol["sessions"])
    executions = list(protocol["executions"])
    _require(len(sessions) == 18, "protocol must contain 18 grasp sessions")
    _require(len(executions) == 36, "protocol must contain 36 executions")

    session_ids = [session["session_id"] for session in sessions]
    execution_ids = [execution["execution_id"] for execution in executions]
    _require(len(set(session_ids)) == 18, "session ids must be unique")
    _require(len(set(execution_ids)) == 36, "execution ids must be unique")
    _require(
        set(protocol["acquisition_session_order"]) == set(session_ids),
        "acquisition order must contain every session exactly once",
    )
    session_by_id = {session["session_id"]: session for session in sessions}
    ordered_contacts = [
        session_by_id[session_id]["contact_region_id"]
        for session_id in protocol["acquisition_session_order"]
    ]
    _require(
        all(
            left != right for left, right in zip(ordered_contacts, ordered_contacts[1:])
        ),
        "acquisition order must interleave contact regions",
    )

    execution_by_id = {execution["execution_id"]: execution for execution in executions}
    for session in sessions:
        _require(session["same_grasp"] is True, "session must preserve the grasp")
        _require(
            session["release_between_executions"] is False,
            "same-grasp session cannot release between commands",
        )
        _require(
            len(session["execution_ids"]) == 2,
            "every session must contain two commands",
        )
        for execution_id in session["execution_ids"]:
            _require(execution_id in execution_by_id, "session execution is missing")
            _require(
                execution_by_id[execution_id]["session_id"] == session["session_id"],
                "execution points to the wrong session",
            )
    session_cells = Counter(
        (session["contact_region_id"], session["realization_condition_id"])
        for session in sessions
    )
    _require(
        len(session_cells) == 18 and set(session_cells.values()) == {1},
        "every contact must contain every realization condition once",
    )
    for contact_id in _CONTACT_IDS:
        pair_indices = {
            session["profile_pair_index"]
            for session in sessions
            if session["contact_region_id"] == contact_id
        }
        _require(
            pair_indices == set(range(6)),
            "every contact must use all six command-profile pairs",
        )

    contact_counts = Counter(execution["contact_region_id"] for execution in executions)
    profile_counts = Counter(
        execution["command_profile_id"] for execution in executions
    )
    condition_counts = Counter(
        execution["realization_condition_id"] for execution in executions
    )
    condition_profile_counts = Counter(
        (
            execution["realization_condition_id"],
            execution["command_profile_id"],
        )
        for execution in executions
    )
    _require(
        contact_counts == Counter({identifier: 12 for identifier in _CONTACT_IDS}),
        "each contact region must have 12 executions",
    )
    _require(
        profile_counts == Counter({identifier: 9 for identifier in _PROFILE_IDS}),
        "each command profile must have 9 executions",
    )
    _require(
        condition_counts == Counter({identifier: 6 for identifier in _CONDITION_IDS}),
        "each realization condition must have 6 executions",
    )
    _require(
        all(
            condition_profile_counts[condition_id, profile_id] in {1, 2}
            for condition_id in _CONDITION_IDS
            for profile_id in _PROFILE_IDS
        ),
        "every realization condition must cover all four command profiles",
    )
    cell_counts = Counter(
        (execution["contact_region_id"], execution["command_profile_id"])
        for execution in executions
    )
    _require(
        set(cell_counts.values()) == {3} and len(cell_counts) == 12,
        "every contact/profile cell must have three replicates",
    )
    block_counts = Counter(
        (
            execution["contact_region_id"],
            execution["command_profile_id"],
            execution["replicate_block"],
        )
        for execution in executions
    )
    _require(
        set(block_counts.values()) == {1} and len(block_counts) == 36,
        "each contact/profile must occur once per replicate block",
    )
    first_counts = Counter(
        execution["command_profile_id"]
        for execution in executions
        if execution["pair_order"] == 0
    )
    second_counts = Counter(
        execution["command_profile_id"]
        for execution in executions
        if execution["pair_order"] == 1
    )
    _require(
        all(
            abs(first_counts[profile_id] - second_counts[profile_id]) <= 1
            for profile_id in _PROFILE_IDS
        ),
        "same-grasp command order is not counterbalanced",
    )

    splits = protocol["splits"]
    _require(
        len(splits["factual_continuation"]) == 36,
        "factual split must include all executions",
    )
    _require(
        len(splits["same_grasp_intervention_prediction"]) == 18,
        "same-grasp split must contain one chronological pair per session",
    )
    for pair in splits["same_grasp_intervention_prediction"]:
        source = execution_by_id[pair["source_execution_id"]]
        target = execution_by_id[pair["target_execution_id"]]
        _require(source["session_id"] == target["session_id"], "grasp changed")
        _require(
            source["pair_order"] == 0 and target["pair_order"] == 1, "time reversed"
        )
        _require(pair["reuse_kappa"] is True, "same grasp must reuse kappa")

    _require(
        len(splits["new_contact_intervention_prediction"]) == 12,
        "new-contact split must contain 12 matched-command transfers",
    )
    new_contact_pairs: Counter[tuple[str, str]] = Counter()
    new_contact_profiles: Counter[str] = Counter()
    new_contact_conditions: Counter[str] = Counter()
    for pair in splits["new_contact_intervention_prediction"]:
        source = execution_by_id[pair["source_execution_id"]]
        target = execution_by_id[pair["target_execution_id"]]
        _require(
            source["contact_region_id"] != target["contact_region_id"],
            "new-contact transfer must change contact",
        )
        _require(
            source["command_profile_id"] == target["command_profile_id"],
            "new-contact transfer must hold the command profile fixed",
        )
        _require(
            source["realization_condition_id"] == target["realization_condition_id"],
            "new-contact transfer must hold persistent phi fixed",
        )
        _require(
            source["acquisition_execution_index"]
            < target["acquisition_execution_index"],
            "new-contact transfer must respect acquisition chronology",
        )
        _require(pair["resample_kappa_cf"] is True, "new contact must resample kappa")
        new_contact_pairs[
            tuple(sorted((source["contact_region_id"], target["contact_region_id"])))
        ] += 1
        new_contact_profiles[source["command_profile_id"]] += 1
        new_contact_conditions[source["realization_condition_id"]] += 1
    _require(
        set(new_contact_pairs.values()) == {4} and len(new_contact_pairs) == 3,
        "new-contact transfers are not balanced across contact pairs",
    )
    _require(
        new_contact_profiles == Counter({identifier: 3 for identifier in _PROFILE_IDS}),
        "new-contact transfers are not balanced across command profiles",
    )
    _require(
        new_contact_conditions
        == Counter({identifier: 2 for identifier in _CONDITION_IDS}),
        "new-contact transfers are not balanced across realization conditions",
    )

    folds = splits["cross_action_contact_calibration_folds"]
    _require(len(folds) == 12, "cross-action calibration requires 12 locked folds")
    target_uses: Counter[str] = Counter()
    session_by_execution = {
        execution_id: session["session_id"]
        for session in sessions
        for execution_id in session["execution_ids"]
    }
    for fold in folds:
        fit_ids = set(fold["fit_execution_ids"])
        calibration_ids = set(fold["calibration_execution_ids"])
        target_ids = set(fold["target_execution_ids"])
        _require(len(fit_ids) == 8, "fold fit set must contain 8 executions")
        _require(
            len(calibration_ids) == 4,
            "fold calibration set must contain 4 executions",
        )
        _require(len(target_ids) == 3, "fold target must contain 3 repetitions")
        _require(
            not (
                fit_ids & calibration_ids
                or fit_ids & target_ids
                or calibration_ids & target_ids
            ),
            "fold partitions overlap",
        )
        fit_sessions = {session_by_execution[identifier] for identifier in fit_ids}
        calibration_sessions = {
            session_by_execution[identifier] for identifier in calibration_ids
        }
        _require(
            not fit_sessions & calibration_sessions,
            "a grasp session crosses fit and calibration",
        )
        held_contact = fold["held_out_contact_region_id"]
        held_profile = fold["held_out_command_profile_id"]
        for identifier in fit_ids | calibration_ids:
            execution = execution_by_id[identifier]
            _require(
                execution["contact_region_id"] != held_contact,
                "held contact leaked into fold source",
            )
            _require(
                execution["command_profile_id"] != held_profile,
                "held action leaked into fold source",
            )
        for identifier in target_ids:
            target_uses[identifier] += 1
            execution = execution_by_id[identifier]
            _require(
                execution["contact_region_id"] == held_contact
                and execution["command_profile_id"] == held_profile,
                "target does not match held factors",
            )
    _require(
        target_uses == Counter({identifier: 1 for identifier in execution_ids}),
        "every execution must be out-of-fold target exactly once",
    )
    _require(
        protocol["design_sha256"] == _CANONICAL_DESIGN_SHA256,
        "protocol differs from the canonical locked v1 design",
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "design_sha256": protocol["design_sha256"],
        "sessions": len(sessions),
        "executions": len(executions),
        "cross_action_contact_folds": len(folds),
        "passed": True,
    }


def write_protocol(path: str | Path, protocol: Mapping[str, Any] | None = None) -> Path:
    """Write a validated protocol as stable, human-readable JSON."""

    output = Path(path)
    payload = dict(protocol or build_same_object_real_protocol())
    validate_protocol(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _acquisition_schedule_rows(
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    session_by_id = {session["session_id"]: session for session in protocol["sessions"]}
    rows = []
    for execution in sorted(
        protocol["executions"],
        key=lambda value: value["acquisition_execution_index"],
    ):
        session = session_by_id[execution["session_id"]]
        rows.append(
            {
                "acquisition_execution_index": execution["acquisition_execution_index"],
                "acquisition_session_index": session["acquisition_session_index"],
                "execution_id": execution["execution_id"],
                "session_id": execution["session_id"],
                "pair_order": execution["pair_order"],
                "contact_region_id": execution["contact_region_id"],
                "command_profile_id": execution["command_profile_id"],
                "realization_condition_id": execution["realization_condition_id"],
                "replicate_block": execution["replicate_block"],
            }
        )
    return rows


def write_acquisition_schedule(
    path: str | Path,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Write the locked execution order as an operator-friendly CSV."""

    payload = dict(protocol or build_same_object_real_protocol())
    validate_protocol(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ACQUISITION_SCHEDULE_FIELDS)
        writer.writeheader()
        writer.writerows(_acquisition_schedule_rows(payload))
    return output


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def execution_manifest_template(
    protocol: Mapping[str, Any], execution_id: str
) -> dict[str, Any]:
    """Build an explicitly incomplete acquisition manifest template."""

    validate_protocol(protocol)
    execution_by_id = {
        execution["execution_id"]: execution for execution in protocol["executions"]
    }
    if execution_id not in execution_by_id:
        raise KeyError(execution_id)
    execution = execution_by_id[execution_id]
    condition = next(
        value
        for value in protocol["realization_conditions"]
        if value["id"] == execution["realization_condition_id"]
    )
    artifact_names = list(protocol["recording_contract"]["required_artifacts"])
    artifact_names.extend(("force_torque", "gripper_normal_force"))
    return {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "execution_id": execution_id,
        "session_id": execution["session_id"],
        "contact_region_id": execution["contact_region_id"],
        "command_profile_id": execution["command_profile_id"],
        "realization_condition_id": execution["realization_condition_id"],
        "replicate_block": execution["replicate_block"],
        "acquisition_status": "template",
        "acquisition": {
            "operator_id": None,
            "hardware_run_id": None,
            "started_at_utc": None,
        },
        "known_injection": deepcopy(condition["injection"]),
        "timing": {
            "frame_count": None,
            "intervention_frame": None,
            "o_plus_prefix_frames": 6,
        },
        "artifacts": {
            name: {
                "path": None,
                "sha256": None,
                "bytes": None,
                **({"clock_id": None} if name in _TIMESTAMPED_ARTIFACTS else {}),
            }
            for name in artifact_names
        },
        "quality": {
            "reset_passed": None,
            "rgbd_actuator_sync_error_ms": None,
            "initial_state_chamfer_m": None,
            "end_effector_reset_error_m": None,
            "contact_centroid_error_m": None,
            "dropped_rgbd_frames": None,
            "slip_displacement_m": None,
            "complete_release_observed": None,
        },
        "drift_indicators": {
            "wear_cycle_count": None,
            "minutes_since_first_execution": None,
            "object_temperature_c": None,
            "room_temperature_c": None,
            "notes": None,
        },
        "information_boundary": {
            "target_frames_used_for_inference": False,
            "target_frames_used_for_hyperparameter_selection": False,
            "individual_counterfactual_ground_truth_claimed": False,
        },
        "exclusion": {
            "status": "pending",
            "reason": None,
            "decided_before_target_evaluation": None,
        },
    }


def object_registration_template(protocol: Mapping[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "object_id": protocol["object"]["object_id"],
        "object_instance_serial": None,
        "phystwin_model_id": None,
        "phystwin_model_sha256": None,
        "contact_regions": {
            region["id"]: {
                "canonical_node_set_path": None,
                "canonical_node_set_sha256": None,
                "node_count": None,
            }
            for region in protocol["contact_regions"]
        },
    }


def slip_pilot_template(protocol: Mapping[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "condition_id": "slip_low_force",
        "pilot_execution_ids": [],
        "contact_region_ids": [],
        "bounded_slip_successes": None,
        "slip_displacement_mean_m": None,
        "slip_displacement_coefficient_of_variation": None,
        "complete_release_count": None,
        "passed": None,
        "decided_before_confirmatory_collection": None,
    }


def scaffold_dataset(
    protocol: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Create non-overwriting acquisition templates for every locked execution."""

    validate_protocol(protocol)
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty dataset root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    write_protocol(root / "protocol.json", protocol)
    write_acquisition_schedule(root / "acquisition_schedule.csv", protocol)
    (root / "object_registration.template.json").write_text(
        json.dumps(object_registration_template(protocol), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (root / "slip_pilot.template.json").write_text(
        json.dumps(slip_pilot_template(protocol), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for session in protocol["sessions"]:
        session_root = root / "sessions" / session["session_id"]
        session_root.mkdir(parents=True, exist_ok=True)
        (session_root / "session.template.json").write_text(
            json.dumps(session, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for execution in protocol["executions"]:
        execution_root = root / "executions" / execution["execution_id"]
        execution_root.mkdir(parents=True, exist_ok=True)
        (execution_root / "manifest.template.json").write_text(
            json.dumps(
                execution_manifest_template(protocol, execution["execution_id"]),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "root": str(root.resolve()),
        "sessions": len(protocol["sessions"]),
        "execution_templates": len(protocol["executions"]),
    }


def _validate_artifact_descriptor(
    name: str,
    descriptor: Mapping[str, Any],
    *,
    execution_root: Path | None,
    verify_files: bool,
) -> None:
    path_value = descriptor.get("path")
    _require(isinstance(path_value, str) and path_value, f"{name} path is missing")
    path = Path(path_value)
    _require(
        not path.is_absolute() and ".." not in path.parts, f"{name} path is unsafe"
    )
    _require(_is_sha256(descriptor.get("sha256")), f"{name} SHA-256 is invalid")
    byte_count = descriptor.get("bytes")
    _require(
        isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count >= 0,
        f"{name} byte count is invalid",
    )
    if name in _TIMESTAMPED_ARTIFACTS:
        _require(
            isinstance(descriptor.get("clock_id"), str)
            and bool(descriptor["clock_id"]),
            f"{name} clock_id is missing",
        )
    if verify_files:
        _require(
            execution_root is not None, "execution root is required for file checks"
        )
        artifact_path = execution_root / path
        _require(artifact_path.is_file(), f"{name} file is missing: {artifact_path}")
        digest, actual_bytes = _sha256_file(artifact_path)
        _require(digest == descriptor["sha256"], f"{name} checksum mismatch")
        _require(actual_bytes == byte_count, f"{name} byte count mismatch")


def validate_execution_manifest(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    execution_root: str | Path | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Validate one completed execution against the locked design."""

    validate_protocol(protocol)
    _require(
        manifest.get("schema_version") == EXECUTION_MANIFEST_SCHEMA_VERSION,
        "unsupported execution manifest schema",
    )
    _require(
        manifest.get("protocol_id") == protocol["protocol_id"], "protocol id mismatch"
    )
    _require(
        manifest.get("protocol_design_sha256") == protocol["design_sha256"],
        "manifest protocol digest mismatch",
    )
    execution_by_id = {
        execution["execution_id"]: execution for execution in protocol["executions"]
    }
    execution_id = manifest.get("execution_id")
    _require(execution_id in execution_by_id, "execution is not in the protocol")
    expected = execution_by_id[execution_id]
    for field in (
        "session_id",
        "contact_region_id",
        "command_profile_id",
        "realization_condition_id",
        "replicate_block",
    ):
        _require(manifest.get(field) == expected[field], f"manifest {field} changed")
    _require(
        manifest.get("acquisition_status") == "complete", "execution is incomplete"
    )
    acquisition = manifest.get("acquisition", {})
    for field in ("operator_id", "hardware_run_id", "started_at_utc"):
        _require(
            isinstance(acquisition.get(field), str) and bool(acquisition[field]),
            f"acquisition {field} is missing",
        )
    condition = next(
        value
        for value in protocol["realization_conditions"]
        if value["id"] == expected["realization_condition_id"]
    )
    _require(
        manifest.get("known_injection") == condition["injection"],
        "known injection differs from the preregistered condition",
    )
    timing = manifest.get("timing", {})
    frame_count = timing.get("frame_count")
    intervention_frame = timing.get("intervention_frame")
    _require(isinstance(frame_count, int) and frame_count > 6, "frame count is invalid")
    _require(
        isinstance(intervention_frame, int)
        and 0 < intervention_frame < frame_count - 6,
        "intervention frame is invalid",
    )
    _require(
        timing.get("o_plus_prefix_frames") == 6, "O+ prefix must remain six frames"
    )

    artifacts = manifest.get("artifacts", {})
    required = set(protocol["recording_contract"]["required_artifacts"])
    root = Path(execution_root) if execution_root is not None else None
    if expected["realization_condition_id"] == "slip_low_force":
        alternatives = set(protocol["recording_contract"]["slip_condition_any_of"])
        selected = [
            name
            for name in alternatives
            if name in artifacts and artifacts[name].get("path")
        ]
        _require(
            bool(selected),
            "slip execution requires force/torque or normal-force data",
        )
        for name in selected:
            _validate_artifact_descriptor(
                name,
                artifacts[name],
                execution_root=root,
                verify_files=verify_files,
            )
    for name in sorted(required):
        _require(name in artifacts, f"required artifact is missing: {name}")
        _validate_artifact_descriptor(
            name,
            artifacts[name],
            execution_root=root,
            verify_files=verify_files,
        )
    quality = manifest.get("quality", {})
    gates = protocol["quality_gates"]
    exclusion = manifest.get("exclusion", {})
    _require(
        exclusion.get("status") in {"included", "excluded"},
        "exclusion status must be locked",
    )
    _require(
        exclusion.get("decided_before_target_evaluation") is True,
        "exclusion was not decided before target evaluation",
    )
    if exclusion["status"] == "excluded":
        _require(
            isinstance(exclusion.get("reason"), str) and bool(exclusion["reason"]),
            "excluded execution needs a reason",
        )

    gate_failures = []
    _require(
        isinstance(quality.get("reset_passed"), bool),
        "reset_passed must be recorded",
    )
    if not quality["reset_passed"]:
        gate_failures.append("reset_passed")
    numeric_gates = (
        ("rgbd_actuator_sync_error_ms", "maximum_rgbd_actuator_sync_error_ms"),
        ("initial_state_chamfer_m", "maximum_initial_state_chamfer_m"),
        ("end_effector_reset_error_m", "maximum_end_effector_reset_error_m"),
        ("contact_centroid_error_m", "maximum_contact_centroid_error_m"),
        ("dropped_rgbd_frames", "maximum_dropped_rgbd_frames"),
    )
    for metric, threshold in numeric_gates:
        value = quality.get(metric)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"quality metric is missing: {metric}",
        )
        if value > gates[threshold]:
            gate_failures.append(metric)
    if expected["realization_condition_id"] == "slip_low_force":
        lower, upper = protocol["slip_activation_gate"]["bounded_slip_interval_m"]
        slip = quality.get("slip_displacement_m")
        _require(
            isinstance(slip, (int, float)) and not isinstance(slip, bool),
            "slip displacement is missing",
        )
        _require(
            isinstance(quality.get("complete_release_observed"), bool),
            "complete-release state is missing",
        )
        if not lower <= slip <= upper:
            gate_failures.append("slip_displacement_m")
        if quality["complete_release_observed"]:
            gate_failures.append("complete_release_observed")
    if gate_failures:
        _require(
            exclusion["status"] == "excluded",
            "failed quality gates require a preregistered exclusion",
        )
    drift = manifest.get("drift_indicators", {})
    _require(
        isinstance(drift.get("wear_cycle_count"), int)
        and drift["wear_cycle_count"] >= 0,
        "wear cycle count is missing",
    )
    _require(
        isinstance(drift.get("minutes_since_first_execution"), (int, float))
        and drift["minutes_since_first_execution"] >= 0.0,
        "drift time is missing",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary
        == {
            "target_frames_used_for_inference": False,
            "target_frames_used_for_hyperparameter_selection": False,
            "individual_counterfactual_ground_truth_claimed": False,
        },
        "information boundary changed",
    )
    return {
        "execution_id": execution_id,
        "quality_gate_failures": gate_failures,
        "included": exclusion["status"] == "included",
        "passed": True,
    }


def validate_object_registration(
    protocol: Mapping[str, Any], registration: Mapping[str, Any]
) -> None:
    validate_protocol(protocol)
    _require(
        registration.get("schema_version") == 1,
        "unsupported object registration schema",
    )
    _require(
        registration.get("protocol_id") == protocol["protocol_id"],
        "registration protocol mismatch",
    )
    _require(
        registration.get("protocol_design_sha256") == protocol["design_sha256"],
        "registration digest mismatch",
    )
    _require(
        registration.get("object_id") == protocol["object"]["object_id"],
        "object mismatch",
    )
    for field in ("object_instance_serial", "phystwin_model_id"):
        _require(
            isinstance(registration.get(field), str) and bool(registration[field]),
            f"registration {field} is missing",
        )
    _require(
        _is_sha256(registration.get("phystwin_model_sha256")),
        "twin SHA-256 is invalid",
    )
    regions = registration.get("contact_regions", {})
    for region_id in _CONTACT_IDS:
        descriptor = regions.get(region_id, {})
        _require(
            isinstance(descriptor.get("canonical_node_set_path"), str)
            and bool(descriptor["canonical_node_set_path"]),
            f"contact node set is missing: {region_id}",
        )
        node_path = Path(descriptor["canonical_node_set_path"])
        _require(
            not node_path.is_absolute() and ".." not in node_path.parts,
            f"contact node-set path is unsafe: {region_id}",
        )
        _require(
            _is_sha256(descriptor.get("canonical_node_set_sha256")),
            f"contact node-set SHA-256 is invalid: {region_id}",
        )
        _require(
            isinstance(descriptor.get("node_count"), int)
            and descriptor["node_count"] > 0,
            f"contact node count is invalid: {region_id}",
        )


def validate_slip_pilot(protocol: Mapping[str, Any], pilot: Mapping[str, Any]) -> None:
    validate_protocol(protocol)
    _require(pilot.get("schema_version") == 1, "unsupported slip pilot schema")
    gate = protocol["slip_activation_gate"]
    _require(
        pilot.get("protocol_id") == protocol["protocol_id"],
        "slip pilot protocol mismatch",
    )
    _require(
        pilot.get("protocol_design_sha256") == protocol["design_sha256"],
        "slip pilot digest mismatch",
    )
    _require(
        pilot.get("condition_id") == gate["applies_to"],
        "wrong slip condition",
    )
    pilot_ids = pilot.get("pilot_execution_ids", [])
    _require(
        len(pilot_ids) >= gate["minimum_pilot_executions"],
        "too few slip pilot executions",
    )
    _require(len(set(pilot_ids)) == len(pilot_ids), "slip pilot ids are duplicated")
    contact_ids = set(pilot.get("contact_region_ids", []))
    _require(
        len(contact_ids) >= 2 and contact_ids <= set(_CONTACT_IDS),
        "slip pilot must cover at least two registered contacts",
    )
    successes = pilot.get("bounded_slip_successes", 0)
    _require(
        successes >= gate["minimum_bounded_slip_successes"],
        "slip pilot has too few bounded successes",
    )
    _require(successes <= len(pilot_ids), "slip successes exceed pilot executions")
    _require(pilot.get("complete_release_count") == 0, "slip pilot released the object")
    mean_slip = pilot.get("slip_displacement_mean_m")
    lower, upper = gate["bounded_slip_interval_m"]
    _require(
        isinstance(mean_slip, (int, float))
        and not isinstance(mean_slip, bool)
        and lower <= mean_slip <= upper,
        "mean pilot slip is outside the bounded interval",
    )
    variation = pilot.get("slip_displacement_coefficient_of_variation")
    _require(
        isinstance(variation, (int, float))
        and not isinstance(variation, bool)
        and variation >= 0.0
        and variation <= gate["maximum_slip_coefficient_of_variation"],
        "slip pilot is not reproducible",
    )
    _require(pilot.get("passed") is True, "slip pilot is not marked passed")
    _require(
        pilot.get("decided_before_confirmatory_collection") is True,
        "slip gate was evaluated after collection began",
    )


def validate_dataset(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate a complete 36-execution acquisition tree."""

    validate_protocol(protocol)
    root = Path(dataset_root)
    dataset_protocol_path = root / "protocol.json"
    schedule_path = root / "acquisition_schedule.csv"
    registration_path = root / "object_registration.json"
    slip_path = root / "slip_pilot.json"
    _require(dataset_protocol_path.is_file(), "dataset protocol.json is missing")
    _require(schedule_path.is_file(), "acquisition_schedule.csv is missing")
    _require(registration_path.is_file(), "object_registration.json is missing")
    _require(slip_path.is_file(), "slip_pilot.json is missing")
    dataset_protocol = json.loads(dataset_protocol_path.read_text(encoding="utf-8"))
    validate_protocol(dataset_protocol)
    _require(
        dataset_protocol == protocol, "dataset protocol differs from locked protocol"
    )
    with schedule_path.open(newline="", encoding="utf-8") as handle:
        schedule_rows = list(csv.DictReader(handle))
    expected_schedule_rows = [
        {field: str(row[field]) for field in _ACQUISITION_SCHEDULE_FIELDS}
        for row in _acquisition_schedule_rows(protocol)
    ]
    _require(
        schedule_rows == expected_schedule_rows,
        "acquisition schedule differs from the locked design",
    )
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    validate_object_registration(
        protocol,
        registration,
    )
    if verify_files:
        for region_id, descriptor in registration["contact_regions"].items():
            node_path = root / descriptor["canonical_node_set_path"]
            _require(node_path.is_file(), f"contact node set is missing: {region_id}")
            digest, _ = _sha256_file(node_path)
            _require(
                digest == descriptor["canonical_node_set_sha256"],
                f"contact node-set checksum mismatch: {region_id}",
            )
    validate_slip_pilot(
        protocol,
        json.loads(slip_path.read_text(encoding="utf-8")),
    )
    included = 0
    excluded = 0
    for execution in protocol["executions"]:
        execution_root = root / "executions" / execution["execution_id"]
        manifest_path = execution_root / "manifest.json"
        _require(
            manifest_path.is_file(), f"manifest is missing: {execution['execution_id']}"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_execution_manifest(
            protocol,
            manifest,
            execution_root=execution_root,
            verify_files=verify_files,
        )
        if manifest["exclusion"]["status"] == "included":
            included += 1
        else:
            excluded += 1
    return {
        "protocol_id": protocol["protocol_id"],
        "design_sha256": protocol["design_sha256"],
        "executions_checked": 36,
        "included": included,
        "excluded": excluded,
        "file_hashes_verified": verify_files,
        "passed": True,
    }
