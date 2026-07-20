"""Fail-closed artifact chain for the untouched Deform360 confirmation panel.

This module deliberately does not discover cases from a directory.  Every
pre-outcome operation is authorized by an immutable lock whose confirmation
and calibration whitelists are validated against the prospective protocol.
The artifact chain is::

    lock -> frame-zero bundle -> physical-prior seal
         -> causal-prefix authorization -> online-prediction seal
         -> complete-cohort outcome permit

The frame-zero bundle may contain the known 76-frame aligned realized robot
kinematics, but object evidence is restricted to frame zero.  Outcome callbacks
are invoked only after every prediction seal in the requested cohort has been
revalidated.
The module never constructs frame-zero assets and never accepts a target or
outcome path as a sealing input.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, TypeVar

import numpy as np

from .deform360_frame_zero_assets import (
    FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
    FRAME_ZERO_CAMERA_SELECTION_RULE,
    FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_POLICY_ID,
    FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_RULE,
)
from .deform360_frame_zero_semantic_gate import (
    FRAME_ZERO_REFERENCE_OPTIONAL_ASSIGNMENT_STRATEGY,
    FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID,
    FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY,
    FRAME_ZERO_SEMANTIC_GATE_CONTRACT,
    FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
    semantic_label_for_object_id,
    validate_semantic_gate_audit,
)
from .deform360_robot_kinematics import (
    ROBOT_KINEMATICS_WINDOW_CONTRACT,
    ROBOT_KINEMATICS_WINDOW_POLICY_ID,
    load_robot_kinematics_archive,
    robot_kinematics_array_records,
    validate_robot_kinematics_selection_audit,
    validate_selected_robot_kinematics_bundle,
)


PROTOCOL_ID = "deform360-held-online-belief-v5"
SCHEMA_VERSION = 1
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
REMOTE_INVENTORY_COMBINED_SHA256 = (
    "c974ea7e2ce217635f31201863e7dad995a26445bcc2352e7f15fe1a0b335156"
)
LOCK_KIND = "Deform360HeldOnlineBeliefLock"
FRAME_ZERO_KIND = "Deform360HeldFrameZeroBundle"
PHYSICAL_SEAL_KIND = "Deform360HeldPhysicalPriorSeal"
PREFIX_AUTHORIZATION_KIND = "Deform360HeldCausalPrefixAuthorization"
ONLINE_SEAL_KIND = "Deform360HeldOnlinePredictionSeal"
CALIBRATION_DECISION_KIND = "Deform360HeldCalibrationGateDecision"
CALIBRATION_SCORE_EVIDENCE_KIND = "Deform360HeldCalibrationScoreEvidence"
CONFIRMATION_DECISION_KIND = "Deform360HeldConfirmationDecision"
CONFIRMATION_SCORE_EVIDENCE_KIND = "Deform360HeldConfirmationScoreEvidence"

FRAME_COUNT = 76
UPDATE_FRAMES = (19, 38, 57)
SCORED_FRAME_INTERVALS_HALF_OPEN = ((20, 38), (39, 57), (58, 76))

PRIMARY_METHOD = {
    "method_id": "support-gated-selected-backbone-full-blend-euclidean-rbf",
    "backbone_selector": "current-observed-center-symmetric-chamfer",
    "update": "full-blend-euclidean-rbf",
    "minimum_support_count": 3,
    "insufficient_support_behavior": "persistence",
    "calibration_selects_method": False,
}
CONTROL_METHODS = (
    "physical_prior",
    "persistence",
    "selected_raw_backbone",
    "frozen_current_state",
    "constant_velocity",
    "independent_cpd",
    "clique_correspondence",
    "adaptive_recursive_cpd",
)
METRIC_LOCK = {
    "primary": "post_update_hidden_symmetric_chamfer_m",
    "secondary": "post_update_hidden_identity_rmse_m",
    "assimilation_centers_excluded_from_both_chamfer_directions": True,
    "assimilation_centers_excluded_from_identity_metric": True,
    "scored_frame_intervals_half_open": [
        list(value) for value in SCORED_FRAME_INTERVALS_HALF_OPEN
    ],
    "episode_aggregation": "equal_case_mean",
    "object_aggregation": "equal_object_mean",
    "comparator": "selected_raw_backbone",
    "lower_is_better": True,
}
CALIBRATION_GATE = {
    "role": "go_no_go_only",
    "method_selection_permitted": False,
    "case_count": 15,
    "minimum_equal_case_mean_chamfer_improvement_fraction": 0.05,
    "aggregate_hidden_identity_rmse_must_improve": True,
    "minimum_case_chamfer_wins": 10,
    "maximum_case_chamfer_regression_fraction": 0.10,
    "no_go_keeps_confirmation_payload_sealed": True,
}
CONFIRMATION_GATE = {
    "case_count": 6,
    "required_case_chamfer_wins": 6,
    "one_sided_sign_test_p": 1.0 / 64.0,
    "minimum_equal_case_mean_chamfer_improvement_fraction": 0.05,
    "aggregate_hidden_identity_rmse_must_improve": True,
    "maximum_case_chamfer_regression_fraction": 0.10,
    "all_cases_must_be_reported": True,
}

# This amendment records the complete lineage through the failed-closed v4
# execution, including the v3 pre-lock filename-only rg incident.  It
# deliberately does not claim
# that protected regular files were unopened: rg is a content scanner and may
# have opened any regular file under its broad search roots.  No held payload
# content or value was returned to the research agent or used to select a
# method/gate.  The three external lineage reports are supplied by checksum
# when the lock is created; those digests are not source constants that could
# silently bless different files.
SOURCE_FEASIBILITY_AMENDMENT_CONTRACT = {
    "contract_id": "deform360-held-source-feasibility-amendment-v5",
    "protocol_id": PROTOCOL_ID,
    "v1_execution": {
        "protocol_id": "deform360-held-online-belief-v1",
        "disposition": "ABANDONED_PREOUTCOME",
        "evidence_binding_key": "v1_preoutcome_feasibility_report",
        "exact_target_free_census": {
            "requested_case_count": 15,
            "sealed_case_count": 5,
            "frame_zero_failure_count": 9,
            "physical_admission_failure_count": 1,
        },
        "outcome_payloads_accessed": False,
        "target_payloads_accessed": False,
        "confirmation_payloads_accessed": False,
        "outcome_permit_created": False,
        "execution_artifacts_reused_by_v5": False,
        "predictions_reused_by_v5": False,
    },
    "v2_design": {
        "protocol_id": "deform360-held-online-belief-v2",
        "disposition": "WITHDRAWN_BEFORE_LOCK_AND_PREDICTION",
        "evidence_binding_key": "v2_design_withdrawal_report",
        "exact_execution_census": {
            "calibration_lock_count": 0,
            "case_attempt_count": 0,
            "deployed_snapshot_count": 0,
            "frame_zero_manifest_count": 0,
            "online_prediction_seal_count": 0,
            "outcome_created_count": 0,
            "outcome_permit_count": 0,
            "outcome_read_count": 0,
            "physical_prior_seal_count": 0,
            "prediction_count": 0,
            "prefix_authorization_count": 0,
            "shard_start_count": 0,
            "target_operation_count": 0,
        },
        "information_access": {
            "confirmation_payload_read": False,
            "episode_payload_read": False,
            "frame_zero_payload_read": False,
            "future_tactile_read": False,
            "outcome_read": False,
            "prediction_payload_read": False,
            "target_data_read": False,
            "target_or_outcome_path_accessed": False,
        },
        "execution_artifacts_reused_by_v5": False,
        "predictions_reused_by_v5": False,
    },
    "v3_design": {
        "protocol_id": "deform360-held-online-belief-v3",
        "disposition": "WITHDRAWN_BEFORE_LOCK_AND_PREDICTION",
        "evidence_binding_key": "v3_prelock_boundary_incident_report",
        "exact_formal_protocol_execution_census": {
            "calibration_lock_count": 0,
            "case_attempt_count": 0,
            "deployed_snapshot_count": 0,
            "deployment_count": 0,
            "frame_zero_manifest_count": 0,
            "online_prediction_seal_count": 0,
            "outcome_api_operation_count": 0,
            "outcome_created_count": 0,
            "outcome_permit_count": 0,
            "physical_prior_seal_count": 0,
            "prediction_count": 0,
            "prefix_authorization_count": 0,
            "shard_start_count": 0,
            "target_operation_count": 0,
        },
        "formal_protocol_execution_scope": (
            "canonical held-v3 pipeline and artifacts only; excludes the "
            "separately disclosed rg content scanner"
        ),
        "prelock_boundary_incident": {
            "execution_context": "SSH",
            "program": "rg",
            "mode": "-l",
            "search_terms": [
                "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834",
                "facebook/cotracker3-scaled-offline",
            ],
            "search_roots": [
                "/mnt/corsair/florianpfaff/bpt-online-belief-v1",
                "/mnt/corsair/florianpfaff/deform360-processing-deps",
                "/mnt/lexar4tb/datasets/deform360",
            ],
            "stdout_consumer": "head",
            "stdout_maximum_line_count": 100,
            "only_matching_absolute_filenames_returned": True,
            "included_unrelated_171_outcome_or_log_paths": True,
            "content_scanner_may_have_opened_any_regular_file": True,
            "protected_file_open_status": "NOT_CLAIMED",
            "payload_bytes_metrics_labels_arrays_or_values_returned": False,
            "held_cohort_payload_content_or_value_returned_to_research_agent": False,
            "method_or_gate_choice_used_outcome_values": False,
        },
        "execution_artifacts_reused_by_v5": False,
        "predictions_reused_by_v5": False,
    },
    "v4_repairs": {
        "robot_window_selection": {
            "withdrawn_v2_defect": (
                "averaged heterogeneous robot.npz:actions rows (translation, "
                "three rotation rows, and opening) into a pseudo-centre"
            ),
            "archive_fields": [
                "format_version",
                "actions",
                "T_worlds",
                "openings",
                "bimanual",
            ],
            "selection_translation": "T_worlds[..., :3, 3]",
            "selection_openings": "openings",
            "bimanual_semantics_preserved": True,
            "redundant_actions_state_parity_required": True,
            "selected_prediction_slice_frame_count": 76,
            "selected_bundle_must_replay_exact_source_slice": True,
            "camera_frame_zero_must_match_selected_action_window_start": True,
        },
        "frame_zero_geometry": {
            "ordered_reference_anchored_strategies": [
                "legacy",
                "same-masks-projected-footprint",
                "common-voxel-assignment-projected-footprint",
            ],
            "direct_common_geometry_fallback_preserved": True,
            "final_fallback": (
                "reference-conditioned-reference-optional-exhaustive-exact-eight"
            ),
            "official_current_frame_urdf_robot_exclusion_required": True,
            "pinned_siglip2_exclusive_semantic_rank_gate_required": True,
            "reference_camera_may_abstain_only_in_final_fallback": True,
            "selection_scope": "source_only_frame_zero",
            "target_or_outcome_guidance_permitted": False,
        },
        "failed_source_only_diagnostics_disclosed": [
            "faithful_groundedsam_reference_produced_zero_detections",
            "multi_reference_groundedsam_failed_object_specificity",
            "official_current_frame_urdf_exclusion_alone_did_not_remove_the_spider4_false_region",
            "bilateral_taxel_interaction_gate_failed_its_frozen_20mm_threshold",
            "first_siglip2_attempt_failed_before_logits_until_the_pinned_cublas_workspace_was_declared",
        ],
        "information_boundary": {
            "rg_content_scanner_may_have_opened_any_regular_file_under_search_roots": True,
            "protected_file_open_status": "NOT_CLAIMED",
            "held_cohort_payload_content_or_value_returned_to_research_agent": False,
            "outcome_metric_label_array_or_value_returned_to_research_agent": False,
            "method_or_gate_choice_used_outcome_values": False,
        },
    },
    "v4_execution": {
        "protocol_id": "deform360-held-online-belief-v4",
        "disposition": "WITHDRAWN_AFTER_FRAME_ZERO_BEFORE_PHYSICAL_PREDICTION",
        "evidence_binding_key": "v4_execution_withdrawal_report",
        "exact_execution_census": {
            "calibration_decision_count": 0,
            "calibration_lock_count": 1,
            "case_attempt_count": 2,
            "confirmation_lock_count": 0,
            "deployed_snapshot_count": 1,
            "deployment_count": 1,
            "frame_zero_bundle_count": 2,
            "frame_zero_manifest_count": 2,
            "formal_online_prediction_count": 0,
            "formal_physical_prediction_count": 0,
            "online_prediction_seal_count": 0,
            "outcome_api_operation_count": 0,
            "outcome_created_count": 0,
            "outcome_permit_count": 0,
            "outcome_read_count": 0,
            "physical_builder_invocation_count": 2,
            "physical_prediction_artifact_count": 0,
            "physical_prior_seal_count": 0,
            "prefix_authorization_count": 0,
            "shard_start_count": 2,
            "target_operation_count": 0,
        },
        "failure": {
            "classification": "IMMUTABLE_PYTHON_INVENTORY_MISMATCH",
            "phase": "physical-build runtime preflight",
            "failure_time_inventory_recorded": False,
            "post_failure_diagnostic_is_not_failure_time_reconstruction": True,
        },
        "information_boundary": {
            "frame_zero_source_artifacts_created": True,
            "episode_payload_read": True,
            "episode_payload_scope": (
                "frame-zero RGB-D and masks; the frame-zero pipeline read the full "
                "realized robot archive to select the window, then sealed the aligned "
                "76-frame robot-kinematics window"
            ),
            "object_future_rgb_depth_or_tracking_read": False,
            "physical_prediction_created": False,
            "online_prediction_created": False,
            "prediction_payload_read": False,
            "target_data_read": False,
            "tactile_data_read": False,
            "future_tactile_read": False,
            "confirmation_payload_read": False,
            "outcome_created_or_read": False,
        },
        "execution_artifacts_reused_by_v5": False,
        "predictions_reused_by_v5": False,
    },
    "reuse": {
        "v1_execution_artifacts_reused_by_v5": False,
        "v1_predictions_reused_by_v5": False,
        "v2_execution_artifacts_reused_by_v5": False,
        "v2_predictions_reused_by_v5": False,
        "v3_execution_artifacts_reused_by_v5": False,
        "v3_predictions_reused_by_v5": False,
        "v4_execution_artifacts_reused_by_v5": False,
        "v4_predictions_reused_by_v5": False,
        "sealed_lineage_reports_bound_by_v5": [
            "v1_preoutcome_feasibility_report",
            "v2_design_withdrawal_report",
            "v3_prelock_boundary_incident_report",
            "v4_execution_withdrawal_report",
        ],
    },
}


@dataclass(frozen=True)
class HeldCaseSpec:
    case_name: str
    object_id: str
    episode_id: int
    remote_inventory_sha256: str
    remote_file_count: int
    remote_total_bytes: int


CONFIRMATION_CASES = (
    HeldCaseSpec(
        "002-rope-silk-ep0001",
        "002-rope-silk",
        1,
        "b33791f6faa8d05717408d7b77cf1405083b614fe42ecef3a3538a0dc2008858",
        32,
        37_863_432,
    ),
    HeldCaseSpec(
        "081-stripe-rope-ep0005",
        "081-stripe-rope",
        5,
        "6055375fb66ea1e0732e808d855e4eecb66687f14dfd6a6a604d5d9a39a194e0",
        32,
        61_222_868,
    ),
    HeldCaseSpec(
        "085-scarf-cloth-ep0002",
        "085-scarf-cloth",
        2,
        "cb9ee9be4c99244e94f676a329b31ecb629c0afef9b7ffbe6060a6b061b81249",
        32,
        31_710_094,
    ),
    HeldCaseSpec(
        "083-blanket-cloth-ep0007",
        "083-blanket-cloth",
        7,
        "102f9edd98b6d3703c3d98625a358c7588d87c79024c795d233771e76b10be84",
        32,
        53_947_570,
    ),
    HeldCaseSpec(
        "092-squirrel-ep0001",
        "092-squirrel",
        1,
        "6f02afc8e8101fdc0e30ee171435162d1d6a4d648f5ee910070f711313d2b960",
        32,
        38_161_504,
    ),
    HeldCaseSpec(
        "170-spider-ep0006",
        "170-spider",
        6,
        "c19cb57b087aa98c5e792e8dfcb2e889cb4b2a52653a78a2cba6591a0fdc80a7",
        32,
        47_269_453,
    ),
)
CONFIRMATION_CASE_NAMES = tuple(case.case_name for case in CONFIRMATION_CASES)

CALIBRATION_CASE_NAMES = (
    "002-rope-silk-ep0003",
    "002-rope-silk-ep0004",
    "002-rope-silk-ep0008",
    "083-blanket-cloth-ep0000",
    "083-blanket-cloth-ep0003",
    "083-blanket-cloth-ep0006",
    "085-scarf-cloth-ep0000",
    "085-scarf-cloth-ep0005",
    "085-scarf-cloth-ep0007",
    "092-squirrel-ep0002",
    "092-squirrel-ep0003",
    "092-squirrel-ep0006",
    "170-spider-ep0002",
    "170-spider-ep0004",
    "170-spider-ep0007",
)

PHYSICAL_ARTIFACT_ROLES = (
    "prediction_only_input",
    "prediction_only_summary",
    "physical_prediction_archive",
    "physical_prediction_manifest",
)
ONLINE_ARTIFACT_ROLES = (
    "measurement_archive",
    "measurement_manifest",
    "measurement_uncertainty_archive",
    "measurement_uncertainty_manifest",
    "cycle_uncertainty_archive",
    "cycle_uncertainty_manifest",
    "online_prediction_archive",
)

# Every implementation, model, data revision, runtime, and reconstruction
# contract that can affect a held prediction or its score must be named before
# the prospective lock is created.  Requiring the exact set prevents a caller
# from silently omitting a decisive component while still supplying one
# syntactically valid checksum.
REQUIRED_IMMUTABLE_BINDING_KEYS = (
    "alltracker_checkpoint",
    "alltracker_molmomotion_revision_literal",
    "alltracker_provenance_tree_literal",
    "alltracker_runtime_tree",
    "cpd_registration_source",
    "cotracker_checkpoint",
    "cotracker_commit_object",
    "cotracker_git_tree_manifest",
    "cotracker_revision_literal",
    "dataset_revision_literal",
    "deform360_code_commit_object",
    "deform360_code_git_tree_manifest",
    "deform360_code_revision_literal",
    "deform360_dataset_containment_source",
    "deform360_hidden_metric_source",
    "deform360_object_sam2_source",
    "deform360_official_outcome_builder_source",
    "deform360_pipeline_config",
    "deform360_pipeline_config_semantic",
    "deform360_robot_kinematics_source",
    "deform360_sam2_source",
    "deform360_stage_script",
    "deform360_strict_reconstruction_source",
    "ffmpeg_executable",
    "ffmpeg_version_literal",
    "frame_zero_builder_cli",
    "frame_zero_builder_source",
    "frame_zero_default_config",
    "frame_zero_deform360_protocol_dependency",
    "frame_zero_object_sam2_source",
    "frame_zero_sam2_constants_source",
    "frame_zero_semantic_gate_contract",
    "frame_zero_semantic_gate_source",
    "frame_zero_siglip2_model_tree",
    "frame_zero_siglip2_revision_literal",
    "frame_zero_siglip2_transformers_sources",
    "frame_zero_visual_hull_source",
    "graph_residual_mapping_source",
    "held_calibration_case_runner_source",
    "held_calibration_gate_contract",
    "held_calibration_outcome_driver_source",
    "held_calibration_shard_runner_source",
    "held_confirmation_case_runner_source",
    "held_confirmation_outcome_driver_source",
    "held_confirmation_shard_runner_source",
    "held_confirmation_gate_contract",
    "held_frozen_runtime_manifest",
    "held_metric_contract",
    "held_online_runner_cli",
    "held_online_runner_source",
    "held_outcome_reconstruction_adapter_source",
    "held_outcome_scorer_source",
    "held_physical_builder_cli",
    "held_physical_builder_source",
    "held_physical_numeric_contract",
    "held_primary_method_contract",
    "held_protocol_lock_operator_source",
    "held_protocol_source",
    "held_source_feasibility_amendment_contract",
    "independent_cpd_source",
    "method_commit_object",
    "method_deployed_snapshot_tree",
    "method_git_tree_manifest",
    "method_head_literal",
    "nearest_distance_metric_source",
    "official_phystwin_commit_object",
    "official_phystwin_git_tree_manifest",
    "official_phystwin_real_config",
    "official_phystwin_revision_literal",
    "outcome_reconstruction_contract",
    "pairwise_clique_source",
    "primary_rbf_config",
    "pyproject_toml",
    "python_executable",
    "python_pip_freeze_sorted",
    "raw_cycle_cli",
    "raw_cycle_default_config",
    "raw_cycle_source",
    "raw_gated_evaluator_cli",
    "raw_gated_evaluator_source",
    "raw_observation_cli",
    "raw_observation_default_config",
    "raw_observation_source",
    "raw_uncertainty_cli",
    "raw_uncertainty_default_config",
    "raw_uncertainty_source",
    "recursive_cpd_source",
    "recursive_rbf_source",
    "remote_confirmation_inventory_combined",
    "robot_kinematics_window_contract",
    "robust_correspondence_source",
    "sam2_checkpoint",
    "sam2_commit_object",
    "sam2_git_tree_manifest",
    "sam2_model_config",
    "sam2_revision_literal",
    "upstream_action_support_source",
    "upstream_automatic_twin_builder",
    "upstream_contact_conditioned_action_source",
    "upstream_dense_reusable_panel_config",
    "upstream_dense_reusable_panel_source",
    "upstream_dense_source",
    "upstream_independent_source_split_config",
    "upstream_official_phystwin_smoke",
    "upstream_partial_graph_state_source",
    "upstream_phystwin_graph_source",
    "upstream_reusable_graph_source",
    "upstream_runtime_bundle_tree",
    "v1_preoutcome_feasibility_report",
    "v2_design_withdrawal_report",
    "v3_prelock_boundary_incident_report",
    "v4_execution_withdrawal_report",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PREOUTCOME_FILENAMES = frozenset({"outcome.json", "target_data.pkl"})
_FORBIDDEN_FRAME_ZERO_SUFFIXES = frozenset({".h5", ".hdf5"})
_OUTCOME_CAPABILITY = object()
_T = TypeVar("_T")


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


def held_artifact_sha256(artifact: Mapping[str, Any]) -> str:
    """Hash an artifact's semantic JSON content, excluding its self digest."""

    unsigned = deepcopy(dict(artifact))
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def held_contract_sha256(value: Any) -> str:
    """Hash one immutable configuration/contract using canonical JSON."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _sha256_file(path: str | Path) -> str:
    source, descriptor, opened = _open_regular_file_snapshot(path)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        _require_unchanged_open_file(source, descriptor, opened)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _open_regular_file_snapshot(
    path: str | Path,
) -> tuple[Path, int, os.stat_result]:
    """Open one canonical regular file without following its final component."""

    source = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(source)
    _require(stat.S_ISREG(before.st_mode), f"{source} is not a regular file")
    _require(
        source.resolve() == source,
        f"{source} has a symlinked or non-canonical ancestor",
    )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
            f"{source} changed while opening",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return source, descriptor, opened


def _require_unchanged_open_file(
    source: Path,
    descriptor: int,
    opened: os.stat_result,
) -> os.stat_result:
    after = os.fstat(descriptor)
    current = os.lstat(source)
    identity = (opened.st_dev, opened.st_ino)
    _require(
        (after.st_dev, after.st_ino) == identity
        and (current.st_dev, current.st_ino) == identity
        and after.st_size == opened.st_size
        and after.st_mtime_ns == opened.st_mtime_ns
        and after.st_ctime_ns == opened.st_ctime_ns,
        f"{source} changed while reading",
    )
    return after


def _load_json(path: str | Path) -> dict[str, Any]:
    source, descriptor, opened = _open_regular_file_snapshot(path)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            value = json.load(stream)
        _require_unchanged_open_file(source, descriptor, opened)
    finally:
        os.close(descriptor)
    _require(isinstance(value, dict), f"{source.name} must contain a JSON object")
    return value


def _write_new_json(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    destination = Path(os.path.abspath(os.fspath(path)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_stat = os.lstat(destination.parent)
    _require(
        stat.S_ISDIR(parent_stat.st_mode)
        and destination.parent.resolve() == destination.parent,
        f"{destination.parent} is linked or non-canonical",
    )
    payload = (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _bound_file(path: str | Path, *, frame_zero_bundle: bool = False) -> dict[str, Any]:
    source, descriptor, opened = _open_regular_file_snapshot(path)
    resolved = source
    _require(
        resolved.name not in _FORBIDDEN_PREOUTCOME_FILENAMES,
        "outcome payload cannot be a pre-outcome sealing input",
    )
    if frame_zero_bundle:
        _require(
            resolved.suffix.lower() not in _FORBIDDEN_FRAME_ZERO_SUFFIXES,
            "frame-zero bundle must be extracted, not a future-bearing HDF5 container",
        )
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = _require_unchanged_open_file(source, descriptor, opened)
    finally:
        os.close(descriptor)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": after.st_size,
    }


def _validate_bound_file(
    record: Mapping[str, Any],
    *,
    role: str,
    frame_zero_bundle: bool = False,
) -> Path:
    _require(isinstance(record, Mapping), f"{role} file record is missing")
    path_value = record.get("path")
    _require(isinstance(path_value, str) and path_value, f"{role} path is missing")
    observed = _bound_file(path_value, frame_zero_bundle=frame_zero_bundle)
    _require(observed == dict(record), f"{role} file binding changed")
    return Path(observed["path"])


def _expected_confirmation_payload() -> list[dict[str, Any]]:
    return [asdict(case) for case in CONFIRMATION_CASES]


def create_held_protocol_lock(
    output_path: str | Path,
    *,
    immutable_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Create the prospective lock without reading any episode payload."""

    bindings = {str(key): str(value) for key, value in immutable_bindings.items()}
    required = set(REQUIRED_IMMUTABLE_BINDING_KEYS)
    observed = set(bindings)
    _require(
        observed == required,
        "immutable binding keys changed; "
        f"missing={sorted(required - observed)!r}, "
        f"unexpected={sorted(observed - required)!r}",
    )
    _require(
        all(key and _valid_sha256(value) for key, value in bindings.items()),
        "every immutable implementation binding must be a named SHA-256",
    )
    _require(
        bindings["held_source_feasibility_amendment_contract"]
        == held_contract_sha256(SOURCE_FEASIBILITY_AMENDMENT_CONTRACT),
        "source-feasibility amendment contract binding changed",
    )
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": LOCK_KIND,
        "protocol_id": PROTOCOL_ID,
        "dataset_revision": DATASET_REVISION,
        "remote_inventory_combined_sha256": REMOTE_INVENTORY_COMBINED_SHA256,
        "cohort": _expected_confirmation_payload(),
        "case_whitelist": [case.case_name for case in CONFIRMATION_CASES],
        "calibration_case_whitelist": list(CALIBRATION_CASE_NAMES),
        "frame_count": FRAME_COUNT,
        "update_frames": list(UPDATE_FRAMES),
        "immutable_bindings": dict(sorted(bindings.items())),
        "primary_method": deepcopy(PRIMARY_METHOD),
        "control_methods": list(CONTROL_METHODS),
        "metric_lock": deepcopy(METRIC_LOCK),
        "calibration_gate": deepcopy(CALIBRATION_GATE),
        "confirmation_gate": deepcopy(CONFIRMATION_GATE),
        "stage": "calibration",
        "confirmation_access_authorized": False,
        "parent_calibration_lock": None,
        "calibration_gate_evidence": None,
        "information_boundary": {
            "target_payload_read_before_lock": False,
            "target_outcome_created_before_lock": False,
            "target_outcome_read_before_lock": False,
            "calibration_outcomes_read_before_lock": False,
            "filesystem_case_discovery_permitted": False,
            "all_cohort_predictions_required_before_outcome": True,
            "calibration_may_select_method": False,
        },
    }
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    _write_new_json(output_path, artifact)
    return load_held_protocol_lock(output_path)


def load_held_protocol_lock(path: str | Path) -> dict[str, Any]:
    """Load a lock and reject any deviation from the prospective design."""

    artifact = _load_json(path)
    _require(
        artifact.get("schema_version") == SCHEMA_VERSION, "unsupported held lock schema"
    )
    _require(
        artifact.get("artifact_kind") == LOCK_KIND, "unsupported held lock artifact"
    )
    _require(artifact.get("protocol_id") == PROTOCOL_ID, "held protocol id changed")
    _require(
        artifact.get("dataset_revision") == DATASET_REVISION, "dataset revision changed"
    )
    _require(
        artifact.get("remote_inventory_combined_sha256")
        == REMOTE_INVENTORY_COMBINED_SHA256,
        "remote target inventory changed",
    )
    _require(
        artifact.get("cohort") == _expected_confirmation_payload(),
        "held cohort changed",
    )
    _require(
        artifact.get("case_whitelist")
        == [case.case_name for case in CONFIRMATION_CASES],
        "confirmation whitelist changed",
    )
    _require(
        artifact.get("calibration_case_whitelist") == list(CALIBRATION_CASE_NAMES),
        "calibration whitelist changed",
    )
    _require(
        artifact.get("frame_count") == FRAME_COUNT
        and artifact.get("update_frames") == list(UPDATE_FRAMES),
        "held temporal protocol changed",
    )
    _require(artifact.get("primary_method") == PRIMARY_METHOD, "primary method changed")
    _require(
        artifact.get("control_methods") == list(CONTROL_METHODS),
        "control methods changed",
    )
    _require(artifact.get("metric_lock") == METRIC_LOCK, "metric definition changed")
    _require(
        artifact.get("calibration_gate") == CALIBRATION_GATE, "calibration gate changed"
    )
    _require(
        artifact.get("confirmation_gate") == CONFIRMATION_GATE,
        "confirmation gate changed",
    )
    stage = artifact.get("stage")
    _require(stage in {"calibration", "confirmation"}, "unsupported held lock stage")
    if stage == "calibration":
        _require(
            artifact.get("confirmation_access_authorized") is False
            and artifact.get("parent_calibration_lock") is None
            and artifact.get("calibration_gate_evidence") is None,
            "calibration lock prematurely authorizes confirmation",
        )
    else:
        _require(
            artifact.get("confirmation_access_authorized") is True,
            "confirmation lock lacks calibration GO authorization",
        )
        parent_path = _validate_bound_file(
            artifact.get("parent_calibration_lock", {}),
            role="parent calibration lock",
        )
        parent = load_held_protocol_lock(parent_path)
        _require(
            parent.get("stage") == "calibration",
            "confirmation parent is not a calibration lock",
        )
        evidence_path = _validate_bound_file(
            artifact.get("calibration_gate_evidence", {}),
            role="calibration gate evidence",
        )
        evidence = validate_calibration_gate_decision(evidence_path, parent_path)
        _require(
            evidence.get("decision") == "GO",
            "calibration gate did not authorize confirmation",
        )
        for key in (
            "dataset_revision",
            "cohort",
            "case_whitelist",
            "calibration_case_whitelist",
            "frame_count",
            "update_frames",
            "immutable_bindings",
            "primary_method",
            "control_methods",
            "metric_lock",
            "calibration_gate",
            "confirmation_gate",
        ):
            _require(
                artifact.get(key) == parent.get(key), f"confirmation lock changed {key}"
            )
    bindings = artifact.get("immutable_bindings")
    _require(
        isinstance(bindings, Mapping)
        and set(bindings) == set(REQUIRED_IMMUTABLE_BINDING_KEYS),
        "immutable binding key set changed",
    )
    _require(
        all(
            isinstance(key, str) and key and _valid_sha256(value)
            for key, value in bindings.items()
        ),
        "immutable implementation binding is invalid",
    )
    _require(
        bindings["held_source_feasibility_amendment_contract"]
        == held_contract_sha256(SOURCE_FEASIBILITY_AMENDMENT_CONTRACT),
        "source-feasibility amendment contract binding changed",
    )
    boundary = artifact.get("information_boundary", {})
    _require(
        boundary.get("target_payload_read_before_lock") is False
        and boundary.get("target_outcome_created_before_lock") is False
        and boundary.get("target_outcome_read_before_lock") is False
        and boundary.get("calibration_outcomes_read_before_lock")
        == (stage == "confirmation")
        and boundary.get("filesystem_case_discovery_permitted") is False
        and boundary.get("all_cohort_predictions_required_before_outcome") is True
        and boundary.get("calibration_may_select_method") is False,
        "held lock crossed or weakened the information boundary",
    )
    _require(
        artifact.get("artifact_sha256") == held_artifact_sha256(artifact),
        "held lock content checksum changed",
    )
    return artifact


def locked_case_names(
    path: str | Path, *, role: str = "confirmation"
) -> tuple[str, ...]:
    """Return a lock-bound whitelist; never inspect a directory for cases."""

    artifact = load_held_protocol_lock(path)
    _require(role in {"calibration", "confirmation"}, "unsupported cohort role")
    key = "case_whitelist" if role == "confirmation" else "calibration_case_whitelist"
    return tuple(str(value) for value in artifact[key])


def _authorize_case(
    lock: Mapping[str, Any], case_name: str, role: str
) -> dict[str, Any]:
    _require(role in {"calibration", "confirmation"}, "unsupported cohort role")
    if role == "confirmation":
        _require(
            lock.get("stage") == "confirmation"
            and lock.get("confirmation_access_authorized") is True,
            "confirmation remains sealed until the calibration GO lock exists",
        )
    else:
        _require(
            lock.get("stage") == "calibration",
            "calibration actions require the pre-outcome calibration lock",
        )
    key = "case_whitelist" if role == "confirmation" else "calibration_case_whitelist"
    _require(case_name in lock[key], f"case is outside the exact {role} whitelist")
    _require(
        case_name
        not in lock[
            "calibration_case_whitelist" if role == "confirmation" else "case_whitelist"
        ],
        "case has ambiguous calibration/confirmation authorization",
    )
    if role == "confirmation":
        case = next(
            value for value in CONFIRMATION_CASES if value.case_name == case_name
        )
        return {
            "case_name": case.case_name,
            "object_id": case.object_id,
            "episode_id": case.episode_id,
            "role": role,
        }
    object_id, encoded_episode = case_name.rsplit("-ep", maxsplit=1)
    return {
        "case_name": case_name,
        "object_id": object_id,
        "episode_id": int(encoded_episode),
        "role": role,
    }


_FRAME_ZERO_ACTION_ALIGNMENT_FIELDS = frozenset(
    {
        "policy_id",
        "trajectory_semantics",
        "selection_audit",
        "selected_raw_frame_range_half_open",
        "prediction_raw_frame_range_half_open",
        "tracking_tail_frame_count",
        "source_robot_frame_count",
        "prediction_frame_count",
        "selected_robot_kinematics_bundle",
        "selected_action_bundle",
        "selected_action_bundle_is_compatibility_alias",
        "selected_action_arrays",
        "selected_bundle_exact_slice_audit",
    }
)
_FRAME_ZERO_CAMERA_ACCESS_FIELDS = frozenset(
    {
        "camera",
        "path",
        "decoded_frame_count",
        "maximum_rgb_frame_read",
        "action_window_frame_index",
        "source_aligned_frame_index",
        "decoded_rgb_sha256",
        "whole_file_hashed_or_read",
    }
)
_FRAME_ZERO_CAMERA_POLICY_FIELDS = frozenset(
    {
        "policy_id",
        "rule",
        "reference_camera",
        "minimum_selected_camera_count",
        "candidate_cameras",
        "candidate_camera_count",
        "selected_cameras",
        "selected_camera_count",
        "abstained_cameras",
        "abstained_camera_count",
    }
)
# Backward-compatible private aliases used by the independent held validator
# and its tests.  The values themselves have one canonical production owner.
_FRAME_ZERO_CAMERA_SELECTION_POLICY_ID = FRAME_ZERO_CAMERA_SELECTION_POLICY_ID
_FRAME_ZERO_CAMERA_SELECTION_RULE = FRAME_ZERO_CAMERA_SELECTION_RULE


def _validated_positive_config_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    _require(
        type(value) is int and value > 0,
        f"frame-zero {key} is not a positive integer",
    )
    return value


def _reference_optional_camera_strategy(manifest: Mapping[str, Any]) -> bool:
    fallback = manifest.get("geometry_fallback")
    if not isinstance(fallback, Mapping):
        return False
    if (
        fallback.get("selected_strategy")
        != FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
    ):
        return False
    attempts = fallback.get("attempts")
    assignment = fallback.get("common_assignment")
    safeguard = fallback.get("reference_optional_safeguard")
    safeguarded_assignment = (
        safeguard.get("assignment") if isinstance(safeguard, Mapping) else None
    )
    semantic_gate = (
        safeguard.get("semantic_gate") if isinstance(safeguard, Mapping) else None
    )
    _require(
        fallback.get("policy_id")
        == FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID
        and fallback.get("ordered_strategies")
        == list(FRAME_ZERO_SEMANTIC_GATE_CONTRACT["application_order"])
        and isinstance(attempts, list)
        and bool(attempts)
        and isinstance(attempts[-1], Mapping)
        and attempts[-1].get("status") == "passed"
        and attempts[-1].get("strategy")
        == FRAME_ZERO_REFERENCE_OPTIONAL_GEOMETRY_STRATEGY
        and isinstance(assignment, Mapping)
        and assignment.get("strategy")
        == FRAME_ZERO_REFERENCE_OPTIONAL_ASSIGNMENT_STRATEGY
        and assignment.get("policy_id")
        == FRAME_ZERO_REFERENCE_OPTIONAL_FALLBACK_POLICY_ID
        and assignment == safeguarded_assignment
        and isinstance(safeguard, Mapping)
        and safeguard.get("contract_sha256")
        == FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256
        and safeguard.get("artifact_sha256") == held_artifact_sha256(safeguard)
        and isinstance(safeguard.get("official_urdf"), Mapping)
        and isinstance(safeguard.get("robot_subtraction"), Mapping)
        and isinstance(semantic_gate, Mapping),
        "reference-optional geometry strategy audit changed",
    )
    semantic_result = validate_semantic_gate_audit(semantic_gate)
    semantic_proposals = safeguard.get("semantic_selected_proposals")
    assignment_proposals = assignment.get("selected_proposals")
    semantic_selected = semantic_gate.get("selected_exact8")
    _require(
        isinstance(manifest.get("object_id"), str)
        and semantic_result["true_label"]
        == semantic_label_for_object_id(str(manifest["object_id"]))
        and isinstance(semantic_proposals, list)
        and isinstance(assignment_proposals, list)
        and isinstance(semantic_selected, list)
        and semantic_result["selected_cameras"]
        == [record.get("camera") for record in semantic_proposals]
        and [
            {
                "camera": record.get("camera"),
                "candidate_index": record.get("candidate_index"),
                "mask_sha256": record.get("mask_sha256"),
            }
            for record in semantic_proposals
        ]
        == [
            {
                "camera": record.get("camera"),
                "candidate_index": record.get("candidate_index"),
                "mask_sha256": record.get("mask_sha256"),
            }
            for record in assignment_proposals
        ]
        == [
            {
                "camera": record.get("camera"),
                "candidate_index": record.get("candidate_index"),
                "mask_sha256": record.get("selected_mask_sha256"),
            }
            for record in semantic_selected
        ],
        "semantic gate differs from held object/assignment",
    )
    return True


def _validate_frame_zero_robot_kinematics(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Independently prove the sealed robot window and camera alignment."""

    action_inputs = manifest.get("action_inputs", {})
    _require(
        isinstance(action_inputs, Mapping)
        and set(action_inputs) == {"robot_trajectory", "robot_metadata"},
        "frame-zero manifest must bind the known robot kinematics and metadata",
    )
    raw_robot_path = _validate_bound_file(
        action_inputs["robot_trajectory"], role="raw robot kinematics"
    )
    _validate_bound_file(action_inputs["robot_metadata"], role="robot metadata")
    source_state = load_robot_kinematics_archive(raw_robot_path)

    window_length = _validated_positive_config_int(
        config, "action_window_length_frames"
    )
    prediction_count = _validated_positive_config_int(
        config, "prediction_frame_count"
    )
    candidate_first_value = config.get("action_candidate_first_frame")
    _require(
        type(candidate_first_value) is int and candidate_first_value >= 0,
        "frame-zero action_candidate_first_frame is not a non-negative integer",
    )
    candidate_first = candidate_first_value
    candidate_stride = _validated_positive_config_int(
        config, "action_candidate_stride_frames"
    )
    _require(
        window_length == 81 and prediction_count == FRAME_COUNT,
        "held robot window dimensions changed",
    )

    alignment = manifest.get("action_alignment")
    _require(
        isinstance(alignment, Mapping)
        and set(alignment) == set(_FRAME_ZERO_ACTION_ALIGNMENT_FIELDS),
        "frame-zero robot alignment fields changed",
    )
    selection = alignment.get("selection_audit")
    _require(
        isinstance(selection, Mapping), "frame-zero robot selection audit is missing"
    )
    expected_selection = validate_robot_kinematics_selection_audit(
        selection,
        source_state,
        window_length_frames=window_length,
        prediction_frame_count=prediction_count,
        candidate_first_frame=candidate_first,
        candidate_stride_frames=candidate_stride,
    )
    selected_range = expected_selection["selected_raw_frame_range_half_open"]
    prediction_range = expected_selection["prediction_raw_frame_range_half_open"]
    _require(
        alignment.get("policy_id") == ROBOT_KINEMATICS_WINDOW_POLICY_ID
        and alignment.get("trajectory_semantics")
        == ROBOT_KINEMATICS_WINDOW_CONTRACT["trajectory_semantics"]
        and alignment.get("selected_raw_frame_range_half_open") == selected_range
        and alignment.get("prediction_raw_frame_range_half_open") == prediction_range
        and alignment.get("tracking_tail_frame_count")
        == window_length - prediction_count
        and alignment.get("source_robot_frame_count") == source_state.frame_count
        and alignment.get("prediction_frame_count") == prediction_count,
        "frame-zero robot selection mirrors changed",
    )

    selected_record = alignment.get("selected_robot_kinematics_bundle")
    _require(
        isinstance(selected_record, Mapping)
        and alignment.get("selected_action_bundle") == selected_record
        and alignment.get("selected_action_bundle_is_compatibility_alias") is True,
        "selected action compatibility alias changed",
    )
    selected_path = _validate_bound_file(
        selected_record, role="selected robot kinematics"
    )
    selected_state = load_robot_kinematics_archive(
        selected_path, expected_frame_count=prediction_count
    )
    _require(
        alignment.get("selected_action_arrays")
        == robot_kinematics_array_records(selected_state),
        "selected robot kinematics array bindings changed",
    )
    exact_slice = validate_selected_robot_kinematics_bundle(
        selected_state,
        source_state=source_state,
        prediction_start_frame=int(prediction_range[0]),
        prediction_frame_count=prediction_count,
    )
    _require(
        alignment.get("selected_bundle_exact_slice_audit") == exact_slice,
        "selected robot kinematics exact-slice proof changed",
    )

    bundle_path = _validate_bound_file(
        manifest.get("bundle", {}), role="frame-zero bundle", frame_zero_bundle=True
    )
    with np.load(bundle_path, allow_pickle=False) as stored:
        _require(
            "camera_names" in stored.files,
            "frame-zero bundle camera_names are missing",
        )
        bundle_cameras_array = np.asarray(stored["camera_names"])
    _require(
        bundle_cameras_array.ndim == 1
        and bundle_cameras_array.dtype.kind == "U"
        and len(bundle_cameras_array) > 0,
        "frame-zero bundle camera_names are invalid",
    )
    bundle_cameras = [str(value) for value in bundle_cameras_array.tolist()]
    _require(
        len(set(bundle_cameras)) == len(bundle_cameras),
        "frame-zero bundle camera_names are duplicated",
    )

    camera_policy = manifest.get("camera_policy")
    camera_access = manifest.get("camera_frame_zero_access")
    _require(
        isinstance(camera_policy, Mapping)
        and set(camera_policy) == set(_FRAME_ZERO_CAMERA_POLICY_FIELDS),
        "frame-zero camera policy fields changed",
    )
    candidates = camera_policy["candidate_cameras"]
    selected = camera_policy["selected_cameras"]
    abstained = camera_policy["abstained_cameras"]
    _require(
        isinstance(candidates, list)
        and isinstance(selected, list)
        and isinstance(abstained, list)
        and all(isinstance(camera, str) and camera for camera in candidates)
        and all(isinstance(camera, str) and camera for camera in selected)
        and all(isinstance(camera, str) and camera for camera in abstained)
        and candidates == sorted(set(candidates))
        and selected == sorted(set(selected))
        and abstained == sorted(set(abstained))
        and set(candidates) == set(selected) | set(abstained)
        and not set(selected) & set(abstained)
        and selected == bundle_cameras,
        "frame-zero camera partition or bundle order changed",
    )
    minimum_camera_count = config.get("minimum_camera_count")
    reference_camera = camera_policy.get("reference_camera")
    reference_optional = _reference_optional_camera_strategy(manifest)
    expected_camera_policy_id = (
        FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_POLICY_ID
        if reference_optional
        else FRAME_ZERO_CAMERA_SELECTION_POLICY_ID
    )
    expected_camera_rule = (
        FRAME_ZERO_REFERENCE_OPTIONAL_CAMERA_SELECTION_RULE
        if reference_optional
        else FRAME_ZERO_CAMERA_SELECTION_RULE
    )
    _require(
        camera_policy.get("policy_id") == expected_camera_policy_id
        and camera_policy.get("rule") == expected_camera_rule
        and type(minimum_camera_count) is int
        and minimum_camera_count >= 2
        and camera_policy.get("minimum_selected_camera_count")
        == minimum_camera_count
        and camera_policy.get("candidate_camera_count") == len(candidates)
        and camera_policy.get("selected_camera_count") == len(selected)
        and camera_policy.get("abstained_camera_count") == len(abstained)
        and len(selected) >= minimum_camera_count
        and isinstance(reference_camera, str)
        and reference_camera == config.get("reference_camera")
        and reference_camera in candidates
        and (reference_camera in selected or reference_optional)
        and (not reference_optional or len(selected) == 8),
        "frame-zero camera counts or reference policy changed",
    )
    _require(
        isinstance(camera_access, list)
        and len(camera_access) == len(candidates)
        and all(
            isinstance(record, Mapping)
            and set(record) == set(_FRAME_ZERO_CAMERA_ACCESS_FIELDS)
            for record in camera_access
        )
        and [record["camera"] for record in camera_access] == candidates,
        "frame-zero camera access audit changed",
    )
    start = int(selected_range[0])
    _require(
        all(
            record["source_aligned_frame_index"] == start
            and record["action_window_frame_index"] == 0
            and record["decoded_frame_count"] == 1
            and record["maximum_rgb_frame_read"] == 0
            and record["whole_file_hashed_or_read"] is False
            for record in camera_access
        ),
        "frame-zero camera source index differs from the robot window start",
    )


def validate_frame_zero_bundle_manifest(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Validate an externally built, single-frame object bundle.

    The bundle builder is intentionally outside this module.  Raw HDF5 masks
    and depths are rejected here because they may carry future frames even if a
    downstream consumer promises to use only index zero.
    """

    lock = load_held_protocol_lock(lock_path)
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "unsupported frame-zero schema",
    )
    _require(
        manifest.get("artifact_kind") == FRAME_ZERO_KIND,
        "unsupported frame-zero artifact",
    )
    _require(
        manifest.get("protocol_id") == PROTOCOL_ID,
        "frame-zero protocol changed",
    )
    case_name = manifest.get("case_name")
    role = manifest.get("role")
    _require(
        isinstance(case_name, str) and case_name, "frame-zero case name is missing"
    )
    _require(isinstance(role, str), "frame-zero role is missing")
    identity = _authorize_case(lock, case_name, role)
    if expected_case_name is not None:
        _require(
            case_name == expected_case_name, "frame-zero manifest binds another case"
        )
    if expected_role is not None:
        _require(role == expected_role, "frame-zero manifest binds another cohort role")
    _require(
        manifest.get("frame_indices") == [0], "frame-zero bundle contains another frame"
    )
    _require(
        manifest.get("lock_sha256") == _sha256_file(lock_path),
        "frame-zero bundle binds another lock",
    )
    _require(
        manifest.get("lock_artifact_sha256") == lock["artifact_sha256"],
        "frame-zero bundle binds another lock artifact",
    )
    config = manifest.get("config")
    _require(isinstance(config, Mapping), "frame-zero configuration is missing")
    _require(
        held_contract_sha256(config)
        == lock["immutable_bindings"]["frame_zero_default_config"],
        "frame-zero configuration changed from the immutable lock",
    )
    for key in ("object_id", "episode_id"):
        _require(manifest.get(key) == identity[key], f"frame-zero {key} changed")
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("maximum_object_rgb_frame_read") == 0
        and boundary.get("object_observation_frames_used") == [0]
        and boundary.get("known_aligned_realized_robot_kinematics_read") is True
        and boundary.get("known_robot_trajectory_semantics")
        == ROBOT_KINEMATICS_WINDOW_CONTRACT["trajectory_semantics"]
        and boundary.get("robot_delta_command_read") is False
        and boundary.get("commanded_control_read") is False
        and boundary.get("known_future_robot_action_read") is True
        and boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_depth_or_mask_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("outcome_created") is False
        and boundary.get("outcome_read") is False
        and boundary.get("whole_future_container_hashed_or_read") is False,
        "frame-zero bundle crossed the object-future boundary",
    )
    _validate_bound_file(
        manifest.get("bundle", {}), role="frame-zero bundle", frame_zero_bundle=True
    )
    _validate_frame_zero_robot_kinematics(manifest, config)
    _require(
        manifest.get("artifact_sha256") == held_artifact_sha256(manifest),
        "frame-zero manifest content checksum changed",
    )
    return manifest


def create_physical_prior_seal(
    output_path: str | Path,
    lock_path: str | Path,
    frame_zero_manifest_path: str | Path,
    physical_artifacts: Mapping[str, str | Path],
    *,
    case_name: str,
    role: str = "confirmation",
) -> dict[str, Any]:
    """Seal a frame-zero physical forecast before any object RGB frame > 0."""

    lock = load_held_protocol_lock(lock_path)
    identity = _authorize_case(lock, case_name, role)
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    _require(
        set(physical_artifacts) == set(PHYSICAL_ARTIFACT_ROLES),
        "physical artifact roles differ from the prediction-only contract",
    )
    bound_artifacts = {
        artifact_role: _bound_file(physical_artifacts[artifact_role])
        for artifact_role in PHYSICAL_ARTIFACT_ROLES
    }
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PHYSICAL_SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        **identity,
        "lock": _bound_file(lock_path),
        "frame_zero_manifest": _bound_file(frame_zero_manifest_path),
        "frame_zero_manifest_artifact_sha256": frame_zero["artifact_sha256"],
        "physical_artifacts": bound_artifacts,
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "physical_prior_sealed_before_rgb_frame_gt_zero": True,
        },
    }
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    _write_new_json(output_path, artifact)
    return validate_physical_prior_seal(output_path, lock_path)


def validate_physical_prior_seal(
    seal_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    lock = load_held_protocol_lock(lock_path)
    seal = _load_json(seal_path)
    _require(
        seal.get("schema_version") == SCHEMA_VERSION, "unsupported physical seal schema"
    )
    _require(
        seal.get("artifact_kind") == PHYSICAL_SEAL_KIND, "unsupported physical seal"
    )
    _require(seal.get("protocol_id") == PROTOCOL_ID, "physical seal protocol changed")
    identity = _authorize_case(lock, str(seal.get("case_name")), str(seal.get("role")))
    for key, value in identity.items():
        _require(seal.get(key) == value, f"physical seal {key} changed")
    if expected_case_name is not None:
        _require(
            seal["case_name"] == expected_case_name, "physical seal binds another case"
        )
    if expected_role is not None:
        _require(
            seal["role"] == expected_role, "physical seal binds another cohort role"
        )
    _require(
        _validate_bound_file(seal.get("lock", {}), role="held lock")
        == Path(lock_path).resolve(),
        "physical seal binds another lock path",
    )
    _require(
        seal["lock"]["sha256"] == _sha256_file(lock_path),
        "physical seal binds another lock",
    )
    frame_zero_path = _validate_bound_file(
        seal.get("frame_zero_manifest", {}), role="frame-zero manifest"
    )
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_path,
        lock_path,
        expected_case_name=seal["case_name"],
        expected_role=seal["role"],
    )
    _require(
        seal.get("frame_zero_manifest_artifact_sha256")
        == frame_zero["artifact_sha256"],
        "physical seal binds another frame-zero artifact",
    )
    physical = seal.get("physical_artifacts", {})
    _require(
        isinstance(physical, Mapping) and set(physical) == set(PHYSICAL_ARTIFACT_ROLES),
        "physical artifact set changed",
    )
    for artifact_role, record in physical.items():
        _validate_bound_file(record, role=artifact_role)
    boundary = seal.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("known_future_robot_action_read") is True
        and boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("outcome_created") is False
        and boundary.get("outcome_read") is False
        and boundary.get("physical_prior_sealed_before_rgb_frame_gt_zero") is True,
        "physical seal crossed the pre-prefix boundary",
    )
    _require(
        seal.get("artifact_sha256") == held_artifact_sha256(seal),
        "physical seal content checksum changed",
    )
    return seal


def create_prefix_stage_authorization(
    output_path: str | Path,
    lock_path: str | Path,
    physical_seal_path: str | Path,
) -> dict[str, Any]:
    """Authorize causal RGB prefixes only after the physical seal exists."""

    physical = validate_physical_prior_seal(physical_seal_path, lock_path)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PREFIX_AUTHORIZATION_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": physical["case_name"],
        "object_id": physical["object_id"],
        "episode_id": physical["episode_id"],
        "role": physical["role"],
        "lock": _bound_file(lock_path),
        "physical_prior_seal": _bound_file(physical_seal_path),
        "physical_prior_artifact_sha256": physical["artifact_sha256"],
        "permitted_update_frames": list(UPDATE_FRAMES),
        "maximum_object_rgb_frame_permitted": UPDATE_FRAMES[-1],
        "information_boundary": {
            "physical_prior_validated_before_prefix_authorization": True,
            "causal_prefix_only": True,
            "future_tactile_permitted": False,
            "outcome_creation_permitted": False,
            "outcome_read_permitted": False,
        },
    }
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    _write_new_json(output_path, artifact)
    return validate_prefix_stage_authorization(output_path, lock_path)


def validate_prefix_stage_authorization(
    authorization_path: str | Path,
    lock_path: str | Path,
) -> dict[str, Any]:
    authorization = _load_json(authorization_path)
    _require(
        authorization.get("schema_version") == SCHEMA_VERSION,
        "unsupported prefix authorization schema",
    )
    _require(
        authorization.get("artifact_kind") == PREFIX_AUTHORIZATION_KIND,
        "unsupported prefix authorization",
    )
    _require(
        authorization.get("protocol_id") == PROTOCOL_ID,
        "prefix authorization protocol changed",
    )
    _require(
        _validate_bound_file(authorization.get("lock", {}), role="held lock")
        == Path(lock_path).resolve(),
        "prefix authorization binds another lock path",
    )
    physical_path = _validate_bound_file(
        authorization.get("physical_prior_seal", {}), role="physical-prior seal"
    )
    physical = validate_physical_prior_seal(
        physical_path,
        lock_path,
        expected_case_name=str(authorization.get("case_name")),
        expected_role=str(authorization.get("role")),
    )
    for key in ("case_name", "object_id", "episode_id", "role"):
        _require(
            authorization.get(key) == physical.get(key),
            f"prefix authorization {key} changed",
        )
    _require(
        authorization.get("physical_prior_artifact_sha256")
        == physical["artifact_sha256"],
        "prefix authorization binds another physical artifact",
    )
    _require(
        authorization.get("permitted_update_frames") == list(UPDATE_FRAMES)
        and authorization.get("maximum_object_rgb_frame_permitted")
        == UPDATE_FRAMES[-1],
        "prefix authorization temporal boundary changed",
    )
    boundary = authorization.get("information_boundary", {})
    _require(
        boundary.get("physical_prior_validated_before_prefix_authorization") is True
        and boundary.get("causal_prefix_only") is True
        and boundary.get("future_tactile_permitted") is False
        and boundary.get("outcome_creation_permitted") is False
        and boundary.get("outcome_read_permitted") is False,
        "prefix authorization permits forbidden evidence",
    )
    _require(
        authorization.get("artifact_sha256") == held_artifact_sha256(authorization),
        "prefix authorization content checksum changed",
    )
    return authorization


def create_online_prediction_seal(
    output_path: str | Path,
    lock_path: str | Path,
    prefix_authorization_path: str | Path,
    online_artifacts: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Seal every frozen online artifact without accepting an outcome input."""

    authorization = validate_prefix_stage_authorization(
        prefix_authorization_path, lock_path
    )
    _require(
        set(online_artifacts) == set(ONLINE_ARTIFACT_ROLES),
        "online artifact roles differ from the frozen contract",
    )
    bound_artifacts = {
        artifact_role: _bound_file(online_artifacts[artifact_role])
        for artifact_role in ONLINE_ARTIFACT_ROLES
    }
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ONLINE_SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": authorization["case_name"],
        "object_id": authorization["object_id"],
        "episode_id": authorization["episode_id"],
        "role": authorization["role"],
        "lock": _bound_file(lock_path),
        "prefix_authorization": _bound_file(prefix_authorization_path),
        "prefix_authorization_artifact_sha256": authorization["artifact_sha256"],
        "online_artifacts": bound_artifacts,
        "primary_method": deepcopy(PRIMARY_METHOD),
        "information_boundary": {
            "maximum_object_rgb_frame_read": UPDATE_FRAMES[-1],
            "object_rgb_frames_read_as_causal_prefixes": True,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "all_frozen_predictions_hashed_before_outcome": True,
        },
    }
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    _write_new_json(output_path, artifact)
    return validate_online_prediction_seal(output_path, lock_path)


def validate_online_prediction_seal(
    seal_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    seal = _load_json(seal_path)
    _require(
        seal.get("schema_version") == SCHEMA_VERSION, "unsupported online seal schema"
    )
    _require(
        seal.get("artifact_kind") == ONLINE_SEAL_KIND,
        "unsupported online prediction seal",
    )
    _require(
        seal.get("protocol_id") == PROTOCOL_ID, "online prediction protocol changed"
    )
    _require(
        _validate_bound_file(seal.get("lock", {}), role="held lock")
        == Path(lock_path).resolve(),
        "online seal binds another lock path",
    )
    authorization_path = _validate_bound_file(
        seal.get("prefix_authorization", {}), role="prefix authorization"
    )
    authorization = validate_prefix_stage_authorization(authorization_path, lock_path)
    for key in ("case_name", "object_id", "episode_id", "role"):
        _require(seal.get(key) == authorization.get(key), f"online seal {key} changed")
    if expected_case_name is not None:
        _require(
            seal["case_name"] == expected_case_name, "online seal binds another case"
        )
    if expected_role is not None:
        _require(seal["role"] == expected_role, "online seal binds another cohort role")
    _require(
        seal.get("prefix_authorization_artifact_sha256")
        == authorization["artifact_sha256"],
        "online seal binds another prefix authorization",
    )
    _require(
        seal.get("primary_method") == PRIMARY_METHOD,
        "online seal primary method changed",
    )
    online = seal.get("online_artifacts", {})
    _require(
        isinstance(online, Mapping) and set(online) == set(ONLINE_ARTIFACT_ROLES),
        "online artifact set changed",
    )
    for artifact_role, record in online.items():
        _validate_bound_file(record, role=artifact_role)
    boundary = seal.get("information_boundary", {})
    _require(
        boundary.get("maximum_object_rgb_frame_read") == UPDATE_FRAMES[-1]
        and boundary.get("object_rgb_frames_read_as_causal_prefixes") is True
        and boundary.get("future_tactile_read") is False
        and boundary.get("outcome_created") is False
        and boundary.get("outcome_read") is False
        and boundary.get("all_frozen_predictions_hashed_before_outcome") is True,
        "online prediction seal crossed the outcome boundary",
    )
    _require(
        seal.get("artifact_sha256") == held_artifact_sha256(seal),
        "online seal content checksum changed",
    )
    return seal


@dataclass(frozen=True)
class OutcomePhasePermit:
    """In-process capability produced only by a complete-cohort validation."""

    lock_path: str
    role: str
    seal_paths: tuple[tuple[str, str], ...]
    cohort_barrier_sha256: str
    _capability: object = field(repr=False, compare=False)


def _validate_cohort_barrier(
    lock_path: str | Path,
    online_seal_paths: Mapping[str, str | Path],
    *,
    role: str,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    expected = locked_case_names(lock_path, role=role)
    _require(
        set(online_seal_paths) == set(expected),
        f"{role} outcome remains sealed until every exact cohort seal is present",
    )
    records: list[dict[str, str]] = []
    normalized: list[tuple[str, str]] = []
    for case_name in expected:
        path = Path(online_seal_paths[case_name]).resolve()
        seal = validate_online_prediction_seal(
            path,
            lock_path,
            expected_case_name=case_name,
            expected_role=role,
        )
        records.append(
            {
                "case_name": case_name,
                "seal_file_sha256": _sha256_file(path),
                "seal_artifact_sha256": str(seal["artifact_sha256"]),
            }
        )
        normalized.append((case_name, str(path)))
    barrier = {
        "protocol_id": PROTOCOL_ID,
        "lock_file_sha256": _sha256_file(lock_path),
        "role": role,
        "ordered_online_prediction_seals": records,
        "complete_cohort": True,
    }
    return hashlib.sha256(_canonical_bytes(barrier)).hexdigest(), tuple(normalized)


def authorize_outcome_phase(
    lock_path: str | Path,
    online_seal_paths: Mapping[str, str | Path],
    *,
    role: str = "confirmation",
) -> OutcomePhasePermit:
    """Return an outcome capability only after all cohort seals validate."""

    barrier_sha256, normalized = _validate_cohort_barrier(
        lock_path,
        online_seal_paths,
        role=role,
    )
    return OutcomePhasePermit(
        lock_path=str(Path(lock_path).resolve()),
        role=role,
        seal_paths=normalized,
        cohort_barrier_sha256=barrier_sha256,
        _capability=_OUTCOME_CAPABILITY,
    )


def _revalidate_outcome_permit(
    permit: OutcomePhasePermit,
) -> dict[str, str]:
    _require(
        isinstance(permit, OutcomePhasePermit)
        and permit._capability is _OUTCOME_CAPABILITY,
        "outcome operation lacks a cohort capability",
    )
    seal_paths = dict(permit.seal_paths)
    observed_sha256, normalized = _validate_cohort_barrier(
        permit.lock_path,
        seal_paths,
        role=permit.role,
    )
    _require(normalized == permit.seal_paths, "outcome seal path set changed")
    _require(
        observed_sha256 == permit.cohort_barrier_sha256,
        "cohort prediction barrier changed after authorization",
    )
    return seal_paths


_CALIBRATION_SCORE_KEYS = (
    "primary_chamfer_m",
    "comparator_chamfer_m",
    "primary_identity_rmse_m",
    "comparator_identity_rmse_m",
)


def _calibration_gate_summary(
    scores: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    expected = set(CALIBRATION_CASE_NAMES)
    _require(
        set(scores) == expected, "calibration scores must contain all 15 locked cases"
    )
    normalized: dict[str, dict[str, float]] = {}
    for case_name in CALIBRATION_CASE_NAMES:
        record = scores[case_name]
        _require(
            isinstance(record, Mapping) and set(record) == set(_CALIBRATION_SCORE_KEYS),
            f"calibration score fields changed for {case_name}",
        )
        values = {key: float(record[key]) for key in _CALIBRATION_SCORE_KEYS}
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in values.values()),
            f"calibration scores are invalid for {case_name}",
        )
        normalized[case_name] = values
    primary_chamfer = [
        normalized[case]["primary_chamfer_m"] for case in CALIBRATION_CASE_NAMES
    ]
    comparator_chamfer = [
        normalized[case]["comparator_chamfer_m"] for case in CALIBRATION_CASE_NAMES
    ]
    primary_identity = [
        normalized[case]["primary_identity_rmse_m"] for case in CALIBRATION_CASE_NAMES
    ]
    comparator_identity = [
        normalized[case]["comparator_identity_rmse_m"]
        for case in CALIBRATION_CASE_NAMES
    ]
    comparator_chamfer_mean = sum(comparator_chamfer) / len(comparator_chamfer)
    _require(
        comparator_chamfer_mean > 0.0,
        "calibration comparator mean Chamfer must be positive",
    )
    primary_chamfer_mean = sum(primary_chamfer) / len(primary_chamfer)
    primary_identity_mean = sum(primary_identity) / len(primary_identity)
    comparator_identity_mean = sum(comparator_identity) / len(comparator_identity)
    chamfer_improvement = (
        comparator_chamfer_mean - primary_chamfer_mean
    ) / comparator_chamfer_mean
    wins = sum(
        primary < comparator
        for primary, comparator in zip(primary_chamfer, comparator_chamfer, strict=True)
    )
    no_large_regression = all(
        primary
        <= (1.0 + CALIBRATION_GATE["maximum_case_chamfer_regression_fraction"])
        * comparator
        for primary, comparator in zip(primary_chamfer, comparator_chamfer, strict=True)
    )
    checks = {
        "mean_chamfer_improvement_at_least_5_percent": (
            chamfer_improvement
            >= CALIBRATION_GATE["minimum_equal_case_mean_chamfer_improvement_fraction"]
        ),
        "aggregate_identity_improves": primary_identity_mean < comparator_identity_mean,
        "at_least_10_of_15_chamfer_wins": wins
        >= CALIBRATION_GATE["minimum_case_chamfer_wins"],
        "no_case_over_10_percent_chamfer_regression": no_large_regression,
    }
    summary = {
        "primary_equal_case_mean_chamfer_m": primary_chamfer_mean,
        "comparator_equal_case_mean_chamfer_m": comparator_chamfer_mean,
        "equal_case_mean_chamfer_improvement_fraction": chamfer_improvement,
        "primary_equal_case_mean_identity_rmse_m": primary_identity_mean,
        "comparator_equal_case_mean_identity_rmse_m": comparator_identity_mean,
        "case_chamfer_wins": wins,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return normalized, summary


def _confirmation_gate_summary(
    scores: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Apply the exact frozen six-case confirmation arithmetic."""

    expected = set(CONFIRMATION_CASE_NAMES)
    _require(
        set(scores) == expected,
        "confirmation scores must contain all six locked cases",
    )
    normalized: dict[str, dict[str, float]] = {}
    for case_name in CONFIRMATION_CASE_NAMES:
        record = scores[case_name]
        _require(
            isinstance(record, Mapping)
            and set(record) == set(_CALIBRATION_SCORE_KEYS),
            f"confirmation score fields changed for {case_name}",
        )
        values = {key: float(record[key]) for key in _CALIBRATION_SCORE_KEYS}
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in values.values()),
            f"confirmation scores are invalid for {case_name}",
        )
        normalized[case_name] = values

    primary_chamfer = [
        normalized[case]["primary_chamfer_m"] for case in CONFIRMATION_CASE_NAMES
    ]
    comparator_chamfer = [
        normalized[case]["comparator_chamfer_m"]
        for case in CONFIRMATION_CASE_NAMES
    ]
    primary_identity = [
        normalized[case]["primary_identity_rmse_m"]
        for case in CONFIRMATION_CASE_NAMES
    ]
    comparator_identity = [
        normalized[case]["comparator_identity_rmse_m"]
        for case in CONFIRMATION_CASE_NAMES
    ]
    count = len(CONFIRMATION_CASE_NAMES)
    _require(
        count == CONFIRMATION_GATE["case_count"] == 6,
        "confirmation gate case count changed",
    )
    comparator_chamfer_mean = math.fsum(comparator_chamfer) / count
    _require(
        comparator_chamfer_mean > 0.0,
        "confirmation comparator mean Chamfer must be positive",
    )
    primary_chamfer_mean = math.fsum(primary_chamfer) / count
    primary_identity_mean = math.fsum(primary_identity) / count
    comparator_identity_mean = math.fsum(comparator_identity) / count
    chamfer_improvement = (
        comparator_chamfer_mean - primary_chamfer_mean
    ) / comparator_chamfer_mean
    wins = sum(
        primary < comparator
        for primary, comparator in zip(primary_chamfer, comparator_chamfer, strict=True)
    )
    sign_test_p = math.fsum(
        math.comb(count, successes) for successes in range(wins, count + 1)
    ) / (2**count)
    no_large_regression = all(
        primary
        <= (1.0 + CONFIRMATION_GATE["maximum_case_chamfer_regression_fraction"])
        * comparator
        for primary, comparator in zip(primary_chamfer, comparator_chamfer, strict=True)
    )
    checks = {
        "all_6_cases_chamfer_win": wins
        == CONFIRMATION_GATE["required_case_chamfer_wins"],
        "one_sided_sign_test_p_is_1_over_64": sign_test_p
        == CONFIRMATION_GATE["one_sided_sign_test_p"],
        "mean_chamfer_improvement_at_least_5_percent": (
            chamfer_improvement
            >= CONFIRMATION_GATE[
                "minimum_equal_case_mean_chamfer_improvement_fraction"
            ]
        ),
        "aggregate_identity_improves": primary_identity_mean
        < comparator_identity_mean,
        "no_case_over_10_percent_chamfer_regression": no_large_regression,
    }
    summary = {
        "primary_equal_case_mean_chamfer_m": primary_chamfer_mean,
        "comparator_equal_case_mean_chamfer_m": comparator_chamfer_mean,
        "equal_case_mean_chamfer_improvement_fraction": chamfer_improvement,
        "primary_equal_case_mean_identity_rmse_m": primary_identity_mean,
        "comparator_equal_case_mean_identity_rmse_m": comparator_identity_mean,
        "case_chamfer_wins": wins,
        "one_sided_sign_test_p": sign_test_p,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return normalized, summary


_SCORE_RECORD_KEYS = frozenset(
    {
        "case_name",
        "gate_score",
        "scored_frames",
        "permanently_excluded_center_ids",
        "identity_transport",
        "scores",
        "sealed_inputs",
        "outcome_provenance",
        "method_selection_or_tuning_performed",
    }
)
_DETAILED_SCORE_KEYS = frozenset(
    {
        "frame_count",
        "scored_frames",
        "permanently_excluded_center_count",
        "post_update_hidden_identity_rmse_m",
        "post_update_hidden_symmetric_chamfer_m",
        "hidden_identity_count_per_frame",
        "by_frame",
    }
)
_TRANSPORT_KEYS = frozenset(
    {
        "algorithm",
        "scipy_version",
        "maximum_assignment_distance_m",
        "candidate_edge_count",
        "sealed_point_coverage_fraction",
        "assigned_official_identity_collision_count",
        "assigned_official_identity_count",
        "official_identity_count",
        "mean_assignment_distance_m",
        "p95_assignment_distance_m",
        "observed_maximum_assignment_distance_m",
        "assignment_ids_sha256",
        "assignment_distances_sha256",
        "eligible_official_frame_zero_identity_count",
        "official_identity_ids_sha256",
        "raw_official_frame_zero_sha256",
        "sealed_frame_zero_sha256",
        "transported_frame_zero_replaced_with_sealed_identity",
        "claim_limitation",
    }
)
_OUTCOME_PROVENANCE_KEYS = frozenset(
    {
        "target_artifact_kind",
        "outcome_artifact_kind",
        "case_name",
        "object_id",
        "episode_id",
        "dataset_revision",
        "cohort_barrier_sha256",
        "target_file",
        "outcome_file",
        "array_sha256",
        "information_boundary",
    }
)


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validate_detailed_score(
    value: object,
    *,
    scored_frames: list[int],
    center_count: int,
    label: str,
) -> Mapping[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == set(_DETAILED_SCORE_KEYS),
        f"{label} detailed score schema changed",
    )
    _require(
        value.get("frame_count") == len(scored_frames)
        and value.get("scored_frames") == scored_frames
        and value.get("permanently_excluded_center_count") == center_count,
        f"{label} detailed score protocol changed",
    )
    identity = value.get("post_update_hidden_identity_rmse_m")
    chamfer = value.get("post_update_hidden_symmetric_chamfer_m")
    _require(
        _finite_nonnegative(identity) and _finite_nonnegative(chamfer),
        f"{label} aggregate score is invalid",
    )
    by_frame = value.get("by_frame")
    _require(
        isinstance(by_frame, Mapping)
        and set(by_frame) == {"hidden_identity_rmse_m", "hidden_symmetric_chamfer_m"},
        f"{label} per-frame score schema changed",
    )
    identity_frames = by_frame["hidden_identity_rmse_m"]
    chamfer_frames = by_frame["hidden_symmetric_chamfer_m"]
    _require(
        isinstance(identity_frames, list)
        and isinstance(chamfer_frames, list)
        and len(identity_frames) == len(scored_frames)
        and len(chamfer_frames) == len(scored_frames)
        and all(_finite_nonnegative(item) for item in identity_frames)
        and all(_finite_nonnegative(item) for item in chamfer_frames),
        f"{label} per-frame scores are invalid",
    )
    _require(
        math.isclose(
            float(identity),
            math.fsum(float(item) for item in identity_frames) / len(identity_frames),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(chamfer),
            math.fsum(float(item) for item in chamfer_frames) / len(chamfer_frames),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        f"{label} aggregate differs from its per-frame evidence",
    )
    counts = value.get("hidden_identity_count_per_frame")
    _require(
        isinstance(counts, Mapping)
        and set(counts) == {"minimum", "mean", "maximum"}
        and isinstance(counts.get("minimum"), int)
        and not isinstance(counts.get("minimum"), bool)
        and isinstance(counts.get("maximum"), int)
        and not isinstance(counts.get("maximum"), bool)
        and _finite_nonnegative(counts.get("mean"))
        and 0 < counts["minimum"] <= counts["mean"] <= counts["maximum"],
        f"{label} hidden-identity support summary is invalid",
    )
    return value


def _validate_case_score_record(
    record: object,
    permit: OutcomePhasePermit,
    case_name: str,
    *,
    expected_score_fields: set[str],
    expected_role: str,
) -> None:
    _require(
        isinstance(record, Mapping) and set(record) == set(_SCORE_RECORD_KEYS),
        f"invalid exact score record schema for {case_name}",
    )
    _require(record.get("case_name") == case_name, "score evidence case changed")
    _require(
        record.get("method_selection_or_tuning_performed") is False,
        "score evidence performed method selection",
    )
    scored_frames = [
        frame
        for start, stop in SCORED_FRAME_INTERVALS_HALF_OPEN
        for frame in range(start, stop)
    ]
    _require(
        record.get("scored_frames") == scored_frames,
        f"score evidence frames changed for {case_name}",
    )
    centers = record.get("permanently_excluded_center_ids")
    _require(
        isinstance(centers, list)
        and bool(centers)
        and all(
            isinstance(item, int) and not isinstance(item, bool) for item in centers
        )
        and len(set(centers)) == len(centers)
        and min(centers) >= 0,
        f"score evidence center IDs are invalid for {case_name}",
    )
    gate_score = record.get("gate_score")
    _require(
        isinstance(gate_score, Mapping)
        and set(gate_score) == expected_score_fields
        and all(_finite_nonnegative(value) for value in gate_score.values()),
        f"invalid gate score evidence for {case_name}",
    )
    detailed = record.get("scores")
    _require(
        isinstance(detailed, Mapping)
        and set(detailed) == {"primary", "selected_raw_backbone"},
        f"score evidence detailed methods changed for {case_name}",
    )
    primary = _validate_detailed_score(
        detailed["primary"],
        scored_frames=scored_frames,
        center_count=len(centers),
        label=f"{case_name} primary",
    )
    comparator = _validate_detailed_score(
        detailed["selected_raw_backbone"],
        scored_frames=scored_frames,
        center_count=len(centers),
        label=f"{case_name} comparator",
    )
    _require(
        gate_score
        == {
            "primary_chamfer_m": primary["post_update_hidden_symmetric_chamfer_m"],
            "comparator_chamfer_m": comparator[
                "post_update_hidden_symmetric_chamfer_m"
            ],
            "primary_identity_rmse_m": primary["post_update_hidden_identity_rmse_m"],
            "comparator_identity_rmse_m": comparator[
                "post_update_hidden_identity_rmse_m"
            ],
        },
        f"{case_name} gate score differs from detailed metric evidence",
    )
    transport = record.get("identity_transport")
    _require(
        isinstance(transport, Mapping) and set(transport) == set(_TRANSPORT_KEYS),
        f"{case_name} identity-transport schema changed",
    )
    assigned_count = transport.get("assigned_official_identity_count")
    _require(
        transport.get("algorithm")
        == "scipy-sparse-minimum-weight-full-bipartite-matching"
        and isinstance(transport.get("scipy_version"), str)
        and bool(transport.get("scipy_version"))
        and transport.get("maximum_assignment_distance_m") == 0.015
        and isinstance(transport.get("candidate_edge_count"), int)
        and transport["candidate_edge_count"] > 0
        and transport.get("sealed_point_coverage_fraction") == 1.0
        and transport.get("assigned_official_identity_collision_count") == 0
        and isinstance(assigned_count, int)
        and assigned_count > max(centers)
        and isinstance(transport.get("official_identity_count"), int)
        and transport["official_identity_count"] >= assigned_count
        and isinstance(
            transport.get("eligible_official_frame_zero_identity_count"), int
        )
        and transport["eligible_official_frame_zero_identity_count"] >= assigned_count
        and all(
            _finite_nonnegative(transport.get(key))
            for key in (
                "mean_assignment_distance_m",
                "p95_assignment_distance_m",
                "observed_maximum_assignment_distance_m",
            )
        )
        and transport["observed_maximum_assignment_distance_m"] <= 0.015
        and all(
            _valid_sha256(transport.get(key))
            for key in (
                "assignment_ids_sha256",
                "assignment_distances_sha256",
                "official_identity_ids_sha256",
                "raw_official_frame_zero_sha256",
                "sealed_frame_zero_sha256",
            )
        )
        and transport.get("transported_frame_zero_replaced_with_sealed_identity")
        is True
        and transport.get("claim_limitation")
        == (
            "one-to-one transported official reconstruction proxy; not native "
            "material identity and not parity with the Deform360 Tables 3-5 "
            "world-model benchmarks "
            "or their native tactile-refined material identities"
        ),
        f"{case_name} identity-transport evidence is invalid",
    )

    seal_path = Path(dict(permit.seal_paths)[case_name]).resolve()
    seal = validate_online_prediction_seal(
        seal_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role=expected_role,
    )
    authorization_path = _validate_bound_file(
        seal["prefix_authorization"], role=f"{case_name} prefix authorization"
    )
    authorization = validate_prefix_stage_authorization(
        authorization_path, permit.lock_path
    )
    physical_path = _validate_bound_file(
        authorization["physical_prior_seal"], role=f"{case_name} physical seal"
    )
    physical = validate_physical_prior_seal(
        physical_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role=expected_role,
    )
    frame_zero_path = _validate_bound_file(
        physical["frame_zero_manifest"], role=f"{case_name} frame-zero manifest"
    )
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role=expected_role,
    )
    expected_sealed_inputs = {
        "online_prediction_seal": _bound_file(seal_path),
        "online_prediction_archive": seal["online_artifacts"][
            "online_prediction_archive"
        ],
        "physical_prediction_archive": physical["physical_artifacts"][
            "physical_prediction_archive"
        ],
        "frame_zero_bundle": frame_zero["bundle"],
    }
    _require(
        record.get("sealed_inputs") == expected_sealed_inputs,
        f"{case_name} score evidence binds different sealed inputs",
    )

    provenance = record.get("outcome_provenance")
    _require(
        isinstance(provenance, Mapping)
        and set(provenance) == set(_OUTCOME_PROVENANCE_KEYS),
        f"{case_name} outcome provenance schema changed",
    )
    object_id, encoded_episode = case_name.rsplit("-ep", maxsplit=1)
    _require(
        provenance.get("target_artifact_kind")
        == "Deform360OfficialReconstructionTarget"
        and provenance.get("outcome_artifact_kind") == "Deform360HeldOfficialOutcome"
        and provenance.get("case_name") == case_name
        and provenance.get("object_id") == object_id
        and provenance.get("episode_id") == int(encoded_episode)
        and provenance.get("dataset_revision") == DATASET_REVISION
        and provenance.get("cohort_barrier_sha256") == permit.cohort_barrier_sha256,
        f"{case_name} outcome provenance identity changed",
    )
    _validate_bound_file(provenance.get("target_file", {}), role="official target")
    _validate_bound_file(provenance.get("outcome_file", {}), role="held outcome")
    array_sha256 = provenance.get("array_sha256")
    _require(
        isinstance(array_sha256, Mapping)
        and set(array_sha256)
        == {"object_points", "object_visibilities", "object_motions_valid"}
        and all(_valid_sha256(value) for value in array_sha256.values()),
        f"{case_name} target array bindings changed",
    )
    outcome_boundary = provenance.get("information_boundary")
    _require(
        outcome_boundary
        == {
            "complete_cohort_barrier_validated_before_future_open": True,
            "official_target_constructed_or_read_after_barrier": True,
            "prediction_metric_computed_during_target_construction": False,
        },
        f"{case_name} outcome provenance crossed the information boundary",
    )


def validate_calibration_score_evidence(
    evidence_path: str | Path,
    permit: OutcomePhasePermit,
) -> dict[str, Any]:
    """Validate the immutable scorer output behind the live cohort barrier."""

    _require(permit.role == "calibration", "score evidence requires calibration")
    _revalidate_outcome_permit(permit)
    evidence = _load_json(evidence_path)
    _require(
        evidence.get("schema_version") == SCHEMA_VERSION,
        "unsupported calibration score evidence schema",
    )
    _require(
        evidence.get("artifact_kind") == CALIBRATION_SCORE_EVIDENCE_KIND,
        "unsupported calibration score evidence",
    )
    _require(
        evidence.get("protocol_id") == PROTOCOL_ID,
        "calibration score evidence protocol changed",
    )
    _require(evidence.get("role") == "calibration", "score evidence role changed")
    _require(
        evidence.get("cohort_barrier_sha256") == permit.cohort_barrier_sha256,
        "score evidence binds another cohort barrier",
    )
    _require(
        _validate_bound_file(evidence.get("lock", {}), role="calibration lock")
        == Path(permit.lock_path).resolve(),
        "score evidence binds another calibration lock",
    )
    _require(
        evidence.get("ordered_case_names") == list(CALIBRATION_CASE_NAMES),
        "score evidence case order changed",
    )
    _require(
        evidence.get("metric_lock") == METRIC_LOCK, "score evidence metric changed"
    )
    lock = load_held_protocol_lock(permit.lock_path)
    _require(
        evidence.get("outcome_reconstruction_contract_sha256")
        == lock["immutable_bindings"]["outcome_reconstruction_contract"],
        "score evidence reconstruction contract changed",
    )
    records = evidence.get("case_records")
    _require(
        isinstance(records, Mapping) and set(records) == set(CALIBRATION_CASE_NAMES),
        "score evidence must contain all 15 locked calibration cases",
    )
    expected_score_fields = {
        "primary_chamfer_m",
        "comparator_chamfer_m",
        "primary_identity_rmse_m",
        "comparator_identity_rmse_m",
    }
    for case_name in CALIBRATION_CASE_NAMES:
        record = records[case_name]
        _validate_case_score_record(
            record,
            permit,
            case_name,
            expected_score_fields=expected_score_fields,
            expected_role="calibration",
        )
    boundary = evidence.get("information_boundary", {})
    _require(
        boundary.get("all_15_online_predictions_sealed_before_any_outcome") is True
        and boundary.get("outcomes_opened_only_through_live_permit") is True
        and boundary.get("method_selection_or_tuning_performed") is False
        and boundary.get("confirmation_payload_read") is False,
        "score evidence crossed the confirmation boundary",
    )
    _require(
        evidence.get("artifact_sha256") == held_artifact_sha256(evidence),
        "calibration score evidence content checksum changed",
    )
    return evidence


def create_calibration_gate_decision(
    output_path: str | Path,
    permit: OutcomePhasePermit,
    scores: Mapping[str, Mapping[str, float]],
    *,
    score_evidence_path: str | Path,
) -> dict[str, Any]:
    """Apply the frozen GO/NO-GO rule after all calibration seals exist."""

    _require(
        permit.role == "calibration", "calibration gate requires a calibration permit"
    )
    seal_paths = _revalidate_outcome_permit(permit)
    lock = load_held_protocol_lock(permit.lock_path)
    _require(
        lock.get("stage") == "calibration", "calibration gate uses another lock stage"
    )
    evidence = validate_calibration_score_evidence(score_evidence_path, permit)
    evidence_scores = {
        case_name: evidence["case_records"][case_name]["gate_score"]
        for case_name in CALIBRATION_CASE_NAMES
    }
    normalized, summary = _calibration_gate_summary(scores)
    normalized_evidence, _ = _calibration_gate_summary(evidence_scores)
    _require(
        normalized == normalized_evidence,
        "calibration gate scores differ from immutable score evidence",
    )
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CALIBRATION_DECISION_KIND,
        "protocol_id": PROTOCOL_ID,
        "decision": "GO" if summary["passed"] else "NO_GO",
        "calibration_lock": _bound_file(permit.lock_path),
        "cohort_barrier_sha256": permit.cohort_barrier_sha256,
        "online_prediction_seals": {
            case_name: _bound_file(seal_paths[case_name])
            for case_name in CALIBRATION_CASE_NAMES
        },
        "calibration_score_evidence": _bound_file(score_evidence_path),
        "calibration_score_evidence_artifact_sha256": evidence["artifact_sha256"],
        "score_definition": {
            "primary_method": deepcopy(PRIMARY_METHOD),
            "comparator": METRIC_LOCK["comparator"],
            "metrics": [METRIC_LOCK["primary"], METRIC_LOCK["secondary"]],
            "aggregation": METRIC_LOCK["episode_aggregation"],
        },
        "scores": normalized,
        "summary": summary,
        "information_boundary": {
            "all_calibration_predictions_sealed_before_outcomes": True,
            "immutable_score_evidence_validated": True,
            "calibration_role": "go_no_go_only",
            "method_selected_from_calibration": False,
            "confirmation_payload_read": False,
            "no_go_keeps_confirmation_payload_sealed": True,
        },
    }
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    _write_new_json(output_path, artifact)
    return validate_calibration_gate_decision(output_path, permit.lock_path)


def validate_calibration_gate_decision(
    decision_path: str | Path,
    calibration_lock_path: str | Path,
) -> dict[str, Any]:
    lock = load_held_protocol_lock(calibration_lock_path)
    _require(
        lock.get("stage") == "calibration", "gate parent is not a calibration lock"
    )
    decision = _load_json(decision_path)
    _require(
        decision.get("schema_version") == SCHEMA_VERSION,
        "unsupported calibration decision schema",
    )
    _require(
        decision.get("artifact_kind") == CALIBRATION_DECISION_KIND,
        "unsupported calibration decision",
    )
    _require(
        decision.get("protocol_id") == PROTOCOL_ID,
        "calibration decision protocol changed",
    )
    _require(
        _validate_bound_file(
            decision.get("calibration_lock", {}), role="calibration lock"
        )
        == Path(calibration_lock_path).resolve(),
        "calibration decision binds another lock path",
    )
    seal_records = decision.get("online_prediction_seals", {})
    _require(
        isinstance(seal_records, Mapping)
        and set(seal_records) == set(CALIBRATION_CASE_NAMES),
        "calibration decision seal cohort changed",
    )
    seal_paths = {
        case_name: _validate_bound_file(record, role=f"{case_name} online seal")
        for case_name, record in seal_records.items()
    }
    barrier_sha256, _ = _validate_cohort_barrier(
        calibration_lock_path,
        seal_paths,
        role="calibration",
    )
    _require(
        decision.get("cohort_barrier_sha256") == barrier_sha256,
        "calibration decision binds another cohort barrier",
    )
    permit = authorize_outcome_phase(
        calibration_lock_path,
        seal_paths,
        role="calibration",
    )
    score_evidence_path = _validate_bound_file(
        decision.get("calibration_score_evidence", {}),
        role="calibration score evidence",
    )
    score_evidence = validate_calibration_score_evidence(
        score_evidence_path,
        permit,
    )
    _require(
        decision.get("calibration_score_evidence_artifact_sha256")
        == score_evidence["artifact_sha256"],
        "calibration decision binds another score evidence artifact",
    )
    _require(
        decision.get("score_definition")
        == {
            "primary_method": PRIMARY_METHOD,
            "comparator": METRIC_LOCK["comparator"],
            "metrics": [METRIC_LOCK["primary"], METRIC_LOCK["secondary"]],
            "aggregation": METRIC_LOCK["episode_aggregation"],
        },
        "calibration score definition changed",
    )
    normalized, summary = _calibration_gate_summary(decision.get("scores", {}))
    evidence_scores = {
        case_name: score_evidence["case_records"][case_name]["gate_score"]
        for case_name in CALIBRATION_CASE_NAMES
    }
    normalized_evidence, _ = _calibration_gate_summary(evidence_scores)
    _require(decision.get("scores") == normalized, "calibration scores changed")
    _require(
        normalized == normalized_evidence,
        "calibration decision scores differ from immutable evidence",
    )
    _require(
        decision.get("summary") == summary, "calibration decision arithmetic changed"
    )
    expected_decision = "GO" if summary["passed"] else "NO_GO"
    _require(
        decision.get("decision") == expected_decision, "calibration decision changed"
    )
    boundary = decision.get("information_boundary", {})
    _require(
        boundary.get("all_calibration_predictions_sealed_before_outcomes") is True
        and boundary.get("immutable_score_evidence_validated") is True
        and boundary.get("calibration_role") == "go_no_go_only"
        and boundary.get("method_selected_from_calibration") is False
        and boundary.get("confirmation_payload_read") is False
        and boundary.get("no_go_keeps_confirmation_payload_sealed") is True,
        "calibration decision weakened the confirmation boundary",
    )
    _require(
        decision.get("artifact_sha256") == held_artifact_sha256(decision),
        "calibration decision content checksum changed",
    )
    return decision


def validate_confirmation_score_evidence(
    evidence_path: str | Path,
    permit: OutcomePhasePermit,
) -> dict[str, Any]:
    """Validate immutable score evidence for the exact six-case panel."""

    _require(permit.role == "confirmation", "score evidence requires confirmation")
    _revalidate_outcome_permit(permit)
    evidence = _load_json(evidence_path)
    _require(
        evidence.get("schema_version") == SCHEMA_VERSION,
        "unsupported confirmation score evidence schema",
    )
    _require(
        evidence.get("artifact_kind") == CONFIRMATION_SCORE_EVIDENCE_KIND,
        "unsupported confirmation score evidence",
    )
    _require(
        evidence.get("protocol_id") == PROTOCOL_ID,
        "confirmation score evidence protocol changed",
    )
    _require(evidence.get("role") == "confirmation", "score evidence role changed")
    _require(
        evidence.get("cohort_barrier_sha256") == permit.cohort_barrier_sha256,
        "score evidence binds another cohort barrier",
    )
    _require(
        _validate_bound_file(evidence.get("lock", {}), role="confirmation lock")
        == Path(permit.lock_path).resolve(),
        "score evidence binds another confirmation lock",
    )
    _require(
        evidence.get("ordered_case_names") == list(CONFIRMATION_CASE_NAMES),
        "confirmation score evidence case order changed",
    )
    _require(
        evidence.get("metric_lock") == METRIC_LOCK,
        "confirmation score evidence metric changed",
    )
    lock = load_held_protocol_lock(permit.lock_path)
    _require(
        lock.get("stage") == "confirmation",
        "confirmation score evidence uses another lock stage",
    )
    _require(
        evidence.get("outcome_reconstruction_contract_sha256")
        == lock["immutable_bindings"]["outcome_reconstruction_contract"],
        "confirmation score evidence reconstruction contract changed",
    )
    records = evidence.get("case_records")
    _require(
        isinstance(records, Mapping) and set(records) == set(CONFIRMATION_CASE_NAMES),
        "score evidence must contain all six locked confirmation cases",
    )
    expected_score_fields = {
        "primary_chamfer_m",
        "comparator_chamfer_m",
        "primary_identity_rmse_m",
        "comparator_identity_rmse_m",
    }
    for case_name in CONFIRMATION_CASE_NAMES:
        _validate_case_score_record(
            records[case_name],
            permit,
            case_name,
            expected_score_fields=expected_score_fields,
            expected_role="confirmation",
        )
    _require(
        evidence.get("information_boundary")
        == {
            "all_6_online_predictions_sealed_before_any_outcome": True,
            "outcomes_opened_only_through_live_permit": True,
            "method_selection_or_tuning_performed": False,
            "calibration_method_and_gate_unchanged": True,
        },
        "confirmation score evidence crossed the frozen-method boundary",
    )
    _require(
        evidence.get("artifact_sha256") == held_artifact_sha256(evidence),
        "confirmation score evidence content checksum changed",
    )
    return evidence


def create_confirmation_gate_decision(
    output_path: str | Path,
    permit: OutcomePhasePermit,
    scores: Mapping[str, Mapping[str, float]],
    *,
    score_evidence_path: str | Path,
) -> dict[str, Any]:
    """Apply the frozen final gate with no selection or tuning."""

    _require(
        permit.role == "confirmation",
        "confirmation gate requires a confirmation permit",
    )
    seal_paths = _revalidate_outcome_permit(permit)
    lock = load_held_protocol_lock(permit.lock_path)
    _require(
        lock.get("stage") == "confirmation",
        "confirmation gate uses another lock stage",
    )
    evidence = validate_confirmation_score_evidence(score_evidence_path, permit)
    evidence_scores = {
        case_name: evidence["case_records"][case_name]["gate_score"]
        for case_name in CONFIRMATION_CASE_NAMES
    }
    normalized, summary = _confirmation_gate_summary(scores)
    normalized_evidence, _ = _confirmation_gate_summary(evidence_scores)
    _require(
        normalized == normalized_evidence,
        "confirmation gate scores differ from immutable score evidence",
    )
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CONFIRMATION_DECISION_KIND,
        "protocol_id": PROTOCOL_ID,
        "decision": "CONFIRMED" if summary["passed"] else "NOT_CONFIRMED",
        "confirmation_lock": _bound_file(permit.lock_path),
        "cohort_barrier_sha256": permit.cohort_barrier_sha256,
        "online_prediction_seals": {
            case_name: _bound_file(seal_paths[case_name])
            for case_name in CONFIRMATION_CASE_NAMES
        },
        "confirmation_score_evidence": _bound_file(score_evidence_path),
        "confirmation_score_evidence_artifact_sha256": evidence["artifact_sha256"],
        "score_definition": {
            "primary_method": deepcopy(PRIMARY_METHOD),
            "comparator": METRIC_LOCK["comparator"],
            "metrics": [METRIC_LOCK["primary"], METRIC_LOCK["secondary"]],
            "aggregation": METRIC_LOCK["episode_aggregation"],
        },
        "confirmation_gate": deepcopy(CONFIRMATION_GATE),
        "scores": normalized,
        "summary": summary,
        "information_boundary": {
            "all_confirmation_predictions_sealed_before_outcomes": True,
            "immutable_score_evidence_validated": True,
            "primary_method_fixed_before_calibration": True,
            "method_selection_or_tuning_performed": False,
            "all_6_cases_reported": True,
        },
    }
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    _write_new_json(output_path, artifact)
    return validate_confirmation_gate_decision(
        output_path,
        permit.lock_path,
    )


def validate_confirmation_gate_decision(
    decision_path: str | Path,
    confirmation_lock_path: str | Path,
) -> dict[str, Any]:
    """Recompute every final-gate value from bound immutable evidence."""

    lock = load_held_protocol_lock(confirmation_lock_path)
    _require(
        lock.get("stage") == "confirmation",
        "final-gate parent is not a confirmation lock",
    )
    decision = _load_json(decision_path)
    _require(
        decision.get("schema_version") == SCHEMA_VERSION,
        "unsupported confirmation decision schema",
    )
    _require(
        decision.get("artifact_kind") == CONFIRMATION_DECISION_KIND,
        "unsupported confirmation decision",
    )
    _require(
        decision.get("protocol_id") == PROTOCOL_ID,
        "confirmation decision protocol changed",
    )
    _require(
        _validate_bound_file(
            decision.get("confirmation_lock", {}), role="confirmation lock"
        )
        == Path(confirmation_lock_path).resolve(),
        "confirmation decision binds another lock path",
    )
    seal_records = decision.get("online_prediction_seals", {})
    _require(
        isinstance(seal_records, Mapping)
        and set(seal_records) == set(CONFIRMATION_CASE_NAMES),
        "confirmation decision seal cohort changed",
    )
    seal_paths = {
        case_name: _validate_bound_file(record, role=f"{case_name} online seal")
        for case_name, record in seal_records.items()
    }
    barrier_sha256, _ = _validate_cohort_barrier(
        confirmation_lock_path,
        seal_paths,
        role="confirmation",
    )
    _require(
        decision.get("cohort_barrier_sha256") == barrier_sha256,
        "confirmation decision binds another cohort barrier",
    )
    permit = authorize_outcome_phase(
        confirmation_lock_path,
        seal_paths,
        role="confirmation",
    )
    score_evidence_path = _validate_bound_file(
        decision.get("confirmation_score_evidence", {}),
        role="confirmation score evidence",
    )
    score_evidence = validate_confirmation_score_evidence(
        score_evidence_path,
        permit,
    )
    _require(
        decision.get("confirmation_score_evidence_artifact_sha256")
        == score_evidence["artifact_sha256"],
        "confirmation decision binds another score evidence artifact",
    )
    _require(
        decision.get("score_definition")
        == {
            "primary_method": PRIMARY_METHOD,
            "comparator": METRIC_LOCK["comparator"],
            "metrics": [METRIC_LOCK["primary"], METRIC_LOCK["secondary"]],
            "aggregation": METRIC_LOCK["episode_aggregation"],
        },
        "confirmation score definition changed",
    )
    _require(
        decision.get("confirmation_gate") == CONFIRMATION_GATE,
        "confirmation gate contract changed",
    )
    normalized, summary = _confirmation_gate_summary(decision.get("scores", {}))
    evidence_scores = {
        case_name: score_evidence["case_records"][case_name]["gate_score"]
        for case_name in CONFIRMATION_CASE_NAMES
    }
    normalized_evidence, _ = _confirmation_gate_summary(evidence_scores)
    _require(decision.get("scores") == normalized, "confirmation scores changed")
    _require(
        normalized == normalized_evidence,
        "confirmation decision scores differ from immutable evidence",
    )
    _require(
        decision.get("summary") == summary,
        "confirmation decision arithmetic changed",
    )
    expected_decision = "CONFIRMED" if summary["passed"] else "NOT_CONFIRMED"
    _require(
        decision.get("decision") == expected_decision,
        "confirmation decision changed",
    )
    _require(
        decision.get("information_boundary")
        == {
            "all_confirmation_predictions_sealed_before_outcomes": True,
            "immutable_score_evidence_validated": True,
            "primary_method_fixed_before_calibration": True,
            "method_selection_or_tuning_performed": False,
            "all_6_cases_reported": True,
        },
        "confirmation decision weakened the frozen-method boundary",
    )
    _require(
        decision.get("artifact_sha256") == held_artifact_sha256(decision),
        "confirmation decision content checksum changed",
    )
    return decision


def create_confirmation_protocol_lock(
    output_path: str | Path,
    calibration_lock_path: str | Path,
    calibration_decision_path: str | Path,
) -> dict[str, Any]:
    """Promote the immutable calibration lock only after a validated GO."""

    calibration_lock = load_held_protocol_lock(calibration_lock_path)
    _require(
        calibration_lock.get("stage") == "calibration",
        "confirmation parent is not a calibration lock",
    )
    decision = validate_calibration_gate_decision(
        calibration_decision_path,
        calibration_lock_path,
    )
    _require(
        decision.get("decision") == "GO",
        "confirmation remains sealed after calibration NO-GO",
    )
    artifact = deepcopy(calibration_lock)
    artifact.pop("artifact_sha256", None)
    artifact["stage"] = "confirmation"
    artifact["confirmation_access_authorized"] = True
    artifact["parent_calibration_lock"] = _bound_file(calibration_lock_path)
    artifact["calibration_gate_evidence"] = _bound_file(calibration_decision_path)
    artifact["information_boundary"]["calibration_outcomes_read_before_lock"] = True
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    _write_new_json(output_path, artifact)
    return load_held_protocol_lock(output_path)


def run_outcome_operation(
    permit: OutcomePhasePermit,
    *,
    case_name: str,
    operation: str,
    callback: Callable[[], _T],
) -> _T:
    """Run one outcome create/read callback behind a live cohort barrier.

    All lock, seal, and bound artifact checks are repeated immediately before
    invoking the callback.  A permit therefore becomes unusable if any sealed
    file changes after authorization.
    """

    _require(operation in {"create", "read"}, "unsupported outcome operation")
    seal_paths = _revalidate_outcome_permit(permit)
    _require(case_name in seal_paths, "outcome case is outside the permitted cohort")
    return callback()


__all__ = [
    "CALIBRATION_CASE_NAMES",
    "CALIBRATION_GATE",
    "CALIBRATION_SCORE_EVIDENCE_KIND",
    "CONFIRMATION_CASES",
    "CONFIRMATION_CASE_NAMES",
    "CONFIRMATION_DECISION_KIND",
    "CONFIRMATION_GATE",
    "CONFIRMATION_SCORE_EVIDENCE_KIND",
    "CONTROL_METHODS",
    "FRAME_COUNT",
    "FRAME_ZERO_KIND",
    "METRIC_LOCK",
    "ONLINE_ARTIFACT_ROLES",
    "OutcomePhasePermit",
    "PHYSICAL_ARTIFACT_ROLES",
    "PRIMARY_METHOD",
    "PROTOCOL_ID",
    "REQUIRED_IMMUTABLE_BINDING_KEYS",
    "SOURCE_FEASIBILITY_AMENDMENT_CONTRACT",
    "UPDATE_FRAMES",
    "authorize_outcome_phase",
    "create_calibration_gate_decision",
    "create_confirmation_gate_decision",
    "create_confirmation_protocol_lock",
    "create_held_protocol_lock",
    "create_online_prediction_seal",
    "create_physical_prior_seal",
    "create_prefix_stage_authorization",
    "held_artifact_sha256",
    "held_contract_sha256",
    "load_held_protocol_lock",
    "locked_case_names",
    "run_outcome_operation",
    "validate_frame_zero_bundle_manifest",
    "validate_calibration_gate_decision",
    "validate_calibration_score_evidence",
    "validate_confirmation_gate_decision",
    "validate_confirmation_score_evidence",
    "validate_online_prediction_seal",
    "validate_physical_prior_seal",
    "validate_prefix_stage_authorization",
]
