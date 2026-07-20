#!/usr/bin/env python3
"""Create the prospective Deform360 held-v6 lock from audited lineage.

This is an operator tool, not part of the prediction method.  It updates every
method-local source binding from an immutable deployed snapshot, recomputes the
v6 contracts, preserves unchanged external identities from v1, binds the exact
failed-closed v4 and v5 execution reports, and admits only the dedicated frozen
v5 Python runtime.
It never accepts or reads an outcome, target, tactile, or confirmation path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


EXPECTED_V1_LOCK_FILE_SHA256 = (
    "eaa06f7e5d7de95497e090bcb706cf65b25d8841de4d10d0629f4391a4c488cd"
)
EXPECTED_V1_LOCK_ARTIFACT_SHA256 = (
    "543d86653edf7c5931e6b3a133d22434f2960072e129d3f95d5c2a252e14bb39"
)
EXPECTED_V1_REPORT_FILE_SHA256 = (
    "430176bfb869127f7658b806c57e4609e40fc738b48a3cef98fb9eb90259eb6d"
)
EXPECTED_V1_REPORT_ARTIFACT_SHA256 = (
    "1337154d8b44508d20d310523c7e94561050f092681acc85da9baae51f5d70af"
)
EXPECTED_V1_COUNTS = {
    "frame_zero_failure": 9,
    "physical_admission_failure": 1,
    "sealed_prediction": 5,
}
EXPECTED_V5_LOCK_FILE_SHA256 = (
    "a917650499b047bdcd7d7baf57212ff82a9e277867bb3ba5389b1a0c126d950e"
)
EXPECTED_V5_LOCK_ARTIFACT_SHA256 = (
    "cfb13c88220e5abbe83937d20e125ed7c22727bcce5c4acbf58df0f2b07d440d"
)
EXPECTED_V5_LOCK_BINDING_COUNT = 112
EXPECTED_V5_REPORT_FILE_SHA256 = (
    "915a556fb8a412b314766c7169878db9880200f63d07dd3ec82476be5d2ea14b"
)
EXPECTED_V5_REPORT_ARTIFACT_SHA256 = (
    "38a2f0b3c08ccac82b3d301fb77b993b7c994ba5f42c260c2a669a0d80a57def"
)
EXPECTED_V5_REPORT_SIZE_BYTES = 13_828
EXPECTED_V6_BINDING_COUNT = 113
EXPECTED_V6_MIGRATION_KEY_COUNT = 22
EXPECTED_V4_LOCK_FILE_SHA256 = (
    "3f5b6b678c095cd16e5aec1fdb8d0a6ad690e7a7e26c373b4740675e3399dacb"
)
EXPECTED_V4_LOCK_ARTIFACT_SHA256 = (
    "54c066064b5bdb208c8ac0cd411db3450d9bdf5c4dddece64cdebc21cdf51829"
)
EXPECTED_V4_REPORT_FILE_SHA256 = (
    "9b585f1340a47c64d787a5489faa1c67738d733d4d577648ccb753361c5dd4ca"
)
EXPECTED_V4_REPORT_ARTIFACT_SHA256 = (
    "72fb6ba1c6f113157e8351f5b470bcc03d4b20ccec64b3504c7e356fc69cfdc0"
)
EXPECTED_V5_RUNTIME_MANIFEST_FILE_SHA256 = (
    "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61"
)
EXPECTED_V5_PIP_FREEZE_SHA256 = (
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
_PYCACHE_PREFIX = "/nonexistent/bpt-held-v6-pycache"
_PINNED_OPERATOR_ENVIRONMENT = {
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": "/home/florianpfaff",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LOGNAME": "florianpfaff",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPYCACHEPREFIX": _PYCACHE_PREFIX,
    "TMPDIR": "/tmp",
    "USER": "florianpfaff",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_GIT_EXECUTABLE = Path("/usr/bin/git")
_PROTECTED_PAYLOAD_BASENAMES = frozenset(
    {
        "outcome.json",
        "target_data.pkl",
        "tactile_data.pkl",
        "future_tactile.pkl",
        "confirmation_payload.json",
    }
)
_CANONICAL_HELD_BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
_CANONICAL_HELD_V1_ROOT = _CANONICAL_HELD_BASE / "held-v1"
_CANONICAL_HELD_V2_ROOT = _CANONICAL_HELD_BASE / "held-v2"
_CANONICAL_HELD_V3_ROOT = _CANONICAL_HELD_BASE / "held-v3"
_CANONICAL_HELD_V4_ROOT = _CANONICAL_HELD_BASE / "held-v4"
_CANONICAL_HELD_V5_ROOT = _CANONICAL_HELD_BASE / "held-v5"
_CANONICAL_HELD_V6_ROOT = _CANONICAL_HELD_BASE / "held-v6"
_CANONICAL_V1_LOCK = _CANONICAL_HELD_V1_ROOT / "calibration-lock.json"
_CANONICAL_V1_REPORT = _CANONICAL_HELD_V1_ROOT / "v1-preoutcome-feasibility-report.json"
_CANONICAL_V2_WITHDRAWAL_REPORT = (
    _CANONICAL_HELD_V4_ROOT / "v2-design-withdrawal-report.json"
)
_CANONICAL_V3_BOUNDARY_INCIDENT_REPORT = (
    _CANONICAL_HELD_V4_ROOT / "v3-prelock-boundary-incident-report.json"
)
_CANONICAL_V4_LOCK = _CANONICAL_HELD_V4_ROOT / "calibration-lock.json"
_CANONICAL_V4_EXECUTION_WITHDRAWAL_REPORT = (
    _CANONICAL_HELD_V4_ROOT / "v4-execution-withdrawal-report.json"
)
_CANONICAL_V5_LOCK = _CANONICAL_HELD_V5_ROOT / "calibration-lock.json"
_CANONICAL_V5_OUTCOME_WITHDRAWAL_REPORT = (
    _CANONICAL_HELD_V5_ROOT / "v5-outcome-withdrawal-report.json"
)
_CANONICAL_V6_LOCK = _CANONICAL_HELD_V6_ROOT / "calibration-lock.json"
_V2_WITHDRAWAL_EXECUTION_COUNTS = {
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
}
_V2_WITHDRAWAL_REPORT_UNSIGNED: Mapping[str, Any] = {
    "artifact_kind": "Deform360HeldProtocolWithdrawalReport",
    "designed_calibration_case_count": 15,
    "disposition": "WITHDRAWN_BEFORE_LOCK_AND_PREDICTION",
    "execution_counts": _V2_WITHDRAWAL_EXECUTION_COUNTS,
    "existence_evidence": {
        "canonical_held_root": os.fspath(_CANONICAL_HELD_V2_ROOT),
        "evidence_scope": "source-only filesystem existence check",
        "held_root_exists": False,
    },
    "information_boundary": {
        "confirmation_payload_read": False,
        "episode_payload_read": False,
        "frame_zero_payload_read": False,
        "future_tactile_read": False,
        "outcome_created": False,
        "outcome_phase_authorized": False,
        "outcome_read": False,
        "prediction_payload_read": False,
        "source_only_evidence": True,
        "target_data_read": False,
        "target_or_outcome_path_accessed": False,
    },
    "protocol_id": "deform360-held-online-belief-v2",
    "replacement_protocol_id": "deform360-held-online-belief-v3",
    "reuse": {
        "v1_predictions_reused_by_v3": False,
        "v2_artifacts_reused_by_v3": False,
        "v2_predictions_reused_by_v3": False,
    },
    "schema_version": 1,
    "withdrawal_reason": {
        "defect": (
            "the abandoned v2 design averaged heterogeneous robot.npz:actions "
            "rows (translation, three rotation rows, and opening) into a "
            "pseudo-centre used for window selection"
        ),
        "replacement": (
            "v3 requires exactly the five robot archive fields, redundant-state "
            "parity, T_worlds translation with openings/bimanual semantics, and "
            "exact selected-source-slice replay"
        ),
    },
}

_V3_WITHDRAWAL_EXECUTION_COUNTS = {
    "calibration_lock_count": 0,
    "case_attempt_count": 0,
    "deployed_snapshot_count": 0,
    "deployment_count": 0,
    "frame_zero_manifest_count": 0,
    "online_prediction_seal_count": 0,
    "outcome_created_count": 0,
    "outcome_permit_count": 0,
    "outcome_api_operation_count": 0,
    "physical_prior_seal_count": 0,
    "prediction_count": 0,
    "prefix_authorization_count": 0,
    "shard_start_count": 0,
    "target_operation_count": 0,
}
_V3_BOUNDARY_INCIDENT_REPORT_UNSIGNED: Mapping[str, Any] = {
    "artifact_kind": "Deform360HeldProtocolBoundaryIncidentWithdrawalReport",
    "designed_calibration_case_count": 15,
    "disposition": "WITHDRAWN_BEFORE_LOCK_AND_PREDICTION",
    "formal_protocol_execution_counts": _V3_WITHDRAWAL_EXECUTION_COUNTS,
    "formal_protocol_execution_scope": (
        "canonical held-v3 pipeline and artifacts only; excludes the separately "
        "disclosed rg content scanner"
    ),
    "existence_evidence": {
        "canonical_held_root": os.fspath(_CANONICAL_HELD_V3_ROOT),
        "evidence_scope": "filesystem existence check after the pre-lock incident",
        "held_root_exists": False,
    },
    "incident": {
        "classification": "PRELOCK_INFORMATION_BOUNDARY_INCIDENT",
        "execution_context": "SSH",
        "search": {
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
        },
        "scanner_scope": {
            "may_have_opened_any_regular_file_under_search_roots": True,
            "protected_file_open_status": "NOT_CLAIMED",
        },
        "returned_output": {
            "only_matching_absolute_filenames": True,
            "included_unrelated_171_outcome_or_log_paths": True,
            "payload_bytes_returned": False,
            "metrics_returned": False,
            "labels_returned": False,
            "arrays_returned": False,
            "payload_values_returned": False,
        },
    },
    "information_boundary": {
        "content_scanner_may_have_opened_any_regular_file_under_search_roots": True,
        "protected_file_open_status": "NOT_CLAIMED",
        "held_cohort_payload_content_or_value_returned_to_research_agent": False,
        "outcome_metric_label_array_or_value_returned_to_research_agent": False,
        "method_or_gate_choice_used_outcome_values": False,
        "stdout_was_filename_only": True,
    },
    "protocol_id": "deform360-held-online-belief-v3",
    "replacement_protocol_id": "deform360-held-online-belief-v4",
    "reuse": {
        "v1_predictions_reused_by_v4": False,
        "v2_execution_artifacts_reused_by_v4": False,
        "v2_predictions_reused_by_v4": False,
        "v3_execution_artifacts_reused_by_v4": False,
        "v3_predictions_reused_by_v4": False,
        "sealed_source_only_lineage_reports_bound_by_v4": [
            "v1_preoutcome_feasibility_report",
            "v2_design_withdrawal_report",
            "v3_prelock_boundary_incident_report",
        ],
    },
    "schema_version": 1,
    "withdrawal_reason": {
        "incident": (
            "an SSH rg -l content search for two public CoTracker identities was "
            "run over broad roots and piped to head; only matching absolute "
            "filenames were returned, but rg may have opened any regular file "
            "under those roots"
        ),
        "replacement": (
            "v4 starts from a fresh absent held root, binds this exact report, and "
            "reuses no v3 prediction or execution artifact"
        ),
    },
}


# These bindings name files in the deployed method snapshot.  Values absent
# from this table are either canonical contracts/configurations recomputed
# below or external runtimes inherited byte-for-byte from the sealed v1 lock.
LOCAL_FILE_BINDINGS: Mapping[str, str] = {
    "cpd_registration_source": "src/bayesian_phystwin/cpd_registration.py",
    "deform360_dataset_containment_source": (
        "src/bayesian_phystwin/deform360_dataset_containment.py"
    ),
    "deform360_hidden_metric_source": (
        "src/bayesian_phystwin/deform360_online_belief_evaluation.py"
    ),
    "deform360_object_sam2_source": ("src/causal4d_public/deform360_object_sam2.py"),
    "deform360_sam2_source": "src/causal4d_public/deform360_sam2.py",
    "deform360_robot_kinematics_source": (
        "src/bayesian_phystwin/deform360_robot_kinematics.py"
    ),
    "frame_zero_builder_cli": (
        "src/bayesian_phystwin/cli/deform360_frame_zero_assets.py"
    ),
    "frame_zero_builder_source": (
        "src/bayesian_phystwin/deform360_frame_zero_assets.py"
    ),
    "frame_zero_deform360_protocol_dependency": ("src/causal4d_public/deform360.py"),
    "frame_zero_object_sam2_source": ("src/causal4d_public/deform360_object_sam2.py"),
    "frame_zero_sam2_constants_source": ("src/causal4d_public/deform360_sam2.py"),
    "frame_zero_semantic_gate_source": (
        "src/bayesian_phystwin/deform360_frame_zero_semantic_gate.py"
    ),
    "frame_zero_visual_hull_source": ("src/causal4d_public/deform360_visual_hull.py"),
    "graph_residual_mapping_source": (
        "src/bayesian_phystwin/phystwin_graph_residual_mapping.py"
    ),
    "held_online_runner_cli": (
        "src/bayesian_phystwin/cli/deform360_held_online_prefix.py"
    ),
    "held_online_runner_source": (
        "src/bayesian_phystwin/deform360_held_online_prefix.py"
    ),
    "held_calibration_case_runner_source": (
        "scripts/held/run_deform360_v6_calibration_case.sh"
    ),
    "held_calibration_outcome_driver_source": (
        "scripts/held/run_deform360_v6_calibration_outcomes.py"
    ),
    "held_calibration_shard_runner_source": (
        "scripts/held/run_deform360_v6_calibration_shard.sh"
    ),
    "held_confirmation_case_runner_source": (
        "scripts/held/run_deform360_v6_confirmation_case.sh"
    ),
    "held_confirmation_outcome_driver_source": (
        "scripts/held/run_deform360_v6_confirmation_outcomes.py"
    ),
    "held_confirmation_shard_runner_source": (
        "scripts/held/run_deform360_v6_confirmation_shard.sh"
    ),
    "held_outcome_reconstruction_adapter_source": (
        "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
    ),
    "held_outcome_scorer_source": (
        "src/bayesian_phystwin/deform360_held_outcome_scoring.py"
    ),
    "held_physical_builder_cli": (
        "src/bayesian_phystwin/cli/deform360_held_physical_prior.py"
    ),
    "held_physical_builder_source": (
        "src/bayesian_phystwin/deform360_held_physical_prior.py"
    ),
    "held_protocol_source": "src/bayesian_phystwin/deform360_held_protocol.py",
    "held_protocol_lock_operator_source": ("scripts/held/prepare_deform360_v6_lock.py"),
    "independent_cpd_source": ("src/bayesian_phystwin/deform360_cpd_diagnostic.py"),
    "nearest_distance_metric_source": (
        "src/bayesian_phystwin/phystwin_official_evaluation.py"
    ),
    "pairwise_clique_source": (
        "src/bayesian_phystwin/deform360_raw_pairwise_correspondence_diagnostic.py"
    ),
    "pyproject_toml": "pyproject.toml",
    "raw_cycle_cli": (
        "src/bayesian_phystwin/cli/deform360_raw_camera_cycle_uncertainty.py"
    ),
    "raw_cycle_source": (
        "src/bayesian_phystwin/deform360_raw_camera_cycle_uncertainty.py"
    ),
    "raw_gated_evaluator_cli": (
        "src/bayesian_phystwin/cli/deform360_raw_camera_gated_evaluation.py"
    ),
    "raw_gated_evaluator_source": (
        "src/bayesian_phystwin/deform360_raw_camera_gated_evaluation.py"
    ),
    "raw_observation_cli": (
        "src/bayesian_phystwin/cli/deform360_raw_camera_observation.py"
    ),
    "raw_observation_source": (
        "src/bayesian_phystwin/deform360_raw_camera_observation.py"
    ),
    "raw_uncertainty_cli": (
        "src/bayesian_phystwin/cli/deform360_raw_camera_uncertainty.py"
    ),
    "raw_uncertainty_source": (
        "src/bayesian_phystwin/deform360_raw_camera_uncertainty.py"
    ),
    "recursive_cpd_source": (
        "src/bayesian_phystwin/deform360_recursive_cpd_diagnostic.py"
    ),
    "recursive_rbf_source": "src/bayesian_phystwin/phystwin_online_belief.py",
    "robust_correspondence_source": (
        "src/bayesian_phystwin/deform360_robust_correspondence_diagnostic.py"
    ),
}

# These are the only v1 values that v6 may inherit.  They identify runtimes,
# checkpoints, external repositories/configurations, and the sealed remote
# inventory.  Everything implemented by this repository is recomputed below.
# Keeping this list explicit prevents a newly added local binding from being
# silently copied out of the parent lock.
INHERITED_EXTERNAL_BINDING_KEYS = frozenset(
    {
        "alltracker_checkpoint",
        "alltracker_molmomotion_revision_literal",
        "alltracker_provenance_tree_literal",
        "alltracker_runtime_tree",
        "cotracker_checkpoint",
        "cotracker_commit_object",
        "cotracker_git_tree_manifest",
        "cotracker_revision_literal",
        "dataset_revision_literal",
        "deform360_code_commit_object",
        "deform360_code_git_tree_manifest",
        "deform360_code_revision_literal",
        "deform360_official_outcome_builder_source",
        "deform360_pipeline_config",
        "deform360_pipeline_config_semantic",
        "deform360_stage_script",
        "deform360_strict_reconstruction_source",
        "ffmpeg_executable",
        "ffmpeg_version_literal",
        "official_phystwin_commit_object",
        "official_phystwin_git_tree_manifest",
        "official_phystwin_real_config",
        "official_phystwin_revision_literal",
        "python_executable",
        "remote_confirmation_inventory_combined",
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
    }
)

# These v6 external identities are not inherited from the v1 lock.  They bind
# the exact semantic exports used by the source-only census and the dedicated
# frozen Python runtime introduced by the failed-closed v4 migration.
V6_PINNED_EXTERNAL_BINDING_VALUES: Mapping[str, str] = {
    "frame_zero_siglip2_model_tree": (
        "b293f9bb1cd86272626d211d7c297c99d3aa1adfc4a3072b83e69f1fa70773ad"
    ),
    "frame_zero_siglip2_revision_literal": (
        "5adaedd88ee9835df44ae379db76ca2e0a84427d427f05d9b268f5556b02ff97"
    ),
    "frame_zero_siglip2_transformers_sources": (
        "9a2cdf1e349a250963d0e18438665d1f6dc4f6e578ee38a3f05234478cc0ab86"
    ),
    "held_frozen_runtime_manifest": (EXPECTED_V5_RUNTIME_MANIFEST_FILE_SHA256),
    "python_pip_freeze_sorted": EXPECTED_V5_PIP_FREEZE_SHA256,
}
V6_PINNED_EXTERNAL_BINDING_KEYS = frozenset(V6_PINNED_EXTERNAL_BINDING_VALUES)
V6_PINNED_SEMANTIC_GATE_CONTRACT_SHA256 = (
    "6c5e9b71b9948724f38dc42bda550f8489e310a448a970f51b60d0ebff59fbf7"
)


def _require_deployed_semantic_binding_exports(
    *,
    model_tree_sha256: str,
    model_revision: str,
    transformers_source_aggregate_sha256: str,
    semantic_gate_contract_sha256: str,
) -> None:
    """Reject a lock operator whose copied semantic identities have drifted."""

    expected_external_bindings = {
        "frame_zero_siglip2_model_tree": model_tree_sha256,
        "frame_zero_siglip2_revision_literal": hashlib.sha256(
            model_revision.encode("ascii")
        ).hexdigest(),
        "frame_zero_siglip2_transformers_sources": (
            transformers_source_aggregate_sha256
        ),
    }
    _require(
        {
            key: V6_PINNED_EXTERNAL_BINDING_VALUES[key]
            for key in expected_external_bindings
        }
        == expected_external_bindings,
        "pinned SigLIP2 bindings diverge from deployed semantic exports",
    )
    _require(
        V6_PINNED_SEMANTIC_GATE_CONTRACT_SHA256 == semantic_gate_contract_sha256,
        "semantic gate contract binding diverges from deployed semantic export",
    )


LOCAL_CONTRACT_BINDING_KEYS = frozenset(
    {
        "frame_zero_default_config",
        "frame_zero_semantic_gate_contract",
        "held_calibration_gate_contract",
        "held_confirmation_gate_contract",
        "held_metric_contract",
        "held_physical_numeric_contract",
        "held_primary_method_contract",
        "held_source_feasibility_amendment_contract",
        "outcome_reconstruction_contract",
        "primary_rbf_config",
        "raw_cycle_default_config",
        "raw_observation_default_config",
        "raw_uncertainty_default_config",
        "robot_kinematics_window_contract",
        "upstream_runtime_bundle_tree",
    }
)
METHOD_PROVENANCE_BINDING_KEYS = frozenset(
    {
        "method_commit_object",
        "method_deployed_snapshot_tree",
        "method_git_tree_manifest",
        "method_head_literal",
    }
)
V6_ONLY_BINDING_KEYS = frozenset(
    {
        "deform360_dataset_containment_source",
        "deform360_robot_kinematics_source",
        "frame_zero_semantic_gate_contract",
        "frame_zero_semantic_gate_source",
        "frame_zero_siglip2_model_tree",
        "frame_zero_siglip2_revision_literal",
        "frame_zero_siglip2_transformers_sources",
        "held_calibration_case_runner_source",
        "held_calibration_outcome_driver_source",
        "held_calibration_shard_runner_source",
        "held_confirmation_case_runner_source",
        "held_confirmation_outcome_driver_source",
        "held_confirmation_shard_runner_source",
        "held_frozen_runtime_manifest",
        "held_protocol_lock_operator_source",
        "held_source_feasibility_amendment_contract",
        "robot_kinematics_window_contract",
        "v1_preoutcome_feasibility_report",
        "v2_design_withdrawal_report",
        "v3_prelock_boundary_incident_report",
        "v4_execution_withdrawal_report",
        "v5_outcome_withdrawal_report",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _absolute(path: Path) -> Path:
    """Return an absolute spelling without following the final symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _ensure_unprotected_path(path: Path, role: str) -> None:
    _require(
        path.name.lower() not in _PROTECTED_PAYLOAD_BASENAMES,
        f"{role} names a protected held payload",
    )


def _read_regular_bytes(path: Path, role: str) -> bytes:
    source = _absolute(path)
    _ensure_unprotected_path(source, role)
    try:
        before = os.lstat(source)
    except FileNotFoundError as error:
        raise ValueError(f"{role} is absent: {source}") from error
    _require(stat.S_ISREG(before.st_mode), f"{role} is not a regular file: {source}")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        after = os.fstat(descriptor)
        _require(
            stat.S_ISREG(after.st_mode)
            and (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
            f"{role} changed while it was opened",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _sha256_file(path: Path, role: str = "file") -> str:
    return hashlib.sha256(_read_regular_bytes(path, role)).hexdigest()


def _load_json(path: Path, role: str) -> tuple[dict[str, Any], str]:
    payload = _read_regular_bytes(path, role)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not UTF-8 JSON: {path}") from error
    _require(isinstance(value, dict), f"{role} JSON root must be an object")
    return value, hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return _canonical_sha256(unsigned)


def _expected_v2_withdrawal_report() -> tuple[dict[str, Any], bytes, str, str]:
    artifact_sha256 = _canonical_sha256(_V2_WITHDRAWAL_REPORT_UNSIGNED)
    report = {
        **_V2_WITHDRAWAL_REPORT_UNSIGNED,
        "artifact_sha256": artifact_sha256,
    }
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return report, encoded, hashlib.sha256(encoded).hexdigest(), artifact_sha256


def _expected_v3_boundary_incident_report() -> tuple[dict[str, Any], bytes, str, str]:
    artifact_sha256 = _canonical_sha256(_V3_BOUNDARY_INCIDENT_REPORT_UNSIGNED)
    report = {
        **_V3_BOUNDARY_INCIDENT_REPORT_UNSIGNED,
        "artifact_sha256": artifact_sha256,
    }
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return report, encoded, hashlib.sha256(encoded).hexdigest(), artifact_sha256


def _require_exact_mode_0400(path: Path, role: str) -> None:
    _require(
        stat.S_IMODE(os.lstat(_absolute(path)).st_mode) == 0o400,
        f"{role} mode is not exactly 0400",
    )


def _validate_v2_withdrawal_report(report_path: Path) -> str:
    path = _absolute(report_path)
    _require(
        path == _CANONICAL_V2_WITHDRAWAL_REPORT,
        "v2 withdrawal report is outside the canonical sealed held-v4 audit path",
    )
    _require(path.resolve() == path, "v2 withdrawal-report path is not canonical")
    _require_exact_mode_0400(path, "v2 withdrawal report")
    observed = _read_regular_bytes(path, "v2 design-withdrawal report")
    expected, expected_bytes, file_sha256, artifact_sha256 = (
        _expected_v2_withdrawal_report()
    )
    _require(
        observed == expected_bytes, "v2 withdrawal report is not exact canonical JSON"
    )
    parsed, observed_file_sha256 = _load_json(path, "v2 design-withdrawal report")
    _require(parsed == expected, "v2 withdrawal-report schema or evidence changed")
    _require(
        observed_file_sha256 == file_sha256
        and parsed.get("artifact_sha256") == artifact_sha256
        and _artifact_sha256(parsed) == artifact_sha256,
        "v2 withdrawal-report checksum changed",
    )
    _require(
        not os.path.lexists(_CANONICAL_HELD_V2_ROOT),
        "held-v2 root now exists; the source-only withdrawal evidence is stale",
    )
    return observed_file_sha256


def _validate_v3_boundary_incident_report(report_path: Path) -> str:
    path = _absolute(report_path)
    _require(
        path == _CANONICAL_V3_BOUNDARY_INCIDENT_REPORT,
        "v3 boundary-incident report is outside the canonical sealed held-v4 audit path",
    )
    _require(
        path.resolve() == path,
        "v3 boundary-incident report path is not canonical",
    )
    _require_exact_mode_0400(path, "v3 boundary-incident report")
    observed = _read_regular_bytes(path, "v3 pre-lock boundary-incident report")
    expected, expected_bytes, file_sha256, artifact_sha256 = (
        _expected_v3_boundary_incident_report()
    )
    _require(
        observed == expected_bytes,
        "v3 boundary-incident report is not exact canonical JSON",
    )
    parsed, observed_file_sha256 = _load_json(
        path, "v3 pre-lock boundary-incident report"
    )
    _require(
        parsed == expected,
        "v3 boundary-incident report schema or evidence changed",
    )
    _require(
        observed_file_sha256 == file_sha256
        and parsed.get("artifact_sha256") == artifact_sha256
        and _artifact_sha256(parsed) == artifact_sha256,
        "v3 boundary-incident report checksum changed",
    )
    _require(
        not os.path.lexists(_CANONICAL_HELD_V3_ROOT),
        "held-v3 root now exists; the pre-lock withdrawal evidence is stale",
    )
    return observed_file_sha256


def _validate_v4_lock(lock_path: Path) -> dict[str, Any]:
    path = _absolute(lock_path)
    _require(path == _CANONICAL_V4_LOCK, "v4 lock path changed")
    _require(path.resolve() == path, "v4 lock path is not canonical")
    _require_exact_mode_0400(path, "v4 calibration lock")
    lock, file_sha256 = _load_json(path, "v4 calibration lock")
    _require(
        file_sha256 == EXPECTED_V4_LOCK_FILE_SHA256,
        "v4 calibration-lock file checksum changed",
    )
    _require(
        lock.get("artifact_sha256") == EXPECTED_V4_LOCK_ARTIFACT_SHA256
        and _artifact_sha256(lock) == EXPECTED_V4_LOCK_ARTIFACT_SHA256,
        "v4 calibration-lock semantic checksum changed",
    )
    _require(
        lock.get("protocol_id") == "deform360-held-online-belief-v4"
        and isinstance(lock.get("immutable_bindings"), Mapping)
        and len(lock["immutable_bindings"]) == 110,
        "v4 calibration-lock identity or binding count changed",
    )
    return lock


def _validate_v4_execution_withdrawal_report(report_path: Path) -> str:
    path = _absolute(report_path)
    _require(
        path == _CANONICAL_V4_EXECUTION_WITHDRAWAL_REPORT,
        "v4 execution-withdrawal report path changed",
    )
    _require(path.resolve() == path, "v4 withdrawal-report path is not canonical")
    _require_exact_mode_0400(path, "v4 execution-withdrawal report")
    report, file_sha256 = _load_json(path, "v4 execution-withdrawal report")
    _require(
        file_sha256 == EXPECTED_V4_REPORT_FILE_SHA256,
        "v4 execution-withdrawal report file checksum changed",
    )
    _require(
        report.get("artifact_sha256") == EXPECTED_V4_REPORT_ARTIFACT_SHA256
        and _artifact_sha256(report) == EXPECTED_V4_REPORT_ARTIFACT_SHA256,
        "v4 execution-withdrawal report semantic checksum changed",
    )
    _require(
        report.get("protocol_id") == "deform360-held-online-belief-v4"
        and report.get("replacement_protocol_id") == "deform360-held-online-belief-v5"
        and report.get("disposition")
        == "WITHDRAWN_AFTER_FRAME_ZERO_BEFORE_PHYSICAL_PREDICTION",
        "v4 execution-withdrawal disposition changed",
    )
    counts = report.get("execution_counts")
    _require(
        isinstance(counts, Mapping)
        and counts.get("case_attempt_count") == 2
        and counts.get("calibration_decision_count") == 0
        and counts.get("calibration_lock_count") == 1
        and counts.get("confirmation_lock_count") == 0
        and counts.get("deployed_snapshot_count") == 1
        and counts.get("deployment_count") == 1
        and counts.get("frame_zero_bundle_count") == 2
        and counts.get("frame_zero_manifest_count") == 2
        and counts.get("physical_builder_invocation_count") == 2
        and counts.get("physical_prediction_artifact_count") == 0
        and counts.get("formal_physical_prediction_count") == 0
        and counts.get("formal_online_prediction_count") == 0
        and counts.get("physical_prior_seal_count") == 0
        and counts.get("online_prediction_seal_count") == 0
        and counts.get("outcome_api_operation_count") == 0
        and counts.get("outcome_created_count") == 0
        and counts.get("outcome_permit_count") == 0
        and counts.get("prefix_authorization_count") == 0
        and counts.get("shard_start_count") == 2
        and counts.get("target_operation_count") == 0
        and counts.get("outcome_read_count") == 0,
        "v4 execution-withdrawal census changed",
    )
    reuse = report.get("reuse")
    _require(
        isinstance(reuse, Mapping)
        and reuse.get("v4_frame_zero_artifacts_reused_by_v5") is False
        and reuse.get("v4_physical_or_online_predictions_reused_by_v5") is False
        and reuse.get("v5_requires_fresh_absent_held_root") is True,
        "v4-to-v5 no-reuse declaration changed",
    )
    return file_sha256


def _validate_v5_lock(lock_path: Path) -> dict[str, Any]:
    path = _absolute(lock_path)
    _require(path == _CANONICAL_V5_LOCK, "v5 lock path changed")
    _require(path.resolve() == path, "v5 lock path is not canonical")
    _require_exact_mode_0400(path, "v5 calibration lock")
    lock, file_sha256 = _load_json(path, "v5 calibration lock")
    _require(
        file_sha256 == EXPECTED_V5_LOCK_FILE_SHA256,
        "v5 calibration-lock file checksum changed",
    )
    _require(
        lock.get("artifact_sha256") == EXPECTED_V5_LOCK_ARTIFACT_SHA256
        and _artifact_sha256(lock) == EXPECTED_V5_LOCK_ARTIFACT_SHA256,
        "v5 calibration-lock semantic checksum changed",
    )
    _require(
        lock.get("protocol_id") == "deform360-held-online-belief-v5"
        and isinstance(lock.get("immutable_bindings"), Mapping)
        and len(lock["immutable_bindings"]) == EXPECTED_V5_LOCK_BINDING_COUNT,
        "v5 calibration-lock identity or binding count changed",
    )
    return lock


def _validate_v5_outcome_withdrawal_report(report_path: Path) -> str:
    """Bind only the exact metadata-only v5 withdrawal report.

    This validator deliberately opens no v5 execution artifact other than the
    sealed lock and report.  The report's complete evidence inventory was
    produced by the one-purpose forensic sealer and is fixed here by both its
    byte digest and semantic artifact digest.
    """

    path = _absolute(report_path)
    _require(
        path == _CANONICAL_V5_OUTCOME_WITHDRAWAL_REPORT,
        "v5 outcome-withdrawal report path changed",
    )
    _require(path.resolve() == path, "v5 withdrawal-report path is not canonical")
    _require_exact_mode_0400(path, "v5 outcome-withdrawal report")
    payload = _read_regular_bytes(path, "v5 outcome-withdrawal report")
    _require(
        len(payload) == EXPECTED_V5_REPORT_SIZE_BYTES
        and hashlib.sha256(payload).hexdigest() == EXPECTED_V5_REPORT_FILE_SHA256,
        "v5 outcome-withdrawal report file checksum changed",
    )
    report, file_sha256 = _load_json(path, "v5 outcome-withdrawal report")
    canonical = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _require(payload == canonical, "v5 withdrawal report is not canonical JSON")
    _require(
        report.get("artifact_sha256") == EXPECTED_V5_REPORT_ARTIFACT_SHA256
        and _artifact_sha256(report) == EXPECTED_V5_REPORT_ARTIFACT_SHA256,
        "v5 outcome-withdrawal report semantic checksum changed",
    )
    _require(
        report.get("artifact_kind") == "Deform360HeldProtocolExecutionWithdrawalReport"
        and report.get("schema_version") == 1
        and report.get("protocol_id") == "deform360-held-online-belief-v5"
        and report.get("replacement_protocol_id") == "deform360-held-online-belief-v6"
        and report.get("disposition")
        == "WITHDRAWN_DURING_FIRST_TARGET_OPERATION_BEFORE_ANY_COMPLETED_OUTCOME",
        "v5 outcome-withdrawal report identity or disposition changed",
    )
    _require(
        report.get("cause")
        == {
            "classification": "PROPAGATED_FRAME_ZERO_MASK_SEAL_MISMATCH",
            "exception_message": (
                "propagated frame-zero mask differs from seal: brics-odroid-001_cam0"
            ),
            "failed_camera": "brics-odroid-001_cam0",
            "failed_case": "002-rope-silk-ep0003",
            "failure_phase": "first calibration target reconstruction operation",
            "terminal_log": {
                "path": "calibration-outcomes.console.log",
                "sha256": (
                    "7f48d2ee1291d37f051a9422e2217fafe7f08d7438285b8b4f4ee14e5d93ab71"
                ),
                "size": 3270,
            },
        },
        "v5 outcome-withdrawal cause changed",
    )
    _require(
        report.get("execution_counts")
        == {
            "calibration_case_execution_count": 15,
            "calibration_decision_count": 0,
            "calibration_lock_count": 1,
            "calibration_score_evidence_count": 0,
            "confirmation_case_execution_count": 0,
            "confirmation_lock_count": 0,
            "confirmation_prediction_seal_count": 0,
            "deployed_snapshot_count": 1,
            "formal_online_prediction_count": 15,
            "formal_physical_prediction_count": 15,
            "frame_zero_bundle_count": 15,
            "frame_zero_manifest_count": 15,
            "online_prediction_seal_count": 15,
            "outcome_created_count": 0,
            "outcome_permit_count": 1,
            "outcome_phase_claim_count": 1,
            "outcome_read_count": 0,
            "partial_target_case_directory_count": 1,
            "partial_target_staging_directory_count": 12,
            "partial_target_staging_file_count": 28,
            "physical_prior_seal_count": 15,
            "prefix_authorization_count": 15,
            "shard_start_count": 2,
            "staged_camera_video_count": 8,
            "target_operation_completed_count": 0,
            "target_operation_failed_count": 1,
            "target_operation_planned_count": 15,
            "target_operation_started_count": 1,
            "target_reconstruction_artifact_count": 0,
        },
        "v5 outcome-withdrawal execution census changed",
    )
    _require(
        report.get("information_boundary")
        == {
            "all_15_calibration_predictions_exist_and_are_sealed": True,
            "all_15_prediction_artifact_sets_revalidated_bytewise_for_outcome_permit": True,
            "calibration_gate_or_metric_created_or_read": False,
            "confirmation_payload_read": False,
            "first_case_online_prediction_arrays_decoded_before_target_callback": True,
            "forensic_audit_disclosed_arrays_images_masks_metrics_or_protected_values": False,
            "forensic_audit_method": (
                "filenames/stat metadata and stable O_NOFOLLOW SHA-256 byte streams "
                "only; no file was deserialized and no video was decoded"
            ),
            "future_tactile_read": False,
            "later_case_online_prediction_arrays_decoded": False,
            "object_future_depth_read": False,
            "object_future_rgb_read": "POSSIBLE_WITHIN_FIRST_CALIBRATION_CASE_ONLY",
            "object_future_rgb_read_case_upper_bound": 1,
            "object_future_rgb_read_reason": (
                "the first target callback started and eight staged video files "
                "exist; the metadata-only audit cannot establish the exact source "
                "frames decoded before failure"
            ),
            "object_future_tracking_read": False,
            "official_target_reconstruction_created": False,
            "partial_target_source_staging_created": True,
            "partial_target_source_staging_scope": (
                "one calibration case, eight camera videos and timestamps/metadata, "
                "camera calibration, and robot metadata/archive"
            ),
            "tactile_read": False,
        },
        "v5 outcome-withdrawal information boundary changed",
    )
    _require(
        report.get("reuse")
        == {
            "v5_evidence_may_be_used_by_v6_only_as_immutable_lineage": True,
            "v5_execution_artifacts_reused_by_v6": False,
            "v5_partial_target_staging_reused_by_v6": False,
            "v5_physical_or_online_predictions_reused_by_v6": False,
            "v5_score_or_gate_available_for_reuse": False,
            "v6_requires_fresh_absent_held_root": True,
            "v6_requires_fresh_predictions_and_outcome_phase": True,
        },
        "v5-to-v6 no-reuse declaration changed",
    )
    evidence = report.get("evidence")
    _require(isinstance(evidence, Mapping), "v5 withdrawal evidence is absent")
    root_files = evidence.get("root_file_inventory")
    _require(isinstance(root_files, list), "v5 root-file inventory is absent")
    indexed = {
        str(record.get("path")): record
        for record in root_files
        if isinstance(record, Mapping)
    }
    _require(
        indexed.get("calibration-lock.json")
        == {
            "path": "calibration-lock.json",
            "sha256": EXPECTED_V5_LOCK_FILE_SHA256,
            "size": 16_956,
        },
        "v5 withdrawal report no longer binds the exact v5 lock",
    )
    _require(
        _CANONICAL_HELD_V5_ROOT.is_dir()
        and not _CANONICAL_HELD_V5_ROOT.is_symlink()
        and _CANONICAL_HELD_V5_ROOT.resolve() == _CANONICAL_HELD_V5_ROOT
        and os.lstat(_CANONICAL_HELD_V5_ROOT).st_mode & 0o222 == 0,
        "sealed held-v5 root is absent, aliased, or writable",
    )
    return file_sha256


def _require_deployed_module_provenance(
    code: Path, module_names: Sequence[str]
) -> None:
    source = (code / "src").resolve()
    for module_name in module_names:
        module = sys.modules.get(module_name)
        _require(
            module is not None,
            f"expected deployed module was not imported: {module_name}",
        )
        module_file = getattr(module, "__file__", None)
        _require(
            bool(module_file), f"deployed module has no source path: {module_name}"
        )
        observed = Path(str(module_file)).resolve()
        try:
            observed.relative_to(source)
        except ValueError as error:
            raise ValueError(
                f"deployed module imported outside the snapshot: {module_name}: {observed}"
            ) from error


def _run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> tuple[int, bytes]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("GIT_CONFIG_KEY_") or key.startswith("GIT_CONFIG_VALUE_"):
            environment.pop(key)
    for key in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_COUNT",
        "GIT_DIR",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CEILING_DIRECTORIES": os.fspath(root.parent),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    completed = subprocess.run(
        [
            os.fspath(_GIT_EXECUTABLE),
            "--no-replace-objects",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "core.excludesFile=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.untrackedCache=false",
            *arguments,
        ],
        cwd=root,
        env=environment,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    _require(
        completed.returncode in allowed_returncodes,
        "Git provenance command failed: "
        f"git {' '.join(arguments)}: {completed.stderr.decode('utf-8', 'replace').strip()}",
    )
    return completed.returncode, completed.stdout


def _enumerate_worktree_files(root: Path) -> tuple[list[str], list[Path]]:
    relative_files: list[str] = []
    directories: list[Path] = [root]
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            if directory == root and entry.name == ".git":
                continue
            observed = entry.stat(follow_symlinks=False)
            path = Path(entry.path)
            _require(
                not stat.S_ISLNK(observed.st_mode),
                f"deployed method contains a symlink: {path}",
            )
            if stat.S_ISDIR(observed.st_mode):
                directories.append(path)
                stack.append(path)
            elif stat.S_ISREG(observed.st_mode):
                relative = path.relative_to(root).as_posix()
                _ensure_unprotected_path(path, "deployed method file")
                _require(
                    "__pycache__" not in path.parts
                    and path.suffix not in {".pyc", ".pyo"},
                    f"deployed method contains generated Python state: {relative}",
                )
                relative_files.append(relative)
            else:
                raise ValueError(f"deployed method contains a special file: {path}")
    return sorted(relative_files), directories


def _require_immutable_repository(
    root: Path, worktree_directories: Sequence[Path]
) -> None:
    stack = [root / ".git"]
    git_paths: list[Path] = []
    while stack:
        directory = stack.pop()
        git_paths.append(directory)
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            observed = entry.stat(follow_symlinks=False)
            path = Path(entry.path)
            _require(
                not stat.S_ISLNK(observed.st_mode),
                f"deployed Git repository contains a symlink: {path}",
            )
            if stat.S_ISDIR(observed.st_mode):
                stack.append(path)
            elif not stat.S_ISREG(observed.st_mode):
                raise ValueError(
                    f"deployed Git repository contains a special file: {path}"
                )
            git_paths.append(path)
    for path in [*worktree_directories, *git_paths]:
        mode = os.lstat(path).st_mode
        _require(mode & 0o222 == 0, f"deployed repository is writable: {path}")


def _parse_git_tree(raw: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, path_bytes = entry.split(b"\t", 1)
            mode_bytes, kind, object_bytes, size_bytes = metadata.split(b" ", 3)
            relative = path_bytes.decode("utf-8")
            mode = mode_bytes.decode("ascii")
            object_id = object_bytes.decode("ascii")
            size = int(size_bytes)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Git tree manifest is malformed") from error
        parts = relative.split("/")
        _require(
            relative
            and not relative.startswith("/")
            and all(part not in {"", ".", ".."} for part in parts),
            "Git tree contains an unsafe path",
        )
        _ensure_unprotected_path(Path(relative), "tracked method file")
        _require(kind == b"blob", f"non-blob Git entry is forbidden: {relative}")
        _require(mode in {"100644", "100755"}, f"unsafe Git mode for {relative}")
        _require(
            _GIT_OBJECT_ID.fullmatch(object_id) is not None, "invalid Git object id"
        )
        _require(size >= 0, "invalid Git blob size")
        records.append(
            {
                "git_object": object_id,
                "mode": mode,
                "path": relative,
                "size_bytes": size,
            }
        )
    _require(bool(records), "deployed Git tree is empty")
    _require(
        [record["path"] for record in records]
        == sorted(record["path"] for record in records),
        "Git tree file order changed",
    )
    _require(
        len({record["path"] for record in records}) == len(records),
        "Git tree contains duplicate paths",
    )
    return records


def _git_blob_digest(path: Path, size: int, algorithm: str) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_digest = hashlib.new(algorithm)
    git_digest.update(f"blob {size}\0".encode("ascii"))
    source = _absolute(path)
    before = os.lstat(source)
    _require(stat.S_ISREG(before.st_mode), f"tracked path is not a file: {source}")
    _require(before.st_mode & 0o222 == 0, f"tracked file is writable: {source}")
    _require(before.st_size == size, f"tracked file size changed: {source}")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        after = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
            f"tracked file changed while opening: {source}",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(block)
                git_digest.update(block)
    finally:
        os.close(descriptor)
    return sha256.hexdigest(), git_digest.hexdigest()


def _validate_deployed_git_repository(root: Path) -> dict[str, Any]:
    code = _absolute(root)
    _require(
        stat.S_ISREG(os.lstat(_GIT_EXECUTABLE).st_mode),
        f"pinned Git executable is unavailable: {_GIT_EXECUTABLE}",
    )
    observed_root = os.lstat(code)
    _require(stat.S_ISDIR(observed_root.st_mode), "invalid deployed code root")
    git_dir = code / ".git"
    _require(
        stat.S_ISDIR(os.lstat(git_dir).st_mode),
        "deployed code must be a non-linked Git clone",
    )
    _require_immutable_repository(code, ())
    git_config = _read_regular_bytes(git_dir / "config", "deployed Git config")
    try:
        git_config_text = git_config.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("deployed Git config is not UTF-8") from error
    _require(
        re.search(
            r"^\s*\[\s*include(?:if)?\b", git_config_text, re.IGNORECASE | re.MULTILINE
        )
        is None,
        "deployed Git config may not include external configuration",
    )
    _require(
        "partialclone" not in git_config_text.lower(),
        "deployed Git repository may not be a partial clone",
    )
    _require(
        not (git_dir / "objects" / "info" / "alternates").exists(),
        "deployed Git clone may not use an alternate object store",
    )
    _require(
        not (git_dir / "objects" / "info" / "http-alternates").exists()
        and not (git_dir / "commondir").exists()
        and not (git_dir / "shallow").exists(),
        "deployed Git clone may not use external, common, or shallow objects",
    )
    _, top_level_raw = _run_git(code, ("rev-parse", "--show-toplevel"))
    _require(
        Path(top_level_raw.decode("utf-8").strip()) == code,
        "deployed Git top level changed",
    )
    _, git_dir_raw = _run_git(code, ("rev-parse", "--absolute-git-dir"))
    _require(
        Path(git_dir_raw.decode("utf-8").strip()) == git_dir,
        "deployed Git directory is not local to the snapshot",
    )
    _, common_dir_raw = _run_git(code, ("rev-parse", "--git-common-dir"))
    common_dir = Path(common_dir_raw.decode("utf-8").strip())
    if not common_dir.is_absolute():
        common_dir = _absolute(code / common_dir)
    _require(common_dir == git_dir, "deployed Git common directory is external")
    symbolic_returncode, _ = _run_git(
        code,
        ("symbolic-ref", "-q", "HEAD"),
        allowed_returncodes=frozenset({0, 1}),
    )
    _require(symbolic_returncode == 1, "deployed repository HEAD is not detached")
    _, head_raw = _run_git(code, ("rev-parse", "--verify", "HEAD"))
    head = head_raw.decode("ascii").strip()
    _require(_GIT_OBJECT_ID.fullmatch(head) is not None, "invalid deployed HEAD")
    _require(code.name == f"code-{head}", "deployed directory is not named code-$HEAD")
    _, object_format_raw = _run_git(code, ("rev-parse", "--show-object-format"))
    object_format = object_format_raw.decode("ascii").strip()
    _require(object_format in {"sha1", "sha256"}, "unsupported Git object format")
    expected_length = 40 if object_format == "sha1" else 64
    _require(len(head) == expected_length, "HEAD length disagrees with Git format")
    _, status = _run_git(
        code,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    )
    _require(status == b"", "deployed worktree is dirty or has untracked files")
    _, tree_raw = _run_git(code, ("ls-tree", "-r", "-l", "-z", "--full-tree", "HEAD"))
    git_records = _parse_git_tree(tree_raw)
    worktree_files, worktree_directories = _enumerate_worktree_files(code)
    tracked_files = [str(record["path"]) for record in git_records]
    _require(
        worktree_files == tracked_files,
        "deployed worktree differs from the exact tracked file set; "
        f"missing={sorted(set(tracked_files) - set(worktree_files))[:8]!r}, "
        f"unexpected={sorted(set(worktree_files) - set(tracked_files))[:8]!r}",
    )
    _require_immutable_repository(code, worktree_directories)
    snapshot_records: list[dict[str, Any]] = []
    for record in git_records:
        relative = str(record["path"])
        sha256, git_object = _git_blob_digest(
            code / relative,
            int(record["size_bytes"]),
            object_format,
        )
        _require(
            git_object == record["git_object"],
            f"deployed bytes disagree with Git object: {relative}",
        )
        snapshot_records.append(
            {
                "path": relative,
                "sha256": sha256,
                "size_bytes": int(record["size_bytes"]),
            }
        )
    _, commit_payload = _run_git(code, ("cat-file", "commit", "HEAD"))
    _require(
        commit_payload.startswith(b"tree "),
        "deployed HEAD commit object is malformed",
    )
    return {
        "method_head": head,
        "method_head_literal": hashlib.sha256(head.encode("ascii")).hexdigest(),
        "method_commit_object": hashlib.sha256(commit_payload).hexdigest(),
        "method_git_tree_manifest": _canonical_sha256(git_records),
        "method_deployed_snapshot_tree": _canonical_sha256(snapshot_records),
        "tracked_file_count": len(git_records),
        "tracked_files": tracked_files,
    }


def _validate_v1_lock(lock_path: Path) -> dict[str, Any]:
    path = _absolute(lock_path)
    _require(path == _CANONICAL_V1_LOCK, "v1 lock path changed")
    _require(path.resolve() == path, "v1 lock path is not canonical")
    _require_exact_mode_0400(path, "v1 calibration lock")
    lock, file_sha256 = _load_json(path, "v1 calibration lock")
    _require(file_sha256 == EXPECTED_V1_LOCK_FILE_SHA256, "v1 lock file changed")
    _require(
        lock.get("artifact_sha256") == EXPECTED_V1_LOCK_ARTIFACT_SHA256
        and _artifact_sha256(lock) == EXPECTED_V1_LOCK_ARTIFACT_SHA256,
        "v1 lock semantic checksum changed",
    )
    _require(
        lock.get("protocol_id") == "deform360-held-online-belief-v1",
        "parent lock is not exact held v1",
    )
    return lock


def _validate_v1_report(report_path: Path) -> str:
    path = _absolute(report_path)
    _require(
        path == _CANONICAL_V1_REPORT,
        "v1 report path changed",
    )
    _require(path.resolve() == path, "v1 report path is not canonical")
    _require_exact_mode_0400(path, "v1 pre-outcome feasibility report")
    report, file_sha256 = _load_json(path, "v1 pre-outcome feasibility report")
    _require(
        file_sha256 == EXPECTED_V1_REPORT_FILE_SHA256,
        "v1 feasibility-report file checksum changed",
    )
    _require(
        report.get("artifact_sha256") == EXPECTED_V1_REPORT_ARTIFACT_SHA256
        and _artifact_sha256(report) == EXPECTED_V1_REPORT_ARTIFACT_SHA256,
        "v1 feasibility-report semantic checksum changed",
    )
    _require(
        report.get("protocol_id") == "deform360-held-online-belief-v1"
        and report.get("disposition") == "ABANDONED_PREOUTCOME"
        and report.get("counts") == EXPECTED_V1_COUNTS,
        "v1 feasibility-report disposition or exact census changed",
    )
    boundary = report.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("source_only_feasibility_evidence") is True
        and boundary.get("outcome_phase_authorized") is False
        and boundary.get("outcome_created") is False
        and boundary.get("outcome_read") is False
        and boundary.get("target_data_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("confirmation_payload_read") is False,
        "v1 information boundary changed",
    )
    return file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-lock", type=Path, required=True)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--v2-withdrawal-report", type=Path, required=True)
    parser.add_argument("--v3-boundary-incident-report", type=Path, required=True)
    parser.add_argument("--v4-lock", type=Path, required=True)
    parser.add_argument("--v4-execution-withdrawal-report", type=Path, required=True)
    parser.add_argument("--v5-lock", type=Path, required=True)
    parser.add_argument("--v5-outcome-withdrawal-report", type=Path, required=True)
    parser.add_argument("--deployed-code", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument(
        "--verify-existing-lock",
        action="store_true",
        help="verify --output-lock without creating or modifying any file",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    _require(
        sys.flags.isolated == 1,
        "lock preparation and verification must run with Python isolated mode (-I)",
    )
    _require(
        sys.flags.dont_write_bytecode == 1 and sys.pycache_prefix == _PYCACHE_PREFIX,
        "lock operator lacks the isolated no-adjacent-pyc runtime flags",
    )
    _require(
        not os.path.lexists("/nonexistent") and not os.path.lexists(_PYCACHE_PREFIX),
        "reserved held-v6 pycache prefix is no longer unavailable",
    )
    _require(
        dict(os.environ) == _PINNED_OPERATOR_ENVIRONMENT,
        "lock operator environment is not the exact source-only allowlist",
    )
    code = _absolute(arguments.deployed_code)
    _require(
        code.parent == _CANONICAL_HELD_V6_ROOT,
        "deployed code is outside the canonical fresh held-v6 root",
    )
    _require(
        _CANONICAL_HELD_V6_ROOT.is_dir()
        and not _CANONICAL_HELD_V6_ROOT.is_symlink()
        and _CANONICAL_HELD_V6_ROOT.resolve() == _CANONICAL_HELD_V6_ROOT,
        "canonical held-v6 root is absent, a symlink, or aliased",
    )
    _require(
        _absolute(arguments.output_lock) == _CANONICAL_V6_LOCK,
        "v6 lock path is outside the canonical fresh held-v6 root",
    )
    if not arguments.verify_existing_lock:
        with os.scandir(_CANONICAL_HELD_V6_ROOT) as entries:
            root_entries = sorted(entries, key=lambda entry: entry.name)
        _require(
            [entry.name for entry in root_entries] == [code.name]
            and root_entries[0].is_dir(follow_symlinks=False),
            "fresh held-v6 root contains anything except the deployed code snapshot",
        )
    expected_operator = code / "scripts" / "held" / "prepare_deform360_v6_lock.py"
    operator = _absolute(Path(__file__))
    _require(
        not Path(__file__).is_symlink() and operator == expected_operator,
        "lock operator is not the tracked deployed v6 operator",
    )
    v1_lock = _validate_v1_lock(arguments.v1_lock)
    v1_report_file_sha256 = _validate_v1_report(arguments.v1_report)
    provenance_before = _validate_deployed_git_repository(code)
    v2_report_file_sha256 = _validate_v2_withdrawal_report(
        arguments.v2_withdrawal_report
    )
    v3_report_file_sha256 = _validate_v3_boundary_incident_report(
        arguments.v3_boundary_incident_report
    )
    v4_lock = _validate_v4_lock(arguments.v4_lock)
    v4_report_file_sha256 = _validate_v4_execution_withdrawal_report(
        arguments.v4_execution_withdrawal_report
    )
    v5_lock = _validate_v5_lock(arguments.v5_lock)
    v5_report_file_sha256 = _validate_v5_outcome_withdrawal_report(
        arguments.v5_outcome_withdrawal_report
    )

    # Imports are permitted only after the deployed bytes have been matched to
    # a clean, detached Git commit.  Disabling bytecode prevents this audit
    # itself from changing the snapshot it is about to bind.
    sys.dont_write_bytecode = True
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "bayesian_phystwin" or name.startswith("bayesian_phystwin.")
    )
    _require(
        not preloaded, f"bayesian_phystwin was preloaded before audit: {preloaded}"
    )
    sys.path.insert(0, str(code / "src"))

    from bayesian_phystwin.deform360_frame_zero_assets import (  # noqa: PLC0415
        FrameZeroAssetConfig,
        artifact_sha256,
    )
    from bayesian_phystwin.deform360_frame_zero_semantic_gate import (  # noqa: PLC0415
        FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
        FRAME_ZERO_SIGLIP2_MODEL_REVISION,
        FRAME_ZERO_SIGLIP2_MODEL_TREE_SHA256,
        FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256,
    )
    from bayesian_phystwin.deform360_held_online_prefix import (  # noqa: PLC0415
        HELD_FRAME_ZERO_SAM2,
        HELD_OBSERVATION_CONFIG,
        HELD_RBF_CONFIG,
        HELD_UNCERTAINTY_CONFIG,
    )
    from bayesian_phystwin.deform360_held_outcome_scoring import (  # noqa: PLC0415
        OUTCOME_RECONSTRUCTION_CONTRACT,
    )
    from bayesian_phystwin.deform360_held_physical_prior import (  # noqa: PLC0415
        HELD_PYTHON_RUNTIME,
        HELD_PHYSICAL_NUMERIC_CONTRACT,
        UPSTREAM_FILE_SHA256,
        UPSTREAM_LOCK_BINDING_BY_PATH,
        UPSTREAM_RUNTIME_BUNDLE_CONTRACT,
        validate_python_runtime,
    )
    from bayesian_phystwin.deform360_held_protocol import (  # noqa: PLC0415
        CALIBRATION_GATE,
        CONFIRMATION_GATE,
        DATASET_REVISION,
        METRIC_LOCK,
        PRIMARY_METHOD,
        PROTOCOL_ID,
        REMOTE_INVENTORY_COMBINED_SHA256,
        REQUIRED_IMMUTABLE_BINDING_KEYS,
        SOURCE_FEASIBILITY_AMENDMENT_CONTRACT,
        create_held_protocol_lock,
        held_contract_sha256,
        load_held_protocol_lock,
    )
    from bayesian_phystwin.deform360_raw_camera_cycle_uncertainty import (  # noqa: PLC0415
        RawCameraCycleUncertaintyConfig,
    )
    from bayesian_phystwin.deform360_raw_camera_observation import (  # noqa: PLC0415
        ALLTRACKER_CHECKPOINT_SHA256,
        ALLTRACKER_MOLMOMOTION_REVISION,
        ALLTRACKER_RUNTIME_SOURCE_SHA256,
        ALLTRACKER_SOURCE_TREE,
    )
    from bayesian_phystwin.deform360_robot_kinematics import (  # noqa: PLC0415
        ROBOT_KINEMATICS_WINDOW_CONTRACT,
        ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
    )

    _require_deployed_module_provenance(
        code,
        (
            "bayesian_phystwin",
            "bayesian_phystwin.deform360_frame_zero_assets",
            "bayesian_phystwin.deform360_frame_zero_semantic_gate",
            "bayesian_phystwin.deform360_held_online_prefix",
            "bayesian_phystwin.deform360_held_outcome_scoring",
            "bayesian_phystwin.deform360_held_physical_prior",
            "bayesian_phystwin.deform360_held_protocol",
            "bayesian_phystwin.deform360_raw_camera_cycle_uncertainty",
            "bayesian_phystwin.deform360_raw_camera_observation",
            "bayesian_phystwin.deform360_robot_kinematics",
        ),
    )

    provenance_after = _validate_deployed_git_repository(code)
    _require(
        provenance_after == provenance_before,
        "deployed repository changed while loading its contracts",
    )
    _require(PROTOCOL_ID == "deform360-held-online-belief-v6", "not v6 code")
    _require_deployed_semantic_binding_exports(
        model_tree_sha256=FRAME_ZERO_SIGLIP2_MODEL_TREE_SHA256,
        model_revision=FRAME_ZERO_SIGLIP2_MODEL_REVISION,
        transformers_source_aggregate_sha256=(
            FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256
        ),
        semantic_gate_contract_sha256=FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
    )
    old_bindings = v1_lock.get("immutable_bindings")
    _require(isinstance(old_bindings, Mapping), "parent lock bindings are absent")
    old_bindings = {str(key): str(value) for key, value in old_bindings.items()}
    required = set(REQUIRED_IMMUTABLE_BINDING_KEYS)
    local_files = set(LOCAL_FILE_BINDINGS)
    binding_groups = (
        set(INHERITED_EXTERNAL_BINDING_KEYS),
        set(V6_PINNED_EXTERNAL_BINDING_KEYS),
        local_files,
        set(LOCAL_CONTRACT_BINDING_KEYS),
        set(METHOD_PROVENANCE_BINDING_KEYS),
        {
            "v1_preoutcome_feasibility_report",
            "v2_design_withdrawal_report",
            "v3_prelock_boundary_incident_report",
            "v4_execution_withdrawal_report",
            "v5_outcome_withdrawal_report",
        },
    )
    for index, group in enumerate(binding_groups):
        for other in binding_groups[index + 1 :]:
            _require(
                group.isdisjoint(other),
                f"operator binding classifications overlap: {sorted(group & other)!r}",
            )
    expected_partition = set().union(*binding_groups)
    _require(
        expected_partition == required,
        "operator binding classification is stale; "
        f"missing={sorted(required - expected_partition)!r}, "
        f"unexpected={sorted(expected_partition - required)!r}",
    )
    expected_v1_keys = required - set(V6_ONLY_BINDING_KEYS)
    _require(set(old_bindings) == expected_v1_keys, "exact v1 binding keys changed")
    _require(
        set(V6_ONLY_BINDING_KEYS) == required - set(old_bindings)
        and len(V6_ONLY_BINDING_KEYS) == EXPECTED_V6_MIGRATION_KEY_COUNT,
        "exact v6 migration-key delta changed",
    )
    _require(
        len(required) == EXPECTED_V6_BINDING_COUNT,
        "exact v6 immutable-binding count changed",
    )
    bindings = {
        key: old_bindings[key] for key in sorted(INHERITED_EXTERNAL_BINDING_KEYS)
    }
    bindings.update(V6_PINNED_EXTERNAL_BINDING_VALUES)
    runtime_identity = validate_python_runtime(
        HELD_PYTHON_RUNTIME / "bin" / "python",
        bindings,
    )
    _require(
        runtime_identity["python_pip_freeze_sorted_sha256"]
        == EXPECTED_V5_PIP_FREEZE_SHA256
        and runtime_identity["runtime_manifest_sha256"]
        == EXPECTED_V5_RUNTIME_MANIFEST_FILE_SHA256,
        "dedicated v5 Python runtime identity changed",
    )

    # Cross-check external identities that are also named by deployed source
    # constants.  Their values must still be byte-for-byte inherited from v1.
    _require(
        hashlib.sha256(DATASET_REVISION.encode("ascii")).hexdigest()
        == bindings["dataset_revision_literal"],
        "dataset revision literal diverged from v1",
    )
    _require(
        REMOTE_INVENTORY_COMBINED_SHA256
        == bindings["remote_confirmation_inventory_combined"],
        "remote confirmation inventory diverged from v1",
    )
    _require(
        ALLTRACKER_CHECKPOINT_SHA256 == bindings["alltracker_checkpoint"]
        and hashlib.sha256(ALLTRACKER_MOLMOMOTION_REVISION.encode("ascii")).hexdigest()
        == bindings["alltracker_molmomotion_revision_literal"]
        and hashlib.sha256(ALLTRACKER_SOURCE_TREE.encode("ascii")).hexdigest()
        == bindings["alltracker_provenance_tree_literal"]
        and ALLTRACKER_RUNTIME_SOURCE_SHA256 == bindings["alltracker_runtime_tree"],
        "AllTracker external identity diverged from v1",
    )
    _require(
        HELD_FRAME_ZERO_SAM2["checkpoint_sha256"] == bindings["sam2_checkpoint"],
        "SAM2 checkpoint identity diverged from v1",
    )
    for relative, binding_key in UPSTREAM_LOCK_BINDING_BY_PATH.items():
        _require(
            UPSTREAM_FILE_SHA256[relative] == bindings[binding_key],
            f"upstream external identity diverged from v1: {relative}",
        )

    for key, relative in LOCAL_FILE_BINDINGS.items():
        path = code / relative
        _require(relative in provenance_before["tracked_files"], "unreachable")
        bindings[key] = _sha256_file(path, f"local method binding {key}")

    bindings.update(
        {
            "frame_zero_default_config": artifact_sha256(
                asdict(FrameZeroAssetConfig())
            ),
            "frame_zero_semantic_gate_contract": (
                V6_PINNED_SEMANTIC_GATE_CONTRACT_SHA256
            ),
            "held_calibration_gate_contract": held_contract_sha256(CALIBRATION_GATE),
            "held_confirmation_gate_contract": held_contract_sha256(CONFIRMATION_GATE),
            "held_metric_contract": held_contract_sha256(METRIC_LOCK),
            "held_physical_numeric_contract": held_contract_sha256(
                HELD_PHYSICAL_NUMERIC_CONTRACT
            ),
            "held_primary_method_contract": held_contract_sha256(PRIMARY_METHOD),
            "held_source_feasibility_amendment_contract": held_contract_sha256(
                SOURCE_FEASIBILITY_AMENDMENT_CONTRACT
            ),
            "outcome_reconstruction_contract": held_contract_sha256(
                OUTCOME_RECONSTRUCTION_CONTRACT
            ),
            "primary_rbf_config": held_contract_sha256(asdict(HELD_RBF_CONFIG)),
            "raw_cycle_default_config": held_contract_sha256(
                asdict(RawCameraCycleUncertaintyConfig())
            ),
            "raw_observation_default_config": held_contract_sha256(
                asdict(HELD_OBSERVATION_CONFIG)
            ),
            "raw_uncertainty_default_config": held_contract_sha256(
                asdict(HELD_UNCERTAINTY_CONFIG)
            ),
            "robot_kinematics_window_contract": held_contract_sha256(
                ROBOT_KINEMATICS_WINDOW_CONTRACT
            ),
            "upstream_runtime_bundle_tree": held_contract_sha256(
                UPSTREAM_RUNTIME_BUNDLE_CONTRACT
            ),
            "v1_preoutcome_feasibility_report": v1_report_file_sha256,
            "v2_design_withdrawal_report": v2_report_file_sha256,
            "v3_prelock_boundary_incident_report": v3_report_file_sha256,
            "v4_execution_withdrawal_report": v4_report_file_sha256,
            "v5_outcome_withdrawal_report": v5_report_file_sha256,
            **{
                key: str(provenance_before[key])
                for key in METHOD_PROVENANCE_BINDING_KEYS
            },
        }
    )
    _require(
        bindings["robot_kinematics_window_contract"]
        == ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
        "robot kinematics contract export disagrees with canonical held hash",
    )
    _require(set(bindings) == required, "prospective v6 binding key set differs")
    _require(
        all(_SHA256.fullmatch(value) for value in bindings.values()),
        "prospective v6 contains an invalid digest",
    )
    provenance_final = _validate_deployed_git_repository(code)
    _require(
        provenance_final == provenance_before,
        "deployed repository changed while recomputing lock bindings",
    )
    _require(
        _validate_v1_lock(arguments.v1_lock) == v1_lock,
        "v1 parent lock changed during v6 preparation",
    )
    _require(
        _validate_v1_report(arguments.v1_report) == v1_report_file_sha256,
        "v1 feasibility report changed during v6 preparation",
    )
    _require(
        _validate_v2_withdrawal_report(arguments.v2_withdrawal_report)
        == v2_report_file_sha256,
        "v2 withdrawal evidence or absence changed during v6 preparation",
    )
    _require(
        _validate_v3_boundary_incident_report(arguments.v3_boundary_incident_report)
        == v3_report_file_sha256,
        "v3 boundary-incident evidence or absence changed during v6 preparation",
    )
    _require(
        _validate_v4_lock(arguments.v4_lock) == v4_lock,
        "v4 calibration lock changed during v6 preparation",
    )
    _require(
        _validate_v4_execution_withdrawal_report(
            arguments.v4_execution_withdrawal_report
        )
        == v4_report_file_sha256,
        "v4 execution-withdrawal evidence changed during v6 preparation",
    )
    _require(
        _validate_v5_lock(arguments.v5_lock) == v5_lock,
        "v5 calibration lock changed during v6 preparation",
    )
    _require(
        _validate_v5_outcome_withdrawal_report(arguments.v5_outcome_withdrawal_report)
        == v5_report_file_sha256,
        "v5 outcome-withdrawal evidence changed during v6 preparation",
    )
    if arguments.verify_existing_lock:
        output = _absolute(arguments.output_lock)
        _ensure_unprotected_path(output, "existing v6 lock")
        _require(output.name == "calibration-lock.json", "v6 lock filename changed")
        _require(code not in output.parents, "v6 lock may not be inside deployed code")
        raw_lock = _read_regular_bytes(output, "existing v6 lock")
        existing_mode = os.lstat(output).st_mode
        _require(
            stat.S_IMODE(existing_mode) == 0o400,
            "existing v6 lock mode is not exactly 0400",
        )
        lock = load_held_protocol_lock(output)
        _require(
            lock.get("immutable_bindings") == dict(sorted(bindings.items())),
            "existing v6 lock bindings disagree with recomputed bindings",
        )
        canonical_lock = (
            json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _require(raw_lock == canonical_lock, "existing v6 lock file is not canonical")
        operation = "verified_existing_lock"
    else:
        output = _absolute(arguments.output_lock)
        _ensure_unprotected_path(output, "prospective v6 lock output")
        _require(output.name == "calibration-lock.json", "v6 lock filename changed")
        _require(code not in output.parents, "v6 lock may not be inside deployed code")
        _require(not os.path.lexists(output), "prospective v6 lock already exists")
        _require(
            output.parent.is_dir() and not output.parent.is_symlink(),
            "v6 lock parent must already be a real directory",
        )
        lock = create_held_protocol_lock(output, immutable_bindings=bindings)
        os.chmod(output, 0o400)
        operation = "created_lock"
    print(
        json.dumps(
            {
                "operation": operation,
                "protocol_id": lock["protocol_id"],
                "artifact_sha256": lock["artifact_sha256"],
                "lock_file_sha256": _sha256_file(output, "v6 lock"),
                "binding_count": len(bindings),
                "method_head": provenance_before["method_head"],
                "method_commit_object": provenance_before["method_commit_object"],
                "method_git_tree_manifest": provenance_before[
                    "method_git_tree_manifest"
                ],
                "method_deployed_snapshot_tree": provenance_before[
                    "method_deployed_snapshot_tree"
                ],
                "tracked_file_count": provenance_before["tracked_file_count"],
                "v1_report_file_sha256": bindings["v1_preoutcome_feasibility_report"],
                "v2_withdrawal_report_file_sha256": bindings[
                    "v2_design_withdrawal_report"
                ],
                "v3_boundary_incident_report_file_sha256": bindings[
                    "v3_prelock_boundary_incident_report"
                ],
                "v4_execution_withdrawal_report_file_sha256": bindings[
                    "v4_execution_withdrawal_report"
                ],
                "v5_outcome_withdrawal_report_file_sha256": bindings[
                    "v5_outcome_withdrawal_report"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
