"""Fail-closed analysis of the selected-RBF Deform360 view-budget transfer.

The primary-only evaluator materializes the already frozen
``selected_backbone_euclidean_rbf_ungated`` method without covariance
calibration or CPD.  This module does not run that evaluator and never opens
the Deform360 targets.  It validates the frozen inputs, complete evaluator
outputs, a separately saved pre-run parity audit, and an independent
eight-view comparison with the older gated evaluator before applying the four
prospectively frozen four-view gates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_online_belief_evaluation import (
    PRIMARY_METRICS,
    _physical_object_cluster_bootstrap,
)
from .deform360_raw_camera_budget_frontier import (
    CAMERA_COUNTS,
    UPDATE_FRAMES,
    TreeInventory,
    _bound_bytes,
    _bound_json,
    _canonical_sha256,
    _expected_case_metadata,
    _is_sha256,
    _require,
    _stable_file_bytes,
    _validate_config as _validate_raw_frontier_config,
    _validate_measurement_root,
    inventory_tree,
)

PROTOCOL_ID = (
    "deform360-open27-selected-rbf-primary-camera-budget-transfer-v1-development"
)
PARENT_TRANSFER_PROTOCOL_ID = (
    "deform360-open27-gated-camera-budget-transfer-v1-development"
)
PARENT_TRANSFER_CONFIG = (
    "configs/sota/deform360_gated_camera_budget_transfer_v1_development.json"
)
PARENT_TRANSFER_CONFIG_SHA256 = (
    "a4997964dccef81ba02a67588e6c3730ce263b841c36b6dab6199b56c73a3291"
)
RAW_FRONTIER_CONFIG = (
    "configs/sota/deform360_raw_camera_budget_frontier_v1_development.json"
)
RAW_FRONTIER_CONFIG_SHA256 = (
    "27abed3caa8a54ff4d0744f3de06407ae9e7908f124c3fbb2663d73a83526609"
)
HELD_PREDICTOR_COMMIT = "3a23946ab3d647be7c836f92551e264258042325"
EVALUATOR_SOURCE = "src/bayesian_phystwin/deform360_raw_camera_primary_evaluation.py"
ANALYZER_SOURCE = "src/bayesian_phystwin/deform360_primary_camera_budget_transfer.py"
HELD_PREDICTOR_SOURCE = "src/bayesian_phystwin/deform360_held_online_prefix.py"
GATED_EVALUATION_PROTOCOL_ID = (
    "deform360-open27-raw-camera-covariance-gated-rbf-cpd-v1-development"
)
RBF_ARM_PREFIX = "selected_backbone_euclidean_rbf"
SELECTED_BACKBONE_ARM = "selected_raw_backbone"
MINIMUM_SELECTOR_SUPPORT = 3
PRIMARY_EVALUATION_PROTOCOL_ID = (
    "deform360-open27-raw-camera-selected-backbone-rbf-primary-v1-development"
)
PRIMARY_PARITY_PROTOCOL_ID = (
    "deform360-open27-raw-camera-selected-backbone-rbf-parity-v1-development"
)
PRIMARY_ARM = f"{RBF_ARM_PREFIX}_ungated"
PRIMARY_ARMS = (
    "physical_prior",
    "persistence",
    SELECTED_BACKBONE_ARM,
    PRIMARY_ARM,
)
COMPARATORS = ("physical_prior", "persistence", SELECTED_BACKBONE_ARM)
EXCLUDED_PARTIAL_ROOTS = {
    "2": (
        "/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/"
        "deform360-raw-camera-budget-v1-cam2-gated-base-open27-development"
    ),
    "4": (
        "/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/"
        "deform360-raw-camera-budget-v1-cam4-gated-base-open27-development"
    ),
}
FROZEN_STATUS = (
    "prospectively frozen before any 2/4-view primary-only outcome was "
    "produced or inspected; 8-view compatibility was already established "
    "read-only"
)
FROZEN_DATE_UTC = "2026-07-23"
RUNTIME_PYTHON = "/home/florianpfaff/.venvs/motioncrafter-v1/bin/python"
RUNTIME_PYTHON_VERSION = "3.12.3"
RUNTIME_NUMPY_VERSION = "2.0.2"
RUNTIME_SCIPY_VERSION = "1.17.1"
RUNTIME_PIP_FREEZE_SHA256 = (
    "3ee155974316d5b200240b47d44328763ba46c83a5e56e93a2a49c4e4f060e90"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _metric(value: Any, *, label: str) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    result = float(value)
    _require(result >= 0.0, f"{label} is negative")
    return result


def _load_stable_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = Path(path).resolve(strict=True)
    payload = _stable_file_bytes(resolved)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, payload


def _validate_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    _require(
        value.get("result_sha256") == primary_artifact_sha256(value),
        f"{label} canonical result hash changed",
    )


def primary_artifact_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical hash used by primary evaluator JSON artifacts."""

    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    return _canonical_sha256(unsigned)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_blob(repository: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"],
        check=False,
        capture_output=True,
    )
    _require(
        result.returncode == 0,
        f"cannot read {relative_path} from bound commit {commit}",
    )
    return result.stdout


def _validate_code_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    implementation = config["implementation"]
    evaluator_commit = implementation["evaluator_commit"]
    analyzer_commit = implementation["analyzer_commit"]
    repository = _repository_root()
    bindings = (
        ("evaluator", evaluator_commit, EVALUATOR_SOURCE),
        ("analyzer", analyzer_commit, ANALYZER_SOURCE),
        ("held_predictor", HELD_PREDICTOR_COMMIT, HELD_PREDICTOR_SOURCE),
    )
    result: dict[str, Any] = {}
    for label, commit, relative_path in bindings:
        current = _stable_file_bytes(repository / relative_path)
        committed = _git_blob(repository, commit, relative_path)
        _require(
            current == committed,
            f"{label} source differs from its bound commit",
        )
        result[label] = {
            "commit": commit,
            "source": relative_path,
            "source_sha256": _sha256_bytes(current),
        }
    return result


def _live_sorted_pip_freeze_sha256(python_executable: str) -> str:
    result = subprocess.run(
        [python_executable, "-m", "pip", "freeze", "--all"],
        check=False,
        capture_output=True,
    )
    _require(result.returncode == 0, "frozen Python pip freeze --all failed")
    lines = result.stdout.splitlines()
    _require(lines, "frozen Python pip freeze --all returned no distributions")
    _require(
        all(line and b"\0" not in line for line in lines),
        "pip freeze output contains an invalid line",
    )
    normalized = b"\n".join(sorted(lines)) + b"\n"
    return _sha256_bytes(normalized)


def _validate_runtime_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    expected_pythonpath = str(_repository_root() / "src")
    _require(
        sys.executable == RUNTIME_PYTHON,
        "analyzer is not running under the frozen Python executable",
    )
    _require(
        platform.python_version() == RUNTIME_PYTHON_VERSION,
        "live Python version differs from the frozen runtime",
    )
    _require(
        np.__version__ == RUNTIME_NUMPY_VERSION,
        "live NumPy version differs from the frozen runtime",
    )
    scipy_version = importlib.metadata.version("scipy")
    _require(
        scipy_version == RUNTIME_SCIPY_VERSION,
        "live SciPy version differs from the frozen runtime",
    )
    _require(sys.flags.isolated == 0, "isolated Python mode is forbidden")
    _require(sys.dont_write_bytecode, "Python bytecode writes are forbidden")
    _require(
        os.environ.get("PYTHONPATH") == expected_pythonpath,
        "PYTHONPATH is not exactly the clean-clone source root",
    )
    pip_freeze_sha256 = _live_sorted_pip_freeze_sha256(sys.executable)
    _require(
        pip_freeze_sha256 == RUNTIME_PIP_FREEZE_SHA256,
        "live pip freeze differs from the frozen package inventory",
    )
    return {
        **runtime,
        "observed": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy_version,
            "pythonpath": os.environ["PYTHONPATH"],
            "isolated_mode": bool(sys.flags.isolated),
            "dont_write_bytecode": bool(sys.dont_write_bytecode),
            "pip_freeze_all_sorted_sha256": pip_freeze_sha256,
        },
    }


def _load_parent_configs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repository = _repository_root()
    parents = config["parents"]
    transfer_path = repository / parents["gated_transfer"]["path"]
    raw_path = repository / parents["raw_frontier"]["path"]
    transfer, transfer_bytes = _load_stable_json(
        transfer_path,
        label="parent gated-transfer config",
    )
    raw, raw_bytes = _load_stable_json(raw_path, label="parent raw-frontier config")
    _require(
        _sha256_bytes(transfer_bytes) == parents["gated_transfer"]["sha256"],
        "parent gated-transfer config hash changed",
    )
    _require(
        _sha256_bytes(raw_bytes) == parents["raw_frontier"]["sha256"],
        "parent raw-frontier config hash changed",
    )
    _require(
        transfer.get("protocol_id") == PARENT_TRANSFER_PROTOCOL_ID,
        "parent gated-transfer protocol changed",
    )
    _validate_raw_frontier_config(raw)
    cohort = transfer.get("cohort", {})
    _require(
        cohort.get("parent_frontier_config") == RAW_FRONTIER_CONFIG
        and cohort.get("parent_frontier_config_sha256") == RAW_FRONTIER_CONFIG_SHA256,
        "gated transfer binds a different raw frontier",
    )
    _require(
        transfer.get("method", {}).get("primary_arm") == PRIMARY_ARM,
        "parent gated transfer primary arm changed",
    )
    return (
        transfer,
        raw,
        {
            "gated_transfer": {
                "path": str(transfer_path.resolve(strict=True)),
                "sha256": _sha256_bytes(transfer_bytes),
            },
            "raw_frontier": {
                "path": str(raw_path.resolve(strict=True)),
                "sha256": _sha256_bytes(raw_bytes),
            },
        },
    )


def _lexical_absolute(path: Any, *, label: str) -> str:
    _require(isinstance(path, str) and path.startswith("/"), f"{label} is not absolute")
    normalized = os.path.normpath(path)
    _require(normalized == path, f"{label} is not lexically normalized")
    return normalized


def _validate_config(config: Mapping[str, Any]) -> None:
    """Validate the complete prospective transfer protocol without opening roots."""

    _require(
        set(config)
        == {
            "schema_version",
            "protocol_id",
            "status",
            "freeze",
            "parents",
            "implementation",
            "runtime",
            "method",
            "bound_inputs",
            "outputs",
            "excluded_partial_roots",
            "decision",
            "claim_boundary",
        },
        "transfer config top-level schema changed",
    )
    _require(config.get("schema_version") == 1, "transfer config schema changed")
    _require(config.get("protocol_id") == PROTOCOL_ID, "transfer protocol ID changed")
    _require(config.get("status") == FROZEN_STATUS, "prospective status changed")
    freeze = config.get("freeze", {})
    _require(
        isinstance(freeze, dict)
        and set(freeze)
        == {
            "date_utc",
            "low_view_primary_outcome_status_at_freeze",
            "eight_view_compatibility_status_at_freeze",
            "saved_parity_artifact_status_at_freeze",
            "method_selection",
        }
        and freeze.get("date_utc") == FROZEN_DATE_UTC,
        "freeze schema or date changed",
    )
    _require(
        freeze.get("low_view_primary_outcome_status_at_freeze")
        == {"2": "not produced or inspected", "4": "not produced or inspected"},
        "low-view primary outcome status changed",
    )
    _require(
        freeze.get("eight_view_compatibility_status_at_freeze")
        == (
            "read-only in-memory parity established and inspected for all 27 "
            "cases; no fresh 8-view evaluation root materialized"
        ),
        "eight-view compatibility status changed",
    )
    _require(
        freeze.get("saved_parity_artifact_status_at_freeze")
        == "not produced or inspected",
        "saved parity artifact status changed",
    )
    _require(
        freeze.get("method_selection")
        == (
            "exact target-free selected-backbone support-gated ungated RBF "
            "implementation transferred without tuning"
        ),
        "method selection changed",
    )
    parents = config.get("parents", {})
    _require(
        parents
        == {
            "gated_transfer": {
                "path": PARENT_TRANSFER_CONFIG,
                "sha256": PARENT_TRANSFER_CONFIG_SHA256,
            },
            "raw_frontier": {
                "path": RAW_FRONTIER_CONFIG,
                "sha256": RAW_FRONTIER_CONFIG_SHA256,
            },
        },
        "parent config bindings changed",
    )
    implementation = config.get("implementation", {})
    _require(
        set(implementation)
        == {
            "analyzer_commit",
            "analyzer_source",
            "evaluator_commit",
            "evaluator_source",
            "held_predictor_commit",
            "held_predictor_source",
        },
        "implementation bindings are incomplete",
    )
    _require(
        _is_git_commit(implementation.get("evaluator_commit")),
        "evaluator commit is invalid",
    )
    _require(
        _is_git_commit(implementation.get("analyzer_commit")),
        "analyzer commit is invalid",
    )
    _require(
        implementation.get("analyzer_source") == ANALYZER_SOURCE
        and implementation.get("evaluator_source") == EVALUATOR_SOURCE
        and implementation.get("held_predictor_commit") == HELD_PREDICTOR_COMMIT
        and implementation.get("held_predictor_source") == HELD_PREDICTOR_SOURCE,
        "analyzer, evaluator, or held-predictor binding changed",
    )
    runtime = config.get("runtime", {})
    _require(
        isinstance(runtime, dict)
        and set(runtime)
        == {
            "python_executable",
            "python_version",
            "numpy_version",
            "scipy_version",
            "pip_freeze_all_sorted_sha256",
            "gpu_required",
            "pythonpath_contract",
            "isolated_mode_forbidden",
            "bytecode_writes_forbidden",
        }
        and runtime.get("python_executable") == RUNTIME_PYTHON
        and runtime.get("python_version") == RUNTIME_PYTHON_VERSION
        and runtime.get("numpy_version") == RUNTIME_NUMPY_VERSION
        and runtime.get("scipy_version") == RUNTIME_SCIPY_VERSION
        and runtime.get("pip_freeze_all_sorted_sha256") == RUNTIME_PIP_FREEZE_SHA256
        and runtime.get("gpu_required") is False
        and runtime.get("pythonpath_contract") == "clean_clone/src only"
        and runtime.get("isolated_mode_forbidden") is True
        and runtime.get("bytecode_writes_forbidden") is True,
        "frozen evaluation runtime changed",
    )
    method = config.get("method", {})
    _require(
        method
        == {
            "camera_counts": list(CAMERA_COUNTS),
            "camera_budget_semantics": (
                "dynamic tracked-view count after all-view frame-zero planning; "
                "not a full sensor-count ablation"
            ),
            "primary_evaluation_protocol_id": PRIMARY_EVALUATION_PROTOCOL_ID,
            "parity_protocol_id": PRIMARY_PARITY_PROTOCOL_ID,
            "gated_reference_protocol_id": GATED_EVALUATION_PROTOCOL_ID,
            "primary_arm": PRIMARY_ARM,
            "comparators": list(COMPARATORS),
            "primary_metrics": list(PRIMARY_METRICS),
            "minimum_selector_support": MINIMUM_SELECTOR_SUPPORT,
            "insufficient_support_default": "persistence",
            "covariance_or_cpd_used": False,
        },
        "selected-RBF method contract changed",
    )
    bound = config.get("bound_inputs", {})
    _require(
        isinstance(bound, dict) and set(bound) == {"measurements", "gated_8_reference"},
        "bound input schema changed",
    )
    measurements = bound.get("measurements", {})
    _require(
        isinstance(measurements, dict)
        and set(measurements) == {str(count) for count in CAMERA_COUNTS},
        "measurement input bindings are incomplete",
    )
    for camera_count in CAMERA_COUNTS:
        record = measurements[str(camera_count)]
        _require(
            isinstance(record, dict)
            and set(record)
            == {"root", "file_count", "total_file_bytes", "inventory_sha256"}
            and type(record["file_count"]) is int
            and record["file_count"] > 0
            and type(record["total_file_bytes"]) is int
            and record["total_file_bytes"] > 0
            and _is_sha256(record["inventory_sha256"]),
            f"{camera_count}-view measurement binding is invalid",
        )
        _lexical_absolute(record["root"], label=f"{camera_count}-view measurement root")
    reference = bound.get("gated_8_reference", {})
    _require(
        isinstance(reference, dict)
        and set(reference)
        == {
            "root",
            "file_count",
            "total_file_bytes",
            "inventory_sha256",
            "summary_file_sha256",
            "summary_result_sha256",
        }
        and type(reference["file_count"]) is int
        and reference["file_count"] > 0
        and type(reference["total_file_bytes"]) is int
        and reference["total_file_bytes"] > 0
        and all(
            _is_sha256(reference[field])
            for field in (
                "inventory_sha256",
                "summary_file_sha256",
                "summary_result_sha256",
            )
        ),
        "gated 8-view reference binding is invalid",
    )
    _lexical_absolute(reference["root"], label="gated 8-view reference root")
    outputs = config.get("outputs", {})
    _require(
        isinstance(outputs, dict)
        and set(outputs)
        == {
            "primary_evaluations",
            "execution_root",
            "parity_artifact",
            "analysis_output",
        },
        "output schema changed",
    )
    primary = outputs.get("primary_evaluations", {})
    _require(
        isinstance(primary, dict)
        and set(primary) == {str(count) for count in CAMERA_COUNTS},
        "fresh primary output roots are incomplete",
    )
    all_output_paths = [
        _lexical_absolute(primary[str(count)], label=f"{count}-view primary output")
        for count in CAMERA_COUNTS
    ]
    execution_root = _lexical_absolute(
        outputs.get("execution_root"),
        label="execution root",
    )
    parity_artifact = _lexical_absolute(
        outputs.get("parity_artifact"),
        label="parity artifact",
    )
    analysis_output = _lexical_absolute(
        outputs.get("analysis_output"),
        label="analysis output",
    )
    _require(
        Path(parity_artifact).parent.as_posix() == execution_root,
        "parity artifact is outside the frozen execution root",
    )
    _require(
        Path(parity_artifact).name == "cam8-parity.json",
        "parity artifact name changed",
    )
    _require(
        config.get("excluded_partial_roots") == EXCLUDED_PARTIAL_ROOTS,
        "excluded failed partial roots changed",
    )
    excluded = list(EXCLUDED_PARTIAL_ROOTS.values())
    _require(
        len(set([*all_output_paths, execution_root, analysis_output, *excluded]))
        == len([*all_output_paths, execution_root, analysis_output, *excluded]),
        "fresh, execution, analysis, and excluded roots overlap",
    )
    _require(
        parity_artifact not in excluded,
        "parity artifact aliases an excluded partial root",
    )
    for active_path in (
        *all_output_paths,
        execution_root,
        analysis_output,
        parity_artifact,
    ):
        active = Path(active_path)
        for excluded_path in excluded:
            failed_partial = Path(excluded_path)
            _require(
                active != failed_partial
                and active not in failed_partial.parents
                and failed_partial not in active.parents,
                "fresh path and excluded partial root have an ancestor overlap",
            )
    decision = config.get("decision", {})
    _require(
        decision
        == {
            "candidate_camera_count": 4,
            "reference_camera_count": 8,
            "descriptive_camera_counts": [2],
            "go_if_all": {
                (
                    "minimum_fraction_of_8_view_relative_improvement_retained_"
                    "each_primary_metric"
                ): 0.8,
                "minimum_joint_case_wins_vs_physical": 18,
                (
                    "all_five_object_mean_differences_vs_physical_improve_on_"
                    "both_primary_metrics"
                ): True,
                "maximum_case_chamfer_relative_regression_vs_physical": 0.1,
            },
            "secondary_field_value_check": (
                "report whether the 4-view primary arm improves both aggregate "
                "primary metrics over selected_raw_backbone; this does not "
                "override the four preregistered GO gates"
            ),
            "tie_policy": (
                "ties are not improvements or wins; the maximum regression "
                "bound is inclusive"
            ),
        },
        "decision rule changed",
    )
    _require(
        isinstance(config.get("claim_boundary"), str)
        and bool(config["claim_boundary"]),
        "claim boundary is missing",
    )


def _validate_inventory_binding(
    inventory: TreeInventory,
    binding: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(
        str(inventory.root) == str(Path(binding["root"]).resolve(strict=True))
        and inventory.file_count == binding["file_count"]
        and inventory.total_file_bytes == binding["total_file_bytes"]
        and inventory.inventory_sha256 == binding["inventory_sha256"],
        f"{label} inventory changed",
    )


def _expected_evaluation_paths(cases: Sequence[str]) -> set[str]:
    return {
        "summary.json",
        *(relative for case in cases for relative in (f"{case}.json", f"{case}.npz")),
    }


def _load_npz(
    inventory: TreeInventory,
    relative_path: str,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    payload = _bound_bytes(inventory, relative_path)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
            return {name: np.asarray(stored[name]).copy() for name in stored.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a valid non-pickle NPZ") from error


def _validate_primary_evaluation_root(
    inventory: TreeInventory,
    measurement_inventory: TreeInventory,
    measurement_manifests: Mapping[str, Mapping[str, Any]],
    cases: Sequence[str],
    objects: Mapping[str, str],
    episodes: Mapping[str, int],
    *,
    camera_count: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    _require(
        set(inventory.sha256_by_relative_path) == _expected_evaluation_paths(cases),
        f"{camera_count}-view primary output file layout changed",
    )
    summary = _bound_json(inventory, "summary.json")
    _validate_self_hash(summary, label=f"{camera_count}-view primary summary")
    _require(
        summary.get("schema_version") == 1
        and summary.get("protocol_id") == PRIMARY_EVALUATION_PROTOCOL_ID
        and summary.get("episode_count") == len(cases)
        and summary.get("physical_object_count") == len(set(objects.values()))
        and summary.get("primary_arm") == PRIMARY_ARM
        and summary.get("comparators") == list(COMPARATORS),
        f"{camera_count}-view primary summary contract changed",
    )
    boundary = summary.get("information_boundary", {})
    _require(
        boundary.get("all_measurements_verified_before_any_target_open") is True
        and boundary.get("all_primary_predictions_completed_before_any_target_open")
        is True
        and boundary.get("uncertainty_sidecars_required") is False
        and boundary.get("target_visible_covariance_calibration_performed") is False,
        f"{camera_count}-view primary summary crossed its boundary",
    )
    artifact_records = summary.get("artifacts")
    _require(
        isinstance(artifact_records, list)
        and [record.get("case") for record in artifact_records] == list(cases),
        f"{camera_count}-view primary artifact order changed",
    )
    reports: dict[str, dict[str, Any]] = {}
    arrays_by_case: dict[str, dict[str, np.ndarray]] = {}
    for artifact, case in zip(artifact_records, cases, strict=True):
        report_relative = f"{case}.json"
        archive_relative = f"{case}.npz"
        _require(
            isinstance(artifact, dict)
            and set(artifact)
            == {
                "case",
                "report_sha256",
                "report_result_sha256",
                "archive_sha256",
            }
            and artifact["report_sha256"]
            == inventory.sha256_by_relative_path[report_relative]
            and artifact["archive_sha256"]
            == inventory.sha256_by_relative_path[archive_relative]
            and _is_sha256(artifact["report_result_sha256"]),
            f"{camera_count}-view artifact binding changed for {case}",
        )
        report = _bound_json(inventory, report_relative)
        _validate_self_hash(report, label=f"{camera_count}-view {case} report")
        _require(
            report["result_sha256"] == artifact["report_result_sha256"]
            and report.get("trajectory_archive_sha256") == artifact["archive_sha256"],
            f"{camera_count}-view report self-binding changed for {case}",
        )
        _require(
            report.get("protocol_id") == PRIMARY_EVALUATION_PROTOCOL_ID
            and report.get("primary_arm") == PRIMARY_ARM
            and report.get("case") == case
            and report.get("object_id") == objects[case]
            and report.get("episode_id") == episodes[case],
            f"{camera_count}-view primary identity changed for {case}",
        )
        manifest = measurement_manifests[case]
        manifest_inputs = manifest.get("inputs", {})
        prediction_seal_sha256 = manifest_inputs.get("prediction_seal", {}).get(
            "sha256"
        )
        prediction_archive_sha256 = manifest_inputs.get("prediction_archive", {}).get(
            "sha256"
        )
        _require(
            report.get("measurement_manifest_sha256")
            == measurement_inventory.sha256_by_relative_path[
                f"{case}/measurement_manifest.json"
            ]
            and report.get("measurement_archive_sha256")
            == measurement_inventory.sha256_by_relative_path[f"{case}/measurement.npz"]
            and report.get("measurement_result_sha256") == manifest["result_sha256"],
            f"{camera_count}-view report binds different measurements for {case}",
        )
        _require(
            _is_sha256(prediction_seal_sha256)
            and _is_sha256(prediction_archive_sha256)
            and report.get("prediction_seal_sha256") == prediction_seal_sha256
            and report.get("prediction_archive_sha256") == prediction_archive_sha256,
            f"{camera_count}-view report binds a different sealed prior for {case}",
        )
        _require(
            report.get("center_ids") == manifest["plan"]["center_ids"]
            and report.get("update_frames") == list(UPDATE_FRAMES)
            and isinstance(report.get("scored_frames"), list)
            and all(
                type(frame) is int and frame >= 0 for frame in report["scored_frames"]
            ),
            f"{camera_count}-view trajectory contract changed for {case}",
        )
        algorithm = report.get("algorithm_binding", {})
        _require(
            algorithm.get("implementation")
            == "predict_support_gated_selected_backbone_rbf"
            and algorithm.get("target_argument_accepted_by_predictor") is False
            and algorithm.get("uncertainty_argument_accepted_by_predictor") is False
            and algorithm.get("held_rbf_config_required") is True
            and algorithm.get("primary_trajectory_scored_without_recomputation")
            is True,
            f"{camera_count}-view algorithm binding changed for {case}",
        )
        support = report.get("support_gate_contract", {})
        _require(
            support.get("minimum_current_observed_centers") == MINIMUM_SELECTOR_SUPPORT
            and support.get("insufficient_support_default") == "persistence"
            and support.get("covariance_required") is False,
            f"{camera_count}-view support gate changed for {case}",
        )
        case_boundary = report.get("information_boundary", {})
        _require(
            case_boundary.get("measurement_verified_before_target_open") is True
            and case_boundary.get("primary_prediction_completed_before_target_open")
            is True
            and case_boundary.get("measurement_builder_target_read") is False
            and case_boundary.get("uncertainty_sidecar_read") is False
            and case_boundary.get("target_visible_covariance_calibration_performed")
            is False
            and case_boundary.get("target_role") == "scoring only",
            f"{camera_count}-view case crossed its information boundary for {case}",
        )
        scores = report.get("scores")
        _require(
            isinstance(scores, dict) and set(scores) == set(PRIMARY_ARMS),
            f"{camera_count}-view score arms changed for {case}",
        )
        for arm in PRIMARY_ARMS:
            _require(
                isinstance(scores[arm], dict),
                f"{camera_count}-view score is invalid for {case}/{arm}",
            )
            for metric_name in PRIMARY_METRICS:
                _metric(
                    scores[arm].get(metric_name),
                    label=f"{camera_count}-view {case}/{arm}/{metric_name}",
                )
        updates = report.get("updates")
        _require(
            isinstance(updates, list)
            and [record.get("frame") for record in updates] == list(UPDATE_FRAMES),
            f"{camera_count}-view updates changed for {case}",
        )
        for update in updates:
            count = update.get("available_center_count")
            sufficient = update.get("selector_support_sufficient")
            gate = update.get("support_gate", {})
            canonical_decision = (
                "current_observed_center_symmetric_chamfer"
                if sufficient
                else "insufficient_support_persistence"
            )
            _require(
                type(count) is int
                and count >= 0
                and sufficient is (count >= MINIMUM_SELECTOR_SUPPORT)
                and update.get("selector_decision") == canonical_decision
                and gate.get("accepted") is sufficient
                and gate.get("decision") == canonical_decision
                and gate.get("selected_backbone") == update.get("selected_backbone")
                and gate.get("fallback_backbone") == update.get("selected_backbone"),
                f"{camera_count}-view support decision changed for {case}",
            )
        arrays = _load_npz(
            inventory,
            archive_relative,
            label=f"{camera_count}-view {case} trajectory archive",
        )
        _require(
            set(arrays) == set(PRIMARY_ARMS),
            f"{camera_count}-view trajectory arms changed for {case}",
        )
        shapes = {array.shape for array in arrays.values()}
        _require(
            len(shapes) == 1
            and next(iter(shapes), ())[-1:] == (3,)
            and all(array.ndim == 3 for array in arrays.values())
            and all(array.dtype.kind == "f" for array in arrays.values())
            and all(np.all(np.isfinite(array)) for array in arrays.values()),
            f"{camera_count}-view trajectory arrays are invalid for {case}",
        )
        reports[case] = report
        arrays_by_case[case] = arrays

    aggregate = summary.get("aggregate")
    _require(
        isinstance(aggregate, dict) and set(aggregate) == set(PRIMARY_ARMS),
        f"{camera_count}-view aggregate arms changed",
    )
    for arm in PRIMARY_ARMS:
        for metric_name in PRIMARY_METRICS:
            recomputed = float(
                np.mean([reports[case]["scores"][arm][metric_name] for case in cases])
            )
            _require(
                aggregate[arm].get(metric_name) == recomputed,
                f"{camera_count}-view aggregate changed for {arm}/{metric_name}",
            )
    _validate_summary_comparisons(summary, reports, cases, objects, camera_count)
    return summary, reports, arrays_by_case


def _validate_summary_comparisons(
    summary: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    cases: Sequence[str],
    objects: Mapping[str, str],
    camera_count: int,
) -> None:
    comparisons = summary.get("comparisons")
    expected_keys = {
        f"{PRIMARY_ARM}:vs:{baseline}:{metric}"
        for baseline in COMPARATORS
        for metric in PRIMARY_METRICS
    }
    _require(
        isinstance(comparisons, dict) and set(comparisons) == expected_keys,
        f"{camera_count}-view summary comparisons changed",
    )
    for baseline in COMPARATORS:
        for metric_name in PRIMARY_METRICS:
            differences = {
                case: float(
                    reports[case]["scores"][PRIMARY_ARM][metric_name]
                    - reports[case]["scores"][baseline][metric_name]
                )
                for case in cases
            }
            expected = _physical_object_cluster_bootstrap(differences, objects)
            expected["episode_wins"] = int(
                np.sum(np.asarray(list(differences.values())) < 0.0)
            )
            expected["per_object_mean_difference_m"] = {
                object_id: float(
                    np.mean(
                        [
                            differences[case]
                            for case in cases
                            if objects[case] == object_id
                        ]
                    )
                )
                for object_id in sorted(set(objects.values()))
            }
            aggregate = summary["aggregate"]
            expected["relative_change"] = (
                aggregate[PRIMARY_ARM][metric_name] / aggregate[baseline][metric_name]
                - 1.0
            )
            key = f"{PRIMARY_ARM}:vs:{baseline}:{metric_name}"
            _require(
                comparisons[key] == expected,
                f"{camera_count}-view comparison changed for {baseline}/{metric_name}",
            )


def _validate_gated_reference(
    inventory: TreeInventory,
    binding: Mapping[str, Any],
    cases: Sequence[str],
    objects: Mapping[str, str],
    episodes: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    _validate_inventory_binding(inventory, binding, label="gated 8-view reference")
    _require(
        set(inventory.sha256_by_relative_path) == _expected_evaluation_paths(cases),
        "gated 8-view reference file layout changed",
    )
    _require(
        inventory.sha256_by_relative_path["summary.json"]
        == binding["summary_file_sha256"],
        "gated 8-view reference summary file hash changed",
    )
    summary = _bound_json(inventory, "summary.json")
    unsigned = dict(summary)
    claimed = unsigned.pop("result_sha256", None)
    _require(
        claimed == _canonical_sha256(unsigned)
        and claimed == binding["summary_result_sha256"],
        "gated 8-view reference summary result hash changed",
    )
    _require(
        summary.get("schema_version") == 1
        and summary.get("protocol_id") == GATED_EVALUATION_PROTOCOL_ID
        and summary.get("episode_count") == len(cases)
        and summary.get("physical_object_count") == len(set(objects.values())),
        "gated 8-view reference summary contract changed",
    )
    artifacts = summary.get("artifacts")
    _require(
        isinstance(artifacts, list)
        and [record.get("case") for record in artifacts] == list(cases),
        "gated 8-view reference artifact order changed",
    )
    reports: dict[str, dict[str, Any]] = {}
    arrays_by_case: dict[str, dict[str, np.ndarray]] = {}
    for artifact, case in zip(artifacts, cases, strict=True):
        report_relative = f"{case}.json"
        archive_relative = f"{case}.npz"
        _require(
            artifact.get("report_sha256")
            == inventory.sha256_by_relative_path[report_relative]
            and artifact.get("archive_sha256")
            == inventory.sha256_by_relative_path[archive_relative],
            f"gated 8-view reference artifact hash changed for {case}",
        )
        report = _bound_json(inventory, report_relative)
        _require(
            report.get("protocol_id") == GATED_EVALUATION_PROTOCOL_ID
            and report.get("case") == case
            and report.get("object_id") == objects[case]
            and report.get("episode_id") == episodes[case],
            f"gated 8-view reference identity changed for {case}",
        )
        arrays = _load_npz(
            inventory,
            archive_relative,
            label=f"gated 8-view {case} trajectory archive",
        )
        _require(
            set(PRIMARY_ARMS).issubset(arrays),
            f"gated 8-view reference lacks primary trajectories for {case}",
        )
        reports[case] = report
        arrays_by_case[case] = arrays
    return summary, reports, arrays_by_case


def _validate_saved_parity(
    config: Mapping[str, Any],
    reference_summary: Mapping[str, Any],
    reference_inventory: TreeInventory,
    cases: Sequence[str],
) -> dict[str, Any]:
    parity_path = Path(config["outputs"]["parity_artifact"])
    parity, payload = _load_stable_json(
        parity_path, label="saved 8-view parity artifact"
    )
    _validate_self_hash(parity, label="saved 8-view parity artifact")
    _require(
        set(parity)
        == {
            "schema_version",
            "protocol_id",
            "reference_protocol_id",
            "reference_summary_binding",
            "episode_count",
            "all_27_cases_primary_arrays_byte_exact",
            "all_27_cases_parity_passed",
            "parity_passed",
            "cases",
            "read_only_contract",
            "result_sha256",
        }
        and parity.get("schema_version") == 1
        and parity.get("protocol_id") == PRIMARY_PARITY_PROTOCOL_ID
        and parity.get("episode_count") == len(cases) == 27
        and parity.get("parity_passed") is True
        and parity.get("all_27_cases_parity_passed") is True,
        "saved 8-view parity top-level contract changed or did not pass",
    )
    reference = parity.get("reference_summary_binding", {})
    _require(
        parity.get("reference_protocol_id") == GATED_EVALUATION_PROTOCOL_ID
        and isinstance(reference, dict)
        and set(reference) == {"file_sha256", "result_sha256"}
        and reference.get("file_sha256")
        == reference_inventory.sha256_by_relative_path["summary.json"]
        and reference.get("result_sha256") == reference_summary["result_sha256"],
        "saved parity binds a different gated reference",
    )
    _require(
        parity.get("all_27_cases_primary_arrays_byte_exact") is True,
        "saved parity arrays are not byte exact",
    )
    read_only_contract = {
        "input_artifacts_written": False,
        "output_artifacts_written": False,
        "reference_hashes_verified_before_comparison": True,
        "reference_summary_externally_bound": True,
        "reference_case_files_loaded_from_hash_verified_bytes": True,
        "all_measurements_verified_before_any_target_open": True,
        "all_primary_predictions_completed_before_any_target_open": True,
        "primary_input_hashes_rechecked_at_parity_completion": True,
    }
    _require(
        parity.get("read_only_contract") == read_only_contract,
        "saved parity read-only contract changed",
    )
    case_records = parity.get("cases")
    _require(
        isinstance(case_records, list)
        and all(isinstance(record, dict) for record in case_records)
        and [record.get("case") for record in case_records] == list(cases)
        and len({record.get("case") for record in case_records}) == len(cases),
        "saved parity case order, uniqueness, or coverage changed",
    )
    case_fields = {
        "case",
        "all_primary_arrays_byte_exact",
        "all_exact_metadata_equal",
        "all_support_semantics_equivalent",
        "parity_passed",
        "trajectory_bit_exact",
        "metadata_exact",
        "score_within_absolute_tolerance",
        "score_absolute_tolerance",
        "updates",
        "reference_trajectory_sha256",
        "primary_trajectory_sha256",
    }
    metadata_fields = {
        "center_ids",
        "update_frames",
        "scored_frames",
        "rbf_config",
        "observed_backbone_selector_normalized",
    }
    update_fields = {
        "frame",
        "selection_metadata_bit_exact",
        "canonical_support_decision",
        "legacy_reference_selector_decision",
        "legacy_reference_gate_decision",
        "support_semantics_equivalent",
    }
    for record, case in zip(case_records, cases, strict=True):
        _require(
            isinstance(record, dict)
            and set(record) == case_fields
            and record["case"] == case
            and record.get("all_primary_arrays_byte_exact") is True
            and record.get("all_exact_metadata_equal") is True
            and record.get("all_support_semantics_equivalent") is True
            and record.get("parity_passed") is True
            and record.get("score_absolute_tolerance") == 1.0e-12,
            f"saved parity case contract changed or failed for {case}",
        )
        for field in (
            "trajectory_bit_exact",
            "score_within_absolute_tolerance",
        ):
            checks = record.get(field)
            _require(
                isinstance(checks, dict)
                and set(checks) == set(PRIMARY_ARMS)
                and all(value is True for value in checks.values()),
                f"saved parity {field} changed or failed for {case}",
            )
        metadata = record.get("metadata_exact")
        _require(
            isinstance(metadata, dict)
            and set(metadata) == metadata_fields
            and all(value is True for value in metadata.values()),
            f"saved parity metadata changed or failed for {case}",
        )
        for field in (
            "reference_trajectory_sha256",
            "primary_trajectory_sha256",
        ):
            hashes = record.get(field)
            _require(
                isinstance(hashes, dict)
                and set(hashes) == set(PRIMARY_ARMS)
                and all(_is_sha256(value) for value in hashes.values()),
                f"saved parity trajectory hashes changed for {case}",
            )
        _require(
            record["reference_trajectory_sha256"]
            == record["primary_trajectory_sha256"],
            f"saved parity trajectory hashes differ for {case}",
        )
        updates = record.get("updates")
        _require(
            isinstance(updates, list)
            and [update.get("frame") for update in updates] == list(UPDATE_FRAMES),
            f"saved parity updates changed for {case}",
        )
        for update in updates:
            canonical = update.get("canonical_support_decision")
            legacy_selector = update.get("legacy_reference_selector_decision")
            legacy_gate = update.get("legacy_reference_gate_decision")
            sufficient = canonical == "current_observed_center_symmetric_chamfer"
            _require(
                isinstance(update, dict)
                and set(update) == update_fields
                and update.get("selection_metadata_bit_exact") is True
                and update.get("support_semantics_equivalent") is True
                and canonical
                in {
                    "current_observed_center_symmetric_chamfer",
                    "insufficient_support_persistence",
                }
                and legacy_selector
                == (
                    "current_observation_chamfer"
                    if sufficient
                    else "insufficient_support_persistence_default"
                )
                and legacy_gate
                in (
                    {"accepted_without_covariance_gate"}
                    if sufficient
                    else {
                        "insufficient_valid_covariance",
                        "insufficient_selector_support",
                    }
                ),
                f"saved parity update detail changed or failed for {case}",
            )
    return {
        "path": str(parity_path.resolve(strict=True)),
        "file_sha256": _sha256_bytes(payload),
        "result_sha256": parity["result_sha256"],
        "parity_passed": True,
        "reference_summary_binding": reference,
    }


def _arrays_bit_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _independent_eight_view_parity(
    cases: Sequence[str],
    fresh_reports: Mapping[str, Mapping[str, Any]],
    fresh_arrays: Mapping[str, Mapping[str, np.ndarray]],
    reference_reports: Mapping[str, Mapping[str, Any]],
    reference_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    selection_fields = (
        "frame",
        "stop_frame_exclusive",
        "available_center_count",
        "selected_backbone",
        "selector_support_sufficient",
        "current_observation_chamfer_m",
    )
    gate_fields = (
        "accepted",
        "selected_backbone",
        "fallback_backbone",
        "rbf_correction_applied",
    )
    metadata_fields = (
        "center_ids",
        "update_frames",
        "scored_frames",
        "rbf_config",
    )
    selector_fields = (
        "metric",
        "tie_break",
        "minimum_reliable_support",
        "insufficient_support_default",
        "insufficient_support_rule_status",
        "selected_by_update",
        "physical_prior_count",
        "persistence_count",
        "insufficient_support_count",
    )
    for case in cases:
        fresh = fresh_reports[case]
        reference = reference_reports[case]
        array_checks = {
            arm: _arrays_bit_exact(
                fresh_arrays[case][arm],
                reference_arrays[case][arm],
            )
            for arm in PRIMARY_ARMS
        }
        metadata_checks = {
            field: fresh.get(field) == reference.get(field) for field in metadata_fields
        }
        metadata_checks["observed_backbone_selector_normalized"] = all(
            fresh.get("observed_backbone_selector", {}).get(field)
            == reference.get("observed_backbone_selector", {}).get(field)
            for field in selector_fields
        )
        score_checks = {
            arm: all(
                math.isclose(
                    float(fresh["scores"][arm][metric]),
                    float(reference["scores"][arm][metric]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for metric in PRIMARY_METRICS
            )
            for arm in PRIMARY_ARMS
        }
        fresh_updates = fresh.get("updates")
        reference_updates = reference.get("updates")
        _require(
            isinstance(fresh_updates, list)
            and isinstance(reference_updates, list)
            and len(fresh_updates) == len(reference_updates) == len(UPDATE_FRAMES),
            f"8-view update parity shape changed for {case}",
        )
        update_checks: list[dict[str, Any]] = []
        for fresh_update, reference_update in zip(
            fresh_updates,
            reference_updates,
            strict=True,
        ):
            fresh_gate = fresh_update.get("support_gate", {})
            reference_gate = reference_update.get("gates", {}).get("ungated", {})
            selection_equal = all(
                fresh_update.get(field) == reference_update.get(field)
                for field in selection_fields
            )
            sufficient = bool(fresh_update.get("selector_support_sufficient"))
            canonical_decision = (
                "current_observed_center_symmetric_chamfer"
                if sufficient
                else "insufficient_support_persistence"
            )
            legacy_selector_decision = (
                "current_observation_chamfer"
                if sufficient
                else "insufficient_support_persistence_default"
            )
            legacy_gate_decision = (
                "accepted_without_covariance_gate"
                if sufficient
                else (
                    "insufficient_valid_covariance"
                    if fresh_update.get("available_center_count") == 0
                    else "insufficient_selector_support"
                )
            )
            support_equal = (
                fresh_update.get("selector_decision") == canonical_decision
                and fresh_gate.get("decision") == canonical_decision
                and reference_update.get("selector_decision")
                == legacy_selector_decision
                and reference_gate.get("decision") == legacy_gate_decision
            ) and all(
                fresh_gate.get(field) == reference_gate.get(field)
                for field in gate_fields
            )
            update_checks.append(
                {
                    "frame": fresh_update.get("frame"),
                    "selection_metadata_bit_exact": selection_equal,
                    "support_semantics_equivalent": support_equal,
                }
            )
        arrays_exact = all(array_checks.values())
        metadata_exact = all(metadata_checks.values()) and all(
            record["selection_metadata_bit_exact"]
            and record["support_semantics_equivalent"]
            for record in update_checks
        )
        scores_close = all(score_checks.values())
        per_case.append(
            {
                "case": case,
                "primary_arrays_bit_exact": arrays_exact,
                "metadata_bit_exact_after_legacy_normalization": metadata_exact,
                "scores_within_absolute_tolerance": scores_close,
                "parity_passed": arrays_exact and metadata_exact and scores_close,
                "trajectory_bit_exact": array_checks,
                "metadata_bit_exact": metadata_checks,
                "score_within_absolute_tolerance": score_checks,
                "score_absolute_tolerance": 1.0e-12,
                "updates": update_checks,
            }
        )
    passed = all(record["parity_passed"] for record in per_case)
    _require(passed, "fresh 8-view output differs from frozen gated reference")
    return {
        "episode_count": len(per_case),
        "parity_passed": passed,
        "all_primary_arrays_bit_exact": all(
            record["primary_arrays_bit_exact"] for record in per_case
        ),
        "all_metadata_bit_exact_after_legacy_normalization": all(
            record["metadata_bit_exact_after_legacy_normalization"]
            for record in per_case
        ),
        "all_scores_within_absolute_tolerance": all(
            record["scores_within_absolute_tolerance"] for record in per_case
        ),
        "score_absolute_tolerance": 1.0e-12,
        "cases": per_case,
    }


def _cross_budget_checks(
    cases: Sequence[str],
    manifests: Mapping[int, Mapping[str, Mapping[str, Any]]],
    reports: Mapping[int, Mapping[str, Mapping[str, Any]]],
    arrays: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Any]:
    for case in cases:
        case_manifests = [manifests[count][case] for count in CAMERA_COUNTS]
        _require(
            case_manifests[0]["plan"]["center_ids"]
            == case_manifests[1]["plan"]["center_ids"]
            == case_manifests[2]["plan"]["center_ids"],
            f"center IDs differ across camera budgets for {case}",
        )
        _require(
            case_manifests[0]["inputs"]
            == case_manifests[1]["inputs"]
            == case_manifests[2]["inputs"],
            f"immutable inputs differ across camera budgets for {case}",
        )
        planning = [
            {
                key: manifest["plan"][key]
                for key in ("candidate_count", "candidate_ids", "selection_inputs")
            }
            for manifest in case_manifests
        ]
        _require(
            planning[0] == planning[1] == planning[2],
            f"all-camera planning differs across budgets for {case}",
        )
        camera_inputs = {
            count: manifests[count][case]["selected_camera_inputs"]
            for count in CAMERA_COUNTS
        }
        for camera in set().union(*(set(value) for value in camera_inputs.values())):
            occurrences = [
                camera_inputs[count][camera]
                for count in CAMERA_COUNTS
                if camera in camera_inputs[count]
            ]
            _require(
                all(record == occurrences[0] for record in occurrences[1:]),
                f"overlapping camera input bytes differ for {case}/{camera}",
            )
        for field in (
            "scored_frames",
            "rbf_config",
            "support_gate_contract",
            "algorithm_binding",
        ):
            values = [reports[count][case][field] for count in CAMERA_COUNTS]
            _require(
                values[0] == values[1] == values[2],
                f"primary method changed across camera budgets for {case}/{field}",
            )
        for arm in ("physical_prior", "persistence"):
            _require(
                _arrays_bit_exact(arrays[2][case][arm], arrays[4][case][arm])
                and _arrays_bit_exact(arrays[4][case][arm], arrays[8][case][arm]),
                f"{arm} trajectory changed across camera budgets for {case}",
            )
            for metric in PRIMARY_METRICS:
                values = [
                    reports[count][case]["scores"][arm][metric]
                    for count in CAMERA_COUNTS
                ]
                _require(
                    values[0] == values[1] == values[2],
                    f"{arm} score changed across camera budgets for {case}/{metric}",
                )
    return {
        "exact_center_id_equality_across_budgets": True,
        "exact_immutable_input_hash_equality_across_budgets": True,
        "exact_all_camera_planning_equality_across_budgets": True,
        "exact_overlapping_camera_input_hash_equality_across_budgets": True,
        "exact_primary_method_contract_equality_across_budgets": True,
        "physical_and_persistence_trajectories_bit_exact_across_budgets": True,
        "physical_and_persistence_scores_exact_across_budgets": True,
    }


def _score(
    reports: Mapping[str, Mapping[str, Any]],
    case: str,
    arm: str,
    metric: str,
) -> float:
    return float(reports[case]["scores"][arm][metric])


def _budget_analysis(
    camera_count: int,
    reports: Mapping[str, Mapping[str, Any]],
    cases: Sequence[str],
    objects: Mapping[str, str],
) -> dict[str, Any]:
    object_ids = tuple(sorted(set(objects.values())))
    aggregate = {
        arm: {
            metric: float(
                np.mean([_score(reports, case, arm, metric) for case in cases])
            )
            for metric in PRIMARY_METRICS
        }
        for arm in PRIMARY_ARMS
    }
    comparisons: dict[str, Any] = {}
    for comparator in COMPARATORS:
        metrics: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            candidate = {
                case: _score(reports, case, PRIMARY_ARM, metric) for case in cases
            }
            baseline = {
                case: _score(reports, case, comparator, metric) for case in cases
            }
            differences = {case: candidate[case] - baseline[case] for case in cases}
            relative_changes = {}
            for case in cases:
                _require(
                    baseline[case] > 0.0,
                    f"zero comparator prevents relative change for {case}/{metric}",
                )
                relative_changes[case] = candidate[case] / baseline[case] - 1.0
            per_object = {}
            for object_id in object_ids:
                object_cases = [case for case in cases if objects[case] == object_id]
                candidate_mean = float(
                    np.mean([candidate[case] for case in object_cases])
                )
                comparator_mean = float(
                    np.mean([baseline[case] for case in object_cases])
                )
                per_object[object_id] = {
                    "case_count": len(object_cases),
                    "candidate_mean_m": candidate_mean,
                    "comparator_mean_m": comparator_mean,
                    "mean_difference_m": candidate_mean - comparator_mean,
                    "relative_change": candidate_mean / comparator_mean - 1.0,
                    "case_wins": int(
                        sum(differences[case] < 0.0 for case in object_cases)
                    ),
                }
            candidate_mean = float(np.mean(list(candidate.values())))
            comparator_mean = float(np.mean(list(baseline.values())))
            metrics[metric] = {
                "candidate_equal_case_mean_m": candidate_mean,
                "comparator_equal_case_mean_m": comparator_mean,
                "equal_case_mean_difference_m": candidate_mean - comparator_mean,
                "relative_change": candidate_mean / comparator_mean - 1.0,
                "relative_improvement": 1.0 - candidate_mean / comparator_mean,
                "case_wins": int(sum(value < 0.0 for value in differences.values())),
                "per_object": per_object,
                "per_case_difference_m": differences,
                "per_case_relative_change": relative_changes,
            }
        comparisons[comparator] = {
            "metrics": metrics,
            "joint_case_wins": int(
                sum(
                    all(
                        _score(reports, case, PRIMARY_ARM, metric)
                        < _score(reports, case, comparator, metric)
                        for metric in PRIMARY_METRICS
                    )
                    for case in cases
                )
            ),
        }
    return {
        "camera_count": camera_count,
        "role": (
            "descriptive_only"
            if camera_count == 2
            else "candidate"
            if camera_count == 4
            else "reference"
        ),
        "aggregate_scores": aggregate,
        "comparisons": comparisons,
    }


def _four_view_decision(analyses: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    physical_4 = analyses[4]["comparisons"]["physical_prior"]
    physical_8 = analyses[8]["comparisons"]["physical_prior"]
    retention: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        improvement_4 = physical_4["metrics"][metric]["relative_improvement"]
        improvement_8 = physical_8["metrics"][metric]["relative_improvement"]
        valid = improvement_8 > 0.0
        fraction = improvement_4 / improvement_8 if valid else None
        retention[metric] = {
            "four_view_relative_improvement": improvement_4,
            "eight_view_relative_improvement": improvement_8,
            "fraction_retained": fraction,
            "minimum": 0.8,
            "passed": bool(fraction is not None and fraction >= 0.8),
        }
    joint = int(physical_4["joint_case_wins"])
    joint_check = {"observed": joint, "minimum": 18, "passed": joint >= 18}
    per_object: dict[str, Any] = {}
    for object_id in physical_4["metrics"][PRIMARY_METRICS[0]]["per_object"]:
        differences = {
            metric: physical_4["metrics"][metric]["per_object"][object_id][
                "mean_difference_m"
            ]
            for metric in PRIMARY_METRICS
        }
        per_object[object_id] = {
            "mean_difference_m": differences,
            "passed": all(value < 0.0 for value in differences.values()),
        }
    _require(len(per_object) == 5, "object-level gate no longer has five objects")
    chamfer = PRIMARY_METRICS[1]
    relative = physical_4["metrics"][chamfer]["per_case_relative_change"]
    maximum = max(relative.values())
    regression = {
        "maximum_observed_relative_regression": maximum,
        "maximum_allowed": 0.1,
        "inclusive_bound": True,
        "passed": maximum <= 0.1,
        "per_case_relative_change": relative,
        "per_case_difference_m": physical_4["metrics"][chamfer][
            "per_case_difference_m"
        ],
    }
    field_value_metrics = {
        metric: (
            analyses[4]["aggregate_scores"][PRIMARY_ARM][metric]
            < analyses[4]["aggregate_scores"][SELECTED_BACKBONE_ARM][metric]
        )
        for metric in PRIMARY_METRICS
    }
    checks = [
        *(record["passed"] for record in retention.values()),
        joint_check["passed"],
        *(record["passed"] for record in per_object.values()),
        regression["passed"],
    ]
    passed = all(checks)
    return {
        "camera_count": 4,
        "reference_camera_count": 8,
        "status": "GO" if passed else "NO_GO",
        "passed": passed,
        "retains_at_least_80_percent_of_8_view_relative_improvement": retention,
        "joint_case_wins_vs_physical": joint_check,
        "all_five_objects_improve_on_both_primary_metrics": {
            "passed": all(record["passed"] for record in per_object.values()),
            "objects": per_object,
        },
        "case_chamfer_regression_vs_physical": regression,
        "secondary_field_value_vs_selected_raw_backbone": {
            "per_metric_improved": field_value_metrics,
            "improved_both_primary_metrics": all(field_value_metrics.values()),
            "overrides_primary_decision": False,
        },
        "two_view_role": "descriptive_only; never enters this decision",
        "tie_policy": "ties are not improvements or wins; 10% regression passes",
    }


def build_transfer_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every frozen artifact and construct the transfer result."""

    _validate_config(config)
    parent_transfer, raw_config, parent_bindings = _load_parent_configs(config)
    code_bindings = _validate_code_bindings(config)
    runtime_binding = _validate_runtime_binding(config)
    cases, objects, episodes = _expected_case_metadata(raw_config)
    measurement_inventories: dict[int, TreeInventory] = {}
    manifests: dict[int, dict[str, dict[str, Any]]] = {}
    for camera_count in CAMERA_COUNTS:
        binding = config["bound_inputs"]["measurements"][str(camera_count)]
        inventory = inventory_tree(binding["root"])
        _validate_inventory_binding(
            inventory,
            binding,
            label=f"{camera_count}-view measurement",
        )
        manifests[camera_count] = _validate_measurement_root(
            inventory,
            raw_config,
            camera_count,
            cases,
            objects,
            episodes,
        )
        measurement_inventories[camera_count] = inventory

    # Inventory and require the complete 55-file layout for every fresh budget
    # before parsing any outcome-bearing summary.
    evaluation_inventories = {
        camera_count: inventory_tree(
            config["outputs"]["primary_evaluations"][str(camera_count)]
        )
        for camera_count in CAMERA_COUNTS
    }
    for camera_count in CAMERA_COUNTS:
        _require(
            set(evaluation_inventories[camera_count].sha256_by_relative_path)
            == _expected_evaluation_paths(cases),
            f"{camera_count}-view primary output is not terminal and complete",
        )

    reference_binding = config["bound_inputs"]["gated_8_reference"]
    reference_inventory = inventory_tree(reference_binding["root"])
    reference_summary, reference_reports, reference_arrays = _validate_gated_reference(
        reference_inventory,
        reference_binding,
        cases,
        objects,
        episodes,
    )
    saved_parity = _validate_saved_parity(
        config,
        reference_summary,
        reference_inventory,
        cases,
    )

    summaries: dict[int, dict[str, Any]] = {}
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    arrays: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for camera_count in CAMERA_COUNTS:
        inventory = evaluation_inventories[camera_count]
        summary, case_reports, case_arrays = _validate_primary_evaluation_root(
            inventory,
            measurement_inventories[camera_count],
            manifests[camera_count],
            cases,
            objects,
            episodes,
            camera_count=camera_count,
        )
        summaries[camera_count] = summary
        reports[camera_count] = case_reports
        arrays[camera_count] = case_arrays

    independent_parity = _independent_eight_view_parity(
        cases,
        reports[8],
        arrays[8],
        reference_reports,
        reference_arrays,
    )
    cross_budget = _cross_budget_checks(cases, manifests, reports, arrays)
    analyses = {
        camera_count: _budget_analysis(
            camera_count,
            reports[camera_count],
            cases,
            objects,
        )
        for camera_count in CAMERA_COUNTS
    }
    decision = _four_view_decision(analyses)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "panel": {
            "role": "outcome-open development only",
            "episode_count": len(cases),
            "physical_object_count": len(set(objects.values())),
            "cases": list(cases),
        },
        "parents": parent_bindings,
        "freeze_boundary": config["freeze"],
        "parent_transfer_result_role": parent_transfer["claim_boundary"],
        "implementation": code_bindings,
        "runtime": runtime_binding,
        "method": config["method"],
        "excluded_partial_roots": {
            **config["excluded_partial_roots"],
            "validation": (
                "lexical exclusion only; failed partial roots were not statted, "
                "opened, inventoried, or interpreted"
            ),
        },
        "input_inventories": {
            "measurements": {
                str(count): measurement_inventories[count].summary()
                for count in CAMERA_COUNTS
            },
            "gated_8_reference": {
                **reference_inventory.summary(),
                "summary_file_sha256": reference_inventory.sha256_by_relative_path[
                    "summary.json"
                ],
                "summary_result_sha256": reference_summary["result_sha256"],
            },
        },
        "saved_parity_precondition": saved_parity,
        "independent_fresh_8_view_parity": independent_parity,
        "fresh_output_inventories": {
            str(count): {
                **evaluation_inventories[count].summary(),
                "summary_file_sha256": evaluation_inventories[
                    count
                ].sha256_by_relative_path["summary.json"],
                "summary_result_sha256": summaries[count]["result_sha256"],
            }
            for count in CAMERA_COUNTS
        },
        "cross_budget_invariants": cross_budget,
        "budgets": {str(count): analyses[count] for count in CAMERA_COUNTS},
        "decision": decision,
        "claim_boundary": config["claim_boundary"],
    }


def _publish_report(report: Mapping[str, Any], destination: str | Path) -> Path:
    output = Path(destination)
    _require(not output.exists(), f"analysis output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        final_report = dict(report)
        final_report.pop("result_sha256", None)
        final_report["result_sha256"] = _canonical_sha256(final_report)
        (staging / "frontier.json").write_text(
            json.dumps(final_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _require(
            not output.exists(),
            f"analysis output appeared during publication: {output}",
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "frontier.json"


def analyze_primary_camera_budget_transfer(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate once and atomically publish the frozen transfer analysis."""

    path = Path(config_path).resolve(strict=True)
    config, config_bytes = _load_stable_json(path, label="transfer config")
    _validate_config(config)
    configured_output = config["outputs"]["analysis_output"]
    if output is not None:
        _require(
            os.path.normpath(str(output)) == configured_output,
            "CLI output override differs from the frozen analysis output",
        )
    report = build_transfer_report(config)
    report["config"] = {
        "path": str(path),
        "sha256": _sha256_bytes(config_bytes),
    }
    report["result_sha256"] = _canonical_sha256(report)
    _publish_report(report, configured_output)
    return report


__all__ = [
    "COMPARATORS",
    "EXCLUDED_PARTIAL_ROOTS",
    "FROZEN_STATUS",
    "PROTOCOL_ID",
    "analyze_primary_camera_budget_transfer",
    "build_transfer_report",
]
