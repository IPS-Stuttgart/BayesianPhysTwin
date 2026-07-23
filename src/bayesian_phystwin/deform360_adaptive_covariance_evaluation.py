"""Strict open-panel evaluation of adaptive covariance-ranked RBF prediction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_adaptive_covariance_rbf import (
    ADAPTIVE_COVARIANCE_PROTOCOL_ID,
    predict_adaptive_covariance_selected_backbone_rbf,
)
from .deform360_online_belief_evaluation import (
    PRIMARY_METRICS,
    _physical_object_cluster_bootstrap,
    score_deform360_hidden_trajectory,
)
from .deform360_raw_camera_gated_evaluation import _load_uncertainty_artifact
from .deform360_raw_camera_budget_frontier import TreeInventory, inventory_tree
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
    _load_open_case_for_evaluation,
    _sha256,
    expected_open_case_names,
)
from .deform360_raw_camera_primary_evaluation import (
    _VerifiedMeasurement,
    _arrays_are_bit_exact,
    _load_verified_measurement,
    _post_update_scored_frames,
    _recheck_verified_inputs,
    _sha256_array,
    _sign_artifact,
)
from .deform360_raw_camera_uncertainty import (
    UNCERTAINTY_ARCHIVE_FILENAME,
    UNCERTAINTY_MANIFEST_FILENAME,
)


ADAPTIVE_EVALUATION_PROTOCOL_ID = (
    "deform360-open27-adaptive-covariance-selected-rbf-v1-development"
)
DEVELOPMENT_CONFIG_PROTOCOL_ID = (
    "deform360-open27-adaptive-covariance-view-budget-v1-development"
)
ADAPTIVE_ARM = "adaptive_covariance_selected_backbone_rbf"
ADAPTIVE_RAW_ARM = "adaptive_covariance_selected_raw_backbone"
ARMS = ("physical_prior", "persistence", ADAPTIVE_RAW_ARM, ADAPTIVE_ARM)
CAMERA_BUDGETS = (4, 8)
EXPECTED_INPUT_INVENTORIES: Mapping[str, Mapping[int, Mapping[str, int | str]]] = {
    "measurement": {
        4: {
            "file_count": 56,
            "total_file_bytes": 1_643_339,
            "inventory_sha256": (
                "baa9c35a91da4eb3843cdcbb63889d0e2fd60532122da183cad1eda3a4c82141"
            ),
        },
        8: {
            "file_count": 56,
            "total_file_bytes": 2_202_895,
            "inventory_sha256": (
                "3aad3d81fb30e5701bf39011c8c9d00483c35555178fc2c7f4f83afeda9342b9"
            ),
        },
    },
    "uncertainty": {
        4: {
            "file_count": 56,
            "total_file_bytes": 1_794_611,
            "inventory_sha256": (
                "e055c903323b8b054c3730618b97caaf85a60e9613933c8633ef397b7c77beeb"
            ),
        },
        8: {
            "file_count": 56,
            "total_file_bytes": 2_358_852,
            "inventory_sha256": (
                "5da46cf3859ada32f9c5d8d38ef4958331c51c4554300f2ae60b2266832552c1"
            ),
        },
    },
}
DEVELOPMENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "sota"
    / "deform360_adaptive_covariance_view_budget_v1_development.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _VerifiedUncertainty:
    root: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    manifest_sha256: str
    archive_sha256: str


@dataclass(frozen=True)
class _VerifiedAdaptiveCase:
    case: str
    measurements: Mapping[int, _VerifiedMeasurement]
    uncertainties: Mapping[int, _VerifiedUncertainty]
    prediction: np.ndarray
    selected_raw: np.ndarray
    diagnostic: Mapping[str, Any]
    prediction_array_sha256: str
    selected_raw_array_sha256: str


def _verify_input_inventory(
    root: Path,
    *,
    role: str,
    budget: int,
) -> TreeInventory:
    observed = inventory_tree(root)
    expected = EXPECTED_INPUT_INVENTORIES[role][budget]
    for field in ("file_count", "total_file_bytes", "inventory_sha256"):
        if getattr(observed, field) != expected[field]:
            raise ValueError(f"{budget}-view {role} inventory {field} changed")
    return observed


def _load_development_config() -> dict[str, Any]:
    try:
        config = json.loads(DEVELOPMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "adaptive development config is unavailable or invalid"
        ) from error
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or config.get("protocol_id") != DEVELOPMENT_CONFIG_PROTOCOL_ID
    ):
        raise ValueError("adaptive development config identity changed")
    method = config.get("method", {})
    if (
        method.get("camera_budgets") != list(CAMERA_BUDGETS)
        or method.get("minimum_covariance_valid_centers") != 8
        or method.get("maximum_normalized_covariance_dispersion") != 0.015
        or method.get("route_order") != "try four views, then eight views"
    ):
        raise ValueError("adaptive development method changed")
    bound = config.get("bound_inputs", {})
    role_keys = {"measurement": "measurements", "uncertainty": "uncertainty"}
    for role, config_key in role_keys.items():
        for budget in CAMERA_BUDGETS:
            record = bound.get(config_key, {}).get(str(budget), {})
            expected = EXPECTED_INPUT_INVENTORIES[role][budget]
            if any(record.get(field) != expected[field] for field in expected):
                raise ValueError(
                    f"{budget}-view {role} config inventory binding changed"
                )
    parent = config.get("parents", {}).get("selected_rbf_camera_budget_config", {})
    parent_path = Path(__file__).resolve().parents[2] / str(parent.get("path", ""))
    if not parent_path.is_file() or parent.get("sha256") != _sha256(parent_path):
        raise ValueError("adaptive development parent config binding changed")
    return {
        "path": DEVELOPMENT_CONFIG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "file_sha256": _sha256(DEVELOPMENT_CONFIG_PATH),
        "protocol_id": DEVELOPMENT_CONFIG_PROTOCOL_ID,
        "parent_config": {
            "path": parent_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "file_sha256": _sha256(parent_path),
        },
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_root_separation(
    panel: Path,
    measurements: Mapping[int, Path],
    uncertainties: Mapping[int, Path],
    output: Path,
) -> None:
    named_inputs = {
        "panel": panel,
        **{
            f"{budget}-view measurement": measurements[budget]
            for budget in CAMERA_BUDGETS
        },
        **{
            f"{budget}-view uncertainty": uncertainties[budget]
            for budget in CAMERA_BUDGETS
        },
    }
    items = tuple(named_inputs.items())
    for index, (left_label, left) in enumerate(items):
        for right_label, right in items[index + 1 :]:
            if _paths_overlap(left, right):
                raise ValueError(f"{left_label} and {right_label} roots overlap")
        if _paths_overlap(left, output):
            raise ValueError(f"output overlaps {left_label} root")


def _selected_cameras(
    measurement: _VerifiedMeasurement,
    *,
    budget: int,
) -> tuple[str, ...]:
    manifest = measurement.manifest
    config = manifest.get("config", {})
    plan = manifest.get("plan", {})
    cameras = plan.get("selected_cameras")
    if (
        not isinstance(config, Mapping)
        or config.get("selected_camera_count") != budget
        or not isinstance(cameras, list)
        or len(cameras) != budget
        or len(set(cameras)) != budget
        or any(not isinstance(camera, str) or not camera for camera in cameras)
    ):
        raise ValueError(f"{budget}-view measurement camera budget changed")
    selected_inputs = manifest.get("selected_camera_inputs")
    if not isinstance(selected_inputs, Mapping) or set(selected_inputs) != set(cameras):
        raise ValueError(f"{budget}-view selected-camera inputs changed")
    if "selected_cameras" not in measurement.arrays:
        raise ValueError(f"{budget}-view archive lacks selected cameras")
    archived = np.asarray(measurement.arrays["selected_cameras"])
    if archived.ndim != 1 or archived.tolist() != cameras:
        raise ValueError(f"{budget}-view manifest/archive cameras differ")
    return tuple(cameras)


def _load_verified_uncertainty(
    measurement: _VerifiedMeasurement,
    uncertainty_root: str | Path,
) -> _VerifiedUncertainty:
    root = Path(uncertainty_root).resolve()
    manifest_path = root / UNCERTAINTY_MANIFEST_FILENAME
    archive_path = root / UNCERTAINTY_ARCHIVE_FILENAME
    manifest_sha256 = _sha256(manifest_path)
    archive_sha256 = _sha256(archive_path)
    manifest, arrays = _load_uncertainty_artifact(
        measurement.measurement_dir,
        root,
        measurement.seal,
    )
    required = {"measurement_covariance_m2", "measurement_covariance_valid"}
    if not required.issubset(arrays):
        raise ValueError("uncertainty artifact lacks adaptive covariance arrays")
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("uncertainty manifest changed while loading")
    if _sha256(archive_path) != archive_sha256:
        raise ValueError("uncertainty archive changed while loading")
    return _VerifiedUncertainty(
        root=root,
        manifest=manifest,
        arrays=arrays,
        manifest_sha256=manifest_sha256,
        archive_sha256=archive_sha256,
    )


def _recheck_uncertainty(
    uncertainty: _VerifiedUncertainty,
    *,
    boundary: str,
) -> None:
    checks = (
        (
            uncertainty.root / UNCERTAINTY_MANIFEST_FILENAME,
            uncertainty.manifest_sha256,
            "uncertainty manifest",
        ),
        (
            uncertainty.root / UNCERTAINTY_ARCHIVE_FILENAME,
            uncertainty.archive_sha256,
            "uncertainty archive",
        ),
    )
    for path, expected, label in checks:
        if _sha256(path) != expected:
            raise ValueError(f"{label} changed before {boundary}")


def _load_verified_adaptive_case(
    panel_case_dir: str | Path,
    measurement_dirs: Mapping[int, str | Path],
    uncertainty_dirs: Mapping[int, str | Path],
) -> _VerifiedAdaptiveCase:
    if set(measurement_dirs) != set(CAMERA_BUDGETS):
        raise ValueError("measurement budgets must be exactly 4 and 8")
    if set(uncertainty_dirs) != set(CAMERA_BUDGETS):
        raise ValueError("uncertainty budgets must be exactly 4 and 8")
    measurements = {
        budget: _load_verified_measurement(
            panel_case_dir,
            measurement_dirs[budget],
        )
        for budget in CAMERA_BUDGETS
    }
    four = measurements[4]
    for budget in CAMERA_BUDGETS[1:]:
        candidate = measurements[budget]
        if candidate.seal != four.seal:
            raise ValueError("camera budgets bind different prediction seals")
        if not _arrays_are_bit_exact(
            candidate.physical_prior,
            four.physical_prior,
        ) or not _arrays_are_bit_exact(candidate.persistence, four.persistence):
            raise ValueError("backbones changed across camera budgets")
        if not np.array_equal(
            candidate.arrays["center_ids"],
            four.arrays["center_ids"],
        ):
            raise ValueError("center IDs changed across camera budgets")

    uncertainties = {
        budget: _load_verified_uncertainty(
            measurements[budget],
            uncertainty_dirs[budget],
        )
        for budget in CAMERA_BUDGETS
    }
    selected_cameras = {
        budget: _selected_cameras(measurements[budget], budget=budget)
        for budget in CAMERA_BUDGETS
    }
    prediction, selected_raw, diagnostic = (
        predict_adaptive_covariance_selected_backbone_rbf(
            four.physical_prior,
            four.persistence,
            selected_cameras,
            {
                budget: measurements[budget].arrays["measurement_m"]
                for budget in CAMERA_BUDGETS
            },
            {
                budget: measurements[budget].arrays["measurement_validity"]
                for budget in CAMERA_BUDGETS
            },
            {
                budget: uncertainties[budget].arrays["measurement_covariance_m2"]
                for budget in CAMERA_BUDGETS
            },
            {
                budget: uncertainties[budget].arrays["measurement_covariance_valid"]
                for budget in CAMERA_BUDGETS
            },
            center_ids=np.asarray(four.arrays["center_ids"], dtype=np.int64),
        )
    )
    return _VerifiedAdaptiveCase(
        case=four.case_dir.name,
        measurements=measurements,
        uncertainties=uncertainties,
        prediction=prediction,
        selected_raw=selected_raw,
        diagnostic=diagnostic,
        prediction_array_sha256=_sha256_array(prediction),
        selected_raw_array_sha256=_sha256_array(selected_raw),
    )


def _recheck_adaptive_inputs(
    verified: _VerifiedAdaptiveCase,
    *,
    boundary: str,
) -> None:
    for budget in CAMERA_BUDGETS:
        _recheck_verified_inputs(
            verified.measurements[budget],
            boundary=boundary,
        )
        _recheck_uncertainty(
            verified.uncertainties[budget],
            boundary=boundary,
        )


def _evaluate_verified_adaptive_case(
    verified: _VerifiedAdaptiveCase,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _recheck_adaptive_inputs(verified, boundary="target open")
    four = verified.measurements[4]
    open_seal, prior, persistence, target, visibility, validity = (
        _load_open_case_for_evaluation(four.case_dir)
    )
    if open_seal != four.seal:
        raise ValueError("prediction seal changed while opening the outcome")
    if not _arrays_are_bit_exact(prior, four.physical_prior):
        raise ValueError("physical prior changed while opening the outcome")
    if not _arrays_are_bit_exact(persistence, four.persistence):
        raise ValueError("persistence changed while opening the outcome")
    if _sha256_array(verified.prediction) != verified.prediction_array_sha256:
        raise ValueError("adaptive prediction changed before scoring")
    if _sha256_array(verified.selected_raw) != verified.selected_raw_array_sha256:
        raise ValueError("adaptive raw prediction changed before scoring")

    centers = np.asarray(four.arrays["center_ids"], dtype=np.int64)
    scored_frames = _post_update_scored_frames(len(target))
    trajectories = {
        "physical_prior": four.physical_prior.copy(),
        "persistence": four.persistence.copy(),
        ADAPTIVE_RAW_ARM: verified.selected_raw.copy(),
        ADAPTIVE_ARM: verified.prediction.copy(),
    }
    scores = {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visibility,
            validity,
            center_ids=centers,
            scored_frames=scored_frames,
        )
        for arm, trajectory in trajectories.items()
    }
    report = {
        "schema_version": 1,
        "protocol_id": ADAPTIVE_EVALUATION_PROTOCOL_ID,
        "predictor_protocol_id": ADAPTIVE_COVARIANCE_PROTOCOL_ID,
        "case": verified.case,
        "object_id": str(four.seal["object_id"]),
        "episode_id": int(four.seal["episode_id"]),
        "center_ids": centers.tolist(),
        "scored_frames": list(scored_frames),
        "adaptive_arm": ADAPTIVE_ARM,
        "adaptive_raw_arm": ADAPTIVE_RAW_ARM,
        "prediction_array_sha256_before_target_open": (
            verified.prediction_array_sha256
        ),
        "selected_raw_array_sha256_before_target_open": (
            verified.selected_raw_array_sha256
        ),
        "inputs": {
            str(budget): {
                "prediction_seal_sha256": verified.measurements[
                    budget
                ].prediction_seal_sha256,
                "measurement_manifest_sha256": verified.measurements[
                    budget
                ].measurement_manifest_sha256,
                "measurement_archive_sha256": verified.measurements[
                    budget
                ].measurement_archive_sha256,
                "uncertainty_manifest_sha256": verified.uncertainties[
                    budget
                ].manifest_sha256,
                "uncertainty_archive_sha256": verified.uncertainties[
                    budget
                ].archive_sha256,
            }
            for budget in CAMERA_BUDGETS
        },
        "diagnostic": verified.diagnostic,
        "scores": scores,
        "information_boundary": {
            "all_measurements_and_uncertainties_verified_before_target_open": True,
            "adaptive_prediction_completed_and_hashed_before_target_open": True,
            "predictor_target_argument_accepted": False,
            "target_role": "scoring only",
        },
        "claim_boundary": (
            "outcome-open development evaluation on reconstructed proxy targets; "
            "covariance is a target-free ranking score, not calibrated "
            "uncertainty; not official Deform360 or SOTA evidence"
        ),
    }
    return _sign_artifact(report), trajectories


def _write_case(
    output: Path,
    report: Mapping[str, Any],
    trajectories: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, str]]:
    case = str(report["case"])
    archive_path = output / f"{case}.npz"
    report_path = output / f"{case}.json"
    np.savez_compressed(archive_path, **trajectories)
    emitted = dict(report)
    emitted.pop("result_sha256", None)
    emitted["trajectory_archive_sha256"] = _sha256(archive_path)
    emitted = _sign_artifact(emitted)
    report_path.write_text(
        json.dumps(emitted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return emitted, {
        "case": case,
        "report_sha256": _sha256(report_path),
        "report_result_sha256": str(emitted["result_sha256"]),
        "archive_sha256": _sha256(archive_path),
    }


def _summary(
    reports: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, str]],
    input_inventories: Mapping[str, Mapping[int, TreeInventory]],
    config_binding: Mapping[str, Any],
) -> dict[str, Any]:
    groups = {str(report["case"]): str(report["object_id"]) for report in reports}
    aggregate = {
        arm: {
            metric: float(
                np.mean([report["scores"][arm][metric] for report in reports])
            )
            for metric in PRIMARY_METRICS
        }
        for arm in ARMS
    }
    comparisons: dict[str, Any] = {}
    for baseline in ("physical_prior", "persistence", ADAPTIVE_RAW_ARM):
        for metric in PRIMARY_METRICS:
            differences = {
                str(report["case"]): float(
                    report["scores"][ADAPTIVE_ARM][metric]
                    - report["scores"][baseline][metric]
                )
                for report in reports
            }
            comparison = _physical_object_cluster_bootstrap(differences, groups)
            comparison["episode_wins"] = int(
                np.sum(np.asarray(list(differences.values())) < 0.0)
            )
            comparison["relative_change"] = (
                aggregate[ADAPTIVE_ARM][metric] / aggregate[baseline][metric] - 1.0
            )
            comparison["per_case_difference_m"] = differences
            comparisons[f"{ADAPTIVE_ARM}:vs:{baseline}:{metric}"] = comparison

    route_counts: dict[str, int] = {
        "4_view_rbf": 0,
        "8_view_rbf": 0,
        "physical_prior_fallback": 0,
    }
    tracked_counts: list[int] = []
    for report in reports:
        for update in report["diagnostic"]["updates"]:
            route_counts[str(update["route"])] += 1
            tracked_counts.append(int(update["tracked_camera_count"]))
    physical_chamfer = {
        str(report["case"]): float(
            report["scores"]["physical_prior"]["post_update_hidden_symmetric_chamfer_m"]
        )
        for report in reports
    }
    adaptive_chamfer = {
        str(report["case"]): float(
            report["scores"][ADAPTIVE_ARM]["post_update_hidden_symmetric_chamfer_m"]
        )
        for report in reports
    }
    relative_chamfer = {
        case: adaptive_chamfer[case] / physical_chamfer[case] - 1.0
        for case in physical_chamfer
    }
    joint_wins_vs_physical = int(
        sum(
            all(
                report["scores"][ADAPTIVE_ARM][metric]
                < report["scores"]["physical_prior"][metric]
                for metric in PRIMARY_METRICS
            )
            for report in reports
        )
    )
    return _sign_artifact(
        {
            "schema_version": 1,
            "protocol_id": ADAPTIVE_EVALUATION_PROTOCOL_ID,
            "predictor_protocol_id": ADAPTIVE_COVARIANCE_PROTOCOL_ID,
            "episode_count": len(reports),
            "physical_object_count": len(set(groups.values())),
            "adaptive_arm": ADAPTIVE_ARM,
            "aggregate": aggregate,
            "comparisons": comparisons,
            "routing": {
                "counts": route_counts,
                "distinct_tracked_camera_count_equal_interval_mean": float(
                    np.mean(tracked_counts)
                ),
                "tracked_camera_count_semantics": (
                    "distinct dynamic RGB prefixes activated by the sequential "
                    "4-to-8 cascade after all-view frame-zero planning"
                ),
            },
            "bound_input_inventories": {
                role: {
                    str(budget): inventory.summary()
                    for budget, inventory in by_budget.items()
                }
                for role, by_budget in input_inventories.items()
            },
            "development_config": dict(config_binding),
            "tail_vs_physical": {
                "per_case_chamfer_relative_change": relative_chamfer,
                "maximum_chamfer_relative_regression": float(
                    max(relative_chamfer.values())
                ),
                "case_count_above_10_percent": int(
                    sum(value > 0.10 for value in relative_chamfer.values())
                ),
            },
            "joint_episode_wins_vs_physical": joint_wins_vs_physical,
            "artifacts": list(artifacts),
            "information_boundary": {
                "all_27_predictions_completed_and_hashed_before_any_target_open": (
                    True
                ),
                "all_inputs_rechecked_before_target_open_and_artifact_emission": (True),
                "target_free_routing": True,
            },
            "claim_boundary": (
                "post-hoc threshold-grid-selected open27 development result; "
                "requires fresh-object confirmation and cannot support a "
                "calibration, safety, or state-of-the-art claim"
            ),
        }
    )


def evaluate_adaptive_covariance_cohort(
    panel_root: str | Path,
    measurement_roots: Mapping[int, str | Path],
    uncertainty_roots: Mapping[int, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate the full open27 panel after a complete target-free verify pass."""

    if set(measurement_roots) != set(CAMERA_BUDGETS):
        raise ValueError("measurement roots must be exactly budgets 4 and 8")
    if set(uncertainty_roots) != set(CAMERA_BUDGETS):
        raise ValueError("uncertainty roots must be exactly budgets 4 and 8")
    panel = Path(panel_root).resolve()
    measurements = {
        budget: Path(root).resolve() for budget, root in measurement_roots.items()
    }
    uncertainties = {
        budget: Path(root).resolve() for budget, root in uncertainty_roots.items()
    }
    input_inventories = {
        "measurement": {
            budget: _verify_input_inventory(
                measurements[budget],
                role="measurement",
                budget=budget,
            )
            for budget in CAMERA_BUDGETS
        },
        "uncertainty": {
            budget: _verify_input_inventory(
                uncertainties[budget],
                role="uncertainty",
                budget=budget,
            )
            for budget in CAMERA_BUDGETS
        },
    }
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"adaptive output already exists: {output}")
    _validate_root_separation(panel, measurements, uncertainties, output)
    config_binding = _load_development_config()
    cases = expected_open_case_names()
    missing = [
        (budget, case)
        for budget in CAMERA_BUDGETS
        for case in cases
        if not (measurements[budget] / case / MANIFEST_FILENAME).is_file()
        or not (measurements[budget] / case / MEASUREMENT_FILENAME).is_file()
        or not (uncertainties[budget] / case / UNCERTAINTY_MANIFEST_FILENAME).is_file()
        or not (uncertainties[budget] / case / UNCERTAINTY_ARCHIVE_FILENAME).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing adaptive inputs: {missing}")

    # Complete every target-free prediction before opening the first target.
    verified = [
        _load_verified_adaptive_case(
            panel / case,
            {budget: measurements[budget] / case for budget in CAMERA_BUDGETS},
            {budget: uncertainties[budget] / case for budget in CAMERA_BUDGETS},
        )
        for case in cases
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        reports: list[dict[str, Any]] = []
        artifacts: list[dict[str, str]] = []
        for case, item in zip(cases, verified, strict=True):
            report, trajectories = _evaluate_verified_adaptive_case(item)
            _recheck_adaptive_inputs(item, boundary="artifact emission")
            emitted, artifact = _write_case(staging, report, trajectories)
            if emitted["case"] != case:
                raise AssertionError("adaptive case order changed")
            reports.append(emitted)
            artifacts.append(artifact)
        summary = _summary(reports, artifacts, input_inventories, config_binding)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(
                f"adaptive output appeared during evaluation: {output}"
            )
        staging.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "ADAPTIVE_ARM",
    "ADAPTIVE_EVALUATION_PROTOCOL_ID",
    "ADAPTIVE_RAW_ARM",
    "evaluate_adaptive_covariance_cohort",
]
