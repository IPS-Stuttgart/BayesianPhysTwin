"""Primary-only selected-backbone RBF evaluation on the open Deform360 panel.

This module deliberately excludes covariance sidecars, covariance calibration,
thresholded covariance gates, and CPD diagnostics.  It implements only the
support-gated ``selected_backbone_euclidean_rbf_ungated`` arm and the raw
backbones needed to interpret it.

Every measurement manifest and archive is checksum-verified before the
corresponding open target is read.  Cohort evaluation strengthens that
boundary by verifying all 27 measurement artifacts before reading any target.
The parity audit is read-only and compares this implementation against the
already materialized covariance-gated 8-view outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_online_belief_evaluation import (
    PRIMARY_METRICS,
    UPDATE_FRAMES,
    _physical_object_cluster_bootstrap,
    _post_update_scored_frames,
    _resolve_prediction_archive,
    score_deform360_hidden_trajectory,
)
from .deform360_raw_camera_gated_evaluation import (
    GATED_EVALUATION_PROTOCOL_ID,
    MINIMUM_SELECTOR_SUPPORT as GATED_MINIMUM_SELECTOR_SUPPORT,
    RBF_ARM_PREFIX,
    SELECTED_BACKBONE_ARM,
)
from .deform360_held_online_prefix import (
    HELD_RBF_CONFIG,
    MINIMUM_SELECTOR_SUPPORT,
    _sha256_array,
    predict_support_gated_selected_backbone_rbf,
)
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
    _canonical_sha256,
    _load_measurement_artifact,
    _load_open_case_for_evaluation,
    _sha256,
    _validate_prediction_seal,
    expected_open_case_names,
)
from .phystwin_online_belief import RecursiveRbfBeliefConfig


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

if MINIMUM_SELECTOR_SUPPORT != GATED_MINIMUM_SELECTOR_SUPPORT:
    raise AssertionError("held and development support thresholds differ")


def primary_artifact_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical self-hash used by primary JSON artifacts."""

    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    return _canonical_sha256(unsigned)


def _sign_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("result_sha256", None)
    result["result_sha256"] = primary_artifact_sha256(result)
    return result


def _validate_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    if value.get("result_sha256") != primary_artifact_sha256(value):
        raise ValueError(f"{label} canonical result hash changed")


def _arrays_are_bit_exact(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bound_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{label} checksum changed")
    return payload


def _values_close(left: Any, right: Any, *, absolute_tolerance: float) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_close(
                left[key],
                right[key],
                absolute_tolerance=absolute_tolerance,
            )
            for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_close(
                left_value,
                right_value,
                absolute_tolerance=absolute_tolerance,
            )
            for left_value, right_value in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    numeric = (int, float, np.integer, np.floating)
    if isinstance(left, numeric) and isinstance(right, numeric):
        return bool(
            np.isclose(
                float(left),
                float(right),
                rtol=0.0,
                atol=absolute_tolerance,
            )
        )
    return left == right


def evaluate_primary_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    measurement_m: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    scored_frames: Sequence[int],
    rbf_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate only the support-gated selected-backbone ungated RBF arm.

    The prediction is produced by the frozen target-free held-method kernel.
    Targets enter only afterward, when the already-produced trajectories are
    scored.
    """

    prediction, selected_raw, diagnostic = predict_support_gated_selected_backbone_rbf(
        physical_prior_m,
        persistence_m,
        measurement_m,
        measurement_validity,
        center_ids=center_ids,
        rbf_config=rbf_config,
    )
    return _score_primary_outputs(
        physical_prior_m,
        persistence_m,
        selected_raw,
        prediction,
        diagnostic,
        target_m,
        visibility,
        validity,
        center_ids=center_ids,
        scored_frames=scored_frames,
    )


def _score_primary_outputs(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    selected_raw_m: np.ndarray,
    prediction_m: np.ndarray,
    diagnostic: Mapping[str, Any],
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    scored_frames: Sequence[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prior_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    target = np.asarray(target_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    if not np.array_equal(
        prior_input[0].astype(np.float32), target[0].astype(np.float32)
    ):
        raise ValueError("frame-zero material identities differ")
    trajectories = {
        "physical_prior": prior_input.copy(),
        "persistence": persistence_input.copy(),
        SELECTED_BACKBONE_ARM: np.asarray(selected_raw_m).copy(),
        PRIMARY_ARM: np.asarray(prediction_m).copy(),
    }
    scored = tuple(int(frame) for frame in scored_frames)
    scores = {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visible,
            valid,
            center_ids=centers,
            scored_frames=scored,
        )
        for arm, trajectory in trajectories.items()
    }
    update_fields = (
        "frame",
        "stop_frame_exclusive",
        "available_center_count",
        "selected_backbone",
        "selector_support_sufficient",
        "current_observation_chamfer_m",
    )
    update_records = [
        {
            **{field: record[field] for field in update_fields},
            "selector_decision": record["selector_decision"],
            "support_gate": {
                "accepted": record["selector_support_sufficient"],
                "decision": record["selector_decision"],
                "selected_backbone": record["selected_backbone"],
                "fallback_backbone": record["selected_backbone"],
                "rbf_correction_applied": record["rbf_correction_applied"],
            },
        }
        for record in diagnostic["updates"]
    ]
    selected_by_update = [
        record["selected_backbone"] for record in diagnostic["updates"]
    ]
    report = {
        "protocol_id": PRIMARY_EVALUATION_PROTOCOL_ID,
        "primary_arm": PRIMARY_ARM,
        "algorithm_binding": {
            "implementation": "predict_support_gated_selected_backbone_rbf",
            "target_argument_accepted_by_predictor": False,
            "uncertainty_argument_accepted_by_predictor": False,
            "held_rbf_config_required": True,
            "primary_trajectory_scored_without_recomputation": True,
        },
        "center_ids": centers.tolist(),
        "update_frames": list(UPDATE_FRAMES),
        "scored_frames": list(scored),
        "rbf_config": diagnostic["rbf_config"],
        "support_gate_contract": {
            "minimum_current_observed_centers": MINIMUM_SELECTOR_SUPPORT,
            "sufficient_support_action": (
                "select current physical/persistence backbone by unordered "
                "observation Chamfer and apply full-blend Euclidean RBF"
            ),
            "insufficient_support_default": "persistence",
            "insufficient_support_fallback": (
                "bit-exact persistence for the complete forecast interval"
            ),
            "tie_break": "physical_prior",
            "covariance_required": False,
        },
        "observed_backbone_selector": {
            "metric": "current observed-centre symmetric set Chamfer",
            "tie_break": "physical_prior",
            "minimum_reliable_support": MINIMUM_SELECTOR_SUPPORT,
            "insufficient_support_default": "persistence",
            "insufficient_support_rule_status": (
                "frozen on the open development panel before held-target use"
            ),
            "selected_by_update": selected_by_update,
            "physical_prior_count": int(
                sum(value == "physical_prior" for value in selected_by_update)
            ),
            "persistence_count": int(
                sum(value == "persistence" for value in selected_by_update)
            ),
            "insufficient_support_count": int(
                sum(
                    not record["selector_support_sufficient"]
                    for record in diagnostic["updates"]
                )
            ),
        },
        "updates": update_records,
        "scores": scores,
    }
    return report, trajectories


@dataclass(frozen=True)
class _VerifiedMeasurement:
    case_dir: Path
    measurement_dir: Path
    seal: Mapping[str, Any]
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    prediction_archive: Path
    physical_prior: np.ndarray
    persistence: np.ndarray
    selected_raw: np.ndarray
    prediction: np.ndarray
    prediction_diagnostic: Mapping[str, Any]
    prediction_seal_sha256: str
    measurement_manifest_sha256: str
    measurement_archive_sha256: str
    prediction_archive_sha256: str


def _load_verified_measurement(
    panel_case_dir: str | Path,
    measurement_dir: str | Path,
) -> _VerifiedMeasurement:
    case_dir = Path(panel_case_dir).resolve()
    measurement = Path(measurement_dir).resolve()
    if case_dir.name not in expected_open_case_names():
        raise ValueError("case is outside the explicit outcome-open panel")
    seal_path = case_dir / "prediction_seal.json"
    seal_payload = seal_path.read_bytes()
    prediction_seal_sha256 = _sha256_bytes(seal_payload)
    seal = json.loads(seal_payload)
    _validate_prediction_seal(seal)
    if seal.get("schema_version") != 1:
        raise ValueError("prediction seal schema changed")
    if seal.get("result_sha256") != primary_artifact_sha256(seal):
        raise ValueError("prediction seal content checksum changed")
    if (
        seal.get("object_id") is None
        or type(seal.get("episode_id")) is not int
        or case_dir.name != f"{seal['object_id']}-ep{int(seal['episode_id']):04d}"
    ):
        raise ValueError("prediction seal case identity changed")
    archive_record = seal.get("prediction_archive")
    if not isinstance(archive_record, dict) or not _is_sha256(
        archive_record.get("file_sha256")
    ):
        raise ValueError("prediction seal lacks a required archive file checksum")
    array_hashes = archive_record.get("array_sha256")
    if (
        not isinstance(array_hashes, dict)
        or not array_hashes
        or not all(
            isinstance(name, str) and _is_sha256(digest)
            for name, digest in array_hashes.items()
        )
    ):
        raise ValueError("prediction seal lacks valid per-array checksums")
    manifest, arrays = _load_measurement_artifact(case_dir, measurement, seal)
    measurement_manifest_sha256 = _sha256(measurement / MANIFEST_FILENAME)
    measurement_archive_sha256 = _sha256(measurement / MEASUREMENT_FILENAME)
    required = {"center_ids", "measurement_m", "measurement_validity", "update_frames"}
    if not required.issubset(arrays):
        raise ValueError("measurement archive is missing primary evaluator arrays")
    centers = np.asarray(arrays["center_ids"])
    measurement_m = np.asarray(arrays["measurement_m"])
    measurement_validity = np.asarray(arrays["measurement_validity"])
    update_frames = np.asarray(arrays["update_frames"])
    if centers.ndim != 1 or centers.dtype.kind not in "iu":
        raise ValueError("measurement center IDs must be an integer vector")
    if len(centers) != len(np.unique(centers)):
        raise ValueError("measurement center IDs must be unique")
    if measurement_m.ndim != 3 or measurement_m.shape[2] != 3:
        raise ValueError("measurement trajectory must have shape (T, N, 3)")
    if measurement_validity.shape != measurement_m.shape[:2]:
        raise ValueError("measurement validity shape differs from trajectory")
    if (
        update_frames.ndim != 1
        or update_frames.dtype.kind not in "iu"
        or update_frames.tolist() != list(UPDATE_FRAMES)
    ):
        raise ValueError("measurement update frames changed")
    if len(measurement_m) <= UPDATE_FRAMES[-1]:
        raise ValueError("measurement trajectory does not cover all updates")
    if np.any(centers < 0) or np.any(centers >= measurement_m.shape[1]):
        raise ValueError("measurement center ID exceeds the trajectory")
    manifest_centers = manifest.get("plan", {}).get("center_ids")
    if manifest_centers != centers.tolist():
        raise ValueError("measurement manifest and archive center IDs differ")
    prediction_archive = _resolve_prediction_archive(case_dir, seal)
    with np.load(prediction_archive, allow_pickle=False) as stored:
        if not {"prediction_m", "persistence_m"}.issubset(stored.files):
            raise ValueError("sealed prediction archive lacks required trajectories")
        if not set(array_hashes).issubset(stored.files):
            raise ValueError("sealed prediction archive lacks checksum-bound arrays")
        for name, expected_sha256 in array_hashes.items():
            if _sha256_array(np.asarray(stored[name])) != expected_sha256:
                raise ValueError(f"sealed prediction array checksum changed: {name}")
        physical_prior = np.asarray(stored["prediction_m"]).copy()
        persistence = np.asarray(stored["persistence_m"]).copy()
    prediction_archive_sha256 = _sha256(prediction_archive)
    if prediction_archive_sha256 != archive_record["file_sha256"]:
        raise ValueError("sealed prediction archive checksum changed after loading")
    prediction, selected_raw, prediction_diagnostic = (
        predict_support_gated_selected_backbone_rbf(
            physical_prior,
            persistence,
            measurement_m,
            measurement_validity,
            center_ids=centers,
            rbf_config=HELD_RBF_CONFIG,
        )
    )
    return _VerifiedMeasurement(
        case_dir=case_dir,
        measurement_dir=measurement,
        seal=seal,
        manifest=manifest,
        arrays=arrays,
        prediction_archive=prediction_archive,
        physical_prior=physical_prior,
        persistence=persistence,
        selected_raw=selected_raw,
        prediction=prediction,
        prediction_diagnostic=prediction_diagnostic,
        prediction_seal_sha256=prediction_seal_sha256,
        measurement_manifest_sha256=measurement_manifest_sha256,
        measurement_archive_sha256=measurement_archive_sha256,
        prediction_archive_sha256=prediction_archive_sha256,
    )


def _evaluate_verified_measurement(
    verified: _VerifiedMeasurement,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _recheck_verified_inputs(verified, boundary="target open")
    open_seal, prior, persistence, target, visibility, validity = (
        _load_open_case_for_evaluation(verified.case_dir)
    )
    if open_seal != verified.seal:
        raise ValueError("prediction seal changed while opening the outcome")
    if not _arrays_are_bit_exact(prior, verified.physical_prior) or not (
        _arrays_are_bit_exact(persistence, verified.persistence)
    ):
        raise ValueError("sealed prediction changed between prediction and scoring")
    centers = np.asarray(verified.arrays["center_ids"], dtype=np.int64)
    scored_frames = _post_update_scored_frames(len(target))
    algorithm_report, trajectories = _score_primary_outputs(
        verified.physical_prior,
        verified.persistence,
        verified.selected_raw,
        verified.prediction,
        verified.prediction_diagnostic,
        target,
        visibility,
        validity,
        center_ids=centers,
        scored_frames=scored_frames,
    )
    report = {
        **algorithm_report,
        "case": verified.case_dir.name,
        "object_id": str(verified.seal["object_id"]),
        "episode_id": int(verified.seal["episode_id"]),
        "prediction_seal_sha256": verified.prediction_seal_sha256,
        "measurement_manifest_sha256": verified.measurement_manifest_sha256,
        "measurement_archive_sha256": verified.measurement_archive_sha256,
        "measurement_result_sha256": verified.manifest["result_sha256"],
        "prediction_archive_sha256": verified.prediction_archive_sha256,
        "information_boundary": {
            "measurement_verified_before_target_open": True,
            "primary_prediction_completed_before_target_open": True,
            "measurement_builder_target_read": False,
            "uncertainty_sidecar_read": False,
            "target_visible_covariance_calibration_performed": False,
            "target_role": "scoring only",
        },
        "claim_boundary": (
            "outcome-open development evaluation on reconstructed proxy targets; "
            "not official Deform360 or held-target SOTA evidence"
        ),
    }
    return _sign_artifact(report), trajectories


def _recheck_verified_inputs(
    verified: _VerifiedMeasurement,
    *,
    boundary: str,
) -> None:
    checks = (
        (
            verified.case_dir / "prediction_seal.json",
            verified.prediction_seal_sha256,
            "prediction seal",
        ),
        (
            verified.measurement_dir / MANIFEST_FILENAME,
            verified.measurement_manifest_sha256,
            "measurement manifest",
        ),
        (
            verified.measurement_dir / MEASUREMENT_FILENAME,
            verified.measurement_archive_sha256,
            "measurement archive",
        ),
        (
            verified.prediction_archive,
            verified.prediction_archive_sha256,
            "prediction archive",
        ),
    )
    for path, expected_sha256, label in checks:
        if _sha256(path) != expected_sha256:
            raise ValueError(f"{label} changed before {boundary}")


def evaluate_primary_case(
    panel_case_dir: str | Path,
    measurement_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one hash-verified measurement without uncertainty sidecars."""

    return _evaluate_verified_measurement(
        _load_verified_measurement(panel_case_dir, measurement_dir)
    )


def _write_case_artifacts(
    output_dir: Path,
    case: str,
    report: Mapping[str, Any],
    trajectories: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, str]]:
    archive_path = output_dir / f"{case}.npz"
    report_path = output_dir / f"{case}.json"
    np.savez_compressed(archive_path, **trajectories)
    archive_sha256 = _sha256(archive_path)
    emitted_report = dict(report)
    emitted_report.pop("result_sha256", None)
    emitted_report["trajectory_archive_sha256"] = archive_sha256
    emitted_report = _sign_artifact(emitted_report)
    report_path.write_text(
        json.dumps(emitted_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return emitted_report, {
        "case": case,
        "report_sha256": _sha256(report_path),
        "report_result_sha256": str(emitted_report["result_sha256"]),
        "archive_sha256": archive_sha256,
    }


def _cohort_summary(
    reports: Sequence[Mapping[str, Any]],
    groups: Mapping[str, str],
    artifacts: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    aggregate = {
        arm: {
            metric: float(
                np.mean([report["scores"][arm][metric] for report in reports])
            )
            for metric in PRIMARY_METRICS
        }
        for arm in PRIMARY_ARMS
    }
    comparisons: dict[str, Any] = {}
    for baseline in ("physical_prior", "persistence", SELECTED_BACKBONE_ARM):
        for metric in PRIMARY_METRICS:
            differences = {
                str(report["case"]): float(
                    report["scores"][PRIMARY_ARM][metric]
                    - report["scores"][baseline][metric]
                )
                for report in reports
            }
            comparison = _physical_object_cluster_bootstrap(differences, groups)
            comparison["episode_wins"] = int(
                np.sum(np.asarray(list(differences.values())) < 0.0)
            )
            comparison["per_object_mean_difference_m"] = {
                object_id: float(
                    np.mean(
                        [
                            differences[case]
                            for case, group in groups.items()
                            if group == object_id
                        ]
                    )
                )
                for object_id in sorted(set(groups.values()))
            }
            comparison["relative_change"] = (
                aggregate[PRIMARY_ARM][metric] / aggregate[baseline][metric] - 1.0
            )
            comparisons[f"{PRIMARY_ARM}:vs:{baseline}:{metric}"] = comparison
    summary = {
        "schema_version": 1,
        "protocol_id": PRIMARY_EVALUATION_PROTOCOL_ID,
        "episode_count": len(reports),
        "physical_object_count": len(set(groups.values())),
        "primary_arm": PRIMARY_ARM,
        "comparators": ["physical_prior", "persistence", SELECTED_BACKBONE_ARM],
        "aggregate": aggregate,
        "comparisons": comparisons,
        "observed_backbone_selector_counts": {
            "physical_prior": int(
                sum(
                    report["observed_backbone_selector"]["physical_prior_count"]
                    for report in reports
                )
            ),
            "persistence": int(
                sum(
                    report["observed_backbone_selector"]["persistence_count"]
                    for report in reports
                )
            ),
            "insufficient_support": int(
                sum(
                    report["observed_backbone_selector"]["insufficient_support_count"]
                    for report in reports
                )
            ),
        },
        "information_boundary": {
            "all_measurements_verified_before_any_target_open": True,
            "all_primary_predictions_completed_before_any_target_open": True,
            "case_inputs_rechecked_before_target_open_and_artifact_emission": True,
            "uncertainty_sidecars_required": False,
            "target_visible_covariance_calibration_performed": False,
        },
        "artifacts": list(artifacts),
        "claim_boundary": (
            "outcome-open fixed-method development comparison on reconstructed "
            "proxy targets; not official Deform360 or held-target SOTA evidence"
        ),
    }
    return _sign_artifact(summary)


def evaluate_primary_cohort(
    panel_root: str | Path,
    measurement_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate all 27 cases after first verifying every measurement artifact."""

    panel = Path(panel_root).resolve()
    measurements = Path(measurement_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"primary output already exists: {output}")
    cases = expected_open_case_names()
    missing = [
        case
        for case in cases
        if not (measurements / case / MANIFEST_FILENAME).is_file()
        or not (measurements / case / MEASUREMENT_FILENAME).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing raw-camera measurements: {missing}")

    # This entire pass is target-free.  A corrupt late-case artifact therefore
    # prevents any target from being opened.
    verified = [
        _load_verified_measurement(panel / case, measurements / case) for case in cases
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        reports: list[dict[str, Any]] = []
        groups: dict[str, str] = {}
        artifacts: list[dict[str, str]] = []
        for case, measurement in zip(cases, verified, strict=True):
            report, trajectories = _evaluate_verified_measurement(measurement)
            _recheck_verified_inputs(measurement, boundary="artifact emission")
            emitted_report, artifact = _write_case_artifacts(
                staging,
                case,
                report,
                trajectories,
            )
            reports.append(emitted_report)
            artifacts.append(artifact)
            groups[case] = str(emitted_report["object_id"])
        summary = _cohort_summary(reports, groups, artifacts)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(
                f"primary output appeared during evaluation: {output}"
            )
        staging.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@dataclass(frozen=True)
class _GatedReference:
    root: Path
    summary: Mapping[str, Any]
    summary_file_sha256: str
    summary_result_sha256: str
    report_sha256_by_case: Mapping[str, str]
    archive_sha256_by_case: Mapping[str, str]


def _validate_gated_reference(
    reference_root: Path,
    cases: Sequence[str],
    *,
    expected_summary_file_sha256: str,
    expected_summary_result_sha256: str,
) -> _GatedReference:
    if not _is_sha256(expected_summary_file_sha256) or not _is_sha256(
        expected_summary_result_sha256
    ):
        raise ValueError("expected gated summary bindings must be SHA-256 digests")
    summary_path = reference_root / "summary.json"
    summary_payload = _read_bound_bytes(
        summary_path,
        expected_summary_file_sha256,
        label="externally bound gated reference summary",
    )
    summary = json.loads(summary_payload)
    _validate_self_hash(summary, label="gated reference summary")
    if summary.get("result_sha256") != expected_summary_result_sha256:
        raise ValueError("gated reference summary result hash differs from binding")
    if summary.get("protocol_id") != GATED_EVALUATION_PROTOCOL_ID:
        raise ValueError("reference is not the covariance-gated evaluator")
    if summary.get("episode_count") != len(cases):
        raise ValueError("gated reference episode count changed")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("gated reference artifact inventory is missing")
    by_case = {
        str(record.get("case")): record
        for record in artifacts
        if isinstance(record, dict)
    }
    if set(by_case) != set(cases) or len(artifacts) != len(cases):
        raise ValueError("gated reference case inventory changed")
    report_hashes: dict[str, str] = {}
    archive_hashes: dict[str, str] = {}
    for case in cases:
        record = by_case[case]
        report_path = reference_root / f"{case}.json"
        archive_path = reference_root / f"{case}.npz"
        report_sha256 = record.get("report_sha256")
        archive_sha256 = record.get("archive_sha256")
        if not _is_sha256(report_sha256) or not report_path.is_file():
            raise ValueError(f"gated reference report binding is invalid for {case}")
        if not _is_sha256(archive_sha256) or not archive_path.is_file():
            raise ValueError(f"gated reference archive binding is invalid for {case}")
        report_hashes[case] = report_sha256
        archive_hashes[case] = archive_sha256
    return _GatedReference(
        root=reference_root,
        summary=summary,
        summary_file_sha256=expected_summary_file_sha256,
        summary_result_sha256=expected_summary_result_sha256,
        report_sha256_by_case=report_hashes,
        archive_sha256_by_case=archive_hashes,
    )


def _read_gated_reference_case(
    binding: _GatedReference,
    case: str,
) -> tuple[dict[str, Any], bytes]:
    report_payload = _read_bound_bytes(
        binding.root / f"{case}.json",
        binding.report_sha256_by_case[case],
        label=f"gated reference report for {case}",
    )
    archive_payload = _read_bound_bytes(
        binding.root / f"{case}.npz",
        binding.archive_sha256_by_case[case],
        label=f"gated reference archive for {case}",
    )
    report = json.loads(report_payload)
    if not isinstance(report, dict):
        raise ValueError(f"gated reference report is not an object for {case}")
    return report, archive_payload


def _compare_primary_case_to_gated(
    report: Mapping[str, Any],
    trajectories: Mapping[str, np.ndarray],
    reference_report: Mapping[str, Any],
    reference_archive_payload: bytes,
) -> dict[str, Any]:
    case = str(report["case"])
    if reference_report.get("case") != case:
        raise ValueError(f"gated reference case identity differs for {case}")
    trajectory_checks: dict[str, bool] = {}
    reference_trajectory_hashes: dict[str, str] = {}
    with np.load(io.BytesIO(reference_archive_payload), allow_pickle=False) as stored:
        for arm in PRIMARY_ARMS:
            if arm not in stored.files:
                raise ValueError(f"gated reference lacks {arm} for {case}")
            reference = np.asarray(stored[arm])
            trajectory_checks[arm] = _arrays_are_bit_exact(trajectories[arm], reference)
            reference_trajectory_hashes[arm] = _array_sha256(reference)

    metadata_checks: dict[str, bool] = {
        field: report[field] == reference_report.get(field)
        for field in (
            "center_ids",
            "update_frames",
            "scored_frames",
            "rbf_config",
        )
    }
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
    primary_selector = report["observed_backbone_selector"]
    reference_selector = reference_report.get("observed_backbone_selector", {})
    metadata_checks["observed_backbone_selector_normalized"] = all(
        primary_selector.get(field) == reference_selector.get(field)
        for field in selector_fields
    )
    score_checks = {
        arm: _values_close(
            report["scores"][arm],
            reference_report.get("scores", {}).get(arm),
            absolute_tolerance=1.0e-12,
        )
        for arm in PRIMARY_ARMS
    }
    primary_updates = report["updates"]
    reference_updates = reference_report.get("updates")
    if not isinstance(reference_updates, list) or len(primary_updates) != len(
        reference_updates
    ):
        raise ValueError(f"gated reference updates differ for {case}")
    update_checks: list[dict[str, Any]] = []
    update_fields = (
        "frame",
        "stop_frame_exclusive",
        "available_center_count",
        "selected_backbone",
        "selector_support_sufficient",
        "current_observation_chamfer_m",
    )
    for primary_update, reference_update in zip(
        primary_updates, reference_updates, strict=True
    ):
        reference_gate = reference_update.get("gates", {}).get("ungated", {})
        primary_gate = primary_update["support_gate"]
        sufficient = bool(primary_update["selector_support_sufficient"])
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
                if primary_update["available_center_count"] == 0
                else "insufficient_selector_support"
            )
        )
        update_checks.append(
            {
                "frame": primary_update["frame"],
                "selection_metadata_bit_exact": all(
                    primary_update[field] == reference_update.get(field)
                    for field in update_fields
                ),
                "canonical_support_decision": primary_update["selector_decision"],
                "legacy_reference_selector_decision": reference_update.get(
                    "selector_decision"
                ),
                "legacy_reference_gate_decision": reference_gate.get("decision"),
                "support_semantics_equivalent": (
                    primary_update["selector_decision"] == canonical_decision
                    and primary_gate.get("decision") == canonical_decision
                    and reference_update.get("selector_decision")
                    == legacy_selector_decision
                    and reference_gate.get("decision") == legacy_gate_decision
                    and primary_gate.get("accepted") == reference_gate.get("accepted")
                    and primary_gate.get("selected_backbone")
                    == reference_gate.get("selected_backbone")
                    and primary_gate.get("fallback_backbone")
                    == reference_gate.get("fallback_backbone")
                    and primary_gate.get("rbf_correction_applied")
                    == reference_gate.get("rbf_correction_applied")
                ),
            }
        )
    all_exact_metadata = all(metadata_checks.values()) and all(
        record["selection_metadata_bit_exact"] for record in update_checks
    )
    all_support_semantics_equivalent = all(
        record["support_semantics_equivalent"] for record in update_checks
    )
    parity_passed = (
        all(trajectory_checks.values())
        and all_exact_metadata
        and all_support_semantics_equivalent
        and all(score_checks.values())
    )
    return {
        "case": case,
        "all_primary_arrays_byte_exact": all(trajectory_checks.values()),
        "all_exact_metadata_equal": all_exact_metadata,
        "all_support_semantics_equivalent": all_support_semantics_equivalent,
        "parity_passed": parity_passed,
        "trajectory_bit_exact": trajectory_checks,
        "metadata_exact": metadata_checks,
        "score_within_absolute_tolerance": score_checks,
        "score_absolute_tolerance": 1.0e-12,
        "updates": update_checks,
        "reference_trajectory_sha256": reference_trajectory_hashes,
        "primary_trajectory_sha256": {
            arm: _array_sha256(trajectories[arm]) for arm in PRIMARY_ARMS
        },
    }


def compare_primary_to_gated_cohort(
    panel_root: str | Path,
    measurement_root: str | Path,
    gated_reference_root: str | Path,
    *,
    expected_gated_summary_file_sha256: str,
    expected_gated_summary_result_sha256: str,
) -> dict[str, Any]:
    """Read-only bit-exact parity audit against all existing 8-view outputs."""

    panel = Path(panel_root).resolve()
    measurements = Path(measurement_root).resolve()
    reference = Path(gated_reference_root).resolve()
    cases = expected_open_case_names()
    if len(cases) != 27:
        raise AssertionError("primary parity audit requires the complete open27")

    # Every measurement is verified and every target-free primary prediction is
    # complete before any gated outcome-derived artifact or scoring target is
    # opened.
    verified = [
        _load_verified_measurement(panel / case, measurements / case) for case in cases
    ]
    reference_binding = _validate_gated_reference(
        reference,
        cases,
        expected_summary_file_sha256=expected_gated_summary_file_sha256,
        expected_summary_result_sha256=expected_gated_summary_result_sha256,
    )
    per_case: list[dict[str, Any]] = []
    for case, measurement in zip(cases, verified, strict=True):
        report, trajectories = _evaluate_verified_measurement(measurement)
        reference_report, reference_archive_payload = _read_gated_reference_case(
            reference_binding,
            case,
        )
        per_case.append(
            _compare_primary_case_to_gated(
                report,
                trajectories,
                reference_report,
                reference_archive_payload,
            )
        )
    for measurement in verified:
        _recheck_verified_inputs(measurement, boundary="parity completion")
    _read_bound_bytes(
        reference / "summary.json",
        reference_binding.summary_file_sha256,
        label="gated reference summary after parity",
    )
    all_cases_parity_passed = all(record["parity_passed"] for record in per_case)
    result = {
        "schema_version": 1,
        "protocol_id": PRIMARY_PARITY_PROTOCOL_ID,
        "reference_protocol_id": reference_binding.summary["protocol_id"],
        "reference_summary_binding": {
            "file_sha256": reference_binding.summary_file_sha256,
            "result_sha256": reference_binding.summary_result_sha256,
        },
        "episode_count": len(per_case),
        "all_27_cases_primary_arrays_byte_exact": all(
            record["all_primary_arrays_byte_exact"] for record in per_case
        ),
        "all_27_cases_parity_passed": all_cases_parity_passed,
        "parity_passed": all_cases_parity_passed,
        "cases": per_case,
        "read_only_contract": {
            "input_artifacts_written": False,
            "output_artifacts_written": False,
            "reference_hashes_verified_before_comparison": True,
            "reference_summary_externally_bound": True,
            "reference_case_files_loaded_from_hash_verified_bytes": True,
            "all_measurements_verified_before_any_target_open": True,
            "all_primary_predictions_completed_before_any_target_open": True,
            "primary_input_hashes_rechecked_at_parity_completion": True,
        },
    }
    return _sign_artifact(result)


__all__ = [
    "PRIMARY_ARM",
    "PRIMARY_ARMS",
    "PRIMARY_EVALUATION_PROTOCOL_ID",
    "PRIMARY_PARITY_PROTOCOL_ID",
    "compare_primary_to_gated_cohort",
    "evaluate_primary_arrays",
    "evaluate_primary_case",
    "evaluate_primary_cohort",
    "primary_artifact_sha256",
]
