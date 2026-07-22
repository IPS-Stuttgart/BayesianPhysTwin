"""Outcome-free artifacts for direct 2-D cross-view guarded updates."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
)
from .deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from .deform360_crossview_2d_guard import (
    DirectCrossView2DConfig,
    predict_direct_crossview_guarded_candidate_arrays,
)
from .deform360_crossview_guard import CrossViewGuardConfig
from .deform360_crossview_observation import (
    MANIFEST_FILENAME as SUPPLEMENT_MANIFEST_FILENAME,
    load_crossview_track_supplement,
)
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME as SOURCE_MANIFEST_FILENAME,
    _canonical_sha256 as source_canonical_sha256,
)


PROTOCOL_ID = "deform360-direct-2d-crossview-guard-v1-postopen-development"
ARTIFACT_KIND = "Deform360Direct2DCrossViewGuardPrediction"
ARCHIVE_FILENAME = "direct_2d_crossview_guarded_prediction.npz"
REPORT_FILENAME = "direct_2d_crossview_guarded_prediction.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    unsigned = dict(manifest)
    claimed = unsigned.pop("result_sha256", None)
    _require(
        isinstance(claimed, str) and claimed == source_canonical_sha256(unsigned),
        "source measurement manifest checksum changed",
    )


def load_direct_crossview_guard_prediction(
    artifact_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and checksum one direct camera-space guarded prediction."""

    root = Path(artifact_dir).resolve()
    report_path = root / REPORT_FILENAME
    archive_path = root / ARCHIVE_FILENAME
    report = _load_json(report_path)
    _require(
        report.get("artifact_kind") == ARTIFACT_KIND
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("result_sha256")
        == canonical_sha256(report, digest_key="result_sha256"),
        "direct cross-view prediction report changed",
    )
    _require(
        report.get("output", {}).get("archive_file_sha256")
        == file_sha256(archive_path),
        "direct cross-view prediction archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    required = {
        "baseline_m",
        "direct_2d_crossview_guarded_m",
        "center_ids",
        "update_frames",
    }
    _require(required.issubset(arrays), "direct prediction archive is incomplete")
    _require(
        arrays["baseline_m"].shape
        == arrays["direct_2d_crossview_guarded_m"].shape,
        "direct guarded prediction shape changed",
    )
    return report, arrays


def build_direct_crossview_guard_prediction(
    measurement_dir: str | Path,
    supplement_dir: str | Path,
    baseline_archive: str | Path,
    baseline_key: str,
    output_dir: str | Path,
    *,
    development_config: Deform360BiasAwareDevelopmentConfig | None = None,
    fit_config: DirectCrossView2DConfig | None = None,
    guard_config: CrossViewGuardConfig | None = None,
) -> dict[str, Any]:
    """Seal a direct 2-D guarded trajectory without accepting an outcome."""

    measurement_root = Path(measurement_dir).resolve()
    supplement_root = Path(supplement_dir).resolve()
    baseline_path = Path(baseline_archive).resolve()
    output = Path(output_dir).resolve()
    _require(not output.exists(), f"direct prediction output exists: {output}")
    source_manifest_path = measurement_root / SOURCE_MANIFEST_FILENAME
    source_manifest = _load_json(source_manifest_path)
    _validate_source_manifest(source_manifest)
    supplement_manifest, supplement = load_crossview_track_supplement(
        supplement_root
    )
    _require(
        supplement_manifest["source_measurement"]["manifest_result_sha256"]
        == source_manifest["result_sha256"],
        "supplement and source measurement differ",
    )
    prediction_record = source_manifest.get("inputs", {}).get(
        "prediction_archive", {}
    )
    prediction_path = Path(str(prediction_record.get("path"))).resolve()
    _require(
        prediction_record.get("sha256") == file_sha256(prediction_path),
        "source physical prediction checksum changed",
    )
    with np.load(prediction_path, allow_pickle=False) as stored:
        required = {
            "driven_readout_m",
            "zero_action_readout_m",
            "frame_zero_points_m",
            "action_support",
        }
        _require(required.issubset(stored.files), "physical prediction is incomplete")
        physical_response = np.asarray(stored["driven_readout_m"]) - np.asarray(
            stored["zero_action_readout_m"]
        )
        frame_zero = np.asarray(stored["frame_zero_points_m"])
        action_support = np.asarray(stored["action_support"])
    with np.load(baseline_path, allow_pickle=False) as stored:
        _require(baseline_key in stored.files, "selected baseline key is absent")
        baseline = np.asarray(stored[baseline_key])
    _require(
        baseline.shape == physical_response.shape,
        "selected baseline and physical response shapes differ",
    )
    development = development_config or Deform360BiasAwareDevelopmentConfig()
    fit = fit_config or DirectCrossView2DConfig()
    guard = guard_config or CrossViewGuardConfig()
    method_report, guarded = predict_direct_crossview_guarded_candidate_arrays(
        baseline,
        physical_response,
        frame_zero,
        action_support,
        supplement,
        development_config=development,
        fit_config=fit,
        guard_config=guard,
    )
    output.mkdir(parents=True)
    archive_path = output / ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        baseline_m=baseline,
        direct_2d_crossview_guarded_m=guarded,
        center_ids=np.asarray(supplement["center_ids"], dtype=np.int64),
        update_frames=np.asarray(supplement["update_frames"], dtype=np.int64),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case": source_manifest["case"],
        "object_id": source_manifest["object_id"],
        "episode_id": source_manifest["episode_id"],
        "episode_key": source_manifest["episode_key"],
        "baseline_key": str(baseline_key),
        "development_config": asdict(development),
        "direct_2d_fit_config": asdict(fit),
        "crossview_guard_config": asdict(guard),
        "method": method_report,
        "inputs_sha256": {
            "source_measurement_manifest": file_sha256(source_manifest_path),
            "source_measurement_result": source_manifest["result_sha256"],
            "crossview_supplement_manifest": file_sha256(
                supplement_root / SUPPLEMENT_MANIFEST_FILENAME
            ),
            "crossview_supplement_result": supplement_manifest["result_sha256"],
            "physical_prediction_archive": file_sha256(prediction_path),
            "selected_baseline_archive": file_sha256(baseline_path),
        },
        "output": {
            "archive": str(archive_path),
            "archive_file_sha256": file_sha256(archive_path),
            "accepted_update_count": int(method_report["accepted_count"]),
            "exact_fallback_update_count": int(
                method_report["exact_fallback_count"]
            ),
            "trajectory_bit_exact_baseline": bool(
                np.array_equal(guarded, baseline)
            ),
        },
        "information_boundary": {
            "target_argument_accepted": False,
            "outcome_argument_accepted": False,
            "future_observation_read": False,
            "prediction_hashed_before_outcome_join": True,
        },
        "claim_boundary": (
            "Post-open direct-camera method development; this cannot modify "
            "either frozen prospective result."
        ),
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    (output / REPORT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "ARCHIVE_FILENAME",
    "ARTIFACT_KIND",
    "PROTOCOL_ID",
    "REPORT_FILENAME",
    "build_direct_crossview_guard_prediction",
    "load_direct_crossview_guard_prediction",
]
