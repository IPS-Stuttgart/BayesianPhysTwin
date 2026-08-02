from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_pairwise_bias_aware_transfer as transfer

CASE = "test-object-ep0003"
OBJECT_ID = "test-object"
EPISODE_ID = 3
FRAME_COUNT = 76
NODE_COUNT = 12
CENTER_IDS = np.arange(9, dtype=np.int64)


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    roots = {
        "source_root": tmp_path / "source",
        "measurement_root": tmp_path / "measurement",
        "uncertainty_root": tmp_path / "uncertainty",
        "selected_baseline_root": tmp_path / "baseline",
    }
    source_case = roots["source_root"] / CASE
    measurement_case = roots["measurement_root"] / CASE
    uncertainty_case = roots["uncertainty_root"] / CASE
    source_case.mkdir(parents=True)
    measurement_case.mkdir(parents=True)
    uncertainty_case.mkdir(parents=True)
    roots["selected_baseline_root"].mkdir(parents=True)

    frame_zero = np.zeros((NODE_COUNT, 3), dtype=np.float32)
    trajectory = np.repeat(frame_zero[None], FRAME_COUNT, axis=0)
    prediction_path = source_case / "physical_prediction.npz"
    np.savez_compressed(
        prediction_path,
        driven_readout_m=trajectory,
        zero_action_readout_m=trajectory,
        action_support=np.zeros(NODE_COUNT, dtype=np.float32),
        frame_zero_points_m=frame_zero,
    )
    target_path = source_case / "target_data.pkl"
    target_path.write_bytes(b"opaque target bytes, deliberately not a pickle")
    seal_path = source_case / "prediction_seal.json"
    seal = {
        "object_id": OBJECT_ID,
        "episode_id": EPISODE_ID,
        "episode_key": f"{OBJECT_ID}/{EPISODE_ID}",
        "prediction_archive": {
            "path": str(prediction_path.resolve()),
            "file_sha256": transfer._sha256(prediction_path),
        },
    }
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    outcome = {
        "artifact_kind": "Deform360IndependentSourceOutcome",
        "object_id": OBJECT_ID,
        "episode_id": EPISODE_ID,
        "episode_key": f"{OBJECT_ID}/{EPISODE_ID}",
        "input_sha256": {"prediction_seal": transfer._sha256(seal_path)},
        "output_sha256": {"target_data": transfer._sha256(target_path)},
    }
    (source_case / "outcome.json").write_text(
        json.dumps(outcome),
        encoding="utf-8",
    )

    measurement = np.full_like(trajectory, np.nan)
    visibility = np.zeros((FRAME_COUNT, NODE_COUNT), dtype=bool)
    validity = np.zeros_like(visibility)
    for frame in (19, 38, 57):
        measurement[frame, CENTER_IDS] = 0.0
        visibility[frame, CENTER_IDS] = True
        validity[frame, CENTER_IDS] = True
    np.savez_compressed(
        measurement_case / "measurement.npz",
        measurement_m=measurement,
        measurement_visibility=visibility,
        measurement_validity=validity,
        center_ids=CENTER_IDS,
        selected_cameras=np.asarray(["cam0", "cam1", "cam2"]),
        update_frames=np.asarray([19, 38, 57], dtype=np.int64),
        triangulation_inlier_view_count=np.full((3, 9), 3, dtype=np.int64),
        triangulation_median_reprojection_px=np.ones((3, 9), dtype=np.float32),
    )
    covariance = np.zeros((FRAME_COUNT, NODE_COUNT, 3, 3), dtype=np.float32)
    covariance_valid = validity.copy()
    for frame in (19, 38, 57):
        covariance[frame, CENTER_IDS] = np.eye(3, dtype=np.float32) * 1e-5
    np.savez_compressed(
        uncertainty_case / "measurement_cycle_uncertainty.npz",
        measurement_covariance_m2=covariance,
        measurement_covariance_valid=covariance_valid,
    )
    np.savez_compressed(
        roots["selected_baseline_root"] / f"{CASE}.npz",
        selected_raw_backbone=trajectory,
    )
    return roots


def _one_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        transfer,
        "_expected_case_records",
        lambda: ((CASE, OBJECT_ID, EPISODE_ID),),
    )


def test_stage_and_validate_bundle_without_deserializing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _one_case(monkeypatch)
    roots = _write_fixture(tmp_path)
    bundle = tmp_path / "bundle"

    staged = transfer.stage_open27_transfer_bundle(
        **roots,
        destination=bundle,
    )

    assert staged["case_count"] == 1
    assert staged["file_count"] == 7
    assert (
        bundle / "source" / CASE / "target_data.pkl"
    ).read_bytes() == b"opaque target bytes, deliberately not a pickle"
    for path in roots.values():
        shutil.rmtree(path)
    repeated = transfer.validate_open27_transfer_bundle(bundle)
    assert repeated["manifest_sha256"] == staged["manifest_sha256"]


def test_bundle_validation_rejects_changed_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _one_case(monkeypatch)
    roots = _write_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    transfer.stage_open27_transfer_bundle(**roots, destination=bundle)
    changed = bundle / "selected_baseline" / f"{CASE}.npz"
    with changed.open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="size changed"):
        transfer.validate_open27_transfer_bundle(bundle)


def test_manifest_rejects_root_escape_even_with_recomputed_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _one_case(monkeypatch)
    roots = _write_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    transfer.stage_open27_transfer_bundle(**roots, destination=bundle)
    manifest_path = bundle / transfer.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["files"]["measurement"]["relative_path"] = "../escape.npz"
    manifest["manifest_sha256"] = transfer._canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        transfer.validate_open27_transfer_bundle(bundle)


def test_fixed_source_panel_has_twenty_seven_cases() -> None:
    assert len(transfer._expected_case_records()) == 27
