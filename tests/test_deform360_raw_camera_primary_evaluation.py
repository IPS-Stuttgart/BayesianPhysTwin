from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_raw_camera_primary_evaluation as primary

from bayesian_phystwin.deform360_raw_camera_gated_evaluation import (
    SELECTED_BACKBONE_ARM,
    evaluate_covariance_gated_arrays,
)
from bayesian_phystwin.deform360_raw_camera_primary_evaluation import (
    PRIMARY_ARM,
    PRIMARY_ARMS,
    _compare_primary_case_to_gated,
    _write_case_artifacts,
    evaluate_primary_arrays,
    primary_artifact_sha256,
)


def _arrays() -> tuple[np.ndarray, ...]:
    frame_count = 76
    point_count = 7
    centers = np.array([0, 1, 2], dtype=np.int64)
    frame_zero = np.stack(
        (
            np.linspace(0.0, 0.06, point_count),
            np.zeros(point_count),
            np.ones(point_count),
        ),
        axis=1,
    ).astype(np.float32)
    prior = np.repeat(frame_zero[None], frame_count, axis=0)
    persistence = np.repeat(frame_zero[None], frame_count, axis=0)
    prior[20:, :, 1] += 0.02
    prior[39:, :, 1] += 0.02
    target = prior.copy()
    measurement = np.full_like(prior, np.nan)
    measurement_validity = np.zeros((frame_count, point_count), dtype=bool)
    measurement[19, centers] = prior[19, centers] + np.array([0.003, 0.0, 0.0])
    measurement_validity[19, centers] = True
    measurement[38, centers] = persistence[38, centers] + np.array([0.002, 0.0, 0.0])
    measurement_validity[38, centers] = True
    # Frame 57 deliberately has zero support.  The exact persistence fallback
    # is part of both the primary algorithm and the parity comparison.
    visible = np.ones((frame_count, point_count), dtype=bool)
    valid = np.ones((frame_count, point_count), dtype=bool)
    return (
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        centers,
    )


def test_primary_arrays_are_bit_exact_to_existing_gated_ungated_arm() -> None:
    (
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        centers,
    ) = _arrays()
    covariance = np.full(prior.shape[:2] + (3, 3), np.nan)
    covariance_validity = np.zeros(prior.shape[:2], dtype=bool)

    primary_report, primary_trajectories = evaluate_primary_arrays(
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        center_ids=centers,
        scored_frames=(20, 39, 58),
    )
    gated_report, gated_trajectories = evaluate_covariance_gated_arrays(
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        covariance,
        covariance_validity,
        center_ids=centers,
        scored_frames=(20, 39, 58),
        gate_thresholds={"ungated": -np.inf},
    )

    for arm in PRIMARY_ARMS:
        assert primary_trajectories[arm].dtype == gated_trajectories[arm].dtype
        assert primary_trajectories[arm].tobytes() == gated_trajectories[arm].tobytes()
        assert primary_report["scores"][arm] == gated_report["scores"][arm]
    assert primary_report["rbf_config"] == gated_report["rbf_config"]
    for field, value in primary_report["observed_backbone_selector"].items():
        assert value == gated_report["observed_backbone_selector"][field]
    assert [record["selected_backbone"] for record in primary_report["updates"]] == [
        "physical_prior",
        "persistence",
        "persistence",
    ]
    assert primary_report["updates"][-1]["available_center_count"] == 0
    assert (
        primary_report["updates"][-1]["support_gate"]["decision"]
        == "insufficient_support_persistence"
    )
    assert "covariance" not in json.dumps(primary_report["updates"][-1]).lower()


def test_zero_support_is_bit_exact_persistence_for_complete_interval() -> None:
    (
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        centers,
    ) = _arrays()
    measurement_validity[:] = False

    report, trajectories = evaluate_primary_arrays(
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        center_ids=centers,
        scored_frames=(20, 39, 58),
    )

    scored = np.asarray(
        [*range(20, 38), *range(39, 57), *range(58, len(prior))],
        dtype=np.int64,
    )
    assert report["observed_backbone_selector"]["insufficient_support_count"] == 3
    np.testing.assert_array_equal(
        trajectories[SELECTED_BACKBONE_ARM][scored], persistence[scored]
    )
    np.testing.assert_array_equal(
        trajectories[PRIMARY_ARM][scored], persistence[scored]
    )


def test_measurement_failure_prevents_target_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = "fixture-ep0000"
    panel_case = tmp_path / case
    panel_case.mkdir()
    seal = primary._sign_artifact(
        {
            "schema_version": 1,
            "object_id": "fixture",
            "episode_id": 0,
            "prediction_archive": {
                "file_sha256": "0" * 64,
                "array_sha256": {"prediction_m": "1" * 64},
            },
        }
    )
    (panel_case / "prediction_seal.json").write_text(
        json.dumps(seal) + "\n",
        encoding="utf-8",
    )
    target_opened = False

    monkeypatch.setattr(primary, "expected_open_case_names", lambda: (case,))
    monkeypatch.setattr(primary, "_validate_prediction_seal", lambda _seal: None)

    def reject_measurement(*_args: object) -> None:
        raise ValueError("measurement archive checksum changed")

    def open_target(*_args: object) -> None:
        nonlocal target_opened
        target_opened = True
        raise AssertionError("target must not open")

    monkeypatch.setattr(primary, "_load_measurement_artifact", reject_measurement)
    monkeypatch.setattr(primary, "_load_open_case_for_evaluation", open_target)

    with pytest.raises(ValueError, match="measurement archive checksum changed"):
        primary.evaluate_primary_case(panel_case, tmp_path / "measurement")
    assert target_opened is False


def test_cohort_verifies_every_measurement_before_any_target_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = ("first-ep0000", "late-corrupt-ep0000")
    panel = tmp_path / "panel"
    measurements = tmp_path / "measurements"
    for case in cases:
        (panel / case).mkdir(parents=True)
        case_measurements = measurements / case
        case_measurements.mkdir(parents=True)
        (case_measurements / primary.MANIFEST_FILENAME).write_bytes(b"manifest")
        (case_measurements / primary.MEASUREMENT_FILENAME).write_bytes(b"archive")
    target_opened = False

    monkeypatch.setattr(primary, "expected_open_case_names", lambda: cases)

    def verify(panel_case: Path, _measurement_case: Path) -> object:
        if Path(panel_case).name == cases[-1]:
            raise ValueError("late measurement checksum changed")
        return object()

    def evaluate(_verified: object) -> None:
        nonlocal target_opened
        target_opened = True
        raise AssertionError("target must not open")

    monkeypatch.setattr(primary, "_load_verified_measurement", verify)
    monkeypatch.setattr(primary, "_evaluate_verified_measurement", evaluate)
    monkeypatch.setattr(primary, "_recheck_verified_inputs", lambda *_args, **_kw: None)

    with pytest.raises(ValueError, match="late measurement checksum changed"):
        primary.evaluate_primary_cohort(
            panel,
            measurements,
            tmp_path / "output",
        )
    assert target_opened is False
    assert not (tmp_path / "output").exists()


def test_late_evaluation_failure_removes_staging_and_never_publishes_final_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = ("first-ep0000", "late-failure-ep0000")
    panel = tmp_path / "panel"
    measurements = tmp_path / "measurements"
    for case in cases:
        (panel / case).mkdir(parents=True)
        case_measurements = measurements / case
        case_measurements.mkdir(parents=True)
        (case_measurements / primary.MANIFEST_FILENAME).write_bytes(b"manifest")
        (case_measurements / primary.MEASUREMENT_FILENAME).write_bytes(b"archive")
    output = tmp_path / "atomic-output"

    monkeypatch.setattr(primary, "expected_open_case_names", lambda: cases)
    monkeypatch.setattr(
        primary,
        "_load_verified_measurement",
        lambda panel_case, _measurement_case: Path(panel_case).name,
    )

    def evaluate(case: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        if case == cases[-1]:
            raise RuntimeError("late scoring failure")
        return (
            {
                "case": case,
                "object_id": "fixture-object",
                "protocol_id": primary.PRIMARY_EVALUATION_PROTOCOL_ID,
            },
            {PRIMARY_ARM: np.zeros((2, 2, 3), dtype=np.float32)},
        )

    monkeypatch.setattr(primary, "_evaluate_verified_measurement", evaluate)
    monkeypatch.setattr(primary, "_recheck_verified_inputs", lambda *_args, **_kw: None)

    with pytest.raises(RuntimeError, match="late scoring failure"):
        primary.evaluate_primary_cohort(panel, measurements, output)
    assert not output.exists()
    assert list(tmp_path.glob(".atomic-output.staging-*")) == []


def test_emitted_case_report_and_summary_hashes_are_self_verifying(
    tmp_path: Path,
) -> None:
    report = {
        "case": "fixture",
        "protocol_id": primary.PRIMARY_EVALUATION_PROTOCOL_ID,
        "result_sha256": "replaced",
    }
    trajectories = {PRIMARY_ARM: np.arange(12, dtype=np.float32).reshape(2, 2, 3)}

    emitted, artifact = _write_case_artifacts(tmp_path, "fixture", report, trajectories)

    stored = json.loads((tmp_path / "fixture.json").read_text(encoding="utf-8"))
    assert emitted == stored
    assert stored["result_sha256"] == primary_artifact_sha256(stored)
    assert artifact["report_result_sha256"] == stored["result_sha256"]
    assert artifact["archive_sha256"] == stored["trajectory_archive_sha256"]
    summary = primary._sign_artifact(
        {
            "schema_version": 1,
            "protocol_id": primary.PRIMARY_EVALUATION_PROTOCOL_ID,
            "artifacts": [artifact],
        }
    )
    assert summary["result_sha256"] == primary_artifact_sha256(summary)


def test_read_only_parity_logic_checks_trajectory_score_and_update_metadata(
    tmp_path: Path,
) -> None:
    (
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        centers,
    ) = _arrays()
    covariance = np.full(prior.shape[:2] + (3, 3), np.nan)
    covariance_validity = np.zeros(prior.shape[:2], dtype=bool)
    report, trajectories = evaluate_primary_arrays(
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        center_ids=centers,
        scored_frames=(20, 39, 58),
    )
    report["case"] = "fixture"
    reference, gated_trajectories = evaluate_covariance_gated_arrays(
        prior,
        persistence,
        target,
        visible,
        valid,
        measurement,
        measurement_validity,
        covariance,
        covariance_validity,
        center_ids=centers,
        scored_frames=(20, 39, 58),
        gate_thresholds={"ungated": -np.inf},
    )
    reference["case"] = "fixture"
    archive = tmp_path / "fixture.npz"
    np.savez_compressed(archive, **gated_trajectories)

    result = _compare_primary_case_to_gated(
        report, trajectories, reference, archive.read_bytes()
    )

    assert result["parity_passed"] is True
    assert all(result["trajectory_bit_exact"].values())
    assert all(result["score_within_absolute_tolerance"].values())
    assert all(
        update["selection_metadata_bit_exact"]
        and update["support_semantics_equivalent"]
        for update in result["updates"]
    )
    with np.load(archive, allow_pickle=False) as stored:
        changed = {name: np.asarray(stored[name]).copy() for name in stored.files}
    changed[PRIMARY_ARM][20, 0, 0] = np.nextafter(
        changed[PRIMARY_ARM][20, 0, 0], np.float32(np.inf)
    )
    np.savez_compressed(archive, **changed)
    changed_result = _compare_primary_case_to_gated(
        report, trajectories, reference, archive.read_bytes()
    )
    assert changed_result["parity_passed"] is False
    assert changed_result["trajectory_bit_exact"][PRIMARY_ARM] is False


def test_parity_completes_all_predictions_before_reference_or_target_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = tuple(f"fixture-{index:02d}" for index in range(27))
    events: list[str] = []
    monkeypatch.setattr(primary, "expected_open_case_names", lambda: cases)

    def predict(panel_case: Path, _measurement_case: Path) -> str:
        case = Path(panel_case).name
        events.append(f"predict:{case}")
        return case

    def bind_reference(
        root: Path,
        bound_cases: tuple[str, ...],
        **_kwargs: str,
    ) -> primary._GatedReference:
        assert tuple(bound_cases) == cases
        assert events == [f"predict:{case}" for case in cases]
        events.append("reference-summary")
        return primary._GatedReference(
            root=root,
            summary={"protocol_id": primary.GATED_EVALUATION_PROTOCOL_ID},
            summary_file_sha256="a" * 64,
            summary_result_sha256="b" * 64,
            report_sha256_by_case={case: "c" * 64 for case in cases},
            archive_sha256_by_case={case: "d" * 64 for case in cases},
        )

    def score(case: str) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        events.append(f"target:{case}")
        return {"case": case}, {}

    monkeypatch.setattr(primary, "_load_verified_measurement", predict)
    monkeypatch.setattr(primary, "_validate_gated_reference", bind_reference)
    monkeypatch.setattr(primary, "_evaluate_verified_measurement", score)
    monkeypatch.setattr(
        primary,
        "_read_gated_reference_case",
        lambda _binding, case: ({"case": case}, b"archive"),
    )
    monkeypatch.setattr(
        primary,
        "_compare_primary_case_to_gated",
        lambda *_args: {
            "all_primary_arrays_byte_exact": True,
            "parity_passed": True,
        },
    )
    monkeypatch.setattr(primary, "_recheck_verified_inputs", lambda *_args, **_kw: None)
    monkeypatch.setattr(primary, "_read_bound_bytes", lambda *_args, **_kwargs: b"")

    result = primary.compare_primary_to_gated_cohort(
        tmp_path / "panel",
        tmp_path / "measurements",
        tmp_path / "reference",
        expected_gated_summary_file_sha256="a" * 64,
        expected_gated_summary_result_sha256="b" * 64,
    )

    assert result["all_27_cases_parity_passed"] is True
    assert events[:27] == [f"predict:{case}" for case in cases]
    assert events[27] == "reference-summary"
    assert events[28:] == [f"target:{case}" for case in cases]


def test_reference_summary_is_externally_bound_and_case_files_rechecked(
    tmp_path: Path,
) -> None:
    case = "fixture"
    report_path = tmp_path / f"{case}.json"
    archive_path = tmp_path / f"{case}.npz"
    report_path.write_text('{"case":"fixture"}\n', encoding="utf-8")
    archive_path.write_bytes(b"bound archive")
    report_sha256 = primary._sha256(report_path)
    archive_sha256 = primary._sha256(archive_path)
    summary = primary._sign_artifact(
        {
            "schema_version": 1,
            "protocol_id": primary.GATED_EVALUATION_PROTOCOL_ID,
            "episode_count": 1,
            "artifacts": [
                {
                    "case": case,
                    "report_sha256": report_sha256,
                    "archive_sha256": archive_sha256,
                }
            ],
        }
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_file_sha256 = primary._sha256(summary_path)
    binding = primary._validate_gated_reference(
        tmp_path,
        (case,),
        expected_summary_file_sha256=summary_file_sha256,
        expected_summary_result_sha256=summary["result_sha256"],
    )

    loaded_report, loaded_archive = primary._read_gated_reference_case(binding, case)
    assert loaded_report["case"] == case
    assert loaded_archive == b"bound archive"
    report_path.write_text('{"case":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="report.*checksum changed"):
        primary._read_gated_reference_case(binding, case)


def test_verified_input_digests_are_rechecked_before_boundary(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    measurement_dir = tmp_path / "measurement"
    case_dir.mkdir()
    measurement_dir.mkdir()
    seal_path = case_dir / "prediction_seal.json"
    manifest_path = measurement_dir / primary.MANIFEST_FILENAME
    measurement_path = measurement_dir / primary.MEASUREMENT_FILENAME
    prediction_path = case_dir / "prediction.npz"
    seal_path.write_bytes(b"seal")
    manifest_path.write_bytes(b"manifest")
    measurement_path.write_bytes(b"measurement")
    prediction_path.write_bytes(b"prediction")
    verified = primary._VerifiedMeasurement(
        case_dir=case_dir,
        measurement_dir=measurement_dir,
        seal={},
        manifest={},
        arrays={},
        prediction_archive=prediction_path,
        physical_prior=np.empty((0,)),
        persistence=np.empty((0,)),
        selected_raw=np.empty((0,)),
        prediction=np.empty((0,)),
        prediction_diagnostic={},
        prediction_seal_sha256=primary._sha256(seal_path),
        measurement_manifest_sha256=primary._sha256(manifest_path),
        measurement_archive_sha256=primary._sha256(measurement_path),
        prediction_archive_sha256=primary._sha256(prediction_path),
    )

    primary._recheck_verified_inputs(verified, boundary="test")
    measurement_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="measurement archive changed before test"):
        primary._recheck_verified_inputs(verified, boundary="test")
