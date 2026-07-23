from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_adaptive_covariance_evaluation as adaptive


def _placeholder_inputs(
    tmp_path: Path,
    cases: tuple[str, ...],
) -> tuple[Path, dict[int, Path], dict[int, Path]]:
    panel = tmp_path / "panel"
    measurements = {budget: tmp_path / f"measurements-{budget}" for budget in (4, 8)}
    uncertainties = {budget: tmp_path / f"uncertainties-{budget}" for budget in (4, 8)}
    for case in cases:
        (panel / case).mkdir(parents=True)
        for budget in (4, 8):
            measurement = measurements[budget] / case
            measurement.mkdir(parents=True)
            (measurement / adaptive.MANIFEST_FILENAME).write_bytes(b"manifest")
            (measurement / adaptive.MEASUREMENT_FILENAME).write_bytes(b"measurement")
            uncertainty = uncertainties[budget] / case
            uncertainty.mkdir(parents=True)
            (uncertainty / adaptive.UNCERTAINTY_MANIFEST_FILENAME).write_bytes(
                b"manifest"
            )
            (uncertainty / adaptive.UNCERTAINTY_ARCHIVE_FILENAME).write_bytes(
                b"uncertainty"
            )
    return panel, measurements, uncertainties


def _fake_inventory(root: Path) -> adaptive.TreeInventory:
    return adaptive.TreeInventory(
        root=root,
        file_count=0,
        total_file_bytes=0,
        inventory_sha256="0" * 64,
        sha256_by_relative_path={},
    )


def test_selected_camera_budget_validates_manifest_and_archive(tmp_path: Path) -> None:
    cameras = [f"camera-{index:02d}" for index in range(4)]
    measurement = adaptive._VerifiedMeasurement(
        case_dir=tmp_path,
        measurement_dir=tmp_path,
        seal={},
        manifest={
            "config": {"selected_camera_count": 4},
            "plan": {"selected_cameras": cameras},
            "selected_camera_inputs": {camera: {} for camera in cameras},
        },
        arrays={"selected_cameras": np.asarray(cameras)},
        prediction_archive=tmp_path / "prediction.npz",
        physical_prior=np.empty((0,)),
        persistence=np.empty((0,)),
        selected_raw=np.empty((0,)),
        prediction=np.empty((0,)),
        prediction_diagnostic={},
        prediction_seal_sha256="0" * 64,
        measurement_manifest_sha256="1" * 64,
        measurement_archive_sha256="2" * 64,
        prediction_archive_sha256="3" * 64,
    )

    assert adaptive._selected_cameras(measurement, budget=4) == tuple(cameras)
    changed = dict(measurement.manifest)
    changed["config"] = {"selected_camera_count": 8}
    wrong = adaptive._VerifiedMeasurement(
        **{**measurement.__dict__, "manifest": changed}
    )
    with pytest.raises(ValueError, match="camera budget changed"):
        adaptive._selected_cameras(wrong, budget=4)


def test_input_inventory_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = adaptive.TreeInventory(
        root=tmp_path,
        file_count=55,
        total_file_bytes=1_643_339,
        inventory_sha256=(
            "baa9c35a91da4eb3843cdcbb63889d0e2fd60532122da183cad1eda3a4c82141"
        ),
        sha256_by_relative_path={},
    )
    monkeypatch.setattr(adaptive, "inventory_tree", lambda _root: observed)

    with pytest.raises(ValueError, match="inventory file_count changed"):
        adaptive._verify_input_inventory(
            tmp_path,
            role="measurement",
            budget=4,
        )


def test_output_must_be_disjoint_from_every_input_root(tmp_path: Path) -> None:
    panel = tmp_path / "panel"
    measurements = {4: tmp_path / "m4", 8: tmp_path / "m8"}
    uncertainties = {4: tmp_path / "u4", 8: tmp_path / "u8"}

    with pytest.raises(ValueError, match="output overlaps panel root"):
        adaptive._validate_root_separation(
            panel,
            measurements,
            uncertainties,
            panel / "output",
        )
    with pytest.raises(ValueError, match="roots overlap"):
        adaptive._validate_root_separation(
            panel,
            {4: measurements[4], 8: measurements[4]},
            uncertainties,
            tmp_path / "output",
        )


def test_cohort_completes_every_prediction_before_first_target_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = ("first-ep0000", "second-ep0000")
    panel, measurements, uncertainties = _placeholder_inputs(tmp_path, cases)
    events: list[str] = []
    monkeypatch.setattr(adaptive, "expected_open_case_names", lambda: cases)
    monkeypatch.setattr(
        adaptive,
        "_verify_input_inventory",
        lambda root, **_kwargs: _fake_inventory(Path(root)),
    )
    monkeypatch.setattr(
        adaptive,
        "_load_development_config",
        lambda: {"file_sha256": "0" * 64},
    )

    def predict(
        panel_case: Path,
        _measurement_dirs: object,
        _uncertainty_dirs: object,
    ) -> str:
        case = Path(panel_case).name
        events.append(f"predict:{case}")
        return case

    def score(case: str) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        events.append(f"target:{case}")
        return {"case": case}, {}

    monkeypatch.setattr(adaptive, "_load_verified_adaptive_case", predict)
    monkeypatch.setattr(adaptive, "_evaluate_verified_adaptive_case", score)
    monkeypatch.setattr(
        adaptive,
        "_recheck_adaptive_inputs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        adaptive,
        "_summary",
        lambda *_args, **_kwargs: {"result_sha256": "0" * 64},
    )

    adaptive.evaluate_adaptive_covariance_cohort(
        panel,
        measurements,
        uncertainties,
        tmp_path / "output",
    )

    assert events == [
        "predict:first-ep0000",
        "predict:second-ep0000",
        "target:first-ep0000",
        "target:second-ep0000",
    ]


def test_late_prediction_failure_opens_no_target_and_publishes_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = ("first-ep0000", "late-corrupt-ep0000")
    panel, measurements, uncertainties = _placeholder_inputs(tmp_path, cases)
    target_opened = False
    monkeypatch.setattr(adaptive, "expected_open_case_names", lambda: cases)
    monkeypatch.setattr(
        adaptive,
        "_verify_input_inventory",
        lambda root, **_kwargs: _fake_inventory(Path(root)),
    )
    monkeypatch.setattr(
        adaptive,
        "_load_development_config",
        lambda: {"file_sha256": "0" * 64},
    )

    def predict(
        panel_case: Path,
        _measurement_dirs: object,
        _uncertainty_dirs: object,
    ) -> str:
        if Path(panel_case).name == cases[-1]:
            raise ValueError("late covariance checksum changed")
        return Path(panel_case).name

    def score(_case: str) -> None:
        nonlocal target_opened
        target_opened = True
        raise AssertionError("target must not open")

    monkeypatch.setattr(adaptive, "_load_verified_adaptive_case", predict)
    monkeypatch.setattr(adaptive, "_evaluate_verified_adaptive_case", score)

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="late covariance checksum changed"):
        adaptive.evaluate_adaptive_covariance_cohort(
            panel,
            measurements,
            uncertainties,
            output,
        )
    assert target_opened is False
    assert not output.exists()
