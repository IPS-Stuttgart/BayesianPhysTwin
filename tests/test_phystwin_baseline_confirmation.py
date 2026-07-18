import json
from pathlib import Path

import pytest

from bayesian_phystwin.phystwin_baseline_confirmation import (
    _fit_case,
    _comparison_manifest,
    run_phystwin_baseline_confirmation,
)
from bayesian_phystwin.phystwin_confirmation_lock import confirmation_output_lock
from bayesian_phystwin.phystwin_confirmatory import PhysTwinConfirmatoryProtocol


def test_baseline_comparison_manifest_uses_method_trajectory(tmp_path: Path):
    case = tmp_path / "data" / "case_a"
    case.mkdir(parents=True)
    (case / "split.json").write_text(
        json.dumps({"frame_len": 10, "train": [0, 7], "test": [7, 10]})
    )

    manifest = _comparison_manifest(
        tmp_path / "data", tmp_path / "output", ("case_a",), "dmdc"
    )

    entry = manifest["cases"][0]
    assert entry["start_frame"] == 7
    assert entry["candidate_trajectory"].endswith(
        "/output/cases/case_a/dmdc/trajectory.pkl"
    )


def test_baseline_confirmation_rejects_nonpositive_workers(tmp_path: Path):
    with pytest.raises(ValueError, match="workers must be positive"):
        run_phystwin_baseline_confirmation(tmp_path, tmp_path / "out", workers=0)


def test_baseline_confirmation_rejects_a_second_output_owner(tmp_path: Path):
    output = tmp_path / "out"

    with confirmation_output_lock(output):
        with pytest.raises(RuntimeError, match="another PhysTwin confirmation"):
            run_phystwin_baseline_confirmation(tmp_path, output)


def test_baseline_confirmation_rejects_duplicate_cases(tmp_path: Path):
    (tmp_path / "evaluation_subset_manifest.json").write_text(
        json.dumps({"selected_cases": ["case_a", "case_a"]})
    )
    with pytest.raises(ValueError, match="duplicate cases"):
        run_phystwin_baseline_confirmation(tmp_path, tmp_path / "out")


def test_baseline_fit_rejects_split_changed_while_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = tmp_path / "data"
    case = data_root / "case_a"
    case.mkdir(parents=True)
    split_path = case / "split.json"
    split_path.write_text(
        json.dumps({"frame_len": 10, "train": [0, 8], "test": [8, 10]})
    )
    for filename in ("final_data.pkl", "inference.pkl", "gt_track_3d.pkl"):
        (case / filename).write_bytes(filename.encode())

    def fake_fit(*args, **kwargs):
        output = Path(args[3])
        output.mkdir(parents=True)
        (output / "trajectory.pkl").write_bytes(b"trajectory")
        split_path.write_text(
            json.dumps({"frame_len": 11, "train": [0, 8], "test": [8, 11]})
        )
        return {"inputs": {}}

    monkeypatch.setattr(
        "bayesian_phystwin.phystwin_baseline_confirmation.fit_residual_dynamics_baselines",
        fake_fit,
    )

    with pytest.raises(RuntimeError, match="sources changed while the fit was running"):
        _fit_case(
            (
                data_root,
                tmp_path / "output",
                "case_a",
                PhysTwinConfirmatoryProtocol(),
                False,
            )
        )
