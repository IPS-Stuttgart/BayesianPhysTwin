import json
from pathlib import Path

import pytest

from bayesian_phystwin.phystwin_bayesian_confirmation import (
    BayesianAnchorConfirmationProtocol,
    _fit_case,
    run_bayesian_anchor_confirmation,
)
from bayesian_phystwin.phystwin_confirmation_lock import confirmation_output_lock


def test_bayesian_confirmation_protocol_freezes_reliability_grid() -> None:
    protocol = BayesianAnchorConfirmationProtocol()

    assert protocol.maximum_residual_m == 0.01
    assert protocol.process_std_candidates_m[-1] == 0.005
    assert protocol.observation_std_candidates_m == (0.001, 0.0025, 0.005)
    assert protocol.inlier_prior == 0.95


def test_bayesian_confirmation_rejects_nonpositive_workers(tmp_path) -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        run_bayesian_anchor_confirmation(tmp_path, tmp_path / "out", workers=0)


def test_bayesian_confirmation_rejects_a_second_output_owner(tmp_path) -> None:
    output = tmp_path / "out"

    with confirmation_output_lock(output):
        with pytest.raises(RuntimeError, match="another PhysTwin confirmation"):
            run_bayesian_anchor_confirmation(tmp_path, output)


def test_bayesian_confirmation_rejects_duplicate_cases(tmp_path) -> None:
    (tmp_path / "evaluation_subset_manifest.json").write_text(
        json.dumps({"selected_cases": ["case_a", "case_a"]})
    )
    with pytest.raises(ValueError, match="duplicate cases"):
        run_bayesian_anchor_confirmation(tmp_path, tmp_path / "out")


def test_bayesian_fit_rejects_input_changed_while_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    case = data_root / "case_a"
    case.mkdir(parents=True)
    (case / "split.json").write_text(
        json.dumps({"frame_len": 10, "train": [0, 8], "test": [8, 10]})
    )
    gt_track = case / "gt_track_3d.pkl"
    for filename in ("final_data.pkl", "inference.pkl", "gt_track_3d.pkl"):
        (case / filename).write_bytes(filename.encode())

    def fake_fit(*args, **kwargs):
        output = Path(args[3])
        output.mkdir(parents=True)
        (output / "trajectory.pkl").write_bytes(b"trajectory")
        gt_track.write_bytes(b"changed during fit")
        return {"inputs": {}}

    monkeypatch.setattr(
        "bayesian_phystwin.phystwin_bayesian_confirmation.fit_bayesian_residual_anchor",
        fake_fit,
    )

    with pytest.raises(RuntimeError, match="sources changed while the fit was running"):
        _fit_case(
            (
                data_root,
                tmp_path / "output",
                "case_a",
                BayesianAnchorConfirmationProtocol(),
                False,
            )
        )
