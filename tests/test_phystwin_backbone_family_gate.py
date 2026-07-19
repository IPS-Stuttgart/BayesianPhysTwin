import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.phystwin_backbone_family_gate as family_gate_module
from bayesian_phystwin.phystwin_backbone_family_gate import (
    _load_stability_control_manifest,
    choose_backbone_family,
    choose_guarded_backbone_family,
    normalized_validation_score,
    open_backbone_family_future,
    trajectory_coordinate_rmse,
)


def _metrics(cd: float, track: float) -> dict[str, float]:
    return {"chamfer_distance_m": cd, "track_error_m": track}


def test_normalized_validation_score_balances_metrics() -> None:
    score = normalized_validation_score(_metrics(0.008, 0.024), _metrics(0.01, 0.02))

    assert score == pytest.approx(1.0)


def test_family_gate_uses_common_reference_and_preserves_tie_order() -> None:
    selected, scores = choose_backbone_family(
        {
            "released": _metrics(0.008, 0.016),
            "learned": _metrics(0.008, 0.016),
        },
        _metrics(0.01, 0.02),
    )

    assert selected == "released"
    assert scores == pytest.approx({"released": 0.8, "learned": 0.8})


def test_family_gate_can_accept_balanced_transfer() -> None:
    selected, scores = choose_backbone_family(
        {
            "released": _metrics(0.008, 0.018),
            "learned": _metrics(0.007, 0.017),
        },
        _metrics(0.01, 0.02),
    )

    assert selected == "learned"
    assert scores["learned"] < scores["released"]


def test_normalized_validation_score_rejects_invalid_reference() -> None:
    with pytest.raises(ValueError, match="positive references"):
        normalized_validation_score(_metrics(0.01, 0.02), _metrics(0.0, 0.02))


def test_guarded_family_gate_requires_both_metric_safety() -> None:
    selected, scores, decisions = choose_guarded_backbone_family(
        {
            "released": _metrics(0.008, 0.016),
            "learned": _metrics(0.007, 0.0161),
        },
        _metrics(0.01, 0.02),
        fallback_family="released",
        minimum_relative_improvement=0.001,
        maximum_metric_regression=0.0,
    )

    assert scores["learned"] < scores["released"]
    assert selected == "released"
    assert decisions["learned"]["no_metric_regression"] is False


def test_guarded_family_gate_rejects_unstable_candidate() -> None:
    selected, _, decisions = choose_guarded_backbone_family(
        {
            "released": _metrics(0.008, 0.016),
            "learned": _metrics(0.007, 0.014),
        },
        _metrics(0.01, 0.02),
        fallback_family="released",
        minimum_relative_improvement=0.001,
        maximum_metric_regression=0.0,
        eligible_families={"released": True, "learned": False},
    )

    assert selected == "released"
    assert decisions["learned"]["stability_eligible"] is False


def test_guarded_family_gate_accepts_safe_improvement() -> None:
    selected, _, decisions = choose_guarded_backbone_family(
        {
            "released": _metrics(0.008, 0.016),
            "learned": _metrics(0.007, 0.014),
        },
        _metrics(0.01, 0.02),
        fallback_family="released",
        minimum_relative_improvement=0.001,
        maximum_metric_regression=0.0,
        eligible_families={"released": True, "learned": True},
    )

    assert selected == "learned"
    assert decisions["learned"]["accepted"] is True


def test_trajectory_coordinate_rmse_uses_all_coordinates() -> None:
    reference = np.zeros((2, 3, 3))
    candidate = reference.copy()
    candidate[0, 0, 0] = 3.0

    assert trajectory_coordinate_rmse(reference, candidate) == pytest.approx(
        3.0 / (18.0**0.5)
    )


def test_stability_control_manifest_binds_family_cases_and_bytes(tmp_path) -> None:
    trajectory = tmp_path / "identity.pkl"
    trajectory.write_bytes(b"identity replay")
    digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    manifest = tmp_path / "controls.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "phystwin-family-stability-control-v1",
                "family": "learned",
                "future_observations_used": False,
                "cases": [
                    {
                        "name": "case_a",
                        "trajectory": {
                            "path": trajectory.name,
                            "sha256": digest,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    controls = _load_stability_control_manifest(
        manifest,
        expected_family="learned",
        expected_cases=("case_a",),
    )

    assert controls == {"case_a": trajectory.resolve()}

    trajectory.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        _load_stability_control_manifest(
            manifest,
            expected_family="learned",
            expected_cases=("case_a",),
        )


def test_future_opener_uses_hash_bound_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    released = tmp_path / "released.pkl"
    learned = tmp_path / "learned.pkl"
    staged = tmp_path / "staged.pkl"
    for path, value in ((released, 1.0), (learned, 2.0)):
        with path.open("wb") as handle:
            pickle.dump(np.full((4, 2, 3), value), handle)
    shutil.copy2(learned, staged)

    def identity(path: Path) -> dict[str, str]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "sealed",
                "future_metrics_opened": False,
                "case_results": {
                    "case_a": {
                        "selected_family": "learned",
                        "train_end_frame_exclusive": 2,
                        "frame_count": 4,
                        "family_outputs": {
                            "released": identity(released),
                            "learned": identity(learned),
                        },
                        "output": identity(staged),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_future_metrics(
        data_root, case, trajectory_path, *, train_end, frame_count
    ):
        del data_root, case, train_end, frame_count
        with Path(trajectory_path).open("rb") as handle:
            value = float(np.mean(pickle.load(handle)))
        return {"chamfer_distance_m": value, "track_error_m": 2.0 * value}

    monkeypatch.setattr(family_gate_module, "_future_metrics", fake_future_metrics)
    result = open_backbone_family_future(
        tmp_path / "data",
        tmp_path / "future",
        selection,
    )

    assert result["future_metrics_opened"] is True
    assert result["comparison"]["selected_equal_case_mean"] == {
        "chamfer_distance_m": 2.0,
        "track_error_m": 4.0,
    }
    assert result["case_results"]["case_a"]["selected_family"] == "learned"


def test_future_opener_rejects_an_already_opened_selection(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "future_metrics_opened": True,
                "case_results": {"case_a": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sealed prefix-only"):
        open_backbone_family_future(
            tmp_path / "data",
            tmp_path / "future",
            selection,
        )
