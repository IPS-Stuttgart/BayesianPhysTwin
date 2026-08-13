import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/matphys_backend_real_replay_smoke_v1.json"
RESULT_ROOT = ROOT / "results/sota/diagnostics/matphys_backend_real_replay_smoke_v1"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_replay_smoke_preserves_information_and_claim_boundaries() -> None:
    protocol = _json(PROTOCOL)
    replay = _json(RESULT_ROOT / "compact_replay_result.json")
    incumbent = _json(RESULT_ROOT / "incumbent_validation_metrics.json")
    candidate = _json(RESULT_ROOT / "candidate_validation_metrics.json")

    assert protocol["status"].startswith("development-only")
    assert protocol["causal_intervals"] == {
        "proposal_evidence_end_frame_exclusive": 30,
        "validation_frame_range_half_open": [30, 40],
        "future_frame_start": 40,
        "future_outcomes_may_be_opened_for_this_smoke": False,
    }
    assert protocol["case"]["target_object_excluded_from_matphys_training"] is True
    assert replay["future_outcomes_scored"] is False
    assert all(
        record["future_metrics_opened"] is False and record["gt_track_3d"] is None
        for record in replay["summary_checks"].values()
    )
    assert incumbent["frame_range_half_open"] == [30, 40]
    assert candidate["frame_range_half_open"] == [30, 40]
    assert incumbent["future_outcomes_opened"] is False
    assert candidate["future_outcomes_opened"] is False


def test_real_replay_smoke_is_compatible_substantive_and_guard_selected() -> None:
    replay = _json(RESULT_ROOT / "compact_replay_result.json")
    identity = _json(RESULT_ROOT / "identity_replay_check.json")
    incumbent = _json(RESULT_ROOT / "incumbent_validation_metrics.json")
    candidate = _json(RESULT_ROOT / "candidate_validation_metrics.json")
    selected = _json(RESULT_ROOT / "matphys-backend.json")

    assert replay["contract"] == {
        "action_support_exact": True,
        "array_names": [
            "action_support",
            "driven_readout_m",
            "frame_zero_points_m",
            "persistence_m",
            "prediction_m",
            "zero_action_readout_m",
        ],
        "array_names_equal": True,
        "finite": True,
        "frame_zero_exact": True,
        "persistence_exact": True,
        "prediction_dtype": "float32",
        "prediction_shape": [58, 4607, 3],
    }
    assert (
        replay["candidate_vs_incumbent_prediction_coordinate_rmse_m"][
            "validation_frames_30_39"
        ]
        > 0.003
    )
    assert identity["byte_exact_archive"] is True
    assert identity["coordinate_rmse_m"] == 0.0
    assert (
        candidate["metrics"]["chamfer_distance_m"]
        < incumbent["metrics"]["chamfer_distance_m"]
    )
    assert candidate["metrics"]["track_error_m"] < incumbent["metrics"]["track_error_m"]
    assert selected["candidate_accepted"] is True
    assert selected["selected_backend"] == "matphys_warp_proposal"
    assert selected["output"]["byte_exact_source_copy"] is True
    assert selected["output"]["exact_incumbent_fallback_verified"] is False
