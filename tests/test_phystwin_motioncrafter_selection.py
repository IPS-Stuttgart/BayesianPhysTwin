import json
from pathlib import Path

from bayesian_phystwin.phystwin_motioncrafter_selection import (
    select_motioncrafter_views,
)


def _write_summary(
    path: Path,
    *,
    camera: int,
    training_error: float,
    endpoint_coverage: float,
    manual_future_error: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "case": "case_a",
                "config": {"camera_index": camera},
                "alignment": {"view_count": 1},
                "frame_indices": [0, 1, 2, 3],
                "train_end_frame": 3,
                "graph": {
                    "association_initial_error_m": {"mean": training_error},
                    "training_motion_error_m": {"mean": training_error},
                    "valid_vertex_fraction_by_sampled_frame": [
                        1.0,
                        0.8,
                        endpoint_coverage,
                        endpoint_coverage / 2.0,
                    ]
                },
                "released_dense_track_error": {
                    "by_sampled_frame_m": [0.01, 0.02, training_error, 0.04],
                    "training_mean_m": training_error,
                    "future_mean_m": 0.04,
                },
                "manual_identity_audit": {
                    "error_by_sampled_frame_m": [0.01, 0.02, 0.03, 0.05],
                    "training_mean_m": 0.02,
                    "future_mean_m": manual_future_error,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_selection_penalizes_low_coverage_and_ignores_manual_audit(
    tmp_path: Path,
) -> None:
    camera_zero = tmp_path / "camera0.json"
    camera_one = tmp_path / "camera1.json"
    _write_summary(
        camera_zero,
        camera=0,
        training_error=0.02,
        endpoint_coverage=0.5,
        manual_future_error=10.0,
    )
    _write_summary(
        camera_one,
        camera=1,
        training_error=0.015,
        endpoint_coverage=0.2,
        manual_future_error=0.0,
    )

    result = select_motioncrafter_views([camera_zero, camera_one])

    assert result["cases"][0]["selected_camera_index"] == 0
    assert result["cases"][0]["perception_only_sensitivity_camera_index"] == 0
    assert result["selected_aggregate"]["audit_only_manual_future_error_m"] == 10.0
    assert "manual" not in result["selection_contract"]["score"]
