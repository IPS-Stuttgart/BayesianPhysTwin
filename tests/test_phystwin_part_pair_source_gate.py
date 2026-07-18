import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.phystwin_part_pair_source_gate import (
    run_part_pair_source_gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(cases: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_name": "phystwin-part-pair-source-v1",
        "source_cases": cases,
        "evidence_boundary": {"fit_fraction_of_released_prefix": 0.75},
        "method": {
            "epochs": 12,
            "learning_rate": 0.003,
            "spring_scale_weight_decay": 0.01,
            "early_stopping_patience": 3,
            "observation_variant": "hard",
            "selection_metric": "official_3d",
            "deterministic_spring_forces": True,
        },
        "family_gate": {
            "minimum_relative_score_improvement": 0.001,
            "maximum_per_metric_regression": 0.0,
        },
        "source_acceptance": {"minimum_learned_case_count": 2},
    }


def _write_case(
    root: Path,
    case: str,
    *,
    candidate_cd: float,
    candidate_track: float,
) -> None:
    prefix = root / case / "prefix"
    learned = root / case / "learned"
    prefix.mkdir(parents=True)
    learned.mkdir(parents=True)
    final_data = prefix / "final_data_prefix.pkl"
    tracks = prefix / "gt_track_3d_prefix.pkl"
    final_data.write_bytes(b"prefix observations")
    tracks.write_bytes(b"prefix tracks")
    (prefix / "manifest.json").write_text(
        json.dumps(
            {
                "contract": "phystwin-observation-prefix-plus-hold-v1",
                "prefix_end_frame": 20,
                "hold_frame_index": 20,
                "output_frame_count": 21,
                "outputs": {
                    "final_data": {
                        "path": str(final_data),
                        "sha256": _sha256(final_data),
                    },
                    "gt_track_3d": {
                        "path": str(tracks),
                        "sha256": _sha256(tracks),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (learned / "trajectory.pkl").write_bytes(b"candidate")
    (learned / "baseline_trajectory.pkl").write_bytes(b"teacher")
    summary = {
        "code_commit": "candidate-commit",
        "official_commit": "official-commit",
        "config": {
            "variant": "hard",
            "train_end_frame": 20,
            "fit_end_frame": 15,
            "epochs": 12,
            "learning_rate": 0.003,
            "spring_scale_weight_decay": 0.01,
            "early_stopping_patience": 3,
            "selection_metric": "official_3d",
            "spring_parameterization": "part_pair",
            "deterministic_spring_forces": True,
            "optimize_collision": False,
            "dashpot_log_scale": 0.0,
            "drag_log_scale": 0.0,
        },
        "inputs": {
            "final_data": {"sha256": _sha256(final_data)},
            "gt_track_3d": {"sha256": _sha256(tracks)},
        },
        "baseline_official_evaluation": {
            "validation": {
                "chamfer_distance_m": 0.010,
                "track_error_m": 0.020,
            }
        },
        "official_evaluation": {
            "validation": {
                "chamfer_distance_m": candidate_cd,
                "track_error_m": candidate_track,
            }
        },
        "selection": {"selected_epoch": 2},
        "selected_baseline_trajectory_parity": {"vector_rmse_m": 0.0},
        "released_baseline_trajectory_parity": {"vector_rmse_m": 0.0002},
        "parameters": {
            "group_log_scales": {
                "controller": 0.1,
                "part_pairs": [
                    {
                        "parts": [0, 0],
                        "log_scale": 0.2,
                        "spring_count": 10,
                    },
                    {
                        "parts": [0, 1],
                        "log_scale": -0.1,
                        "spring_count": 5,
                    },
                ],
            }
        },
    }
    (learned / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_source_gate_applies_balanced_fallback_and_aggregate_rule(
    tmp_path: Path,
) -> None:
    cases = ["accepted_a", "metric_regression", "accepted_b"]
    root = tmp_path / "source"
    root.mkdir()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol(cases)), encoding="utf-8")
    (root / "locked_protocol.json").write_bytes(protocol_path.read_bytes())
    _write_case(root, cases[0], candidate_cd=0.009, candidate_track=0.019)
    _write_case(root, cases[1], candidate_cd=0.008, candidate_track=0.0201)
    _write_case(root, cases[2], candidate_cd=0.0095, candidate_track=0.0195)

    result = run_part_pair_source_gate(
        root,
        tmp_path / "result.json",
        protocol_path,
    )

    assert result["future_metrics_opened"] is False
    assert result["learned_acceptance_count"] == 2
    assert result["source_gate_passed"] is True
    rejected = result["case_results"]["metric_regression"]
    assert rejected["decision"]["selected_family"] == "exact_teacher"
    assert rejected["artifacts"]["selected_validation_trajectory"]["path"].endswith(
        "baseline_trajectory.pkl"
    )
    assert result["aggregate_validation_metrics"]["selected"][
        "track_error_m"
    ] < result["aggregate_validation_metrics"]["teacher"]["track_error_m"]


def test_source_gate_rejects_a_changed_protocol_lock(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol(["case"])), encoding="utf-8")
    root = tmp_path / "source"
    root.mkdir()
    (root / "locked_protocol.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="differs"):
        run_part_pair_source_gate(root, tmp_path / "result.json", protocol_path)
