import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.matphys_part_family_gate as family_gate
from bayesian_phystwin.matphys_graph_parts import GRAPH_PART_PROXY_CONTRACT
from bayesian_phystwin.matphys_part_model import PART_AWARE_MODEL_CONTRACT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)
    return {"path": str(path), "sha256": _sha256(path)}


def test_part_family_requires_balanced_disjoint_improvement() -> None:
    accepted = family_gate.choose_part_family(
        {"chamfer_distance_m": 0.01, "track_error_m": 0.02},
        {"chamfer_distance_m": 0.009, "track_error_m": 0.019},
    )
    regressed = family_gate.choose_part_family(
        {"chamfer_distance_m": 0.01, "track_error_m": 0.02},
        {"chamfer_distance_m": 0.008, "track_error_m": 0.0201},
    )

    assert accepted["selected_family"] == "learned_part_residual"
    assert regressed["selected_family"] == "exact_teacher"
    assert regressed["candidate_normalized_score"] < 1.0
    assert regressed["no_metric_regression"] is False


def test_part_family_rejects_improvements_below_replay_floor_gate() -> None:
    decision = family_gate.choose_part_family(
        {"chamfer_distance_m": 0.01, "track_error_m": 0.02},
        {"chamfer_distance_m": 0.0099999, "track_error_m": 0.0199998},
        minimum_relative_score_improvement=0.001,
    )

    assert decision["relative_score_improvement"] == pytest.approx(1e-5)
    assert decision["selected_family"] == "exact_teacher"


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data_root = tmp_path / "data"
    case_root = data_root / "case_a"
    case_root.mkdir(parents=True)
    frame_count = 8
    observed = np.zeros((frame_count, 3, 3), dtype=np.float32)
    observed[:, :, 0] = np.asarray([0.0, 0.01, 0.02])
    observed[:, :, 1] = np.arange(frame_count)[:, None] * 0.001
    final_data = {
        "object_points": observed,
        "object_visibilities": np.ones((frame_count, 3), dtype=bool),
        "surface_points": np.asarray([[0.03, 0.0, 0.0]], dtype=np.float32),
    }
    with (case_root / "final_data.pkl").open("wb") as handle:
        pickle.dump(final_data, handle)
    tracks = observed[:, :1].copy()
    with (case_root / "gt_track_3d.pkl").open("wb") as handle:
        pickle.dump(tracks, handle)
    (case_root / "split.json").write_text(
        json.dumps({"train": [0, 6], "test": [6, 8], "frame_len": 8}),
        encoding="utf-8",
    )

    teacher = np.zeros((6, 4, 3), dtype=np.float32)
    teacher[:, :3] = observed[:6]
    teacher[:, 3, 0] = 0.03
    teacher[1:, :, 1] += 0.002
    candidate = teacher.copy()
    candidate[1:, :, 1] -= 0.001
    candidate_identity = _dump(tmp_path / "candidate_validation.pkl", candidate)
    teacher_identity = _dump(tmp_path / "teacher_validation.pkl", teacher)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    audit = tmp_path / "audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    spring_summary = tmp_path / "spring_summary.json"
    spring_summary.write_text(
        json.dumps(
            {
                "overall": {"minimum": 0.8, "maximum": 1.2},
                "by_part": [{"mean": 0.9}, {"mean": 1.1}],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "backbone": {
            "future_observations_used": False,
            "proxy_contract": GRAPH_PART_PROXY_CONTRACT,
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "causal_training_audit": {
                "path": str(audit),
                "sha256": _sha256(audit),
            },
            "parameterization": {
                "part_model_contract": PART_AWARE_MODEL_CONTRACT,
                "residual_log_scale": float(np.log(2.0)),
            },
        },
        "cases": [
            {
                "name": "case_a",
                "spring_field_summary": {
                    "path": str(spring_summary),
                    "sha256": _sha256(spring_summary),
                },
                "causal_validation": {
                    "contract": family_gate.PAIRED_VALIDATION_CONTRACT,
                    "fit_end_frame_exclusive": 4,
                    "validation_end_frame_exclusive": 6,
                    "candidate": candidate_identity,
                    "teacher": teacher_identity,
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return data_root, manifest_path, checkpoint, audit


def test_gate_reads_only_prefix_truncated_pair_and_falls_back_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, manifest_path, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        family_gate,
        "validate_causal_training_audit",
        lambda audit, checkpoint: {
            "cases": [
                {
                    "name": "case_a",
                    "evidence_end_frame_exclusive": 4,
                    "validation_frame_interval": [4, 6],
                }
            ]
        },
    )

    result = family_gate.run_matphys_part_family_gate(
        data_root,
        tmp_path / "gate",
        manifest_path,
        case_names=("case_a",),
        source_protocol=manifest_path,
    )

    assert result["future_metrics_opened"] is False
    assert result["learned_acceptance_count"] == 1
    assert result["source_gate_passed"] is True
    assert result["aggregate_both_metrics_improved"] is True
    assert result["contract"]["source_protocol"]["sha256"] == _sha256(
        manifest_path
    )
    case = result["case_results"]["case_a"]
    assert case["decision"]["selected_family"] == "learned_part_residual"
    assert case["candidate_validation_metrics"]["track_error_m"] < case[
        "teacher_validation_metrics"
    ]["track_error_m"]

    case_root = data_root / "case_a"
    with (case_root / "final_data.pkl").open("rb") as handle:
        future_mutated_data = pickle.load(handle)
    future_mutated_data["object_points"][6:] += 1000.0
    future_mutated_data["object_visibilities"][6:] = False
    with (case_root / "final_data.pkl").open("wb") as handle:
        pickle.dump(future_mutated_data, handle)
    with (case_root / "gt_track_3d.pkl").open("rb") as handle:
        future_mutated_tracks = pickle.load(handle)
    future_mutated_tracks[6:] -= 1000.0
    with (case_root / "gt_track_3d.pkl").open("wb") as handle:
        pickle.dump(future_mutated_tracks, handle)

    mutated = family_gate.run_matphys_part_family_gate(
        data_root,
        tmp_path / "future_mutated_gate",
        manifest_path,
        case_names=("case_a",),
        source_protocol=manifest_path,
    )
    assert mutated["learned_acceptance_count"] == result[
        "learned_acceptance_count"
    ]
    assert mutated["aggregate_validation_metrics"] == result[
        "aggregate_validation_metrics"
    ]

    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["causal_validation"][
        "validation_end_frame_exclusive"
    ] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="boundaries changed"):
        family_gate.run_matphys_part_family_gate(
            data_root,
            tmp_path / "bad_gate",
            manifest_path,
            case_names=("case_a",),
        )
