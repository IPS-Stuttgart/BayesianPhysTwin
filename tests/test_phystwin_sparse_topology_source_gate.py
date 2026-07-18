import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_piecewise_topology import (
    build_piecewise_topology_candidate,
    write_piecewise_topology_artifact,
)
from bayesian_phystwin.phystwin_sparse_topology_source_gate import (
    run_sparse_topology_source_gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(cases: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_name": "phystwin-global-sparse-topology-source-v1",
        "transfer_cases": cases,
        "evidence_boundary": {"fit_fraction_of_released_prefix": 0.75},
        "candidate": {
            "radius_multiplier": 0.45,
            "neighbour_multiplier": 2.0 / 3.0,
            "object_scale_normalization": "preserve_total_object_stiffness",
        },
        "source_acceptance": {
            "minimum_both_metric_win_count": 2,
            "minimum_aggregate_balanced_improvement": 0.03,
        },
    }


def _topologies(case_root: Path, final_data: Path) -> tuple[dict, dict]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    controls = np.array([[0.3, 0.0, 0.0]], dtype=np.float32)
    assignments = np.array([0, 0, 1, 1], dtype=np.int32)
    config = PhysTwinSpringGraphConfig(
        object_radius=1.0,
        object_max_neighbours=3,
        controller_radius=0.5,
        controller_max_neighbours=2,
    )
    graph = build_phystwin_spring_graph(points, controls, config=config)
    spring_y = np.arange(1, len(graph.springs) + 1, dtype=np.float32)
    common = {
        "structure_points": points,
        "controller_points": controls,
        "region_assignments": assignments,
        "teacher_spring_y": spring_y,
        "teacher_config": config,
        "preserve_total_object_stiffness": True,
    }
    identity = build_piecewise_topology_candidate(
        **common,
        radius_multipliers=(1.0, 1.0),
        neighbour_multipliers=(1.0, 1.0),
    )
    candidate = build_piecewise_topology_candidate(
        **common,
        radius_multipliers=(0.45, 0.45),
        neighbour_multipliers=(2.0 / 3.0, 2.0 / 3.0),
    )

    sidecars = []
    for name, artifact, radius, neighbour in (
        ("identity", identity, 1.0, 1.0),
        ("candidate", candidate, 0.45, 2.0 / 3.0),
    ):
        directory = case_root / name
        directory.mkdir(parents=True)
        artifact_identity = write_piecewise_topology_artifact(
            directory / "topology.npz", artifact
        )
        sidecar = {
            "contract": "phystwin-piecewise-topology-v1",
            "future_observations_used": False,
            "artifact": artifact_identity,
            "inputs": {"final_data": {"sha256": _sha256(final_data)}},
            "search_coordinates": {
                "radius_multipliers": [radius, radius],
                "neighbour_multipliers": [neighbour, neighbour],
                "requested_object_log_scale": 0.0,
                "controller_log_scale": 0.0,
                "object_scale_normalization": (
                    "preserve_total_object_stiffness"
                ),
            },
        }
        (directory / "topology.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
        sidecars.append(sidecar)
    return tuple(sidecars)


def _summary(
    topology_sha: str,
    final_data_sha: str,
    tracks_sha: str,
    *,
    cd: float,
    track: float,
) -> dict[str, object]:
    return {
        "code_commit": "candidate-commit",
        "official_commit": "official-commit",
        "config": {
            "variant": "hard",
            "train_end_frame": 20,
            "fit_end_frame": 15,
            "epochs": 0,
            "spring_parameterization": "dense",
            "selection_metric": "official_3d",
            "deterministic_spring_forces": True,
            "optimize_collision": False,
            "dashpot_log_scale": 0.0,
            "drag_log_scale": 0.0,
        },
        "inputs": {
            "final_data": {"sha256": final_data_sha},
            "gt_track_3d": {"sha256": tracks_sha},
            "spring_topology": {"sha256": topology_sha},
        },
        "official_evaluation": {
            "validation": {
                "chamfer_distance_m": cd,
                "track_error_m": track,
            }
        },
        "selected_baseline_trajectory_parity": {"vector_rmse_m": 0.0},
    }


def _write_case(
    root: Path,
    case: str,
    *,
    candidate_cd: float,
    candidate_track: float,
) -> None:
    case_root = root / "cases" / case
    prefix = case_root / "prefix"
    prefix.mkdir(parents=True)
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
                    "final_data": {"sha256": _sha256(final_data)},
                    "gt_track_3d": {"sha256": _sha256(tracks)},
                },
            }
        ),
        encoding="utf-8",
    )
    identity_sidecar, candidate_sidecar = _topologies(case_root, final_data)
    identity = case_root / "identity"
    candidate = case_root / "candidate"
    (identity / "trajectory.pkl").write_bytes(b"teacher")
    (candidate / "trajectory.pkl").write_bytes(b"candidate")
    (identity / "summary.json").write_text(
        json.dumps(
            _summary(
                identity_sidecar["artifact"]["sha256"],
                _sha256(final_data),
                _sha256(tracks),
                cd=0.010,
                track=0.020,
            )
        ),
        encoding="utf-8",
    )
    (candidate / "summary.json").write_text(
        json.dumps(
            _summary(
                candidate_sidecar["artifact"]["sha256"],
                _sha256(final_data),
                _sha256(tracks),
                cd=candidate_cd,
                track=candidate_track,
            )
        ),
        encoding="utf-8",
    )


def test_sparse_topology_gate_requires_transfer_and_preserved_stiffness(
    tmp_path: Path,
) -> None:
    cases = ["win_a", "win_b", "loss"]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol(cases)), encoding="utf-8")
    root = tmp_path / "source"
    root.mkdir()
    (root / "locked_protocol.json").write_bytes(protocol_path.read_bytes())
    _write_case(root, cases[0], candidate_cd=0.009, candidate_track=0.018)
    _write_case(root, cases[1], candidate_cd=0.0095, candidate_track=0.019)
    _write_case(root, cases[2], candidate_cd=0.0101, candidate_track=0.0201)

    result = run_sparse_topology_source_gate(
        root, tmp_path / "result.json", protocol_path
    )

    assert result["future_metrics_opened"] is False
    assert result["both_metric_win_count"] == 2
    assert result["source_gate_passed"] is True
    assert result["selected_family"] == "global_sparse_density_matched"
    assert result["aggregate_validation_metrics"]["candidate"][
        "track_error_m"
    ] < result["aggregate_validation_metrics"]["identity"]["track_error_m"]


def test_sparse_topology_gate_rejects_changed_protocol_lock(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol(["case"])), encoding="utf-8")
    root = tmp_path / "source"
    root.mkdir()
    (root / "locked_protocol.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="differs"):
        run_sparse_topology_source_gate(
            root, tmp_path / "result.json", protocol_path
        )
