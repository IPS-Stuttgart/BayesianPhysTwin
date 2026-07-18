import hashlib
import json
from dataclasses import asdict
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
from bayesian_phystwin.phystwin_zero_order_source_gate import (
    run_zero_order_source_gate,
)
from bayesian_phystwin.phystwin_zero_order_topology import (
    ZERO_ORDER_TOPOLOGY_CONTRACT,
    ZeroOrderTopologySearchConfig,
    generate_topology_field_candidates,
    select_topology_field_candidate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(cases: list[str], config: ZeroOrderTopologySearchConfig) -> dict:
    return {
        "schema_version": 1,
        "protocol_name": "phystwin-zero-order-topology-field-source-v1",
        "transfer_cases": cases,
        "evidence_boundary": {
            "outer_fit_fraction_of_released_prefix": 0.75,
            "nested_selection_fraction_of_outer_fit": 0.75,
        },
        "search": {
            "region_count": config.region_count,
            "candidates_per_family": config.candidates_per_family,
            "total_candidate_count_including_teacher": (
                1 + 3 * config.candidates_per_family
            ),
            "seed": config.seed,
            "radius_bounds": list(config.radius_bounds),
            "neighbour_bounds": list(config.neighbour_bounds),
            "object_log_scale_bounds": list(config.object_log_scale_bounds),
            "region_log_scale_bounds": list(config.region_log_scale_bounds),
            "controller_log_scale_bounds": list(
                config.controller_log_scale_bounds
            ),
            "minimum_nested_fit_improvement": config.minimum_fit_improvement,
            "maximum_nested_fit_metric_ratio": config.maximum_fit_metric_ratio,
        },
        "source_acceptance": {
            "minimum_non_teacher_selection_count": 1,
            "minimum_both_metric_win_count": 1,
            "minimum_aggregate_balanced_improvement": 0.03,
        },
    }


def _write_prefixes(case_root: Path) -> tuple[dict, dict]:
    outer = case_root / "outer_prefix"
    fit = case_root / "fit_prefix"
    outer.mkdir(parents=True)
    fit.mkdir(parents=True)
    outer_outputs = {}
    for name in ("final_data", "gt_track_3d", "released_trajectory"):
        path = outer / f"{name}.pkl"
        path.write_bytes(f"outer {name}".encode())
        outer_outputs[name] = {"path": str(path), "sha256": _sha256(path)}
    outer_manifest = {
        "contract": "phystwin-observation-prefix-plus-hold-v1",
        "prefix_end_frame": 20,
        "hold_frame_index": 20,
        "output_frame_count": 21,
        "outputs": outer_outputs,
    }
    (outer / "manifest.json").write_text(
        json.dumps(outer_manifest), encoding="utf-8"
    )

    fit_outputs = {}
    for name in ("final_data", "gt_track_3d", "released_trajectory"):
        path = fit / f"{name}.pkl"
        path.write_bytes(f"fit {name}".encode())
        fit_outputs[name] = {"path": str(path), "sha256": _sha256(path)}
    fit_manifest = {
        "contract": "phystwin-observation-prefix-plus-hold-v1",
        "prefix_end_frame": 15,
        "hold_frame_index": 15,
        "output_frame_count": 16,
        "inputs": outer_outputs,
        "outputs": fit_outputs,
    }
    (fit / "manifest.json").write_text(
        json.dumps(fit_manifest), encoding="utf-8"
    )
    return outer_manifest, fit_manifest


def _candidate_artifacts(case_root: Path, config, candidates) -> dict[str, dict]:
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
    graph_config = PhysTwinSpringGraphConfig(
        object_radius=1.0,
        object_max_neighbours=3,
        controller_radius=0.5,
        controller_max_neighbours=2,
    )
    graph = build_phystwin_spring_graph(points, controls, config=graph_config)
    spring_y = np.arange(1, len(graph.springs) + 1, dtype=np.float32)
    result = {}
    for candidate in candidates:
        artifact = build_piecewise_topology_candidate(
            points,
            controls,
            assignments,
            spring_y,
            teacher_config=graph_config,
            radius_multipliers=candidate.radius_multipliers,
            neighbour_multipliers=candidate.neighbour_multipliers,
            object_log_scale=candidate.object_log_scale,
            controller_log_scale=candidate.controller_log_scale,
            preserve_total_object_stiffness=True,
            region_object_log_scales=candidate.region_object_log_scales,
        )
        candidate_root = case_root / "search" / "candidates" / candidate.candidate_id
        candidate_root.mkdir(parents=True)
        identity = write_piecewise_topology_artifact(
            candidate_root / "topology.npz", artifact
        )
        result[candidate.candidate_id] = {
            "status": "evaluated",
            "candidate": asdict(candidate),
            "topology": {"artifact": identity},
        }
    return result


def _full_summary(
    final_sha: str,
    tracks_sha: str,
    topology_sha: str,
    *,
    cd: float,
    track: float,
) -> dict:
    return {
        "code_commit": "test-commit",
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
            "final_data": {"sha256": final_sha},
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


def _write_case(root: Path, case: str, config) -> None:
    case_root = root / "cases" / case
    outer, fit = _write_prefixes(case_root)
    candidates = generate_topology_field_candidates(config)
    candidate_results = _candidate_artifacts(case_root, config, candidates)
    metrics = {
        candidate.candidate_id: {
            "chamfer_distance_m": 0.0101,
            "track_error_m": 0.0201,
        }
        for candidate in candidates
    }
    metrics["exact_teacher"] = {
        "chamfer_distance_m": 0.010,
        "track_error_m": 0.020,
    }
    metrics["field_000"] = {
        "chamfer_distance_m": 0.009,
        "track_error_m": 0.018,
    }
    for candidate_id, values in metrics.items():
        candidate_results[candidate_id]["selection_metrics"] = values
    selection = select_topology_field_candidate(metrics, candidates, config)
    selected_id = selection["selected_candidate_id"]
    search_root = case_root / "search"
    plan = {
        "contract": ZERO_ORDER_TOPOLOGY_CONTRACT,
        "config": asdict(config),
        "candidates": [asdict(candidate) for candidate in candidates],
        "selection_interval": [11, 15],
        "inputs": {"fit_final_data": fit["outputs"]["final_data"]},
        "future_observations_used": False,
    }
    plan_path = search_root / "search_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    search_summary = {
        "contract": ZERO_ORDER_TOPOLOGY_CONTRACT,
        "future_observations_used": False,
        "search_plan": {"sha256": _sha256(plan_path)},
        "selection": selection,
        "candidate_results": candidate_results,
    }
    (search_root / "search_summary.json").write_text(
        json.dumps(search_summary), encoding="utf-8"
    )

    identity_topology = candidate_results["exact_teacher"]["topology"]["artifact"]
    selected_topology = candidate_results[selected_id]["topology"]["artifact"]
    for name, topology, cd, track in (
        ("identity", identity_topology, 0.010, 0.020),
        ("selected", selected_topology, 0.009, 0.018),
    ):
        directory = case_root / name
        directory.mkdir()
        (directory / "trajectory.pkl").write_bytes(name.encode())
        summary = _full_summary(
            outer["outputs"]["final_data"]["sha256"],
            outer["outputs"]["gt_track_3d"]["sha256"],
            topology["sha256"],
            cd=cd,
            track=track,
        )
        (directory / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )


def test_zero_order_source_gate_recomputes_nested_selection(tmp_path: Path) -> None:
    config = ZeroOrderTopologySearchConfig(
        region_count=2,
        candidates_per_family=1,
        seed=17,
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(_protocol(["case"], config)), encoding="utf-8"
    )
    root = tmp_path / "source"
    root.mkdir()
    (root / "locked_protocol.json").write_bytes(protocol_path.read_bytes())
    _write_case(root, "case", config)

    result = run_zero_order_source_gate(
        root, tmp_path / "result.json", protocol_path
    )

    assert result["source_gate_passed"] is True
    assert result["future_metrics_opened"] is False
    assert result["non_teacher_selection_count"] == 1
    assert result["both_metric_win_count"] == 1
    assert result["case_results"]["case"]["selected_candidate_id"] == "field_000"


def test_zero_order_source_gate_rejects_changed_lock(tmp_path: Path) -> None:
    config = ZeroOrderTopologySearchConfig(
        region_count=2,
        candidates_per_family=1,
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(_protocol(["case"], config)), encoding="utf-8"
    )
    root = tmp_path / "source"
    root.mkdir()
    (root / "locked_protocol.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="differs"):
        run_zero_order_source_gate(
            root, tmp_path / "result.json", protocol_path
        )
