from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin._causal4d_provider_v2_replay as replay_impl
import bayesian_phystwin.causal4d_provider_v2 as provider


def _write_pickle(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_files(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    case_dir = tmp_path / "double_unit"
    data_path = case_dir / "final_data.pkl"
    optimal_path = case_dir / "optimal.pkl"
    baseline_path = case_dir / "baseline.pkl"
    data = {
        "object_points": np.zeros((4, 3, 3), dtype=np.float32),
        "object_visibilities": np.ones((4, 3), dtype=bool),
        "object_motions_valid": np.ones((4, 3), dtype=bool),
        "controller_points": np.asarray(
            [
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[-1.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
                [[-1.0, 0.2, 0.0], [1.0, 0.2, 0.0]],
                [[-1.0, 0.3, 0.0], [1.0, 0.3, 0.0]],
            ],
            dtype=np.float32,
        ),
        "surface_points": np.asarray([[0.0, 0.5, 0.0]], dtype=np.float32),
        "interior_points": np.asarray([[0.0, -0.5, 0.0]], dtype=np.float32),
    }
    optimal = {
        "object_radius": 2.0,
        "object_max_neighbours": 3,
        "controller_radius": 2.0,
        "controller_max_neighbours": 2,
    }
    baseline = np.zeros((4, 5, 3), dtype=np.float32)
    digests = {
        "final_data": _write_pickle(data_path, data),
        "optimal_params": _write_pickle(optimal_path, optimal),
        "baseline_trajectory": _write_pickle(baseline_path, baseline),
    }
    return data_path, optimal_path, baseline_path, digests


def test_manifest_exposes_canonical_contract_fingerprint() -> None:
    encoded = json.dumps(
        provider.CAUSAL4D_PROVIDER_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == (
        provider.CAUSAL4D_PROVIDER_CONTRACT_SHA256
    )
    manifest = provider.causal4d_provider_manifest(provider_revision="abc123")
    assert manifest["schema_version"] == 2
    assert manifest["provider_revision"] == "abc123"
    assert manifest["metadata"]["contract_sha256"] == (
        provider.CAUSAL4D_PROVIDER_CONTRACT_SHA256
    )
    assert manifest["artifact_schema_versions"]["PhysTwinCase"] == 2
    assert manifest["artifact_schema_versions"]["PhysTwinSpringGraph"] == 2


def test_case_graph_and_controller_layout_are_provider_owned(tmp_path: Path) -> None:
    data_path, optimal_path, baseline_path, digests = _case_files(tmp_path)
    case = provider.load_official_phystwin_case(
        data_path,
        optimal_path,
        baseline_path,
        expected_sha256=digests,
    )
    assert case.case_name == "double_unit"
    assert case.frame_count == 4
    assert case.original_count == 3
    assert case.structure_points_m.shape == (5, 3)
    assert case.baseline_trajectory_m is not None
    assert not case.object_points_m.flags.writeable

    graph = provider.build_phystwin_spring_graph(
        case.structure_points_m,
        case.controller_points_m[0],
        config=case.graph_config,
    )
    assert graph.vertices.shape == (7, 3)
    assert graph.num_object_points == 5
    assert not graph.vertices.flags.writeable
    assert not graph.springs.flags.writeable

    layout = provider.released_controller_layout(
        case.case_name, case.controller_points_m[0]
    )
    assert layout.hand_count == 2
    assert set(layout.group_ids.tolist()) == {0, 1}
    assert not layout.group_ids.flags.writeable


def test_digest_bound_case_loading_rejects_changed_bytes(tmp_path: Path) -> None:
    data_path, optimal_path, baseline_path, digests = _case_files(tmp_path)
    data_path.write_bytes(data_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        provider.load_official_phystwin_case(
            data_path,
            optimal_path,
            baseline_path,
            expected_sha256=digests,
        )


def test_digest_mapping_must_cover_every_requested_pickle(tmp_path: Path) -> None:
    data_path, optimal_path, baseline_path, digests = _case_files(tmp_path)
    with pytest.raises(ValueError, match="baseline_trajectory"):
        provider.load_official_phystwin_case(
            data_path,
            optimal_path,
            baseline_path,
            expected_sha256={
                "final_data": digests["final_data"],
                "optimal_params": digests["optimal_params"],
            },
        )


def test_case_loader_rejects_cross_array_shape_mismatch(tmp_path: Path) -> None:
    data_path, optimal_path, baseline_path, _ = _case_files(tmp_path)
    with data_path.open("rb") as stream:
        data = pickle.load(stream)
    data["object_visibilities"] = np.ones((3, 3), dtype=bool)
    _write_pickle(data_path, data)
    with pytest.raises(ValueError, match="object_visibilities"):
        provider.load_official_phystwin_case(
            data_path,
            optimal_path,
            baseline_path,
        )


def test_graph_contract_rejects_out_of_range_endpoints() -> None:
    with pytest.raises(ValueError, match="spring endpoint"):
        provider.PhysTwinSpringGraph(
            vertices=np.zeros((2, 3)),
            springs=np.asarray([[0, 2]]),
            rest_lengths=np.ones(1),
            masses=np.ones(2),
            num_object_springs=1,
            num_object_points=2,
        )


def test_case_replay_factory_forwards_validated_case(
    monkeypatch, tmp_path: Path
) -> None:
    data_path, optimal_path, baseline_path, _ = _case_files(tmp_path)
    case = provider.load_official_phystwin_case(
        data_path, optimal_path, baseline_path
    )
    graph = provider.build_phystwin_spring_graph(
        case.structure_points_m,
        case.controller_points_m[0],
        config=case.graph_config,
    )
    captured: dict[str, object] = {}
    sentinel = object()

    def factory(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(replay_impl, "create_official_replay_provider", factory)
    result = provider.create_official_case_replay_provider(
        tmp_path,
        case,
        tmp_path / "checkpoint.pt",
        graph,
        dt=0.01,
        num_substeps=4,
        self_collision=False,
        spring_parameterization="grouped",
        device="cpu",
    )
    assert result is sentinel
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["num_surface_points"] == case.num_surface_points
    assert kwargs["original_count"] == case.original_count
    assert kwargs["spring_parameterization"] == "grouped"
