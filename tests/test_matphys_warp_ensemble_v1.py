import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.matphys_warp_ensemble_v1 import (
    hierarchical_trajectory_ensemble_arrays,
    load_matphys_spring_ensemble,
    load_registered_replay_graph,
    trajectory_ensemble_arrays,
)
from scripts.remote.run_matphys_warp_ensemble_v1 import (
    EXPECTED_OFFICIAL_WARP_VERSION,
    EXPECTED_REPLAY_RUNTIME,
    _validate_independent_reference,
    _validate_replay_runtime,
    _validate_warp_runtime,
)


def _graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
        dtype=np.float32,
    )
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    np.savez_compressed(
        path,
        vertices=points,
        springs=edges,
        rest_lengths=np.array([0.01, 0.01], dtype=np.float32),
        masses=np.ones(3, dtype=np.float32),
        contact_anchor_indices=np.array([0, 2], dtype=np.int64),
    )
    return points, edges


def test_registered_replay_graph_preserves_object_and_rebuilds_controls(
    tmp_path: Path,
) -> None:
    points, edges = _graph(tmp_path / "graph.npz")
    controls = np.array(
        [
            [0.0, 0.001, 0.0],
            [0.0, 0.002, 0.0],
            [0.02, 0.001, 0.0],
            [0.02, 0.002, 0.0],
        ],
        dtype=np.float32,
    )

    replay = load_registered_replay_graph(
        tmp_path / "graph.npz",
        expected_points_m=points,
        expected_edges=edges,
        controller_reference_m=controls,
        controller_radius_m=0.011,
        controller_patch_size=2,
    )

    np.testing.assert_array_equal(replay.vertices[:3], points)
    np.testing.assert_array_equal(replay.springs[:2], edges)
    assert replay.num_object_springs == 2
    assert replay.num_controller_springs == 4
    assert replay.controller_group_count == 2
    assert replay.vertices.shape == (7, 3)
    assert np.all(replay.springs[2:, 0] >= len(points))


def test_registered_replay_graph_rejects_matphys_topology_mismatch(
    tmp_path: Path,
) -> None:
    points, edges = _graph(tmp_path / "graph.npz")
    controls = np.array([[0.0, 0.001, 0.0], [0.02, 0.001, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="edges differ"):
        load_registered_replay_graph(
            tmp_path / "graph.npz",
            expected_points_m=points,
            expected_edges=edges[::-1],
            controller_reference_m=controls,
            controller_radius_m=0.011,
            controller_patch_size=1,
        )


def test_spring_ensemble_binds_member_count_and_graph(tmp_path: Path) -> None:
    archive = tmp_path / "springs.npz"
    np.savez_compressed(
        archive,
        incumbent_spring_y_pa=np.array([1000.0, 2000.0], dtype=np.float32),
        member_spring_y_pa=np.array(
            [[900.0, 1800.0], [1100.0, 2200.0]], dtype=np.float32
        ),
        graph_points_m=np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
            dtype=np.float32,
        ),
        graph_edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
    )

    result = load_matphys_spring_ensemble(archive, expected_member_count=2)

    assert result.member_spring_y_pa.shape == (2, 2)
    assert result.graph_edges.dtype == np.int64
    with pytest.raises(ValueError, match="registered ensemble"):
        load_matphys_spring_ensemble(archive, expected_member_count=3)


def test_trajectory_arrays_keep_incumbent_mean_separate() -> None:
    incumbent: np.ndarray = np.zeros((2, 3, 3), dtype=np.float32)
    first = np.full_like(incumbent, 0.01)
    second = np.full_like(incumbent, 0.03)

    arrays = trajectory_ensemble_arrays(incumbent, np.stack((first, second)))

    assert arrays["incumbent_trajectory_m"].tobytes() == incumbent.tobytes()
    np.testing.assert_allclose(arrays["member_mean_trajectory_m"], 0.02)
    np.testing.assert_allclose(arrays["member_covariance_m2"][..., 0, 0], 0.0001)
    np.testing.assert_array_equal(arrays["unique_member_indices"], [0, 1])


def test_hierarchical_moments_separate_member_and_replay_variance() -> None:
    shape = (2, 1, 3)
    incumbent = np.stack(
        (
            np.full(shape, -0.001, dtype=np.float32),
            np.full(shape, 0.001, dtype=np.float32),
        )
    )
    members = np.stack(
        (
            np.stack(
                (
                    np.full(shape, 0.009, dtype=np.float32),
                    np.full(shape, 0.011, dtype=np.float32),
                )
            ),
            np.stack(
                (
                    np.full(shape, 0.029, dtype=np.float32),
                    np.full(shape, 0.031, dtype=np.float32),
                )
            ),
        )
    )

    arrays = hierarchical_trajectory_ensemble_arrays(incumbent, members)

    np.testing.assert_allclose(arrays["incumbent_replay_mean_m"], 0.0)
    np.testing.assert_allclose(arrays["member_mean_trajectory_m"], 0.02)
    np.testing.assert_allclose(
        arrays["between_member_covariance_m2"][..., 0, 0], 0.0001
    )
    np.testing.assert_allclose(
        arrays["within_member_replay_covariance_m2"][..., 0, 0],
        0.000001,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        arrays["member_total_covariance_m2"][..., 0, 0],
        0.000101,
        rtol=1e-6,
    )


def test_official_warp_runtime_is_exactly_pinned() -> None:
    assert _validate_warp_runtime("1.16.0") == EXPECTED_OFFICIAL_WARP_VERSION
    with pytest.raises(ValueError, match="runtime version changed"):
        _validate_warp_runtime("1.15.0")


def test_official_replay_runtime_is_exactly_pinned() -> None:
    observed = dict(EXPECTED_REPLAY_RUNTIME)

    assert _validate_replay_runtime(observed) == observed
    for name, changed in (
        ("python_version", "3.12.3"),
        ("numpy_version", "2.2.0"),
        ("torch_version", "2.5.0+cu121"),
        ("torch_cuda_version", "12.4"),
        ("warp_version", "1.15.0"),
    ):
        with pytest.raises(ValueError, match="replay runtime changed"):
            _validate_replay_runtime({**observed, name: changed})


def test_independent_reference_binds_producer_inputs_and_dynamics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "official_runner.py"
    runner.write_text("# frozen official producer\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.remote.run_matphys_warp_ensemble_v1.EXPECTED_REFERENCE_RUNNER_SHA256",
        hashlib.sha256(runner.read_bytes()).hexdigest(),
    )
    data = tmp_path / "data.pkl"
    data.write_bytes(b"data")
    graph = tmp_path / "graph.npz"
    graph.write_bytes(b"graph")
    trajectory = tmp_path / "official_phystwin_trajectory.npz"
    trajectory.write_bytes(b"trajectory")
    result_path = tmp_path / "official_phystwin_smoke.json"
    result = {
        "passed": True,
        "source_only_smoke": True,
        "official_phystwin_revision": "official-revision",
        "config_sha256": "config-sha",
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "config_overrides": {
            "controller_max_neighbours": 1,
            "controller_radius": 0.03,
            "dashpot_damping": 100.0,
            "drag_damping": 10.0,
            "init_spring_Y": 10000.0,
        },
        "canonical_reusable_graph": {
            "file_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
            "controller_patch_size_per_anchor": 16,
        },
        "support_dynamics": {
            "mode": "official-ground",
            "reverse_factor": -1.0,
            "uses_official_cuda_graph": True,
        },
        "realized_actuation": {"controller_displacement_scale": 1.0},
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    kwargs = {
        "result_path": result_path,
        "trajectory_path": trajectory,
        "runner_path": runner,
        "data_path": data,
        "config_sha256": "config-sha",
        "official_revision": "official-revision",
        "registered_graph_path": graph,
        "controller_max_neighbours": 1,
        "controller_radius_m": 0.03,
        "controller_patch_size": 16,
        "init_spring_y": 10000.0,
        "drag_damping": 10.0,
        "dashpot_damping": 100.0,
    }

    assert _validate_independent_reference(**kwargs) == result

    result["config_overrides"]["init_spring_Y"] = 9000.0
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="reference dynamics changed"):
        _validate_independent_reference(**kwargs)
