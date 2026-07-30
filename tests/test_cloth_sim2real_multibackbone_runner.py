from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _runner():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "development"
        / "run_cloth_sim2real_backbone_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cloth_sim2real_multibackbone_runner",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTriangles:
    def array(self) -> np.ndarray:
        return np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)


class _FakeTopology:
    triangles = _FakeTriangles()


class _FakeSquareGravity:
    def getObject(self, name: str) -> _FakeTopology:
        assert name == "topo"
        return _FakeTopology()


class _FakeEnvironment:
    SquareGravity = _FakeSquareGravity()

    def __init__(self) -> None:
        self.reset_count = 0
        self.start_arguments = None

    def reset(self) -> None:
        self.reset_count += 1

    def start_simulation(self, *args) -> None:
        self.start_arguments = args


def test_default_simulator_preserves_mujoco_v1_behavior() -> None:
    runner = _runner()

    args = runner._parser().parse_args(
        [
            "--benchmark-code-root",
            "/benchmark",
            "--case-id",
            "chequered_rag_0/dynamic",
            "--output",
            "/tmp/baseline.npz",
        ]
    )

    assert args.simulator == "mujoco3"


def test_sofa_uses_scene_initialization_and_topology_faces() -> None:
    runner = _runner()
    environment = _FakeEnvironment()
    trajectory = [np.zeros(6), np.ones(6)]

    runner._initialize_environment(environment, "sofa", trajectory, 1)
    faces = runner._extract_faces(environment, {}, "sofa")

    assert environment.reset_count == 0
    assert environment.start_arguments[:2] == (trajectory, 1)
    assert np.array_equal(
        faces,
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )


def test_mujoco_uses_reset_and_info_faces() -> None:
    runner = _runner()
    environment = _FakeEnvironment()
    expected = np.asarray([[0, 1, 2]], dtype=np.int32)

    runner._initialize_environment(environment, "mujoco3", [], 0)
    faces = runner._extract_faces(
        environment,
        {"faces": expected},
        "mujoco3",
    )

    assert environment.reset_count == 1
    assert environment.start_arguments is None
    assert np.array_equal(faces, expected)
    assert faces.dtype == np.int64


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("chequered_rag_0/dynamic", ("chequered_rag_0", "dynamic")),
        ("linen_rag_2/quasi_static", ("linen_rag_2", "quasi_static")),
    ],
)
def test_case_parts_are_strict(case_id: str, expected: tuple[str, str]) -> None:
    runner = _runner()
    assert runner._case_parts(case_id) == expected


def test_unknown_task_is_rejected() -> None:
    runner = _runner()
    with pytest.raises(ValueError, match="unknown task"):
        runner._case_parts("chequered_rag_0/future")
