from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.material_backend_v1 import (
    describe_material_backend_profiles,
    resolve_material_backend_profile,
)
from bayesian_phystwin.material_trajectory_backend_v1 import (
    RUNTIME_FILENAME,
    get_material_backend_profile,
    validate_material_trajectory_backend,
)
from bayesian_phystwin.material_trajectory_producer_v1 import (
    MaterialTrajectoryReplayV1,
    produce_material_trajectory_backend,
)
from bayesian_phystwin.material_trajectory_replay_adapters_v1 import (
    DolfinxDisplacementReplayV1,
    PyElasticaRodReplayV1,
)

NEW_BACKENDS = (
    "fenicsx-fem-v1",
    "pyelastica-cosserat-rod-v1",
)


class _ArrayFacade:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.calls: list[str] = []

    def block_until_ready(self) -> _ArrayFacade:
        self.calls.append("block")
        return self

    def detach(self) -> _ArrayFacade:
        self.calls.append("detach")
        return self

    def cpu(self) -> _ArrayFacade:
        self.calls.append("cpu")
        return self

    def numpy(self) -> np.ndarray:
        self.calls.append("numpy")
        return self.value


class _Rod:
    def __init__(self, positions: object) -> None:
        self.position_collection = positions


class _Replay:
    def __init__(self) -> None:
        self.positions = np.stack(
            (
                np.linspace(0.0, 0.08, 5, dtype=np.float32),
                np.zeros(5, dtype=np.float32),
                np.zeros(5, dtype=np.float32),
            ),
            axis=1,
        )
        self.pending = np.zeros_like(self.positions)

    def synchronize(self) -> None:
        return None

    def get_material_positions_m(self) -> np.ndarray:
        return self.positions

    def step(self) -> None:
        self.positions += self.pending
        self.pending.fill(0.0)


def _driven_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    concrete = cast(_Replay, replay)
    concrete.pending[:, 2] = np.linspace(
        0.0,
        0.001 * (transition_index + 1),
        len(concrete.positions),
        dtype=np.float32,
    )


def _zero_control(
    transition_index: int,
    replay: MaterialTrajectoryReplayV1,
) -> None:
    del transition_index
    cast(_Replay, replay).pending.fill(0.0)


def test_new_backend_profiles_extend_the_drake_registry_tail() -> None:
    records = cast(
        list[dict[str, object]],
        describe_material_backend_profiles()["profiles"],
    )
    assert [item["profile_id"] for item in records[-3:]] == [
        "drake-fem-v1",
        "fenicsx-fem-v1",
        "pyelastica-cosserat-rod-v1",
    ]
    assert [item["priority"] for item in records[-3:]] == [8, 9, 10]

    expected = {
        "fenicsx-fem-v1": (
            "FEniCS/dolfinx",
            "distributed-finite-element-method",
            "global-geometry-node-index",
        ),
        "pyelastica-cosserat-rod-v1": (
            "GazzolaLab/PyElastica",
            "cosserat-rod-dynamics",
            "rod-node-index",
        ),
    }
    for profile_id, facts in expected.items():
        canonical = resolve_material_backend_profile(profile_id)
        transport = get_material_backend_profile(profile_id)
        assert canonical.profile_id == profile_id
        assert canonical.transport == "material-trajectory-v1"
        assert canonical.spec.maturity == "experimental"
        assert (
            transport.engine_repository,
            transport.solver_family,
            transport.identity_kind,
        ) == facts


def test_dolfinx_replay_adds_registered_displacement_and_copies_facades() -> None:
    reference = np.arange(15, dtype=np.float32).reshape(5, 3) / 100.0
    displacement = np.zeros_like(reference)
    reference_facade = _ArrayFacade(reference)
    displacement_facade = _ArrayFacade(displacement)
    events: list[str] = []
    replay = DolfinxDisplacementReplayV1(
        reference_positions_m=reference_facade,
        displacement_callback=lambda: displacement_facade,
        step_callback=lambda: events.append("step"),
        synchronize_callback=lambda: events.append("synchronize"),
        context="dolfinx-scene",
    )

    replay.synchronize()
    np.testing.assert_array_equal(replay.get_material_positions_m(), reference)
    displacement[:, 2] = np.linspace(0.0, 0.01, len(displacement))
    np.testing.assert_allclose(
        replay.get_material_positions_m(),
        reference + displacement,
    )
    replay.step()

    assert replay.context == "dolfinx-scene"
    assert events == ["synchronize", "step"]
    assert reference_facade.calls == ["block", "detach", "cpu", "numpy"]
    assert displacement_facade.calls == [
        "block",
        "detach",
        "cpu",
        "numpy",
        "block",
        "detach",
        "cpu",
        "numpy",
    ]


@pytest.mark.parametrize(
    ("reference", "message"),
    (
        (np.zeros((5, 2), dtype=np.float32), r"shape \(S,3\)"),
        (np.zeros((5, 3), dtype=np.int64), "floating point"),
        (
            np.array(
                [[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "non-finite",
        ),
    ),
)
def test_dolfinx_replay_rejects_invalid_reference_geometry(
    reference: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DolfinxDisplacementReplayV1(
            reference_positions_m=reference,
            displacement_callback=lambda: np.zeros_like(reference),
            step_callback=lambda: None,
        )


def test_dolfinx_replay_rejects_noncallable_callbacks() -> None:
    with pytest.raises(TypeError, match="displacement_callback must be callable"):
        DolfinxDisplacementReplayV1(
            reference_positions_m=np.zeros((2, 3), dtype=np.float32),
            displacement_callback=cast(Any, None),
            step_callback=lambda: None,
        )


@pytest.mark.parametrize(
    ("displacement", "message"),
    (
        (np.zeros((4, 3), dtype=np.float32), "registered reference shape"),
        (np.zeros((5, 3), dtype=np.float64), "registered reference dtype"),
        (
            np.array(
                [
                    [np.inf, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "non-finite",
        ),
    ),
)
def test_dolfinx_replay_rejects_invalid_displacements(
    displacement: np.ndarray,
    message: str,
) -> None:
    replay = DolfinxDisplacementReplayV1(
        reference_positions_m=np.zeros((5, 3), dtype=np.float32),
        displacement_callback=lambda: displacement,
        step_callback=lambda: None,
    )
    with pytest.raises(ValueError, match=message):
        replay.get_material_positions_m()


def test_pyelastica_replay_transposes_native_rod_nodes() -> None:
    expected = np.arange(15, dtype=np.float64).reshape(5, 3) / 100.0
    facade = _ArrayFacade(expected.T)
    events: list[str] = []
    replay = PyElasticaRodReplayV1(
        rod=_Rod(facade),
        step_callback=lambda: events.append("step"),
        context="rod-scene",
    )

    assert replay.synchronize() is None
    portable = replay.get_material_positions_m()
    replay.step()

    np.testing.assert_array_equal(portable, expected)
    assert portable.flags.c_contiguous
    assert replay.context == "rod-scene"
    assert events == ["step"]
    assert facade.calls == ["block", "detach", "cpu", "numpy"]


def test_pyelastica_replay_rejects_missing_surface_and_noncallables() -> None:
    with pytest.raises(TypeError, match="must expose position_collection"):
        PyElasticaRodReplayV1(rod=object(), step_callback=lambda: None)

    with pytest.raises(TypeError, match="step_callback must be callable"):
        PyElasticaRodReplayV1(
            rod=_Rod(np.zeros((3, 2), dtype=np.float64)),
            step_callback=cast(Any, None),
        )


@pytest.mark.parametrize(
    ("positions", "message"),
    (
        (np.zeros((2, 5), dtype=np.float64), r"shape \(3,N\)"),
        (np.zeros((3, 5), dtype=np.int64), "floating point"),
        (
            np.array(
                [[0.0, np.nan], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float64,
            ),
            "non-finite",
        ),
    ),
)
def test_pyelastica_replay_rejects_invalid_native_node_matrices(
    positions: np.ndarray,
    message: str,
) -> None:
    replay = PyElasticaRodReplayV1(
        rod=_Rod(positions),
        step_callback=lambda: None,
    )
    with pytest.raises(ValueError, match=message):
        replay.get_material_positions_m()


@pytest.mark.parametrize("backend_kind", NEW_BACKENDS)
def test_new_profiles_publish_the_common_material_trajectory_contract(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    output = tmp_path / backend_kind
    artifact = produce_material_trajectory_backend(
        output_dir=output,
        backend_kind=backend_kind,
        replay_factory=_Replay,
        driven_control=_driven_control,
        zero_action_control=_zero_control,
        frame_count=4,
        material_query_indices=np.array([4, 0, 2], dtype=np.int64),
        action_support=np.array([1.0, 0.0, 0.5], dtype=np.float32),
        engine_revision="a" * 40,
        engine_version="test-engine-v1",
        producer_repository="example/backend-producer",
        producer_revision="b" * 40,
        producer_version="producer-v1",
        producer_artifacts={"producer.py": "c" * 64},
        topology_sha256="d" * 64,
        device="cpu",
        device_name="contract-test-device",
        time_step_s=1.0 / 120.0,
        scene_id="registered-material-scene-v1",
        model_kind="deformable-material",
        constitutive_model="registered-model",
        integrator="backend-native-integrator",
        solver="backend-native-solver",
        substeps=2,
        engine_parameters={"density_kg_m3": 1000.0},
    )

    assert artifact == validate_material_trajectory_backend(output)
    assert artifact["backend_kind"] == backend_kind
    runtime = json.loads(
        (output / "provenance" / RUNTIME_FILENAME).read_text(encoding="utf-8")
    )
    profile = get_material_backend_profile(backend_kind)
    assert runtime["engine_repository"] == profile.engine_repository
    assert runtime["solver_family"] == profile.solver_family
    assert runtime["identity_kind"] == profile.identity_kind
