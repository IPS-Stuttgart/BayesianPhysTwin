import numpy as np
import pytest

from bayesian_phystwin.phystwin_state_injection import (
    _released_self_collision_for_case,
    _rollout_initial,
    _rollout_restart,
    _trajectory_error,
    estimate_endpoint_velocity_delta,
)


class _ArrayTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)

    def contiguous(self) -> "_ArrayTensor":
        return self

    def detach(self) -> "_ArrayTensor":
        return self

    def cpu(self) -> "_ArrayTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeWarp:
    vec3 = object()

    @staticmethod
    def to_torch(value: np.ndarray) -> _ArrayTensor:
        return _ArrayTensor(value)

    @staticmethod
    def from_torch(
        value: _ArrayTensor,
        *,
        dtype: object,
        requires_grad: bool,
    ) -> np.ndarray:
        del dtype, requires_grad
        return value.value

    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def capture_launch(graph) -> None:
        graph()


class _FakeTorch:
    float32 = np.float32

    @staticmethod
    def as_tensor(value: np.ndarray, *, dtype: object, device: str) -> _ArrayTensor:
        del dtype, device
        return _ArrayTensor(value)


class _FakeState:
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.wp_x = np.asarray(position).copy()
        self.wp_v = np.asarray(velocity).copy()


class _FakeSimulator:
    def __init__(self) -> None:
        self.wp_init_vertices = np.zeros((2, 3), dtype=np.float32)
        self.wp_init_velocities = np.zeros((2, 3), dtype=np.float32)
        self.wp_states = [
            _FakeState(self.wp_init_vertices, self.wp_init_velocities),
            _FakeState(self.wp_init_vertices, self.wp_init_velocities),
        ]
        self.object_collision_flag = False
        self.init_pure_inference: list[bool] = []
        self.target_pure_inference: list[bool] = []
        self.forward_graph = self._forward

    def set_init_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        *,
        pure_inference: bool = False,
    ) -> None:
        self.init_pure_inference.append(pure_inference)
        self.wp_states[0].wp_x = np.asarray(position).copy()
        self.wp_states[0].wp_v = np.asarray(velocity).copy()

    def set_controller_target(
        self,
        frame: int,
        *,
        pure_inference: bool = False,
    ) -> None:
        del frame
        self.target_pure_inference.append(pure_inference)

    def update_collision_graph(self) -> None:
        raise AssertionError("collision graph should not be updated in this test")

    def _forward(self) -> None:
        self.wp_states[-1].wp_x = self.wp_states[0].wp_x + 1.0
        self.wp_states[-1].wp_v = self.wp_states[0].wp_v + 0.5


def test_deterministic_vertex_spring_adjacency_has_fixed_sign_order() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("warp")
    from bayesian_phystwin._phystwin_warp_backend import (
        deterministic_vertex_spring_adjacency,
    )

    offsets, spring_ids, signs = deterministic_vertex_spring_adjacency(
        np.array([[0, 1], [2, 0], [3, 1]], dtype=np.int32),
        num_object_points=3,
    )

    np.testing.assert_array_equal(offsets, [0, 2, 4, 5])
    np.testing.assert_array_equal(spring_ids, [0, 1, 0, 2, 1])
    np.testing.assert_array_equal(signs, [1, -1, -1, -1, 1])


def test_velocity_delta_recovers_linear_correction_motion() -> None:
    frame_dt = 0.05
    velocity = np.array([[0.2, -0.1, 0.05], [-0.3, 0.0, 0.4]])
    offset = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    time = frame_dt * np.arange(4)
    history = offset[None] + time[:, None, None] * velocity[None]

    estimated = estimate_endpoint_velocity_delta(history, frame_dt=frame_dt)

    np.testing.assert_allclose(estimated, velocity, atol=1e-12)


def test_velocity_delta_rejects_invalid_history() -> None:
    with pytest.raises(ValueError, match="T>=2"):
        estimate_endpoint_velocity_delta(np.zeros((1, 3, 3)), frame_dt=0.1)
    with pytest.raises(ValueError, match="frame_dt"):
        estimate_endpoint_velocity_delta(np.zeros((2, 3, 3)), frame_dt=0.0)


def test_trajectory_error_uses_vector_and_coordinate_units() -> None:
    reference = np.zeros((2, 1, 3))
    candidate = np.ones((2, 1, 3))

    result = _trajectory_error(reference, candidate)

    assert result["coordinate_rmse_m"] == pytest.approx(1.0)
    assert result["vector_rmse_m"] == pytest.approx(np.sqrt(3.0))
    assert result["maximum_norm_m"] == pytest.approx(np.sqrt(3.0))


def test_initial_rollout_uses_pure_inference_for_every_state_copy() -> None:
    simulator = _FakeSimulator()

    positions, velocities = _rollout_initial(
        simulator,
        _FakeWarp(),
        frame_count=3,
    )

    assert positions.shape == (3, 2, 3)
    assert velocities.shape == (3, 2, 3)
    assert simulator.init_pure_inference == [True, True, True]
    assert simulator.target_pure_inference == [True, True]


def test_restart_rollout_uses_pure_inference_for_every_state_copy() -> None:
    simulator = _FakeSimulator()

    continuation = _rollout_restart(
        simulator,
        _FakeTorch(),
        _FakeWarp(),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        start_frame=1,
        stop_frame=3,
        device="cuda:0",
    )

    assert continuation.shape == (2, 2, 3)
    assert simulator.init_pure_inference == [True, True, True]
    assert simulator.target_pure_inference == [True, True]


@pytest.mark.parametrize(
    ("case_name", "expected"),
    (
        ("double_lift_cloth_1", True),
        ("cloth_blue_fold", True),
        ("double_push_package", True),
        ("single_lift_sloth", False),
        ("single_push_rope", False),
    ),
)
def test_released_self_collision_matches_phystwin_case_rule(
    case_name: str, expected: bool
) -> None:
    assert _released_self_collision_for_case(case_name) is expected
