import numpy as np
import pytest

from bayesian_phystwin.pokeflex_released_robot_checkpoint import (
    PokeFlexReleasedRobotCheckpoint,
    prepare_pokeflex_robot_history,
)
from bayesian_phystwin.pokeflex_robot_fusion import (
    PokeFlexRobotFusionConfig,
    pokeflex_robot_fusion_candidates,
    pokeflex_robot_fusion_features,
)


def _record(force: tuple[float, float, float], position: np.ndarray) -> dict:
    transform = np.eye(4)
    transform[:3, 3] = position
    return {"forces": [*force, 0.0, 0.0, 0.0], "T_WT": transform.tolist()}


def test_robot_history_matches_official_normalization() -> None:
    records = [
        _record((10.0, -20.0, 30.0), np.array([1.1, 2.0, 3.0])),
        _record((20.0, 0.0, -10.0), np.array([0.9, 2.2, 3.0])),
    ]

    result = prepare_pokeflex_robot_history(
        records,
        template_center_m=np.array([1.0, 2.0, 3.0]),
        template_scale_m=0.2,
    )

    np.testing.assert_allclose(
        result.values,
        [
            [0.1, -0.2, 0.3, 0.5, 0.0, 0.0],
            [0.2, 0.0, -0.1, -0.5, 1.0, 0.0],
        ],
        atol=1e-6,
    )
    assert result.maximum_force_n == 100.0


def test_injected_robot_models_preserve_template_identity() -> None:
    torch = pytest.importorskip("torch")

    class ForceEncoder:
        def forward(self, value):
            assert value.shape == (5, 6)
            return torch.zeros((5, 32), dtype=value.dtype, device=value.device)

    class Attention:
        def forward(self, value):
            assert value.shape == (5, 1, 32)
            return torch.zeros((1, 64), dtype=value.dtype, device=value.device)

    class Decoder:
        def forward(self, _feature, template):
            return template

    template = np.array(
        [
            [-0.1, -0.1, -0.1],
            [0.1, -0.1, -0.1],
            [-0.1, 0.1, -0.1],
            [-0.1, -0.1, 0.1],
            [0.1, 0.1, 0.1],
        ]
    )
    adapter = PokeFlexReleasedRobotCheckpoint(
        template,
        force_encoder=ForceEncoder(),
        attention_model=Attention(),
        decoder=Decoder(),
        torch_module=torch,
        device="cpu",
    )
    records = [_record((1.0, 2.0, 3.0), np.zeros(3)) for _ in range(5)]

    prediction = adapter.predict_from_records(records)

    np.testing.assert_allclose(prediction.vertices_m, template, atol=1e-7)
    assert prediction.history_frame_count == 5


def test_robot_checkpoint_rejects_noncausal_history_length() -> None:
    torch = pytest.importorskip("torch")
    template = np.eye(3)
    adapter = PokeFlexReleasedRobotCheckpoint(
        template,
        force_encoder=object(),
        attention_model=object(),
        decoder=object(),
        torch_module=torch,
        device="cpu",
    )

    with pytest.raises(ValueError, match="exactly five"):
        adapter.predict_from_records(
            [_record((1.0, 0.0, 0.0), np.zeros(3)) for _ in range(6)]
        )


def test_robot_fusion_has_byte_exact_fallback() -> None:
    baseline = np.arange(18, dtype=np.float64).reshape(6, 3) / 100.0
    robot = baseline + 0.01

    result = pokeflex_robot_fusion_candidates(
        baseline,
        robot,
        config=PokeFlexRobotFusionConfig(scales=(0.0, 0.1)),
    )

    np.testing.assert_array_equal(result["robot_convex_scale_0"], baseline)
    np.testing.assert_allclose(
        result["robot_convex_scale_0.1"],
        baseline + 0.001,
    )


def test_robot_fusion_features_do_not_depend_on_target_state() -> None:
    template = np.zeros((4, 3), dtype=np.float64)
    baseline = template + np.array([0.0, 0.01, 0.0])
    robot = template + np.array([0.0, 0.02, 0.0])
    records = [
        _record((0.0, 3.0, 0.0), np.array([0.0, 0.0, 0.0])),
        _record((0.0, 5.0, 0.0), np.array([0.0, 0.002, 0.0])),
    ]

    features = pokeflex_robot_fusion_features(
        baseline,
        robot,
        template,
        records,
    )

    assert set(features) == {
        "baseline_deformation_rms_m",
        "robot_deformation_rms_m",
        "model_disagreement_rms_m",
        "deformation_cosine",
        "force_norm_n",
        "force_delta_norm_n",
        "tool_step_m",
    }
    assert features["deformation_cosine"] == pytest.approx(1.0)
    assert features["force_norm_n"] == pytest.approx(5.0)
    assert features["force_delta_norm_n"] == pytest.approx(2.0)
    assert features["tool_step_m"] == pytest.approx(0.002)
