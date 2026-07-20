import numpy as np
import pytest

from bayesian_phystwin.pokeflex_released_checkpoint import (
    PokeFlexReleasedCheckpoint,
    prepare_pokeflex_checkpoint_point_cloud,
)


def test_checkpoint_preprocessing_is_bounded_and_deterministic() -> None:
    generator = np.random.default_rng(7)
    first = generator.normal(size=(140, 3)) * 0.01
    second = generator.normal(size=(140, 3)) * 0.01

    result = prepare_pokeflex_checkpoint_point_cloud(
        (first, second),
        template_center_m=np.zeros(3),
        template_scale_m=0.1,
        maximum_points=80,
        initial_voxel_size_m=0.002,
        voxel_step_m=0.002,
    )
    repeated = prepare_pokeflex_checkpoint_point_cloud(
        (first, second),
        template_center_m=np.zeros(3),
        template_scale_m=0.1,
        maximum_points=80,
        initial_voxel_size_m=0.002,
        voxel_step_m=0.002,
    )

    assert result.input_point_count == 280
    assert result.retained_point_count <= 80
    assert result.voxel_size_m > 0.0
    assert result.points.shape == (80, 3)
    np.testing.assert_array_equal(result.points, repeated.points)


def test_checkpoint_preprocessing_does_not_gain_samples_from_duplicates() -> None:
    points = np.array(
        [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
        dtype=np.float64,
    )
    result = prepare_pokeflex_checkpoint_point_cloud(
        (np.repeat(points, 50, axis=0),),
        template_center_m=np.zeros(3),
        template_scale_m=0.1,
        maximum_points=16,
        initial_voxel_size_m=0.005,
    )

    assert result.retained_point_count == 3
    actual = result.points[:3][np.lexsort(result.points[:3].T[::-1])]
    expected = (points / 0.1)[np.lexsort((points / 0.1).T[::-1])]
    np.testing.assert_allclose(actual, expected)


def test_injected_checkpoint_models_preserve_template_identity() -> None:
    torch = pytest.importorskip("torch")

    class Encoder:
        encoder: "Encoder"

        def __init__(self) -> None:
            self.encoder = self

        def forward(self, value):
            return torch.zeros((len(value), 256), dtype=value.dtype, device=value.device)

    class Attention:
        def forward(self, value):
            assert value.shape == (5, 1, 256)
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
    adapter = PokeFlexReleasedCheckpoint(
        template,
        pointcloud_encoder=Encoder(),
        attention_model=Attention(),
        decoder=Decoder(),
        torch_module=torch,
        device="cpu",
    )
    encoded = []
    prepared = []
    for _ in range(5):
        feature, metadata = adapter.encode_frame((template, template.copy()))
        encoded.append(feature)
        prepared.append(metadata)

    prediction = adapter.predict_from_encoded_history(encoded, prepared)

    np.testing.assert_allclose(prediction.vertices_m, template, atol=1e-7)
    assert prediction.history_retained_point_counts == (10, 10, 10, 10, 10)
