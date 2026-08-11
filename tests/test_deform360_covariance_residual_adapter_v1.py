from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_covariance_residual_adapter_v1 import (
    Deform360ResidualHistoryAdapterConfigV1,
    Deform360ResidualHistoryIdentityV1,
    adapt_deform360_covariance_residual_history_v1,
)


def _identity() -> Deform360ResidualHistoryIdentityV1:
    return Deform360ResidualHistoryIdentityV1(
        object_id="source-object",
        session_id="source-session",
        material_id="source-material",
        coordinate_frame="source-metric-frame",
        provider_camera_ids=("cam-00", "cam-01"),
        scoring_camera_ids=("cam-02", "cam-03"),
        provider_artifact_ids=("provider-reconstruction",),
        scoring_artifact_ids=("scoring-reconstruction",),
    )


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(
        [
            [[0.001, 0.002, 0.003], [0.010, 0.020, 0.030], [1.0, 1.0, 1.0]],
            [[0.002, 0.003, 0.004], [9.0, 9.0, 9.0], [0.100, 0.200, 0.300]],
            [[0.003, 0.004, 0.005], [0.011, 0.021, 0.031], [9.0, 9.0, 9.0]],
            [[0.004, 0.005, 0.006], [0.012, 0.022, 0.032], [0.101, 0.201, 0.301]],
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [
            [True, True, False],
            [True, False, True],
            [True, True, False],
            [True, True, True],
        ],
        dtype=bool,
    )
    residual[~valid] = np.nan
    physical = np.arange(27, dtype=np.float64).reshape(3, 3, 3) / 100.0
    return residual, valid, np.ascontiguousarray(physical)


def _provider(
    residual: np.ndarray,
    valid: np.ndarray,
    steps: tuple[int, ...],
) -> np.ndarray:
    assert residual.shape == (4, 3, 3)
    assert valid.shape == (4, 3)
    assert steps == (1, 2, 3)
    assert np.all(residual[~valid] == 0.0)
    covariance = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for horizon, step in enumerate(steps):
        covariance[horizon, :, 0, 0] = step * 1.0e-6
        covariance[horizon, :, 1, 1] = step * 2.0e-6
        covariance[horizon, :, 2, 2] = step * 3.0e-6
    return covariance


def test_adapter_preserves_validity_mean_and_exact_track_fallback() -> None:
    residual, valid, physical = _fixture()
    prediction = adapt_deform360_covariance_residual_history_v1(
        residual,
        valid,
        physical,
        horizon_labels=("early", "middle", "late"),
        horizon_steps=(1, 2, 3),
        identity=_identity(),
        covariance_provider=_provider,
    )

    np.testing.assert_array_equal(prediction.supported_track_mask, [True, True, False])
    np.testing.assert_array_equal(
        prediction.mean_m[:, 0],
        physical[:, 0] + residual[3, 0],
    )
    np.testing.assert_array_equal(
        prediction.mean_m[:, 1],
        physical[:, 1] + residual[3, 1],
    )
    np.testing.assert_array_equal(prediction.mean_m[:, 2], physical[:, 2])
    provider_valid = valid & np.asarray([True, True, False])[None, :]
    provider_residual = np.zeros_like(residual)
    provider_residual[provider_valid] = residual[provider_valid]
    expected = _provider(provider_residual, provider_valid, (1, 2, 3))
    expected[:, 2] = 0.0
    expected *= np.asarray([8.0, 16.0, 16.0])[:, None, None, None]
    np.testing.assert_array_equal(prediction.covariance_m2, expected)
    assert not prediction.covariance_m2.flags.writeable
    assert not prediction.supported_track_mask.flags.writeable
    assert prediction.record.provider_status == "success"
    assert prediction.record.valid_observation_count_by_track == (4, 3, 2)
    assert prediction.record.supported_track_count == 2
    assert prediction.record.unsupported_track_count == 1
    assert prediction.record.validity_preserved
    assert not prediction.record.nearest_fill_used
    assert prediction.record.unsupported_tracks_use_physical_fallback
    assert prediction.record.mean_object_identity_preserved
    assert prediction.record.covariance_hybrid_artifact_id is not None
    assert prediction.record.target_payload_opened is False
    assert prediction.record.target_outcomes_opened is False
    assert prediction.record.claim_authorized is False


def test_masked_values_do_not_change_prediction_or_content_identity() -> None:
    residual, valid, physical = _fixture()
    first = adapt_deform360_covariance_residual_history_v1(
        residual,
        valid,
        physical,
        horizon_labels=("early", "middle", "late"),
        horizon_steps=(1, 2, 3),
        identity=_identity(),
        covariance_provider=_provider,
    )
    alternate = residual.copy()
    alternate[~valid] = 1.0e100
    second = adapt_deform360_covariance_residual_history_v1(
        alternate,
        valid,
        physical,
        horizon_labels=("early", "middle", "late"),
        horizon_steps=(1, 2, 3),
        identity=_identity(),
        covariance_provider=_provider,
    )

    np.testing.assert_array_equal(second.mean_m, first.mean_m)
    np.testing.assert_array_equal(second.covariance_m2, first.covariance_m2)
    assert second.record.validity_sha256 == first.record.validity_sha256
    assert (
        second.record.canonical_residual_sha256
        == first.record.canonical_residual_sha256
    )
    assert second.record.artifact_id == first.record.artifact_id


def test_provider_failure_returns_exact_reference_and_zero_covariance() -> None:
    residual, valid, physical = _fixture()

    def failing_provider(
        _residual: np.ndarray,
        _valid: np.ndarray,
        _steps: tuple[int, ...],
    ) -> np.ndarray:
        raise RuntimeError("deliberate provider failure")

    prediction = adapt_deform360_covariance_residual_history_v1(
        residual,
        valid,
        physical,
        horizon_labels=("early", "middle", "late"),
        horizon_steps=(1, 2, 3),
        identity=_identity(),
        covariance_provider=failing_provider,
    )

    np.testing.assert_array_equal(
        prediction.mean_m[:, 0],
        physical[:, 0] + residual[3, 0],
    )
    np.testing.assert_array_equal(
        prediction.mean_m[:, 1],
        physical[:, 1] + residual[3, 1],
    )
    np.testing.assert_array_equal(prediction.mean_m[:, 2], physical[:, 2])
    assert np.count_nonzero(prediction.covariance_m2) == 0
    assert prediction.record.provider_status == "fallback-provider-failure"
    assert prediction.record.provider_error_type == "RuntimeError"
    assert prediction.record.provider_error_sha256 is not None
    assert prediction.record.covariance_hybrid_artifact_id is None


def test_no_supported_track_preserves_physical_object_and_skips_provider() -> None:
    residual, valid, physical = _fixture()
    insufficient = np.zeros_like(valid)

    def forbidden_provider(
        _residual: np.ndarray,
        _valid: np.ndarray,
        _steps: tuple[int, ...],
    ) -> np.ndarray:
        raise AssertionError("provider must not be called")

    prediction = adapt_deform360_covariance_residual_history_v1(
        residual,
        insufficient,
        physical,
        horizon_labels=("early", "middle", "late"),
        horizon_steps=(1, 2, 3),
        identity=_identity(),
        covariance_provider=forbidden_provider,
    )

    assert prediction.mean_m is physical
    assert np.count_nonzero(prediction.covariance_m2) == 0
    assert prediction.record.provider_status == "fallback-no-supported-tracks"
    assert prediction.record.supported_track_count == 0
    assert prediction.record.unsupported_track_count == 3


def test_identity_rejects_shared_camera_or_reconstruction_artifact() -> None:
    values = {
        "object_id": "object",
        "session_id": "session",
        "material_id": "material",
        "coordinate_frame": "frame",
        "provider_camera_ids": ("cam-00",),
        "scoring_camera_ids": ("cam-01",),
        "provider_artifact_ids": ("provider",),
        "scoring_artifact_ids": ("scoring",),
    }
    with pytest.raises(ValueError, match="cameras must be disjoint"):
        Deform360ResidualHistoryIdentityV1(
            **{**values, "scoring_camera_ids": ("cam-00",)}
        )
    with pytest.raises(ValueError, match="artifacts must be disjoint"):
        Deform360ResidualHistoryIdentityV1(
            **{**values, "scoring_artifact_ids": ("provider",)}
        )


@pytest.mark.parametrize(
    ("labels", "steps", "match"),
    [
        (("middle", "early", "late"), (1, 2, 3), "ordered"),
        (("early", "middle", "late"), (1, 1, 3), "strictly increasing"),
        (("early", "unknown", "late"), (1, 2, 3), "unregistered"),
    ],
)
def test_adapter_rejects_horizon_misalignment(
    labels: tuple[str, str, str],
    steps: tuple[int, int, int],
    match: str,
) -> None:
    residual, valid, physical = _fixture()
    with pytest.raises(ValueError, match=match):
        adapt_deform360_covariance_residual_history_v1(
            residual,
            valid,
            physical,
            horizon_labels=labels,
            horizon_steps=steps,
            identity=_identity(),
            covariance_provider=_provider,
        )


def test_non_psd_provider_fails_closed_to_zero_covariance() -> None:
    residual, valid, physical = _fixture()

    def non_psd_provider(
        _residual: np.ndarray,
        _valid: np.ndarray,
        _steps: tuple[int, ...],
    ) -> np.ndarray:
        covariance = np.zeros((3, 3, 3, 3), dtype=np.float64)
        covariance[..., 0, 0] = -1.0
        return covariance

    prediction = adapt_deform360_covariance_residual_history_v1(
        residual,
        valid,
        physical,
        horizon_labels=("early", "middle", "late"),
        horizon_steps=(1, 2, 3),
        identity=_identity(),
        covariance_provider=non_psd_provider,
    )

    assert prediction.record.provider_status == "fallback-provider-failure"
    assert prediction.record.provider_error_type == "ValueError"
    assert np.count_nonzero(prediction.covariance_m2) == 0


def test_config_retains_registered_scales_and_positive_support() -> None:
    assert Deform360ResidualHistoryAdapterConfigV1().horizon_scales == (
        ("early", 8.0),
        ("middle", 16.0),
        ("late", 16.0),
    )
    with pytest.raises(ValueError, match="positive"):
        Deform360ResidualHistoryAdapterConfigV1(
            minimum_valid_observations_per_track=0
        )
    with pytest.raises(ValueError, match="order"):
        Deform360ResidualHistoryAdapterConfigV1(
            horizon_scales=(
                ("middle", 16.0),
                ("early", 8.0),
                ("late", 16.0),
            )
        )
