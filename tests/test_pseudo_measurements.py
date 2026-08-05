import numpy as np
import pytest

from bayesian_phystwin import (
    PseudoMeasurementBatch,
    ReliabilityConfig,
    ReliabilityResult,
    reliability_weighted_loss,
    score_reliability,
)


def test_low_confidence_occluded_outlier_is_downweighted() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        predicted=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        variance=1.0,
        confidence=[1.0, 0.2],
        occluded=[False, True],
        flow_inconsistency=[0.0, 1.0],
    )

    result = score_reliability(batch, ReliabilityConfig(residual_scale=1.0))

    assert result.weights[0] > 0.99
    assert result.weights[1] < 0.01
    assert result.inflated_variance[1, 0] > result.inflated_variance[0, 0]


def test_weighted_loss_is_finite() -> None:
    batch = PseudoMeasurementBatch(
        observed=np.ones((4, 3)),
        predicted=np.zeros((4, 3)),
        variance=np.full((4, 3), 0.5),
    )

    loss = reliability_weighted_loss(batch)

    assert np.isfinite(loss)
    assert loss >= 0.0


def test_weighted_loss_applies_reliability_once() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[2.0]],
        predicted=[[0.0]],
        variance=1.0,
        confidence=[0.5],
    )

    result = score_reliability(batch)
    loss = reliability_weighted_loss(batch)

    assert result.weights[0] == pytest.approx(0.5)
    assert result.inflated_variance[0, 0] == pytest.approx(2.0)
    assert loss == pytest.approx(2.0)


def test_shape_mismatch_raises() -> None:
    batch = PseudoMeasurementBatch(
        observed=np.ones((4, 3)),
        predicted=np.ones((5, 3)),
    )

    with pytest.raises(ValueError):
        score_reliability(batch)


def test_residual_gating_is_opt_in() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[10.0]],
        predicted=[[0.0]],
        confidence=[1.0],
    )

    cue_only = score_reliability(batch)
    residual_gated = score_reliability(batch, ReliabilityConfig(residual_scale=1.0))

    assert cue_only.weights[0] > 0.99
    assert residual_gated.weights[0] == pytest.approx(1e-3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed", [[np.nan]]),
        ("observed", [[np.inf]]),
        ("predicted", [[np.nan]]),
        ("predicted", [[-np.inf]]),
    ],
)
def test_nonfinite_measurements_fail_closed(
    field: str,
    value: list[list[float]],
) -> None:
    kwargs: dict[str, object] = {
        "observed": [[0.0]],
        "predicted": [[0.0]],
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        score_reliability(PseudoMeasurementBatch(**kwargs))


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((0, 1), "at least one value"),
        ((1, 0), "at least one value"),
    ],
)
def test_empty_measurement_axes_fail_closed(
    shape: tuple[int, int],
    message: str,
) -> None:
    batch = PseudoMeasurementBatch(
        observed=np.empty(shape),
        predicted=np.empty(shape),
    )

    with pytest.raises(ValueError, match=message):
        score_reliability(batch)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("confidence", [-0.01], "confidence"),
        ("confidence", [1.01], "confidence"),
        ("confidence", [np.nan], "confidence"),
        ("occluded", [2], "occluded"),
        ("occluded", [np.nan], "occluded"),
        ("boundary_distance", [-0.01], "boundary_distance"),
        ("boundary_distance", [np.nan], "boundary_distance"),
        ("flow_inconsistency", [-0.01], "flow_inconsistency"),
        ("flow_inconsistency", [np.inf], "flow_inconsistency"),
    ],
)
def test_malformed_reliability_cues_fail_closed(
    field: str,
    value: list[float],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "observed": [[0.0]],
        "predicted": [[0.0]],
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        score_reliability(PseudoMeasurementBatch(**kwargs))


def test_exact_numeric_occlusion_mask_and_infinite_boundary_are_supported() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0], [0.0]],
        predicted=[[0.0], [0.0]],
        occluded=[0, 1],
        boundary_distance=[np.inf, np.inf],
    )

    result = score_reliability(batch)

    assert result.weights == pytest.approx([1.0, 0.05])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_weight", np.nan),
        ("confidence_power", np.inf),
        ("residual_scale", np.nan),
        ("boundary_scale", np.inf),
        ("flow_scale", np.nan),
        ("occlusion_weight", np.inf),
        ("covariance_inflation_at_min_weight", np.nan),
        ("min_weight", True),
    ],
)
def test_nonfinite_or_boolean_config_values_fail_closed(
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        ReliabilityConfig(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed", [["0.0"]]),
        ("predicted", [["0.0"]]),
        ("variance", "1.0"),
        ("confidence", ["0.5"]),
    ],
)
def test_string_encoded_numeric_inputs_fail_closed(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "observed": [[0.0]],
        "predicted": [[0.0]],
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        score_reliability(PseudoMeasurementBatch(**kwargs))


def test_reliability_result_defensively_owns_read_only_arrays() -> None:
    weights = np.array([0.5])
    inflated = np.array([[2.0]])
    residual = np.array([1.0])

    result = ReliabilityResult(
        weights=weights,
        inflated_variance=inflated,
        residual_norm=residual,
    )
    weights[0] = 0.2
    inflated[0, 0] = 3.0
    residual[0] = 4.0

    assert result.weights[0] == pytest.approx(0.5)
    assert result.inflated_variance[0, 0] == pytest.approx(2.0)
    assert result.residual_norm[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="read-only"):
        result.weights[0] = 0.1


def test_one_dimensional_measurements_fail_closed() -> None:
    batch = PseudoMeasurementBatch(observed=[0.0], predicted=[0.0])

    with pytest.raises(ValueError, match="shape"):
        score_reliability(batch)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_weight", 0.0, "min_weight"),
        ("min_weight", 1.1, "min_weight"),
        ("confidence_power", -1.0, "confidence_power"),
        ("residual_scale", 0.0, "residual_scale"),
        ("boundary_scale", 0.0, "boundary_scale"),
        ("flow_scale", 0.0, "flow_scale"),
        ("occlusion_weight", -0.1, "occlusion_weight"),
        ("occlusion_weight", 1.1, "occlusion_weight"),
        (
            "covariance_inflation_at_min_weight",
            0.9,
            "covariance inflation cap",
        ),
        ("min_weight", "0.1", "finite real"),
    ],
)
def test_out_of_domain_config_values_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ReliabilityConfig(**{field: value})


def test_scalar_cues_are_broadcast() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0], [0.0]],
        predicted=[[0.0], [0.0]],
        confidence=0.5,  # type: ignore[arg-type]
        occluded=True,  # type: ignore[arg-type]
    )

    result = score_reliability(batch)

    assert result.weights == pytest.approx([0.025, 0.025])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence", [0.5, 0.5]),
        ("occluded", [False, True]),
    ],
)
def test_cue_length_mismatch_fails_closed(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "observed": [[0.0]],
        "predicted": [[0.0]],
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match="scalar or shape"):
        score_reliability(PseudoMeasurementBatch(**kwargs))


def test_string_occlusion_mask_fails_closed() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0]],
        predicted=[[0.0]],
        occluded=["False"],  # type: ignore[list-item]
    )

    with pytest.raises(ValueError, match="occluded"):
        score_reliability(batch)


def test_per_measurement_variance_is_broadcast_across_coordinates() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0, 0.0], [0.0, 0.0]],
        predicted=[[0.0, 0.0], [0.0, 0.0]],
        variance=[1.0, 2.0],
    )

    result = score_reliability(batch)

    assert result.inflated_variance == pytest.approx(np.array([[1.0, 1.0], [2.0, 2.0]]))


@pytest.mark.parametrize(
    "variance",
    [
        [1.0, 2.0, 3.0],
        0.0,
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_variance_fails_closed(variance: object) -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0], [0.0]],
        predicted=[[0.0], [0.0]],
        variance=variance,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="variance"):
        score_reliability(batch)


def test_wrong_config_type_fails_closed() -> None:
    batch = PseudoMeasurementBatch(observed=[[0.0]], predicted=[[0.0]])

    with pytest.raises(TypeError, match="ReliabilityConfig"):
        score_reliability(batch, config=object())  # type: ignore[arg-type]


def test_unrepresentable_residual_fails_closed() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[1e308]],
        predicted=[[-1e308]],
    )

    with pytest.raises(ValueError, match="residual must"):
        score_reliability(batch)


def test_unrepresentable_residual_norm_fails_closed() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[1e308]],
        predicted=[[0.0]],
    )

    with pytest.raises(ValueError, match="residual norm"):
        score_reliability(batch)


def test_tampered_config_cannot_produce_nonfinite_weights() -> None:
    batch = PseudoMeasurementBatch(
        observed=[[0.0]],
        predicted=[[0.0]],
        boundary_distance=[1.0],
    )
    config = ReliabilityConfig()
    object.__setattr__(config, "boundary_scale", np.nan)

    with pytest.raises(ValueError, match="weights"):
        score_reliability(batch, config)


def test_effective_sample_size_handles_zero_and_nonzero_weights() -> None:
    zero = ReliabilityResult(
        weights=np.array([0.0, 0.0]),
        inflated_variance=np.ones((2, 1)),
        residual_norm=np.zeros(2),
    )
    nonzero = ReliabilityResult(
        weights=np.array([0.5, 0.5]),
        inflated_variance=np.ones((2, 1)),
        residual_norm=np.zeros(2),
    )

    assert zero.effective_sample_size == 0.0
    assert nonzero.effective_sample_size == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("weights", "inflated", "residual", "message"),
    [
        (np.array([]), np.empty((0, 1)), np.array([]), "nonempty vector"),
        (np.array([[0.5]]), np.ones((1, 1)), np.array([0.0]), "nonempty vector"),
        (np.array([0.5]), np.ones((1, 1)), np.array([0.0, 0.0]), "same shape"),
        (np.array([0.5]), np.array([1.0]), np.array([0.0]), "matrix"),
        (np.array([0.5]), np.empty((1, 0)), np.array([0.0]), "coordinate"),
        (np.array([-0.1]), np.ones((1, 1)), np.array([0.0]), "weights"),
        (np.array([np.nan]), np.ones((1, 1)), np.array([0.0]), "weights"),
        (np.array([0.5]), np.zeros((1, 1)), np.array([0.0]), "inflated"),
        (np.array([0.5]), np.ones((1, 1)), np.array([-0.1]), "residual"),
    ],
)
def test_reliability_result_rejects_malformed_arrays(
    weights: np.ndarray,
    inflated: np.ndarray,
    residual: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ReliabilityResult(
            weights=weights,
            inflated_variance=inflated,
            residual_norm=residual,
        )


def test_reliability_result_rejects_string_arrays() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        ReliabilityResult(
            weights=np.array(["0.5"]),
            inflated_variance=np.ones((1, 1)),
            residual_norm=np.zeros(1),
        )


def test_weighted_loss_fails_closed_on_nonfinite_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesian_phystwin.pseudo_measurements as pseudo_measurements

    def fake_score_reliability(
        batch: PseudoMeasurementBatch,
        config: ReliabilityConfig | None = None,
    ) -> ReliabilityResult:
        del batch, config
        return ReliabilityResult(
            weights=np.array([1.0]),
            inflated_variance=np.array([[np.nextafter(0.0, 1.0)]]),
            residual_norm=np.array([1.0]),
        )

    monkeypatch.setattr(
        pseudo_measurements,
        "score_reliability",
        fake_score_reliability,
    )
    batch = PseudoMeasurementBatch(observed=[[1e154]], predicted=[[0.0]])

    with pytest.raises(ValueError, match="weighted loss"):
        reliability_weighted_loss(batch)
