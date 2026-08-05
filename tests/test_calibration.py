import numpy as np
import pytest

from bayesian_phystwin import BinaryCalibrationMetrics, binary_calibration_metrics


def test_binary_calibration_metrics_for_ranked_predictions() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([True, True, False, False]),
        n_bins=5,
    )

    assert metrics.count == 4
    assert metrics.positive_rate == pytest.approx(0.5)
    assert metrics.brier_score == pytest.approx(0.025)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert 0.0 <= metrics.expected_calibration_error <= 1.0


def test_auc_is_undefined_for_one_class() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.8, 0.9]),
        np.array([True, True]),
    )

    assert metrics.roc_auc is None


def test_exact_numeric_binary_labels_are_supported() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.9, 0.1]),
        np.array([1, 0]),
    )

    assert metrics.roc_auc == pytest.approx(1.0)


@pytest.mark.parametrize("probability", [[-0.01], [1.01], [np.nan], [np.inf]])
def test_invalid_probabilities_fail_closed(probability: list[float]) -> None:
    with pytest.raises(ValueError, match="probability"):
        binary_calibration_metrics(np.asarray(probability), np.array([True]))


@pytest.mark.parametrize(
    "target",
    [
        np.array([2]),
        np.array([-1]),
        np.array([0.5]),
        np.array([np.nan]),
        np.array(["True"]),
    ],
)
def test_nonbinary_targets_fail_closed(target: np.ndarray) -> None:
    with pytest.raises(ValueError, match="target"):
        binary_calibration_metrics(np.array([0.5]), target)


@pytest.mark.parametrize("n_bins", [True, 0, -1, 1.5])
def test_invalid_bin_counts_fail_closed(n_bins: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        binary_calibration_metrics(
            np.array([0.5]),
            np.array([True]),
            n_bins=n_bins,  # type: ignore[arg-type]
        )


def test_string_encoded_probability_fails_closed() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        binary_calibration_metrics(
            np.array(["0.5"]),
            np.array([True]),
        )


def test_metrics_contract_rejects_inconsistent_scalars() -> None:
    with pytest.raises(ValueError, match="positive_rate"):
        BinaryCalibrationMetrics(
            count=1,
            positive_rate=1.1,
            brier_score=0.0,
            log_loss=0.0,
            expected_calibration_error=0.0,
            roc_auc=None,
        )


def test_tied_probabilities_use_average_ranks() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.5, 0.5, 0.1, 0.9]),
        np.array([True, False, False, True]),
    )

    assert metrics.roc_auc == pytest.approx(0.875)


def test_metrics_as_dict_preserves_validated_values() -> None:
    metrics = binary_calibration_metrics(
        np.array([0.9, 0.1]),
        np.array([True, False]),
    )

    assert metrics.as_dict()["count"] == 2
    assert metrics.as_dict()["roc_auc"] == pytest.approx(1.0)


@pytest.mark.parametrize("count", [0, True, 1.5])
def test_metrics_contract_rejects_invalid_counts(count: object) -> None:
    with pytest.raises(ValueError, match="count"):
        BinaryCalibrationMetrics(
            count=count,  # type: ignore[arg-type]
            positive_rate=0.5,
            brier_score=0.25,
            log_loss=0.5,
            expected_calibration_error=0.1,
            roc_auc=0.5,
        )


@pytest.mark.parametrize("log_loss", [-0.1, np.nan, "0.5"])
def test_metrics_contract_rejects_invalid_log_loss(log_loss: object) -> None:
    with pytest.raises(ValueError, match="log_loss"):
        BinaryCalibrationMetrics(
            count=1,
            positive_rate=0.5,
            brier_score=0.25,
            log_loss=log_loss,  # type: ignore[arg-type]
            expected_calibration_error=0.1,
            roc_auc=0.5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("positive_rate", True),
        ("positive_rate", "0.5"),
        ("brier_score", 1.1),
        ("expected_calibration_error", np.nan),
        ("roc_auc", -0.1),
    ],
)
def test_metrics_contract_rejects_invalid_unit_interval_values(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "count": 1,
        "positive_rate": 0.5,
        "brier_score": 0.25,
        "log_loss": 0.5,
        "expected_calibration_error": 0.1,
        "roc_auc": 0.5,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        BinaryCalibrationMetrics(**kwargs)  # type: ignore[arg-type]


def test_target_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        binary_calibration_metrics(
            np.array([0.5, 0.5]),
            np.array([True]),
        )


def test_nonvector_probabilities_fail_closed() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        binary_calibration_metrics(
            np.array([[0.5]]),
            np.array([[True]]),
        )


def test_empty_probabilities_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least one"):
        binary_calibration_metrics(
            np.array([], dtype=float),
            np.array([], dtype=bool),
        )
