from __future__ import annotations

import numpy as np
import pytest

import bayesian_phystwin.deform360_registered_covariance_source_v1 as source


class _BrokenArray:
    def __array__(self, dtype: object = None) -> np.ndarray:
        del dtype
        raise TypeError("cannot convert")


@pytest.mark.parametrize("value", ["", "line\nbreak"])
def test_canonical_string_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(ValueError):
        source._canonical_string(value, name="source_unit_id")


def test_owned_mean_rejects_non_arrays() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        source._owned_mean([[[0.0, 0.0, 0.0]]], name="mean")


def test_owned_mean_rejects_non_float64_arrays() -> None:
    value = np.zeros((1, 1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="dtype float64"):
        source._owned_mean(value, name="mean")


def test_owned_mean_rejects_invalid_geometry() -> None:
    value = np.zeros((1, 1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"shape \(H, N, 3\)"):
        source._owned_mean(value, name="mean")


def test_owned_mean_rejects_changed_shape() -> None:
    value = np.zeros((1, 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="shape changed"):
        source._owned_mean(value, name="mean", shape=(2, 1, 3))


def test_owned_mean_rejects_nonfinite_arrays() -> None:
    value = np.zeros((1, 1, 3), dtype=np.float64)
    value[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite and C-contiguous"):
        source._owned_mean(value, name="mean")


def test_owned_covariance_rejects_non_arrays() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        source._owned_covariance([], mean_shape=(1, 1, 3))


def test_owned_covariance_rejects_wrong_dtype() -> None:
    value = np.zeros((1, 1, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="float64 with shape"):
        source._owned_covariance(value, mean_shape=(1, 1, 3))


def test_owned_covariance_rejects_nonfinite_arrays() -> None:
    value = np.eye(3, dtype=np.float64)[None, None, ...]
    value[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite and C-contiguous"):
        source._owned_covariance(value, mean_shape=(1, 1, 3))


def test_owned_covariance_rejects_asymmetry() -> None:
    value = np.eye(3, dtype=np.float64)[None, None, ...]
    value[0, 0, 0, 1] = 1.0

    with pytest.raises(ValueError, match="symmetric"):
        source._owned_covariance(value, mean_shape=(1, 1, 3))


def test_owned_covariance_rejects_negative_eigenvalues() -> None:
    value = np.eye(3, dtype=np.float64)[None, None, ...]
    value[0, 0, 0, 0] = -1.0

    with pytest.raises(ValueError, match="positive semidefinite"):
        source._owned_covariance(value, mean_shape=(1, 1, 3))


def test_source_inputs_normalize_conversion_failures() -> None:
    with pytest.raises(ValueError, match="real numeric values"):
        source._source_inputs(
            _BrokenArray(),
            np.ones((1, 1), dtype=bool),
            end_frame=1,
        )


def test_source_inputs_reject_nonnumeric_values() -> None:
    residual = np.full((1, 1, 3), "not-numeric")

    with pytest.raises(ValueError, match="real numeric values"):
        source._source_inputs(
            residual,
            np.ones((1, 1), dtype=bool),
            end_frame=1,
        )


def test_source_inputs_reject_invalid_geometry() -> None:
    with pytest.raises(ValueError, match=r"shape \(T, N>=1, 3\)"):
        source._source_inputs(
            np.zeros((1, 3), dtype=np.float64),
            np.ones((1, 1), dtype=bool),
            end_frame=1,
        )


def test_source_inputs_reject_nonfinite_values() -> None:
    residual = np.zeros((1, 1, 3), dtype=np.float64)
    residual[0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="must be finite"):
        source._source_inputs(
            residual,
            np.ones((1, 1), dtype=bool),
            end_frame=1,
        )


def test_source_inputs_require_boolean_validity() -> None:
    residual = np.zeros((1, 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="matching Boolean matrix"):
        source._source_inputs(
            residual,
            np.ones((1, 1), dtype=np.int64),
            end_frame=1,
        )


def test_source_inputs_reject_end_frame_outside_history() -> None:
    residual = np.zeros((1, 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="outside residual_history_m"):
        source._source_inputs(
            residual,
            np.ones((1, 1), dtype=bool),
            end_frame=2,
        )


def test_bins_require_integer_vector() -> None:
    with pytest.raises(ValueError, match="integer vector"):
        source._bins(np.asarray([0.0]), count=1)


def test_bins_reject_unknown_horizon_indices() -> None:
    with pytest.raises(ValueError, match="indices 0, 1, 2"):
        source._bins(np.asarray([3], dtype=np.int64), count=1)
