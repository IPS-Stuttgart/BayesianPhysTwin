from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.causal4d_belief_provider_v1 import (
    infer_fixed_bayesian_anchor_endpoint,
)


def _provider_inputs() -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros((3, 2, 3), dtype=np.float64)
    residual[:, 0, 0] = (0.001, 0.002, 0.003)
    residual[:, 1, 1] = (-0.001, -0.002, -0.003)
    valid = np.asarray(
        (
            (True, True),
            (False, True),
            (True, False),
        ),
        dtype=bool,
    )
    return residual, valid


def test_fixed_anchor_provider_rejects_complex_residual_without_lossy_cast() -> None:
    residual, valid = _provider_inputs()
    complex_residual = residual.astype(np.complex128)
    complex_residual[0, 0, 0] += 0.5j

    with pytest.raises(ValueError, match="residual_m must contain real numeric values"):
        infer_fixed_bayesian_anchor_endpoint(
            complex_residual,
            valid,
            end_frame=len(residual),
        )


@pytest.mark.parametrize(
    "invalid_validity",
    (
        np.asarray(((1, 1), (0, 2), (1, 0)), dtype=np.int64),
        np.asarray(((1.0, 1.0), (0.0, np.nan), (1.0, 0.0))),
        np.asarray(((1, 1), (-1, 1), (1, 0)), dtype=np.int64),
        np.asarray((("1", "1"), ("0", "1"), ("1", "0")), dtype=object),
        np.ones((3, 2), dtype=np.complex128),
    ),
)
def test_fixed_anchor_provider_rejects_nonbinary_validity(
    invalid_validity: np.ndarray,
) -> None:
    residual, _ = _provider_inputs()

    with pytest.raises(
        ValueError,
        match="valid must contain booleans or exact 0/1 values",
    ):
        infer_fixed_bayesian_anchor_endpoint(
            residual,
            invalid_validity,
            end_frame=len(residual),
        )


@pytest.mark.parametrize("dtype", (np.uint8, np.int64, np.float32, np.float64))
def test_fixed_anchor_provider_accepts_exact_zero_one_validity(dtype: type) -> None:
    residual, valid = _provider_inputs()
    expected = infer_fixed_bayesian_anchor_endpoint(
        residual,
        valid,
        end_frame=len(residual),
    )
    actual = infer_fixed_bayesian_anchor_endpoint(
        residual,
        valid.astype(dtype),
        end_frame=len(residual),
    )

    np.testing.assert_array_equal(actual.mean_m, expected.mean_m)
    np.testing.assert_array_equal(actual.variance_m2, expected.variance_m2)
    np.testing.assert_array_equal(
        actual.final_nominal_probability,
        expected.final_nominal_probability,
    )
    np.testing.assert_array_equal(actual.update_count, expected.update_count)
