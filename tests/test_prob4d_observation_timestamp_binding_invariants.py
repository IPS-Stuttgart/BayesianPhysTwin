"""Adversarial invariants for direct Prob4D timestamp binding construction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.prob4d_observation_timestamps import (
    Prob4DObservationTimestampBindingV1,
)


def _binding(**overrides: Any) -> Prob4DObservationTimestampBindingV1:
    values: dict[str, Any] = {
        "observation_artifact_id": "1" * 64,
        "bundle_manifest_sha256": "2" * 64,
        "timestamp_lineage_artifact_id": "3" * 64,
        "clock_domain": "camera-hardware-clock",
        "time_scale": "device-monotonic",
        "timestamp_source": "camera-packet-timestamp",
        "factor_ids": ("factor-0", "factor-1"),
        "factor_frame_indices": np.asarray([0, 1], dtype=np.int64),
        "factor_timestamps_ns": np.asarray(
            [1_000_000_000, 2_000_000_000],
            dtype=np.int64,
        ),
        "conditional_timestamp_std_ns": np.asarray(
            [1_000_000.0, 2_000_000.0],
            dtype=np.float64,
        ),
        "row_factor_indices": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "row_timestamps_s": np.asarray([1.0, 1.0, 2.0, 2.0]),
        "row_conditional_timestamp_std_s": np.asarray([0.001, 0.001, 0.002, 0.002]),
        "shared_clock_offset_prior_artifact_id": "4" * 64,
        "metadata": {"protocol": "direct-binding-invariants-v1"},
    }
    values.update(overrides)
    return Prob4DObservationTimestampBindingV1(**values)


def test_direct_binding_accepts_only_canonical_derived_rows() -> None:
    binding = _binding()

    with pytest.raises(ValueError, match="row_timestamps_s do not match"):
        replace(
            binding,
            row_timestamps_s=np.asarray([1.0, 1.0, 2.0, 2.001]),
            binding_id=None,
        )
    with pytest.raises(
        ValueError,
        match="row_conditional_timestamp_std_s does not match",
    ):
        replace(
            binding,
            row_conditional_timestamp_std_s=np.asarray([0.001, 0.001, 0.002, 0.003]),
            binding_id=None,
        )


def test_direct_binding_rejects_negative_factor_coordinates() -> None:
    binding = _binding()

    with pytest.raises(ValueError, match="factor_frame_indices must be nonnegative"):
        replace(
            binding,
            factor_frame_indices=np.asarray([-1, 1], dtype=np.int64),
            binding_id=None,
        )
    with pytest.raises(ValueError, match="factor_timestamps_ns must be nonnegative"):
        replace(
            binding,
            factor_timestamps_ns=np.asarray([-1, 2_000_000_000], dtype=np.int64),
            binding_id=None,
        )


def test_direct_binding_uses_one_canonical_nanosecond_conversion() -> None:
    factor_timestamps = np.asarray(
        [1_000_000_001, 2_000_000_003],
        dtype=np.int64,
    )
    factor_jitter = np.asarray([1_000_001.0, 2_000_003.0], dtype=np.float64)
    row_factors = np.asarray([1, 0, 1], dtype=np.int64)
    row_timestamps = np.asarray(
        factor_timestamps[row_factors],
        dtype=np.float64,
    )
    row_timestamps *= 1e-9
    row_jitter = np.asarray(factor_jitter[row_factors], dtype=np.float64)
    row_jitter *= 1e-9

    binding = _binding(
        factor_timestamps_ns=factor_timestamps,
        conditional_timestamp_std_ns=factor_jitter,
        row_factor_indices=row_factors,
        row_timestamps_s=row_timestamps,
        row_conditional_timestamp_std_s=row_jitter,
    )

    np.testing.assert_array_equal(binding.row_timestamps_s, row_timestamps)
    np.testing.assert_array_equal(
        binding.row_conditional_timestamp_std_s,
        row_jitter,
    )
