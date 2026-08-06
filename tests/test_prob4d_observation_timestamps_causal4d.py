from __future__ import annotations

import math

import numpy as np
import pytest

from bayesian_phystwin.prob4d_observation_timestamps import (
    Prob4DObservationTimestampBindingV1,
)
from causal4d.observation_clock_offset_prior import (
    OBSERVATION_TIME_CORRECTION_CONVENTION,
    ObservationClockOffsetPriorV1,
)


def _causal4d_prior() -> ObservationClockOffsetPriorV1:
    offsets = (-0.011, -0.010, -0.009)
    sample_standard_deviation = 0.001
    grid_standard_deviation = 0.001 / math.sqrt(12.0)
    predictive_standard_deviation = math.sqrt(
        (1.0 + 1.0 / len(offsets)) * sample_standard_deviation**2
        + grid_standard_deviation**2
    )
    return ObservationClockOffsetPriorV1(
        clock_domain="camera-hardware-clock",
        reference_clock_domain="actuator-command-clock",
        time_scale="device-monotonic",
        source_revision="a" * 40,
        source_artifact_ids=("1" * 64, "2" * 64, "3" * 64),
        execution_ids=("source-01", "source-02", "source-03"),
        source_offsets_s=offsets,
        source_group_count=len(offsets),
        mean_offset_s=-0.010,
        sample_standard_deviation_s=sample_standard_deviation,
        grid_quantization_standard_deviation_s=grid_standard_deviation,
        minimum_predictive_standard_deviation_s=5e-4,
        predictive_standard_deviation_s=predictive_standard_deviation,
    )


def _binding(prior_artifact_id: str) -> Prob4DObservationTimestampBindingV1:
    return Prob4DObservationTimestampBindingV1(
        observation_artifact_id="4" * 64,
        bundle_manifest_sha256="5" * 64,
        timestamp_lineage_artifact_id="6" * 64,
        clock_domain="camera-hardware-clock",
        time_scale="device-monotonic",
        timestamp_source="camera-packet-timestamp",
        factor_ids=("factor-0",),
        factor_frame_indices=np.asarray([0], dtype=np.int64),
        factor_timestamps_ns=np.asarray([1_000_000_000], dtype=np.int64),
        conditional_timestamp_std_ns=np.asarray([1_000_000.0]),
        row_factor_indices=np.asarray([0], dtype=np.int64),
        row_timestamps_s=np.asarray([1.0]),
        row_conditional_timestamp_std_s=np.asarray([0.001]),
        shared_clock_offset_prior_artifact_id=prior_artifact_id,
        metadata={"protocol": "causal4d-clock-prior-parity"},
    )


def test_actual_causal4d_prior_payload_round_trips_exactly() -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None
    payload = producer.bayesian_phystwin_prior_payload()

    consumer = _binding(producer.artifact_id).shared_clock_prior_from_payload(payload)

    assert consumer.clock_domain == producer.clock_domain
    assert consumer.source_artifact_id == producer.artifact_id
    assert consumer.mean_offset_s == pytest.approx(producer.mean_offset_s)
    assert consumer.standard_deviation_s == pytest.approx(
        producer.predictive_standard_deviation_s
    )
    assert payload["offset_convention"] == OBSERVATION_TIME_CORRECTION_CONVENTION


def test_causal4d_prior_payload_cannot_be_relabelled() -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None
    payload = producer.bayesian_phystwin_prior_payload()
    binding = _binding(producer.artifact_id)

    with pytest.raises(ValueError, match="domain differs"):
        binding.shared_clock_prior_from_payload(
            {**payload, "clock_domain": "different-clock"}
        )
    with pytest.raises(ValueError, match="artifact ID differs"):
        binding.shared_clock_prior_from_payload(
            {**payload, "source_artifact_id": "f" * 64}
        )
    with pytest.raises(ValueError, match="convention"):
        binding.shared_clock_prior_from_payload(
            {**payload, "offset_convention": "reversed"}
        )
