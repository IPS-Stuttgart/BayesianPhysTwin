from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REQUIRED_WORKFLOW = "Prob4D observation timestamp consumer"

try:
    from causal4d.observation_clock_offset_prior import (
        OBSERVATION_TIME_CORRECTION_CONVENTION,
        ObservationClockOffsetPriorV1,
    )
except ModuleNotFoundError:
    if (
        os.environ.get("BPT_REQUIRE_TIMESTAMP_COMPANIONS") == "1"
        or os.environ.get("GITHUB_WORKFLOW") == _REQUIRED_WORKFLOW
    ):
        raise
    pytest.skip(
        "Causal4D parity is validated by the dedicated timestamp consumer workflow",
        allow_module_level=True,
    )

from bayesian_phystwin.causal4d_observation_clock_prior import (
    Causal4DObservationClockOffsetPriorV1,
    causal4d_observation_timing_prior_from_record,
    load_causal4d_observation_timing_prior,
)
from bayesian_phystwin.observation_timing_nuisance import (
    ObservationTimingPrior,
)
from bayesian_phystwin.prob4d_observation_timestamps import (
    Prob4DObservationTimestampBindingV1,
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


def _binding(
    prior_artifact_id: str,
    *,
    clock_domain: str = "camera-hardware-clock",
    time_scale: str = "device-monotonic",
) -> Prob4DObservationTimestampBindingV1:
    return Prob4DObservationTimestampBindingV1(
        observation_artifact_id="4" * 64,
        bundle_manifest_sha256="5" * 64,
        timestamp_lineage_artifact_id="6" * 64,
        clock_domain=clock_domain,
        time_scale=time_scale,
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


def _consume(
    binding: Prob4DObservationTimestampBindingV1,
    record: Mapping[str, Any],
) -> ObservationTimingPrior:
    expected_id = binding.shared_clock_offset_prior_artifact_id
    if expected_id is None:
        raise ValueError("timestamp lineage declares no shared clock prior")
    return causal4d_observation_timing_prior_from_record(
        record,
        expected_artifact_id=expected_id,
        expected_clock_domain=binding.clock_domain,
        expected_time_scale=binding.time_scale,
    )


def test_strict_clock_prior_file_loader_round_trip_and_duplicates(
    tmp_path: Path,
) -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None
    binding = _binding(producer.artifact_id)
    path = tmp_path / "clock-prior.json"
    path.write_text(json.dumps(producer.to_record()), encoding="utf-8")

    prior = load_causal4d_observation_timing_prior(
        path,
        expected_artifact_id=producer.artifact_id,
        expected_clock_domain=binding.clock_domain,
        expected_time_scale=binding.time_scale,
    )
    assert prior.source_artifact_id == producer.artifact_id

    path.write_text(
        '{"schema":"first","schema":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_causal4d_observation_timing_prior(
            path,
            expected_artifact_id=producer.artifact_id,
            expected_clock_domain=binding.clock_domain,
            expected_time_scale=binding.time_scale,
        )


def test_actual_causal4d_prior_record_round_trips_exactly() -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None
    record = producer.to_record()

    reconstructed = Causal4DObservationClockOffsetPriorV1.from_mapping(record)
    consumer = _consume(
        _binding(producer.artifact_id),
        record,
    )

    assert reconstructed.artifact_id == producer.artifact_id
    assert reconstructed.identity_record() == producer.identity_record()
    assert consumer.clock_domain == producer.clock_domain
    assert consumer.source_artifact_id == producer.artifact_id
    assert consumer.mean_offset_s == pytest.approx(producer.mean_offset_s)
    assert consumer.standard_deviation_s == pytest.approx(
        producer.predictive_standard_deviation_s
    )
    assert record["offset_convention"] == OBSERVATION_TIME_CORRECTION_CONVENTION


def test_compact_clock_payload_is_not_claim_bearing() -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None

    with pytest.raises(ValueError, match="fields changed"):
        _consume(
            _binding(producer.artifact_id),
            producer.bayesian_phystwin_prior_payload(),
        )


def test_causal4d_prior_record_rejects_changed_gaussian_with_same_id() -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None
    binding = _binding(producer.artifact_id)

    with pytest.raises(ValueError, match="summary does not match"):
        _consume(
            binding,
            {**producer.to_record(), "mean_offset_s": 0.25},
        )
    with pytest.raises(ValueError, match="summary does not match"):
        _consume(
            binding,
            {
                **producer.to_record(),
                "predictive_standard_deviation_s": 0.25,
            },
        )


def test_causal4d_prior_record_cannot_be_relabelled() -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None

    different_domain = replace(
        producer,
        clock_domain="different-clock",
        artifact_id=None,
    )
    assert different_domain.artifact_id is not None
    with pytest.raises(ValueError, match="domain differs"):
        _consume(
            _binding(different_domain.artifact_id),
            different_domain.to_record(),
        )

    different_scale = replace(
        producer,
        time_scale="tai",
        artifact_id=None,
    )
    assert different_scale.artifact_id is not None
    with pytest.raises(ValueError, match="time scale differs"):
        _consume(
            _binding(different_scale.artifact_id),
            different_scale.to_record(),
        )

    different_source = replace(
        producer,
        source_revision="b" * 40,
        artifact_id=None,
    )
    assert different_source.artifact_id is not None
    with pytest.raises(ValueError, match="artifact ID differs"):
        _consume(
            _binding(producer.artifact_id),
            different_source.to_record(),
        )

    with pytest.raises(ValueError, match="convention changed"):
        _consume(
            _binding(producer.artifact_id),
            {**producer.to_record(), "offset_convention": "reversed"},
        )


def test_causal4d_prior_record_preserves_information_boundary() -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None
    record = producer.to_record()
    boundary = dict(record["information_boundary"])
    boundary["target_outcomes_used"] = True

    with pytest.raises(ValueError, match="information boundary changed"):
        _consume(
            _binding(producer.artifact_id),
            {**record, "information_boundary": boundary},
        )


def test_timestamp_lineage_must_declare_shared_clock_prior() -> None:
    producer = _causal4d_prior()
    assert producer.artifact_id is not None
    binding = replace(
        _binding(producer.artifact_id),
        shared_clock_offset_prior_artifact_id=None,
        binding_id=None,
    )

    with pytest.raises(ValueError, match="declares no shared clock prior"):
        _consume(binding, producer.to_record())
