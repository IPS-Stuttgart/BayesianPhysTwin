from __future__ import annotations

from pathlib import Path

import numpy as np
from prob4d.observation_timestamp_lineage import (
    ObservationTimestampLineageV1,
    write_observation_timestamp_lineage,
)

from bayesian_phystwin.prob4d_observation_timestamps import (
    load_prob4d_observation_timestamp_lineage,
)


def test_actual_prob4d_timestamp_sidecar_has_exact_consumer_identity(
    tmp_path: Path,
) -> None:
    producer = ObservationTimestampLineageV1(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="stream-a",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        causal_frame_stop=3,
        clock_domain="camera-hardware-clock",
        time_scale="device-monotonic",
        timestamp_source="camera-packet-timestamp",
        factor_ids=("factor-0", "factor-1"),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        timestamps_ns=np.asarray(
            [1_000_000_000, 2_000_000_000],
            dtype=np.int64,
        ),
        conditional_timestamp_std_ns=np.asarray(
            [1_000_000.0, 2_000_000.0],
            dtype=np.float64,
        ),
        shared_clock_offset_prior_artifact_id="c" * 64,
        metadata={"calibration_partition": "source-only"},
    )
    path = tmp_path / "timestamp-lineage.json"
    write_observation_timestamp_lineage(producer, path)

    consumer = load_prob4d_observation_timestamp_lineage(path)

    assert consumer.artifact_id == producer.artifact_id
    assert consumer.identity_record() == producer.identity_record()
    assert consumer.factor_ids == producer.factor_ids
    np.testing.assert_array_equal(consumer.frame_indices, producer.frame_indices)
    np.testing.assert_array_equal(consumer.timestamps_ns, producer.timestamps_ns)
    np.testing.assert_allclose(
        consumer.conditional_timestamp_std_ns,
        producer.conditional_timestamp_std_ns,
    )
