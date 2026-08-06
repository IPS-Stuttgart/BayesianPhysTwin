from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.prob4d_observation_timestamps import (
    PROB4D_CONDITIONAL_JITTER_FACTOR_SEMANTICS,
    PROB4D_TIMESTAMP_UNCERTAINTY_SEMANTICS,
    Prob4DObservationTimestampLineageV1,
    load_prob4d_observation_timestamp_binding,
    load_prob4d_observation_timestamp_lineage,
)

REVISION = "a" * 40
SOURCE_ARTIFACT = "b" * 64
SHARED_PRIOR = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(*, frame_ids: np.ndarray | None = None) -> ObservationBeliefV1:
    frames = (
        np.asarray([0, 0, 1, 1], dtype=np.int64)
        if frame_ids is None
        else np.asarray(frame_ids, dtype=np.int64)
    )
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    return ObservationBeliefV1(
        case_id="case-a",
        stream_id="stream-a",
        causal_frame_stop=3,
        view_names=("camera-0",),
        window_names=("window-0",),
        factor_names=(),
        source_repository="FlorianPfaff/Prob4D",
        source_revision=REVISION,
        source_artifact_sha256="d" * 64,
        declared_frame_ids=np.asarray([0, 1], dtype=np.int64),
        mean_xyz_m=np.zeros((4, 3), dtype=np.float64),
        frame_ids=frames,
        entity_ids=np.arange(4, dtype=np.int64),
        view_indices=np.zeros(4, dtype=np.int64),
        window_indices=np.zeros(4, dtype=np.int64),
        correlation_group_ids=groups,
        factor_group_ids=groups,
        prior_reliability=np.ones(4, dtype=np.float64),
        association_probability=np.ones(4, dtype=np.float64),
        local_covariance_m2=np.repeat(
            (1e-4 * np.eye(3, dtype=np.float64))[None],
            4,
            axis=0,
        ),
        low_rank_factor_m=np.zeros((4, 3, 0), dtype=np.float64),
        group_ids=np.asarray([0, 1], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([0.9, 0.9]),
        group_composite_weight=np.ones(2, dtype=np.float64),
        metadata={"source": "timestamp-consumer-test"},
    )


def _write_bundle(
    path: Path,
    *,
    factor_ids: tuple[str, str] = ("factor-0", "factor-1"),
    frame_indices: tuple[int, int] = (0, 1),
    source_repository: str = "FlorianPfaff/Prob4D",
) -> str:
    value = {
        "schema": "prob4d.observation-factor-bundle",
        "schema_version": 4,
        "sequence_id": "sequence-a",
        "case_id": "case-a",
        "stream_id": "stream-a",
        "source_repository": source_repository,
        "source_revision": REVISION,
        "causal_frame_stop": 3,
        "factors": [
            {"factor_id": factor_id, "frame_index": frame_index}
            for factor_id, frame_index in zip(
                factor_ids,
                frame_indices,
                strict=True,
            )
        ],
    }
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _lineage(**overrides: object) -> Prob4DObservationTimestampLineageV1:
    values: dict[str, object] = {
        "sequence_id": "sequence-a",
        "case_id": "case-a",
        "stream_id": "stream-a",
        "source_revision": REVISION,
        "source_artifact_sha256": SOURCE_ARTIFACT,
        "causal_frame_stop": 3,
        "clock_domain": "camera-hardware-clock",
        "time_scale": "device-monotonic",
        "timestamp_source": "camera-packet-timestamp",
        "factor_ids": ("factor-0", "factor-1"),
        "frame_indices": np.asarray([0, 1], dtype=np.int64),
        "timestamps_ns": np.asarray([1_000_000_000, 2_000_000_000]),
        "conditional_timestamp_std_ns": np.asarray([1_000_000.0, 2_000_000.0]),
        "shared_clock_offset_prior_artifact_id": SHARED_PRIOR,
        "metadata": {"calibration_partition": "source-only"},
    }
    values.update(overrides)
    return Prob4DObservationTimestampLineageV1(**values)  # type: ignore[arg-type]


def _write_lineage(path: Path, **overrides: object) -> None:
    lineage = _lineage(**overrides)
    value = {**lineage.identity_record(), "artifact_id": lineage.artifact_id}
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _binding(tmp_path: Path, *, observation: ObservationBeliefV1 | None = None):
    bundle_path = tmp_path / "bundle.json"
    lineage_path = tmp_path / "timestamp-lineage.json"
    bundle_sha = _write_bundle(bundle_path)
    _write_lineage(lineage_path)
    return load_prob4d_observation_timestamp_binding(
        _observation() if observation is None else observation,
        timestamp_lineage_path=lineage_path,
        bundle_manifest_path=bundle_path,
        expected_bundle_manifest_sha256=bundle_sha,
        row_factor_ids=("factor-0", "factor-0", "factor-1", "factor-1"),
        metadata={"protocol": "timestamp-binding-test"},
    )


def test_exact_factor_to_row_timestamp_binding(tmp_path: Path) -> None:
    binding = _binding(tmp_path)

    np.testing.assert_allclose(binding.row_timestamps_s, [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(
        binding.row_conditional_timestamp_std_s,
        [0.001, 0.001, 0.002, 0.002],
    )
    assert binding.row_factor_indices.tolist() == [0, 0, 1, 1]
    assert binding.timestamp_uncertainty_semantics == (
        PROB4D_TIMESTAMP_UNCERTAINTY_SEMANTICS
    )
    assert binding.conditional_jitter_factor_semantics == (
        PROB4D_CONDITIONAL_JITTER_FACTOR_SEMANTICS
    )
    assert binding.binding_id is not None


def test_factor_local_jitter_and_shared_clock_are_separate(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    derivative = np.zeros((4, 3), dtype=np.float64)
    derivative[:, 0] = [1.0, 2.0, 3.0, 4.0]

    factor = binding.conditional_jitter_low_rank_factor(derivative)
    dense = factor.reshape(12, binding.factor_count)
    covariance = dense @ dense.T
    x0, x1, x2, x3 = 0, 3, 6, 9

    assert covariance[x0, x1] == pytest.approx(2e-6)
    assert covariance[x2, x3] == pytest.approx(48e-6)
    assert covariance[x0, x2] == pytest.approx(0.0)

    shared = binding.shared_clock_design(derivative)
    assert shared.shape == (12, 1)
    np.testing.assert_allclose(shared[:, 0], derivative.reshape(-1))

    shared_covariance = shared @ shared.T
    assert shared_covariance[x0, x2] != 0.0
    assert covariance[x0, x2] == 0.0


def test_shared_clock_prior_binds_artifact_domain_and_sign(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    payload: dict[str, object] = {
        "clock_domain": "camera-hardware-clock",
        "mean_offset_s": 0.001,
        "standard_deviation_s": 0.0005,
        "source_artifact_id": SHARED_PRIOR,
        "offset_convention": (
            "aligned_observation_time_s = observation_time_s + offset_s"
        ),
    }

    prior = binding.shared_clock_prior_from_payload(payload)
    assert prior.clock_domain == binding.clock_domain
    assert prior.source_artifact_id == SHARED_PRIOR

    with pytest.raises(ValueError, match="artifact ID"):
        binding.shared_clock_prior_from_payload(
            {**payload, "source_artifact_id": "f" * 64}
        )
    with pytest.raises(ValueError, match="domain"):
        binding.shared_clock_prior_from_payload(
            {**payload, "clock_domain": "other-clock"}
        )
    with pytest.raises(ValueError, match="convention"):
        binding.shared_clock_prior_from_payload(
            {**payload, "offset_convention": "reversed"}
        )


def test_source_order_frame_and_checksum_mismatches_fail_closed(
    tmp_path: Path,
) -> None:
    observation = _observation()
    bundle_path = tmp_path / "bundle.json"
    lineage_path = tmp_path / "timestamp-lineage.json"
    bundle_sha = _write_bundle(bundle_path)
    _write_lineage(
        lineage_path,
        factor_ids=("factor-1", "factor-0"),
        frame_indices=np.asarray([1, 0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="factor order"):
        load_prob4d_observation_timestamp_binding(
            observation,
            timestamp_lineage_path=lineage_path,
            bundle_manifest_path=bundle_path,
            expected_bundle_manifest_sha256=bundle_sha,
            row_factor_ids=("factor-0",) * 4,
        )

    _write_lineage(lineage_path)
    bundle_path.write_text(bundle_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_prob4d_observation_timestamp_binding(
            observation,
            timestamp_lineage_path=lineage_path,
            bundle_manifest_path=bundle_path,
            expected_bundle_manifest_sha256=bundle_sha,
            row_factor_ids=("factor-0",) * 4,
        )

    clean_sha = _write_bundle(bundle_path)
    with pytest.raises(ValueError, match="row frames"):
        load_prob4d_observation_timestamp_binding(
            _observation(frame_ids=np.asarray([0, 1, 1, 1])),
            timestamp_lineage_path=lineage_path,
            bundle_manifest_path=bundle_path,
            expected_bundle_manifest_sha256=clean_sha,
            row_factor_ids=("factor-0", "factor-0", "factor-1", "factor-1"),
        )


def test_lineage_tampering_duplicates_and_noninteger_time_fail_closed(
    tmp_path: Path,
) -> None:
    lineage_path = tmp_path / "timestamp-lineage.json"
    _write_lineage(lineage_path)
    value = json.loads(lineage_path.read_text(encoding="utf-8"))
    value["artifact_id"] = "0" * 64
    lineage_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        load_prob4d_observation_timestamp_lineage(lineage_path)

    lineage_path.write_text(
        '{"schema":"first","schema":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_prob4d_observation_timestamp_lineage(lineage_path)

    with pytest.raises(ValueError, match="integers"):
        _lineage(timestamps_ns=np.asarray([1.0, 2.0]))


def test_binding_arrays_are_irreversibly_immutable(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    for array in (
        binding.factor_frame_indices,
        binding.factor_timestamps_ns,
        binding.conditional_timestamp_std_ns,
        binding.row_factor_indices,
        binding.row_timestamps_s,
        binding.row_conditional_timestamp_std_s,
        binding.conditional_jitter_low_rank_factor(
            np.ones((4, 3), dtype=np.float64)
        ),
        binding.shared_clock_design(np.ones((4, 3), dtype=np.float64)),
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_binding_identity_rejects_tampering(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    with pytest.raises(ValueError, match="binding ID mismatch"):
        replace(binding, binding_id="f" * 64)
