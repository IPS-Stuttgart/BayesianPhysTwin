from __future__ import annotations

import io
import os
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.observation_belief_io_v2 as io_v2
from bayesian_phystwin.observation_belief import (
    ObservationBeliefV1,
    load_observation_belief,
    save_observation_belief,
)
from bayesian_phystwin.observation_belief_io_v2 import (
    ObservationBeliefIOLimitsV2,
    load_observation_belief_bounded_v2,
    save_observation_belief_atomic_v2,
)


def _belief(*, observation_count: int = 3, factor_rank: int = 2) -> ObservationBeliefV1:
    local_covariance = np.repeat(
        np.diag([0.002, 0.003, 0.004])[None, :, :],
        observation_count,
        axis=0,
    )
    low_rank = np.zeros((observation_count, 3, factor_rank), dtype=np.float64)
    if factor_rank:
        low_rank[:, 0, 0] = np.linspace(0.001, 0.002, observation_count)
    return ObservationBeliefV1(
        case_id="case-a",
        stream_id="stream-a",
        causal_frame_stop=2,
        view_names=("camera-a",),
        window_names=("window-a",),
        factor_names=tuple(f"factor-{index}" for index in range(factor_rank)),
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([0, 1], dtype=np.int64),
        mean_xyz_m=np.arange(observation_count * 3, dtype=np.float64).reshape(
            observation_count, 3
        ),
        frame_ids=np.arange(observation_count, dtype=np.int64) % 2,
        entity_ids=np.arange(observation_count, dtype=np.int64),
        view_indices=np.zeros(observation_count, dtype=np.int64),
        window_indices=np.zeros(observation_count, dtype=np.int64),
        correlation_group_ids=np.zeros(observation_count, dtype=np.int64),
        factor_group_ids=np.zeros(observation_count, dtype=np.int64),
        prior_reliability=np.linspace(0.6, 1.0, observation_count),
        association_probability=np.linspace(0.7, 1.0, observation_count),
        local_covariance_m2=local_covariance,
        low_rank_factor_m=low_rank,
        group_ids=np.asarray([0], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([0.9], dtype=np.float64),
        group_composite_weight=np.asarray([1.0], dtype=np.float64),
        metadata={"split": "calibration"},
    )


def test_v2_round_trip_preserves_the_v1_artifact_identity(tmp_path: Path) -> None:
    source = _belief()
    path = tmp_path / "belief.npz"

    save_observation_belief_atomic_v2(path, source)

    strict = load_observation_belief_bounded_v2(path)
    legacy = load_observation_belief(path)
    assert strict.artifact_id == source.artifact_id
    assert legacy.artifact_id == source.artifact_id


def test_v2_loads_a_legacy_writer_archive(tmp_path: Path) -> None:
    source = _belief()
    path = tmp_path / "legacy.npz"
    save_observation_belief(path, source)

    loaded = load_observation_belief_bounded_v2(path)

    assert loaded.artifact_id == source.artifact_id


def test_atomic_write_leaves_the_previous_target_on_serialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "belief.npz"
    target.write_bytes(b"previous-authoritative-bytes")

    def fail_after_partial_write(stream: io.BufferedWriter, **_: object) -> None:
        stream.write(b"partial-new-archive")
        stream.flush()
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(io_v2.np, "savez_compressed", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="injected serialization failure"):
        save_observation_belief_atomic_v2(target, _belief())

    assert target.read_bytes() == b"previous-authoritative-bytes"
    assert list(tmp_path.glob(f".{target.name}.*.partial")) == []


def test_atomic_write_verifies_before_replacing_the_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "belief.npz"
    target.write_bytes(b"previous-authoritative-bytes")

    def reject(*_: object, **__: object) -> ObservationBeliefV1:
        raise ValueError("injected verification failure")

    monkeypatch.setattr(io_v2, "load_observation_belief_bounded_v2", reject)

    with pytest.raises(ValueError, match="injected verification failure"):
        save_observation_belief_atomic_v2(target, _belief())

    assert target.read_bytes() == b"previous-authoritative-bytes"
    assert list(tmp_path.glob(f".{target.name}.*.partial")) == []


def test_archive_byte_budget_is_enforced_before_numpy_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    called = False

    def forbidden_load(*_: object, **__: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("np.load must not run before archive preflight")

    monkeypatch.setattr(io_v2.np, "load", forbidden_load)
    limits = ObservationBeliefIOLimitsV2(maximum_archive_bytes=path.stat().st_size - 1)

    with pytest.raises(ValueError, match="archive byte budget"):
        load_observation_belief_bounded_v2(path, limits=limits)

    assert not called


def test_observation_count_budget_is_enforced_from_the_npy_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief(observation_count=3))
    called = False

    def forbidden_load(*_: object, **__: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("np.load must not run before shape preflight")

    monkeypatch.setattr(io_v2.np, "load", forbidden_load)
    limits = ObservationBeliefIOLimitsV2(maximum_observation_count=2)

    with pytest.raises(ValueError, match="observation count"):
        load_observation_belief_bounded_v2(path, limits=limits)

    assert not called


def test_factor_rank_budget_is_enforced_from_the_npy_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief(factor_rank=2))

    with pytest.raises(ValueError, match="factor rank"):
        load_observation_belief_bounded_v2(
            path,
            limits=ObservationBeliefIOLimitsV2(maximum_factor_rank=1),
        )


def test_duplicate_zip_members_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    with zipfile.ZipFile(path, "r") as archive:
        duplicate = archive.read("descriptor_json.npy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("descriptor_json.npy", duplicate)

    with pytest.raises(ValueError, match="duplicate ZIP members"):
        load_observation_belief_bounded_v2(path)


def test_duplicate_descriptor_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_observation_belief(path, _belief())
    with zipfile.ZipFile(path, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    malformed = (
        '{"schema_name":"phys4d.observation_belief",'
        '"schema_name":"phys4d.observation_belief"}'
    )
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(malformed), allow_pickle=False)
    members["descriptor_json.npy"] = buffer.getvalue()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    with pytest.raises(ValueError, match="duplicate key"):
        load_observation_belief_bounded_v2(path)


def test_symbolic_link_input_is_rejected_when_no_follow_is_available(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform has no O_NOFOLLOW")
    source = tmp_path / "source.npz"
    link = tmp_path / "link.npz"
    save_observation_belief(source, _belief())
    link.symlink_to(source)

    with pytest.raises(ValueError, match="ordinary file"):
        load_observation_belief_bounded_v2(link)


def test_limits_require_consistent_positive_integer_budgets() -> None:
    with pytest.raises(TypeError, match="genuine integer"):
        ObservationBeliefIOLimitsV2(maximum_archive_bytes=True)
    with pytest.raises(ValueError, match="positive"):
        ObservationBeliefIOLimitsV2(maximum_archive_bytes=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        ObservationBeliefIOLimitsV2(
            maximum_uncompressed_bytes=10,
            maximum_member_bytes=11,
            maximum_descriptor_bytes=1,
        )
