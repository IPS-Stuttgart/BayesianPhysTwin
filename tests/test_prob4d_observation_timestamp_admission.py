from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.prob4d_observation_timestamp_admission import (
    load_claim_bearing_prob4d_observation_timestamp_binding,
)

SOURCE_SHA = "a" * 64
LINEAGE_ID = "b" * 64
BUNDLE_SHA = "c" * 64


def _call(
    path: Path,
    *,
    expected_source: str = SOURCE_SHA,
    metadata: dict[str, object] | None = None,
):
    return load_claim_bearing_prob4d_observation_timestamp_binding(
        cast(ObservationBeliefV1, object()),
        timestamp_lineage_path=path,
        expected_timestamp_source_sha256=expected_source,
        bundle_manifest_path=path.parent / "bundle.json",
        expected_bundle_manifest_sha256=BUNDLE_SHA,
        row_factor_ids=("factor-0",),
        metadata=metadata,
    )


def test_admission_binds_independent_source_and_exact_sidecar_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timestamps.json"
    path.write_bytes(b"exact timestamp sidecar\n")
    lineage = SimpleNamespace(
        source_artifact_sha256=SOURCE_SHA,
        artifact_id=LINEAGE_ID,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_lineage",
        lambda _: lineage,
    )

    def fake_binding(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            timestamp_lineage_artifact_id=LINEAGE_ID,
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_binding",
        fake_binding,
    )

    binding = _call(path, metadata={"protocol": "source-frozen-v1"})
    admitted = cast(dict[str, object], captured["metadata"])

    assert binding.timestamp_lineage_artifact_id == LINEAGE_ID
    assert admitted["protocol"] == "source-frozen-v1"
    assert admitted["prob4d_timestamp_source_sha256"] == SOURCE_SHA
    assert admitted["prob4d_timestamp_source_independently_verified"] is True
    assert admitted["prob4d_timestamp_lineage_artifact_id"] == LINEAGE_ID
    assert len(cast(str, admitted["prob4d_timestamp_lineage_file_sha256"])) == 64


def test_wrong_independent_source_digest_fails_before_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timestamps.json"
    path.write_bytes(b"sidecar")
    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_lineage",
        lambda _: SimpleNamespace(
            source_artifact_sha256="d" * 64,
            artifact_id=LINEAGE_ID,
        ),
    )

    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("lower-level binding must not run")

    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_binding",
        forbidden,
    )

    with pytest.raises(ValueError, match="independent evidence"):
        _call(path)
    assert not called


def test_sidecar_replacement_during_admission_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timestamps.json"
    path.write_bytes(b"first snapshot")
    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_lineage",
        lambda _: SimpleNamespace(
            source_artifact_sha256=SOURCE_SHA,
            artifact_id=LINEAGE_ID,
        ),
    )

    def replace_sidecar(*args: object, **kwargs: object) -> SimpleNamespace:
        path.write_bytes(b"replacement snapshot")
        return SimpleNamespace(
            timestamp_lineage_artifact_id=LINEAGE_ID,
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_binding",
        replace_sidecar,
    )

    with pytest.raises(ValueError, match="changed during admission"):
        _call(path)


def test_changed_lineage_identity_and_reserved_metadata_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timestamps.json"
    path.write_bytes(b"sidecar")
    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_lineage",
        lambda _: SimpleNamespace(
            source_artifact_sha256=SOURCE_SHA,
            artifact_id=LINEAGE_ID,
        ),
    )
    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_binding",
        lambda *args, **kwargs: SimpleNamespace(
            timestamp_lineage_artifact_id="e" * 64,
            metadata=kwargs["metadata"],
        ),
    )

    with pytest.raises(ValueError, match="identity changed"):
        _call(path)

    with pytest.raises(ValueError, match="reserves fields"):
        _call(
            path,
            metadata={"prob4d_timestamp_source_sha256": SOURCE_SHA},
        )


def test_symlinked_sidecar_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"sidecar")
    link = tmp_path / "timestamps.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _call(link)


def test_non_regular_sidecar_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "timestamps.json"
    directory.mkdir()

    with pytest.raises(ValueError, match="ordinary file"):
        _call(directory)
