from __future__ import annotations

import hashlib
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
BUNDLE_BYTES = b'{"schema":"test-bundle"}\n'
BUNDLE_SHA = hashlib.sha256(BUNDLE_BYTES).hexdigest()
VERIFICATION_ID = "f" * 64


def _call(
    path: Path,
    *,
    expected_source: str = SOURCE_SHA,
    verification_id: str = VERIFICATION_ID,
    metadata: dict[str, object] | None = None,
):
    bundle_path = path.parent / "bundle.json"
    if not bundle_path.exists():
        bundle_path.write_bytes(BUNDLE_BYTES)
    return load_claim_bearing_prob4d_observation_timestamp_binding(
        cast(ObservationBeliefV1, object()),
        timestamp_lineage_path=path,
        expected_timestamp_source_sha256=expected_source,
        timestamp_source_verification_artifact_id=verification_id,
        bundle_manifest_path=bundle_path,
        expected_bundle_manifest_sha256=BUNDLE_SHA,
        row_factor_ids=("factor-0",),
        metadata=metadata,
    )


def test_admission_binds_independent_source_and_private_exact_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timestamps.json"
    timestamp_bytes = b"exact timestamp sidecar\n"
    path.write_bytes(timestamp_bytes)
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
        timestamp_snapshot = Path(cast(str | Path, kwargs["timestamp_lineage_path"]))
        bundle_snapshot = Path(cast(str | Path, kwargs["bundle_manifest_path"]))
        captured["timestamp_snapshot_bytes"] = timestamp_snapshot.read_bytes()
        captured["bundle_snapshot_bytes"] = bundle_snapshot.read_bytes()
        captured["timestamp_snapshot_is_private"] = timestamp_snapshot != path
        captured["bundle_snapshot_is_private"] = bundle_snapshot != (
            path.parent / "bundle.json"
        )
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
    assert captured["timestamp_snapshot_bytes"] == timestamp_bytes
    assert captured["bundle_snapshot_bytes"] == BUNDLE_BYTES
    assert captured["timestamp_snapshot_is_private"] is True
    assert captured["bundle_snapshot_is_private"] is True
    assert admitted["protocol"] == "source-frozen-v1"
    assert admitted["prob4d_timestamp_source_sha256"] == SOURCE_SHA
    assert admitted["prob4d_timestamp_source_independently_verified"] is True
    assert admitted["prob4d_timestamp_source_verification_artifact_id"] == (
        VERIFICATION_ID
    )
    assert admitted["prob4d_timestamp_lineage_artifact_id"] == LINEAGE_ID
    assert admitted["prob4d_timestamp_lineage_file_sha256"] == hashlib.sha256(
        timestamp_bytes
    ).hexdigest()


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


def test_sidecar_cannot_verify_itself(
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

    with pytest.raises(ValueError, match="own verification"):
        _call(path, verification_id=LINEAGE_ID)


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

    with pytest.raises(ValueError, match="timestamp lineage changed"):
        _call(path)


def test_bundle_replacement_during_admission_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timestamps.json"
    path.write_bytes(b"timestamp snapshot")
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(BUNDLE_BYTES)
    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_lineage",
        lambda _: SimpleNamespace(
            source_artifact_sha256=SOURCE_SHA,
            artifact_id=LINEAGE_ID,
        ),
    )

    def replace_bundle(*args: object, **kwargs: object) -> SimpleNamespace:
        bundle_path.write_bytes(b"replacement bundle")
        return SimpleNamespace(
            timestamp_lineage_artifact_id=LINEAGE_ID,
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(
        "bayesian_phystwin.prob4d_observation_timestamp_admission."
        "load_prob4d_observation_timestamp_binding",
        replace_bundle,
    )

    with pytest.raises(ValueError, match="bundle changed"):
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


def test_symlinked_bundle_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.json"
    path.write_bytes(b"sidecar")
    bundle_target = tmp_path / "bundle-target.json"
    bundle_target.write_bytes(BUNDLE_BYTES)
    (tmp_path / "bundle.json").symlink_to(bundle_target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _call(path)


def test_non_regular_sidecar_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "timestamps.json"
    directory.mkdir()

    with pytest.raises(ValueError, match="ordinary file"):
        _call(directory)
