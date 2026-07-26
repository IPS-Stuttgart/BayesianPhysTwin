from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.legacy_artifacts as legacy_artifacts
from bayesian_phystwin.causal4d_provider_v1 import (
    causal4d_provider_manifest,
    load_trusted_legacy_phystwin_pickle,
)


def _write_pickle(path: Path, value) -> str:
    with path.open("wb") as stream:
        pickle.dump(value, stream)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provider_advertises_trusted_legacy_loading() -> None:
    manifest = causal4d_provider_manifest(provider_revision="test-revision")
    assert "trusted_legacy_pickle_loading" in manifest["capabilities"]


def test_loads_hash_locked_mapping_and_validates_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pkl"
    digest = _write_pickle(path, {"object_points": np.arange(3), "surface_points": []})

    loaded = load_trusted_legacy_phystwin_pickle(
        path,
        expected_sha256=digest,
        artifact_kind="mapping",
        required_keys=("object_points", "surface_points"),
    )

    np.testing.assert_array_equal(loaded["object_points"], np.arange(3))


def test_rejects_digest_mismatch_before_deserialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "artifact.pkl"
    _write_pickle(path, {"value": 1})
    monkeypatch.setattr(
        legacy_artifacts.pickle,
        "load",
        lambda stream: pytest.fail("pickle must not be opened after digest mismatch"),
    )

    with pytest.raises(ValueError, match="refusing to deserialize"):
        load_trusted_legacy_phystwin_pickle(
            path,
            expected_sha256="0" * 64,
            artifact_kind="mapping",
        )


def test_rejects_top_level_contract_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pkl"
    digest = _write_pickle(path, [1, 2, 3])

    with pytest.raises(TypeError, match="mapping"):
        load_trusted_legacy_phystwin_pickle(
            path,
            expected_sha256=digest,
            artifact_kind="mapping",
        )


def test_rejects_missing_mapping_keys_and_invalid_digest(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pkl"
    digest = _write_pickle(path, {"value": 1})

    with pytest.raises(ValueError, match="missing required keys"):
        load_trusted_legacy_phystwin_pickle(
            path,
            expected_sha256=digest,
            artifact_kind="mapping",
            required_keys=("other",),
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_trusted_legacy_phystwin_pickle(
            path,
            expected_sha256=digest.upper(),
            artifact_kind="mapping",
        )
