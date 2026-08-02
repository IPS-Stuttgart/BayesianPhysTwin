from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_exclusion_union import (
    EXCLUSION_KIND,
    HASH_NAMESPACE,
    build_exclusion_union,
    object_exclusion_hash,
    validate_exclusion_manifest,
)
from bayesian_phystwin.deform360_tactile_features import canonical_artifact_sha256


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _exclusion(path: Path) -> tuple[Path, dict[str, object]]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": EXCLUSION_KIND,
        "hash_namespace": HASH_NAMESPACE,
        "owner": "independent",
        "object_hashes": [object_exclusion_hash("001-rope")],
        "source_artifact_sha256s": ["a" * 64],
        "information_boundary": {
            "target_artifact_read": False,
            "object_ids_emitted": False,
        },
    }
    stripped = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["exclusion_sha256"] = hashlib.sha256(stripped).hexdigest()
    return _write(path, payload), payload


def _source(path: Path) -> tuple[Path, dict[str, object]]:
    payload: dict[str, object] = {
        "artifact_kind": "Deform360TactileRegretGuardSourceDiagnostic",
        "cross_fitted": {
            "combined": {
                "cases": [
                    {"object": "002-rope-silk"},
                    {"object": "002-rope-silk"},
                ]
            }
        },
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return _write(path, payload), payload


def test_union_is_hash_only_and_counts_new_objects(tmp_path: Path) -> None:
    exclusion = _exclusion(tmp_path / "external.json")
    source = _source(tmp_path / "source.json")
    support = tmp_path / "support.py"
    support.write_text("opened fixture\n", encoding="utf-8")
    result = build_exclusion_union(
        [exclusion],
        [source],
        additional_opened_object_ids=["003-cable"],
        additional_source_artifacts=[support],
        owner="prospective-test",
    )
    validate_exclusion_manifest(result)
    assert result["accounting"] == {
        "independent_hash_count": 1,
        "opened_source_object_count": 2,
        "new_opened_source_hash_count": 2,
        "union_hash_count": 3,
    }
    encoded = json.dumps(result, sort_keys=True)
    assert "002-rope-silk" not in encoded
    assert "003-cable" not in encoded


def test_mutated_input_exclusion_is_rejected(tmp_path: Path) -> None:
    path, exclusion = _exclusion(tmp_path / "external.json")
    exclusion["owner"] = "mutated"
    _write(path, exclusion)
    with pytest.raises(ValueError, match="checksum changed"):
        build_exclusion_union(
            [(path, exclusion)],
            [_source(tmp_path / "source.json")],
            owner="prospective-test",
        )


def test_object_hash_uses_shared_namespace() -> None:
    expected = hashlib.sha256(
        b"deform360-fresh-object-exclusion-v1\0" + b"002-rope-silk"
    ).hexdigest()
    assert object_exclusion_hash("002-rope-silk") == expected
