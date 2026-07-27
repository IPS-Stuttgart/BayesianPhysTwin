from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.cli.deform360_object_exclusion import main
from bayesian_phystwin.deform360_object_exclusion import (
    EXCLUSION_KIND,
    HASH_NAMESPACE,
    canonical_sha256,
    file_sha256,
    load_object_exclusion_manifest,
    merge_object_exclusion_manifests,
    validate_object_exclusion_manifest,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _manifest(owner: str, objects: list[str], sources: list[str]) -> dict[str, object]:
    artifact: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": EXCLUSION_KIND,
        "hash_namespace": HASH_NAMESPACE,
        "owner": owner,
        "object_hashes": sorted(objects),
        "source_artifact_sha256s": sorted(sources),
        "information_boundary": {
            "target_artifact_read": False,
            "object_ids_emitted": False,
        },
    }
    artifact["exclusion_sha256"] = canonical_sha256(artifact)
    return artifact


def _write(path: Path, artifact: dict[str, object]) -> None:
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_merge_is_hash_only_deterministic_and_deduplicated(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, _manifest("first", [_digest(1), _digest(2)], [_digest(11)]))
    _write(second, _manifest("second", [_digest(2), _digest(3)], [_digest(12)]))

    merged = merge_object_exclusion_manifests(
        [second, first],
        owner="fresh-dynamic-provider-v1",
    )
    reverse = merge_object_exclusion_manifests(
        [first, second],
        owner="fresh-dynamic-provider-v1",
    )

    assert merged == reverse
    assert merged["object_hashes"] == [_digest(1), _digest(2), _digest(3)]
    assert merged["source_artifact_sha256s"] == sorted(
        [file_sha256(first), file_sha256(second)]
    )
    assert merged["composition"] == {
        "rule": "set-union-of-validated-hash-only-manifests",
        "member_count": 2,
        "input_hash_count": 4,
        "unique_object_hash_count": 3,
        "members": sorted(
            [
                {
                    "owner": "first",
                    "exclusion_sha256": load_object_exclusion_manifest(first)[
                        "exclusion_sha256"
                    ],
                    "file_sha256": file_sha256(first),
                    "object_hash_count": 2,
                },
                {
                    "owner": "second",
                    "exclusion_sha256": load_object_exclusion_manifest(second)[
                        "exclusion_sha256"
                    ],
                    "file_sha256": file_sha256(second),
                    "object_hash_count": 2,
                },
            ],
            key=lambda item: (
                item["owner"],
                item["exclusion_sha256"],
                item["file_sha256"],
            ),
        ),
    }
    assert merged["information_boundary"]["target_artifact_read"] is False
    assert merged["information_boundary"]["object_ids_emitted"] is False
    validate_object_exclusion_manifest(merged)


def test_validation_rejects_checksum_identity_and_boundary_changes() -> None:
    artifact = _manifest("source", [_digest(1)], [_digest(2)])
    changed = dict(artifact)
    changed["owner"] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        validate_object_exclusion_manifest(changed)

    exposed = dict(artifact)
    exposed["object_ids"] = ["plaintext-object"]
    exposed["exclusion_sha256"] = canonical_sha256(exposed)
    with pytest.raises(ValueError, match="plaintext identity"):
        validate_object_exclusion_manifest(exposed)

    crossed = json.loads(json.dumps(artifact))
    crossed["information_boundary"]["target_artifact_read"] = True
    crossed["exclusion_sha256"] = canonical_sha256(crossed)
    with pytest.raises(ValueError, match="information boundary"):
        validate_object_exclusion_manifest(crossed)


def test_cli_creates_once_and_reports_only_hash_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "merged.json"
    _write(source, _manifest("source", [_digest(1)], [_digest(2)]))

    assert (
        main(
            [
                str(output),
                "--owner",
                "fresh-dynamic-provider-v1",
                "--input",
                str(source),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["unique_object_hash_count"] == 1
    assert summary["member_count"] == 1
    assert summary["file_sha256"] == file_sha256(output)
    assert "object_hashes" not in summary

    with pytest.raises(ValueError, match="refusing to replace"):
        main(
            [
                str(output),
                "--owner",
                "fresh-dynamic-provider-v1",
                "--input",
                str(source),
            ]
        )
