from __future__ import annotations

import json
import subprocess
from pathlib import Path

from bayesian_phystwin.pokeflex_fresh_take_selection import (
    build_fresh_take_selection_manifest,
    canonical_payload_sha256,
    select_fresh_takes,
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
    )


def test_selector_uses_salted_hash_and_reports_exhausted_objects() -> None:
    takes = ("A_T1", "A_T2", "B_T1", "C_T1", "C_T2")
    selected, exhausted = select_fresh_takes(
        takes,
        ("A_T1", "B_T1"),
        salt_label="fresh-v2",
    )

    assert len(selected) == 2
    assert {value.rpartition("_T")[0] for value in selected} == {"A", "C"}
    assert exhausted == ("B",)


def test_manifest_scans_reachable_history_not_only_current_tree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    evidence = repository / "evidence.json"
    evidence.write_text(json.dumps({"take": "A_T1"}), encoding="utf-8")
    _git(repository, "add", "evidence.json")
    _git(repository, "commit", "-qm", "Add old evidence")
    evidence.write_text("{}", encoding="utf-8")
    _git(repository, "commit", "-qam", "Remove visible reference")

    manifest = build_fresh_take_selection_manifest(
        repository,
        ("A_T1", "A_T2", "B_T1"),
        salt_label="fresh-v2",
        selection_id="test-selection",
        created_at_utc="2026-08-03T00:00:00Z",
    )

    assert manifest["referenced_take_count"] == 1
    assert manifest["eligible_object_count"] == 2
    assert "A_T1" not in manifest["selected_take_ids"]
    assert manifest["selection_manifest_sha256"] == canonical_payload_sha256(
        manifest, digest_field="selection_manifest_sha256"
    )
    assert manifest["repository_clean"] is True
