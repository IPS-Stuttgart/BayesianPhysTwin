from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.run_manifest import (
    ArtifactDigest,
    RepositoryState,
    RunManifestV1,
    artifact_digest,
    discover_git_repository_state,
    load_run_manifest,
    normalize_github_repository,
    verify_run_manifest_artifacts,
    write_run_manifest,
)


def _manifest(root: Path) -> RunManifestV1:
    source = root / "input.txt"
    result = root / "output.txt"
    source.write_text("input\n", encoding="utf-8")
    result.write_text("output\n", encoding="utf-8")
    return RunManifestV1(
        run_id="unit-test",
        repository="FlorianPfaff/Bayesian-PhysTwin",
        revision="a" * 40,
        dirty=False,
        related_repositories=(
            RepositoryState(
                repository="Jianghanxiao/PhysTwin",
                revision="b" * 40,
                dirty=False,
                role="upstream",
            ),
        ),
        command=("bpt", "benchmark", "synthetic"),
        classification="infrastructure",
        statistical_unit="test case",
        information_boundary={"causal_frame_stop": 10},
        configuration={"seeds": [1, 2]},
        seeds=(1, 2),
        inputs=(artifact_digest(source, name="input", role="input", root=root),),
        outputs=(artifact_digest(result, name="output", role="output", root=root),),
        package_versions={"bayesian-phystwin": "0.4.0"},
        runtime_environment={"python_version": "3.12.0"},
        claim_ids=("bpt.full22_anchor_released_contract",),
        method_freeze_id="method-v1",
        protocol_id="protocol-v1",
        split_id="split-v1",
        baseline_id="released-phystwin",
        created_utc="2026-07-26T18:00:00+00:00",
    )


def test_run_manifest_round_trip_and_artifact_verification(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)

    loaded = load_run_manifest(path)

    assert loaded == manifest
    assert loaded.manifest_id == manifest.manifest_id
    assert loaded.evidence_fingerprint == manifest.evidence_fingerprint
    verify_run_manifest_artifacts(loaded, root=tmp_path)


def test_evidence_fingerprint_ignores_timestamp_and_notes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    later = replace(
        manifest,
        created_utc="2026-07-27T18:00:00+00:00",
        notes="copied into the paper bundle",
    )

    assert later.evidence_fingerprint == manifest.evidence_fingerprint
    assert later.manifest_id != manifest.manifest_id


def test_run_manifest_rejects_payload_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["classification"] = "confirmatory"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint|digest"):
        load_run_manifest(path)


def test_run_manifest_rejects_uncovered_schema_fields(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["uncovered"] = "tampering"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match schema"):
        load_run_manifest(path)


def test_run_manifest_rejects_uncovered_artifact_fields(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outputs"][0]["uncovered"] = "tampering"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match schema"):
        load_run_manifest(path)


def test_run_manifest_detects_artifact_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "output.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact .* mismatch"):
        verify_run_manifest_artifacts(manifest, root=tmp_path)


def test_run_manifest_rejects_escaping_artifact_paths(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    manifest = replace(
        _manifest(tmp_path),
        outputs=(
            ArtifactDigest(
                name="outside",
                role="output",
                path="../outside.txt",
                sha256="0" * 64,
                size_bytes=outside.stat().st_size,
            ),
        ),
    )

    with pytest.raises(ValueError, match="escapes verification root"):
        verify_run_manifest_artifacts(manifest, root=tmp_path)


def test_repository_states_must_be_exact_and_unique(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="unique repository"):
        replace(
            manifest,
            related_repositories=(
                RepositoryState(
                    repository=manifest.repository,
                    revision="c" * 40,
                    dirty=False,
                    role="dependency",
                ),
            ),
        )
    with pytest.raises(ValueError, match="exact 40-character"):
        RepositoryState(
            repository="FlorianPfaff/Prob4D",
            revision="main",
            dirty=False,
            role="observation",
        )


def test_discover_git_repository_state_tracks_uncommitted_files(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "git@github.com:FlorianPfaff/Bayesian-PhysTwin.git",
        ],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "tracked.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
    )

    clean = discover_git_repository_state(tmp_path)
    assert clean.repository == "FlorianPfaff/Bayesian-PhysTwin"
    assert len(clean.revision) == 40
    assert clean.dirty is False

    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    assert discover_git_repository_state(tmp_path).dirty is True


def test_normalize_github_repository_accepts_https_and_ssh() -> None:
    expected = "FlorianPfaff/Bayesian-PhysTwin"
    assert (
        normalize_github_repository(
            "https://github.com/FlorianPfaff/Bayesian-PhysTwin.git"
        )
        == expected
    )
    assert (
        normalize_github_repository(
            "git@github.com:FlorianPfaff/Bayesian-PhysTwin.git"
        )
        == expected
    )
