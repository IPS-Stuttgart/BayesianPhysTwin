from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.run_manifest import (
    RunManifestV1,
    artifact_digest,
    load_run_manifest,
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
        command=("bpt", "benchmark", "synthetic"),
        classification="infrastructure",
        statistical_unit="test case",
        information_boundary={"causal_frame_stop": 10},
        configuration={"seeds": [1, 2]},
        seeds=(1, 2),
        inputs=(artifact_digest(source, name="input", role="input", root=root),),
        outputs=(artifact_digest(result, name="output", role="output", root=root),),
        package_versions={"bayesian-phystwin": "0.4.0"},
        created_utc="2026-07-26T18:00:00+00:00",
    )


def test_run_manifest_round_trip_and_artifact_verification(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)

    loaded = load_run_manifest(path)

    assert loaded == manifest
    assert loaded.manifest_id == manifest.manifest_id
    verify_run_manifest_artifacts(loaded, root=tmp_path)


def test_run_manifest_rejects_payload_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["classification"] = "confirmatory"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_run_manifest(path)


def test_run_manifest_detects_artifact_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "output.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact .* mismatch"):
        verify_run_manifest_artifacts(manifest, root=tmp_path)
