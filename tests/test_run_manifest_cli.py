from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.cli.run_manifest import main
from bayesian_phystwin.run_manifest import load_run_manifest


def test_run_manifest_cli_create_and_validate(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    artifact_root = tmp_path / "bundle"
    artifact_root.mkdir()
    artifact = artifact_root / "result.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    related_path = tmp_path / "repositories.json"
    related_path.write_text(
        json.dumps(
            [
                {
                    "repository": "Jianghanxiao/PhysTwin",
                    "revision": "c" * 40,
                    "dirty": False,
                    "role": "upstream",
                }
            ]
        ),
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps({"gpu_model": "test-gpu", "container_digest": "sha256:test"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "create",
                str(manifest_path),
                "--run-id",
                "smoke",
                "--revision",
                "b" * 40,
                "--classification",
                "infrastructure",
                "--statistical-unit",
                "test case",
                "--command-line",
                "bpt provider manifest",
                "--artifact-root",
                str(artifact_root),
                "--output-artifact",
                "result=result.json",
                "--related-repositories-json",
                str(related_path),
                "--runtime-json",
                str(runtime_path),
                "--claim-id",
                "bpt.infrastructure.run_manifest_v1",
                "--method-freeze-id",
                "method-v1",
                "--protocol-id",
                "protocol-v1",
                "--split-id",
                "split-v1",
                "--baseline-id",
                "baseline-v1",
            ]
        )
        == 0
    )
    create_output = capsys.readouterr().out
    assert '"evidence_fingerprint"' in create_output
    manifest = load_run_manifest(manifest_path)
    assert manifest.outputs[0].path == "result.json"
    assert manifest.related_repositories[0].repository == "Jianghanxiao/PhysTwin"
    assert manifest.runtime_environment["gpu_model"] == "test-gpu"
    assert manifest.claim_ids == ("bpt.infrastructure.run_manifest_v1",)

    assert (
        main(
            [
                "validate",
                str(manifest_path),
                "--artifact-root",
                str(artifact_root),
            ]
        )
        == 0
    )
    assert '"status": "valid"' in capsys.readouterr().out
