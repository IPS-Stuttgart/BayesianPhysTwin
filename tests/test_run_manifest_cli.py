from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.cli import run_manifest as run_manifest_cli
from bayesian_phystwin.cli.run_manifest import main
from bayesian_phystwin.run_manifest import (
    RunManifestV1,
    artifact_digest,
)
from bayesian_phystwin.run_manifest import (
    write_run_manifest as write_run_manifest_v1,
)
from bayesian_phystwin.run_manifest_v2 import (
    RunManifestV2,
    load_run_manifest,
)


def test_run_manifest_optional_inputs_default_empty() -> None:
    assert run_manifest_cli._load_json_mapping(None, name="configuration") == {}
    assert run_manifest_cli._load_repository_states(None) == ()


def test_run_manifest_helpers_reject_invalid_json_shapes(tmp_path: Path) -> None:
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration JSON must contain an object"):
        run_manifest_cli._load_json_mapping(
            configuration_path,
            name="configuration",
        )

    repositories_path = tmp_path / "repositories.json"
    repositories_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an array"):
        run_manifest_cli._load_repository_states(repositories_path)

    repositories_path.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record 0 must contain an object"):
        run_manifest_cli._load_repository_states(repositories_path)


def test_run_manifest_cli_creates_v2_and_validates_both_versions(
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
        json.dumps(
            {
                "gpu_model": "test-gpu",
                "container_digest": "sha256:test",
            }
        ),
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
                "bpt.infrastructure.run_manifest_v2",
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
    assert '"schema_version": 2' in create_output
    assert '"evidence_fingerprint"' in create_output
    manifest = load_run_manifest(manifest_path)
    assert isinstance(manifest, RunManifestV2)
    assert manifest.outputs[0].path == "result.json"
    assert manifest.related_repositories[0].repository == "Jianghanxiao/PhysTwin"
    assert manifest.runtime_environment["gpu_model"] == "test-gpu"

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
    validation_output = capsys.readouterr().out
    assert '"status": "valid"' in validation_output
    assert '"schema_version": 2' in validation_output

    legacy_path = tmp_path / "legacy.json"
    legacy = RunManifestV1(
        run_id="legacy",
        repository="FlorianPfaff/Bayesian-PhysTwin",
        revision="legacy-revision",
        dirty=False,
        command=("bpt-provider-manifest",),
        classification="infrastructure",
        statistical_unit="test case",
        information_boundary={},
        configuration={},
        outputs=(
            artifact_digest(
                artifact,
                name="result",
                role="output",
                root=artifact_root,
            ),
        ),
        created_utc="2026-07-26T18:00:00+00:00",
    )
    write_run_manifest_v1(legacy_path, legacy)

    assert (
        main(
            [
                "validate",
                str(legacy_path),
                "--artifact-root",
                str(artifact_root),
            ]
        )
        == 0
    )
    legacy_output = capsys.readouterr().out
    assert '"schema_version": 1' in legacy_output
    assert '"status": "valid"' in legacy_output


def test_run_manifest_v2_rejects_empty_run_id() -> None:
    with pytest.raises(ValueError, match="run ID must be nonempty"):
        RunManifestV2(
            run_id="",
            repository="FlorianPfaff/Bayesian-PhysTwin",
            revision="a" * 40,
            dirty=False,
            command=("bpt", "provider", "manifest"),
            classification="infrastructure",
            statistical_unit="test case",
            information_boundary={},
            configuration={},
        )
