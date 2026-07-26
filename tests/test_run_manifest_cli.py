from __future__ import annotations

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
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert load_run_manifest(manifest_path).outputs[0].path == "result.json"

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
