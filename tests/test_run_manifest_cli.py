from __future__ import annotations

from pathlib import Path

from bayesian_phystwin.cli.run_manifest import main
from bayesian_phystwin.run_manifest import load_run_manifest


def test_run_manifest_cli_create_and_validate(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

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
                str(tmp_path),
                "--output-artifact",
                f"result={artifact}",
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
                str(tmp_path),
            ]
        )
        == 0
    )
    assert '"status": "valid"' in capsys.readouterr().out
