from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin.cli.run_manifest import main
from bayesian_phystwin.paper_evidence_v1 import PAPER_EVIDENCE_PROFILE_KEY
from bayesian_phystwin.run_manifest_v2 import (
    RunManifestV2,
    load_run_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_embeds_and_requires_paper_evidence_profile(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    files = {
        "provider_manifest": root / "provider-manifest.json",
        "observation_belief": root / "observation-belief.npz",
        "twin_belief": root / "twin-belief.json",
        "bayesian_phystwin_wheel": root / "bayesian_phystwin.whl",
        "bayesian_phystwin_sdist": root / "bayesian_phystwin.tar.gz",
    }
    for name, path in files.items():
        path.write_bytes(f"{name}\n".encode())

    profile = {
        "schema_name": "bayesian_phystwin.paper_evidence_bindings",
        "schema_version": 1,
        "primary_distribution_project": "bayesian-phystwin",
        "provider_manifest": {
            "artifact_name": "provider_manifest",
            "artifact_id": _sha256(files["provider_manifest"]),
            "role": "input",
        },
        "prob4d_stream_contract": {
            "version": 2,
            "resolution": "declared",
        },
        "observation_belief": {
            "artifact_name": "observation_belief",
            "artifact_id": _sha256(files["observation_belief"]),
            "role": "input",
        },
        "twin_belief": {
            "artifact_name": "twin_belief",
            "artifact_id": _sha256(files["twin_belief"]),
            "role": "output",
        },
        "distributions": [
            {
                "project": "bayesian-phystwin",
                "kind": "wheel",
                "artifact_name": "bayesian_phystwin_wheel",
                "artifact_id": _sha256(files["bayesian_phystwin_wheel"]),
            },
            {
                "project": "bayesian-phystwin",
                "kind": "sdist",
                "artifact_name": "bayesian_phystwin_sdist",
                "artifact_id": _sha256(files["bayesian_phystwin_sdist"]),
            },
        ],
    }
    profile_path = tmp_path / "paper-evidence.json"
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"

    arguments = [
        "create",
        str(manifest_path),
        "--run-id",
        "paper-evidence-cli",
        "--revision",
        "a" * 40,
        "--classification",
        "confirmatory",
        "--statistical-unit",
        "interaction",
        "--command-line",
        "bpt run manifest create",
        "--artifact-root",
        str(root),
        "--paper-evidence-json",
        str(profile_path),
        "--claim-id",
        "bpt.paper_evidence_cli",
        "--method-freeze-id",
        "method-v1",
        "--protocol-id",
        "protocol-v1",
        "--split-id",
        "split-v1",
        "--baseline-id",
        "baseline-v1",
        "--input",
        "provider_manifest=provider-manifest.json",
        "--input",
        "observation_belief=observation-belief.npz",
        "--input",
        "bayesian_phystwin_wheel=bayesian_phystwin.whl",
        "--input",
        "bayesian_phystwin_sdist=bayesian_phystwin.tar.gz",
        "--output-artifact",
        "twin_belief=twin-belief.json",
    ]

    assert main(arguments) == 0
    capsys.readouterr()
    manifest = load_run_manifest(manifest_path)
    assert isinstance(manifest, RunManifestV2)
    assert PAPER_EVIDENCE_PROFILE_KEY in manifest.information_boundary

    assert (
        main(
            [
                "validate",
                str(manifest_path),
                "--artifact-root",
                str(root),
                "--require-paper-evidence",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"paper_evidence_profile": "valid"' in output
