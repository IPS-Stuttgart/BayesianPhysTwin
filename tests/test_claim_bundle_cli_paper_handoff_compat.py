from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from bayesian_phystwin.cli import claim_bundle as cli


@dataclass(frozen=True)
class _Artifact:
    kind: str


@dataclass(frozen=True)
class _Bundle:
    bundle_id: str = "a" * 64
    run_manifest_id: str = "b" * 64
    evidence_fingerprint: str = "c" * 64
    run_id: str = "run-v1"
    classification: str = "controlled"
    claim_ids: tuple[str, ...] = ("bpt.claim",)
    repositories: tuple[str, ...] = ("primary",)
    artifacts: tuple[_Artifact, ...] = ()


def _build_args(tmp_path: Path, *, strict: bool) -> argparse.Namespace:
    return argparse.Namespace(
        artifact_root=tmp_path,
        run_manifest=tmp_path / "run-manifest.json",
        evidence_summary=tmp_path / "evidence-summary.json",
        claim_binding=(tmp_path / "claim-binding.json") if strict else None,
        figure=[],
        table_data=[],
        supporting=[],
        verify_paper_handoff=strict,
        bundle=tmp_path / "claim-bundle.json",
        force=False,
    )


def test_generic_build_output_keeps_the_stable_key_set(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "build_claim_bundle", lambda **_kwargs: _Bundle())
    monkeypatch.setattr(cli, "write_claim_bundle", lambda *_args, **_kwargs: None)

    assert cli._build(_build_args(tmp_path, strict=False)) == 0
    output = json.loads(capsys.readouterr().out)

    assert set(output) == {
        "artifact_count",
        "bundle",
        "bundle_id",
        "claim_count",
        "evidence_fingerprint",
        "repository_count",
        "run_manifest_id",
        "schema_version",
    }


def test_strict_build_output_adds_paper_handoff_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "build_claim_bundle", lambda **_kwargs: _Bundle())
    monkeypatch.setattr(cli, "write_claim_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "verify_claim_bundle_paper_handoff",
        lambda *_args, **_kwargs: {"compact_table_row_count": 1},
    )

    assert cli._build(_build_args(tmp_path, strict=True)) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["paper_handoff_verified"] is True
    assert output["paper_handoff"] == {"compact_table_row_count": 1}


def _validate_args(tmp_path: Path, *, strict: bool) -> argparse.Namespace:
    return argparse.Namespace(
        bundle=tmp_path / "claim-bundle.json",
        artifact_root=tmp_path if strict else None,
        require_claim_binding=False,
        verify_paper_handoff=strict,
    )


def test_generic_validate_output_keeps_the_stable_key_set(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "load_claim_bundle", lambda _path: _Bundle())

    assert cli._validate(_validate_args(tmp_path, strict=False)) == 0
    output = json.loads(capsys.readouterr().out)

    assert set(output) == {
        "artifacts_verified",
        "bundle_id",
        "claim_binding",
        "claim_count",
        "classification",
        "run_id",
        "schema_version",
        "status",
    }


def test_strict_validate_output_adds_paper_handoff_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    bundle = _Bundle(artifacts=(_Artifact(kind="claim_binding"),))
    monkeypatch.setattr(cli, "load_claim_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        cli,
        "verify_claim_bundle_paper_handoff",
        lambda *_args, **_kwargs: {"compact_table_row_count": 1},
    )

    assert cli._validate(_validate_args(tmp_path, strict=True)) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["paper_handoff_verified"] is True
    assert output["paper_handoff"] == {"compact_table_row_count": 1}
