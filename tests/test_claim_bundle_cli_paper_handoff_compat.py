from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import test_paper_handoff_v1 as paper_cases

from bayesian_phystwin import claim_bundle_v1
from bayesian_phystwin import paper_handoff_v1 as handoff
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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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


def test_cli_strict_error_paths_and_generic_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "build_claim_bundle", lambda **_kwargs: _Bundle())
    build_args = _build_args(tmp_path, strict=True)
    build_args.claim_binding = None
    with pytest.raises(ValueError, match="requires --claim-binding"):
        cli._build(build_args)

    monkeypatch.setattr(cli, "load_claim_bundle", lambda _path: _Bundle())
    require_binding = _validate_args(tmp_path, strict=False)
    require_binding.require_claim_binding = True
    with pytest.raises(ValueError, match="no claim-binding artifact"):
        cli._validate(require_binding)

    missing_root = _validate_args(tmp_path, strict=True)
    missing_root.artifact_root = None
    with pytest.raises(ValueError, match="requires --artifact-root"):
        cli._validate(missing_root)

    calls: list[tuple[object, Path]] = []
    rehash = _validate_args(tmp_path, strict=False)
    rehash.artifact_root = tmp_path
    monkeypatch.setattr(
        cli,
        "verify_claim_bundle_artifacts",
        lambda bundle, *, root: calls.append((bundle, root)),
    )
    assert cli._validate(rehash) == 0
    assert calls == [(_Bundle(), tmp_path)]


def test_paper_validate_and_unhandled_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _Bundle(artifacts=(_Artifact(kind="claim_binding"),))
    monkeypatch.setattr(cli, "load_claim_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        cli,
        "verify_claim_bundle_paper_handoff",
        lambda *_args, **_kwargs: {
            "binding_claim_count": 1,
            "compact_table_count": 1,
            "compact_table_row_count": 1,
        },
    )
    args = argparse.Namespace(
        bundle=tmp_path / "bundle.json",
        artifact_root=tmp_path,
    )
    assert cli._paper_validate(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["paper_handoff_verified"] is True
    assert output["compact_table_row_count"] == 1

    class _Parser:
        @staticmethod
        def parse_args(_argv: object) -> argparse.Namespace:
            return argparse.Namespace(command_name="unknown")

    monkeypatch.setattr(cli, "build_parser", lambda: _Parser())
    with pytest.raises(AssertionError, match="unhandled command"):
        cli.main([])


def _rewrite_binding(
    bundle: paper_cases._Bundle,
    root: Path,
    mutation: Any,
) -> paper_cases._Bundle:
    path = root / "claim-binding.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return paper_cases._with_rewritten_binding(bundle, root)


def _write_direct_table(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_table_payload() -> dict[str, object]:
    return {
        "schema_name": handoff.COMPACT_CLAIM_TABLE_SCHEMA,
        "schema_version": handoff.COMPACT_CLAIM_TABLE_SCHEMA_VERSION,
        "rows": [
            {
                "id": "bpt.claim.primary",
                "claim_id": "bpt.claim",
                "evidence": [],
            }
        ],
    }


def test_paper_handoff_strict_scalar_and_schema_helpers() -> None:
    with pytest.raises(ValueError, match="literal strings"):
        handoff._require_exact_fields({1: "value"}, expected=frozenset(), name="x")
    with pytest.raises(ValueError, match="missing"):
        handoff._require_exact_fields({}, expected=frozenset({"a"}), name="x")
    with pytest.raises(ValueError, match="unknown"):
        handoff._require_exact_fields(
            {"a": 1, "b": 2},
            expected=frozenset({"a"}),
            name="x",
        )
    with pytest.raises(ValueError, match="JSON object"):
        handoff._require_mapping([], name="x")
    with pytest.raises(ValueError, match="JSON array"):
        handoff._require_sequence("x", name="x")
    with pytest.raises(ValueError, match="nonempty literal text"):
        handoff._require_text("", name="x")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        handoff._require_text(" x ", name="x")
    with pytest.raises(ValueError, match="integer"):
        handoff._require_integer(True, name="x")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        handoff._require_sha256("A" * 64, name="x")
    for value in ("a\\b", "/absolute", ".", "a/../b", "a//b"):
        with pytest.raises(ValueError):
            handoff._require_relative_path(value, name="x")
    assert handoff._require_binding_root(".", name="x").as_posix() == "."


def test_paper_handoff_strict_json_reader_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be read"):
        handoff._load_json_mapping(tmp_path / "missing.json", name="payload")

    invalid_utf8 = tmp_path / "invalid.json"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="cannot be read"):
        handoff._load_json_mapping(invalid_utf8, name="payload")


def test_paper_handoff_snapshot_and_artifact_byte_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="cannot be opened"):
        handoff._stable_regular_file_snapshot(tmp_path / "missing.bin")

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        handoff._stable_regular_file_snapshot(directory)

    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    real_fstat = handoff.os.fstat
    calls = 0

    def changed_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        if calls == 1:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(handoff.os, "fstat", changed_fstat)
    with pytest.raises(ValueError, match="changed while hashing"):
        handoff._stable_regular_file_snapshot(source)

    monkeypatch.setattr(handoff.os, "fstat", real_fstat)
    artifact = paper_cases._artifact(
        source,
        name="source",
        kind="supporting",
        root=tmp_path,
    )
    with pytest.raises(ValueError, match="size differs"):
        handoff._verify_artifact_bytes(
            replace(artifact, size_bytes=artifact.size_bytes + 1),
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="digest differs"):
        handoff._verify_artifact_bytes(
            replace(artifact, sha256="0" * 64),
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="is not a file"):
        handoff._verify_artifact_bytes(
            replace(artifact, path="missing.bin"),
            root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(schema_name="wrong"), "schema"),
        (lambda payload: payload.update(schema_version=2), "schema version"),
        (lambda payload: payload.update(rows="not-an-array"), "JSON array"),
        (lambda payload: payload.update(rows=["not-an-object"]), "JSON object"),
        (
            lambda payload: payload["rows"][0].update(unknown=True),
            "does not match schema",
        ),
        (lambda payload: payload["rows"][0].update(id=""), "nonempty"),
        (
            lambda payload: payload["rows"][0].update(evidence="not-an-array"),
            "JSON array",
        ),
    ),
)
def test_direct_compact_table_schema_failures(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    payload = _valid_table_payload()
    mutation(payload)
    table = _write_direct_table(tmp_path / "table.json", payload)
    with pytest.raises(ValueError, match=message):
        handoff._verify_compact_table_row(
            table,
            row_id="bpt.claim.primary",
            claim_id="bpt.claim",
        )


def test_paper_handoff_binding_inventory_and_root_schema_failures(
    tmp_path: Path,
) -> None:
    bundle = paper_cases._bundle(tmp_path)
    with pytest.raises(ValueError, match="exactly one claim-binding"):
        handoff.verify_compact_claim_table_bindings(
            replace(bundle, artifacts=bundle.artifacts[1:]),
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="exactly one claim-binding"):
        handoff.verify_compact_claim_table_bindings(
            replace(
                bundle,
                artifacts=(bundle.artifacts[0], bundle.artifacts[0], *bundle.artifacts[1:]),
            ),
            root=tmp_path,
        )

    duplicate_path = replace(bundle.artifacts[1], path=bundle.artifacts[2].path)
    with pytest.raises(ValueError, match="duplicate claim-bundle artifact path"):
        handoff.verify_compact_claim_table_bindings(
            replace(bundle, artifacts=(bundle.artifacts[0], duplicate_path, bundle.artifacts[2])),
            root=tmp_path,
        )

    wrong_schema = _rewrite_binding(
        bundle,
        tmp_path,
        lambda payload: payload.update(schema_name="wrong"),
    )
    with pytest.raises(ValueError, match="binding schema"):
        handoff.verify_compact_claim_table_bindings(wrong_schema, root=tmp_path)

    bundle = paper_cases._bundle(tmp_path)
    wrong_version = _rewrite_binding(
        bundle,
        tmp_path,
        lambda payload: payload.update(schema_version=2),
    )
    with pytest.raises(ValueError, match="schema version"):
        handoff.verify_compact_claim_table_bindings(wrong_version, root=tmp_path)

    for claim_ids in ((), ("bpt.claim", "bpt.claim"), ("",)):
        bundle = paper_cases._bundle(tmp_path, claim_ids=claim_ids)
        with pytest.raises(ValueError, match="unique and nonempty"):
            handoff.verify_compact_claim_table_bindings(bundle, root=tmp_path)


def test_paper_handoff_bound_artifact_failures(tmp_path: Path) -> None:
    bundle = paper_cases._bundle(tmp_path)
    with pytest.raises(ValueError, match="absent from the claim bundle"):
        handoff.verify_compact_claim_table_bindings(
            replace(bundle, artifacts=(bundle.artifacts[0], bundle.artifacts[2])),
            root=tmp_path,
        )

    wrong_kind = replace(bundle.artifacts[1], kind="figure")
    with pytest.raises(ValueError, match="absent from the claim bundle"):
        handoff.verify_compact_claim_table_bindings(
            replace(bundle, artifacts=(bundle.artifacts[0], wrong_kind, bundle.artifacts[2])),
            root=tmp_path,
        )

    wrong_size = replace(bundle.artifacts[1], size_bytes=bundle.artifacts[1].size_bytes + 1)
    with pytest.raises(ValueError, match="size differs"):
        handoff.verify_compact_claim_table_bindings(
            replace(bundle, artifacts=(bundle.artifacts[0], wrong_size, bundle.artifacts[2])),
            root=tmp_path,
        )

    binding_path = tmp_path / "claim-binding.json"
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload["bindings"][0]["result_artifact"]["sha256"] = "0" * 64
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    changed_binding = paper_cases._artifact(
        binding_path,
        name="paper_claim_binding",
        kind="claim_binding",
        root=tmp_path,
    )
    changed_result = replace(bundle.artifacts[1], sha256="0" * 64)
    changed = replace(
        bundle,
        artifacts=(changed_binding, changed_result, bundle.artifacts[2]),
    )
    with pytest.raises(ValueError, match="digest differs"):
        handoff.verify_compact_claim_table_bindings(changed, root=tmp_path)


def test_paper_handoff_duplicate_bindings_and_migration_exceptions(
    tmp_path: Path,
) -> None:
    bundle = paper_cases._bundle(tmp_path)
    duplicate = _rewrite_binding(
        bundle,
        tmp_path,
        lambda payload: payload["bindings"].append(dict(payload["bindings"][0])),
    )
    with pytest.raises(ValueError, match="duplicate claim-evidence binding"):
        handoff.verify_compact_claim_table_bindings(duplicate, root=tmp_path)

    bundle = paper_cases._bundle(tmp_path)
    overlap = _rewrite_binding(
        bundle,
        tmp_path,
        lambda payload: payload["migration_exceptions"].append(
            {"claim_id": "bpt.claim", "reason": "legacy"}
        ),
    )
    with pytest.raises(ValueError, match="cannot rely on migration exceptions"):
        handoff.verify_compact_claim_table_bindings(overlap, root=tmp_path)

    bundle = paper_cases._bundle(tmp_path)
    duplicate_exception = _rewrite_binding(
        bundle,
        tmp_path,
        lambda payload: payload["migration_exceptions"].extend(
            [
                {"claim_id": "legacy", "reason": "one"},
                {"claim_id": "legacy", "reason": "two"},
            ]
        ),
    )
    with pytest.raises(ValueError, match="duplicate claim-evidence migration"):
        handoff.verify_compact_claim_table_bindings(
            duplicate_exception,
            root=tmp_path,
        )

    bundle = paper_cases._bundle(tmp_path)
    malformed_exception = _rewrite_binding(
        bundle,
        tmp_path,
        lambda payload: payload.update(migration_exceptions=[{"claim_id": "legacy"}]),
    )
    with pytest.raises(ValueError, match="does not match schema"):
        handoff.verify_compact_claim_table_bindings(
            malformed_exception,
            root=tmp_path,
        )


def test_full_paper_handoff_runs_generic_and_compact_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = paper_cases._bundle(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        claim_bundle_v1,
        "verify_claim_bundle_artifacts",
        lambda *_args, **_kwargs: calls.append("generic"),
    )
    monkeypatch.setattr(
        handoff,
        "verify_compact_claim_table_bindings",
        lambda *_args, **_kwargs: calls.append("compact") or {"ok": True},
    )

    assert handoff.verify_claim_bundle_paper_handoff(bundle, root=tmp_path) == {
        "ok": True
    }
    assert calls == ["generic", "compact"]
