from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from bayesian_phystwin.paper_handoff_v1 import (
    CLAIM_EVIDENCE_BINDING_SCHEMA,
    CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION,
    COMPACT_CLAIM_TABLE_SCHEMA,
    COMPACT_CLAIM_TABLE_SCHEMA_VERSION,
    verify_compact_claim_table_bindings,
)


@dataclass(frozen=True)
class _Artifact:
    name: str
    kind: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _Bundle:
    claim_ids: tuple[str, ...]
    artifacts: tuple[_Artifact, ...]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, *, name: str, kind: str, root: Path) -> _Artifact:
    return _Artifact(
        name=name,
        kind=kind,
        path=path.relative_to(root).as_posix(),
        sha256=_digest(path),
        size_bytes=path.stat().st_size,
    )


def _write_table(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_name": COMPACT_CLAIM_TABLE_SCHEMA,
                "schema_version": COMPACT_CLAIM_TABLE_SCHEMA_VERSION,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _bundle(
    root: Path,
    *,
    claim_ids: tuple[str, ...] = ("bpt.claim",),
    rows: list[dict] | None = None,
    table_text: str | None = None,
    table_row_id: str = "bpt.claim.primary",
    binding_claim_id: str = "bpt.claim",
    artifact_root: str = ".",
) -> _Bundle:
    result_path = root / "result.json"
    result_path.write_text('{"status":"complete"}\n', encoding="utf-8")
    table_path = root / "table.json"
    if table_text is not None:
        table_path.write_text(table_text, encoding="utf-8")
    else:
        _write_table(
            table_path,
            rows
            if rows is not None
            else [
                {
                    "id": "bpt.claim.primary",
                    "claim_id": "bpt.claim",
                    "evidence": [{"estimate": 1.0, "metric": "loss"}],
                }
            ],
        )

    result_artifact = _artifact(
        result_path,
        name="compact_result",
        kind="supporting",
        root=root,
    )
    table_artifact = _artifact(
        table_path,
        name="compact_table",
        kind="table_data",
        root=root,
    )
    binding_path = root / "claim-binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "schema_name": CLAIM_EVIDENCE_BINDING_SCHEMA,
                "schema_version": CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION,
                "bindings": [
                    {
                        "claim_id": binding_claim_id,
                        "manifest": "run-manifest.json",
                        "artifact_root": artifact_root,
                        "expected_manifest_id": "a" * 64,
                        "expected_evidence_fingerprint": "b" * 64,
                        "result_artifact": {
                            "name": result_artifact.name,
                            "path": result_artifact.path,
                            "sha256": result_artifact.sha256,
                        },
                        "table_artifact": {
                            "name": table_artifact.name,
                            "path": table_artifact.path,
                            "sha256": table_artifact.sha256,
                        },
                        "table_row_id": table_row_id,
                    }
                ],
                "migration_exceptions": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    binding_artifact = _artifact(
        binding_path,
        name="paper_claim_binding",
        kind="claim_binding",
        root=root,
    )
    return _Bundle(
        claim_ids=claim_ids,
        artifacts=(binding_artifact, result_artifact, table_artifact),
    )


def _with_rewritten_binding(bundle: _Bundle, root: Path) -> _Bundle:
    binding_artifact = _artifact(
        root / "claim-binding.json",
        name="paper_claim_binding",
        kind="claim_binding",
        root=root,
    )
    return _Bundle(
        claim_ids=bundle.claim_ids,
        artifacts=(binding_artifact, *bundle.artifacts[1:]),
    )


def test_compact_claim_table_binding_verifies_real_row(tmp_path: Path) -> None:
    summary = verify_compact_claim_table_bindings(
        _bundle(tmp_path),
        root=tmp_path,
    )

    assert summary == {
        "binding_claim_count": 1,
        "compact_table_count": 1,
        "compact_table_row_count": 1,
    }


def test_compact_claim_table_binding_rejects_missing_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    missing = _bundle(tmp_path, table_row_id="bpt.claim.missing")
    with pytest.raises(ValueError, match="has no row"):
        verify_compact_claim_table_bindings(missing, root=tmp_path)

    rows = [
        {
            "id": "bpt.claim.primary",
            "claim_id": "bpt.claim",
            "evidence": [],
        },
        {
            "id": "bpt.claim.primary",
            "claim_id": "bpt.claim",
            "evidence": [],
        },
    ]
    duplicate = _bundle(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="duplicate compact claim table row"):
        verify_compact_claim_table_bindings(duplicate, root=tmp_path)


def test_compact_claim_table_binding_rejects_wrong_claim_and_claim_set(
    tmp_path: Path,
) -> None:
    wrong_row = _bundle(
        tmp_path,
        rows=[
            {
                "id": "bpt.claim.primary",
                "claim_id": "bpt.other",
                "evidence": [],
            }
        ],
    )
    with pytest.raises(ValueError, match="bound to another claim"):
        verify_compact_claim_table_bindings(wrong_row, root=tmp_path)

    wrong_set = _bundle(tmp_path, claim_ids=("bpt.other",))
    with pytest.raises(ValueError, match="do not match bundle claim IDs"):
        verify_compact_claim_table_bindings(wrong_set, root=tmp_path)


def test_compact_claim_table_binding_rejects_malformed_json_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    malformed = _bundle(tmp_path, table_text="not JSON\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        verify_compact_claim_table_bindings(malformed, root=tmp_path)

    duplicate_key = _bundle(
        tmp_path,
        table_text=(
            '{"schema_name":"bayesian_phystwin.compact_claim_table",'
            '"schema_name":"bayesian_phystwin.compact_claim_table",'
            '"schema_version":1,"rows":[]}\n'
        ),
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_compact_claim_table_bindings(duplicate_key, root=tmp_path)


def test_compact_claim_table_binding_rejects_nonfinite_json_constant(
    tmp_path: Path,
) -> None:
    nonfinite = _bundle(
        tmp_path,
        table_text=(
            '{"schema_name":"bayesian_phystwin.compact_claim_table",'
            '"schema_version":1,"rows":[{"id":"bpt.claim.primary",'
            '"claim_id":"bpt.claim","evidence":[NaN]}]}\n'
        ),
    )

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        verify_compact_claim_table_bindings(nonfinite, root=tmp_path)


def test_compact_claim_table_binding_rejects_surrounding_identifier_whitespace(
    tmp_path: Path,
) -> None:
    padded = _bundle(tmp_path, binding_claim_id=" bpt.claim ")

    with pytest.raises(ValueError, match="surrounding whitespace"):
        verify_compact_claim_table_bindings(padded, root=tmp_path)


def test_compact_claim_table_binding_rejects_reference_name_drift(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    binding_path = tmp_path / "claim-binding.json"
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload["bindings"][0]["table_artifact"]["name"] = "another_table"
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    altered = _with_rewritten_binding(bundle, tmp_path)

    with pytest.raises(ValueError, match="selects another bundle artifact name"):
        verify_compact_claim_table_bindings(altered, root=tmp_path)


def test_compact_claim_table_binding_rejects_noncanonical_reference_path(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    binding_path = tmp_path / "claim-binding.json"
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload["bindings"][0]["table_artifact"]["path"] = "./table.json"
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    altered = _with_rewritten_binding(bundle, tmp_path)

    with pytest.raises(ValueError, match="canonical normalized relative path"):
        verify_compact_claim_table_bindings(altered, root=tmp_path)


def test_compact_claim_table_binding_rejects_escaping_root_and_symlink(
    tmp_path: Path,
) -> None:
    escaping = _bundle(tmp_path, artifact_root="../outside")
    with pytest.raises(ValueError, match="normalized relative path"):
        verify_compact_claim_table_bindings(escaping, root=tmp_path)

    target = tmp_path / "target-table.json"
    _write_table(
        target,
        [
            {
                "id": "bpt.claim.primary",
                "claim_id": "bpt.claim",
                "evidence": [],
            }
        ],
    )
    link = tmp_path / "table.json"
    link.unlink(missing_ok=True)
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    result_path = tmp_path / "result.json"
    result_artifact = _artifact(
        result_path,
        name="compact_result",
        kind="supporting",
        root=tmp_path,
    )
    table_artifact = _Artifact(
        name="compact_table",
        kind="table_data",
        path="table.json",
        sha256=_digest(target),
        size_bytes=target.stat().st_size,
    )
    binding_path = tmp_path / "claim-binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "schema_name": CLAIM_EVIDENCE_BINDING_SCHEMA,
                "schema_version": 1,
                "bindings": [
                    {
                        "claim_id": "bpt.claim",
                        "manifest": "run-manifest.json",
                        "artifact_root": ".",
                        "expected_manifest_id": "a" * 64,
                        "expected_evidence_fingerprint": "b" * 64,
                        "result_artifact": {
                            "name": result_artifact.name,
                            "path": result_artifact.path,
                            "sha256": result_artifact.sha256,
                        },
                        "table_artifact": {
                            "name": table_artifact.name,
                            "path": table_artifact.path,
                            "sha256": table_artifact.sha256,
                        },
                        "table_row_id": "bpt.claim.primary",
                    }
                ],
                "migration_exceptions": [],
            }
        ),
        encoding="utf-8",
    )
    binding_artifact = _artifact(
        binding_path,
        name="paper_claim_binding",
        kind="claim_binding",
        root=tmp_path,
    )
    symlink_bundle = _Bundle(
        claim_ids=("bpt.claim",),
        artifacts=(binding_artifact, result_artifact, table_artifact),
    )
    with pytest.raises(ValueError, match="must not use symbolic links"):
        verify_compact_claim_table_bindings(symlink_bundle, root=tmp_path)
