from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesian_phystwin.claim_bundle_v1 import (
    CLAIM_EVIDENCE_BINDING_SCHEMA,
    CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION,
    ClaimBundleV1,
    claim_bundle_artifact,
    write_claim_bundle,
)
from bayesian_phystwin.cli.claim_bundle import main as claim_bundle_main
from bayesian_phystwin.paper_projection_v1 import (
    COMPACT_CLAIM_TABLE_SCHEMA,
    build_paper_projection,
    load_compact_claim_table_row,
    load_paper_projection,
    render_paper_projection_markdown,
)
from bayesian_phystwin.repository_provenance import RepositoryState

CLAIMS = ("bpt.claim.alpha", "bpt.claim.beta")


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": "bpt.claim.alpha.primary",
            "claim_id": "bpt.claim.alpha",
            "evidence": [{"estimate": -12.09, "metric": "chamfer"}],
        },
        {
            "id": "bpt.claim.beta.primary",
            "claim_id": "bpt.claim.beta",
            "evidence": {"status": "negative", "harmful_updates": 0},
        },
    ]


def _fixture(
    root: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    suffix: str = ".json",
) -> tuple[Path, SimpleNamespace]:
    manifest = root / "run-manifest.json"
    summary = root / "summary.json"
    table = root / f"compact{suffix}"
    binding = root / "claim-binding.json"
    manifest.write_text("{}\n", encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")
    _json(
        table,
        {
            "schema_name": COMPACT_CLAIM_TABLE_SCHEMA,
            "schema_version": 1,
            "rows": _rows() if rows is None else rows,
        },
    )
    manifest_artifact = claim_bundle_artifact(
        manifest, name="run_manifest", kind="run_manifest", root=root
    )
    result_artifact = claim_bundle_artifact(
        summary, name="summary", kind="evidence_summary", root=root
    )
    table_artifact = claim_bundle_artifact(
        table, name="compact_table", kind="table_data", root=root
    )
    _json(
        binding,
        {
            "schema_name": CLAIM_EVIDENCE_BINDING_SCHEMA,
            "schema_version": CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION,
            "migration_exceptions": [],
            "bindings": [
                {
                    "artifact_root": ".",
                    "claim_id": claim_id,
                    "expected_evidence_fingerprint": "b" * 64,
                    "expected_manifest_id": "a" * 64,
                    "manifest": manifest_artifact.path,
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
                    "table_row_id": f"{claim_id}.primary",
                }
                for claim_id in CLAIMS
            ],
        },
    )
    binding_artifact = claim_bundle_artifact(
        binding, name="claim_binding", kind="claim_binding", root=root
    )
    bundle = ClaimBundleV1(
        run_manifest_id="a" * 64,
        evidence_fingerprint="b" * 64,
        run_id="projection-test",
        classification="confirmatory",
        protocol_id="deform360-independent-object-v1",
        statistical_unit="physical object",
        claim_boundary="fresh-object evidence only",
        claim_ids=CLAIMS,
        method_freeze_id="method-v1",
        split_id="calibration-10-confirmation-12-v1",
        baseline_id="physical-fallback-v1",
        repositories=(
            RepositoryState(
                repository="IPS-Stuttgart/BayesianPhysTwin",
                revision="c" * 40,
                dirty=False,
                role="primary",
            ),
        ),
        artifacts=(
            manifest_artifact,
            result_artifact,
            table_artifact,
            binding_artifact,
        ),
    )
    bundle_path = root / "claim-bundle.json"
    write_claim_bundle(bundle_path, bundle)
    run_manifest = SimpleNamespace(
        outputs=tuple(
            SimpleNamespace(
                name=artifact.name,
                path=artifact.path,
                sha256=artifact.sha256,
                role="output",
            )
            for artifact in (result_artifact, table_artifact)
        )
    )
    return bundle_path, run_manifest


def _accept(monkeypatch: pytest.MonkeyPatch, manifest: object) -> None:
    monkeypatch.setattr(
        "bayesian_phystwin.paper_projection_v1.verify_claim_bundle_artifacts",
        lambda bundle, root: manifest,
    )


def test_compact_table_requires_one_unique_owned_row(tmp_path: Path) -> None:
    table = tmp_path / "compact.json"
    rows = _rows()
    _json(
        table,
        {"schema_name": COMPACT_CLAIM_TABLE_SCHEMA, "schema_version": 1, "rows": rows},
    )
    evidence = load_compact_claim_table_row(
        table, row_id="bpt.claim.alpha.primary", claim_id="bpt.claim.alpha"
    )
    assert evidence[0]["estimate"] == -12.09

    rows[0]["claim_id"] = "bpt.claim.beta"
    _json(
        table,
        {"schema_name": COMPACT_CLAIM_TABLE_SCHEMA, "schema_version": 1, "rows": rows},
    )
    with pytest.raises(ValueError, match="another claim"):
        load_compact_claim_table_row(
            table, row_id="bpt.claim.alpha.primary", claim_id="bpt.claim.alpha"
        )

    rows.append(dict(rows[0]))
    _json(
        table,
        {"schema_name": COMPACT_CLAIM_TABLE_SCHEMA, "schema_version": 1, "rows": rows},
    )
    with pytest.raises(ValueError, match="duplicate compact-table row ID"):
        load_compact_claim_table_row(
            table, row_id="bpt.claim.alpha.primary", claim_id="bpt.claim.alpha"
        )


def test_projection_round_trip_markdown_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    bundle, manifest = _fixture(tmp_path)
    _accept(monkeypatch, manifest)
    projection = build_paper_projection(bundle_path=bundle, artifact_root=tmp_path)
    assert projection.claim_ids == CLAIMS
    assert projection.claims[1].evidence["status"] == "negative"

    output = tmp_path / "projection.json"
    markdown = tmp_path / "projection.md"
    args = [
        "project-paper",
        str(bundle),
        "--artifact-root",
        str(tmp_path),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
    ]
    assert claim_bundle_main(args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["projection_id"] == projection.projection_id
    assert load_paper_projection(output) == projection
    assert markdown.read_text(encoding="utf-8") == render_paper_projection_markdown(
        projection
    )
    with pytest.raises(FileExistsError, match="already exists"):
        claim_bundle_main(args)
    assert claim_bundle_main([*args, "--force"]) == 0
    capsys.readouterr()


def test_projection_rejects_missing_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fixture(tmp_path, rows=_rows()[1:])
    _accept(monkeypatch, manifest)
    with pytest.raises(ValueError, match="exactly one row"):
        build_paper_projection(bundle_path=bundle, artifact_root=tmp_path)


def test_projection_requires_manifest_output_and_json_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fixture(tmp_path)
    manifest.outputs = manifest.outputs[:1]
    _accept(monkeypatch, manifest)
    with pytest.raises(ValueError, match="run-manifest outputs"):
        build_paper_projection(bundle_path=bundle, artifact_root=tmp_path)

    other = tmp_path / "csv"
    other.mkdir()
    bundle, manifest = _fixture(other, suffix=".csv")
    _accept(monkeypatch, manifest)
    with pytest.raises(ValueError, match="application/json"):
        build_paper_projection(bundle_path=bundle, artifact_root=other)


def test_projection_loader_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fixture(tmp_path)
    _accept(monkeypatch, manifest)
    projection = build_paper_projection(bundle_path=bundle, artifact_root=tmp_path)
    path = tmp_path / "projection.json"
    _json(path, projection.as_dict())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "tampered"
    _json(path, payload)
    with pytest.raises(ValueError, match="digest does not match"):
        load_paper_projection(path)
