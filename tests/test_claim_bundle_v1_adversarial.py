from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import test_claim_bundle_v1 as cases

import bayesian_phystwin.claim_bundle_v1 as claim_bundle
from bayesian_phystwin.cli import claim_bundle as claim_bundle_cli


def _different_digest(value: str) -> str:
    return "0" * 64 if value != "0" * 64 else "1" * 64


def test_claim_bundle_low_level_fail_closed_contracts(tmp_path: Path) -> None:
    claim_bundle._require_exact_fields(
        {"value": 1},
        expected=frozenset({"value"}),
        name="value",
    )
    for payload in ({}, {"value": 1, "extra": 2}):
        with pytest.raises(ValueError, match="does not match schema"):
            claim_bundle._require_exact_fields(
                payload,
                expected=frozenset({"value"}),
                name="value",
            )

    with pytest.raises(ValueError, match="JSON object"):
        claim_bundle._require_mapping([], name="value")
    for value in ("text", b"bytes", {"not": "a sequence"}):
        with pytest.raises(ValueError, match="JSON array"):
            claim_bundle._require_sequence(value, name="value")
    for value in ("", "   ", 1):
        with pytest.raises(ValueError, match="nonempty text"):
            claim_bundle._require_text(value, name="value")
    for value in (1, "A" * 64, "0" * 63, "z" * 64):
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            claim_bundle._require_sha256(value, name="value")
    for value in (True, 1.5, "1"):
        with pytest.raises(ValueError, match="integer"):
            claim_bundle._require_integer(value, name="value")
    for value in ("a\\b", "/absolute", ".", "../outside"):
        with pytest.raises(ValueError, match="portable|normalized relative path"):
            claim_bundle._require_relative_path(value, name="value")
    assert claim_bundle._require_binding_root(".", name="root").as_posix() == "."
    with pytest.raises(ValueError, match="unsupported claim-bundle artifact kind"):
        claim_bundle._require_artifact_kind("unknown")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        claim_bundle._load_json_mapping(malformed, name="payload")
    not_mapping = tmp_path / "not-mapping.json"
    not_mapping.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        claim_bundle._load_json_mapping(not_mapping, name="payload")
    assert claim_bundle._media_type(Path("opaque.bin")) == "application/octet-stream"

    missing = tmp_path / "missing.bin"
    with pytest.raises(ValueError, match="cannot be opened"):
        claim_bundle._stable_regular_file_snapshot(missing)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        claim_bundle._stable_regular_file_snapshot(directory)

    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="not a file"):
        claim_bundle._resolved_artifact_path(root.resolve(), Path("missing.txt"))
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="remain below artifact root"):
        claim_bundle._resolved_artifact_path(root.resolve(), outside.resolve())

    with pytest.raises(ValueError, match="size_bytes must be nonnegative"):
        claim_bundle.ClaimBundleArtifactV1(
            name="negative",
            kind="supporting",
            path="negative.bin",
            sha256="a" * 64,
            size_bytes=-1,
            media_type="application/octet-stream",
        )


def test_claim_bundle_descriptor_invariants_fail_closed(tmp_path: Path) -> None:
    bundle, _figure = cases._build(tmp_path)

    with pytest.raises(ValueError, match="controlled or confirmatory"):
        replace(bundle, classification="exploratory")
    for claims in ((), (bundle.claim_ids[0], bundle.claim_ids[0])):
        with pytest.raises(ValueError, match="claim_ids must be unique and nonempty"):
            replace(bundle, claim_ids=claims)

    without_primary = tuple(
        replace(state, role="upstream") if state.role == "primary" else state
        for state in bundle.repositories
    )
    with pytest.raises(ValueError, match="exactly one primary repository"):
        replace(bundle, repositories=without_primary)

    two_primary = (
        bundle.repositories[0],
        replace(bundle.repositories[1], role="primary"),
        *bundle.repositories[2:],
    )
    with pytest.raises(ValueError, match="exactly one primary repository"):
        replace(bundle, repositories=two_primary)

    dirty = (
        replace(bundle.repositories[0], dirty=True),
        *bundle.repositories[1:],
    )
    with pytest.raises(ValueError, match="dirty repository state"):
        replace(bundle, repositories=dirty)

    duplicate_repository = (
        bundle.repositories[0],
        replace(
            bundle.repositories[1],
            repository=bundle.repositories[0].repository,
        ),
        *bundle.repositories[2:],
    )
    with pytest.raises(ValueError, match="repository names must be unique"):
        replace(bundle, repositories=duplicate_repository)

    with pytest.raises(ValueError, match="must contain artifacts"):
        replace(bundle, artifacts=())

    first, second, *remaining = bundle.artifacts
    with pytest.raises(ValueError, match="artifact names must be unique"):
        replace(
            bundle,
            artifacts=(first, replace(second, name=first.name), *remaining),
        )
    with pytest.raises(ValueError, match="artifact paths must be unique"):
        replace(
            bundle,
            artifacts=(first, replace(second, path=first.path), *remaining),
        )

    for kind, message in (
        ("run_manifest", "exactly one run manifest"),
        ("evidence_summary", "exactly one evidence summary"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(
                bundle,
                artifacts=tuple(
                    artifact for artifact in bundle.artifacts if artifact.kind != kind
                ),
            )

    claim_binding = next(
        artifact for artifact in bundle.artifacts if artifact.kind == "claim_binding"
    )
    second_binding = replace(
        claim_binding,
        name="paper_claim_binding_2",
        path="claim-binding-2.json",
    )
    with pytest.raises(ValueError, match="at most one claim binding"):
        replace(bundle, artifacts=(*bundle.artifacts, second_binding))


def test_decisive_evidence_summary_fail_closed_branches() -> None:
    base = cases._summary()
    mutations: list[tuple[str, object]] = [
        ("schema_version", 2),
        ("schema_version", True),
        ("contract", "wrong-contract"),
        ("source_contract", "wrong-source-contract"),
        ("reference_method", 1),
    ]
    for key, value in mutations:
        payload = deepcopy(base)
        payload[key] = value
        with pytest.raises(ValueError):
            claim_bundle.validate_decisive_evidence_summary(payload)

    configuration_mutations = {
        "matched_fallback": False,
        "primary_risk_coverage_contract": "wrong-primary",
        "secondary_risk_coverage_contract": "wrong-secondary",
        "confirmatory_thresholds_must_be_source_or_calibration_frozen": False,
    }
    for key, value in configuration_mutations.items():
        payload = deepcopy(base)
        payload["analysis_configuration"][key] = value
        with pytest.raises(ValueError):
            claim_bundle.validate_decisive_evidence_summary(payload)

    payload = deepcopy(base)
    payload["metrics"] = {}
    with pytest.raises(ValueError, match="must not be empty"):
        claim_bundle.validate_decisive_evidence_summary(payload)

    metric = deepcopy(next(iter(base["metrics"].values())))
    payload = deepcopy(base)
    payload["metrics"] = {"": metric}
    with pytest.raises(ValueError, match="nonempty text"):
        claim_bundle.validate_decisive_evidence_summary(payload)

    payload = deepcopy(base)
    payload["metrics"] = {"loss": []}
    with pytest.raises(ValueError, match="JSON object"):
        claim_bundle.validate_decisive_evidence_summary(payload)

    for section in ("threshold_risk_coverage", "matched_count_risk_coverage"):
        payload = deepcopy(base)
        del payload["metrics"]["track_error_mm"][section]
        with pytest.raises(ValueError, match="JSON object"):
            claim_bundle.validate_decisive_evidence_summary(payload)

        payload = deepcopy(base)
        payload["metrics"]["track_error_mm"][section]["contract"] = "wrong-contract"
        with pytest.raises(ValueError, match="wrong .*contract"):
            claim_bundle.validate_decisive_evidence_summary(payload)


def test_claim_binding_fail_closed_branches(tmp_path: Path) -> None:
    manifest_path, summary_path, binding_path, figure_path, table_path = (
        cases._bundle_inputs(tmp_path)
    )
    manifest = claim_bundle._require_v2_manifest(manifest_path)
    artifacts = (
        claim_bundle.claim_bundle_artifact(
            manifest_path,
            name="run_manifest",
            kind="run_manifest",
            root=tmp_path,
            media_type="application/json",
        ),
        claim_bundle.claim_bundle_artifact(
            summary_path,
            name="decisive_evidence_summary",
            kind="evidence_summary",
            root=tmp_path,
            media_type="application/json",
        ),
        claim_bundle.claim_bundle_artifact(
            figure_path,
            name="risk_coverage_figure",
            kind="figure",
            root=tmp_path,
        ),
        claim_bundle.claim_bundle_artifact(
            table_path,
            name="object_results_table",
            kind="table_data",
            root=tmp_path,
        ),
    )
    valid = json.loads(binding_path.read_text(encoding="utf-8"))

    def rejects(payload: dict, match: str, current_artifacts=artifacts) -> None:
        with pytest.raises(ValueError, match=match):
            claim_bundle.validate_claim_evidence_bindings(
                payload,
                manifest=manifest,
                artifacts=current_artifacts,
            )

    payload = deepcopy(valid)
    payload["schema_name"] = "wrong-schema"
    rejects(payload, "unsupported claim-evidence binding schema")

    payload = deepcopy(valid)
    payload["schema_version"] = 2
    rejects(payload, "unsupported claim-evidence binding schema version")

    rejects(deepcopy(valid), "require the bound run manifest", artifacts[1:])

    payload = deepcopy(valid)
    payload["bindings"].append(deepcopy(payload["bindings"][0]))
    rejects(payload, "duplicate claim-evidence binding")

    payload = deepcopy(valid)
    payload["bindings"][0]["manifest"] = "other-manifest.json"
    rejects(payload, "selects another manifest path")

    payload = deepcopy(valid)
    payload["bindings"][0]["expected_evidence_fingerprint"] = _different_digest(
        manifest.evidence_fingerprint
    )
    rejects(payload, "selects another evidence fingerprint")

    payload = deepcopy(valid)
    payload["bindings"][0]["result_artifact"]["sha256"] = _different_digest(
        payload["bindings"][0]["result_artifact"]["sha256"]
    )
    rejects(payload, "result artifact is absent")

    payload = deepcopy(valid)
    payload["bindings"][0]["result_artifact"] = {
        "name": "object_results_table",
        "path": table_path.name,
        "sha256": cases.sha256_file(table_path),
    }
    rejects(payload, "result artifact is absent")

    payload = deepcopy(valid)
    payload["bindings"][0]["result_artifact"]["extra"] = True
    rejects(payload, "does not match schema")

    payload = deepcopy(valid)
    payload["bindings"][0]["table_artifact"]["sha256"] = _different_digest(
        payload["bindings"][0]["table_artifact"]["sha256"]
    )
    rejects(payload, "table artifact is absent")

    payload = deepcopy(valid)
    payload["bindings"][0]["table_artifact"] = {
        "name": "decisive_evidence_summary",
        "path": summary_path.name,
        "sha256": cases.sha256_file(summary_path),
    }
    rejects(payload, "table artifact is absent")

    payload = deepcopy(valid)
    payload["bindings"][1]["claim_id"] = "unknown.claim"
    rejects(payload, "do not match manifest claim IDs")

    payload = deepcopy(valid)
    payload["migration_exceptions"] = [
        {"claim_id": "legacy.claim", "reason": "legacy"},
        {"claim_id": "legacy.claim", "reason": "duplicate"},
    ]
    rejects(payload, "duplicate claim-evidence migration exception")


def test_claim_bundle_cli_fail_closed_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected NAME=PATH"):
        claim_bundle_cli._named_path("invalid")
    assert claim_bundle_cli._named_path(" name = artifact.json ") == (
        "name",
        Path("artifact.json"),
    )

    manifest_path, summary_path, _binding_path, _figure_path, _table_path = (
        cases._bundle_inputs(tmp_path)
    )
    bundle = claim_bundle.build_claim_bundle(
        run_manifest_path=manifest_path,
        evidence_summary_path=summary_path,
        artifact_root=tmp_path,
    )
    bundle_path = tmp_path / "without-binding.json"
    claim_bundle.write_claim_bundle(bundle_path, bundle)

    assert claim_bundle_cli.main(["validate", str(bundle_path)]) == 0
    assert '"artifacts_verified": false' in capsys.readouterr().out
    with pytest.raises(ValueError, match="no claim-binding artifact"):
        claim_bundle_cli.main(
            ["validate", str(bundle_path), "--require-claim-binding"]
        )

    class UnknownParser:
        def parse_args(self, _argv):
            return argparse.Namespace(command_name="unknown")

    monkeypatch.setattr(claim_bundle_cli, "build_parser", lambda: UnknownParser())
    with pytest.raises(AssertionError, match="unhandled command"):
        claim_bundle_cli.main([])


def test_claim_bundle_load_and_verify_fail_closed_branches(tmp_path: Path) -> None:
    bundle, figure_path = cases._build(tmp_path)
    bundle_path = tmp_path / "claim-bundle.json"
    claim_bundle.write_claim_bundle(bundle_path, bundle)

    repository = bundle.repositories[0].as_dict()
    invalid_role = dict(repository)
    invalid_role["role"] = "invalid-role"
    with pytest.raises(ValueError, match="unsupported claim-bundle repository role"):
        claim_bundle._repository_from_mapping(invalid_role)
    invalid_dirty = dict(repository)
    invalid_dirty["dirty"] = 1
    with pytest.raises(ValueError, match="dirty field must be boolean"):
        claim_bundle._repository_from_mapping(invalid_dirty)

    for key, value, message in (
        ("schema_name", "wrong-schema", "unsupported claim-bundle schema"),
        ("schema_version", 2, "unsupported claim-bundle schema version"),
    ):
        payload = bundle.as_dict()
        payload[key] = value
        path = tmp_path / f"invalid-{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            claim_bundle.load_claim_bundle(path)

    semantic_drift = replace(bundle, run_id="changed-run")
    with pytest.raises(ValueError, match="run_id differs from bound evidence"):
        claim_bundle.verify_claim_bundle_artifacts(semantic_drift, root=tmp_path)

    original = figure_path.read_text(encoding="utf-8")
    replacement = "<SVG></svg>\n"
    assert len(replacement.encode()) == len(original.encode())
    figure_path.write_text(replacement, encoding="utf-8")
    with pytest.raises(ValueError, match="artifact digest differs"):
        claim_bundle.verify_claim_bundle_artifacts(bundle, root=tmp_path)


def test_paper_evidence_empty_distribution_profile_is_rejected(tmp_path: Path) -> None:
    inputs, twin = cases._paper_artifacts(tmp_path)
    profile = cases._paper_bindings(inputs, twin)
    with pytest.raises(ValueError, match="requires distribution artifacts"):
        replace(profile, distributions=())
