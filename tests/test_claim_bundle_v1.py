from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.claim_bundle_v1 import (
    CLAIM_EVIDENCE_BINDING_SCHEMA,
    CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION,
    ClaimBundleArtifactV1,
    build_claim_bundle,
    claim_bundle_artifact,
    load_claim_bundle,
    verify_claim_bundle_artifacts,
    write_claim_bundle,
)
from bayesian_phystwin.cli.claim_bundle import main as claim_bundle_main
from bayesian_phystwin.cli.command_registry import COMMANDS_BY_ID
from bayesian_phystwin.cli.main import main as grouped_main
from bayesian_phystwin.decisive_evidence import (
    DECISIVE_EVIDENCE_INPUT_CONTRACT,
    DECISIVE_EVIDENCE_SUMMARY_CONTRACT,
    MATCHED_COUNT_RISK_COVERAGE_CONTRACT,
    THRESHOLD_RISK_COVERAGE_CONTRACT,
)
from bayesian_phystwin.paper_evidence_v1 import (
    ArtifactBindingV1,
    DistributionBindingV1,
    PaperEvidenceBindingsV1,
    Prob4DStreamBindingV1,
    embed_paper_evidence_bindings,
)
from bayesian_phystwin.repository_provenance import RepositoryState
from bayesian_phystwin.run_manifest import (
    ArtifactDigest,
    artifact_digest,
    sha256_file,
)
from bayesian_phystwin.run_manifest_v2 import RunManifestV2, write_run_manifest


def _paper_artifacts(root: Path) -> tuple[dict[str, ArtifactDigest], ArtifactDigest]:
    paths = {
        "provider_manifest": root / "provider-manifest.json",
        "observation_belief": root / "observation-belief.npz",
        "twin_belief": root / "twin-belief.json",
        "bayesian_phystwin_wheel": root / "bayesian_phystwin.whl",
        "bayesian_phystwin_sdist": root / "bayesian_phystwin.tar.gz",
    }
    for name, path in paths.items():
        path.write_bytes(f"{name}\n".encode())
    inputs = {
        name: artifact_digest(path, name=name, role="input", root=root)
        for name, path in paths.items()
        if name != "twin_belief"
    }
    twin = artifact_digest(
        paths["twin_belief"],
        name="twin_belief",
        role="output",
        root=root,
    )
    return inputs, twin


def _paper_bindings(
    inputs: dict[str, ArtifactDigest],
    twin: ArtifactDigest,
) -> PaperEvidenceBindingsV1:
    return PaperEvidenceBindingsV1(
        primary_distribution_project="bayesian-phystwin",
        provider_manifest=ArtifactBindingV1(
            artifact_name="provider_manifest",
            artifact_id=inputs["provider_manifest"].sha256,
            role="input",
        ),
        prob4d_stream_contract=Prob4DStreamBindingV1(
            version=2,
            resolution="declared",
        ),
        observation_belief=ArtifactBindingV1(
            artifact_name="observation_belief",
            artifact_id=inputs["observation_belief"].sha256,
            role="input",
        ),
        twin_belief=ArtifactBindingV1(
            artifact_name="twin_belief",
            artifact_id=twin.sha256,
            role="output",
        ),
        distributions=(
            DistributionBindingV1(
                project="bayesian-phystwin",
                kind="wheel",
                artifact_name="bayesian_phystwin_wheel",
                artifact_id=inputs["bayesian_phystwin_wheel"].sha256,
            ),
            DistributionBindingV1(
                project="bayesian-phystwin",
                kind="sdist",
                artifact_name="bayesian_phystwin_sdist",
                artifact_id=inputs["bayesian_phystwin_sdist"].sha256,
            ),
        ),
    )


def _manifest(root: Path, *, classification: str = "confirmatory") -> RunManifestV2:
    inputs, twin = _paper_artifacts(root)
    return RunManifestV2(
        run_id="claim-bundle-test",
        repository="IPS-Stuttgart/BayesianPhysTwin",
        revision="a" * 40,
        dirty=False,
        related_repositories=(
            RepositoryState(
                repository="IPS-Stuttgart/Prob4D",
                revision="b" * 40,
                dirty=False,
                role="observation",
            ),
            RepositoryState(
                repository="FlorianPfaff/BayesianPhysTwin-Paper",
                revision="c" * 40,
                dirty=False,
                role="paper",
            ),
        ),
        command=("bpt", "evidence", "summarize"),
        classification=classification,
        statistical_unit="physical object",
        information_boundary=embed_paper_evidence_bindings(
            {"confirmation_opened_once": True},
            _paper_bindings(inputs, twin),
        ),
        configuration={"guard": "baseline-relative-v1"},
        inputs=tuple(inputs.values()),
        outputs=(twin,),
        package_versions={"bayesian-phystwin": "0.4.0"},
        runtime_environment={"python_version": "3.12.0"},
        claim_ids=("bpt.deform360.guard", "bpt.deform360.calibration"),
        method_freeze_id="method-v1",
        protocol_id="deform360-independent-object-v1",
        split_id="calibration-10-confirmation-12-v1",
        baseline_id="physical-fallback-v1",
        created_utc="2026-08-06T12:00:00+00:00",
    )


def _summary(*, protocol_id: str = "deform360-independent-object-v1") -> dict:
    return {
        "schema_version": 1,
        "contract": DECISIVE_EVIDENCE_SUMMARY_CONTRACT,
        "source_contract": DECISIVE_EVIDENCE_INPUT_CONTRACT,
        "protocol_id": protocol_id,
        "statistical_unit": "physical object",
        "claim_boundary": (
            "fresh-object guarded physical-query evidence; no deployment claim"
        ),
        "reference_method": "physical_fallback",
        "analysis_configuration": {
            "matched_fallback": True,
            "primary_risk_coverage_contract": THRESHOLD_RISK_COVERAGE_CONTRACT,
            "secondary_risk_coverage_contract": (MATCHED_COUNT_RISK_COVERAGE_CONTRACT),
            "confirmatory_thresholds_must_be_source_or_calibration_frozen": True,
        },
        "metrics": {
            "track_error_mm": {
                "threshold_risk_coverage": {
                    "contract": THRESHOLD_RISK_COVERAGE_CONTRACT,
                    "methods": {},
                },
                "matched_count_risk_coverage": {
                    "contract": MATCHED_COUNT_RISK_COVERAGE_CONTRACT,
                    "methods": {},
                },
            }
        },
    }


def _claim_binding_payload(
    manifest: RunManifestV2,
    *,
    summary_path: Path,
    table_path: Path,
) -> dict:
    bindings = []
    for claim_id in manifest.claim_ids:
        bindings.append(
            {
                "artifact_root": ".",
                "claim_id": claim_id,
                "expected_evidence_fingerprint": manifest.evidence_fingerprint,
                "expected_manifest_id": manifest.manifest_id,
                "manifest": "run-manifest.json",
                "result_artifact": {
                    "name": "decisive_evidence_summary",
                    "path": summary_path.name,
                    "sha256": sha256_file(summary_path),
                },
                "table_artifact": {
                    "name": "object_results_table",
                    "path": table_path.name,
                    "sha256": sha256_file(table_path),
                },
                "table_row_id": f"{claim_id}.primary",
            }
        )
    return {
        "bindings": bindings,
        "migration_exceptions": [],
        "schema_name": CLAIM_EVIDENCE_BINDING_SCHEMA,
        "schema_version": CLAIM_EVIDENCE_BINDING_SCHEMA_VERSION,
    }


def _bundle_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    manifest = _manifest(root)
    manifest_path = root / "run-manifest.json"
    write_run_manifest(manifest_path, manifest)
    summary_path = root / "evidence-summary.json"
    summary_path.write_text(
        json.dumps(_summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure_path = root / "risk-coverage.svg"
    figure_path.write_text("<svg></svg>\n", encoding="utf-8")
    table_path = root / "object-results.csv"
    table_path.write_text("object,loss\n001,1.0\n", encoding="utf-8")
    binding_path = root / "claim-binding.json"
    binding_path.write_text(
        json.dumps(
            _claim_binding_payload(
                manifest,
                summary_path=summary_path,
                table_path=table_path,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, summary_path, binding_path, figure_path, table_path


def _build(root: Path):
    manifest_path, summary_path, binding_path, figure_path, table_path = _bundle_inputs(
        root
    )
    extras = (
        claim_bundle_artifact(
            figure_path,
            name="risk_coverage_figure",
            kind="figure",
            root=root,
        ),
        claim_bundle_artifact(
            table_path,
            name="object_results_table",
            kind="table_data",
            root=root,
        ),
    )
    bundle = build_claim_bundle(
        run_manifest_path=manifest_path,
        evidence_summary_path=summary_path,
        claim_binding_path=binding_path,
        artifact_root=root,
        additional_artifacts=extras,
    )
    return bundle, figure_path


def test_claim_bundle_round_trips_and_revalidates_bound_evidence(
    tmp_path: Path,
) -> None:
    bundle, _figure_path = _build(tmp_path)
    bundle_path = tmp_path / "claim-bundle.json"
    write_claim_bundle(bundle_path, bundle)

    loaded = load_claim_bundle(bundle_path)
    manifest = verify_claim_bundle_artifacts(loaded, root=tmp_path)
    rebuilt, _ = _build(tmp_path)

    assert loaded == bundle
    assert rebuilt.bundle_id == bundle.bundle_id
    assert manifest.manifest_id == bundle.run_manifest_id
    assert bundle.artifacts[0].kind == "claim_binding"
    assert {artifact.kind for artifact in bundle.artifacts} == {
        "run_manifest",
        "evidence_summary",
        "claim_binding",
        "figure",
        "table_data",
    }


def test_claim_bundle_detects_artifact_and_descriptor_tampering(
    tmp_path: Path,
) -> None:
    bundle, figure_path = _build(tmp_path)
    bundle_path = tmp_path / "claim-bundle.json"
    write_claim_bundle(bundle_path, bundle)

    figure_path.write_text("<svg>tampered</svg>\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="artifact size differs|artifact digest differs"
    ):
        verify_claim_bundle_artifacts(bundle, root=tmp_path)

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["run_id"] = "changed-run"
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        load_claim_bundle(bundle_path)

    bundle_path.write_text(
        '{"bundle_id":"'
        + bundle.bundle_id
        + '","bundle_id":"'
        + bundle.bundle_id
        + '"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_claim_bundle(bundle_path)


def test_claim_bundle_rejects_semantic_drift_and_nonclaim_runs(
    tmp_path: Path,
) -> None:
    manifest_path, summary_path, _binding, _figure, _table = _bundle_inputs(tmp_path)
    summary_path.write_text(
        json.dumps(_summary(protocol_id="different-protocol")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protocol_id differs"):
        build_claim_bundle(
            run_manifest_path=manifest_path,
            evidence_summary_path=summary_path,
            artifact_root=tmp_path,
        )

    write_run_manifest(
        manifest_path,
        _manifest(tmp_path, classification="exploratory"),
    )
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    with pytest.raises(ValueError, match="controlled or confirmatory"):
        build_claim_bundle(
            run_manifest_path=manifest_path,
            evidence_summary_path=summary_path,
            artifact_root=tmp_path,
        )


def test_claim_bundle_rejects_missing_paper_profile_and_reserved_extra(
    tmp_path: Path,
) -> None:
    manifest_path, summary_path, _binding, figure_path, _table = _bundle_inputs(
        tmp_path
    )
    manifest = _manifest(tmp_path)
    write_run_manifest(
        manifest_path,
        replace(manifest, information_boundary={}),
    )
    with pytest.raises(ValueError, match="no paper-evidence profile"):
        build_claim_bundle(
            run_manifest_path=manifest_path,
            evidence_summary_path=summary_path,
            artifact_root=tmp_path,
        )

    write_run_manifest(manifest_path, manifest)
    reserved = claim_bundle_artifact(
        figure_path,
        name="not-a-manifest",
        kind="run_manifest",
        root=tmp_path,
    )
    with pytest.raises(ValueError, match="supporting artifact kind"):
        build_claim_bundle(
            run_manifest_path=manifest_path,
            evidence_summary_path=summary_path,
            artifact_root=tmp_path,
            additional_artifacts=(reserved,),
        )


def test_claim_bundle_rejects_unbound_or_migrated_paper_claims(
    tmp_path: Path,
) -> None:
    manifest_path, summary_path, binding_path, _figure, table_path = _bundle_inputs(
        tmp_path
    )
    manifest = _manifest(tmp_path)
    table_artifact = claim_bundle_artifact(
        table_path,
        name="object_results_table",
        kind="table_data",
        root=tmp_path,
    )

    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    payload["bindings"][0]["expected_manifest_id"] = "f" * 64
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selects another manifest"):
        build_claim_bundle(
            run_manifest_path=manifest_path,
            evidence_summary_path=summary_path,
            claim_binding_path=binding_path,
            artifact_root=tmp_path,
            additional_artifacts=(table_artifact,),
        )

    payload = _claim_binding_payload(
        manifest,
        summary_path=summary_path,
        table_path=table_path,
    )
    payload["bindings"].pop()
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match manifest claim IDs"):
        build_claim_bundle(
            run_manifest_path=manifest_path,
            evidence_summary_path=summary_path,
            claim_binding_path=binding_path,
            artifact_root=tmp_path,
            additional_artifacts=(table_artifact,),
        )

    payload = _claim_binding_payload(
        manifest,
        summary_path=summary_path,
        table_path=table_path,
    )
    payload["migration_exceptions"] = [
        {
            "claim_id": manifest.claim_ids[0],
            "reason": "legacy exception must not authorize a new bundle",
        }
    ]
    binding_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot rely on migration exceptions"):
        build_claim_bundle(
            run_manifest_path=manifest_path,
            evidence_summary_path=summary_path,
            claim_binding_path=binding_path,
            artifact_root=tmp_path,
            additional_artifacts=(table_artifact,),
        )

    binding_path.write_text('{"claims": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match schema"):
        build_claim_bundle(
            run_manifest_path=manifest_path,
            evidence_summary_path=summary_path,
            claim_binding_path=binding_path,
            artifact_root=tmp_path,
            additional_artifacts=(table_artifact,),
        )


def test_claim_bundle_publication_refuses_replacement_by_default(
    tmp_path: Path,
) -> None:
    bundle, _figure = _build(tmp_path)
    destination = tmp_path / "claim-bundle.json"

    write_claim_bundle(destination, bundle)
    with pytest.raises(FileExistsError, match="already exists"):
        write_claim_bundle(destination, bundle)
    write_claim_bundle(destination, bundle, overwrite=True)
    assert load_claim_bundle(destination) == bundle


def test_claim_bundle_rejects_symbolic_link_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("payload\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(ValueError, match="must not be symbolic links"):
        claim_bundle_artifact(
            link,
            name="linked",
            kind="supporting",
            root=tmp_path,
        )


def test_claim_bundle_cli_builds_validates_and_registers_route(
    tmp_path: Path,
    capsys,
) -> None:
    manifest_path, summary_path, binding_path, figure_path, table_path = _bundle_inputs(
        tmp_path
    )
    bundle_path = tmp_path / "claim-bundle.json"
    build_args = [
        "build",
        str(bundle_path),
        "--artifact-root",
        str(tmp_path),
        "--run-manifest",
        str(manifest_path),
        "--evidence-summary",
        str(summary_path),
        "--claim-binding",
        str(binding_path),
        "--figure",
        f"risk_coverage_figure={figure_path}",
        "--table-data",
        f"object_results_table={table_path}",
    ]

    assert claim_bundle_main(build_args) == 0
    build_output = json.loads(capsys.readouterr().out)
    assert build_output["artifact_count"] == 5
    assert build_output["claim_count"] == 2

    with pytest.raises(FileExistsError, match="already exists"):
        claim_bundle_main(build_args)
    assert claim_bundle_main([*build_args, "--force"]) == 0
    capsys.readouterr()

    assert (
        claim_bundle_main(
            [
                "validate",
                str(bundle_path),
                "--artifact-root",
                str(tmp_path),
                "--require-claim-binding",
            ]
        )
        == 0
    )
    validation = json.loads(capsys.readouterr().out)
    assert validation["status"] == "valid"
    assert validation["artifacts_verified"] is True
    assert validation["claim_binding"] == "present"

    command = COMMANDS_BY_ID["claim-bundle"]
    assert command.route == ("evidence", "bundle")
    assert grouped_main(["evidence"]) == 0
    assert "bundle" in capsys.readouterr().out


def test_claim_bundle_artifact_contract_rejects_nonportable_paths() -> None:
    with pytest.raises(ValueError, match="normalized relative path"):
        ClaimBundleArtifactV1(
            name="bad",
            kind="supporting",
            path="../outside.json",
            sha256="a" * 64,
            size_bytes=1,
            media_type="application/json",
        )
