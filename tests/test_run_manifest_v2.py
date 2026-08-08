from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.decisive_evidence import analyze_decisive_evidence
from bayesian_phystwin.repository_provenance import (
    RepositoryState,
    discover_git_repository_state,
    normalize_github_repository,
)
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
    load_run_manifest_v2,
    verify_run_manifest_artifacts,
    write_run_manifest,
)


def _manifest(root: Path) -> RunManifestV2:
    source = root / "input.txt"
    result = root / "output.txt"
    source.write_text("input\n", encoding="utf-8")
    result.write_text("output\n", encoding="utf-8")
    return RunManifestV2(
        run_id="unit-test-v2",
        repository="FlorianPfaff/Bayesian-PhysTwin",
        revision="a" * 40,
        dirty=False,
        related_repositories=(
            RepositoryState(
                repository="Jianghanxiao/PhysTwin",
                revision="b" * 40,
                dirty=False,
                role="upstream",
            ),
        ),
        command=("bpt", "benchmark", "synthetic"),
        classification="infrastructure",
        statistical_unit="test case",
        information_boundary={"causal_frame_stop": 10},
        configuration={"seeds": [1, 2]},
        seeds=(1, 2),
        inputs=(artifact_digest(source, name="input", role="input", root=root),),
        outputs=(artifact_digest(result, name="output", role="output", root=root),),
        package_versions={"bayesian-phystwin": "0.4.0"},
        runtime_environment={"python_version": "3.12.0", "gpu_model": "test"},
        claim_ids=("bpt.infrastructure.run_manifest_v2",),
        method_freeze_id="method-v1",
        protocol_id="protocol-v1",
        split_id="split-v1",
        baseline_id="baseline-v1",
        created_utc="2026-07-26T18:00:00+00:00",
    )


def test_v2_round_trip_and_artifact_verification(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)

    loaded = load_run_manifest_v2(path)

    assert loaded == manifest
    assert loaded.manifest_id == manifest.manifest_id
    assert loaded.evidence_fingerprint == manifest.evidence_fingerprint
    verify_run_manifest_artifacts(loaded, root=tmp_path)


def test_generic_loader_preserves_v1_compatibility(tmp_path: Path) -> None:
    artifact = tmp_path / "result.txt"
    artifact.write_text("result\n", encoding="utf-8")
    legacy = RunManifestV1(
        run_id="legacy",
        repository="FlorianPfaff/Bayesian-PhysTwin",
        revision="legacy-revision",
        dirty=False,
        command=("bpt", "provider", "manifest"),
        classification="infrastructure",
        statistical_unit="test case",
        information_boundary={},
        configuration={},
        outputs=(
            artifact_digest(
                artifact,
                name="result",
                role="output",
                root=tmp_path,
            ),
        ),
        created_utc="2026-07-26T18:00:00+00:00",
    )
    path = tmp_path / "legacy.json"
    write_run_manifest_v1(path, legacy)

    loaded = load_run_manifest(path)

    assert isinstance(loaded, RunManifestV1)
    assert loaded == legacy
    verify_run_manifest_artifacts(loaded, root=tmp_path)


def test_evidence_fingerprint_ignores_timestamp_and_notes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    copied = replace(
        manifest,
        created_utc="2026-07-27T18:00:00+00:00",
        notes="copied into the paper bundle",
    )

    assert copied.evidence_fingerprint == manifest.evidence_fingerprint
    assert copied.manifest_id != manifest.manifest_id


def test_v2_rejects_payload_and_schema_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["classification"] = "confirmatory"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint|digest"):
        load_run_manifest_v2(path)

    write_run_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["uncovered"] = "tampering"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match schema"):
        load_run_manifest_v2(path)


def test_repository_states_require_exact_unique_revisions(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="unique repository"):
        replace(
            manifest,
            related_repositories=(
                RepositoryState(
                    repository=manifest.repository,
                    revision="c" * 40,
                    dirty=False,
                    role="dependency",
                ),
            ),
        )
    with pytest.raises(ValueError, match="exact lowercase Git revision"):
        RepositoryState(
            repository="FlorianPfaff/Prob4D",
            revision="main",
            dirty=False,
            role="observation",
        )


def test_v2_requires_nonempty_run_id_and_owner_name_repository(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="run ID must be nonempty"):
        replace(manifest, run_id="")
    with pytest.raises(ValueError, match="repository must use owner/name"):
        replace(manifest, repository="Bayesian-PhysTwin")


def test_stable_core_exercises_threshold_risk_coverage_contract() -> None:
    units = (
        (
            "u1",
            "object-a",
            "early",
            (8.0, 0.10, True, 0.90, 2, True, 2.0),
            (9.0, 0.15, True, 0.70, 1, True, 3.0),
        ),
        (
            "u2",
            "object-a",
            "early",
            (12.0, 0.30, False, 0.40, 1, False, 4.0),
            (11.0, 0.25, True, 0.60, 1, False, 5.0),
        ),
        (
            "u3",
            "object-b",
            "late",
            (7.0, 0.30, True, 0.80, 2, True, 2.5),
            (8.0, 0.35, False, 0.30, 0, True, 4.0),
        ),
    )
    records: list[dict[str, object]] = []
    for unit_id, group_id, horizon, bayesian, reference in units:
        for method, values in (
            ("bayesian", bayesian),
            ("reference", reference),
        ):
            loss, risk, accepted, reliability, rank, covered, width = values
            records.append(
                {
                    "unit_id": unit_id,
                    "group_id": group_id,
                    "metric": "track_error_m",
                    "method": method,
                    "loss": loss,
                    "fallback_loss": 10.0,
                    "risk_score": risk,
                    "accepted": accepted,
                    "deployed_loss": loss if accepted else 10.0,
                    "horizon": horizon,
                    "reliability": reliability,
                    "identifiable_rank": rank,
                    "intervals": [
                        {
                            "nominal_coverage": 0.9,
                            "covered": covered,
                            "width": width,
                        }
                    ],
                }
            )
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract": "bayesian-phystwin-decisive-evidence-v1",
        "protocol_id": "stable-core-threshold-risk-test-v1",
        "statistical_unit": "case-horizon",
        "claim_boundary": "synthetic stable-core coverage test",
        "reference_method": "reference",
        "records": records,
    }

    summary = analyze_decisive_evidence(
        payload,
        target_coverages=(0.0, 0.5, 1.0),
        regression_quantiles=(0.5, 0.95),
        reliability_edges=(0.0, 0.5, 1.0),
    )
    metric = summary["metrics"]["track_error_m"]
    threshold_view = metric["threshold_risk_coverage"]
    curve = threshold_view["methods"]["bayesian"]

    assert threshold_view["contract"] == (
        "bayesian-phystwin-threshold-risk-coverage-v1"
    )
    assert [point["accepted_count"] for point in curve] == [0, 1, 3]
    assert curve[-1]["boundary_tie_count"] == 2
    assert curve[-1]["boundary_tie_split"] is False
    assert metric["matched_count_risk_coverage"]["contract"] == (
        "bayesian-phystwin-matched-count-risk-coverage-v1"
    )

    with pytest.raises(ValueError, match="reference method .* is absent"):
        analyze_decisive_evidence({**payload, "reference_method": "missing"})


def test_discover_git_repository_state_tracks_uncommitted_files(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "git@github.com:FlorianPfaff/Bayesian-PhysTwin.git",
        ],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "tracked.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
    )

    clean = discover_git_repository_state(tmp_path)
    assert clean.repository == "FlorianPfaff/Bayesian-PhysTwin"
    assert len(clean.revision) == 40
    assert clean.dirty is False

    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    assert discover_git_repository_state(tmp_path).dirty is True


def test_normalize_github_repository_accepts_https_and_ssh() -> None:
    expected = "FlorianPfaff/Bayesian-PhysTwin"
    assert (
        normalize_github_repository(
            "https://github.com/FlorianPfaff/Bayesian-PhysTwin.git"
        )
        == expected
    )
    assert (
        normalize_github_repository("git@github.com:FlorianPfaff/Bayesian-PhysTwin.git")
        == expected
    )
