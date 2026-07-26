from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from bayesian_phystwin.paper_evidence_v1 import (
    PAPER_EVIDENCE_PROFILE_KEY,
    ArtifactBindingV1,
    DistributionBindingV1,
    PaperEvidenceBindingsV1,
    Prob4DStreamBindingV1,
    embed_paper_evidence_bindings,
    load_paper_evidence_bindings,
    validate_paper_evidence_manifest,
)
from bayesian_phystwin.repository_provenance import RepositoryState
from bayesian_phystwin.run_manifest import ArtifactDigest, artifact_digest
from bayesian_phystwin.run_manifest_v2 import RunManifestV2


def _artifact_files(root: Path) -> dict[str, Path]:
    files = {
        "provider_manifest": root / "provider-manifest.json",
        "observation_belief": root / "observation-belief.npz",
        "twin_belief": root / "twin-belief.json",
        "bayesian_phystwin_wheel": root / "bayesian_phystwin.whl",
        "bayesian_phystwin_sdist": root / "bayesian_phystwin.tar.gz",
    }
    for name, path in files.items():
        path.write_bytes(f"{name}\n".encode())
    return files


def _bindings(
    artifacts: dict[str, ArtifactDigest],
    *,
    resolution: Literal["declared", "inferred"] = "declared",
) -> PaperEvidenceBindingsV1:
    return PaperEvidenceBindingsV1(
        primary_distribution_project="bayesian-phystwin",
        provider_manifest=ArtifactBindingV1(
            artifact_name="provider_manifest",
            artifact_id=artifacts["provider_manifest"].sha256,
            role="input",
        ),
        prob4d_stream_contract=Prob4DStreamBindingV1(
            version=2,
            resolution=resolution,
        ),
        observation_belief=ArtifactBindingV1(
            artifact_name="observation_belief",
            artifact_id=artifacts["observation_belief"].sha256,
            role="input",
        ),
        twin_belief=ArtifactBindingV1(
            artifact_name="twin_belief",
            artifact_id=artifacts["twin_belief"].sha256,
            role="output",
        ),
        distributions=(
            DistributionBindingV1(
                project="bayesian-phystwin",
                kind="wheel",
                artifact_name="bayesian_phystwin_wheel",
                artifact_id=artifacts["bayesian_phystwin_wheel"].sha256,
            ),
            DistributionBindingV1(
                project="bayesian-phystwin",
                kind="sdist",
                artifact_name="bayesian_phystwin_sdist",
                artifact_id=artifacts["bayesian_phystwin_sdist"].sha256,
            ),
        ),
    )


def _manifest(root: Path) -> RunManifestV2:
    files = _artifact_files(root)
    inputs = {
        name: artifact_digest(path, name=name, role="input", root=root)
        for name, path in files.items()
        if name != "twin_belief"
    }
    twin = artifact_digest(
        files["twin_belief"],
        name="twin_belief",
        role="output",
        root=root,
    )
    bindings = _bindings({**inputs, "twin_belief": twin})
    return RunManifestV2(
        run_id="paper-evidence-test",
        repository="FlorianPfaff/Bayesian-PhysTwin",
        revision="a" * 40,
        dirty=False,
        related_repositories=(
            RepositoryState(
                repository="FlorianPfaff/Prob4D",
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
        command=("bpt", "run", "manifest", "create"),
        classification="confirmatory",
        statistical_unit="interaction",
        information_boundary=embed_paper_evidence_bindings(
            {"causal_frame_stop": 10},
            bindings,
        ),
        configuration={"guard": "baseline-relative-v1"},
        inputs=tuple(inputs.values()),
        outputs=(twin,),
        package_versions={"bayesian-phystwin": "0.4.0"},
        runtime_environment={"python_version": "3.12.0"},
        claim_ids=("bpt.paper_evidence_profile",),
        method_freeze_id="method-v1",
        protocol_id="protocol-v1",
        split_id="split-v1",
        baseline_id="baseline-v1",
        created_utc="2026-07-27T10:00:00+00:00",
    )


def test_paper_evidence_profile_matches_manifest_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    bindings = validate_paper_evidence_manifest(manifest)

    assert bindings.prob4d_stream_contract.version == 2
    assert (
        manifest.information_boundary[PAPER_EVIDENCE_PROFILE_KEY]
        == bindings.as_dict()
    )


def test_paper_evidence_profile_round_trips_strict_json(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = manifest.information_boundary[PAPER_EVIDENCE_PROFILE_KEY]
    profile_path = tmp_path / "paper-evidence.json"
    profile_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded = load_paper_evidence_bindings(profile_path)

    assert loaded.as_dict() == payload


def test_stream_resolution_is_part_of_evidence_fingerprint(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    original = validate_paper_evidence_manifest(manifest)
    inferred = replace(
        original,
        prob4d_stream_contract=Prob4DStreamBindingV1(
            version=2,
            resolution="inferred",
        ),
    )
    changed = replace(
        manifest,
        information_boundary=embed_paper_evidence_bindings(
            {"causal_frame_stop": 10},
            inferred,
        ),
    )

    validate_paper_evidence_manifest(changed)
    assert changed.evidence_fingerprint != manifest.evidence_fingerprint


def test_profile_rejects_artifact_id_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    profile = validate_paper_evidence_manifest(manifest)
    changed = replace(
        profile,
        provider_manifest=replace(
            profile.provider_manifest,
            artifact_id="d" * 64,
        ),
    )
    tampered = replace(
        manifest,
        information_boundary=embed_paper_evidence_bindings(
            {"causal_frame_stop": 10},
            changed,
        ),
    )

    with pytest.raises(ValueError, match="provider manifest artifact ID"):
        validate_paper_evidence_manifest(tampered)


def test_profile_requires_claim_and_freeze_identifiers(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="at least one claim"):
        validate_paper_evidence_manifest(replace(manifest, claim_ids=()))

    with pytest.raises(ValueError, match="method_freeze_id"):
        validate_paper_evidence_manifest(
            replace(manifest, method_freeze_id="")
        )


def test_profile_rejects_dirty_participating_repository(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    dirty_related = replace(
        manifest.related_repositories[0],
        dirty=True,
    )

    with pytest.raises(ValueError, match="dirty repository"):
        validate_paper_evidence_manifest(
            replace(
                manifest,
                related_repositories=(
                    dirty_related,
                    manifest.related_repositories[1],
                ),
            )
        )


def test_primary_distribution_requires_wheel_and_sdist(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    profile = validate_paper_evidence_manifest(manifest)

    with pytest.raises(ValueError, match="wheel and one sdist"):
        replace(profile, distributions=(profile.distributions[0],))
