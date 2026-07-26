from __future__ import annotations

from typing import Any

import pytest

from bayesian_phystwin.paper_evidence_v1 import (
    ArtifactBindingV1,
    DistributionBindingV1,
    PaperEvidenceBindingsV1,
    Prob4DStreamBindingV1,
)


def _artifact(
    name: str,
    role: str,
    *,
    digest_character: str,
) -> ArtifactBindingV1:
    return ArtifactBindingV1(
        artifact_name=name,
        artifact_id=digest_character * 64,
        role=role,  # type: ignore[arg-type]
    )


def _profile(*, twin_role: str = "output") -> PaperEvidenceBindingsV1:
    return PaperEvidenceBindingsV1(
        primary_distribution_project="bayesian-phystwin",
        provider_manifest=_artifact(
            "provider_manifest",
            "input",
            digest_character="a",
        ),
        prob4d_stream_contract=Prob4DStreamBindingV1(
            version=2,
            resolution="declared",
        ),
        observation_belief=_artifact(
            "observation_belief",
            "input",
            digest_character="b",
        ),
        twin_belief=_artifact(
            "twin_belief",
            twin_role,
            digest_character="c",
        ),
        distributions=(
            DistributionBindingV1(
                project="bayesian-phystwin",
                kind="wheel",
                artifact_name="bayesian_phystwin_wheel",
                artifact_id="d" * 64,
            ),
            DistributionBindingV1(
                project="bayesian-phystwin",
                kind="sdist",
                artifact_name="bayesian_phystwin_sdist",
                artifact_id="e" * 64,
            ),
        ),
    )


def test_twin_belief_must_be_an_output_artifact() -> None:
    with pytest.raises(ValueError, match="TwinBelief must be bound as an output"):
        _profile(twin_role="input")


def test_profile_schema_version_does_not_coerce_text() -> None:
    payload: dict[str, Any] = _profile().as_dict()
    payload["schema_version"] = "1"

    with pytest.raises(ValueError, match="schema version must be an integer"):
        PaperEvidenceBindingsV1.from_mapping(payload)


def test_profile_artifact_names_must_be_unique() -> None:
    profile = _profile()

    with pytest.raises(ValueError, match="artifact names must be unique"):
        PaperEvidenceBindingsV1(
            primary_distribution_project=profile.primary_distribution_project,
            provider_manifest=profile.provider_manifest,
            prob4d_stream_contract=profile.prob4d_stream_contract,
            observation_belief=profile.observation_belief,
            twin_belief=profile.twin_belief,
            distributions=(
                DistributionBindingV1(
                    project="bayesian-phystwin",
                    kind="wheel",
                    artifact_name=profile.provider_manifest.artifact_name,
                    artifact_id="d" * 64,
                ),
                profile.distributions[1],
            ),
        )


def test_artifact_digest_must_be_canonical_lowercase() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ArtifactBindingV1(
            artifact_name="provider_manifest",
            artifact_id="A" * 64,
            role="input",
        )
