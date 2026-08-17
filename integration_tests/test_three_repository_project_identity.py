from __future__ import annotations

from prob4d.project_identity import (
    PROB4D_CANONICAL_REPOSITORY,
    PROB4D_FROZEN_ARTIFACT_REPOSITORY,
    PROB4D_GITHUB_REPOSITORY_ID,
    PROB4D_PROJECT_ID,
    prob4d_project_identity,
    validate_prob4d_project_identity,
)

from bayesian_phystwin.prob4d_observation_contract import PROB4D_SOURCE_REPOSITORY


def test_current_prob4d_identity_preserves_frozen_artifact_alias() -> None:
    identity = prob4d_project_identity()

    assert validate_prob4d_project_identity(identity) == identity
    assert identity["project_id"] == PROB4D_PROJECT_ID
    assert identity["github_repository_id"] == PROB4D_GITHUB_REPOSITORY_ID
    assert identity["canonical_repository"] == PROB4D_CANONICAL_REPOSITORY
    assert identity["frozen_artifact_repository"] == PROB4D_FROZEN_ARTIFACT_REPOSITORY
    assert PROB4D_SOURCE_REPOSITORY == PROB4D_FROZEN_ARTIFACT_REPOSITORY
    assert PROB4D_CANONICAL_REPOSITORY != PROB4D_SOURCE_REPOSITORY
