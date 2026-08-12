from __future__ import annotations

import pytest

from bayesian_phystwin.prospective_study_lifecycle_v1 import (
    ProspectiveStudyProtocolV1,
    ProspectiveStudyStateV1,
    advance_prospective_study,
    lock_prospective_study,
)


def _digest(character: str) -> str:
    return character * 64


def _protocol() -> ProspectiveStudyProtocolV1:
    return ProspectiveStudyProtocolV1(
        protocol_id="artifact-identity-uniqueness-v1",
        method_set_id=_digest("1"),
        decision_rule_id=_digest("2"),
        fallback_identity_id=_digest("3"),
        information_boundary_id=_digest("4"),
        statistical_unit="physical object session",
        development_group_ids=("source-a",),
        target_group_ids=("target-a",),
    )


def test_direct_state_rejects_reused_artifact_identity() -> None:
    protocol = _protocol()

    with pytest.raises(ValueError, match="artifact identities must be unique"):
        ProspectiveStudyStateV1(
            protocol_id=protocol.protocol_id,
            protocol_content_id=protocol.protocol_content_id,
            stage="source-scored",
            sequence_number=2,
            previous_state_id=_digest("5"),
            source_prediction_bundle_id=_digest("6"),
            source_score_bundle_id=_digest("6"),
        )


def test_transition_rejects_reused_artifact_identity() -> None:
    locked = lock_prospective_study(_protocol())
    predictions = advance_prospective_study(
        locked,
        next_stage="source-predictions-sealed",
        artifact_id=_digest("5"),
    )

    with pytest.raises(ValueError, match="artifact identities must be unique"):
        advance_prospective_study(
            predictions,
            next_stage="source-scored",
            artifact_id=_digest("5"),
        )


def test_terminal_decision_cannot_reuse_prior_artifact_identity() -> None:
    locked = lock_prospective_study(_protocol())
    predictions = advance_prospective_study(
        locked,
        next_stage="source-predictions-sealed",
        artifact_id=_digest("5"),
    )

    with pytest.raises(ValueError, match="artifact identities must be unique"):
        advance_prospective_study(
            predictions,
            next_stage="terminal-source-negative",
            artifact_id=_digest("5"),
        )
