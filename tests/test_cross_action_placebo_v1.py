from __future__ import annotations

from dataclasses import replace

import pytest

from bayesian_phystwin_experiments.cross_action_placebo_v1 import (
    CrossActionPlaceboProtocolV1,
    CrossActionPlaceboResultV1,
    CrossActionPlaceboScoreRowV1,
    PlaceboArm,
    PlaceboDecision,
    SealedCrossActionPlaceboPredictionV1,
)

DIGEST = "a" * 64


def _protocol(session_count: int = 12) -> CrossActionPlaceboProtocolV1:
    placebos = (
        PlaceboArm.WRONG_ACTION,
        PlaceboArm.WRONG_OBJECT,
        PlaceboArm.PHASE_SHIFTED,
        PlaceboArm.IDENTITY_PERMUTED,
    )
    physical = "guarded_physical"
    arm_labels = (physical, *(arm.value for arm in placebos))
    return CrossActionPlaceboProtocolV1(
        parent_transport_protocol_id=DIGEST,
        target_roster_id="b" * 64,
        action_ids=("action-b", "action-a"),
        target_session_ids=tuple(
            f"session-{index:02d}" for index in range(session_count)
        ),
        physical_arm_label=physical,
        placebo_arms=placebos,
        arm_construction_ids={
            arm: f"{index + 20:064x}" for index, arm in enumerate(arm_labels)
        },
        minimum_sessions=min(10, session_count),
        bootstrap_replicates=1000,
        bootstrap_seed=20260824,
        confidence_level=0.95,
        minimum_placebo_contrast=0.1,
        metadata={"target_outcomes_used": False},
    )


def _rows(
    protocol: CrossActionPlaceboProtocolV1,
    *,
    physical_score: float = 1.0,
    placebo_scores: dict[str, float] | None = None,
    physical_selected: bool = True,
) -> list[CrossActionPlaceboScoreRowV1]:
    scores = placebo_scores or {arm.value: 2.0 for arm in protocol.placebo_arms}
    rows: list[CrossActionPlaceboScoreRowV1] = []
    for session_index, session in enumerate(protocol.target_session_ids):
        for pair_index, (source, target) in enumerate(
            protocol.off_diagonal_action_pairs
        ):
            outcome_id = f"{pair_index + 1:064x}"
            parent_prediction_id = f"{session_index * 10 + pair_index + 40:064x}"
            fallback_artifact = f"{session_index * 10 + pair_index + 60:064x}"
            for arm_index, arm in enumerate(protocol.arm_labels):
                prediction_artifact_id = (
                    f"{session_index * 100 + pair_index * 10 + arm_index + 100:064x}"
                    if physical_selected
                    else fallback_artifact
                )
                prediction = SealedCrossActionPlaceboPredictionV1(
                    protocol_id=protocol.protocol_id,
                    object_session_id=session,
                    source_action_id=source,
                    target_action_id=target,
                    arm_label=arm,
                    parent_transport_prediction_id=parent_prediction_id,
                    construction_id=protocol.arm_construction_ids[arm],
                    prediction_artifact_id=prediction_artifact_id,
                    prediction_batch_id="c" * 64,
                    commit_id="d" * 40,
                    candidate_selected=physical_selected,
                    exact_fallback=not physical_selected,
                    prediction_sealed_before_target=True,
                )
                rows.append(
                    CrossActionPlaceboScoreRowV1(
                        prediction=prediction,
                        target_outcome_id=outcome_id,
                        target_access_attestation_id="e" * 64,
                        scorer_id="f" * 64,
                        proper_score=(
                            physical_score
                            if arm == protocol.physical_arm_label
                            else scores[arm]
                        ),
                    )
                )
    return rows


def test_placebo_separation_requires_all_registered_controls() -> None:
    protocol = _protocol()
    result = CrossActionPlaceboResultV1(protocol, _rows(protocol))

    assert result.decision is PlaceboDecision.SUPPORTED
    assert result.selected_physical_prediction_count == 24
    assert result.mean_placebo_contrasts.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert (result.placebo_contrast_intervals[:, 0] > 0.1).all()
    assert not result.session_placebo_contrasts.flags.writeable
    assert result.to_record()["result_id"] == result.result_id


def test_one_tied_placebo_blocks_the_conjunctive_claim() -> None:
    protocol = _protocol()
    scores = {arm.value: 2.0 for arm in protocol.placebo_arms}
    scores[PlaceboArm.WRONG_ACTION.value] = 1.0
    result = CrossActionPlaceboResultV1(
        protocol,
        _rows(protocol, placebo_scores=scores),
    )

    assert result.decision is PlaceboDecision.NOT_SUPPORTED
    index = [arm.value for arm in protocol.placebo_arms].index("wrong_action")
    assert result.mean_placebo_contrasts[index] == 0.0


def test_all_fallback_predictions_cannot_support_transport() -> None:
    protocol = _protocol()
    result = CrossActionPlaceboResultV1(
        protocol,
        _rows(protocol, physical_selected=False),
    )
    assert result.decision is PlaceboDecision.NOT_SUPPORTED
    assert result.selected_physical_prediction_count == 0


def test_complete_matrix_and_common_outcome_are_fail_closed() -> None:
    protocol = _protocol()
    rows = _rows(protocol)
    with pytest.raises(ValueError, match="complete matrix"):
        CrossActionPlaceboResultV1(protocol, rows[:-1])

    drifted = rows.copy()
    drifted[1] = replace(drifted[1], target_outcome_id="1" * 64)
    with pytest.raises(ValueError, match="same outcome"):
        CrossActionPlaceboResultV1(protocol, drifted)


def test_construction_parent_batch_and_scorer_drift_are_rejected() -> None:
    protocol = _protocol()
    rows = _rows(protocol)

    changed_prediction = replace(rows[1].prediction, construction_id="1" * 64)
    with pytest.raises(ValueError, match="construction"):
        CrossActionPlaceboResultV1(
            protocol,
            [rows[0], replace(rows[1], prediction=changed_prediction), *rows[2:]],
        )

    changed_parent = replace(
        rows[1].prediction,
        parent_transport_prediction_id="2" * 64,
    )
    with pytest.raises(ValueError, match="same parent"):
        CrossActionPlaceboResultV1(
            protocol,
            [rows[0], replace(rows[1], prediction=changed_parent), *rows[2:]],
        )

    changed_batch = replace(rows[1].prediction, prediction_batch_id="3" * 64)
    with pytest.raises(ValueError, match="one sealed batch"):
        CrossActionPlaceboResultV1(
            protocol,
            [rows[0], replace(rows[1], prediction=changed_batch), *rows[2:]],
        )

    with pytest.raises(ValueError, match="one frozen scorer"):
        CrossActionPlaceboResultV1(
            protocol,
            [rows[0], replace(rows[1], scorer_id="4" * 64), *rows[2:]],
        )


def test_fallback_must_be_byte_identical_across_all_controls() -> None:
    protocol = _protocol()
    rows = _rows(protocol, physical_selected=False)
    changed = replace(rows[1].prediction, prediction_artifact_id="5" * 64)
    with pytest.raises(ValueError, match="identical prediction artifact"):
        CrossActionPlaceboResultV1(
            protocol,
            [rows[0], replace(rows[1], prediction=changed), *rows[2:]],
        )


def test_result_identity_is_invariant_to_score_row_order() -> None:
    protocol = _protocol()
    rows = _rows(protocol)
    forward = CrossActionPlaceboResultV1(protocol, rows)
    reverse = CrossActionPlaceboResultV1(protocol, tuple(reversed(rows)))
    assert forward.result_id == reverse.result_id


def test_protocol_and_rows_reject_target_informed_designs() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="target-selection free"):
        replace(protocol, target_outcomes_used_for_selection=True)
    with pytest.raises(ValueError, match="duplicates"):
        replace(protocol, action_ids=("same", "same"))
    with pytest.raises(ValueError, match="exactly every registered arm"):
        replace(protocol, arm_construction_ids={"guarded_physical": DIGEST})

    row = _rows(protocol)[0]
    with pytest.raises(ValueError, match="sealed before target access"):
        replace(row.prediction, target_outcomes_used=True)
    with pytest.raises(ValueError, match="target-side"):
        replace(row, target_side_selection_used=True)
