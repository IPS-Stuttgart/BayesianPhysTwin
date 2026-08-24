from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.trajectory_value_v1 import (
    TRAJECTORY_VALUE_CLAIM_BOUNDARY,
    FrozenActionDecisionValueV1,
    TrajectoryProperScoreConfigV1,
    TrajectoryProperScoreV1,
    VariogramPairV1,
)

SHA = "a" * 64


def _config(
    scale: object,
    *,
    energy_weight: float = 1.0,
    variogram_weight: float = 1.0,
    variogram_power: float = 1.0,
    pairs: tuple[VariogramPairV1, ...] = (VariogramPairV1(0, 1, 1.0),),
    **kwargs: object,
) -> TrajectoryProperScoreConfigV1:
    return TrajectoryProperScoreConfigV1(
        score_definition_id=SHA,
        coordinate_scale_id=SHA,
        coordinate_scale=np.asarray(scale),
        energy_weight=energy_weight,
        variogram_weight=variogram_weight,
        variogram_power=variogram_power,
        variogram_pairs=pairs,
        **kwargs,
    )


def _score(
    samples: object,
    target: object,
    *,
    config: TrajectoryProperScoreConfigV1 | None = None,
    **kwargs: object,
) -> TrajectoryProperScoreV1:
    target_array = np.asarray(target)
    if config is None:
        coordinate_count = target_array.size
        if coordinate_count >= 2:
            config = _config(np.ones(coordinate_count))
        else:
            config = _config(
                np.ones(coordinate_count),
                variogram_weight=0.0,
                pairs=(),
            )
    return TrajectoryProperScoreV1(
        config=config,
        prediction_artifact_id=SHA,
        target_artifact_id=SHA,
        object_session_id="object/session",
        action_id="action-a",
        arm_id="candidate",
        predictive_samples=np.asarray(samples),
        target_trajectory=target_array,
        prediction_sealed_before_target=True,
        **kwargs,
    )


def _decision(
    predictive: object,
    realized: object,
    *,
    actions: tuple[str, ...] = ("action-a", "action-b"),
    **kwargs: object,
) -> FrozenActionDecisionValueV1:
    return FrozenActionDecisionValueV1(
        decision_protocol_id=SHA,
        loss_definition_id=SHA,
        prediction_batch_id=SHA,
        target_access_attestation_id=SHA,
        object_session_id="object/session",
        method_id="candidate",
        action_ids=actions,
        predictive_loss_samples=np.asarray(predictive),
        realized_losses=np.asarray(realized),
        predictions_sealed_before_target=True,
        **kwargs,
    )


def test_identical_distribution_and_target_have_zero_scores() -> None:
    score = _score(
        samples=[
            [[0.0], [1.0]],
            [[0.0], [1.0]],
        ],
        target=[[0.0], [1.0]],
    )

    assert score.energy_score == pytest.approx(0.0)
    assert score.variogram_score == pytest.approx(0.0)
    assert score.total_score == pytest.approx(0.0)


def test_energy_score_penalizes_disperse_distribution() -> None:
    config = _config(
        [1.0],
        variogram_weight=0.0,
        pairs=(),
    )
    score = _score(
        samples=[
            [[-1.0]],
            [[1.0]],
        ],
        target=[[0.0]],
        config=config,
    )

    assert score.energy_score == pytest.approx(0.5)
    assert score.variogram_score == pytest.approx(0.0)
    assert score.total_score == pytest.approx(0.5)


def test_variogram_detects_wrong_dependence_with_equal_marginals() -> None:
    config = _config(
        [1.0, 1.0],
        energy_weight=0.0,
        variogram_weight=1.0,
        variogram_power=1.0,
    )
    coherent = _score(
        samples=[
            [[-1.0], [-1.0]],
            [[1.0], [1.0]],
        ],
        target=[[0.0], [0.0]],
        config=config,
    )
    anticorrelated = _score(
        samples=[
            [[-1.0], [1.0]],
            [[1.0], [-1.0]],
        ],
        target=[[0.0], [0.0]],
        config=config,
    )

    assert coherent.variogram_score == pytest.approx(0.0)
    assert anticorrelated.variogram_score == pytest.approx(4.0)
    np.testing.assert_allclose(
        coherent.predictive_samples[:, :, 0].mean(axis=0),
        anticorrelated.predictive_samples[:, :, 0].mean(axis=0),
    )


def test_coordinate_scale_makes_scores_invariant_to_unit_change() -> None:
    metric = _score(
        samples=[
            [[0.0], [2.0]],
            [[2.0], [0.0]],
        ],
        target=[[0.5], [1.5]],
        config=_config([1.0, 1.0]),
    )
    millimetres = _score(
        samples=[
            [[0.0], [2000.0]],
            [[2000.0], [0.0]],
        ],
        target=[[500.0], [1500.0]],
        config=_config([1000.0, 1000.0]),
    )

    assert millimetres.energy_score == pytest.approx(metric.energy_score)
    assert millimetres.variogram_score == pytest.approx(metric.variogram_score)
    assert millimetres.total_score == pytest.approx(metric.total_score)


def test_action_decision_reports_realized_regret() -> None:
    decision = _decision(
        predictive=[
            [1.0, 1.2, 0.8],
            [2.0, 2.2, 1.8],
        ],
        realized=[3.0, 1.0],
    )

    assert decision.selected_action_id == "action-a"
    assert decision.oracle_action_id == "action-b"
    assert decision.selected_realized_loss == pytest.approx(3.0)
    assert decision.oracle_realized_loss == pytest.approx(1.0)
    assert decision.realized_regret == pytest.approx(2.0)
    assert decision.predictive_selection_margin == pytest.approx(1.0)
    assert not decision.oracle_match


def test_action_decision_ties_break_lexicographically() -> None:
    decision = _decision(
        predictive=[
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        realized=[0.0, 2.0],
    )

    assert decision.selected_action_id == "action-a"
    assert decision.oracle_action_id == "action-a"
    assert decision.realized_regret == pytest.approx(0.0)
    assert decision.predictive_selection_margin == pytest.approx(0.0)
    assert decision.oracle_match


def test_records_are_immutable_content_addressed_and_bounded() -> None:
    samples = np.array(
        [
            [[0.0], [1.0]],
            [[0.0], [1.0]],
        ]
    )
    score = _score(samples, [[0.0], [1.0]], metadata={"split": "held-out"})
    score_id = score.artifact_id
    samples[0, 0, 0] = 9.0

    assert score.predictive_samples[0, 0, 0] == 0.0
    assert not score.predictive_samples.flags.writeable
    assert not score.target_trajectory.flags.writeable
    assert score.to_record()["artifact_id"] == score_id
    assert score.summary()["claim_boundary"] == TRAJECTORY_VALUE_CLAIM_BOUNDARY

    decision = _decision(
        predictive=[[1.0, 1.0], [2.0, 2.0]],
        realized=[1.0, 2.0],
        metadata={"split": "held-out"},
    )
    for array in decision.arrays().values():
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0.0
    assert decision.summary()["claim_boundary"] == (TRAJECTORY_VALUE_CLAIM_BOUNDARY)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "energy_weight": 0.0,
                "variogram_weight": 0.0,
                "pairs": (),
            },
            "at least one score component",
        ),
        (
            {
                "variogram_weight": 1.0,
                "pairs": (),
            },
            "requires registered variogram_pairs",
        ),
        (
            {
                "scale": [1.0, 0.0],
            },
            "strictly positive",
        ),
    ],
)
def test_invalid_score_configurations_fail_closed(
    kwargs: dict[str, object],
    match: str,
) -> None:
    scale = kwargs.pop("scale", [1.0, 1.0])
    with pytest.raises(ValueError, match=match):
        _config(scale, **kwargs)


def test_duplicate_or_out_of_range_variogram_pairs_fail_closed() -> None:
    pair = VariogramPairV1(0, 1, 1.0)
    with pytest.raises(ValueError, match="must not repeat"):
        _config([1.0, 1.0], pairs=(pair, pair))
    with pytest.raises(ValueError, match="exceeds coordinate_scale"):
        _config([1.0, 1.0], pairs=(VariogramPairV1(0, 2, 1.0),))


@pytest.mark.parametrize(
    ("samples", "target", "match"),
    [
        (
            [[[0.0], [1.0]]],
            [[0.0], [1.0]],
            "at least two samples",
        ),
        (
            [
                [[0.0], [1.0]],
                [[0.0], [1.0]],
            ],
            [[0.0]],
            "shapes are incompatible",
        ),
        (
            [
                [[0.0], [np.nan]],
                [[0.0], [1.0]],
            ],
            [[0.0], [1.0]],
            "must be finite",
        ),
    ],
)
def test_invalid_trajectory_scores_fail_closed(
    samples: object,
    target: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _score(samples, target)


def test_target_informed_predictions_fail_closed() -> None:
    with pytest.raises(ValueError, match="target-outcome free"):
        _score(
            [
                [[0.0], [1.0]],
                [[0.0], [1.0]],
            ],
            [[0.0], [1.0]],
            target_outcomes_used_for_prediction=True,
        )
    with pytest.raises(ValueError, match="target-outcome free"):
        _decision(
            [[1.0, 1.0], [2.0, 2.0]],
            [1.0, 2.0],
            target_outcomes_used_for_prediction=True,
        )


@pytest.mark.parametrize(
    ("predictive", "realized", "actions", "match"),
    [
        (
            [[1.0, 1.0]],
            [1.0],
            ("action-a",),
            "at least two actions",
        ),
        (
            [[1.0, 1.0], [2.0, 2.0]],
            [1.0],
            ("action-a", "action-b"),
            "one value per action",
        ),
        (
            [[1.0, 1.0], [2.0, 2.0]],
            [1.0, 2.0],
            ("action-b", "action-a"),
            "must be sorted",
        ),
    ],
)
def test_invalid_decisions_fail_closed(
    predictive: object,
    realized: object,
    actions: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _decision(predictive, realized, actions=actions)
