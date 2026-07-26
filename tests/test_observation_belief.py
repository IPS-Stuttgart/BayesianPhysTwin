from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.observation_belief import (
    ObservationBeliefV1,
    load_observation_belief,
    save_observation_belief,
)

GOLDEN_ARTIFACT_ID = (
    "9c02e638f60424cca7738d347d1258acd208eb562f422efacd077db4edb2fe80"
)


def _belief() -> ObservationBeliefV1:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 1e-4
    factors = np.zeros((4, 3, 2))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return ObservationBeliefV1(
        case_id="case-1",
        stream_id="prob4d:points",
        causal_frame_stop=12,
        view_names=("camera0",),
        window_names=("window0", "window1"),
        factor_names=("gauge_latent_0", "gauge_latent_1"),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([8, 9]),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ]
        ),
        frame_ids=np.asarray([8, 8, 9, 9]),
        entity_ids=np.asarray([0, 1, 0, 1]),
        view_indices=np.zeros(4, dtype=int),
        window_indices=np.asarray([0, 0, 1, 1]),
        correlation_group_ids=np.asarray([0, 0, 1, 1]),
        factor_group_ids=np.asarray([0, 0, 1, 1]),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=np.ones(4),
        local_covariance_m2=local,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata={"causal_source": "prefix only"},
    )


def test_observation_belief_round_trip_and_digest(tmp_path: Path) -> None:
    belief = _belief()
    path = tmp_path / "belief.npz"
    save_observation_belief(path, belief)
    restored = load_observation_belief(path)

    assert belief.artifact_id == GOLDEN_ARTIFACT_ID
    assert restored.artifact_id == belief.artifact_id
    assert restored.summary()["observation_count"] == 4
    assert restored.mean_xyz_m.flags.writeable is False
    np.testing.assert_array_equal(restored.mean_xyz_m, belief.mean_xyz_m)


def test_observation_belief_rejects_future_frame() -> None:
    belief = _belief()
    with pytest.raises(ValueError, match="causal boundary"):
        ObservationBeliefV1(
            **{
                **belief.__dict__,
                "frame_ids": np.asarray([8, 8, 9, 12]),
            }
        )


def test_observation_belief_rejects_duplicate_identity() -> None:
    belief = _belief()
    with pytest.raises(ValueError, match="must be unique"):
        ObservationBeliefV1(
            **{
                **belief.__dict__,
                "entity_ids": np.asarray([0, 0, 0, 1]),
            }
        )


def test_sim3_transform_moves_covariance_and_factors() -> None:
    belief = _belief()
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = belief.transformed(
        rotation=rotation,
        translation_m=np.asarray([1.0, 2.0, 3.0]),
        scale=2.0,
        stream_id="world",
    )

    expected = 2.0 * (rotation @ belief.mean_xyz_m[0]) + np.asarray(
        [1.0, 2.0, 3.0]
    )
    np.testing.assert_allclose(transformed.mean_xyz_m[0], expected)
    np.testing.assert_allclose(
        transformed.local_covariance_m2[0], 4.0 * belief.local_covariance_m2[0]
    )
    assert transformed.artifact_id != belief.artifact_id
