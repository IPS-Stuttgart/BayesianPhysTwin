from __future__ import annotations

import itertools

import numpy as np
import pytest

from bayesian_phystwin.independent_group_inference_v1 import (
    IndependentGroupInferenceV1,
    analyze_independent_group_inference_v1,
)


def _result(
    effects: np.ndarray | None = None,
    *,
    group_ids: tuple[str, ...] = ("g3", "g1", "g2"),
    estimand_ids: tuple[str, ...] = ("candidate-vs-last", "candidate-vs-physics"),
    seed: int = 17,
    replicates: int = 256,
    metadata: dict[str, object] | None = None,
) -> IndependentGroupInferenceV1:
    if effects is None:
        effects = np.asarray(
            [
                [-1.0, -0.4],
                [-0.2, 0.3],
                [-0.8, -0.1],
            ]
        )
    return analyze_independent_group_inference_v1(
        protocol_id="protocol-sha256",
        family_id="two-contrast-family-v1",
        statistical_unit="complete physical object-session",
        within_group_aggregation="equal-horizon mean before group inference",
        group_ids=group_ids,
        estimand_ids=estimand_ids,
        group_effects=effects,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        metadata={} if metadata is None else metadata,
    )


def _brute_force_p_values(effects: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rms = np.sqrt(np.mean(np.square(effects), axis=0))
    observed = np.divide(
        np.mean(effects, axis=0),
        rms,
        out=np.zeros(effects.shape[1]),
        where=rms > 0.0,
    )
    statistics: list[np.ndarray] = []
    for signs in itertools.product((1.0, -1.0), repeat=len(effects)):
        signed = np.asarray(signs)[:, None] * effects
        statistics.append(
            np.divide(
                np.mean(signed, axis=0),
                rms,
                out=np.zeros(effects.shape[1]),
                where=rms > 0.0,
            )
        )
    values = np.asarray(statistics)
    tolerance = 32.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(observed))
    unadjusted = np.mean(values <= observed[None, :] + tolerance[None, :], axis=0)
    minima = np.min(values, axis=1)
    familywise = np.mean(
        minima[:, None] <= observed[None, :] + tolerance[None, :], axis=0
    )
    global_p = float(
        np.mean(minima <= float(np.min(observed)) + float(np.max(tolerance)))
    )
    return unadjusted, familywise, global_p


def test_exact_sign_flip_matches_independent_brute_force() -> None:
    effects = np.asarray(
        [
            [-1.0, 0.4],
            [-0.2, -0.1],
            [0.3, -0.8],
            [-0.7, -0.5],
        ]
    )
    result = _result(
        effects,
        group_ids=("g4", "g2", "g1", "g3"),
        replicates=64,
    )
    expected_unadjusted, expected_familywise, expected_global = (
        _brute_force_p_values(effects[[2, 1, 3, 0]])
    )

    np.testing.assert_array_equal(
        result.exact_unadjusted_p_value,
        expected_unadjusted,
    )
    np.testing.assert_array_equal(
        result.exact_familywise_p_value,
        expected_familywise,
    )
    assert result.exact_global_family_p_value == expected_global
    assert np.all(
        result.exact_familywise_p_value >= result.exact_unadjusted_p_value
    )
    assert result.sign_pattern_count == 2 ** len(effects)


def test_constant_negative_effect_has_exact_smallest_sign_flip_probability() -> None:
    result = _result(
        np.full((3, 2), [-1.0, -2.0]),
        replicates=32,
    )

    np.testing.assert_array_equal(result.exact_unadjusted_p_value, [1 / 8, 1 / 8])
    np.testing.assert_array_equal(result.exact_familywise_p_value, [1 / 8, 1 / 8])
    assert result.exact_global_family_p_value == 1 / 8
    np.testing.assert_array_equal(result.standard_error, [0.0, 0.0])
    np.testing.assert_array_equal(result.simultaneous_superiority_upper, [-1.0, -2.0])


def test_zero_effect_is_neutral_and_has_probability_one() -> None:
    result = _result(np.zeros((3, 2)), replicates=32)

    np.testing.assert_array_equal(result.observed_mean, [0.0, 0.0])
    np.testing.assert_array_equal(result.exact_unadjusted_p_value, [1.0, 1.0])
    np.testing.assert_array_equal(result.exact_familywise_p_value, [1.0, 1.0])
    assert result.exact_global_family_p_value == 1.0
    np.testing.assert_array_equal(result.win_count, [0, 0])
    np.testing.assert_array_equal(result.tie_count, [3, 3])
    np.testing.assert_array_equal(result.harm_count, [0, 0])


def test_group_input_order_is_canonicalized_without_changing_identity() -> None:
    first = _result()
    second = _result(
        np.asarray(
            [
                [-0.2, 0.3],
                [-0.8, -0.1],
                [-1.0, -0.4],
            ]
        ),
        group_ids=("g1", "g2", "g3"),
    )

    assert first.group_ids == ("g1", "g2", "g3")
    assert first.artifact_id == second.artifact_id
    assert first.to_payload() == second.to_payload()


def test_estimand_input_order_is_canonicalized_with_columns() -> None:
    first = _result()
    second = _result(
        np.asarray(
            [
                [-0.4, -1.0],
                [0.3, -0.2],
                [-0.1, -0.8],
            ]
        ),
        estimand_ids=("candidate-vs-physics", "candidate-vs-last"),
    )

    assert first.estimand_ids == (
        "candidate-vs-last",
        "candidate-vs-physics",
    )
    assert first.artifact_id == second.artifact_id
    assert first.to_payload() == second.to_payload()


def test_count_properties_report_the_independent_family_shape() -> None:
    result = _result(replicates=16)

    assert result.group_count == 3
    assert result.estimand_count == 2


def test_seed_changes_only_bootstrap_record_not_exact_randomization() -> None:
    first = _result(seed=3)
    second = _result(seed=4)

    np.testing.assert_array_equal(
        first.exact_unadjusted_p_value,
        second.exact_unadjusted_p_value,
    )
    np.testing.assert_array_equal(
        first.exact_familywise_p_value,
        second.exact_familywise_p_value,
    )
    assert first.sign_pattern_sha256 == second.sign_pattern_sha256
    assert first.bootstrap_index_sha256 != second.bootstrap_index_sha256
    assert first.bootstrap_mean_sha256 != second.bootstrap_mean_sha256
    assert first.artifact_id != second.artifact_id


def test_shared_bootstrap_pairing_preserves_proportional_estimands() -> None:
    first = np.asarray([-1.0, -0.4, 0.2, -0.8])
    effects = np.column_stack((first, 2.0 * first))
    result = _result(
        effects,
        group_ids=("g1", "g2", "g3", "g4"),
        replicates=512,
    )

    np.testing.assert_allclose(result.observed_mean[1], 2.0 * result.observed_mean[0])
    np.testing.assert_allclose(
        result.pointwise_interval_lower[1],
        2.0 * result.pointwise_interval_lower[0],
    )
    np.testing.assert_allclose(
        result.pointwise_interval_upper[1],
        2.0 * result.pointwise_interval_upper[0],
    )
    np.testing.assert_allclose(
        result.simultaneous_interval_lower[1],
        2.0 * result.simultaneous_interval_lower[0],
    )
    np.testing.assert_allclose(
        result.simultaneous_interval_upper[1],
        2.0 * result.simultaneous_interval_upper[0],
    )


def test_outputs_own_immutable_bytes_and_ignore_caller_mutation() -> None:
    effects = np.asarray([[-1.0, -0.4], [-0.2, 0.3], [-0.8, -0.1]])
    result = _result(effects)
    original = result.group_effects.copy()
    effects[:] = 100.0

    np.testing.assert_array_equal(result.group_effects, original)
    for array in (
        result.group_effects,
        result.observed_mean,
        result.exact_unadjusted_p_value,
        result.win_count,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_metadata_is_copied_before_content_addressing() -> None:
    metadata: dict[str, object] = {"repository": {"revision": "abc"}}
    result = _result(metadata=metadata)
    artifact_id = result.artifact_id
    cast_nested = metadata["repository"]
    assert isinstance(cast_nested, dict)
    cast_nested["revision"] = "changed"

    assert result.metadata["repository"]["revision"] == "abc"
    assert result.artifact_id == artifact_id
