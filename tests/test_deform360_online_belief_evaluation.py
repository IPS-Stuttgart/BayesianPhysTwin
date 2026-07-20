import hashlib
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_online_belief import (
    deterministic_farthest_point_ids,
)
from bayesian_phystwin.deform360_online_belief_evaluation import (
    UPDATE_FRAMES,
    _validate_deform360_outcome_manifest,
    evaluate_deform360_online_belief_arrays,
    score_deform360_hidden_trajectory,
)


def _synthetic_deform360_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    frame_count = 76
    point_count = 24
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    initial = np.stack(
        (
            0.08 * np.cos(angle),
            0.05 * np.sin(angle),
            np.linspace(-0.02, 0.02, point_count),
        ),
        axis=1,
    ).astype(np.float32)
    physical_prior = np.repeat(initial[None], frame_count, axis=0)
    persistence = physical_prior.copy()
    target = physical_prior.copy()
    target[1:, :, 0] += 0.01
    visibility = np.ones((frame_count, point_count), dtype=bool)
    validity = np.ones_like(visibility)
    return physical_prior, persistence, target, visibility, validity


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_outcome_manifest_binds_seal_target_and_episode(tmp_path: Path) -> None:
    seal_path = tmp_path / "prediction_seal.json"
    target_path = tmp_path / "target_data.pkl"
    seal_path.write_text('{"sealed": true}\n', encoding="utf-8")
    target_path.write_bytes(b"target-payload")
    seal = {
        "object_id": "object",
        "episode_id": 3,
        "episode_key": "object/3",
    }
    outcome = {
        "artifact_kind": "Deform360IndependentSourceOutcome",
        **seal,
        "input_sha256": {"prediction_seal": _file_sha256(seal_path)},
        "output_sha256": {"target_data": _file_sha256(target_path)},
    }

    _validate_deform360_outcome_manifest(seal_path, target_path, seal, outcome)

    mismatched_target = {**outcome, "output_sha256": {"target_data": "0" * 64}}
    with pytest.raises(ValueError, match="target-data checksum"):
        _validate_deform360_outcome_manifest(
            seal_path,
            target_path,
            seal,
            mismatched_target,
        )

    mismatched_episode = {**outcome, "episode_id": 4}
    with pytest.raises(ValueError, match="episode_id differs"):
        _validate_deform360_outcome_manifest(
            seal_path,
            target_path,
            seal,
            mismatched_episode,
        )


def test_risk_rejection_is_bit_exact_physical_prior() -> None:
    prior, persistence, target, visibility, validity = _synthetic_deform360_arrays()
    # No update can see more than eight points.  The exact FPS identities do
    # not need to be known: their intersection with these eight is at most 8.
    for frame in UPDATE_FRAMES:
        visibility[frame] = False
        visibility[frame, :8] = True

    report, arrays = evaluate_deform360_online_belief_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
    )

    np.testing.assert_array_equal(arrays["recursive_rbf_risk_limited_m"], prior)
    np.testing.assert_array_equal(arrays["recursive_global_translation_m"], prior)
    np.testing.assert_array_equal(arrays["recursive_rbf_causal_continuation_m"], prior)
    np.testing.assert_array_equal(arrays["recursive_rbf_correspondence_safe_m"], prior)
    np.testing.assert_array_equal(arrays["risk_limited_frozen_current_state_m"], prior)
    assert report["risk_gate"]["accepted_update_count"] == 0
    assert all(not update["accepted"] for update in report["updates"])
    assert all(
        not update["correspondence_safe_accepted"] for update in report["updates"]
    )
    assert all(
        update["decision"] == "insufficient_support_exact_prior"
        for update in report["updates"]
    )


def test_incoherent_residual_rejection_is_bit_exact_physical_prior() -> None:
    prior, persistence, target, visibility, validity = _synthetic_deform360_arrays()
    # The history has a coherent global residual and therefore retains the
    # fixed 10 mm threshold.  At each update, a geometry-dependent residual
    # spans the object and must be rejected despite complete support.
    for frame in UPDATE_FRAMES:
        target[frame] += 2.0 * prior[0]

    report, arrays = evaluate_deform360_online_belief_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
    )

    np.testing.assert_array_equal(arrays["recursive_rbf_risk_limited_m"], prior)
    np.testing.assert_array_equal(arrays["recursive_global_translation_m"], prior)
    np.testing.assert_array_equal(arrays["recursive_rbf_causal_continuation_m"], prior)
    np.testing.assert_array_equal(arrays["recursive_rbf_correspondence_safe_m"], prior)
    np.testing.assert_array_equal(arrays["risk_limited_frozen_current_state_m"], prior)
    assert not np.array_equal(arrays["recursive_rbf_ungated_m"], prior)
    assert report["risk_gate"]["accepted_update_count"] == 0
    assert all(update["available_center_count"] == 16 for update in report["updates"])
    assert all(not update["accepted"] for update in report["updates"])
    assert all(
        not update["correspondence_safe_accepted"] for update in report["updates"]
    )
    assert all(
        update["decision"] == "incoherent_residual_exact_prior"
        for update in report["updates"]
    )


def test_accepted_update_exports_only_post_update_conformal_half_widths() -> None:
    prior, persistence, target, visibility, validity = _synthetic_deform360_arrays()

    report, arrays = evaluate_deform360_online_belief_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
    )

    half_width = arrays["recursive_rbf_risk_limited_conformal_q90_half_width_m"]
    assert np.all(np.isnan(half_width[: UPDATE_FRAMES[0] + 1]))
    for frame in range(UPDATE_FRAMES[0] + 1, len(half_width)):
        if frame in UPDATE_FRAMES:
            assert np.isnan(half_width[frame])
        else:
            assert half_width[frame] == np.float32(0.01)
    assert np.isclose(
        report["updates"][0]["conformal_style_absolute_residual_half_width_m"]["0.90"],
        0.01,
    )
    assert (
        "formal iid coverage guarantee"
        in report["uncertainty_contract"]["dependence_warning"]
    )


def test_causal_continuation_selector_freezes_unobserved_physical_motion() -> None:
    prior, persistence, _, visibility, validity = _synthetic_deform360_arrays()
    for frame in range(len(prior)):
        prior[frame, :, 0] += 0.001 * frame
    target = persistence.copy()

    report, arrays = evaluate_deform360_online_belief_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
    )

    assert all(update["accepted"] for update in report["updates"])
    assert all(
        update["causal_huber_continuation_gain"] == 0.0 for update in report["updates"]
    )
    assert all(
        update["causal_continuation_selected"] is False for update in report["updates"]
    )
    np.testing.assert_array_equal(
        arrays["recursive_rbf_causal_continuation_m"],
        arrays["risk_limited_frozen_current_state_m"],
    )
    assert not np.array_equal(
        arrays["recursive_rbf_causal_continuation_m"],
        arrays["recursive_rbf_risk_limited_m"],
    )


def test_correspondence_safe_arm_rejects_four_of_sixteen_mismatches() -> None:
    prior, persistence, target, visibility, validity = _synthetic_deform360_arrays()
    centers = deterministic_farthest_point_ids(
        prior[0],
        np.arange(prior.shape[1]),
        16,
    )
    measurement = target.copy()
    destinations = centers[:4]
    sources = np.roll(destinations, 2)
    for frame in UPDATE_FRAMES:
        measurement[frame, destinations] = target[frame, sources]

    report, arrays = evaluate_deform360_online_belief_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
        measurement_m=measurement,
    )

    assert all(update["accepted"] for update in report["updates"])
    assert all(
        update["correspondence_safe_inlier_count"] == 12 for update in report["updates"]
    )
    assert all(
        not update["correspondence_safe_accepted"] for update in report["updates"]
    )
    np.testing.assert_array_equal(
        arrays["recursive_rbf_correspondence_safe_m"],
        prior,
    )
    assert not np.array_equal(arrays["recursive_rbf_risk_limited_m"], prior)
    assert (
        report["observation_contract"]["measurement_stream_is_scoring_target"] is False
    )


def test_assimilation_centres_are_excluded_from_both_metrics() -> None:
    target = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ],
        dtype=float,
    )
    trajectory = target.copy()
    trajectory[1, 0] = [100.0, -100.0, 50.0]
    visibility = np.ones((2, 4), dtype=bool)
    validity = np.ones_like(visibility)

    score = score_deform360_hidden_trajectory(
        trajectory,
        target,
        visibility,
        validity,
        center_ids=np.array([0]),
        scored_frames=(1,),
    )

    assert score["post_update_hidden_identity_rmse_m"] == 0.0
    assert score["post_update_hidden_symmetric_chamfer_m"] == 0.0
    assert score["permanently_excluded_center_count"] == 1
