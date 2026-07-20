import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_query_field_development as query_development
import bayesian_phystwin.deform360_rigid_residual_development as ablation


def _proper_rotation(angle: float) -> np.ndarray:
    axis = np.asarray([0.3, -0.5, 0.8], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def test_pure_rigid_field_is_exact_and_exact_anchors_bypass_decoder() -> None:
    rng = np.random.default_rng(21)
    anchors = rng.normal(size=(12, 3)).astype(np.float32)
    queries = np.concatenate(
        (anchors[[5, 2]], rng.normal(size=(9, 3)).astype(np.float32)), axis=0
    )
    rotation = _proper_rotation(0.61)
    translation = np.asarray([0.4, -0.3, 0.2])
    moved = (anchors.astype(np.float64) @ rotation.T + translation).astype(np.float32)
    trajectory = np.stack((anchors, moved))
    neighbor_indices = np.tile(
        np.asarray([0, 1, 2, 3], dtype=np.int64), (len(queries), 1)
    )
    weights = np.full((len(queries), 4), 0.25, dtype=np.float64)
    exact = np.full(len(queries), -1, dtype=np.int64)
    exact[:2] = [5, 2]

    output, determinants = ablation.query_proper_kabsch_residual_trajectory(
        trajectory,
        anchors,
        queries,
        neighbor_indices,
        weights,
        exact,
    )

    truth = (queries.astype(np.float64) @ rotation.T + translation).astype(np.float32)
    np.testing.assert_allclose(output[1], truth, atol=2e-7, rtol=0.0)
    np.testing.assert_array_equal(output[:, :2], trajectory[:, [5, 2]])
    np.testing.assert_allclose(determinants, 1.0, atol=1e-12, rtol=0.0)
    assert not output.flags.writeable
    assert not determinants.flags.writeable


def test_kabsch_enforces_proper_rotation_for_reflected_target() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(20, 3))
    reflected = source.copy()
    reflected[:, 2] *= -1.0

    rotation, _ = ablation.fit_proper_kabsch_transform(source, reflected)

    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)


def _rotating_case(
    *, frames: int = 4, identities: int = 280
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    frame_zero = rng.uniform(-0.1, 0.1, size=(identities, 3)).astype(np.float32)
    comparator = np.repeat(frame_zero[None], frames, axis=0)
    primary = comparator.copy()
    for frame in range(1, frames):
        rotation = _proper_rotation(0.09 * frame)
        translation = np.asarray([0.002 * frame, -0.001 * frame, 0.0])
        primary[frame] = (
            frame_zero.astype(np.float64) @ rotation.T + translation
        ).astype(np.float32)
    target = primary.copy()
    visible = np.ones((frames, identities), dtype=bool)
    valid = np.ones_like(visible)
    return primary, comparator, target, visible, valid


def test_case_ablation_reuses_center_excluded_split_and_scores_both_operators() -> None:
    primary, comparator, target, visible, valid = _rotating_case()
    centers = np.arange(16, dtype=np.int64)
    candidate = next(
        value
        for value in query_development._candidate_grid()
        if value.neighbor_count == 4 and value.length_scale_fraction == 0.05
    )

    report, arrays = ablation.evaluate_rigid_residual_case_arrays(
        primary,
        comparator,
        target,
        visible,
        valid,
        centers,
        anchor_count=64,
        candidate=candidate,
        scored_frames=(1, 2, 3),
    )

    assert set(report["operators"]) == {
        ablation.TOTAL_OPERATOR_ID,
        ablation.RIGID_RESIDUAL_OPERATOR_ID,
    }
    assert not np.any(np.isin(arrays["anchor_ids"], centers))
    assert not np.any(np.isin(arrays["query_ids"], centers))
    assert not np.any(np.isin(arrays["query_ids"], arrays["anchor_ids"]))
    rigid = report["operators"][ablation.RIGID_RESIDUAL_OPERATOR_ID]
    total = report["operators"][ablation.TOTAL_OPERATOR_ID]
    assert rigid["field_native_fidelity"]["primary"]["identity_rmse_m"] < 2e-8
    assert (
        rigid["target_scores"]["primary"]["identity_rmse_m"]
        < rigid["target_scores"]["comparator"]["identity_rmse_m"]
    )
    assert (
        rigid["field_native_fidelity"]["primary"]["identity_rmse_m"]
        < total["field_native_fidelity"]["primary"]["identity_rmse_m"]
    )
    assert rigid["proper_kabsch"]["exact_anchor_bypass_count"] == 0


def test_writer_is_strict_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = {
        "artifact_kind": ablation.ARTIFACT_KIND,
        "operator_decision": {"selected_operator_id": ablation.TOTAL_OPERATOR_ID},
    }
    monkeypatch.setattr(
        ablation,
        "build_rigid_residual_development_ablation",
        lambda *_: decision,
    )
    output = tmp_path / "nested" / "ablation.json"

    returned = ablation.write_rigid_residual_development_ablation("a", "b", output)

    assert returned == decision
    assert json.loads(output.read_text()) == decision
    with pytest.raises(FileExistsError):
        ablation.write_rigid_residual_development_ablation("a", "b", output)


def test_writer_rejects_nonfinite_json_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ablation,
        "build_rigid_residual_development_ablation",
        lambda *_: {"bad": float("nan")},
    )
    output = tmp_path / "bad.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        ablation.write_rigid_residual_development_ablation("a", "b", output)
    assert not output.exists()
