from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_fresh_object_session_candidate_v6_1 as candidate
from bayesian_phystwin.deform360_fresh_object_session_source_v6 import (
    B0,
    B1,
    D1_NATIVE,
    VT1_OBSERVED,
    VT1_SANDWICH,
    VT1_WORKING,
)
from bayesian_phystwin.deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseVisualWindowRowsV5,
)

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_candidate_producer.json"
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _physical(*, nodes: int = 8) -> np.ndarray:
    frame = np.arange(76, dtype=np.float64)[:, None, None]
    node = np.linspace(-0.04, 0.04, nodes, dtype=np.float64)[None, :, None]
    result = np.zeros((76, nodes, 3), dtype=np.float64)
    result[..., 0] = node[..., 0] + 0.0001 * frame[..., 0]
    result[..., 1] = np.linspace(-0.01, 0.01, nodes)[None]
    result[..., 2] = 0.5
    return result


def _window(
    physical: np.ndarray,
    *,
    residual_scale: float = 1.0,
    duplicate: bool = False,
    covariance_scale: float = 1.0,
) -> Deform360JointSparseVisualWindowRowsV5:
    samples_per_frame = min(4, physical.shape[1])
    frames = np.repeat(np.arange(50, 58, dtype=np.int64), samples_per_frame)
    node_indices = np.tile(np.arange(samples_per_frame, dtype=np.int64), 8)
    points = physical[frames, node_indices].copy()
    points[:, 0] += residual_scale * (frames - 49) * 0.0002
    count = len(frames)
    if duplicate:
        frames = np.repeat(frames, 2)
        points = np.repeat(points, 2, axis=0)
        count *= 2
    return Deform360JointSparseVisualWindowRowsV5(
        camera_id="cam0",
        window_id="motioncrafter-disjoint-baseline:cam0",
        frame_indices=frames,
        pixel_yx=np.column_stack(
            (
                np.arange(count, dtype=np.int64) // 64,
                np.arange(count, dtype=np.int64) % 64,
            )
        ),
        point_world_m=points,
        point_covariance_m2=np.broadcast_to(
            covariance_scale * 4e-6 * np.eye(3), (count, 3, 3)
        ).copy(),
        source_confidence=np.full(count, 0.9),
        mask_distance_pixels=np.full(count, 8.0),
        overlap_disagreement_m=np.full(count, 0.001),
        contributor_count=np.ones(count, dtype=np.int64),
        source_artifact_ids={"prefix/cam0.npz": _digest("prefix")},
    )


def _arrays(*, residual_scale: float = 1.0) -> candidate.Deform360V61CandidateArrays:
    physical = _physical()
    b0 = physical.astype(np.float32)
    b1 = b0.copy()
    b1[58:76, :, 0] += 0.001
    return candidate.build_deform360_v61_candidate_arrays(
        physical_prediction_m=physical,
        b0_trajectory_m=b0,
        b1_trajectory_m=b1,
        visual_windows=(_window(physical, residual_scale=residual_scale),),
    )


def _seal_args() -> dict[str, object]:
    return {
        "candidate_revision": "2" * 40,
        "outer_held_out_object_id": "object-0",
        "object_id": "object-1",
        "episode_id": 1,
        "stratum": "sheet",
        "fit_object_ids": tuple(f"object-{index}" for index in range(2, 10)),
        "source_artifacts": {"upstream/prediction.json": _digest("prediction")},
    }


def test_candidate_amendment_is_content_addressed_and_outcome_closed() -> None:
    amendment = candidate.load_deform360_v61_candidate_amendment(AMENDMENT)

    assert amendment["amendment_id"] == candidate.CANDIDATE_AMENDMENT_ID
    assert amendment["information_boundary"]["source_outcomes_used"] is False
    assert amendment["candidate_producer"]["cross_object_parameters_fitted"] is False
    assert (
        amendment["candidate_producer"]["d1_evidence_pooling_scope"]
        == "within-target-object-graph-nodes-only"
    )
    assert amendment["candidate_producer"]["missing_node_policy"] == (
        "invalid-no-nearest-fill"
    )
    assert (
        amendment["public_tactile_boundary"]["vt1_covariance_variants_available"]
        is False
    )


def test_candidate_amendment_rejects_semantic_drift(tmp_path: Path) -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    amendment["candidate_producer"]["missing_node_policy"] = "nearest-fill"
    body = {key: value for key, value in amendment.items() if key != "amendment_id"}
    amendment["amendment_id"] = candidate.content_id(body)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(amendment), encoding="utf-8")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(candidate, "CANDIDATE_AMENDMENT_ID", amendment["amendment_id"])
        with pytest.raises(ValueError, match="producer semantics changed"):
            candidate.load_deform360_v61_candidate_amendment(changed)


def test_residual_history_does_not_nearest_fill_missing_nodes() -> None:
    physical = _physical()
    history = candidate.estimate_deform360_causal_graph_residual_history_v6_1(
        visual_windows=(_window(physical),),
        physical_prediction_m=physical,
    )

    assert np.any(history.valid)
    assert np.any(~history.valid)
    assert np.array_equal(
        history.residual_m[~history.valid], np.zeros((np.sum(~history.valid), 3))
    )


def test_duplicate_correlated_rows_do_not_increase_update_count_or_change_mean() -> (
    None
):
    physical = _physical()
    unique = candidate.build_deform360_v61_candidate_arrays(
        physical_prediction_m=physical,
        b0_trajectory_m=physical,
        b1_trajectory_m=physical,
        visual_windows=(_window(physical),),
    )
    duplicate = candidate.build_deform360_v61_candidate_arrays(
        physical_prediction_m=physical,
        b0_trajectory_m=physical,
        b1_trajectory_m=physical,
        visual_windows=(_window(physical, duplicate=True),),
    )

    assert np.array_equal(
        unique.arrays["posterior_update_count"],
        duplicate.arrays["posterior_update_count"],
    )
    assert np.allclose(
        unique.arrays[f"trajectory__{D1_NATIVE}"],
        duplicate.arrays[f"trajectory__{D1_NATIVE}"],
        atol=1e-14,
        rtol=0.0,
    )
    assert np.allclose(
        unique.arrays["observation_covariance_m2"],
        duplicate.arrays["observation_covariance_m2"],
        atol=1e-14,
        rtol=0.0,
    )


def test_covariance_intersection_is_not_more_confident_than_independent_fusion() -> (
    None
):
    physical = _physical()
    history = candidate.estimate_deform360_causal_graph_residual_history_v6_1(
        visual_windows=(_window(physical), _window(physical)),
        physical_prediction_m=physical,
    )
    single = candidate.estimate_deform360_causal_graph_residual_history_v6_1(
        visual_windows=(_window(physical),),
        physical_prediction_m=physical,
    )

    assert np.allclose(
        history.observation_covariance_m2,
        single.observation_covariance_m2,
        atol=1e-14,
        rtol=0.0,
    )
    finite = history.valid
    naive_independent = single.observation_covariance_m2[finite] / 2.0
    assert np.all(
        np.linalg.eigvalsh(
            history.observation_covariance_m2[finite] - naive_independent
        )
        >= -1e-12
    )


def test_assignment_ambiguity_increases_metric_observation_covariance() -> None:
    physical = _physical(nodes=2)
    physical[:, 0, 0] = -0.001
    physical[:, 1, 0] = 0.001
    ambiguous = _window(physical)
    points = np.array(ambiguous.point_world_m, copy=True)
    points[:, 0] = 0.0
    ambiguous = candidate.Deform360JointSparseVisualWindowRowsV5(
        camera_id=ambiguous.camera_id,
        window_id=ambiguous.window_id,
        frame_indices=ambiguous.frame_indices,
        pixel_yx=ambiguous.pixel_yx,
        point_world_m=points,
        point_covariance_m2=ambiguous.point_covariance_m2,
        source_confidence=ambiguous.source_confidence,
        mask_distance_pixels=ambiguous.mask_distance_pixels,
        overlap_disagreement_m=ambiguous.overlap_disagreement_m,
        contributor_count=ambiguous.contributor_count,
        source_artifact_ids=ambiguous.source_artifact_ids,
    )
    history = candidate.estimate_deform360_causal_graph_residual_history_v6_1(
        visual_windows=(ambiguous,),
        physical_prediction_m=physical,
    )

    assert np.max(history.observation_covariance_m2[..., 0, 0]) > 4e-6


def test_increasing_state_residual_does_not_lower_prior_reliability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = _physical()
    visual = _window(physical)
    captured: list[np.ndarray] = []
    implementation = candidate._prior_perception_reliability  # noqa: SLF001

    def capture(
        source_confidence: np.ndarray,
        mask_distance_pixels: np.ndarray,
        overlap_disagreement_m: np.ndarray,
    ) -> np.ndarray:
        result = implementation(
            source_confidence,
            mask_distance_pixels,
            overlap_disagreement_m,
        )
        captured.append(result.copy())
        return result

    monkeypatch.setattr(candidate, "_prior_perception_reliability", capture)
    candidate.estimate_deform360_causal_graph_residual_history_v6_1(
        visual_windows=(visual,),
        physical_prediction_m=physical,
    )
    shifted = physical.copy()
    shifted[..., 0] += 0.02
    candidate.estimate_deform360_causal_graph_residual_history_v6_1(
        visual_windows=(visual,),
        physical_prediction_m=shifted,
    )

    assert len(captured) == 16
    for first, second in zip(captured[:8], captured[8:], strict=True):
        assert np.array_equal(first, second)
        assert np.all((first >= 0.0) & (first <= 1.0))


def test_candidate_arrays_preserve_upstream_baselines_and_metric_covariance() -> None:
    physical = _physical()
    b0 = physical.astype(np.float32)
    b1 = b0.copy()
    b1[58:76, :, 0] += 0.001
    result = candidate.build_deform360_v61_candidate_arrays(
        physical_prediction_m=physical,
        b0_trajectory_m=b0,
        b1_trajectory_m=b1,
        visual_windows=(_window(physical),),
    )

    assert result.risk_score < 1.0
    for variant_id in (B0, B1, D1_NATIVE):
        covariance = result.arrays[f"covariance__{variant_id}"]
        assert covariance.shape == (18, 8, 3, 3)
        assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    assert result.arrays[f"trajectory__{B0}"].dtype == np.float32
    assert result.arrays[f"trajectory__{B1}"].dtype == np.float32
    assert np.array_equal(result.arrays[f"trajectory__{B0}"], b0)
    assert np.array_equal(result.arrays[f"trajectory__{B1}"], b1)
    assert not np.array_equal(
        result.arrays[f"trajectory__{D1_NATIVE}"],
        result.arrays[f"trajectory__{B0}"],
    )


def test_technical_failure_uses_exact_b0_d1_and_maximum_risk() -> None:
    physical = _physical().astype(np.float32)
    fallback = candidate.build_deform360_v61_technical_fallback_arrays(
        physical_prediction_m=physical,
        b0_trajectory_m=physical,
        b1_trajectory_m=physical,
    )

    assert fallback.risk_score == 1.0
    assert np.array_equal(
        fallback.arrays[f"trajectory__{D1_NATIVE}"],
        fallback.arrays[f"trajectory__{B0}"],
    )
    assert not np.any(fallback.arrays["residual_valid"])


def test_candidate_artifact_round_trip_is_deterministic_and_vt1_unavailable(
    tmp_path: Path,
) -> None:
    arrays = _arrays()
    first = tmp_path / "first"
    second = tmp_path / "second"
    seal = candidate.publish_deform360_v61_candidate_artifact(
        arrays, first, **_seal_args()
    )
    candidate.publish_deform360_v61_candidate_artifact(arrays, second, **_seal_args())
    loaded_seal, loaded = candidate.load_deform360_v61_candidate_artifact(first)

    assert seal == loaded_seal
    assert (first / candidate.CANDIDATE_ARCHIVE_FILENAME).read_bytes() == (
        second / candidate.CANDIDATE_ARCHIVE_FILENAME
    ).read_bytes()
    assert np.array_equal(
        loaded.arrays[f"trajectory__{D1_NATIVE}"],
        arrays.arrays[f"trajectory__{D1_NATIVE}"],
    )
    for variant_id in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH):
        row = seal["variant_artifacts"][variant_id]
        assert row["available"] is False
        assert row["unavailable_reason"] == candidate.PUBLIC_TACTILE_UNAVAILABLE_REASON


def test_candidate_artifact_tamper_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    candidate.publish_deform360_v61_candidate_artifact(_arrays(), root, **_seal_args())
    seal = json.loads((root / candidate.CANDIDATE_SEAL_FILENAME).read_text())
    changed = copy.deepcopy(seal)
    changed["risk_score"] = 0.0
    (root / candidate.CANDIDATE_SEAL_FILENAME).write_text(
        json.dumps(changed), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="identity changed"):
        candidate.load_deform360_v61_candidate_artifact(root)
