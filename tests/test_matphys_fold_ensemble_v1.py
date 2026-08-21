import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.matphys_fold_ensemble_v1 import (
    MATPHYS_FOLD_ENSEMBLE_CLAIM_BOUNDARY,
    apply_bounded_spring_residual,
    assert_target_excluded,
    build_matphys_fold_ensemble_source,
    causal_frame_indices,
    matphys_graph_features,
    trajectory_ensemble_moments,
    validate_matphys_fold_ensemble_source,
)


def _source(tmp_path: Path) -> dict[str, object]:
    universe = ("cloth", "rope", "sloth")
    members = []
    for index, held_out in enumerate(universe):
        checkpoint = tmp_path / f"fold-{index}.pth"
        checkpoint.write_bytes(f"checkpoint {index}".encode())
        audit = tmp_path / f"fold-{index}-audit.json"
        audit.write_text(
            json.dumps({"fold": index, "held_out": held_out}),
            encoding="utf-8",
        )
        members.append(
            {
                "fold_index": index,
                "held_out_object_id": held_out,
                "training_object_ids": tuple(
                    item for item in universe if item != held_out
                ),
                "checkpoint_path": str(checkpoint),
                "training_audit_path": str(audit),
            }
        )
    return build_matphys_fold_ensemble_source(
        source_revision="a" * 40,
        training_universe_object_ids=universe,
        members=members,
        source_artifacts={"protocol/source.json": "b" * 64},
    )


def test_causal_frames_match_training_sampling_and_never_cross_prefix() -> None:
    np.testing.assert_array_equal(
        causal_frame_indices(32),
        np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31]),
    )
    np.testing.assert_array_equal(causal_frame_indices(3), np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="positive integer"):
        causal_frame_indices(0)


def test_source_manifest_binds_target_excluded_fold_checkpoints(tmp_path: Path) -> None:
    source = _source(tmp_path)

    assert source["member_count"] == 3
    assert source["claim_boundary"] == MATPHYS_FOLD_ENSEMBLE_CLAIM_BOUNDARY
    assert validate_matphys_fold_ensemble_source(source, verify_files=True) == source
    assert_target_excluded(source, target_object_id="fresh-deform360-object")
    with pytest.raises(ValueError, match="training includes the target"):
        assert_target_excluded(source, target_object_id="cloth")

    checkpoint = Path(source["members"][0]["checkpoint"]["path"])
    checkpoint.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checkpoint SHA-256 changed"):
        validate_matphys_fold_ensemble_source(source, verify_files=True)


def test_source_manifest_rejects_duplicate_checkpoint_identity(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source["members"][1]["checkpoint"] = dict(source["members"][0]["checkpoint"])
    identity = {key: value for key, value in source.items() if key != "ensemble_id"}
    source["ensemble_id"] = content_id(identity)

    with pytest.raises(ValueError, match="checkpoint SHA-256 values must be unique"):
        validate_matphys_fold_ensemble_source(source, verify_files=False)


def test_graph_features_match_matphys_shapes_and_preserve_edge_order() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.02, 0.01, 0.0],
            [0.02, 0.02, 0.0],
            [0.01, 0.02, 0.0],
        ],
        dtype=np.float32,
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64)
    parts = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    features = matphys_graph_features(points, edges, parts)

    assert features.edge_features.shape == (5, 11)
    assert features.scene_features.shape == (6,)
    np.testing.assert_array_equal(features.edge_part_index, parts[edges[:, 0]])
    assert np.all(np.isfinite(features.edge_features))
    assert features.scene_features[2] == pytest.approx(np.log(7), rel=1e-6)
    assert features.scene_features[3] == pytest.approx(np.log(6), rel=1e-6)

    reversed_order = matphys_graph_features(points, edges[::-1], parts)
    assert reversed_order.graph_sha256 != features.graph_sha256
    np.testing.assert_allclose(
        reversed_order.edge_features,
        features.edge_features[::-1],
        rtol=1e-6,
        atol=1e-6,
    )


def test_bounded_residual_has_exact_zero_identity_and_twofold_cap() -> None:
    incumbent = np.array([1000.0, 5000.0, 10000.0], dtype=np.float32)
    raw = np.array([-1.0e6, 0.0, 1.0e6], dtype=np.float32)

    identity = apply_bounded_spring_residual(
        incumbent,
        raw,
        proposal_strength=0.0,
    )
    assert identity is incumbent
    assert identity.tobytes() == incumbent.tobytes()

    candidate = apply_bounded_spring_residual(
        incumbent,
        raw,
        proposal_strength=1.0,
    )
    np.testing.assert_allclose(candidate / incumbent, np.array([0.5, 1.0, 2.0]))


def test_ensemble_moments_are_psd_and_duplicate_safe() -> None:
    first = np.zeros((3, 2, 3), dtype=np.float32)
    second = first.copy()
    second[..., 0] = 0.01
    members = np.stack((first, second), axis=0)

    base = trajectory_ensemble_moments(members)
    duplicated = trajectory_ensemble_moments(
        np.stack((first, second, first, second), axis=0)
    )

    np.testing.assert_array_equal(duplicated.mean_m, base.mean_m)
    np.testing.assert_array_equal(duplicated.covariance_m2, base.covariance_m2)
    assert duplicated.unique_member_indices.tolist() == [0, 1]
    assert np.min(np.linalg.eigvalsh(base.covariance_m2)) >= -1e-12
    np.testing.assert_allclose(base.covariance_m2[..., 0, 0], 0.000025)

