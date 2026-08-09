from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_joint_sparse_geometric_batch_v4 import (
    _build_object_batch,
    _mode_matrices,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_candidates_v4 import (
    _Candidate,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_common_v4 import (
    MATERIALIZER_CLAIM_BOUNDARY,
    validate_materializer_policy,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_npz_v4 import (
    _load_metric_sparse_frames,
    _load_prediction_support_windows,
)
from bayesian_phystwin.deform360_joint_sparse_observability_v4 import (
    Deform360JointSparseObservabilityPolicyV4,
    evaluate_deform360_joint_sparse_observability_v4,
)

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_POLICY = (
    ROOT
    / "protocols/locks/"
    "deform360_official_hub_joint_sparse_geometric_materializer_v4.json"
)
V4_POLICY = (
    ROOT
    / "protocols/locks/"
    "deform360_official_hub_joint_sparse_observability_v4.json"
)


def _policy() -> dict[str, object]:
    return validate_materializer_policy(
        json.loads(MATERIALIZER_POLICY.read_text(encoding="utf-8"))
    )


def _candidate(
    *,
    camera: str,
    window: str,
    frame: int,
    cluster: int,
    point: tuple[float, float, float],
) -> _Candidate:
    group = hashlib.sha256(f"group:{frame}:{cluster}".encode()).hexdigest()
    return _Candidate(
        job_id=hashlib.sha256(f"job:{camera}".encode()).hexdigest(),
        camera_id=camera,
        window_id=window,
        frame=frame,
        row=cluster,
        column=cluster + 1,
        point_world_m=np.asarray(point, dtype=np.float64),
        camera_center_world_m=np.asarray(
            (-0.4 if camera == "camera-a" else 0.4, 0.0, -1.0),
            dtype=np.float64,
        ),
        spatial_cluster_id=hashlib.sha256(f"cluster:{cluster}".encode()).hexdigest(),
        correlation_group_id=group,
        support_digest=hashlib.sha256(f"support:{window}".encode()).hexdigest(),
    )


def _candidates() -> list[_Candidate]:
    points = (
        (-0.20, -0.10, -0.05),
        (-0.20, 0.10, 0.05),
        (-0.10, -0.20, 0.10),
        (-0.10, 0.20, -0.10),
        (0.10, -0.20, -0.10),
        (0.10, 0.20, 0.10),
        (0.20, -0.10, 0.05),
        (0.20, 0.10, -0.05),
    )
    result: list[_Candidate] = []
    for camera in ("camera-a", "camera-b"):
        for window_index, window in enumerate(("window-0", "window-1")):
            frame = 10 + window_index
            for cluster, point in enumerate(points):
                result.append(
                    _candidate(
                        camera=camera,
                        window=window,
                        frame=frame,
                        cluster=cluster,
                        point=point,
                    )
                )
    return result


def _batch(candidates: list[_Candidate]):
    policy = _policy()
    return _build_object_batch(
        candidates=candidates,
        selection_artifact_sha256=str(policy["selection_artifact_sha256"]),
        visual_provider_lock_id=str(policy["visual_provider_lock_id"]),
        implementation_revision="a" * 40,
        object_id="development-object",
        episode_id=0,
        stratum="sheet",
        excluded_factor_count=0,
        source_artifacts={"source/metric.json": "b" * 64},
        policy=policy,
        metadata={"test_fixture": True},
    )


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _write_prediction_fixture(root: Path) -> Path:
    window = root / "window.npz"
    with zipfile.ZipFile(window, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("window_id.npy", _npy_bytes(np.asarray("window-0")))
        archive.writestr("frame_indices.npy", _npy_bytes(np.asarray([10, 11])))
        archive.writestr(
            "valid_mask.npy",
            _npy_bytes(np.ones((2, 3, 4), dtype=np.bool_)),
        )
        archive.writestr(
            "point_map.npy",
            b"this payload is intentionally not a valid NPY array",
        )
    digest = hashlib.sha256(window.read_bytes()).hexdigest()
    run_spec = {"seed": 1, "support_only_test": True}
    run_spec_sha = hashlib.sha256(
        json.dumps(
            run_spec,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "format_version": 1,
        "motioncrafter_commit": str(_policy()["motioncrafter_revision"]),
        "overlap_windows": [
            {
                "window_id": "window-0",
                "path": window.name,
                "start_frame": 10,
                "stop_frame": 12,
            }
        ],
        "artifact_integrity": {
            "schema": "prob4d.motioncrafter-artifact-integrity.v1",
            "run_spec": run_spec,
            "run_spec_sha256": run_spec_sha,
            "members": [
                {
                    "path": window.name,
                    "sha256": digest,
                    "bytes": window.stat().st_size,
                    "kind": "independently_decoded_overlap_window",
                }
            ],
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_materializer_policy_is_content_addressed_and_target_closed() -> None:
    policy = _policy()
    assert policy["artifact_id"] == (
        "4a643a9e28118257fc56dafbadb11aee6a3f66db96c2b6b43cc89c1a8b4306ed"
    )
    assert policy["claim_boundary"] == MATERIALIZER_CLAIM_BOUNDARY
    assert policy["prediction_support_masks_used"] is True
    assert policy["robot_metric_points_used"] is True
    for field in (
        "prediction_point_values_used",
        "prediction_residuals_used",
        "calibration_outcomes_used",
        "future_frames_used",
        "adaptive_confirmation_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "replacement_allowed",
    ):
        assert policy[field] is False


def test_deformation_basis_is_symmetric_trace_free_and_orthonormal() -> None:
    modes = _mode_matrices()
    assert modes.shape == (5, 3, 3)
    assert np.allclose(modes, modes.swapaxes(1, 2))
    assert np.allclose(np.trace(modes, axis1=1, axis2=2), 0.0)
    gram = np.einsum("aij,bij->ab", modes, modes)
    assert np.allclose(gram, np.eye(5))


def test_batch_is_order_invariant_and_group_power_is_not_duplicated() -> None:
    candidates = _candidates()
    forward = _batch(candidates)
    reverse = _batch(list(reversed(candidates)))
    assert forward.input_id == reverse.input_id
    assert forward.query_jacobian.shape == (5, 5)
    assert np.array_equal(forward.query_jacobian, np.eye(5))
    assert len(set(forward.camera_ids)) == 2
    assert len(set(forward.window_ids)) == 2
    assert len(set(forward.spatial_cluster_ids)) == 8
    grouped: defaultdict[str, float] = defaultdict(float)
    for group, weight in zip(
        forward.correlation_group_ids,
        forward.composite_weight,
        strict=True,
    ):
        grouped[group] += float(weight)
    assert grouped
    assert all(np.isclose(value, 1.0) for value in grouped.values())
    assert forward.metadata["prediction_point_values_used"] is False
    assert forward.metadata["prediction_residuals_used"] is False


def test_joint_sparse_batch_reaches_the_object_level_evaluator() -> None:
    batch = _batch(_candidates())
    policy = Deform360JointSparseObservabilityPolicyV4.from_record(
        json.loads(V4_POLICY.read_text(encoding="utf-8"))
    )
    result = evaluate_deform360_joint_sparse_observability_v4(
        batch,
        policy,
        implementation_revision="a" * 40,
    )
    assert result.status == "evaluated"
    assert result.query_dimension == 5
    assert result.query_rank is not None
    assert result.query_precision_eigenvalues is not None
    assert result.information_boundary["target_outcomes_used"] is False
    assert result.information_boundary["confirmation_payloads_opened"] is False


def test_metric_grid_loader_streams_only_valid_world_points(tmp_path: Path) -> None:
    frames = np.asarray([10, 11], dtype=np.int64)
    points = np.arange(2 * 3 * 4 * 3, dtype=np.float64).reshape(2, 3, 4, 3)
    valid = np.zeros((2, 3, 4), dtype=np.bool_)
    valid[0, 0, 1] = True
    valid[0, 2, 3] = True
    valid[1, 1, 2] = True
    path = tmp_path / "metric.npz"
    np.savez_compressed(
        path,
        frame_indices=frames,
        points_world_m=points,
        valid_mask=valid,
    )
    loaded, image_shape = _load_metric_sparse_frames(
        path,
        causal_range=(10, 12),
    )
    assert image_shape == (3, 4)
    assert sorted(loaded) == [10, 11]
    assert loaded[10].rows.tolist() == [0, 2]
    assert loaded[10].columns.tolist() == [1, 3]
    assert np.array_equal(
        loaded[10].points_world_m,
        points[0, [0, 2], [1, 3]],
    )


def test_prediction_loader_uses_support_only_and_verifies_file_digest(
    tmp_path: Path,
) -> None:
    path = _write_prediction_fixture(tmp_path)
    windows, run_spec = _load_prediction_support_windows(
        path,
        causal_range=(10, 12),
        image_shape=(3, 4),
        expected_motioncrafter_revision=str(_policy()["motioncrafter_revision"]),
    )
    assert len(windows) == 1
    assert windows[0].valid_mask.shape == (2, 3, 4)
    assert len(run_spec) == 64

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifact_integrity"]["members"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _load_prediction_support_windows(
            path,
            causal_range=(10, 12),
            image_shape=(3, 4),
            expected_motioncrafter_revision=str(
                _policy()["motioncrafter_revision"]
            ),
        )
