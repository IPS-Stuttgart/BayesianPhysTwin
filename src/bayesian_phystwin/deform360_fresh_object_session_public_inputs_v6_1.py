"""Disjoint MotionCrafter prefix adapter for the Deform360 v6.1 candidate.

The frozen v5 public-input implementation is reused for its numerical path.
This module changes only typed provenance: the sealed input is the disjoint
MotionCrafter baseline, not Prob4D decoded-uniform overlap fusion. It exposes no
endpoint, suffix, target, or outcome interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._portable_contracts import nonempty_string
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseExtractionConfigV5,
    Deform360JointSparsePrefixFitV5,
    Deform360JointSparseVisualWindowRowsV5,
)
from .deform360_joint_sparse_public_inputs_v5 import (
    Deform360JointSparseMetricGaugeFitV5,
    prepare_deform360_joint_sparse_visual_window_v5,
)


def _disjoint_sources(
    sources: Mapping[str, str],
    *,
    camera_id: str,
) -> dict[str, str]:
    legacy = f"prob4d-decoded-uniform/{camera_id}.npz"
    disjoint = f"motioncrafter-disjoint-baseline/{camera_id}.npz"
    result = dict(sources)
    digest = result.pop(legacy)
    if disjoint in result and result[disjoint] != digest:
        raise ValueError("disjoint MotionCrafter source digest conflicts")
    result[disjoint] = digest
    return result


def prepare_deform360_disjoint_visual_window_v6_1(
    *,
    camera_id: str,
    disjoint_motioncrafter_path: str | Path,
    metric_prefix_path: str | Path,
    raw_prefix_range_half_open: tuple[int, int],
    fit: Deform360JointSparsePrefixFitV5,
    source_artifact_ids: Mapping[str, str],
    extraction_config: Deform360JointSparseExtractionConfigV5 | None = None,
    metric_cluster_size_pixels: int = 32,
) -> tuple[
    Deform360JointSparseVisualWindowRowsV5,
    Deform360JointSparseMetricGaugeFitV5,
]:
    """Run the frozen v5 numerical adapter with truthful disjoint provenance."""

    camera = nonempty_string(camera_id, name="camera_id")
    rows, gauge = prepare_deform360_joint_sparse_visual_window_v5(
        camera_id=camera,
        decoded_uniform_path=disjoint_motioncrafter_path,
        metric_prefix_path=metric_prefix_path,
        raw_prefix_range_half_open=raw_prefix_range_half_open,
        fit=fit,
        source_artifact_ids=source_artifact_ids,
        extraction_config=extraction_config,
        metric_cluster_size_pixels=metric_cluster_size_pixels,
    )
    gauge_sources = _disjoint_sources(gauge.source_artifact_ids, camera_id=camera)
    disjoint_gauge = Deform360JointSparseMetricGaugeFitV5(
        camera_id=gauge.camera_id,
        raw_frame_index=gauge.raw_frame_index,
        linear=gauge.linear,
        translation=gauge.translation,
        input_pair_count=gauge.input_pair_count,
        inlier_pair_count=gauge.inlier_pair_count,
        independent_cluster_count=gauge.independent_cluster_count,
        inlier_independent_cluster_count=gauge.inlier_independent_cluster_count,
        inlier_rmse_m=gauge.inlier_rmse_m,
        source_artifact_ids=gauge_sources,
    )
    row_sources = _disjoint_sources(rows.source_artifact_ids, camera_id=camera)
    row_sources[f"metric-gauge/{camera}.json"] = disjoint_gauge.artifact_id
    disjoint_rows = Deform360JointSparseVisualWindowRowsV5(
        camera_id=rows.camera_id,
        window_id=f"motioncrafter-disjoint-baseline:{camera}",
        frame_indices=rows.frame_indices,
        pixel_yx=rows.pixel_yx,
        point_world_m=rows.point_world_m,
        point_covariance_m2=rows.point_covariance_m2,
        source_confidence=rows.source_confidence,
        mask_distance_pixels=rows.mask_distance_pixels,
        overlap_disagreement_m=rows.overlap_disagreement_m,
        contributor_count=rows.contributor_count,
        source_artifact_ids=row_sources,
    )
    return disjoint_rows, disjoint_gauge


__all__ = ["prepare_deform360_disjoint_visual_window_v6_1"]
