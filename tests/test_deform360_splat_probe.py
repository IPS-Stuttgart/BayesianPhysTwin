from __future__ import annotations

import numpy as np
import pytest

from causal4d_public.deform360_splat_probe import (
    ThinRopeSplatProbeConfig,
    gaussian_splat_geometry_diagnostics,
    splat_probe_artifact_sha256,
    validate_splat_probe_artifact,
)


def _camera(center_x: float) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray([[120.0, 0.0, 64.0], [0.0, 120.0, 64.0], [0.0, 0.0, 1.0]])
    camera_to_world = np.eye(4)
    camera_to_world[0, 3] = center_x
    return intrinsics, camera_to_world


def test_thin_splat_passes_shape_and_projection_gates() -> None:
    x = np.linspace(-0.2, 0.2, 600)
    positions = np.column_stack((x, 0.004 * np.sin(20.0 * x), np.full_like(x, 1.0)))
    masks = {}
    intrinsics = {}
    extrinsics = {}
    for name, center in (("left", -0.03), ("middle", 0.0), ("right", 0.03)):
        masks[name] = np.ones((128, 128), dtype=bool)
        intrinsics[name], extrinsics[name] = _camera(center)
    config = ThinRopeSplatProbeConfig(
        minimum_camera_count=2,
        minimum_gaussian_count=256,
    )

    diagnostics = gaussian_splat_geometry_diagnostics(
        positions,
        opacity=np.ones(len(positions)),
        masks_by_camera=masks,
        intrinsics_by_camera=intrinsics,
        camera_to_world_by_camera=extrinsics,
        config=config,
    )

    assert diagnostics["probe_passed"] is True
    assert diagnostics["pca_q01_to_q99_span_m"][0] == pytest.approx(0.392, rel=0.02)
    assert diagnostics["mask_containment"]["minimum"] == 1.0


def test_broad_splat_fails_minor_span_gate() -> None:
    rng = np.random.default_rng(4)
    positions = rng.uniform((-0.2, -0.2, 0.8), (0.2, 0.2, 1.2), size=(800, 3))
    intrinsics, extrinsics = _camera(0.0)
    diagnostics = gaussian_splat_geometry_diagnostics(
        positions,
        opacity=None,
        masks_by_camera={
            "a": np.ones((128, 128), dtype=bool),
            "b": np.ones((128, 128), dtype=bool),
        },
        intrinsics_by_camera={"a": intrinsics, "b": intrinsics},
        camera_to_world_by_camera={"a": extrinsics, "b": extrinsics},
        config=ThinRopeSplatProbeConfig(minimum_camera_count=2),
    )

    assert diagnostics["acceptance_gates"]["minor_spans"] is False
    assert diagnostics["probe_passed"] is False


def test_splat_probe_artifact_is_source_only_and_checksummed() -> None:
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360ThinRopeSplatProbe",
        "split": "source",
        "information_boundary": {"target_frames_read": False},
        "diagnostics": {"probe_passed": True},
    }
    payload["result_sha256"] = splat_probe_artifact_sha256(payload)

    assert validate_splat_probe_artifact(payload)["probe_passed"] is True
    payload["split"] = "target"
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_splat_probe_artifact(payload)
