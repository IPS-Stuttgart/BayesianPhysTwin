"""Fresh-only camera eligibility adapter around the frozen generic builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import deform360_raw_camera_observation as raw_camera
from .deform360_fresh_pairwise_protocol import canonical_sha256


def materialized_calibrated_camera_names(
    processed_episode_dir: str | Path,
) -> tuple[str, ...]:
    """Return calibrated cameras with every source asset required by the builder."""

    processed = Path(processed_episode_dir).resolve()
    intrinsics, extrinsics = raw_camera._load_calibration(processed)
    return tuple(
        sorted(
            camera
            for camera in set(intrinsics) & set(extrinsics)
            if (processed / camera / "undistorted.mp4").is_file()
            and (processed / camera / "mask_refined.h5").is_file()
            and (processed / camera / "rendered_depth.h5").is_file()
        )
    )


def build_fresh_raw_camera_measurement_case_with_contract(
    panel_case_dir: str | Path,
    processed_episode_dir: str | Path,
    output_dir: str | Path,
    runtime: raw_camera.AllTrackerPrefixRuntime,
    *,
    protocol_id: str,
    expected_case_names: Sequence[str],
    prediction_seal_validator: Callable[[Mapping[str, Any]], None],
    claim_boundary: str,
    minimum_eligible_camera_count: int,
    config: raw_camera.RawCameraObservationConfig | None = None,
) -> dict[str, Any]:
    """Apply the fresh camera panel without changing frozen generic source bytes."""

    processed = Path(processed_episode_dir).resolve()
    eligible = materialized_calibrated_camera_names(processed)
    cfg = config or runtime.config
    if (
        minimum_eligible_camera_count != cfg.selected_camera_count
        or len(eligible) < minimum_eligible_camera_count
    ):
        raise ValueError("fresh case has too few fully materialized cameras")

    original_loader = raw_camera._load_calibration

    def filtered_loader(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        intrinsics, extrinsics = original_loader(directory)
        return (
            {camera: intrinsics[camera] for camera in eligible},
            {camera: extrinsics[camera] for camera in eligible},
        )

    # The generic builder is hash-pinned by earlier studies. This isolated,
    # single-threaded adapter injects the fresh source panel without editing it.
    raw_camera._load_calibration = filtered_loader
    try:
        manifest = raw_camera.build_raw_camera_measurement_case_with_contract(
            panel_case_dir,
            processed,
            output_dir,
            runtime,
            protocol_id=protocol_id,
            expected_case_names=expected_case_names,
            prediction_seal_validator=prediction_seal_validator,
            claim_boundary=claim_boundary,
            config=cfg,
        )
    finally:
        raw_camera._load_calibration = original_loader

    manifest["plan"]["eligible_cameras"] = list(eligible)
    manifest["plan"]["eligible_camera_policy"] = (
        "lexically sorted intersection of calibrated cameras with materialized "
        "RGB, frame-zero mask, and frame-zero depth assets"
    )
    manifest["result_sha256"] = canonical_sha256(
        manifest, digest_key="result_sha256"
    )
    manifest_path = Path(output_dir).resolve() / raw_camera.MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "build_fresh_raw_camera_measurement_case_with_contract",
    "materialized_calibrated_camera_names",
]
