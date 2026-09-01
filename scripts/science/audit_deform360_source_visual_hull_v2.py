#!/usr/bin/env python3
"""Run the Deform360 source visual-hull gate at episode-frame quantiles.

This v2 wrapper deliberately removes the invalid raw-positive-tactile frame
selector from v1. It derives the common episode length from source object-mask
carrier headers, chooses preregistered episode-frame quantiles, and then reuses
the already tested v1 camera split, voxel carving, held-out reprojection,
calibration perturbation, and camera-block bootstrap implementation.

No tactile payload, RGB pixel, Splatfacto model, target, or paper claim is
opened by this wrapper.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BASE_IMPLEMENTATION = Path(__file__).with_name(
    "audit_deform360_source_visual_hull_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_deform360_source_visual_hull_v1_for_v2",
    BASE_IMPLEMENTATION,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load base implementation {BASE_IMPLEMENTATION}")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_frame_indices(
    frame_population: Sequence[Any] | np.ndarray,
    quantiles: Iterable[float],
) -> list[int]:
    """Select unique nearest episode-frame quantiles without image labels."""
    frame_count = len(frame_population)
    if frame_count < 3:
        raise ValueError("episode timeline has fewer than three frames")
    values = [float(value) for value in quantiles]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("episode-frame quantiles must lie in [0, 1]")
    selected = [
        int(np.rint(value * (frame_count - 1))) for value in values
    ]
    if len(set(selected)) != len(selected):
        raise ValueError("episode-frame quantiles map to duplicate frames")
    return selected


def _mask_timeline_population(
    source_episode_root: Path,
    _unused_sensor_names: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Read only HDF5 mask-carrier headers and return a dummy frame population."""
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ModuleNotFoundError(
            "visual-hull v2 requires the optional h5py dependency"
        ) from exc

    root = Path(source_episode_root).resolve(strict=True)
    carrier_paths = sorted(root.glob("*/mask_refined.h5"))
    if not carrier_paths:
        raise FileNotFoundError("source episode has no object-mask carrier")
    records = []
    frame_counts = set()
    for path in carrier_paths:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"{path} has no HDF5 dataset named 'data'")
            data = handle["data"]
            if data.ndim != 3:
                raise ValueError(
                    f"{path} has shape {data.shape}, expected (T,H,W)"
                )
            frame_count, height, width = map(int, data.shape)
            dtype = str(data.dtype)
        frame_counts.add(frame_count)
        records.append(
            {
                "camera": path.parent.name,
                "carrier": str(path),
                "carrier_shape": [frame_count, height, width],
                "carrier_dtype": dtype,
                "payload_frames_opened_for_selection": 0,
            }
        )
    if len(frame_counts) != 1:
        raise ValueError(
            f"object-mask carrier timelines disagree: {sorted(frame_counts)}"
        )
    frame_count = frame_counts.pop()
    return np.zeros(frame_count, dtype=np.uint8), records


def _parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-episode-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments, _unknown = parser.parse_known_args()
    return arguments


def _write_v2_result(
    *,
    output_dir: Path,
    protocol: dict[str, Any],
    timeline_records: list[dict[str, Any]],
) -> None:
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    base_implementation_sha256 = result["implementation_sha256"]
    selected_frames = list(result["frame_selection"]["selected_frames"])
    episode_frame_count = int(timeline_records[0]["carrier_shape"][0])

    result["schema"] = "bayesian-phystwin/deform360-source-visual-hull-v2"
    result["base_implementation_path"] = str(BASE_IMPLEMENTATION)
    result["base_implementation_sha256"] = base_implementation_sha256
    result["implementation_sha256"] = _sha256(Path(__file__).resolve())
    result["frame_selection"] = {
        "method": "episode-frame-quantiles",
        "episode_frame_quantiles": list(
            protocol["frame_selection"]["episode_frame_quantiles"]
        ),
        "selected_frames": selected_frames,
        "episode_frame_count": episode_frame_count,
        "frame_count_source": protocol["frame_selection"][
            "frame_count_source"
        ],
        "uses_tactile_threshold": False,
        "uses_image_content": False,
        "mask_carrier_header_records": timeline_records,
    }
    boundary = result["information_boundary"]
    boundary["source_mask_carrier_headers_opened_for_frame_selection"] = True
    boundary["source_tactile_payloads_opened"] = False
    boundary["source_camera_pixels_opened"] = False
    boundary["target_directory_contents_listed"] = False
    boundary["target_numeric_payload_opened"] = False
    boundary["target_scoring_performed"] = False
    boundary["fresh_confirmation_authorized"] = False
    boundary["paper_claim_authorized"] = False

    result.pop("result_sha256", None)
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    aggregate = result["aggregate"]
    bootstrap = aggregate["heldout_camera_block_bootstrap"]
    (output_dir / "report.md").write_text(
        "# Deform360 source visual-hull gate v2\n\n"
        f"Decision: `{result['decision']}`\n\n"
        "Frame selection: episode-frame quantiles "
        f"`{protocol['frame_selection']['episode_frame_quantiles']}` over "
        f"`{episode_frame_count}` mask-carrier frames\n\n"
        f"Selected frames: `{selected_frames}`\n\n"
        f"Cameras: `{len(result['camera_split']['training_cameras'])}` training / "
        f"`{len(result['camera_split']['heldout_cameras'])}` held out\n\n"
        f"Median training IoU: `{aggregate['training_iou']['median']:.6f}`\n\n"
        f"Median held-out IoU: `{aggregate['heldout_iou']['median']:.6f}`\n\n"
        f"Held-out IoU p25: `{aggregate['heldout_iou']['p25']:.6f}`\n\n"
        f"Median held-out boundary F1: "
        f"`{aggregate['heldout_boundary_f1']['median']:.6f}`\n\n"
        f"Correct-minus-perturbed-extrinsic median IoU: "
        f"`{aggregate['correct_vs_perturbed_extrinsic_median_iou_gain']:.6f}`\n\n"
        f"Held-out camera-block mean-IoU 95% interval: "
        f"`[{bootstrap['ci95'][0]:.6f}, {bootstrap['ci95'][1]:.6f}]`\n\n"
        "No tactile threshold or RGB image content was used to choose frames. "
        "This remains a single-source-episode geometry diagnostic, not an "
        "independent-object result or paper claim.\n",
        encoding="utf-8",
    )


def main() -> int:
    wrapper_args = _parse_wrapper_args()
    protocol_path = wrapper_args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != (
        "bayesian-phystwin/deform360-source-visual-hull-protocol-v2"
    ):
        raise ValueError("unexpected visual-hull v2 protocol schema")
    frame_selection = protocol.get("frame_selection", {})
    if frame_selection.get("method") != "episode-frame-quantiles":
        raise ValueError("visual-hull v2 requires episode-frame-quantiles")
    if frame_selection.get("uses_tactile_threshold") is not False:
        raise ValueError("visual-hull v2 must not use a tactile threshold")
    if frame_selection.get("uses_image_content") is not False:
        raise ValueError("visual-hull v2 must not use image content for selection")

    timeline_population, timeline_records = _mask_timeline_population(
        wrapper_args.source_episode_root,
        [],
    )
    expected_frames = _episode_frame_indices(
        timeline_population,
        frame_selection["episode_frame_quantiles"],
    )

    original_json_loads = BASE.json.loads
    original_tactile_total = BASE._tactile_total
    original_selector = BASE._select_contact_frames

    def adapted_json_loads(text: str, *args: Any, **kwargs: Any) -> Any:
        value = original_json_loads(text, *args, **kwargs)
        if isinstance(value, dict) and value.get("schema") == protocol["schema"]:
            value = copy.deepcopy(value)
            selection = value["frame_selection"]
            selection["positive_signal_quantiles"] = list(
                selection["episode_frame_quantiles"]
            )
            selection["tactile_sensors"] = []
        return value

    def fixed_population(
        _root: Path,
        _sensors: Sequence[str],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        return timeline_population.copy(), copy.deepcopy(timeline_records)

    try:
        BASE.json.loads = adapted_json_loads
        BASE._tactile_total = fixed_population
        BASE._select_contact_frames = _episode_frame_indices
        return_code = int(BASE.main())
    finally:
        BASE.json.loads = original_json_loads
        BASE._tactile_total = original_tactile_total
        BASE._select_contact_frames = original_selector

    if return_code != 0:
        return return_code
    output_dir = wrapper_args.output_dir.resolve()
    _write_v2_result(
        output_dir=output_dir,
        protocol=protocol,
        timeline_records=timeline_records,
    )
    observed = json.loads(
        (output_dir / "result.json").read_text(encoding="utf-8")
    )["frame_selection"]["selected_frames"]
    if observed != expected_frames:
        raise RuntimeError(
            f"selected frames changed: expected {expected_frames}, got {observed}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
