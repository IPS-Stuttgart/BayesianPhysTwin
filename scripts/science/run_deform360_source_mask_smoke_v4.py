#!/usr/bin/env python3
"""Run one target-closed Deform360 source mask qualification smoke.

This driver opens only one registered source episode. It writes compact aggregate
and provenance evidence; no mask payload is retained outside the runner scratch
area and no target directory is inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--source-episode-root", required=True, type=Path)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--source-episode", required=True, type=int)
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workflow-run-id", required=True, type=int)
    parser.add_argument("--workflow-run-attempt", required=True, type=int)
    parser.add_argument("--official-deform360-revision", required=True)
    parser.add_argument("--sam3-revision", required=True)
    parser.add_argument("--einops-version", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    raw_root = args.raw_root.resolve(strict=True)
    source_episode = args.source_episode_root.resolve(strict=True)
    smoke_root = args.smoke_root.resolve()
    evidence = args.evidence_root.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    if smoke_root.exists():
        raise FileExistsError(f"smoke root already exists: {smoke_root}")
    smoke_root.mkdir(parents=True)

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-source-mask-smoke-v4",
        "repository": args.repository,
        "revision": args.revision,
        "workflow_run_id": args.workflow_run_id,
        "workflow_run_attempt": args.workflow_run_attempt,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "required_runner_label": "gpuserver4090",
        "official_deform360_revision": args.official_deform360_revision,
        "sam3_revision": args.sam3_revision,
        "einops_version": args.einops_version,
        "source_object": args.source_object,
        "source_episode": args.source_episode,
        "mask": {},
        "errors": [],
        "information_boundary": {
            "source_video_decoded": False,
            "persistent_runtime_or_model_cache_written": True,
            "persistent_source_mask_written": False,
            "requested_processed_tree_modified": False,
            "target_directory_contents_listed": False,
            "target_numeric_payload_opened": False,
            "target_scoring_performed": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
    }

    predictor = None
    try:
        import h5py
        import numpy as np
        import torch

        from deform360.processing.masks import process_masks_episode
        from deform360.processing.sam3_predictor import Sam3MaskPredictor

        metadata_path = raw_root / "raw" / args.source_object / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompt = str(metadata.get("sam_prompt", "")).strip()
        if not prompt:
            raise ValueError("source metadata has no sam_prompt")

        cameras = sorted(
            path.name
            for path in source_episode.iterdir()
            if path.is_dir()
            and (path / "undistorted.mp4").is_file()
            and (path / "aligned_timestamps.txt").is_file()
        )
        if not cameras:
            raise FileNotFoundError("source episode has no aligned cameras")
        camera = cameras[0]

        aligned_object = smoke_root / args.source_object
        smoke_episode = aligned_object / f"episode_{args.source_episode:04d}"
        smoke_camera = smoke_episode / camera
        smoke_camera.mkdir(parents=True, exist_ok=False)
        for name in ("undistorted.mp4", "aligned_timestamps.txt"):
            (smoke_camera / name).symlink_to(source_episode / camera / name)

        result["runtime"] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
        result["mask"].update(
            {
                "camera_selection": "lexicographically-first-aligned-camera",
                "camera": camera,
                "aligned_camera_count": len(cameras),
                "prompt": prompt,
                "prompt_source": "raw-metadata-sam_prompt",
            }
        )

        predictor = Sam3MaskPredictor()
        outputs = process_masks_episode(
            aligned_object,
            args.source_episode,
            prompt=prompt,
            predictor=predictor,
            cameras=[camera],
            first_frame_box_fn=None,
            overwrite=True,
            preview=False,
        )
        result["information_boundary"]["source_video_decoded"] = True
        mask_path = outputs[camera]
        meta_path = mask_path.with_name("mask_refined.meta.json")
        mask_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with h5py.File(mask_path, "r") as store:
            dataset = store["data"]
            indices = np.linspace(
                0,
                dataset.shape[0] - 1,
                min(12, dataset.shape[0]),
                dtype=int,
            )
            sampled_nonzero = [
                int(np.count_nonzero(dataset[int(index)])) for index in indices
            ]
            shape = [int(value) for value in dataset.shape]
            dtype = str(dataset.dtype)

        frames_with_mask = int(mask_meta["outputs"]["frames_with_mask"])
        grounded = bool(mask_meta["parameters"]["grounded"])
        if not grounded or frames_with_mask < 1:
            raise RuntimeError("SAM3 produced no grounded source mask")
        result["mask"].update(
            {
                "status": "success",
                "model_id": mask_meta.get("model", {}).get("id"),
                "grounded": grounded,
                "frames_with_mask": frames_with_mask,
                "shape": shape,
                "dtype": dtype,
                "sampled_frame_indices": indices.astype(int).tolist(),
                "sampled_nonzero_pixels": sampled_nonzero,
                "mask_size_bytes": mask_path.stat().st_size,
                "mask_sha256": _sha256(mask_path),
                "metadata_sha256": _sha256(meta_path),
            }
        )
        result["decision"] = "source-mask-runtime-qualified"
        result["next_action"] = (
            "Materialize source masks under the exact qualified runtime, then run "
            "gripper masks and a bounded source reconstruction/pcd pilot."
        )
    except Exception as error:  # evidence must retain the exact technical negative
        result["mask"]["status"] = "failure"
        result["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(limit=24),
            }
        )
        result["decision"] = "source-mask-runtime-or-model-not-qualified"
        result["next_action"] = (
            "Repair only the recorded runtime, dependency, model-access, or source "
            "prompt failure before any broader processing."
        )
    finally:
        if predictor is not None:
            try:
                predictor.close()
            except Exception:
                pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        shutil.rmtree(smoke_root, ignore_errors=True)

    result["claim_boundary"] = (
        "Source-only one-camera segmentation qualification on public Deform360. "
        "No retained source mask, multi-view geometry, physical-parameter result, "
        "Prob4D qualification, BayesianPhysTwin benefit, cross-action transport, "
        "Causal4D decision value, target result, calibration, safety, or SOTA claim."
    )
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    _write_json(evidence / "result.json", result)
    (evidence / "report.md").write_text(
        "# Deform360 source mask smoke v4\n\n"
        f"Decision: `{result['decision']}`\n\n"
        f"Camera: `{result['mask'].get('camera', 'unresolved')}`\n\n"
        f"Frames with mask: `{result['mask'].get('frames_with_mask', 0)}`\n\n"
        f"Next action: {result['next_action']}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
