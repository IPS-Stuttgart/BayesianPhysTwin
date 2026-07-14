#!/usr/bin/env python3
"""Resume source fitting or prefix-only calibration preprocessing."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_phystwin_feasibility import (
    WarpRopeFeasibilityConfig,
)
from causal4d_public.deform360_replication import (
    load_deform360_replication_protocol,
)
from causal4d_public.deform360_replication_contact import (
    ReplicationOpeningContactModel,
    fit_replication_opening_contact_model,
    load_replication_contact_episode,
    prefix_window_from_visual_contact,
    visual_contact_schedule,
)
from causal4d_public.deform360_replication_fit import (
    pool_source_warp_candidate_grids,
    score_source_warp_candidate_grid,
    validate_source_warp_candidate_grid,
    write_replication_fit_artifact,
)
from causal4d_public.deform360_replication_geometry import (
    ReplicationGeometryConfig,
    build_replication_hull_archive,
    build_replication_mask_archive,
    load_replication_hull_archive,
    load_replication_mask_archive,
    replication_geometry_frame_indices,
)
from causal4d_public.deform360_replication_source_qa import (
    validate_source_qa_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("contact", "geometry", "grid", "pool", "all")
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-qa", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, action="append")
    parser.add_argument(
        "--split", choices=("source", "calibration"), default="source"
    )
    parser.add_argument("--sam2-repo", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--official-phystwin-repo", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cohort_record(protocol: dict[str, Any], object_id: str) -> dict[str, Any]:
    records = [
        record
        for record in protocol["config"]["cohort"]
        if record["object_id"] == object_id
    ]
    if len(records) != 1:
        raise ValueError(f"object is not unique in the locked cohort: {object_id}")
    return records[0]


def _qa_record(source_qa: dict[str, Any], object_id: str) -> dict[str, Any]:
    records = [
        record for record in source_qa["objects"] if record["object_id"] == object_id
    ]
    if len(records) != 1:
        raise ValueError(f"object is not unique in source QA: {object_id}")
    return records[0]


def _episode_metadata(cohort: dict[str, Any], episode_id: int) -> dict[str, Any]:
    return cohort["episodes"][str(episode_id)]


def _load_contact_episode(
    root: Path, cohort: dict[str, Any], object_id: str, episode_id: int
):
    metadata = _episode_metadata(cohort, episode_id)
    return load_replication_contact_episode(
        root / "aligned" / object_id / f"episode_{episode_id:04d}",
        episode_id=f"{object_id}/episode_{episode_id:04d}",
        bimanual=metadata["bimanual"] == "yes",
        nonprehensile=metadata["nonprehensile"] == "yes",
    )


def _fit_contact(
    root: Path, cohort: dict[str, Any], object_id: str
) -> ReplicationOpeningContactModel:
    source = [
        _load_contact_episode(root, cohort, object_id, int(episode_id))
        for episode_id in cohort["source_episode_ids"]
    ]
    calibration = [
        _load_contact_episode(root, cohort, object_id, int(episode_id))
        for episode_id in cohort["calibration_episode_ids"]
    ]
    model = fit_replication_opening_contact_model(source, calibration)
    _write_json(root / "observations" / object_id / "contact_model.json", asdict(model))
    return model


def _load_contact_model(root: Path, object_id: str) -> ReplicationOpeningContactModel:
    path = root / "observations" / object_id / "contact_model.json"
    return ReplicationOpeningContactModel(**json.loads(path.read_text(encoding="utf-8")))


def _decode_rgb(video: Path, frame_index: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - remote integration
        raise RuntimeError("OpenCV is required for source geometry") from error
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"cannot decode source reference frame: {video}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _reference_appearance(
    root: Path,
    object_id: str,
    episode_id: int,
    reference_camera: str,
    predictor: DeformableObjectSam2VideoPredictor,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    episode_dir = root / "aligned" / object_id / f"episode_{episode_id:04d}"
    video = episode_dir / reference_camera / "undistorted.mp4"
    mask_json = (
        root
        / "observations"
        / object_id
        / f"episode_{episode_id:04d}"
        / "sampled_masks.json"
    )
    if mask_json.is_file():
        cameras, frames, masks = load_replication_mask_archive(
            json.loads(mask_json.read_text(encoding="utf-8"))
        )
        camera_index = cameras.index(reference_camera)
        fallback = {
            camera: np.asarray(masks[index, 0], dtype=bool)
            for index, camera in enumerate(cameras)
        }
        return (
            _decode_rgb(video, int(frames[0])),
            masks[camera_index, 0],
            fallback,
        )
    mask, _ = predictor.select_initial_mask(video)
    return _decode_rgb(video, 0), mask, {reference_camera: mask}


def _build_source_geometry(
    root: Path,
    cohort: dict[str, Any],
    qa: dict[str, Any],
    model: ReplicationOpeningContactModel,
    object_id: str,
    episode_ids: list[int],
    predictor: DeformableObjectSam2VideoPredictor,
    *,
    prefix_only: bool = False,
) -> None:
    config = ReplicationGeometryConfig()
    reference_camera = "brics-odroid-001_cam0"
    reference_episode = int(qa["source_episode_index"])
    reference_rgb, reference_mask, fallback_masks = _reference_appearance(
        root, object_id, reference_episode, reference_camera, predictor
    )
    cameras = qa["selected_cameras"]
    stratum = cohort["stratum"]
    for episode_id in episode_ids:
        started = time.time()
        episode = _load_contact_episode(root, cohort, object_id, episode_id)
        visual_contact = visual_contact_schedule(episode, model)
        prefix_start, _ = prefix_window_from_visual_contact(
            visual_contact, prefix_frame_count=config.prefix_frame_count
        )
        frames = replication_geometry_frame_indices(
            len(episode.openings_m),
            prefix_start,
            config,
            prefix_only=prefix_only,
        )
        episode_dir = root / "aligned" / object_id / f"episode_{episode_id:04d}"
        output = root / "observations" / object_id / f"episode_{episode_id:04d}"
        output.mkdir(parents=True, exist_ok=True)
        hull_json = output / "sampled_hulls.json"
        if hull_json.is_file():
            existing_hull = json.loads(hull_json.read_text())
            try:
                existing_frames, _ = load_replication_hull_archive(existing_hull)
            except ValueError:
                pass
            else:
                if (
                    existing_hull.get("config") == asdict(config)
                    and existing_frames.astype(int).tolist() == list(frames)
                ):
                    print(
                        f"{object_id} episode {episode_id}: hull already valid",
                        flush=True,
                    )
                    continue
        mask_json = output / "sampled_masks.json"
        mask_payload = None
        if mask_json.is_file():
            candidate_mask_payload = json.loads(mask_json.read_text(encoding="utf-8"))
            _, existing_mask_frames, _ = load_replication_mask_archive(
                candidate_mask_payload
            )
            if existing_mask_frames.astype(int).tolist() == list(frames):
                mask_payload = candidate_mask_payload
        if mask_payload is None:
            mask_payload = build_replication_mask_archive(
                episode_dir,
                cameras,
                frames,
                predictor,
                reference_rgb,
                reference_mask,
                output / "sampled_masks.npz",
                reference_camera=reference_camera,
                fallback_initial_masks_by_camera=fallback_masks,
            )
            _write_json(mask_json, mask_payload)
            if episode_id == reference_episode:
                archived_cameras, _, archived_masks = load_replication_mask_archive(
                    mask_payload
                )
                fallback_masks = {
                    camera: np.asarray(archived_masks[index, 0], dtype=bool)
                    for index, camera in enumerate(archived_cameras)
                }
        hull_payload = build_replication_hull_archive(
            episode_dir,
            mask_payload,
            stratum,
            output / "sampled_hulls.npz",
            config=config,
        )
        _write_json(hull_json, hull_payload)
        print(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "hull_result_sha256": hull_payload["result_sha256"],
                    "object_id": object_id,
                    "seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _score_source_grids(
    root: Path,
    cohort: dict[str, Any],
    model: ReplicationOpeningContactModel,
    object_id: str,
    episode_ids: list[int],
    official_repo: Path,
    device: str,
) -> None:
    config = WarpRopeFeasibilityConfig()
    for episode_id in episode_ids:
        output = root / "fits" / object_id / f"source_episode_{episode_id:04d}_grid.json"
        if output.is_file():
            existing = json.loads(output.read_text())
            try:
                validate_source_warp_candidate_grid(existing)
            except ValueError:
                pass
            else:
                hull_json = (
                    root
                    / "observations"
                    / object_id
                    / f"episode_{episode_id:04d}"
                    / "sampled_hulls.json"
                )
                current_hull = json.loads(hull_json.read_text(encoding="utf-8"))
                if (
                    existing["reference_geometry_result_sha256"]
                    == current_hull["result_sha256"]
                ):
                    print(
                        f"{object_id} episode {episode_id}: source grid already valid",
                        flush=True,
                    )
                    continue
        episode = _load_contact_episode(root, cohort, object_id, episode_id)
        episode_dir = root / "aligned" / object_id / f"episode_{episode_id:04d}"
        hull_json = (
            root
            / "observations"
            / object_id
            / f"episode_{episode_id:04d}"
            / "sampled_hulls.json"
        )
        hull_payload = json.loads(hull_json.read_text(encoding="utf-8"))
        frames, hulls = load_replication_hull_archive(hull_payload)
        total_frame_count = len(hulls)
        available = np.asarray([len(hull) > 0 for hull in hulls], dtype=bool)
        if not available[0]:
            raise ValueError("prefix endpoint hull is unavailable")
        frames = frames[available]
        hulls = tuple(hull for hull, keep in zip(hulls, available, strict=True) if keep)
        started = time.time()
        payload = score_source_warp_candidate_grid(
            episode_dir,
            episode,
            model,
            cohort["stratum"],
            frames,
            hulls,
            hull_payload["result_sha256"],
            total_frame_count,
            official_repo,
            device=device,
            config=config,
        )
        write_replication_fit_artifact(output, payload)
        print(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "object_id": object_id,
                    "result_sha256": payload["result_sha256"],
                    "seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _pool_source_grids(root: Path, cohort: dict[str, Any], object_id: str) -> None:
    grids = [
        json.loads(
            (
                root
                / "fits"
                / object_id
                / f"source_episode_{int(episode_id):04d}_grid.json"
            ).read_text(encoding="utf-8")
        )
        for episode_id in cohort["source_episode_ids"]
    ]
    payload = pool_source_warp_candidate_grids(grids)
    output = root / "fits" / object_id / "pooled_source_fit.json"
    write_replication_fit_artifact(output, payload)
    print(json.dumps(payload["source_backend_competence"], sort_keys=True), flush=True)


def main() -> None:
    args = _parse_args()
    protocol = load_deform360_replication_protocol(args.protocol)
    source_qa = json.loads(args.source_qa.read_text(encoding="utf-8"))
    validate_source_qa_artifact(source_qa)
    cohort = _cohort_record(protocol, args.object_id)
    qa = _qa_record(source_qa, args.object_id)
    source_ids = list(map(int, cohort["source_episode_ids"]))
    calibration_ids = list(map(int, cohort["calibration_episode_ids"]))
    allowed_ids = source_ids if args.split == "source" else calibration_ids
    episode_ids = args.episode_id or allowed_ids
    if not set(episode_ids).issubset(allowed_ids):
        raise ValueError(f"runner was given an episode outside the {args.split} split")
    if args.split != "source" and args.phase in {"grid", "pool", "all"}:
        raise ValueError("calibration split permits prefix geometry only")
    root = args.data_root.resolve()

    phases = (
        ("contact", "geometry", "grid", "pool")
        if args.phase == "all"
        else (args.phase,)
    )
    if "contact" in phases:
        model = _fit_contact(root, cohort, args.object_id)
    else:
        model = _load_contact_model(root, args.object_id)
    if "geometry" in phases:
        if args.sam2_repo is None or args.sam2_checkpoint is None:
            raise ValueError("geometry requires --sam2-repo and --sam2-checkpoint")
        predictor = DeformableObjectSam2VideoPredictor(
            args.sam2_repo, args.sam2_checkpoint, device=args.device
        )
        _build_source_geometry(
            root,
            cohort,
            qa,
            model,
            args.object_id,
            episode_ids,
            predictor,
            prefix_only=args.split == "calibration",
        )
    if "grid" in phases:
        if args.official_phystwin_repo is None:
            raise ValueError("grid requires --official-phystwin-repo")
        _score_source_grids(
            root,
            cohort,
            model,
            args.object_id,
            episode_ids,
            args.official_phystwin_repo,
            args.device,
        )
    if "pool" in phases:
        _pool_source_grids(root, cohort, args.object_id)


if __name__ == "__main__":
    main()
