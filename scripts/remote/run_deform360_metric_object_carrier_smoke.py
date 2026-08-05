#!/usr/bin/env python3
"""Run the frozen source-only metric object-carrier smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_metric_object_carrier import (
    METRIC_OBJECT_CARRIER_POLICY,
    build_metric_object_carrier,
    cover_resize_mask_nearest,
    load_metric_object_carrier_lock,
    reduce_masked_point_map,
)
from bayesian_phystwin.deform360_tactile_metric_gauge import SimilarityTransform


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _load_selector(source: Path):
    package_name = "causal4d_public"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(source.parent)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    name = "causal4d_public.deform360_object_sam2"
    spec = importlib.util.spec_from_file_location(name, source)
    _require(spec is not None and spec.loader is not None, "cannot load SAM2 selector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.DeformableObjectSam2VideoPredictor


def _extract_frames(
    video: Path,
    destination: Path,
    *,
    start: int,
    stop: int,
) -> list[Path]:
    destination.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video))
    paths = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for source_frame in range(start, stop):
            ok, bgr = capture.read()
            observed = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            _require(ok and observed == source_frame, f"cannot decode frame {source_frame}: {video}")
            path = destination / f"{source_frame - start:06d}.jpg"
            _require(
                cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]),
                f"cannot stage SAM2 frame: {path}",
            )
            paths.append(path)
    finally:
        capture.release()
    _require(len(paths) == stop - start, "staged SAM2 frame count changed")
    return paths


def _rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    _require(bgr is not None, f"cannot read RGB frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _transform(record: dict[str, object]) -> SimilarityTransform:
    value = record["similarity_transform"]
    assert isinstance(value, dict)
    return SimilarityTransform(
        scale=float(value["scale"]),
        rotation=np.asarray(value["rotation"], dtype=np.float64),
        translation=np.asarray(value["translation_m"], dtype=np.float64),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--metric-gauge-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repository.resolve()
    lock_path = args.lock.resolve()
    lock = load_metric_object_carrier_lock(lock_path)
    _require(_git_head(repository) == lock["implementation"]["revision"], "runtime revision changed")
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status, "runtime checkout is dirty")
    module_source = repository / "src/bayesian_phystwin/deform360_metric_object_carrier.py"
    runner_source = repository / "scripts/remote/run_deform360_metric_object_carrier_smoke.py"
    _require(_sha256(module_source) == lock["implementation"]["module_source_sha256"], "carrier module changed")
    _require(_sha256(runner_source) == lock["implementation"]["runner_source_sha256"], "carrier runner changed")
    metric_result_path = args.metric_gauge_result.resolve()
    _require(_sha256(metric_result_path) == lock["parents"]["metric_gauge_result"]["sha256"], "metric result changed")
    metric_result = _json(metric_result_path)
    _require(metric_result.get("artifact_id") == lock["parents"]["metric_gauge_result"]["artifact_id"], "metric result identity changed")
    _require(metric_result.get("status") == "admitted", "metric gauge is not admitted")
    gate = metric_result.get("gate")
    _require(isinstance(gate, dict) and gate.get("metric_gauge_authorized") is True, "metric gauge unauthorized")

    cameras = tuple(str(camera) for camera in lock["cameras"])
    reference_camera = str(lock["reference_camera"])
    providers = {str(row["camera"]): row for row in lock["providers"]}
    for camera in cameras:
        provider = providers[camera]
        for field, sha_field in (
            ("video_path", "video_sha256"),
            ("prediction_manifest_path", "prediction_manifest_sha256"),
            ("window_path", "window_sha256"),
        ):
            path = Path(str(provider[field]))
            _require(path.is_file() and _sha256(path) == provider[sha_field], f"provider input changed: {camera}/{field}")
    sam2 = lock["sam2"]
    selector_source = Path(str(sam2["selector_source_path"]))
    sam2_repository = Path(str(sam2["repository_path"]))
    checkpoint = Path(str(sam2["checkpoint_path"]))
    _require(_sha256(selector_source) == sam2["selector_source_sha256"], "SAM2 selector changed")
    _require(_git_head(sam2_repository) == sam2["repository_revision"], "SAM2 revision changed")
    _require(_sha256(checkpoint) == sam2["checkpoint_sha256"], "SAM2 checkpoint changed")

    output = args.output.resolve()
    _require(not output.exists(), "carrier output already exists")
    scratch = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    _require(not scratch.exists(), "carrier scratch already exists")
    scratch.mkdir(parents=True)
    try:
        staged: dict[str, list[Path]] = {}
        for camera in cameras:
            staged[camera] = _extract_frames(
                Path(str(providers[camera]["video_path"])),
                scratch / "sam2-frames" / camera,
                start=int(METRIC_OBJECT_CARRIER_POLICY["source_frame_start"]),
                stop=int(METRIC_OBJECT_CARRIER_POLICY["source_frame_stop_exclusive"]),
            )
        selector_class = _load_selector(selector_source)
        selector = selector_class(sam2_repository, checkpoint, device=args.device)
        masks_by_camera: dict[str, np.ndarray] = {}
        selection_diagnostics: dict[str, object] = {}
        reference_rgb = _rgb(staged[reference_camera][0])
        try:
            reference_mask, reference_diagnostic = selector.select_initial_mask(
                scratch / "sam2-frames" / reference_camera
            )
            initial_masks = {reference_camera: reference_mask}
            selection_diagnostics[reference_camera] = reference_diagnostic
            for camera in cameras:
                if camera == reference_camera:
                    continue
                ranked, diagnostic = selector.initial_mask_candidates_with_reference(
                    scratch / "sam2-frames" / camera,
                    reference_rgb,
                    reference_mask,
                    reference_camera=reference_camera,
                    maximum_candidates=1,
                )
                initial_masks[camera] = np.asarray(ranked[0]["mask"], dtype=bool)
                selection_diagnostics[camera] = {
                    **diagnostic,
                    "selected": ranked[0]["diagnostic"],
                }
            for camera in cameras:
                propagated = list(
                    selector.segment_from_initial_mask(
                        scratch / "sam2-frames" / camera,
                        initial_masks[camera],
                        initialization=selection_diagnostics[camera],
                    )
                )
                _require([frame for frame, _ in propagated] == list(range(42)), f"SAM2 frame sequence changed: {camera}")
                masks_by_camera[camera] = np.stack([mask for _, mask in propagated])
        finally:
            selector.close()

        camera_results = {str(row["camera"]): row for row in metric_result["camera_results"]}
        assignment_count = int(METRIC_OBJECT_CARRIER_POLICY["required_assignment_count"])
        candidates_by_assignment: list[dict[str, object]] = [dict() for _ in range(assignment_count)]
        candidate_counts = np.zeros((assignment_count, len(cameras)), dtype=np.int64)
        gauge_covariance = np.zeros((assignment_count, len(cameras), 3, 3), dtype=np.float64)
        final_masks = []
        for camera_index, camera in enumerate(cameras):
            target_mask = cover_resize_mask_nearest(
                masks_by_camera[camera][-1],
                target_shape=tuple(METRIC_OBJECT_CARRIER_POLICY["target_shape"]),
            )
            final_masks.append(target_mask)
            with np.load(Path(str(providers[camera]["window_path"])), allow_pickle=False) as archive:
                frame_ids = np.asarray(archive["frame_indices"], dtype=np.int64)
                matches = np.flatnonzero(frame_ids == METRIC_OBJECT_CARRIER_POLICY["carrier_source_frame"])
                _require(len(matches) == 1, f"carrier frame missing: {camera}")
                row = int(matches[0])
                point_map = np.asarray(archive["point_map"][row], dtype=np.float64)
                valid_mask = np.asarray(archive["valid_mask"][row], dtype=bool)
                deform_mask = np.asarray(archive["deform_mask"][row], dtype=bool)
            hypotheses = camera_results[camera]["assignment_hypotheses"]
            _require(len(hypotheses) == assignment_count, "assignment count changed")
            for assignment_index, hypothesis in enumerate(hypotheses):
                covariance = np.asarray(hypothesis["covariance_m2"], dtype=np.float64)
                gauge_covariance[assignment_index, camera_index] = covariance
                candidates = reduce_masked_point_map(
                    point_map,
                    valid_mask,
                    target_mask,
                    deform_mask,
                    transform=_transform(hypothesis),
                    gauge_covariance_m2=covariance,
                    block_size_px=int(METRIC_OBJECT_CARRIER_POLICY["block_size_px"]),
                    minimum_mask_pixels=int(METRIC_OBJECT_CARRIER_POLICY["minimum_mask_pixels_per_block"]),
                    minimum_valid_fraction=float(METRIC_OBJECT_CARRIER_POLICY["minimum_valid_fraction_per_block"]),
                    full_reliability_deform_fraction=float(METRIC_OBJECT_CARRIER_POLICY["minimum_deform_fraction_for_full_reliability"]),
                    covariance_floor_m=float(METRIC_OBJECT_CARRIER_POLICY["local_covariance_floor_m"]),
                )
                candidates_by_assignment[assignment_index][camera] = candidates
                candidate_counts[assignment_index, camera_index] = len(candidates.points_world_m)

        reason_codes: list[str] = []
        carrier = None
        try:
            carrier = build_metric_object_carrier(
                candidates_by_assignment,
                camera_order=cameras,
                reference_camera=reference_camera,
                maximum_distance_m=float(METRIC_OBJECT_CARRIER_POLICY["cross_view_maximum_distance_m"]),
                node_count=int(METRIC_OBJECT_CARRIER_POLICY["carrier_node_count"]),
            )
        except ValueError as error:
            reason_codes.append(f"carrier-construction:{error}")
        pairwise_p90 = []
        if carrier is not None:
            pairwise_p90 = [
                float(np.quantile(carrier.pairwise_distance_m[index], 0.9))
                for index in range(assignment_count)
            ]
            if any(
                value > float(METRIC_OBJECT_CARRIER_POLICY["maximum_pairwise_percentile_90_m"])
                for value in pairwise_p90
            ):
                reason_codes.append("cross-view-p90-too-large")
        admitted = carrier is not None and not reason_codes
        mask_array = np.stack([masks_by_camera[camera] for camera in cameras])
        flat_masks = mask_array.reshape(mask_array.shape[0], mask_array.shape[1], -1)
        arrays: dict[str, np.ndarray] = {
            "source_frame_ids": np.arange(108, 150, dtype=np.int64),
            "mask_shape": np.asarray(mask_array.shape[2:], dtype=np.int64),
            "mask_packed_little": np.packbits(flat_masks, axis=2, bitorder="little"),
            "final_target_masks": np.asarray(final_masks, dtype=bool),
            "candidate_counts": candidate_counts,
            "gauge_covariance_m2": gauge_covariance,
        }
        if carrier is not None:
            arrays.update(
                {
                    "points_world_m": carrier.points_world_m,
                    "covariance_m2": carrier.covariance_m2,
                    "assignment_mixture_covariance_m2": carrier.assignment_mixture_covariance_m2,
                    "marginal_covariance_m2": carrier.marginal_covariance_m2,
                    "prior_reliability": carrier.prior_reliability,
                    "reference_pixel_xy": carrier.reference_pixel_xy,
                    "contributor_indices": carrier.contributor_indices,
                    "pairwise_distance_m": carrier.pairwise_distance_m,
                }
            )
        arrays_path = scratch / "metric_object_carrier.npz"
        _deterministic_npz(arrays_path, arrays)
        descriptor: dict[str, object] = {
            "schema": "bayesian-phystwin.deform360-metric-object-carrier-result",
            "schema_version": 1,
            "status": "admitted" if admitted else "rejected-exact-fallback",
            "lock_id": lock["artifact_id"],
            "implementation_revision": lock["implementation"]["revision"],
            "source_case": lock["source_case"],
            "cameras": list(cameras),
            "reference_camera": reference_camera,
            "gate": {
                "object_carrier_authorized": admitted,
                "contact_anchor_authorized": False,
                "reason_codes": reason_codes,
                "candidate_counts_by_assignment_camera": candidate_counts.tolist(),
                "pairwise_percentile_90_m_by_assignment": pairwise_p90,
                "carrier_node_count": 0 if carrier is None else int(carrier.points_world_m.shape[1]),
            },
            "mask_selection": selection_diagnostics,
            "mask_propagation": selector.diagnostics,
            "outputs": {
                "arrays": "metric_object_carrier.npz",
                "arrays_sha256": _sha256(arrays_path),
            },
            "information_boundary": lock["information_boundary"],
            "claim_boundary": lock["claim_boundary"],
        }
        result = {"artifact_id": content_id(descriptor), **descriptor}
        (scratch / "metric_object_carrier_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(scratch / "sam2-frames")
        os.replace(scratch, output)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "artifact_id": result["artifact_id"],
                "reason_codes": result["gate"]["reason_codes"],
                "arrays_sha256": result["outputs"]["arrays_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
