"""Fresh held-v8 adapter for the pinned official Deform360 reconstruction.

The released numerical pipeline is reused as source code, never as a v7
execution artifact.  This adapter opens the aligned episode only after the
caller has consumed a case-specific v8 target-reconstruction capability.  It
returns the official point-axis order directly: no frame-zero assignment,
transport, interpolation, or score is performed here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Literal, Mapping

import numpy as np

from . import deform360_held_outcome_reconstruction as numerical
from . import deform360_held_v8_protocol as protocol
from .deform360_dataset_containment import validate_aligned_episode


ADAPTER_ID = "deform360-held-v8-fresh-official-reconstruction-adapter-v1"
_FRAME_ZERO_ARRAYS = frozenset(
    {
        "frame_indices",
        "camera_names",
        "rgb_frame0",
        "mask_frame0",
        "depth_frame0_m",
        "depth_valid_frame0",
        "intrinsics",
        "camera_to_world",
        "projection_world_to_pixel",
        "object_points_world_m",
        "object_colors_rgb",
        "object_color_support_count",
        "visual_hull_points_world_m",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: str | Path) -> str:
    source = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(source)
    _require(stat.S_ISREG(before.st_mode), f"not a regular file: {source}")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
            f"file changed while opening: {source}",
        )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        _require(
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"file changed while hashing: {source}",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _bound_file(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.fspath(path)))
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "size_bytes": os.lstat(source).st_size,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.fspath(path)))
    value = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON is not an object: {source}")
    return value


def _load_frame_zero_arrays(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = _load_json(manifest_path)
    bundle_record = manifest.get("bundle")
    _require(isinstance(bundle_record, Mapping), "frame-zero bundle binding is absent")
    bundle_path = Path(str(bundle_record.get("path", "")))
    _require(
        bundle_record == _bound_file(bundle_path),
        "frame-zero bundle bytes changed after v8 validation",
    )
    with np.load(bundle_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == _FRAME_ZERO_ARRAYS,
            "frame-zero bundle array set changed",
        )
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o400, follow_symlinks=False)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _seal_tree(root: Path) -> None:
    entries = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    _require(
        all(not path.is_symlink() for path in entries), "reconstruction tree linked"
    )
    for path in entries:
        observed = os.lstat(path)
        if stat.S_ISREG(observed.st_mode):
            os.chmod(path, 0o400, follow_symlinks=False)
        elif stat.S_ISDIR(observed.st_mode):
            os.chmod(path, 0o500, follow_symlinks=False)
        else:
            raise ValueError(f"reconstruction tree contains a special file: {path}")
    os.chmod(root, 0o500, follow_symlinks=False)


def reconstruct_fresh_official_target(
    *,
    lock_path: str | Path,
    role: Literal["calibration", "confirmation"],
    case_name: str,
    online_prediction_seal_path: str | Path,
    aligned_episode_dir: str | Path,
    output_dir: str | Path,
    cohort_barrier_sha256: str,
    backend: numerical.PinnedOfficialPipelineBackend,
) -> dict[str, Any]:
    """Run the pinned official pipeline after a v8 capability was consumed."""

    _require(
        isinstance(cohort_barrier_sha256, str)
        and len(cohort_barrier_sha256) == 64
        and all(character in "0123456789abcdef" for character in cohort_barrier_sha256),
        "first-barrier digest is invalid",
    )
    lock = protocol.validate_protocol_lock(lock_path)
    _require(lock.get("stage") == role, "reconstruction role and lock stage differ")
    online_path = Path(os.path.abspath(os.fspath(online_prediction_seal_path)))
    online = protocol.validate_online_prediction_seal(
        online_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    authorization = _load_json(str(online["prefix_authorization"]["path"]))
    physical_path = Path(str(authorization["physical_prior_seal"]["path"]))
    physical = protocol.validate_physical_prior_seal(
        physical_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    frame_zero_path = Path(str(physical["frame_zero_manifest"]["path"]))
    protocol.validate_frame_zero_bundle_manifest(
        frame_zero_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    frame_zero_manifest, arrays = _load_frame_zero_arrays(frame_zero_path)
    cameras = tuple(str(value) for value in arrays["camera_names"].tolist())
    selected = frame_zero_manifest.get("action_alignment", {}).get(
        "selected_raw_frame_range_half_open"
    )
    _require(
        isinstance(selected, list)
        and len(selected) == 2
        and all(isinstance(value, int) for value in selected)
        and selected[1] - selected[0] == numerical.TRACKING_CONTEXT_FRAME_COUNT,
        "sealed reconstruction window is not the exact 81-frame context",
    )
    object_id = str(online["object_id"])
    episode_id = int(online["episode_id"])
    aligned = validate_aligned_episode(
        aligned_episode_dir,
        object_id=object_id,
        episode_id=episode_id,
    ).episode_dir
    output = Path(os.path.abspath(os.fspath(output_dir)))
    _require(not os.path.lexists(output), "fresh reconstruction output already exists")

    contract = numerical._copy_contract()
    numerical._validate_contract_semantics(contract)
    request = numerical.ReconstructionRequest(
        case_name=case_name,
        object_id=object_id,
        episode_id=episode_id,
        role=role,
        cohort_barrier_sha256=cohort_barrier_sha256,
        aligned_episode_dir=aligned,
        output_dir=output,
        source_frame_start=int(selected[0]),
        source_frame_stop=int(selected[1]),
        camera_names=cameras,
        frame_zero_arrays=arrays,
        frame_zero_manifest=frame_zero_manifest,
        frame_zero_manifest_path=frame_zero_path,
        online_seal_path=online_path,
        contract=contract,
        immutable_bindings=dict(lock["immutable_bindings"]),
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = numerical._validate_backend_result(
            request,
            backend.build(request),
            expected_resource_lifecycle_policy=(
                numerical.resource_lifecycle_policy_for_backend(backend)
            ),
        )
        audit_path = output / "held-v8-official-reconstruction-audit.json"
        audit = {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV8OfficialReconstructionAudit",
            "protocol_id": protocol.PROTOCOL_ID,
            "adapter_id": ADAPTER_ID,
            "case_name": case_name,
            "role": role,
            "cohort_barrier_sha256": cohort_barrier_sha256,
            "online_prediction_seal": _bound_file(online_path),
            "frame_zero_manifest": _bound_file(frame_zero_path),
            "backend_audit": dict(result.audit),
            "information_boundary": {
                "target_capability_consumed_before_adapter_call": True,
                "fresh_v8_reconstruction": True,
                "v7_execution_artifact_reused": False,
                "source_to_target_assignment_performed": False,
                "identity_transport_performed": False,
                "score_computed": False,
            },
        }
        audit["artifact_sha256"] = protocol.held_artifact_sha256(audit)
        _write_new_json(audit_path, audit)
        audit_record = _bound_file(audit_path)
        _seal_tree(output)
    except BaseException:
        # Partial staging is intentionally retained for diagnosis.  The
        # write-once outer outcome claim prevents it from being resumed.
        raise

    return {
        "object_points": result.object_points,
        "object_visibilities": result.object_visibilities,
        "object_motions_valid": result.object_motions_valid,
        "provenance": {
            "adapter_id": ADAPTER_ID,
            "fresh_v8_reconstruction": True,
            "v7_execution_artifact_reused": False,
            "reconstruction_audit": audit_record,
            "cohort_barrier_sha256": cohort_barrier_sha256,
        },
    }


__all__ = [
    "ADAPTER_ID",
    "reconstruct_fresh_official_target",
]
