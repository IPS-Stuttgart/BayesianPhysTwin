from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remote" / (
    "prepare_deform360_causal_response_direct_depth_v14_source.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("v14_source_preparation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: bytes = b"source") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _source_tree(root: Path, object_id: str) -> None:
    metadata = {
        "object": object_id,
        "sequences": {"0": {"bimanual": "no"}},
    }
    _write(
        root / "metadata.json",
        json.dumps(metadata).encode("utf-8"),
    )
    for name in ("intrinsics.npy", "extrinsics.npy", "dist.npy"):
        _write(root / "calibration_refined" / name)
    for camera in REGISTERED_CAMERA_IDS:
        _write(root / camera / f"{camera}_100.mp4")
        _write(root / camera / f"{camera}_100.txt")
    sensor = "brics-odroid_tactilel_left"
    _write(root / sensor / f"{sensor}_100.npy")
    _write(root / sensor / f"{sensor}_100.txt")
    _write(root / sensor / "median_050.npy")


def _manifest(
    module: ModuleType,
    *,
    queue_path: Path,
    queue: dict[str, object],
    object_id: str,
    raw_object: Path,
) -> dict[str, object]:
    files = [path for path in raw_object.rglob("*") if path.is_file()]
    payload: dict[str, object] = {
        "artifact_kind": "Deform360FreshSourceDownload",
        "revision": queue["dataset"]["revision"],
        "queue_sha256": queue["queue_sha256"],
        "queue_file_sha256": file_sha256(queue_path),
        "download_scope": "ranked_queued_episode_causal_source",
        "tactile_included": True,
        "objects": [
            {
                "queue_rank": 1,
                "object_id": object_id,
                "episode_id": 0,
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "metadata_sha256": file_sha256(raw_object / "metadata.json"),
            }
        ],
        "information_boundary": {
            "episode_payload_deserialized": False,
            "future_object_positions_deserialized": False,
            "target_metrics_opened": False,
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    assert payload["manifest_sha256"] == module._download_manifest_sha256(payload)
    return payload


def test_preparation_validates_exact_ranked_causal_source(tmp_path: Path) -> None:
    module = _load_script()
    object_id = "001-source-object"
    raw_object = tmp_path / "raw" / object_id
    _source_tree(raw_object, object_id)
    queue: dict[str, object] = {
        "dataset": {"revision": "1" * 40},
        "queue_sha256": "2" * 64,
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    manifest = _manifest(
        module,
        queue_path=queue_path,
        queue=queue,
        object_id=object_id,
        raw_object=raw_object,
    )
    manifest_path = tmp_path / "download.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    row = module._validate_download(
        manifest_path,
        queue_path=queue_path,
        queue=queue,
        queue_rank=1,
        object_id=object_id,
        episode_id=0,
        raw_object=raw_object,
    )

    assert row["file_count"] == 31


def test_preparation_rejects_download_boundary_tampering(tmp_path: Path) -> None:
    module = _load_script()
    object_id = "001-source-object"
    raw_object = tmp_path / "raw" / object_id
    _source_tree(raw_object, object_id)
    queue: dict[str, object] = {
        "dataset": {"revision": "1" * 40},
        "queue_sha256": "2" * 64,
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    manifest = _manifest(
        module,
        queue_path=queue_path,
        queue=queue,
        object_id=object_id,
        raw_object=raw_object,
    )
    manifest["information_boundary"]["target_metrics_opened"] = True
    manifest_path = tmp_path / "download.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="binding changed"):
        module._validate_download(
            manifest_path,
            queue_path=queue_path,
            queue=queue,
            queue_rank=1,
            object_id=object_id,
            episode_id=0,
            raw_object=raw_object,
        )
