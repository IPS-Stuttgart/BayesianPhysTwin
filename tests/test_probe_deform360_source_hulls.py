from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "probe_deform360_source_hulls.py"
)
_SPEC = importlib.util.spec_from_file_location("_deform360_source_hull_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

load_probe_protocol = _MODULE.load_probe_protocol
probe_locked_source_hulls = _MODULE.probe_locked_source_hulls
write_probe = _MODULE.write_probe


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_protocol(
    path: Path,
    entries: list[dict[str, object]],
) -> None:
    objects = {str(entry["object_id"]) for entry in entries}
    config = {
        "protocol_id": "deform360-source-hull-contract-probe-v1",
        "status": "locked-before-source-hull-payload-metadata-access",
        "cohort": {
            "object_count": len(objects),
            "episode_count": len(entries),
            "reserved_target_object_count": 0,
            "unit_of_replication": "physical object",
            "entries": entries,
        },
        "probe": {
            "minimum_point_count_per_frame": 1,
            "required_members": [
                "frame_indices.npy",
                "point_offsets.npy",
                "points_world_m.npy",
            ],
        },
        "source_inventory": {
            "content_inventory_sha256": "1" * 64,
            "inventory_sha256": "2" * 64,
            "product_head_sha": "3" * 64,
            "evaluated_merge_sha": "4" * 64,
            "workflow_artifact_sha256": "5" * 64,
        },
    }
    payload = {
        "schema": (
            "bayesian-phystwin/"
            "deform360-source-hull-contract-probe-protocol-v1"
        ),
        "config": config,
        "config_sha256": __import__("hashlib").sha256(
            _canonical_bytes(config)
        ).hexdigest(),
        "schema_version": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_hulls(
    path: Path,
    *,
    frames: np.ndarray,
    point_counts: tuple[int, ...],
) -> None:
    offsets = np.zeros(len(point_counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(point_counts)
    points = np.arange(int(offsets[-1]) * 3, dtype=np.float64).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_indices=frames,
        point_offsets=offsets,
        points_world_m=points,
    )


def _fixture(tmp_path: Path, *, irregular: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "data"
    relative = (
        "data-7fea8e2/replication-v1/observations/"
        "002-rope-silk/episode_0000/sampled_hulls.npz"
    )
    frames = np.array([0, 2, 5], dtype=np.int32)
    if not irregular:
        frames = np.array([0, 2, 4], dtype=np.int32)
    _write_hulls(root / relative, frames=frames, point_counts=(4, 5, 6))
    protocol = tmp_path / "protocol.json"
    _write_protocol(
        protocol,
        [
            {
                "object_id": "002-rope-silk",
                "episode_id": 0,
                "classification": "prior_open_or_reserved",
                "relative_path": relative,
                "representation": "packed_visual_hulls",
            }
        ],
    )
    return root, protocol


def test_probe_reads_cadence_and_headers_without_reporting_coordinates(
    tmp_path: Path,
) -> None:
    root, protocol = _fixture(tmp_path)

    result = probe_locked_source_hulls(
        root,
        protocol_path=protocol,
        revision="revision-test",
    )

    assert result["object_count"] == 1
    assert result["episode_count"] == 1
    assert result["all_archives_constant_frame_stride"] is True
    archive = result["archives"][0]
    assert archive["frame_indices"] == [0, 2, 4]
    assert archive["frame_stride_counts"] == {"2": 2}
    assert archive["points_world_m_header"]["shape"] == [15, 3]
    assert archive["points_world_m_header"]["coordinate_values_decoded"] is False
    assert "points_world_m" not in archive
    assert result["information_boundary"]["model_prediction_run"] is False
    assert result["information_boundary"]["complete_archive_bytes_hashed"] is True
    assert result["constant_frame_stride_counts"] == {"2": 1}
    assert result["irregular_frame_stride_archive_count"] == 0
    assert len(result["content_probe_sha256"]) == 64
    assert len(result["probe_sha256"]) == 64


def test_probe_reports_irregular_stride_without_hiding_it(tmp_path: Path) -> None:
    root, protocol = _fixture(tmp_path, irregular=True)

    result = probe_locked_source_hulls(root, protocol_path=protocol)

    assert result["all_archives_constant_frame_stride"] is False
    assert result["archives"][0]["frame_stride_counts"] == {"2": 1, "3": 1}
    assert result["irregular_frame_stride_archive_count"] == 1


def test_probe_fails_closed_on_offset_contract(tmp_path: Path) -> None:
    root, protocol = _fixture(tmp_path)
    archive = next(root.rglob("sampled_hulls.npz"))
    with np.load(archive, allow_pickle=False) as stored:
        frames = stored["frame_indices"]
        points = stored["points_world_m"]
    np.savez_compressed(
        archive,
        frame_indices=frames,
        point_offsets=np.array([0, 4, 9, 14], dtype=np.int64),
        points_world_m=points,
    )

    with pytest.raises(ValueError, match="final value"):
        probe_locked_source_hulls(root, protocol_path=protocol)


def test_protocol_rejects_reserved_target(tmp_path: Path) -> None:
    root, protocol = _fixture(tmp_path)
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    payload["config"]["cohort"]["entries"][0]["classification"] = "reserved_target"
    payload["config"]["cohort"]["reserved_target_object_count"] = 1
    payload["config_sha256"] = __import__("hashlib").sha256(
        _canonical_bytes(payload["config"])
    ).hexdigest()
    protocol.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved targets"):
        load_probe_protocol(protocol)

    assert root.is_dir()


def test_probe_output_is_newline_terminated(tmp_path: Path) -> None:
    root, protocol = _fixture(tmp_path)
    result = probe_locked_source_hulls(root, protocol_path=protocol)
    output = tmp_path / "probe.json"

    write_probe(output, result)

    assert output.read_text(encoding="utf-8").endswith("\n")
