from __future__ import annotations

import hashlib
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
_SPEC = importlib.util.spec_from_file_location(
    "_deform360_source_hull_probe_amendment",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

load_probe_amendment = _MODULE.load_probe_amendment
load_probe_protocol = _MODULE.load_probe_protocol
probe_locked_source_hulls = _MODULE.probe_locked_source_hulls


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "data"
    relative = (
        "data-7fea8e2/replication-v1/observations/"
        "002-rope-silk/episode_0000/sampled_hulls.npz"
    )
    archive = root / relative
    archive.parent.mkdir(parents=True)
    np.savez_compressed(
        archive,
        frame_indices=np.array([0, 2, 4, 6], dtype=np.int32),
        point_offsets=np.array([0, 4, 4, 9, 14], dtype=np.int64),
        points_world_m=np.arange(42, dtype=np.float64).reshape(14, 3),
    )

    config = {
        "protocol_id": "deform360-source-hull-contract-probe-v1",
        "status": "locked-before-source-hull-payload-metadata-access",
        "cohort": {
            "object_count": 1,
            "episode_count": 1,
            "reserved_target_object_count": 0,
            "unit_of_replication": "physical object",
            "entries": [
                {
                    "object_id": "002-rope-silk",
                    "episode_id": 0,
                    "classification": "prior_open_or_reserved",
                    "relative_path": relative,
                    "representation": "packed_visual_hulls",
                }
            ],
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
            "product_head_sha": "3" * 40,
            "evaluated_merge_sha": "4" * 40,
            "workflow_artifact_sha256": "5" * 64,
            "workflow_run_id": 123,
            "workflow_artifact_id": 456,
        },
    }
    protocol = tmp_path / "protocol.json"
    _write_json(
        protocol,
        {
            "schema": (
                "bayesian-phystwin/"
                "deform360-source-hull-contract-probe-protocol-v1"
            ),
            "schema_version": 1,
            "config": config,
            "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        },
    )

    amendment_config = {
        "amendment_id": "deform360-source-hull-contract-probe-v2",
        "base_protocol": {
            "path": "protocol.json",
            "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        },
        "status": "locked-after-v1-structural-failure-before-coordinate-access",
        "policy": {
            "point_offsets_order": "nondecreasing",
            "empty_frame_handling": (
                "retain in archive custody; exclude from the prediction sequence"
            ),
            "minimum_points_for_usable_frame": 1,
            "minimum_usable_frames_for_rolling_prediction": 3,
            "cadence_basis": (
                "strictly increasing frame_indices after empty-frame exclusion"
            ),
            "object_balancing_for_future_scoring": True,
        },
        "information_boundary": {
            "points_world_m_coordinate_values_decoded": False,
            "model_prediction_run": False,
            "score_bearing_outcome_computed": False,
            "reserved_target_outcomes_opened": False,
        },
        "trigger_evidence": {
            "workflow_run_id": 789,
            "workflow_job_id": 790,
            "evaluated_merge_sha": "6" * 40,
            "failure": "point_offsets must be strictly increasing",
            "artifact_id": 791,
            "artifact_sha256": "7" * 64,
        },
    }
    amendment = tmp_path / "amendment.json"
    _write_json(
        amendment,
        {
            "schema": (
                "bayesian-phystwin/"
                "deform360-source-hull-contract-probe-amendment-v2"
            ),
            "schema_version": 1,
            "config": amendment_config,
            "config_sha256": hashlib.sha256(
                _canonical_bytes(amendment_config)
            ).hexdigest(),
        },
    )
    return root, protocol, amendment


def test_v1_fails_but_v2_classifies_empty_frames(tmp_path: Path) -> None:
    root, protocol, amendment = _fixture(tmp_path)

    with pytest.raises(ValueError, match="strictly increasing"):
        probe_locked_source_hulls(root, protocol_path=protocol)

    result = probe_locked_source_hulls(
        root,
        protocol_path=protocol,
        amendment_path=amendment,
        revision="revision-test",
    )

    assert result["amendment_id"] == "deform360-source-hull-contract-probe-v2"
    assert result["empty_frame_archive_count"] == 1
    assert result["total_empty_frame_count"] == 1
    assert result["prediction_eligible_episode_count"] == 1
    assert result["prediction_ineligible_episode_count"] == 0
    record = result["archives"][0]
    assert record["frame_indices"] == [0, 2, 4, 6]
    assert record["usable_frame_indices"] == [0, 4, 6]
    assert record["empty_frame_indices"] == [2]
    assert record["frame_stride_counts"] == {"2": 1, "4": 1}
    assert record["prediction_eligible"] is True
    assert record["points_world_m_header"]["coordinate_values_decoded"] is False


def test_amendment_rejects_changed_information_boundary(tmp_path: Path) -> None:
    _, protocol_path, amendment_path = _fixture(tmp_path)
    protocol = load_probe_protocol(protocol_path)
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["config"]["information_boundary"]["model_prediction_run"] = True
    amendment["config_sha256"] = hashlib.sha256(
        _canonical_bytes(amendment["config"])
    ).hexdigest()
    _write_json(amendment_path, amendment)

    with pytest.raises(ValueError, match="information boundary"):
        load_probe_amendment(amendment_path, base_protocol=protocol)
