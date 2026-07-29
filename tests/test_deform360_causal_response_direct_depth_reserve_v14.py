from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_v14 import (
    RESERVE_BATCH_RANKS,
    RESERVE_GEOMETRY_RANKS,
    load_v14_reserve_batch_protocol,
    load_v14_reserve_geometry_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_reserve_batch_v1.json"
)
GEOMETRY_CONFIG = (
    ROOT
    / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_reserve_geometry_v1.json"
)
RUNTIME_CONFIG = (
    ROOT
    / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_reserve_geometry_runtime_v2.json"
)
RUNTIME_WRAPPER = (
    ROOT
    / "scripts/remote/"
    "build_deform360_causal_response_direct_depth_v14_"
    "prefix_geometry_reserve_runtime_v2.py"
)


def _load_runtime_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_v14_reserve_runtime_test",
        RUNTIME_WRAPPER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v14_reserve_batch_binds_exact_parents_and_ranks() -> None:
    protocol = load_v14_reserve_batch_protocol(
        CONFIG,
        method_protocol_path=(
            ROOT
            / "configs/sota/deform360_causal_response_direct_depth_v14.json"
        ),
        asset_protocol_path=(
            ROOT
            / "configs/sota/"
            "deform360_causal_response_direct_depth_v14_assets.json"
        ),
        queue_path=(
            ROOT
            / "configs/sota/"
            "deform360_causal_response_direct_depth_v14_staging_queue.json"
        ),
    )
    assert tuple(protocol["batch_contract"]["batch_queue_ranks"]) == (
        RESERVE_BATCH_RANKS
    )
    assert protocol["batch_contract"]["admissions_required"] == 4


def test_v14_reserve_batch_rejects_rank_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["batch_contract"]["batch_queue_ranks"][-1] = 23
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|selection contract"):
        load_v14_reserve_batch_protocol(mutated)


def test_v14_reserve_geometry_binds_masks_and_preparation_failure() -> None:
    protocol = load_v14_reserve_geometry_protocol(
        GEOMETRY_CONFIG,
        reserve_batch_path=CONFIG,
        asset_protocol_path=(
            ROOT
            / "configs/sota/"
            "deform360_causal_response_direct_depth_v14_assets.json"
        ),
        queue_path=(
            ROOT
            / "configs/sota/"
            "deform360_causal_response_direct_depth_v14_staging_queue.json"
        ),
    )
    assert tuple(
        row["queue_rank"] for row in protocol["mask_inputs"]
    ) == RESERVE_GEOMETRY_RANKS
    assert protocol["technical_dispositions"][0]["queue_rank"] == 20


def test_v14_reserve_geometry_rejects_mask_mutation(tmp_path: Path) -> None:
    payload = json.loads(GEOMETRY_CONFIG.read_text(encoding="utf-8"))
    payload["mask_inputs"][0]["successful_camera_count"] = 11
    mutated = tmp_path / "mutated-geometry.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|mask ledger"):
        load_v14_reserve_geometry_protocol(mutated)


def test_v14_reserve_runtime_v2_binds_failed_parent_adapter() -> None:
    module = _load_runtime_wrapper()
    runtime = module._load_runtime(
        RUNTIME_CONFIG,
        geometry_path=GEOMETRY_CONFIG,
        wrapper_path=RUNTIME_WRAPPER,
    )
    assert runtime["trigger"]["failure_key"] == "parent_prefix_assets"
    assert runtime["application_policy"]["applies_to_queue_ranks"] == [
        15,
        16,
        17,
        18,
        19,
        21,
        22,
    ]
