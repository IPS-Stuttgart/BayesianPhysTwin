from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from bayesian_phystwin.deform360_causal_response_direct_depth_assets import (
    canonical_sha256,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_V1 = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_prefix_geometry_runtime.json"
)
RUNTIME_V2 = ROOT / "configs" / "sota" / (
    "deform360_causal_response_direct_depth_v14_prefix_geometry_runtime_v2.json"
)
WRAPPER_V1 = ROOT / "scripts" / "remote" / (
    "build_deform360_causal_response_direct_depth_v14_prefix_geometry_runtime.py"
)
WRAPPER_V2 = ROOT / "scripts" / "remote" / (
    "build_deform360_causal_response_direct_depth_v14_prefix_geometry_runtime_v2.py"
)


def _load_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_v14_geometry_runtime_v2_wrapper",
        WRAPPER_V2,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v14_geometry_runtime_v2_binds_parent_and_interpreter() -> None:
    wrapper = _load_wrapper()
    amendment = wrapper._load_amendment(RUNTIME_V2)

    assert amendment["parent_runtime_amendment"]["file_sha256"] == file_sha256(
        RUNTIME_V1
    )
    assert amendment["parent_runtime_amendment"]["runtime_wrapper_sha256"] == (
        file_sha256(WRAPPER_V1)
    )
    assert amendment["implementation_file_sha256"]["runtime_wrapper_v2"] == (
        file_sha256(WRAPPER_V2)
    )
    assert amendment["runtime_amendment"]["semantic_runtime_contract_changed"] is False
    assert amendment["application_policy"]["applies_to_queue_ranks"] == list(
        range(4, 15)
    )


def test_v14_geometry_runtime_v2_application_binds_one_result(
    tmp_path: Path,
) -> None:
    wrapper = _load_wrapper()
    amendment = wrapper._load_amendment(RUNTIME_V2)
    result = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalDirectDepthPrefixGeometryResultV14",
        "queue_rank": 4,
        "object_hash": "a" * 64,
        "case_hash": "b" * 64,
        "status": "ready_for_source_lock",
    }
    result["artifact_sha256"] = canonical_sha256(
        result,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-"
            b"result-v14\0"
        ),
        digest_key="artifact_sha256",
    )
    result_path = tmp_path / "rank-004.json"
    result_path.write_text(
        json.dumps(result, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    wrapper._write_application(
        result_path,
        amendment=amendment,
        amendment_path=RUNTIME_V2,
        wrapper_path=WRAPPER_V2,
    )

    application_path = tmp_path / "rank-004.runtime-v2.json"
    application = json.loads(application_path.read_text(encoding="utf-8"))
    assert application["geometry_result_file_sha256"] == file_sha256(result_path)
    assert application["runtime_amendment_config_sha256"] == amendment[
        "config_sha256"
    ]
    assert application["artifact_sha256"] == canonical_sha256(
        application,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-"
            b"runtime-application-v14-v2\0"
        ),
        digest_key="artifact_sha256",
    )
