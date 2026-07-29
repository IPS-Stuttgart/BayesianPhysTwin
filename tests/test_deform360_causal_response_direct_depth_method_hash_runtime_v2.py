from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_method_hash_runtime_v2 import (
    correct_v14_method_config_sha256,
    legacy_v14_method_config_sha256,
    load_v14_method_hash_runtime_v2,
)

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "configs/sota/deform360_causal_response_direct_depth_v14.json"
RUNTIME = (
    ROOT / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_method_hash_runtime_v2.json"
)


def test_method_hash_runtime_repairs_only_the_registered_namespace() -> None:
    method = json.loads(METHOD.read_text(encoding="utf-8"))
    runtime = load_v14_method_hash_runtime_v2(
        RUNTIME,
        repository=ROOT,
        method_protocol_path=METHOD,
    )

    assert correct_v14_method_config_sha256(method) == method["config_sha256"]
    assert legacy_v14_method_config_sha256(method) != method["config_sha256"]
    assert runtime["trigger"]["source_admission_artifact_created"] is False
    assert runtime["trigger"]["source_outcome_read"] is False
    assert runtime["amendment"]["estimator_or_gate_changed"] is False


def test_method_hash_runtime_rejects_scope_mutation(tmp_path: Path) -> None:
    payload = json.loads(RUNTIME.read_text(encoding="utf-8"))
    payload["amendment"]["estimator_or_gate_changed"] = True
    changed = tmp_path / "changed-runtime.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity or checksum"):
        load_v14_method_hash_runtime_v2(
            changed,
            repository=ROOT,
            method_protocol_path=METHOD,
        )
