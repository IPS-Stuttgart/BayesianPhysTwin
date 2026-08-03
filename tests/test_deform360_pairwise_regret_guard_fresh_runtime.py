from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    canonical_sha256,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_runtime import (
    RUNTIME_AMENDMENT_KIND,
    build_fresh_runtime_amendment,
    validate_fresh_runtime_amendment,
    validate_fresh_runtime_identity,
    validate_fresh_runtime_sources,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs/sota/deform360_pairwise_regret_guard_fresh_processing_v1.json"
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identity(tmp_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360PairwiseRegretGuardFreshRuntimeIdentity",
        "python": {
            "executable": "/runtime/bin/python",
            "executable_sha256": "1" * 64,
            "version": "3.10.12",
        },
        "platform": {"node": "host", "system": "Linux", "machine": "x86_64"},
        "packages": {
            "torch": "2.4.0+cu121",
            "gsplat": "1.4.0",
            "nerfstudio": "1.1.5",
            "numpy": "1.26.4",
            "ninja": "1.13.0",
        },
        "cuda": {
            "available": True,
            "cuda_home": "/usr/local/cuda-12.4",
            "torch_cuda": "12.1",
            "torch_cuda_arch_list": "8.9",
            "device_name": "NVIDIA GeForce RTX 4090",
            "device_capability": [8, 9],
            "nvcc": "/usr/local/cuda-12.4/bin/nvcc",
            "nvcc_sha256": "2" * 64,
            "nvcc_version": "Cuda compilation tools, release 12.4",
        },
        "gsplat": {
            "backend": "/runtime/gsplat_cuda.so",
            "backend_sha256": "3" * 64,
            "camera_model": "CameraModelType.PINHOLE",
            "synthetic_rasterization": {
                "render_shape": [1, 16, 16, 3],
                "alpha_shape": [1, 16, 16, 1],
                "finite": True,
                "nonzero_alpha": True,
            },
        },
        "information_boundary": {
            "source_rgb_read": False,
            "processed_geometry_read": False,
            "target_metric_read": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    payload["identity_sha256"] = canonical_sha256(payload, digest_key="identity_sha256")
    path = tmp_path / "identity.json"
    _write(path, payload)
    return path


def _failure(tmp_path: Path) -> tuple[Path, Path]:
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360PairwiseRegretGuardFreshProcessing",
        "case": "197-hand-sanitizer-ep0000",
        "status": "technical_failure",
        "error": {
            "type": "AttributeError",
            "message": "'NoneType' object has no attribute 'CameraModelType'",
        },
        "information_boundary": {
            "target_metric_read": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    artifact = tmp_path / "failure.json"
    log = tmp_path / "failure.log"
    _write(artifact, payload)
    log.write_text("gsplat: No CUDA toolkit found.\n", encoding="utf-8")
    return artifact, log


def test_runtime_amendment_preserves_failed_attempt(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    failure, log = _failure(tmp_path)
    payload = build_fresh_runtime_amendment(
        PROTOCOL,
        failure,
        log,
        identity,
        validator_commit="a" * 40,
    )
    assert payload["artifact_kind"] == RUNTIME_AMENDMENT_KIND
    assert payload["first_attempt"]["retained_as_technical_failure"] is True
    assert payload["repair_scope"]["separate_output_tree_required"] is True
    validate_fresh_runtime_amendment(payload)

    amendment = tmp_path / "amendment.json"
    _write(amendment, payload)
    validate_fresh_runtime_sources(amendment, PROTOCOL, failure, log)


def test_runtime_identity_rejects_changed_backend(tmp_path: Path) -> None:
    identity = json.loads(_identity(tmp_path).read_text(encoding="utf-8"))
    validate_fresh_runtime_identity(identity)
    identity["gsplat"]["backend_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="checksum changed"):
        validate_fresh_runtime_identity(identity)


def test_runtime_source_binding_rejects_changed_failure_log(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    failure, log = _failure(tmp_path)
    payload = build_fresh_runtime_amendment(
        PROTOCOL,
        failure,
        log,
        identity,
        validator_commit="b" * 40,
    )
    amendment = tmp_path / "amendment.json"
    _write(amendment, payload)
    assert payload["first_attempt"]["artifact_file_sha256"] == file_sha256(failure)
    log.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="first runtime attempt binding changed"):
        validate_fresh_runtime_sources(amendment, PROTOCOL, failure, log)
