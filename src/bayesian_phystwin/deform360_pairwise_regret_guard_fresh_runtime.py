"""Runtime amendment for the fresh pairwise-regret-guard processing study.

The first source-processing smoke reached Nerfstudio but failed before training
because ``gsplat`` could not locate a CUDA toolkit.  This module binds the
outcome-independent runtime repair while preserving that first attempt as an
immutable technical failure.  It deliberately contains no target reader.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_pairwise_regret_guard_fresh_processing import (
    PROCESSING_KIND,
    canonical_sha256,
    validate_fresh_processing_protocol,
)
from .deform360_pairwise_regret_guard_fresh_protocol import file_sha256

RUNTIME_IDENTITY_KIND = "Deform360PairwiseRegretGuardFreshRuntimeIdentity"
RUNTIME_AMENDMENT_KIND = "Deform360PairwiseRegretGuardFreshRuntimeAmendment"
RUNTIME_CERTIFICATE_KIND = "Deform360PairwiseRegretGuardFreshRuntimeCertificate"
RUNTIME_AMENDMENT_ID = "deform360-pairwise-regret-guard-fresh-runtime-v1"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PACKAGES = ("torch", "gsplat", "nerfstudio", "numpy", "ninja")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def collect_fresh_runtime_identity() -> dict[str, Any]:
    """Exercise the exact CUDA backend and return a deterministic identity."""

    cuda_home = os.environ.get("CUDA_HOME")
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    _require(bool(cuda_home), "CUDA_HOME is not set")
    _require(bool(arch_list), "TORCH_CUDA_ARCH_LIST is not set")
    nvcc = shutil.which("nvcc")
    _require(nvcc is not None, "nvcc is not available on PATH")

    import torch  # noqa: PLC0415
    from gsplat import rasterization  # noqa: PLC0415
    from gsplat.cuda import _backend  # noqa: PLC0415

    _require(torch.cuda.is_available(), "CUDA is unavailable to PyTorch")
    _require(_backend._C is not None, "gsplat CUDA backend is unavailable")
    camera_model = str(_backend._C.CameraModelType.PINHOLE)

    device = torch.device("cuda")
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
    opacities = torch.tensor([0.9], device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    viewmats = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor(
        [[[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rendered, alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        16,
        16,
    )
    _require(bool(torch.isfinite(rendered).all()), "synthetic render is nonfinite")
    _require(float(alpha.max()) > 0.0, "synthetic render has zero support")

    executable = Path(sys.executable).resolve()
    nvcc_path = Path(nvcc).resolve()
    backend_path = Path(_backend._C.__file__).resolve()
    device_index = torch.cuda.current_device()
    identity: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RUNTIME_IDENTITY_KIND,
        "python": {
            "executable": str(executable),
            "executable_sha256": file_sha256(executable),
            "version": platform.python_version(),
        },
        "platform": {
            "node": platform.node(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "packages": {name: importlib.metadata.version(name) for name in _PACKAGES},
        "cuda": {
            "available": True,
            "cuda_home": str(Path(cuda_home).resolve()),
            "torch_cuda": torch.version.cuda,
            "torch_cuda_arch_list": arch_list,
            "device_name": torch.cuda.get_device_name(device_index),
            "device_capability": list(torch.cuda.get_device_capability(device_index)),
            "nvcc": str(nvcc_path),
            "nvcc_sha256": file_sha256(nvcc_path),
            "nvcc_version": _command_output([str(nvcc_path), "--version"]),
        },
        "gsplat": {
            "backend": str(backend_path),
            "backend_sha256": file_sha256(backend_path),
            "camera_model": camera_model,
            "synthetic_rasterization": {
                "render_shape": list(rendered.shape),
                "alpha_shape": list(alpha.shape),
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
    identity["identity_sha256"] = canonical_sha256(
        identity, digest_key="identity_sha256"
    )
    validate_fresh_runtime_identity(identity)
    return identity


def validate_fresh_runtime_identity(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "wrong runtime identity schema")
    _require(
        payload.get("artifact_kind") == RUNTIME_IDENTITY_KIND,
        "wrong runtime identity kind",
    )
    _require(
        payload.get("identity_sha256")
        == canonical_sha256(payload, digest_key="identity_sha256"),
        "runtime identity checksum changed",
    )
    python = payload.get("python")
    cuda = payload.get("cuda")
    gsplat = payload.get("gsplat")
    packages = payload.get("packages")
    boundary = payload.get("information_boundary")
    _require(isinstance(python, Mapping), "runtime Python identity is missing")
    _require(isinstance(cuda, Mapping), "runtime CUDA identity is missing")
    _require(isinstance(gsplat, Mapping), "runtime gsplat identity is missing")
    _require(isinstance(packages, Mapping), "runtime package identity is missing")
    _require(set(packages) == set(_PACKAGES), "runtime package set changed")
    for key in ("executable_sha256",):
        _require(bool(_HEX64.fullmatch(str(python.get(key, "")))), f"bad {key}")
    for key in ("nvcc_sha256",):
        _require(bool(_HEX64.fullmatch(str(cuda.get(key, "")))), f"bad {key}")
    _require(cuda.get("available") is True, "runtime CUDA is unavailable")
    _require(cuda.get("device_capability") == [8, 9], "GPU capability changed")
    _require(
        bool(_HEX64.fullmatch(str(gsplat.get("backend_sha256", "")))),
        "bad gsplat backend hash",
    )
    synthetic = gsplat.get("synthetic_rasterization")
    _require(isinstance(synthetic, Mapping), "runtime rasterization test is missing")
    _require(
        synthetic.get("finite") is True and synthetic.get("nonzero_alpha") is True,
        "runtime rasterization test failed",
    )
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("source_rgb_read") is False
        and boundary.get("processed_geometry_read") is False
        and boundary.get("target_metric_read") is False
        and boundary.get("held_v8_runtime_or_target_artifact_access") is False,
        "runtime identity crossed its information boundary",
    )


def build_fresh_runtime_amendment(
    processing_protocol_path: str | Path,
    failed_artifact_path: str | Path,
    failed_log_path: str | Path,
    runtime_identity_path: str | Path,
    *,
    validator_commit: str,
) -> dict[str, Any]:
    """Bind the runtime repair after a source-only infrastructure failure."""

    _require(bool(_HEX40.fullmatch(validator_commit)), "bad validator commit")
    protocol = _read_json(processing_protocol_path)
    failure = _read_json(failed_artifact_path)
    identity = _read_json(runtime_identity_path)
    validate_fresh_processing_protocol(protocol)
    validate_fresh_runtime_identity(identity)
    _require(failure.get("artifact_kind") == PROCESSING_KIND, "wrong failure kind")
    _require(failure.get("status") == "technical_failure", "attempt did not fail")
    error = failure.get("error")
    _require(isinstance(error, Mapping), "failure error is missing")
    _require(
        error.get("type") == "AttributeError"
        and error.get("message")
        == "'NoneType' object has no attribute 'CameraModelType'",
        "runtime amendment is not tied to the observed gsplat failure",
    )
    boundary = failure.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("target_metric_read") is False
        and boundary.get("held_v8_runtime_or_target_artifact_access") is False,
        "failed attempt crossed its information boundary",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RUNTIME_AMENDMENT_KIND,
        "amendment_id": RUNTIME_AMENDMENT_ID,
        "status": "locked_before_runtime_repair_attempt",
        "validator_commit": validator_commit,
        "processing_protocol": {
            "protocol_sha256": protocol["protocol_sha256"],
            "file_sha256": file_sha256(processing_protocol_path),
            "implementation_commit": protocol["implementation_commit"],
        },
        "first_attempt": {
            "case": failure["case"],
            "status": failure["status"],
            "result_sha256": failure["result_sha256"],
            "artifact_file_sha256": file_sha256(failed_artifact_path),
            "log_file_sha256": file_sha256(failed_log_path),
            "error": dict(error),
            "retained_as_technical_failure": True,
        },
        "runtime_identity": identity,
        "repair_scope": {
            "algorithm_changed": False,
            "source_bytes_changed": False,
            "window_or_camera_panel_changed": False,
            "mask_or_admission_rule_changed": False,
            "target_or_outcome_used": False,
            "separate_output_tree_required": True,
            "first_attempt_may_not_be_deleted_or_relabelled": True,
        },
        "information_boundary": {
            "source_processing_status_read": True,
            "target_metric_read": False,
            "future_object_positions_deserialized": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    payload["amendment_sha256"] = canonical_sha256(
        payload, digest_key="amendment_sha256"
    )
    validate_fresh_runtime_amendment(payload)
    return payload


def validate_fresh_runtime_amendment(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "wrong runtime amendment schema")
    _require(
        payload.get("artifact_kind") == RUNTIME_AMENDMENT_KIND,
        "wrong runtime amendment kind",
    )
    _require(payload.get("amendment_id") == RUNTIME_AMENDMENT_ID, "wrong amendment")
    _require(
        payload.get("status") == "locked_before_runtime_repair_attempt",
        "runtime amendment is not locked",
    )
    _require(
        bool(_HEX40.fullmatch(str(payload.get("validator_commit", "")))),
        "bad runtime validator commit",
    )
    _require(
        payload.get("amendment_sha256")
        == canonical_sha256(payload, digest_key="amendment_sha256"),
        "runtime amendment checksum changed",
    )
    protocol = payload.get("processing_protocol")
    first = payload.get("first_attempt")
    repair = payload.get("repair_scope")
    boundary = payload.get("information_boundary")
    _require(isinstance(protocol, Mapping), "processing protocol binding is missing")
    _require(isinstance(first, Mapping), "first attempt binding is missing")
    _require(first.get("retained_as_technical_failure") is True, "failure not retained")
    _require(isinstance(repair, Mapping), "runtime repair scope is missing")
    expected_false = (
        "algorithm_changed",
        "source_bytes_changed",
        "window_or_camera_panel_changed",
        "mask_or_admission_rule_changed",
        "target_or_outcome_used",
    )
    _require(
        all(repair.get(key) is False for key in expected_false)
        and repair.get("separate_output_tree_required") is True
        and repair.get("first_attempt_may_not_be_deleted_or_relabelled") is True,
        "runtime repair scope changed",
    )
    validate_fresh_runtime_identity(payload.get("runtime_identity", {}))
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("target_metric_read") is False
        and boundary.get("future_object_positions_deserialized") is False
        and boundary.get("held_v8_runtime_or_target_artifact_access") is False,
        "runtime amendment crossed its information boundary",
    )


def validate_fresh_runtime_sources(
    amendment_path: str | Path,
    processing_protocol_path: str | Path,
    failed_artifact_path: str | Path,
    failed_log_path: str | Path,
) -> dict[str, Any]:
    amendment = _read_json(amendment_path)
    validate_fresh_runtime_amendment(amendment)
    protocol = _read_json(processing_protocol_path)
    validate_fresh_processing_protocol(protocol)
    binding = amendment["processing_protocol"]
    _require(
        binding["protocol_sha256"] == protocol["protocol_sha256"]
        and binding["file_sha256"] == file_sha256(processing_protocol_path),
        "processing protocol binding changed",
    )
    first = amendment["first_attempt"]
    _require(
        first["artifact_file_sha256"] == file_sha256(failed_artifact_path)
        and first["log_file_sha256"] == file_sha256(failed_log_path),
        "first runtime attempt binding changed",
    )
    return amendment


def certify_fresh_runtime(
    amendment: Mapping[str, Any], *, validator_commit: str
) -> dict[str, Any]:
    """Require an exact runtime match and emit a source-free certificate."""

    validate_fresh_runtime_amendment(amendment)
    _require(
        validator_commit == amendment["validator_commit"],
        "runtime validator revision changed",
    )
    identity = collect_fresh_runtime_identity()
    _require(identity == amendment["runtime_identity"], "runtime identity changed")
    certificate: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RUNTIME_CERTIFICATE_KIND,
        "amendment_id": RUNTIME_AMENDMENT_ID,
        "status": "runtime_passed",
        "validator_commit": validator_commit,
        "amendment_sha256": amendment["amendment_sha256"],
        "runtime_identity_sha256": identity["identity_sha256"],
        "information_boundary": {
            "source_rgb_read": False,
            "processed_geometry_read": False,
            "target_metric_read": False,
            "held_v8_runtime_or_target_artifact_access": False,
        },
    }
    certificate["result_sha256"] = canonical_sha256(
        certificate, digest_key="result_sha256"
    )
    return certificate


__all__ = [
    "RUNTIME_AMENDMENT_ID",
    "RUNTIME_AMENDMENT_KIND",
    "RUNTIME_CERTIFICATE_KIND",
    "RUNTIME_IDENTITY_KIND",
    "build_fresh_runtime_amendment",
    "certify_fresh_runtime",
    "collect_fresh_runtime_identity",
    "validate_fresh_runtime_amendment",
    "validate_fresh_runtime_identity",
    "validate_fresh_runtime_sources",
]
