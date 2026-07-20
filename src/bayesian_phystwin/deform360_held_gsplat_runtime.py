"""Frozen gsplat CUDA supplement and target-free held-runtime smoke.

The held-v7 outcome process uses the unchanged held-v5 Python environment.
That environment contains :mod:`gsplat` 1.4.0 but no packaged CUDA extension,
and its deliberately restricted ``PATH`` excludes both ``nvcc`` and ``ninja``.
This module loads one separately frozen ahead-of-time extension by exact byte
identity, installs it into gsplat's already-disabled backend, and exercises the
same non-packed forward/backward rasterization path used by Splatfacto.

No function in this module accepts a dataset, episode, target, prediction, or
outcome path.  The smoke is therefore safe to run before the held cohort
barrier.  The loaded extension is retained globally so the later reconstruction
in the same process uses the exact backend that passed the smoke.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import sys
from types import ModuleType
from typing import Any, Mapping


GSPLAT_CUDA_EXTENSION_PATH = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v7-runtimes/"
    "gsplat-cuda-py312-cu121-"
    "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64/"
    "gsplat_cuda.so"
)
GSPLAT_RUNTIME_SUPPLEMENT_MANIFEST_PATH = (
    GSPLAT_CUDA_EXTENSION_PATH.parent / "runtime-supplement-manifest.json"
)
GSPLAT_RUNTIME_SMOKE_EVIDENCE_PATH = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v7/"
    "gsplat-runtime-smoke-evidence.json"
)

GSPLAT_CUDA_EXTENSION_CONTRACT: Mapping[str, Any] = {
    "contract_id": "deform360-held-gsplat-cuda-extension-v1",
    "canonical_path": os.fspath(GSPLAT_CUDA_EXTENSION_PATH),
    "file_size_bytes": 6_982_312,
    "file_mode_octal": "0444",
    "parent_mode_octal": "0555",
    "sha256": "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64",
    "python_abi": "cp312",
    "torch_version": "2.4.0+cu121",
    "torch_cuda_version": "12.1",
    "gsplat_version": "1.4.0",
    "build_cuda_toolkit_version": "12.9",
    "compute_capability": "8.9",
    "cuda_architecture": "sm_89",
    "cuda_cpp_header_source_aggregate_sha256": (
        "d3895ab1a0fc389d7e42ce796dcc7b7fca9ec07b36d8ecbc54e941f2dcbef59f"
    ),
    "base_runtime_manifest_sha256": (
        "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61"
    ),
    "base_runtime_pip_freeze_sha256": (
        "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
    ),
    "jit_compilation_permitted": False,
}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


GSPLAT_CUDA_EXTENSION_CONTRACT_SHA256 = _canonical_sha256(
    GSPLAT_CUDA_EXTENSION_CONTRACT
)

GSPLAT_RUNTIME_SMOKE_CONTRACT: Mapping[str, Any] = {
    "contract_id": "deform360-held-gsplat-runtime-smoke-v1",
    "extension_contract_sha256": GSPLAT_CUDA_EXTENSION_CONTRACT_SHA256,
    "execution_host": "workstation2",
    "logical_device": "cuda:0",
    "formal_physical_gpu_indices": [0, 1],
    "gpu_name": "NVIDIA RTX 6000 Ada Generation",
    "compute_capability": "8.9",
    "python_major_minor": [3, 12],
    "torch_version": "2.4.0+cu121",
    "torch_cuda_version": "12.1",
    "gsplat_version": "1.4.0",
    "normalized_path": "/usr/local/bin:/usr/bin:/bin",
    "nvcc_visible": False,
    "ninja_visible": False,
    "torch_jit_extension_loading_permitted": False,
    "ambient_backend_must_initially_be_none": True,
    "extension_required_exports": [
        "CameraModelType",
        "fully_fused_projection_fwd",
        "fully_fused_projection_bwd",
        "rasterize_to_pixels_fwd",
        "rasterize_to_pixels_bwd",
    ],
    "rasterization": {
        "gaussian_count": 2,
        "camera_count": 1,
        "width": 16,
        "height": 16,
        "packed": False,
        "near_plane": 0.01,
        "far_plane": 1.0e10,
        "render_mode": "RGB",
        "sh_degree": None,
        "sparse_grad": False,
        "absgrad": False,
        "rasterize_mode": "classic",
    },
    "required_checks": [
        "extension_stable_nofollow_identity_before_and_after",
        "extension_loaded_from_exact_path",
        "forward_render_and_alpha_finite_nonempty_nonzero",
        "both_gaussians_have_positive_projected_radius",
        "backward_gradients_finite_and_nonzero_for_all_five_parameter_groups",
        "cuda_synchronized",
        "backend_retained_for_same_process_outcome_reconstruction",
    ],
    "forbidden_input_classes": [
        "dataset",
        "episode",
        "target",
        "prediction",
        "outcome",
        "tactile",
    ],
}
GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256 = _canonical_sha256(GSPLAT_RUNTIME_SMOKE_CONTRACT)

_EXPECTED_NORMALIZED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONHASHSEED": "0",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONPYCACHEPREFIX": "/nonexistent/bpt-held-v7-pycache",
    "WANDB_MODE": "disabled",
}

_LOADED_GSPLAT_CUDA: ModuleType | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    signed = dict(unsigned)
    signed["artifact_sha256"] = _canonical_sha256(unsigned)
    return signed


def _sha256_regular_file_snapshot(path: Path) -> tuple[tuple[int, int, int, int], str]:
    """Hash one exact regular file without following its final component."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(absolute)
    _require(stat.S_ISREG(before.st_mode), "gsplat extension is not a regular file")
    _require(not absolute.is_symlink(), "gsplat extension is a symlink")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            "gsplat extension changed while opening",
        )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
    finally:
        os.close(descriptor)
    after = os.lstat(absolute)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    _require(
        identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "gsplat extension changed while hashing",
    )
    return identity, digest.hexdigest()


def _validate_extension_file(path: Path) -> tuple[tuple[int, int, int, int], str]:
    expected = Path(str(GSPLAT_CUDA_EXTENSION_CONTRACT["canonical_path"]))
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require(absolute == expected, "gsplat extension path changed")
    _require(absolute.resolve(strict=True) == absolute, "gsplat extension is aliased")
    parent = absolute.parent
    _require(
        parent.resolve(strict=True) == parent and not parent.is_symlink(),
        "gsplat extension parent is aliased",
    )
    _require(
        stat.S_IMODE(os.lstat(parent).st_mode)
        == int(str(GSPLAT_CUDA_EXTENSION_CONTRACT["parent_mode_octal"]), 8),
        "gsplat extension parent mode changed",
    )
    _require(
        stat.S_IMODE(os.lstat(absolute).st_mode)
        == int(str(GSPLAT_CUDA_EXTENSION_CONTRACT["file_mode_octal"]), 8),
        "gsplat extension mode changed",
    )
    identity, digest = _sha256_regular_file_snapshot(absolute)
    _require(
        identity[2] == GSPLAT_CUDA_EXTENSION_CONTRACT["file_size_bytes"],
        "gsplat extension size changed",
    )
    _require(
        digest == GSPLAT_CUDA_EXTENSION_CONTRACT["sha256"],
        "gsplat extension checksum changed",
    )
    return identity, digest


def _validate_formal_environment() -> int:
    _require(socket.gethostname() == "workstation2", "gsplat smoke host changed")
    _require(
        sys.version_info[:2]
        == tuple(GSPLAT_RUNTIME_SMOKE_CONTRACT["python_major_minor"]),
        "gsplat smoke Python ABI changed",
    )
    for name, expected in _EXPECTED_NORMALIZED_ENVIRONMENT.items():
        _require(
            os.environ.get(name) == expected,
            f"gsplat smoke environment changed: {name}",
        )
    _require(
        sys.flags.dont_write_bytecode == 1
        and sys.pycache_prefix
        == _EXPECTED_NORMALIZED_ENVIRONMENT["PYTHONPYCACHEPREFIX"],
        "gsplat smoke process may create or consult adjacent bytecode",
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(visible in {"0", "1"}, "gsplat smoke requires exactly physical GPU 0 or 1")
    _require(shutil.which("nvcc") is None, "nvcc unexpectedly visible to held runtime")
    _require(
        shutil.which("ninja") is None, "ninja unexpectedly visible to held runtime"
    )
    _require(
        "TORCH_EXTENSIONS_DIR" not in os.environ,
        "mutable Torch extension cache was enabled",
    )
    return int(visible)


def _load_exact_extension(path: Path, backend: ModuleType) -> ModuleType:
    global _LOADED_GSPLAT_CUDA

    _require(_LOADED_GSPLAT_CUDA is None, "gsplat CUDA supplement was already loaded")
    _require(
        getattr(backend, "_C", None) is None,
        "ambient gsplat CUDA backend was not disabled before supplement loading",
    )
    _require("gsplat_cuda" not in sys.modules, "ambient gsplat_cuda module is present")
    spec = importlib.util.spec_from_file_location("gsplat_cuda", os.fspath(path))
    _require(
        spec is not None and spec.loader is not None,
        "cannot construct gsplat extension spec",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gsplat_cuda"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("gsplat_cuda", None)
        raise
    _require(
        Path(str(module.__file__)).resolve(strict=True) == path,
        "gsplat CUDA supplement loaded from another path",
    )
    for name in GSPLAT_RUNTIME_SMOKE_CONTRACT["extension_required_exports"]:
        _require(hasattr(module, str(name)), f"gsplat CUDA export is absent: {name}")
    backend._C = module  # type: ignore[attr-defined]
    _LOADED_GSPLAT_CUDA = module
    return module


def _run_fixed_rasterization_smoke(torch: Any, rasterization: Any) -> Mapping[str, Any]:
    device = "cuda:0"
    dtype = torch.float32
    means = torch.tensor(
        [[-0.055, -0.018, 1.85], [0.063, 0.041, 2.15]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    quats = torch.tensor(
        [[1.0, 0.04, -0.02, 0.01], [0.96, -0.08, 0.12, 0.03]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    scales = torch.tensor(
        [[0.075, 0.052, 0.043], [0.061, 0.082, 0.049]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    opacities = torch.tensor(
        [0.78, 0.67], device=device, dtype=dtype, requires_grad=True
    )
    colors = torch.tensor(
        [[0.82, 0.21, 0.13], [0.11, 0.57, 0.91]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    intrinsics = torch.tensor(
        [[[34.0, 0.0, 8.0], [0.0, 33.0, 8.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )
    render, alpha, metadata = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=16,
        height=16,
        packed=False,
        near_plane=0.01,
        far_plane=1.0e10,
        render_mode="RGB",
        sh_degree=None,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode="classic",
    )
    _require(tuple(render.shape) == (1, 16, 16, 3), "gsplat render shape changed")
    _require(tuple(alpha.shape) == (1, 16, 16, 1), "gsplat alpha shape changed")
    _require(
        bool(torch.isfinite(render).all().item())
        and bool(torch.isfinite(alpha).all().item()),
        "gsplat forward output is non-finite",
    )
    _require(
        float(render.detach().abs().sum().item()) > 0.0
        and float(alpha.detach().sum().item()) > 0.0,
        "gsplat forward output is empty",
    )
    radii = metadata.get("radii")
    _require(radii is not None and int(radii.numel()) == 2, "gsplat radii changed")
    _require(
        bool((radii > 0).all().item()), "a fixed smoke Gaussian was not rasterized"
    )

    render_weights = torch.linspace(
        0.25, 1.25, render.numel(), device=device, dtype=dtype
    ).reshape(render.shape)
    alpha_weights = torch.linspace(
        1.15, 0.35, alpha.numel(), device=device, dtype=dtype
    ).reshape(alpha.shape)
    loss = (render * render_weights).sum() + 0.41 * (alpha * alpha_weights).sum()
    _require(bool(torch.isfinite(loss).item()), "gsplat smoke loss is non-finite")
    loss.backward()
    gradient_groups = {
        "means": means.grad,
        "quats": quats.grad,
        "scales": scales.grad,
        "opacities": opacities.grad,
        "colors": colors.grad,
    }
    for name, gradient in gradient_groups.items():
        _require(gradient is not None, f"gsplat backward omitted {name} gradient")
        _require(
            bool(torch.isfinite(gradient).all().item()),
            f"gsplat backward produced non-finite {name} gradient",
        )
        _require(
            float(gradient.detach().abs().sum().item()) > 0.0,
            f"gsplat backward produced zero {name} gradient",
        )
    torch.cuda.synchronize(device)
    return {
        "render_shape": [1, 16, 16, 3],
        "alpha_shape": [1, 16, 16, 1],
        "positive_radius_count": 2,
        "gradient_groups_finite_and_nonzero": sorted(gradient_groups),
        "forward_finite_nonempty_nonzero": True,
        "backward_complete": True,
        "cuda_synchronized": True,
    }


def load_and_smoke_gsplat_runtime() -> Mapping[str, Any]:
    """Load the exact AOT extension and run a target-free CUDA smoke once.

    The default-only signature is intentional: accepting caller-provided paths
    would weaken the pre-barrier information boundary.  Raises ``RuntimeError``
    on every validation or kernel failure.
    """

    physical_gpu_index = _validate_formal_environment()
    before_identity, extension_sha256 = _validate_extension_file(
        GSPLAT_CUDA_EXTENSION_PATH
    )

    try:
        torch = importlib.import_module("torch")
        _require(
            str(torch.__version__) == GSPLAT_RUNTIME_SMOKE_CONTRACT["torch_version"],
            "held torch version changed",
        )
        _require(
            str(torch.version.cuda)
            == GSPLAT_RUNTIME_SMOKE_CONTRACT["torch_cuda_version"],
            "held torch CUDA version changed",
        )
        _require(torch.cuda.is_available(), "CUDA is unavailable to gsplat smoke")
        _require(torch.cuda.device_count() == 1, "gsplat smoke sees more than one GPU")
        _require(
            torch.cuda.get_device_name(0) == GSPLAT_RUNTIME_SMOKE_CONTRACT["gpu_name"],
            "formal GPU model changed",
        )
        capability = torch.cuda.get_device_capability(0)
        _require(capability == (8, 9), "formal GPU compute capability changed")
        _require(
            importlib.metadata.version("gsplat")
            == GSPLAT_RUNTIME_SMOKE_CONTRACT["gsplat_version"],
            "held gsplat version changed",
        )
        backend = importlib.import_module("gsplat.cuda._backend")
        module = _load_exact_extension(GSPLAT_CUDA_EXTENSION_PATH, backend)
        rendering = importlib.import_module("gsplat.rendering")
        rasterization = getattr(rendering, "rasterization", None)
        _require(callable(rasterization), "gsplat rasterization API is absent")
        smoke = _run_fixed_rasterization_smoke(torch, rasterization)
        _require(
            getattr(backend, "_C", None) is module and _LOADED_GSPLAT_CUDA is module,
            "gsplat CUDA backend was not retained",
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"gsplat CUDA supplement smoke failed: {type(error).__name__}: {error}"
        ) from error

    after_identity, after_sha256 = _validate_extension_file(GSPLAT_CUDA_EXTENSION_PATH)
    _require(
        after_identity == before_identity and after_sha256 == extension_sha256,
        "gsplat extension changed during smoke",
    )
    return _artifact(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
            "contract_sha256": GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256,
            "physical_gpu_index": physical_gpu_index,
            "logical_device": "cuda:0",
            "gpu_name": GSPLAT_RUNTIME_SMOKE_CONTRACT["gpu_name"],
            "compute_capability": "8.9",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "gsplat_version": importlib.metadata.version("gsplat"),
            "extension_path": os.fspath(GSPLAT_CUDA_EXTENSION_PATH),
            "extension_sha256": extension_sha256,
            "extension_loaded_and_retained": True,
            "nvcc_visible": False,
            "ninja_visible": False,
            "target_or_outcome_path_accessed": False,
            "predicates": dict(smoke),
        }
    )


__all__ = [
    "GSPLAT_CUDA_EXTENSION_CONTRACT",
    "GSPLAT_CUDA_EXTENSION_CONTRACT_SHA256",
    "GSPLAT_CUDA_EXTENSION_PATH",
    "GSPLAT_RUNTIME_SMOKE_CONTRACT",
    "GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256",
    "GSPLAT_RUNTIME_SMOKE_EVIDENCE_PATH",
    "GSPLAT_RUNTIME_SUPPLEMENT_MANIFEST_PATH",
    "load_and_smoke_gsplat_runtime",
]
