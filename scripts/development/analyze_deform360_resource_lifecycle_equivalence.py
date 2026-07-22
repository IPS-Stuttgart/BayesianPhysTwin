#!/usr/bin/env python3
"""Distributional numerical-equivalence check for Deform360 splat fits.

This is a development-only qualification operator.  It compares repeated
Gaussian-Ply exports from the released Deform360 trainer (``original``) and
the resource-bounded adapter (``wrapped``).  It must never be pointed at a
formal held execution tree.

Exact equality of identically paired repeats is the primary result.  If exact
equality does not hold, a predeclared secondary gate compares every
non-negative pairwise distance against the empirical within-mode variation.
The secondary gate is deliberately an engineering equivalence envelope, not
a calibrated statistical hypothesis test: pairwise values share repeats and
bidirectional nearest-neighbour matching is not bijective.  Count differences
and fixed-camera RGB/alpha comparisons are retained to expose two important
failure modes of nearest-neighbour geometry.

The input is a self-signed JSON manifest.  Every input file is bound by its
exact size and SHA-256, the two modes have the same five-or-more pairing IDs,
and the manifest binds the clean code and Deform360 Git trees.  All PLYs are
rendered exactly once, in one 21-camera gsplat call per PLY, using the frozen
held-v8 AOT CUDA entry point, a fixed 4x downscale, and SH degree zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import stat
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_ID = "deform360-resource-lifecycle-distributional-equivalence-v1"
MANIFEST_KIND = "Deform360ResourceLifecycleRepeatManifestV1"
RESULT_KIND = "Deform360ResourceLifecycleDistributionalEquivalenceV1"
FORMAL_HELD_PARENT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
PINNED_PYTHON_RUNTIME = PINNED_PYTHON.parent.parent
PINNED_PYTHON_FREEZE = Path(f"{PINNED_PYTHON_RUNTIME}.freeze.sorted.txt")
PINNED_PYTHON_TREE_MANIFEST = Path(f"{PINNED_PYTHON_RUNTIME}.tree-manifest.json")
PINNED_PYTHON_FREEZE_SHA256 = (
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
PINNED_PYTHON_TREE_MANIFEST_SHA256 = (
    "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61"
)
PINNED_PYTHON_TREE_MANIFEST_KIND = "Deform360HeldPythonRuntimeTreeManifestV1"
PINNED_PYTHON_RUNTIME_SYMLINKS = {
    "bin/python": "/usr/bin/python3",
    "bin/python3": "python",
    "bin/python3.12": "python",
}
PINNED_PYTHON_SYMLINK_TARGET = "/usr/bin/python3"
PINNED_PYTHON_RESOLVED = Path("/usr/bin/python3.12")
PINNED_PYTHON_BASE_PREFIX = Path("/usr")
PINNED_PYTHON_RESOLVED_SHA256 = (
    "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7"
)
PINNED_DEFORM360_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
PINNED_DEFORM360_TREE = "c566ed29db7e0fd6a4cb768d840a4aa662864680"
GENERATOR_CODE_HEAD = "0db75dc3a54cba682e5398caac4d301b96aa412f"
GENERATOR_CODE_TREE = "e11526534afe21ce65cfa963ee9b35052fe452ba"
GENERATOR_SOURCE_BINDINGS: Mapping[str, Mapping[str, Any]] = {
    "scripts/development/qualify_deform360_resource_lifecycle.py": {
        "git_blob_oid": "96111dc1af2ef56bc91db38f5791b9f5625e5be5",
        "sha256": "64ace2a7785effc384fbe0193506ba1ef7e63aea675e8fd6bca0ced61932db9c",
        "size_bytes": 73_102,
    },
    "src/bayesian_phystwin/deform360_held_gsplat_runtime.py": {
        "git_blob_oid": "2756631693df63835624dc86d6c54932e55337cb",
        "sha256": "61be5d30d6fd049e03e67d555bbf6ecd54f0411b2ec91c7b1d60226e94df2e18",
        "size_bytes": 18_363,
    },
    "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py": {
        "git_blob_oid": "35ac4cd19ceb56c82d9393f816486cb30e498515",
        "sha256": "ea58daa482e93f7c32dbf65ede6cfc2f9a1a377cf2896c7be130a8baf316f242",
        "size_bytes": 98_437,
    },
    "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py": {
        "git_blob_oid": "18bdc5bfcd77e1a73be41f8852c3cf4637e7db73",
        "sha256": "2985de3b4e3f6bea7e98eb0e36148f52d8ee96bce027eb13bad98e87fd7f875c",
        "size_bytes": 696,
    },
}
FIT_EVIDENCE_KIND = "Deform360ResourceLifecycleFitChildEvidence"
FIT_QUALIFICATION_IDS = {
    "historical-0db": "deform360-nerfstudio-resource-lifecycle-qualification-v1",
    "same-as-analyzer": "deform360-nerfstudio-resource-lifecycle-qualification-v2",
}
PROFILE_PHYSICAL_GPU_INDEX = {"historical-0db": 0, "same-as-analyzer": 1}
QUALIFIER_SOURCE_RELATIVE = (
    "scripts/development/qualify_deform360_resource_lifecycle.py"
)
CORE_GENERATOR_SOURCE_PATHS = frozenset(
    {
        "src/bayesian_phystwin/deform360_held_gsplat_runtime.py",
        "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py",
        "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py",
    }
)
CANONICAL_PUBLIC_SOURCE_DATASET = Path(
    "/mnt/corsair/florianpfaff/deform360-reusable-sota-v1/"
    "processing-sam2-dev-smoke/004-rubber-band/episode_0001/"
    "splatfacto/.scratch_000000"
)
FIT_ITERATIONS = 250
FIT_SEED = 0
PINNED_TORCH_VERSION = "2.4.0+cu121"
PINNED_TORCH_CUDA_VERSION = "12.1"
PINNED_GPU_NAME = "NVIDIA RTX 6000 Ada Generation"
PINNED_GSPLAT_VERSION = "1.4.0"
PINNED_GSPLAT_EXTENSION_SHA256 = (
    "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64"
)
PINNED_GSPLAT_EXTENSION_PATH = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v7-runtimes/"
    "gsplat-cuda-py312-cu121-"
    "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64/"
    "gsplat_cuda.so"
)
PINNED_GSPLAT_SMOKE_CONTRACT_SHA256 = (
    "0c2786579530037e32e6b7e39291cbae9b06f9113828d602864f13d84d335962"
)
RELATIVE_ANALYZER_SOURCE = Path(
    "scripts/development/analyze_deform360_resource_lifecycle_equivalence.py"
)
RELATIVE_GSPLAT_ADAPTER_SOURCE = Path(
    "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
)
MINIMUM_REPEATS_PER_MODE = 5
MINIMUM_WITHIN_PAIRS_PER_MODE = 10
MINIMUM_CROSS_PAIRS = 25
CANONICAL_CAMERA_COUNT = 21
RENDER_DOWNSCALE = 4
INFERENCE_BACKGROUND_RGB = (0.1490, 0.1647, 0.2157)
REQUIRED_EXECUTION_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/home/florianpfaff",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LOGNAME": "florianpfaff",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYNPUT_BACKEND": "dummy",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPYCACHEPREFIX": "/nonexistent/bpt-held-v8-pycache",
    "PYTHONSAFEPATH": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "USER": "florianpfaff",
    "WANDB_MODE": "disabled",
}
FORBIDDEN_EXECUTION_ENVIRONMENT = (
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "TORCH_EXTENSIONS_DIR",
)

EXPECTED_PLY_FIELDS = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(45)),
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
SH_FIELDS = (
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(45)),
)

PAIR_METRIC_NAMES = (
    "relative_count_delta",
    "xyz_distance_mean_m",
    "xyz_distance_p95_m",
    "xyz_distance_max_m",
    "opacity_probability_abs_mean",
    "opacity_probability_abs_p95",
    "log_scale_vector_l2_mean",
    "log_scale_vector_l2_p95",
    "quaternion_angle_mean_rad",
    "quaternion_angle_p95_rad",
    "sh_vector_l2_mean",
    "sh_vector_l2_p95",
    "rgb_rmse",
    "alpha_rmse",
)

RENDER_CONTRACT: Mapping[str, Any] = {
    "contract_id": "deform360-equivalence-fixed-gsplat-render-v1",
    "camera_count": CANONICAL_CAMERA_COUNT,
    "integer_downscale": RENDER_DOWNSCALE,
    "logical_device": "cuda:0",
    "dtype": "torch.float32",
    "packed": False,
    "near_plane": 0.01,
    "far_plane": 1.0e10,
    "radius_clip": 0.0,
    "eps2d": 0.3,
    "sh_degree": 0,
    "tile_size": 16,
    "rasterization_backgrounds": None,
    "inference_background_rgb": list(INFERENCE_BACKGROUND_RGB),
    "rgb_compositing": "clamp(render_rgb + (1 - alpha) * background, 0, 1)",
    "render_mode": "RGB",
    "sparse_grad": False,
    "absgrad": False,
    "rasterize_mode": "classic",
    "channel_chunk": 32,
    "distributed": False,
    "camera_model": "pinhole",
    "camera_convention": (
        "Nerfstudio OpenGL camera-to-world; flip columns 1 and 2, then invert "
        "to gsplat OpenCV world-to-camera"
    ),
    "gaussian_activation": (
        "sigmoid opacity; exp log-scale; unit-normalized quaternion; raw SH DC"
    ),
    "one_batched_rasterization_call_per_ply": True,
}

GATE_CONTRACT: Mapping[str, Any] = {
    "contract_id": "deform360-empirical-pairwise-equivalence-envelope-v1",
    "metric_names": list(PAIR_METRIC_NAMES),
    "within_original_minimum_pair_count": MINIMUM_WITHIN_PAIRS_PER_MODE,
    "within_wrapped_minimum_pair_count": MINIMUM_WITHIN_PAIRS_PER_MODE,
    "cross_mode_minimum_pair_count": MINIMUM_CROSS_PAIRS,
    "percentile_method": "linear",
    "per_metric_conditions": [
        "cross_mode_median <= max(within_original_p95, within_wrapped_p95)",
        "cross_mode_p95 <= max(within_original_max, within_wrapped_max)",
    ],
    "all_metrics_required": True,
    "exact_matched_structured_array_equality_is_primary": True,
    "distributional_gate_is_secondary": True,
}

STATISTICAL_LIMITATIONS = (
    "Bidirectional nearest-neighbour matching is assignment-invariant but not "
    "bijective; duplicated or redistributed density can reduce its distances.",
    "Pairwise values are dependent because each fitted repeat occurs in several "
    "pairs; this envelope is not a calibrated independent-sample hypothesis test.",
    "The empirical envelope can be exactly zero when both within-mode processes "
    "are deterministic, in which case any nonzero cross-mode discrepancy fails.",
    "Fixed-camera rendering only tests the declared 21 cameras at SH degree zero "
    "and cannot establish equivalence of unseen views or higher-order SH effects.",
)

_PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _artifact_sha256(result)
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_formal_held_path(path: str | Path) -> bool:
    candidate = _absolute(path)
    try:
        relative = candidate.relative_to(FORMAL_HELD_PARENT)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0].startswith("held-")


def _reject_symlink_ancestors(path: Path, *, label: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            value = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise ValueError(f"{label} ancestor is unavailable: {current}") from error
        _require(not stat.S_ISLNK(value.st_mode), f"{label} has a symlink ancestor")


def _assert_nonheld_path(
    path: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    absolute = _absolute(path)
    _reject_symlink_ancestors(absolute, label=label)
    _require(
        not _is_formal_held_path(absolute),
        f"{label} points into a formal held root",
    )
    if must_exist:
        try:
            lexical = os.lstat(absolute)
        except OSError as error:
            raise ValueError(f"{label} is unavailable: {absolute}") from error
        _require(not stat.S_ISLNK(lexical.st_mode), f"{label} is a symlink")
    try:
        resolved = absolute.resolve(strict=must_exist)
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {absolute}") from error
    _require(
        not _is_formal_held_path(resolved),
        f"{label} resolves into a formal held root",
    )
    return resolved


def _reject_output_within_roots(
    output: Path,
    roots: Iterable[Path],
    *,
    label: str,
) -> None:
    candidate = _absolute(output)
    for raw_root in roots:
        root = _absolute(raw_root)
        _require(
            candidate != root and root not in candidate.parents,
            f"{label} is inside a protected source/input root: {root}",
        )


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_nofollow(path: str | Path, *, label: str) -> bytes:
    source = _assert_nonheld_path(path, label=label, must_exist=True)
    before = os.lstat(source)
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    _require(before.st_nlink == 1, f"{label} is a hardlink")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_identity(opened) == _stable_identity(before),
            f"{label} changed while opening",
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(source)
        _require(
            _stable_identity(after) == _stable_identity(opened)
            and _stable_identity(current) == _stable_identity(opened),
            f"{label} changed while reading",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _bound_file(path: str | Path, *, label: str = "file") -> dict[str, Any]:
    source = _assert_nonheld_path(path, label=label, must_exist=True)
    payload = _read_regular_nofollow(source, label=label)
    mode = stat.S_IMODE(os.lstat(source).st_mode)
    return {
        "path": os.fspath(source),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode_octal": f"{mode:04o}",
    }


def _validated_pinned_aot_binding() -> dict[str, Any]:
    """Bind the frozen gsplat extension before any cohort input is opened."""
    binding = _bound_file(
        PINNED_GSPLAT_EXTENSION_PATH,
        label="pinned gsplat AOT extension",
    )
    _require(
        binding["sha256"] == PINNED_GSPLAT_EXTENSION_SHA256,
        "gsplat AOT changed",
    )
    _require(
        binding["mode_octal"] == "0444",
        "gsplat AOT file mode changed",
    )
    parent = _assert_nonheld_path(
        PINNED_GSPLAT_EXTENSION_PATH.parent,
        label="pinned gsplat AOT parent",
        must_exist=True,
    )
    parent_stat = os.lstat(parent)
    _require(
        stat.S_ISDIR(parent_stat.st_mode)
        and stat.S_IMODE(parent_stat.st_mode) == 0o555,
        "gsplat AOT parent mode changed",
    )
    return binding


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = _assert_nonheld_path(path, label="JSON output", must_exist=False)
    _reject_symlink_ancestors(destination.parent, label="JSON output parent")
    _require(
        destination.parent.is_dir() and not destination.parent.is_symlink(),
        "JSON output parent must already be a real directory",
    )
    _require(not os.path.lexists(destination), f"output already exists: {destination}")
    temporary = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    parent_descriptor: int | None = None
    temporary_created = False
    published = False
    try:
        parent_before = os.lstat(destination.parent)
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        _require(
            _stable_identity(os.fstat(parent_descriptor))
            == _stable_identity(parent_before),
            "JSON output parent changed while opening",
        )
        descriptor = os.open(
            temporary.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_created = True
        payload = _canonical_json(value)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary.name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary.name, dir_fd=parent_descriptor)
        temporary_created = False
        os.fsync(parent_descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created and parent_descriptor is not None:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if published and os.path.lexists(destination):
            # Publication only happens after a complete fsynced file exists.
            # Retain it rather than turning a crash/error into silent absence.
            pass
        raise
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    _require(os.lstat(destination).st_nlink == 1, "published JSON is hardlinked")
    return destination


def _load_signed_json(path: str | Path, *, label: str) -> dict[str, Any]:
    payload = _read_regular_nofollow(path, label=label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    observed = value.get("artifact_sha256")
    _require(
        isinstance(observed, str)
        and len(observed) == 64
        and observed == _artifact_sha256(value),
        f"{label} signature is invalid",
    )
    return value


def _hex_digest(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _git_object_id(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase Git object ID",
    )
    return value


def _git_binding(root: str | Path) -> dict[str, Any]:
    repository = _assert_nonheld_path(root, label="Git repository", must_exist=True)
    _require(repository.is_dir() and not repository.is_symlink(), "bad Git root")
    environment = {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["/usr/bin/git", "-C", repository, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        _require(result.returncode == 0, f"Git command failed: {' '.join(arguments)}")
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    ordinary = git("ls-files", "--others", "--exclude-standard")
    ignored = git("ls-files", "--others", "--ignored", "--exclude-standard")
    _require(
        not status and not ordinary and not ignored, f"dirty Git root: {repository}"
    )
    return {
        "path": os.fspath(repository),
        "head": head,
        "tree": tree,
        "clean": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
    }


def _historical_generator_binding(root: str | Path) -> dict[str, Any]:
    """Recompute the exact generator source identity from immutable Git blobs."""

    repository = _assert_nonheld_path(
        root, label="generator object repository", must_exist=True
    )
    environment = {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }

    def git_bytes(*arguments: str) -> bytes:
        result = subprocess.run(
            ["/usr/bin/git", "-C", repository, *arguments],
            check=False,
            capture_output=True,
            timeout=60,
            env=environment,
        )
        _require(
            result.returncode == 0, f"Git object read failed: {' '.join(arguments)}"
        )
        return result.stdout

    tree = git_bytes("rev-parse", f"{GENERATOR_CODE_HEAD}^{{tree}}").decode().strip()
    _require(tree == GENERATOR_CODE_TREE, "generator Git tree changed")
    sources: dict[str, Any] = {}
    for path, expected in sorted(GENERATOR_SOURCE_BINDINGS.items()):
        object_id = (
            git_bytes("rev-parse", f"{GENERATOR_CODE_HEAD}:{path}").decode().strip()
        )
        _require(
            object_id == expected["git_blob_oid"], f"generator blob changed: {path}"
        )
        payload = git_bytes("cat-file", "blob", object_id)
        digest = hashlib.sha256(payload).hexdigest()
        _require(digest == expected["sha256"], f"generator source changed: {path}")
        _require(
            len(payload) == expected["size_bytes"], f"generator size changed: {path}"
        )
        sources[path] = {
            "git_blob_oid": object_id,
            "sha256": digest,
            "size_bytes": len(payload),
        }
    return {
        "head": GENERATOR_CODE_HEAD,
        "tree": GENERATOR_CODE_TREE,
        "sources": sources,
    }


def _git_blob_oid(root: Path, relative: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", root, "rev-parse", f"HEAD:{relative}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/home/florianpfaff",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    _require(result.returncode == 0, f"cannot bind generator source: {relative}")
    return _git_object_id(result.stdout.strip(), label=f"generator blob {relative}")


def _generator_checkout_binding(
    root: str | Path,
    *,
    profile: str,
    analyzer_root: Path,
    analyzer_git: Mapping[str, Any],
) -> dict[str, Any]:
    _require(profile in FIT_QUALIFICATION_IDS, "unknown generator profile")
    repository = _assert_nonheld_path(
        root, label="generator code root", must_exist=True
    )
    git = _git_binding(repository)
    if profile == "historical-0db":
        _require(
            git["head"] == GENERATOR_CODE_HEAD and git["tree"] == GENERATOR_CODE_TREE,
            "historical generator checkout is not the pinned clean revision",
        )
    else:
        _require(
            repository == analyzer_root, "final generator root differs from analyzer"
        )
        _require(git == dict(analyzer_git), "final generator Git differs from analyzer")
    sources: dict[str, Any] = {}
    for relative, expected in sorted(GENERATOR_SOURCE_BINDINGS.items()):
        binding = _bound_file(
            repository / relative, label=f"generator source {relative}"
        )
        blob_oid = _git_blob_oid(repository, relative)
        if relative in CORE_GENERATOR_SOURCE_PATHS or profile == "historical-0db":
            _require(
                binding["sha256"] == expected["sha256"]
                and binding["size_bytes"] == expected["size_bytes"]
                and blob_oid == expected["git_blob_oid"],
                f"generator checkout source changed: {relative}",
            )
        sources[relative] = {**binding, "git_blob_oid": blob_oid}
    return {
        "profile": profile,
        "qualification_id": FIT_QUALIFICATION_IDS[profile],
        "physical_gpu_index": PROFILE_PHYSICAL_GPU_INDEX[profile],
        "git": git,
        "sources": sources,
    }


def _live_pip_freeze_binding(python: Path = PINNED_PYTHON) -> dict[str, Any]:
    environment = {
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    result = subprocess.run(
        [os.fspath(python), "-I", "-B", "-m", "pip", "freeze", "--all"],
        check=False,
        capture_output=True,
        timeout=120,
        env=environment,
    )
    _require(result.returncode == 0, "pinned Python pip freeze --all failed")
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("pinned Python pip freeze output is not UTF-8") from error
    lines = sorted(line.strip() for line in decoded.splitlines() if line.strip())
    _require(lines, "pinned Python pip freeze --all returned no distributions")
    normalized = ("\n".join(lines) + "\n").encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()
    _require(
        digest == PINNED_PYTHON_FREEZE_SHA256,
        "live pip freeze differs from the frozen package inventory",
    )
    return {
        "command": [os.fspath(python), "-I", "-B", "-m", "pip", "freeze", "--all"],
        "normalized_sha256": digest,
        "normalized_line_count": len(lines),
        "normalized_size_bytes": len(normalized),
        "equals_frozen_package_inventory": True,
    }


def _runtime_tree_paths(root: Path) -> list[str]:
    paths: list[str] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as scan:
                children = sorted(scan, key=lambda item: os.fsencode(item.name))
        except OSError as error:
            raise ValueError(
                f"cannot scan pinned Python runtime: {directory}"
            ) from error
        for child in children:
            path = Path(child.path)
            relative = os.path.relpath(path, root)
            observed = os.lstat(path)
            paths.append(relative)
            if stat.S_ISDIR(observed.st_mode):
                visit(path)
            else:
                _require(
                    stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode),
                    f"unsupported pinned Python runtime entry: {relative}",
                )

    visit(root)
    return sorted(paths, key=os.fsencode)


def _sha256_regular_streaming(path: Path, *, expected: os.stat_result) -> str:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_identity(opened) == _stable_identity(expected),
            f"runtime file changed while opening: {path}",
        )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_identity(after) == _stable_identity(opened)
            and _stable_identity(current) == _stable_identity(opened),
            f"runtime file changed while hashing: {path}",
        )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_runtime_tree() -> dict[str, Any]:
    root = _assert_nonheld_path(
        PINNED_PYTHON_RUNTIME, label="pinned Python runtime root", must_exist=True
    )
    root_stat = os.lstat(root)
    _require(
        stat.S_ISDIR(root_stat.st_mode) and stat.S_IMODE(root_stat.st_mode) == 0o555,
        "pinned Python runtime root mode differs from 0555",
    )
    expected_manifest = root.parent / f"{root.name}.tree-manifest.json"
    _require(
        PINNED_PYTHON_TREE_MANIFEST == expected_manifest,
        "pinned Python runtime manifest is not the exact sibling path",
    )
    raw = _read_regular_nofollow(
        PINNED_PYTHON_TREE_MANIFEST, label="pinned runtime tree manifest"
    )
    _require(
        hashlib.sha256(raw).hexdigest() == PINNED_PYTHON_TREE_MANIFEST_SHA256,
        "pinned Python runtime manifest digest changed",
    )
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("pinned Python runtime manifest is invalid JSON") from error
    _require(isinstance(manifest, Mapping), "runtime tree manifest is not an object")
    _require(
        set(manifest)
        == {
            "artifact_kind",
            "root_path",
            "python_pip_freeze_sorted_sha256",
            "entry_counts",
            "total_regular_file_bytes",
            "tree_sha256",
            "entries",
        },
        "runtime tree manifest fields changed",
    )
    _require(
        raw
        == json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n",
        "runtime tree manifest is not canonical JSON",
    )
    _require(
        manifest["artifact_kind"] == PINNED_PYTHON_TREE_MANIFEST_KIND
        and manifest["root_path"] == os.fspath(root)
        and manifest["python_pip_freeze_sorted_sha256"] == PINNED_PYTHON_FREEZE_SHA256,
        "runtime tree manifest identity changed",
    )
    entries = manifest["entries"]
    _require(isinstance(entries, list), "runtime tree entries are not a list")
    paths: list[str] = []
    counts = {"directory": 0, "file": 0, "symlink": 0}
    total_bytes = 0
    symlinks: dict[str, str] = {}
    for index, entry in enumerate(entries):
        _require(isinstance(entry, Mapping), f"runtime entry {index} is not an object")
        path_value = entry.get("path")
        entry_type = entry.get("type")
        mode = entry.get("mode")
        _require(
            isinstance(path_value, str)
            and path_value
            and "\x00" not in path_value
            and "\\" not in path_value
            and not path_value.startswith("/")
            and all(part not in {"", ".", ".."} for part in path_value.split("/")),
            f"runtime entry {index} path is invalid",
        )
        _require(
            isinstance(mode, str)
            and len(mode) == 4
            and all(character in "01234567" for character in mode),
            f"runtime entry {path_value} mode is invalid",
        )
        _require(entry_type in counts, f"runtime entry {path_value} type is invalid")
        if entry_type == "directory":
            _require(
                set(entry) == {"path", "mode", "type"},
                "runtime directory fields changed",
            )
        elif entry_type == "file":
            _require(
                set(entry) == {"path", "mode", "type", "size", "sha256"},
                "runtime file fields changed",
            )
            _strict_int(
                entry.get("size"), label=f"runtime file {path_value} size", minimum=0
            )
            _hex_digest(entry.get("sha256"), label=f"runtime file {path_value}")
        else:
            _require(
                set(entry) == {"path", "mode", "type", "target"}
                and isinstance(entry.get("target"), str)
                and bool(entry.get("target")),
                "runtime symlink fields changed",
            )
        paths.append(path_value)
    _require(
        paths == sorted(paths, key=os.fsencode) and len(paths) == len(set(paths)),
        "runtime tree paths are unsorted or duplicated",
    )
    _require(
        _runtime_tree_paths(root) == paths, "runtime path set differs from manifest"
    )
    for entry in entries:
        relative = str(entry["path"])
        path = root.joinpath(*relative.split("/"))
        observed = os.lstat(path)
        observed_mode = stat.S_IMODE(observed.st_mode)
        _require(
            f"{observed_mode:04o}" == entry["mode"], f"runtime mode changed: {relative}"
        )
        entry_type = str(entry["type"])
        if entry_type == "directory":
            _require(
                stat.S_ISDIR(observed.st_mode) and observed_mode & 0o222 == 0,
                f"runtime directory changed or is writable: {relative}",
            )
        elif entry_type == "file":
            _require(
                stat.S_ISREG(observed.st_mode)
                and observed_mode & 0o222 == 0
                and observed.st_size == entry["size"],
                f"runtime file metadata changed: {relative}",
            )
            _require(
                _sha256_regular_streaming(path, expected=observed) == entry["sha256"],
                f"runtime file content changed: {relative}",
            )
            total_bytes += observed.st_size
        else:
            _require(
                stat.S_ISLNK(observed.st_mode) and os.readlink(path) == entry["target"],
                f"runtime symlink changed: {relative}",
            )
            symlinks[relative] = str(entry["target"])
        counts[entry_type] += 1
    _require(
        symlinks == PINNED_PYTHON_RUNTIME_SYMLINKS, "runtime symlink policy changed"
    )
    declared_counts = manifest["entry_counts"]
    _require(
        isinstance(declared_counts, Mapping)
        and set(declared_counts) == set(counts)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in declared_counts.values()
        )
        and dict(declared_counts) == counts,
        "runtime entry counts changed",
    )
    _require(
        _strict_int(
            manifest["total_regular_file_bytes"],
            label="runtime total regular file bytes",
            minimum=0,
        )
        == total_bytes,
        "runtime total byte count changed",
    )
    tree_sha = hashlib.sha256(
        json.dumps(
            entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    _require(tree_sha == manifest["tree_sha256"], "runtime tree digest changed")
    return {
        "runtime_root": os.fspath(root),
        "runtime_root_mode_octal": "0555",
        "runtime_manifest_path": os.fspath(PINNED_PYTHON_TREE_MANIFEST),
        "runtime_manifest_sha256": PINNED_PYTHON_TREE_MANIFEST_SHA256,
        "runtime_tree_sha256": tree_sha,
        "entry_counts": counts,
        "total_regular_file_bytes": total_bytes,
        "all_directories_and_regular_files_nonwritable": True,
        "all_entry_metadata_and_file_hashes_verified": True,
    }


def _runtime_binding() -> dict[str, Any]:
    executable = _absolute(sys.executable)
    base_executable_value = getattr(sys, "_base_executable", None)
    _require(
        isinstance(base_executable_value, str) and base_executable_value,
        "analyzer Python has no base executable",
    )
    base_executable = _absolute(base_executable_value)
    _require(executable == PINNED_PYTHON, "analyzer is not using pinned Python")
    lexical = os.lstat(PINNED_PYTHON)
    _require(stat.S_ISLNK(lexical.st_mode), "pinned Python is not a symlink")
    _require(
        os.readlink(PINNED_PYTHON) == PINNED_PYTHON_SYMLINK_TARGET,
        "pinned Python symlink target changed",
    )
    _require(
        PINNED_PYTHON.resolve(strict=True) == PINNED_PYTHON_RESOLVED
        and base_executable == PINNED_PYTHON_RESOLVED,
        "resolved pinned Python changed",
    )
    _require(
        _absolute(sys.prefix) == PINNED_PYTHON_RUNTIME,
        "pinned Python prefix changed",
    )
    _require(
        _absolute(sys.base_prefix) == PINNED_PYTHON_BASE_PREFIX,
        "pinned Python base prefix changed",
    )
    resolved = _bound_file(
        PINNED_PYTHON_RESOLVED, label="resolved pinned Python executable"
    )
    _require(
        resolved["sha256"] == PINNED_PYTHON_RESOLVED_SHA256,
        "resolved pinned Python digest changed",
    )
    freeze = _bound_file(PINNED_PYTHON_FREEZE, label="pinned package inventory")
    tree = _bound_file(
        PINNED_PYTHON_TREE_MANIFEST, label="pinned runtime tree manifest"
    )
    _require(
        freeze["sha256"] == PINNED_PYTHON_FREEZE_SHA256
        and freeze["mode_octal"] == "0400",
        "pinned package inventory changed",
    )
    _require(
        tree["sha256"] == PINNED_PYTHON_TREE_MANIFEST_SHA256
        and tree["mode_octal"] == "0400",
        "pinned runtime tree manifest changed",
    )
    return {
        "sys_executable": os.fspath(executable),
        "sys_base_executable": os.fspath(base_executable),
        "sys_prefix": os.fspath(_absolute(sys.prefix)),
        "sys_base_prefix": os.fspath(_absolute(sys.base_prefix)),
        "lexical_python": {
            "path": os.fspath(PINNED_PYTHON),
            "mode_octal": f"{stat.S_IMODE(lexical.st_mode):04o}",
            "symlink_target": PINNED_PYTHON_SYMLINK_TARGET,
        },
        "resolved_python": resolved,
        "frozen_package_inventory": freeze,
        "frozen_runtime_tree_manifest": tree,
        "verified_runtime_tree": _validate_runtime_tree(),
        "live_pip_freeze_all": _live_pip_freeze_binding(),
    }


def _verified_descriptor(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(
        set(value) == {"path", "size_bytes", "sha256"},
        f"{label} fields changed",
    )
    path_value = value.get("path")
    size_value = value.get("size_bytes")
    _require(isinstance(path_value, str) and path_value, f"{label} path is invalid")
    _require(
        isinstance(size_value, int)
        and not isinstance(size_value, bool)
        and size_value > 0,
        f"{label} size is invalid",
    )
    expected_sha = _hex_digest(value.get("sha256"), label=f"{label} sha256")
    observed = _bound_file(path_value, label=label)
    _require(observed["size_bytes"] == size_value, f"{label} size changed")
    _require(observed["sha256"] == expected_sha, f"{label} checksum changed")
    return observed


def _verified_bound_record(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(
        set(value) == {"path", "size_bytes", "sha256", "mode_octal"},
        f"{label} fields changed",
    )
    observed = _bound_file(str(value.get("path")), label=label)
    _require(observed == dict(value), f"{label} binding changed")
    return observed


def _validate_execution_host() -> str:
    hostname = socket.gethostname()
    _require(hostname == "workstation2", "analyzer host is not workstation2")
    return hostname


def _validate_live_torch_device(torch: Any) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is unavailable to the analyzer")
    _require(torch.cuda.device_count() == 1, "analyzer must see exactly one GPU")
    name = str(torch.cuda.get_device_name(0))
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    _require(name == PINNED_GPU_NAME, "analyzer GPU model changed")
    _require(capability == (8, 9), "analyzer GPU compute capability changed")
    _require(str(torch.__version__) == PINNED_TORCH_VERSION, "analyzer Torch changed")
    _require(
        str(torch.version.cuda) == PINNED_TORCH_CUDA_VERSION,
        "analyzer Torch CUDA changed",
    )
    return {
        "visible_cuda_device_count": 1,
        "logical_device": "cuda:0",
        "gpu_name": name,
        "compute_capability": "8.9",
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }


def _validate_process_flags(flags: Any = sys.flags) -> dict[str, int]:
    required = {
        "isolated": 1,
        "no_user_site": 1,
        "dont_write_bytecode": 1,
        "safe_path": 1,
        "ignore_environment": 1,
    }
    observed: dict[str, int] = {}
    for name, expected in required.items():
        value = getattr(flags, name, None)
        _require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == expected,
            f"analyzer Python flag changed: {name}",
        )
        observed[name] = value
    return observed


def _validate_execution_environment(
    physical_gpu_index: int,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    _require(
        isinstance(physical_gpu_index, int)
        and not isinstance(physical_gpu_index, bool)
        and physical_gpu_index in (0, 1),
        "physical GPU index is invalid",
    )
    for name in FORBIDDEN_EXECUTION_ENVIRONMENT:
        _require(
            name not in environment, f"forbidden analyzer environment variable: {name}"
        )
    exact_names = set(REQUIRED_EXECUTION_ENVIRONMENT) | {
        "CUDA_VISIBLE_DEVICES",
        "TMPDIR",
    }
    _require(
        set(environment) == exact_names,
        "analyzer environment variable set is not exactly sanitized",
    )
    for name, expected in REQUIRED_EXECUTION_ENVIRONMENT.items():
        _require(
            environment.get(name) == expected, f"analyzer environment changed: {name}"
        )
    _require(
        environment.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu_index),
        "CUDA_VISIBLE_DEVICES differs from the manifest physical GPU",
    )
    temporary_value = environment.get("TMPDIR")
    _require(isinstance(temporary_value, str) and temporary_value, "TMPDIR is absent")
    temporary = _assert_nonheld_path(
        temporary_value, label="analyzer TMPDIR", must_exist=True
    )
    _require(temporary.is_dir(), "analyzer TMPDIR is not a directory")
    return {
        "required": dict(REQUIRED_EXECUTION_ENVIRONMENT),
        "cuda_visible_devices": str(physical_gpu_index),
        "forbidden_absent": list(FORBIDDEN_EXECUTION_ENVIRONMENT),
        "tmpdir": os.fspath(temporary),
    }


def _import_path_binding(code_root: Path) -> dict[str, Any]:
    entries: list[str] = []
    for index, value in enumerate(sys.path):
        _require(isinstance(value, str) and value, f"sys.path entry {index} is empty")
        _require(Path(value).is_absolute(), f"sys.path entry {index} is relative")
        path = _absolute(value)
        _require(not _is_formal_held_path(path), "sys.path enters a formal held root")
        _require(
            "/.local/" not in os.fspath(path), "sys.path enters user site-packages"
        )
        entries.append(os.fspath(path))
    _require(len(entries) == len(set(entries)), "sys.path contains duplicate entries")
    code_source = os.fspath((code_root / "src").resolve(strict=True))
    expected_entries = [
        code_source,
        "/usr/lib/python312.zip",
        "/usr/lib/python3.12",
        "/usr/lib/python3.12/lib-dynload",
        os.fspath(PINNED_PYTHON_RUNTIME / "lib/python3.12/site-packages"),
    ]
    _require(
        entries == expected_entries,
        "analyzer sys.path differs from the exact sanitized path",
    )
    site_packages = (PINNED_PYTHON_RUNTIME / "lib/python3.12/site-packages").resolve(
        strict=True
    )
    modules: dict[str, Any] = {}
    for name in ("numpy", "scipy", "torch", "gsplat"):
        module = importlib.import_module(name)
        file_value = getattr(module, "__file__", None)
        _require(
            isinstance(file_value, str) and file_value, f"{name} has no source path"
        )
        path = Path(file_value).resolve(strict=True)
        _require(
            path == site_packages or site_packages in path.parents,
            f"{name} imported outside the pinned runtime",
        )
        modules[name] = _bound_file(path, label=f"{name} import source")
    return {"sys_path": entries, "modules": modules}


def _install_controlled_code_source(code_root: Path) -> None:
    source = os.fspath((code_root / "src").resolve(strict=True))
    _require(source not in sys.path, "analyzer source root was already injected")
    sys.path.insert(0, source)


def _execution_binding(
    code_root: Path,
    *,
    physical_gpu_index: int,
    torch: Any,
) -> dict[str, Any]:
    return {
        "host": _validate_execution_host(),
        "process_flags": _validate_process_flags(),
        "environment": _validate_execution_environment(physical_gpu_index),
        "live_device": _validate_live_torch_device(torch),
        "imports": _import_path_binding(code_root),
    }


def _validate_gsplat_smoke(
    value: Any,
    *,
    label: str,
    expected_physical_gpu_index: int,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    smoke = dict(value)
    _require(
        set(smoke)
        == {
            "schema_version",
            "artifact_kind",
            "contract_sha256",
            "physical_gpu_index",
            "logical_device",
            "gpu_name",
            "compute_capability",
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "gsplat_version",
            "extension_path",
            "extension_sha256",
            "extension_loaded_and_retained",
            "nvcc_visible",
            "ninja_visible",
            "target_or_outcome_path_accessed",
            "predicates",
            "artifact_sha256",
        },
        f"{label} fields changed",
    )
    _require(
        smoke.get("artifact_sha256") == _artifact_sha256(smoke),
        f"{label} signature is invalid",
    )
    _require(
        _strict_int(smoke.get("schema_version"), label=f"{label} schema") == 1
        and smoke.get("artifact_kind") == "Deform360HeldGsplatRuntimeSmokeV1"
        and smoke.get("contract_sha256") == PINNED_GSPLAT_SMOKE_CONTRACT_SHA256
        and isinstance(smoke.get("physical_gpu_index"), int)
        and not isinstance(smoke.get("physical_gpu_index"), bool)
        and smoke.get("physical_gpu_index") == expected_physical_gpu_index
        and smoke.get("logical_device") == "cuda:0"
        and smoke.get("gpu_name") == PINNED_GPU_NAME
        and smoke.get("compute_capability") == "8.9"
        and smoke.get("python_version") == "3.12"
        and smoke.get("torch_version") == PINNED_TORCH_VERSION
        and smoke.get("torch_cuda_version") == PINNED_TORCH_CUDA_VERSION
        and smoke.get("gsplat_version") == PINNED_GSPLAT_VERSION
        and smoke.get("extension_path") == os.fspath(PINNED_GSPLAT_EXTENSION_PATH)
        and smoke.get("extension_sha256") == PINNED_GSPLAT_EXTENSION_SHA256
        and smoke.get("extension_loaded_and_retained") is True
        and smoke.get("nvcc_visible") is False
        and smoke.get("ninja_visible") is False
        and smoke.get("target_or_outcome_path_accessed") is False,
        f"{label} frozen runtime fields changed",
    )
    predicates = smoke.get("predicates")
    _require(
        isinstance(predicates, Mapping)
        and set(predicates)
        == {
            "render_shape",
            "alpha_shape",
            "positive_radius_count",
            "gradient_groups_finite_and_nonzero",
            "forward_finite_nonempty_nonzero",
            "backward_complete",
            "cuda_synchronized",
        }
        and predicates.get("render_shape") == [1, 16, 16, 3]
        and predicates.get("alpha_shape") == [1, 16, 16, 1]
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in [
                *predicates.get("render_shape", []),
                *predicates.get("alpha_shape", []),
            ]
        )
        and _strict_int(
            predicates.get("positive_radius_count"),
            label=f"{label} positive radius count",
        )
        == 2
        and predicates.get("gradient_groups_finite_and_nonzero")
        == ["colors", "means", "opacities", "quats", "scales"]
        and predicates.get("forward_finite_nonempty_nonzero") is True
        and predicates.get("backward_complete") is True
        and predicates.get("cuda_synchronized") is True,
        f"{label} smoke predicates changed",
    )
    _require(
        dict(predicates)
        == {
            "render_shape": [1, 16, 16, 3],
            "alpha_shape": [1, 16, 16, 1],
            "positive_radius_count": 2,
            "gradient_groups_finite_and_nonzero": [
                "colors",
                "means",
                "opacities",
                "quats",
                "scales",
            ],
            "forward_finite_nonempty_nonzero": True,
            "backward_complete": True,
            "cuda_synchronized": True,
        },
        f"{label} smoke predicates changed",
    )
    return smoke


def _strict_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} is not a strict integer",
    )
    if minimum is not None:
        _require(value >= minimum, f"{label} is below its minimum")
    return value


def _validate_resource_boundary(value: Any, *, label: str) -> dict[str, int]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(
        set(value)
        == {
            "file_descriptor_count",
            "task_count",
            "rss_kib",
            "rlimit_nofile_soft",
            "rlimit_nofile_hard",
        },
        f"{label} fields changed",
    )
    result = {
        "file_descriptor_count": _strict_int(
            value["file_descriptor_count"], label=f"{label} FD count", minimum=0
        ),
        "task_count": _strict_int(
            value["task_count"], label=f"{label} task count", minimum=1
        ),
        "rss_kib": _strict_int(value["rss_kib"], label=f"{label} RSS", minimum=1),
        "rlimit_nofile_soft": _strict_int(
            value["rlimit_nofile_soft"], label=f"{label} soft limit", minimum=1
        ),
        "rlimit_nofile_hard": _strict_int(
            value["rlimit_nofile_hard"], label=f"{label} hard limit", minimum=1
        ),
    }
    _require(
        result["rlimit_nofile_hard"] >= result["rlimit_nofile_soft"],
        f"{label} file limit ordering changed",
    )
    return result


def _validate_global_snapshot(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(
        set(value)
        == {
            "event_writers_object_id",
            "event_writer_ids",
            "event_storage_object_id",
            "event_storage_ids",
            "global_buffer_object_id",
            "global_buffer_items",
            "profiler_object_id",
            "profiler_ids",
            "pytorch_profiler_id",
        },
        f"{label} fields changed",
    )
    for name in (
        "event_writers_object_id",
        "event_storage_object_id",
        "global_buffer_object_id",
        "profiler_object_id",
    ):
        _strict_int(value[name], label=f"{label} {name}", minimum=1)
    for name in (
        "event_writer_ids",
        "event_storage_ids",
        "profiler_ids",
    ):
        entries = value[name]
        _require(isinstance(entries, list), f"{label} {name} is not a list")
        for index, item in enumerate(entries):
            _strict_int(item, label=f"{label} {name} {index}", minimum=1)
    buffer_items = value["global_buffer_items"]
    _require(isinstance(buffer_items, list), f"{label} global buffer is not a list")
    for index, item in enumerate(buffer_items):
        _require(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and item[0],
            f"{label} global buffer item {index} changed",
        )
        _strict_int(item[1], label=f"{label} global buffer ID {index}", minimum=1)
    pytorch_profiler = value["pytorch_profiler_id"]
    _require(
        pytorch_profiler is None
        or (
            isinstance(pytorch_profiler, int)
            and not isinstance(pytorch_profiler, bool)
            and pytorch_profiler > 0
        ),
        f"{label} PyTorch profiler ID changed",
    )
    return dict(value)


def _validate_fit_evidence(
    descriptor: Any,
    *,
    mode: str,
    ply_binding: Mapping[str, Any],
    expected_adapter_path: Path,
    generator_profile: str,
    expected_physical_gpu_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(mode in {"original", "wrapped"}, "fit evidence mode is invalid")
    _require(generator_profile in FIT_QUALIFICATION_IDS, "generator profile is invalid")
    _require(
        expected_physical_gpu_index == PROFILE_PHYSICAL_GPU_INDEX[generator_profile],
        "fit evidence GPU differs from the generator profile",
    )
    evidence_binding = _verified_descriptor(descriptor, label=f"{mode} fit evidence")
    evidence = _load_signed_json(evidence_binding["path"], label=f"{mode} fit evidence")
    _require(
        set(evidence)
        == {
            "schema_version",
            "artifact_kind",
            "qualification_id",
            "variant",
            "passed",
            "parameters",
            "runtime",
            "gsplat_runtime_smoke",
            "dataset",
            "output",
            "resource_boundary",
            "global_state",
            "predicates",
            "formal_held_path_supplied",
            "artifact_sha256",
        },
        f"{mode} fit evidence fields changed",
    )
    _require(
        _strict_int(evidence.get("schema_version"), label=f"{mode} schema") == 1
        and evidence.get("artifact_kind") == FIT_EVIDENCE_KIND
        and evidence.get("qualification_id")
        == FIT_QUALIFICATION_IDS[generator_profile],
        f"{mode} fit evidence identity changed",
    )
    _require(evidence.get("variant") == mode, f"{mode} fit variant changed")
    _require(evidence.get("passed") is True, f"{mode} fit did not pass")
    parameters = evidence.get("parameters")
    _require(
        isinstance(parameters, Mapping)
        and set(parameters) == {"iterations", "seed"}
        and _strict_int(parameters["iterations"], label=f"{mode} iterations")
        == FIT_ITERATIONS
        and _strict_int(parameters["seed"], label=f"{mode} seed") == FIT_SEED,
        f"{mode} fit parameters changed",
    )
    _require(
        evidence.get("formal_held_path_supplied") is False,
        f"{mode} fit crossed the formal boundary",
    )
    runtime = evidence.get("runtime")
    _require(
        isinstance(runtime, Mapping)
        and set(runtime)
        == {
            "seed",
            "python_random_seeded",
            "numpy_seeded",
            "torch_cpu_seeded",
            "torch_cuda_seeded",
            "torch_version",
            "torch_cuda_version",
            "cuda_device_name",
            "cuda_device_count",
            "python_version",
        },
        f"{mode} fit runtime fields changed",
    )
    _require(
        _strict_int(runtime.get("seed"), label=f"{mode} runtime seed") == FIT_SEED
        and runtime.get("python_random_seeded") is True
        and runtime.get("numpy_seeded") is True
        and runtime.get("torch_cpu_seeded") is True
        and runtime.get("torch_cuda_seeded") is True
        and runtime.get("torch_version") == PINNED_TORCH_VERSION
        and runtime.get("torch_cuda_version") == PINNED_TORCH_CUDA_VERSION
        and runtime.get("cuda_device_name") == PINNED_GPU_NAME
        and _strict_int(
            runtime.get("cuda_device_count"), label=f"{mode} CUDA device count"
        )
        == 1
        and isinstance(runtime.get("python_version"), str)
        and str(runtime.get("python_version")).startswith("3.12."),
        f"{mode} fit runtime changed",
    )
    gsplat = evidence.get("gsplat_runtime_smoke")
    _require(
        isinstance(gsplat, Mapping)
        and set(gsplat) == {"adapter_source", "evidence", "evidence_artifact_sha256"},
        f"{mode} gsplat wrapper fields changed",
    )
    adapter = _verified_bound_record(
        gsplat.get("adapter_source"), label=f"{mode} gsplat adapter source"
    )
    expected_adapter_sha = GENERATOR_SOURCE_BINDINGS[
        "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    ]["sha256"]
    _require(
        adapter["path"] == os.fspath(expected_adapter_path)
        and adapter["sha256"] == expected_adapter_sha,
        "generator gsplat adapter path or content changed",
    )
    smoke = _validate_gsplat_smoke(
        gsplat.get("evidence"),
        label=f"{mode} gsplat smoke",
        expected_physical_gpu_index=expected_physical_gpu_index,
    )
    _require(
        gsplat.get("evidence_artifact_sha256") == smoke.get("artifact_sha256"),
        f"{mode} gsplat smoke signature is invalid",
    )
    predicates = evidence.get("predicates")
    _require(
        isinstance(predicates, Mapping)
        and set(predicates)
        == {
            "output_created",
            "wrapped_fit_requires_global_restoration",
            "rlimit_nofile_soft_is_1024",
            "rlimit_nofile_unchanged",
            "gsplat_runtime_smoke_validated_and_retained",
        }
        and all(value is True for value in predicates.values()),
        f"{mode} fit predicates changed",
    )
    resource = evidence.get("resource_boundary")
    _require(
        isinstance(resource, Mapping) and set(resource) == {"before", "after"},
        f"{mode} resource boundary fields changed",
    )
    before = _validate_resource_boundary(
        resource.get("before"), label=f"{mode} resource before"
    )
    after = _validate_resource_boundary(
        resource.get("after"), label=f"{mode} resource after"
    )
    _require(
        isinstance(before, Mapping)
        and isinstance(after, Mapping)
        and before.get("rlimit_nofile_soft") == 1024
        and after.get("rlimit_nofile_soft") == before.get("rlimit_nofile_soft")
        and after.get("rlimit_nofile_hard") == before.get("rlimit_nofile_hard"),
        f"{mode} resource boundary changed",
    )
    global_state = evidence.get("global_state")
    _require(
        isinstance(global_state, Mapping)
        and set(global_state) == {"before", "after", "restored"}
        and isinstance(global_state.get("restored"), bool),
        f"{mode} global state fields changed",
    )
    _validate_global_snapshot(
        global_state.get("before"), label=f"{mode} globals before"
    )
    _validate_global_snapshot(global_state.get("after"), label=f"{mode} globals after")
    _require(
        mode != "wrapped" or global_state.get("restored") is True,
        "wrapped fit did not restore globals",
    )
    dataset = evidence.get("dataset")
    _require(isinstance(dataset, str) and dataset, f"{mode} dataset is absent")
    dataset_path = _assert_nonheld_path(
        dataset, label=f"{mode} dataset", must_exist=True
    )
    _require(dataset_path.is_dir(), f"{mode} dataset is not a directory")
    output = _verified_bound_record(evidence.get("output"), label=f"{mode} fit output")
    _require(
        all(
            output[key] == ply_binding[key] for key in ("path", "size_bytes", "sha256")
        ),
        f"{mode} evidence output differs from manifest PLY",
    )
    return evidence_binding, evidence


def _strict_dataset_input(root: Path, value: str, *, label: str) -> tuple[Path, Path]:
    raw = Path(value)
    if raw.is_absolute():
        absolute = _absolute(raw)
        try:
            relative = absolute.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} escapes the dataset") from error
    else:
        normalized = Path(os.path.normpath(os.fspath(raw)))
        _require(
            normalized.parts
            and normalized.parts[0] not in ("", ".", "..")
            and ".." not in normalized.parts,
            f"{label} escapes the dataset",
        )
        relative = normalized
        absolute = root / relative
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            entry = os.lstat(current)
        except OSError as error:
            raise ValueError(f"{label} is unavailable: {current}") from error
        _require(not stat.S_ISLNK(entry.st_mode), f"{label} contains a symlink")
        if index + 1 < len(relative.parts):
            _require(stat.S_ISDIR(entry.st_mode), f"{label} parent is not a directory")
        else:
            _require(stat.S_ISREG(entry.st_mode), f"{label} is not a regular file")
            _require(entry.st_nlink == 1, f"{label} is a hardlink")
    resolved = absolute.resolve(strict=True)
    _require(resolved == absolute, f"{label} is aliased")
    return relative, resolved


def _strict_materialized_seed_input(
    root: Path,
    value: str,
    *,
    label: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve a seed, permitting only an exact materialized canonical alias."""
    raw = Path(value)
    if not raw.is_absolute():
        relative, resolved = _strict_dataset_input(root, value, label=label)
        return (
            relative,
            resolved,
            {
                "declared_path": value,
                "canonical_absolute_alias_used": False,
            },
        )

    declared = _absolute(raw)
    try:
        declared.relative_to(root)
    except ValueError:
        canonical_root = _assert_nonheld_path(
            CANONICAL_PUBLIC_SOURCE_DATASET,
            label="canonical public source dataset",
            must_exist=True,
        )
        _require(
            root != canonical_root,
            f"{label} escapes the canonical dataset",
        )
        try:
            relative = declared.relative_to(canonical_root)
        except ValueError as error:
            raise ValueError(f"{label} escapes the dataset") from error
        canonical_relative, canonical_path = _strict_dataset_input(
            canonical_root,
            os.fspath(declared),
            label=f"{label} canonical target",
        )
        _require(
            canonical_relative == relative,
            f"{label} canonical target changed",
        )
        materialized_relative, materialized_path = _strict_dataset_input(
            root,
            relative.as_posix(),
            label=f"{label} materialized copy",
        )
        canonical_binding = _bound_file(
            canonical_path,
            label=f"{label} canonical target",
        )
        materialized_binding = _bound_file(
            materialized_path,
            label=f"{label} materialized copy",
        )
        _require(
            all(
                canonical_binding[key] == materialized_binding[key]
                for key in ("size_bytes", "sha256")
            ),
            f"{label} materialized copy differs from canonical target",
        )
        return (
            materialized_relative,
            materialized_path,
            {
                "declared_path": os.fspath(declared),
                "canonical_absolute_alias_used": True,
                "canonical_target": canonical_binding,
                "materialized_copy": materialized_binding,
            },
        )

    relative, resolved = _strict_dataset_input(root, value, label=label)
    return (
        relative,
        resolved,
        {
            "declared_path": os.fspath(declared),
            "canonical_absolute_alias_used": False,
        },
    )


def _normalized_transforms_descriptor(payload: bytes) -> dict[str, Any]:
    try:
        transforms = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset transforms are not UTF-8 JSON") from error
    _require(isinstance(transforms, dict), "dataset transforms are not an object")
    portable = json.loads(json.dumps(transforms, allow_nan=False))
    portable["ply_file_path"] = "<MATERIALIZED-SEED-PLY>"
    normalized_bytes = _canonical_bytes(portable)
    return {
        "size_bytes": len(normalized_bytes),
        "sha256": hashlib.sha256(normalized_bytes).hexdigest(),
    }


def _dataset_input_inventory(dataset: str | Path) -> dict[str, Any]:
    root = _assert_nonheld_path(dataset, label="fit dataset", must_exist=True)
    _require(root.is_dir() and not root.is_symlink(), "fit dataset is not a directory")
    transforms_relative, transforms_path = _strict_dataset_input(
        root, "transforms.json", label="dataset transforms"
    )
    transforms_bytes = _read_regular_nofollow(
        transforms_path, label="dataset transforms"
    )
    try:
        transforms = json.loads(transforms_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset transforms are not UTF-8 JSON") from error
    _require(isinstance(transforms, dict), "dataset transforms are not an object")
    seed_value = transforms.get("ply_file_path")
    _require(isinstance(seed_value, str) and seed_value, "dataset seed PLY is absent")
    frames = transforms.get("frames")
    _require(isinstance(frames, list) and frames, "dataset frames are absent")
    references: list[tuple[str, Path, Path]] = []
    seed_relative, seed_path, seed_reference = _strict_materialized_seed_input(
        root, seed_value, label="dataset seed PLY"
    )
    references.append(("seed_ply", seed_relative, seed_path))
    for index, frame in enumerate(frames):
        _require(isinstance(frame, Mapping), f"dataset frame {index} is not an object")
        frame_value = frame.get("file_path")
        _require(
            isinstance(frame_value, str) and frame_value,
            f"dataset frame {index} path is absent",
        )
        relative, path = _strict_dataset_input(
            root, frame_value, label=f"dataset frame {index} image"
        )
        references.append(("frame_image", relative, path))
    relative_names = [relative.as_posix() for _, relative, _ in references]
    _require(
        len(relative_names) == len(set(relative_names)), "dataset input path repeats"
    )
    inodes: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    input_bindings: list[dict[str, Any]] = []
    for role, relative, path in sorted(
        references, key=lambda item: (item[1].as_posix(), item[0])
    ):
        identity = os.lstat(path)
        inode = (identity.st_dev, identity.st_ino)
        _require(inode not in inodes, "dataset input has a duplicate inode alias")
        inodes.add(inode)
        binding = _bound_file(path, label=f"dataset input {relative.as_posix()}")
        rows.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                "size_bytes": binding["size_bytes"],
                "sha256": binding["sha256"],
            }
        )
        input_bindings.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                **binding,
            }
        )
    normalized = _normalized_transforms_descriptor(transforms_bytes)
    content_identity = {
        "normalized_transforms": normalized,
        "referenced_files": rows,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "Deform360DeclaredDatasetInputClosureV1",
        "root": os.fspath(root),
        "raw_transforms": _bound_file(transforms_path, label="raw dataset transforms"),
        "transforms_relative_path": transforms_relative.as_posix(),
        "seed_relative_path": seed_relative.as_posix(),
        "seed_reference": seed_reference,
        "frame_count": len(frames),
        "regular_input_file_count": len(rows) + 1,
        "referenced_input_bindings": input_bindings,
        "content_identity": content_identity,
        "content_artifact_sha256": hashlib.sha256(
            _canonical_bytes(content_identity)
        ).hexdigest(),
        "generated_outputs_excluded": True,
        "symlinks_special_files_and_hardlink_aliases_accepted": False,
    }


@dataclass(frozen=True)
class RepeatInput:
    mode: str
    pairing_id: str
    path: Path
    binding: Mapping[str, Any]
    fit_evidence_binding: Mapping[str, Any]
    fit_evidence: Mapping[str, Any]


def _validate_git_manifest_binding(value: Any, *, label: str) -> Path:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(
        set(value)
        == {
            "path",
            "head",
            "tree",
            "clean",
            "ordinary_untracked_file_count",
            "ignored_untracked_file_count",
        },
        f"{label} fields changed",
    )
    root_value = value.get("path")
    _require(isinstance(root_value, str) and root_value, f"{label} path is absent")
    root = _assert_nonheld_path(root_value, label=f"{label} root", must_exist=True)
    _require(
        root.is_dir() and not root.is_symlink(), f"{label} root is not a directory"
    )
    _git_object_id(value.get("head"), label=f"{label} head")
    _git_object_id(value.get("tree"), label=f"{label} tree")
    _require(value.get("clean") is True, f"{label} is not clean")
    _require(
        _strict_int(
            value.get("ordinary_untracked_file_count"),
            label=f"{label} ordinary untracked count",
            minimum=0,
        )
        == 0
        and _strict_int(
            value.get("ignored_untracked_file_count"),
            label=f"{label} ignored untracked count",
            minimum=0,
        )
        == 0,
        f"{label} has untracked files",
    )
    return root


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    environment_only: bool = False,
) -> tuple[dict[str, list[RepeatInput]], Path, Mapping[str, Any]] | Mapping[str, Any]:
    _require(
        set(manifest)
        == {
            "schema_version",
            "artifact_kind",
            "analysis_id",
            "expected_environment",
            "canonical_source_dataset",
            "canonical_transforms",
            "modes",
            "artifact_sha256",
        },
        "manifest fields changed",
    )
    _require(
        _strict_int(manifest.get("schema_version"), label="manifest schema") == 1,
        "manifest schema changed",
    )
    _require(manifest.get("artifact_kind") == MANIFEST_KIND, "manifest kind changed")
    _require(manifest.get("analysis_id") == ANALYSIS_ID, "analysis identity changed")
    expected = manifest.get("expected_environment")
    _require(isinstance(expected, Mapping), "expected environment is absent")
    _require(
        set(expected)
        == {
            "generator_profile",
            "physical_gpu_index",
            "generator_code",
            "analyzer_code",
            "deform360_git_head",
            "deform360_git_tree",
            "python_freeze_sha256",
            "python_tree_manifest_sha256",
        },
        "expected environment fields changed",
    )
    generator = expected.get("generator_code")
    analyzer = expected.get("analyzer_code")
    profile = expected.get("generator_profile")
    _require(profile in FIT_QUALIFICATION_IDS, "manifest generator profile changed")
    physical_gpu_index = _strict_int(
        expected.get("physical_gpu_index"), label="manifest physical GPU", minimum=0
    )
    _require(
        physical_gpu_index == PROFILE_PHYSICAL_GPU_INDEX[str(profile)],
        "manifest physical GPU differs from generator profile",
    )
    _require(isinstance(generator, Mapping), "generator code binding is absent")
    _require(isinstance(analyzer, Mapping), "analyzer code binding is absent")
    _require(
        set(generator)
        == {"profile", "qualification_id", "physical_gpu_index", "git", "sources"},
        "generator code fields changed",
    )
    _require(
        generator.get("profile") == profile
        and generator.get("qualification_id") == FIT_QUALIFICATION_IDS[str(profile)]
        and _strict_int(
            generator.get("physical_gpu_index"), label="generator physical GPU"
        )
        == physical_gpu_index,
        "generator profile binding changed",
    )
    generator_git = generator.get("git")
    generator_root = _validate_git_manifest_binding(
        generator_git, label="generator code Git"
    )
    analyzer_root = _validate_git_manifest_binding(analyzer, label="analyzer code Git")
    if profile == "historical-0db":
        _require(
            generator_git.get("head") == GENERATOR_CODE_HEAD
            and generator_git.get("tree") == GENERATOR_CODE_TREE,
            "historical generator code identity changed",
        )
    else:
        _require(
            generator_root == analyzer_root and generator_git == analyzer,
            "same-as-analyzer generator identity differs from analyzer",
        )
    generator_sources = generator.get("sources")
    _require(isinstance(generator_sources, Mapping), "generator sources are absent")
    _require(
        set(generator_sources) == set(GENERATOR_SOURCE_BINDINGS),
        "generator source set changed",
    )
    for path, pinned in GENERATOR_SOURCE_BINDINGS.items():
        observed = generator_sources.get(path)
        _require(isinstance(observed, Mapping), f"generator source is absent: {path}")
        _require(
            set(observed)
            == {"path", "size_bytes", "sha256", "mode_octal", "git_blob_oid"},
            f"generator source fields changed: {path}",
        )
        expected_path = (generator_root / path).resolve(strict=True)
        _require(
            observed.get("path") == os.fspath(expected_path)
            and isinstance(observed.get("size_bytes"), int)
            and not isinstance(observed.get("size_bytes"), bool)
            and isinstance(observed.get("mode_octal"), str)
            and len(observed.get("mode_octal")) == 4
            and all(
                character in "01234567" for character in str(observed.get("mode_octal"))
            ),
            f"generator source binding changed: {path}",
        )
        _hex_digest(observed.get("sha256"), label=f"generator source {path}")
        _git_object_id(observed.get("git_blob_oid"), label=f"generator source {path}")
        if path in CORE_GENERATOR_SOURCE_PATHS or profile == "historical-0db":
            _require(
                observed.get("sha256") == pinned["sha256"]
                and observed.get("size_bytes") == pinned["size_bytes"]
                and observed.get("git_blob_oid") == pinned["git_blob_oid"],
                f"pinned generator source changed: {path}",
            )
    expected_adapter_path = (
        generator_root / "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    ).resolve(strict=True)
    _require(
        _git_object_id(expected.get("deform360_git_head"), label="Deform360 Git head")
        == PINNED_DEFORM360_REVISION,
        "Deform360 revision is not pinned",
    )
    _require(
        _git_object_id(expected.get("deform360_git_tree"), label="Deform360 Git tree")
        == PINNED_DEFORM360_TREE,
        "Deform360 tree is not pinned",
    )
    _require(
        _hex_digest(expected.get("python_freeze_sha256"), label="Python freeze")
        == PINNED_PYTHON_FREEZE_SHA256,
        "Python package inventory is not pinned",
    )
    _require(
        _hex_digest(
            expected.get("python_tree_manifest_sha256"),
            label="Python tree manifest",
        )
        == PINNED_PYTHON_TREE_MANIFEST_SHA256,
        "Python runtime tree is not pinned",
    )
    if environment_only:
        return expected

    source_value = manifest.get("canonical_source_dataset")
    _require(isinstance(source_value, Mapping), "canonical source dataset is absent")
    canonical_source_root = _assert_nonheld_path(
        CANONICAL_PUBLIC_SOURCE_DATASET,
        label="canonical public source dataset",
        must_exist=True,
    )
    _require(
        canonical_source_root == CANONICAL_PUBLIC_SOURCE_DATASET,
        "canonical public source dataset path is aliased",
    )
    canonical_source = _dataset_input_inventory(canonical_source_root)
    _require(
        source_value == canonical_source,
        "canonical public source dataset identity changed",
    )

    transforms_value = manifest.get("canonical_transforms")
    _require(isinstance(transforms_value, Mapping), "canonical transforms are absent")
    _require(
        set(transforms_value) == {"raw_representative", "normalized"},
        "canonical transforms fields changed",
    )
    transforms_binding = _verified_descriptor(
        transforms_value.get("raw_representative"), label="canonical source transforms"
    )
    transforms_path = Path(str(transforms_binding["path"]))
    source_transforms = canonical_source["raw_transforms"]
    _require(
        all(
            transforms_binding[key] == source_transforms[key]
            for key in ("path", "size_bytes", "sha256")
        ),
        "canonical render transforms are not the public source transforms",
    )
    normalized_transforms = _normalized_transforms_descriptor(
        _read_regular_nofollow(
            transforms_path, label="canonical transforms representative"
        )
    )
    _require(
        transforms_value.get("normalized") == normalized_transforms,
        "canonical normalized transforms changed",
    )

    modes = manifest.get("modes")
    _require(isinstance(modes, Mapping), "manifest modes are absent")
    _require(set(modes) == {"original", "wrapped"}, "manifest modes changed")
    parsed: dict[str, list[RepeatInput]] = {}
    pairing_sets: dict[str, set[str]] = {}
    all_paths: set[Path] = {transforms_path}
    repeat_inodes: set[tuple[int, int]] = set()
    for mode in ("original", "wrapped"):
        records = modes.get(mode)
        _require(isinstance(records, list), f"{mode} records are not a list")
        _require(
            len(records) >= MINIMUM_REPEATS_PER_MODE,
            f"{mode} has fewer than {MINIMUM_REPEATS_PER_MODE} repeats",
        )
        mode_inputs: list[RepeatInput] = []
        identifiers: set[str] = set()
        for index, record in enumerate(records):
            label = f"{mode} repeat {index}"
            _require(isinstance(record, Mapping), f"{label} is not an object")
            _require(
                set(record)
                == {"pairing_id", "ply", "fit_evidence", "dataset_input_inventory"},
                f"{label} fields changed",
            )
            pairing_id = record.get("pairing_id")
            _require(
                isinstance(pairing_id, str)
                and pairing_id
                and pairing_id.isascii()
                and all(
                    character.isalnum() or character in "-_."
                    for character in pairing_id
                ),
                f"{label} pairing ID is invalid",
            )
            _require(pairing_id not in identifiers, f"duplicate {mode} pairing ID")
            identifiers.add(pairing_id)
            binding = _verified_descriptor(record.get("ply"), label=f"{label} PLY")
            path = Path(str(binding["path"]))
            _require(path.suffix.lower() == ".ply", f"{label} is not a PLY")
            _require(path not in all_paths, "an input path is reused")
            all_paths.add(path)
            path_stat = os.lstat(path)
            path_inode = (path_stat.st_dev, path_stat.st_ino)
            _require(
                path_inode not in repeat_inodes, "a repeat PLY/evidence inode is reused"
            )
            repeat_inodes.add(path_inode)
            fit_binding, fit_evidence = _validate_fit_evidence(
                record.get("fit_evidence"),
                mode=mode,
                ply_binding=binding,
                expected_adapter_path=expected_adapter_path,
                generator_profile=str(profile),
                expected_physical_gpu_index=physical_gpu_index,
            )
            fit_path = Path(str(fit_binding["path"]))
            _require(fit_path not in all_paths, "an input path is reused")
            all_paths.add(fit_path)
            fit_stat = os.lstat(fit_path)
            fit_inode = (fit_stat.st_dev, fit_stat.st_ino)
            _require(
                fit_inode not in repeat_inodes, "a repeat PLY/evidence inode is reused"
            )
            repeat_inodes.add(fit_inode)
            inventory = _dataset_input_inventory(str(fit_evidence["dataset"]))
            _require(
                record.get("dataset_input_inventory") == inventory,
                f"{label} dataset input inventory changed",
            )
            _require(
                inventory["content_identity"] == canonical_source["content_identity"]
                and inventory["content_artifact_sha256"]
                == canonical_source["content_artifact_sha256"],
                f"{label} differs from the canonical public source after only "
                "absolute seed-path normalization",
            )
            mode_inputs.append(
                RepeatInput(
                    mode=mode,
                    pairing_id=pairing_id,
                    path=path,
                    binding=binding,
                    fit_evidence_binding=fit_binding,
                    fit_evidence=fit_evidence,
                )
            )
        parsed[mode] = sorted(mode_inputs, key=lambda item: item.pairing_id)
        pairing_sets[mode] = identifiers
    _require(
        pairing_sets["original"] == pairing_sets["wrapped"],
        "original and wrapped pairing IDs differ",
    )
    total_repeats = sum(len(records) for records in parsed.values())
    expected_inode_count = 2 * total_repeats
    _require(
        len(repeat_inodes) == expected_inode_count,
        "repeat PLY/evidence files are not all inode-distinct",
    )
    return parsed, transforms_path, expected


def _validate_manifest_environment(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = _validate_manifest(manifest, environment_only=True)
    _require(isinstance(expected, Mapping), "manifest environment preflight failed")
    return expected


def _read_header_line(payload: bytes, offset: int) -> tuple[bytes, int]:
    newline = payload.find(b"\n", offset)
    _require(newline >= 0, "PLY header is unterminated")
    _require(newline + 1 <= 1024 * 1024, "PLY header exceeds one MiB")
    return payload[offset:newline].rstrip(b"\r"), newline + 1


@dataclass(frozen=True)
class GaussianCloud:
    pairing_id: str
    mode: str
    binding: Mapping[str, Any]
    schema: tuple[tuple[str, str], ...]
    vertices: np.ndarray
    xyz: np.ndarray
    opacity_logits: np.ndarray
    opacity_probability: np.ndarray
    log_scales: np.ndarray
    raw_quaternions: np.ndarray
    quaternions: np.ndarray
    sh: np.ndarray


def _parse_gaussian_ply(
    payload: bytes,
    *,
    mode: str,
    pairing_id: str,
    binding: Mapping[str, Any],
) -> GaussianCloud:
    offset = 0
    line, offset = _read_header_line(payload, offset)
    _require(line == b"ply", "PLY magic changed")
    line, offset = _read_header_line(payload, offset)
    _require(
        line == b"format binary_little_endian 1.0",
        "PLY must be binary_little_endian 1.0",
    )
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    current_element: str | None = None
    while True:
        line, offset = _read_header_line(payload, offset)
        try:
            text = line.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("PLY header is not ASCII") from error
        if text == "end_header":
            break
        if not text or text.startswith("comment ") or text.startswith("obj_info "):
            continue
        fields = text.split()
        if fields[:1] == ["element"]:
            _require(len(fields) == 3, "malformed PLY element")
            _require(vertex_count is None, "PLY has more than one element")
            _require(fields[1] == "vertex", "PLY has a non-vertex element")
            _require(fields[2].isdigit(), "PLY vertex count is invalid")
            vertex_count = int(fields[2])
            _require(vertex_count > 0, "PLY vertex element is empty")
            current_element = "vertex"
            continue
        if fields[:1] == ["property"]:
            _require(current_element == "vertex", "PLY property precedes vertex")
            _require(len(fields) == 3 and fields[1] != "list", "bad PLY property")
            scalar_type, name = fields[1], fields[2]
            _require(scalar_type in _PLY_SCALAR_DTYPES, "unknown PLY scalar type")
            properties.append((name, scalar_type))
            continue
        raise ValueError(f"unsupported PLY header directive: {text}")
    _require(vertex_count is not None, "PLY has no vertex element")
    names = tuple(name for name, _ in properties)
    _require(len(names) == 62, "PLY does not have exactly 62 fields")
    _require(names == EXPECTED_PLY_FIELDS, "PLY field names or order changed")
    _require(len(set(names)) == len(names), "PLY has duplicate fields")
    _require(
        all(kind == "float" for _, kind in properties),
        "Gaussian PLY fields must all be literal float/f4",
    )
    dtype = np.dtype(
        [(name, _PLY_SCALAR_DTYPES[scalar_type]) for name, scalar_type in properties],
        align=False,
    )
    expected_bytes = vertex_count * dtype.itemsize
    _require(len(payload) - offset == expected_bytes, "PLY binary payload size changed")
    vertices = np.frombuffer(
        payload, dtype=dtype, count=vertex_count, offset=offset
    ).copy()
    for name in names:
        _require(
            bool(np.all(np.isfinite(vertices[name]))), f"PLY field {name} is non-finite"
        )
    for name in ("nx", "ny", "nz"):
        _require(
            bool(np.all(vertices[name] == np.float32(0.0))),
            f"PLY inert normal field {name} is not exactly zero",
        )
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    opacity_logit = np.asarray(vertices["opacity"], dtype=np.float64)
    opacity = np.empty_like(opacity_logit)
    positive = opacity_logit >= 0.0
    opacity[positive] = 1.0 / (1.0 + np.exp(-opacity_logit[positive]))
    exponential = np.exp(opacity_logit[~positive])
    opacity[~positive] = exponential / (1.0 + exponential)
    log_scales = np.column_stack(
        [vertices[f"scale_{index}"] for index in range(3)]
    ).astype(np.float64)
    raw_quaternions = np.column_stack(
        [vertices[f"rot_{index}"] for index in range(4)]
    ).astype(np.float64)
    norms = np.linalg.norm(raw_quaternions, axis=1)
    _require(bool(np.all(np.isfinite(norms) & (norms > 0.0))), "zero quaternion")
    quaternions = raw_quaternions / norms[:, None]
    sh = np.column_stack([vertices[name] for name in SH_FIELDS]).astype(np.float64)
    _require(
        all(
            bool(np.all(np.isfinite(value)))
            for value in (xyz, opacity, log_scales, quaternions, sh)
        ),
        "derived Gaussian fields are non-finite",
    )
    return GaussianCloud(
        pairing_id=pairing_id,
        mode=mode,
        binding=dict(binding),
        schema=tuple(properties),
        vertices=vertices,
        xyz=xyz,
        opacity_logits=opacity_logit,
        opacity_probability=opacity,
        log_scales=log_scales,
        raw_quaternions=raw_quaternions,
        quaternions=quaternions,
        sh=sh,
    )


@dataclass(frozen=True)
class CanonicalCameras:
    identifiers: tuple[str, ...]
    viewmats: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int


def _finite_number(value: Any, *, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is non-finite")
    return result


def _frame_value(
    frame: Mapping[str, Any], root: Mapping[str, Any], name: str, *, index: int
) -> float:
    value = frame.get(name, root.get(name))
    return _finite_number(value, label=f"camera {index} {name}")


def _load_canonical_cameras(payload: bytes) -> CanonicalCameras:
    try:
        root = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("canonical transforms are not UTF-8 JSON") from error
    _require(isinstance(root, Mapping), "canonical transforms are not an object")
    _require(root.get("camera_model") == "OPENCV", "camera model is not OPENCV")
    frames = root.get("frames")
    _require(
        isinstance(frames, list) and len(frames) == CANONICAL_CAMERA_COUNT,
        f"canonical transforms must have exactly {CANONICAL_CAMERA_COUNT} cameras",
    )
    identifiers: list[str] = []
    viewmats: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    image_sizes: set[tuple[int, int]] = set()
    for index, frame in enumerate(frames):
        _require(isinstance(frame, Mapping), f"camera {index} is not an object")
        identifier = frame.get("file_path")
        _require(
            isinstance(identifier, str) and identifier and identifier.isascii(),
            f"camera {index} identifier is invalid",
        )
        _require(identifier not in identifiers, "canonical camera is duplicated")
        identifiers.append(identifier)
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        _require(matrix.shape == (4, 4), f"camera {index} transform shape changed")
        _require(bool(np.all(np.isfinite(matrix))), f"camera {index} is non-finite")
        _require(
            bool(np.array_equal(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]))),
            f"camera {index} has a non-homogeneous transform",
        )
        rotation = matrix[:3, :3]
        determinant = float(np.linalg.det(rotation))
        _require(
            math.isfinite(determinant)
            and abs(determinant - 1.0) <= 1.0e-5
            and bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5)),
            f"camera {index} rotation is not orthonormal",
        )
        # This is the exact algorithm in pinned Splatfacto ``get_viewmat``:
        # flip the c2w rotation's Y/Z columns and use the analytic rigid inverse.
        matrix32 = matrix.astype(np.float32)
        rotation_opencv = (
            matrix32[:3, :3] * np.asarray([1.0, -1.0, -1.0], dtype=np.float32)[None, :]
        )
        rotation_inverse = rotation_opencv.T
        translation_inverse = -(rotation_inverse @ matrix32[:3, 3:4])
        viewmat = np.zeros((4, 4), dtype=np.float32)
        viewmat[3, 3] = 1.0
        viewmat[:3, :3] = rotation_inverse
        viewmat[:3, 3:4] = translation_inverse
        _require(bool(np.all(np.isfinite(viewmat))), f"camera {index} inverse failed")
        viewmats.append(viewmat)

        width_value = _frame_value(frame, root, "w", index=index)
        height_value = _frame_value(frame, root, "h", index=index)
        width = int(width_value)
        height = int(height_value)
        _require(
            width_value == width and height_value == height, "image size is fractional"
        )
        _require(
            width > 0
            and height > 0
            and width % RENDER_DOWNSCALE == 0
            and height % RENDER_DOWNSCALE == 0,
            f"camera {index} cannot be exactly downscaled by {RENDER_DOWNSCALE}",
        )
        image_sizes.add((width // RENDER_DOWNSCALE, height // RENDER_DOWNSCALE))
        fx = _frame_value(frame, root, "fl_x", index=index) / RENDER_DOWNSCALE
        fy = _frame_value(frame, root, "fl_y", index=index) / RENDER_DOWNSCALE
        cx = _frame_value(frame, root, "cx", index=index) / RENDER_DOWNSCALE
        cy = _frame_value(frame, root, "cy", index=index) / RENDER_DOWNSCALE
        _require(fx > 0.0 and fy > 0.0, f"camera {index} focal length is invalid")
        intrinsics.append(np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]))
        for distortion in ("k1", "k2", "k3", "k4", "p1", "p2"):
            if distortion in frame or distortion in root:
                _require(
                    _frame_value(frame, root, distortion, index=index) == 0.0,
                    f"camera {index} has nonzero distortion",
                )
    _require(len(image_sizes) == 1, "canonical cameras have unequal image sizes")
    width, height = next(iter(image_sizes))
    return CanonicalCameras(
        identifiers=tuple(identifiers),
        viewmats=np.asarray(viewmats, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        width=width,
        height=height,
    )


def _percentile(values: np.ndarray, probability: float) -> float:
    _require(values.ndim == 1 and values.size > 0, "empty metric vector")
    _require(bool(np.all(np.isfinite(values))), "metric vector is non-finite")
    result = float(np.quantile(values, probability, method="linear"))
    _require(math.isfinite(result) and result >= 0.0, "invalid metric percentile")
    return result


def _distance_summary(values: np.ndarray, *, include_max: bool) -> dict[str, float]:
    result = {
        "mean": float(np.mean(values, dtype=np.float64)),
        "p95": _percentile(values, 0.95),
    }
    if include_max:
        result["max"] = float(np.max(values))
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in result.values()),
        "distance summary is invalid",
    )
    return result


def _pair_geometry_metrics(
    left: GaussianCloud, right: GaussianCloud
) -> dict[str, float]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - pinned GPU integration
        raise RuntimeError("scipy is required for equivalence matching") from error
    left_tree = cKDTree(left.xyz)
    right_tree = cKDTree(right.xyz)
    left_distance, left_to_right = right_tree.query(left.xyz, k=1, workers=1)
    right_distance, right_to_left = left_tree.query(right.xyz, k=1, workers=1)
    xyz = np.concatenate(
        [
            np.asarray(left_distance, dtype=np.float64),
            np.asarray(right_distance, dtype=np.float64),
        ]
    )

    def scalar_difference(
        left_value: np.ndarray, right_value: np.ndarray
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.abs(left_value - right_value[left_to_right]),
                np.abs(right_value - left_value[right_to_left]),
            ]
        )

    def vector_difference(
        left_value: np.ndarray, right_value: np.ndarray
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.linalg.norm(left_value - right_value[left_to_right], axis=1),
                np.linalg.norm(right_value - left_value[right_to_left], axis=1),
            ]
        )

    opacity = scalar_difference(left.opacity_probability, right.opacity_probability)
    scale = vector_difference(left.log_scales, right.log_scales)
    sh = vector_difference(left.sh, right.sh)
    left_dots = np.sum(left.quaternions * right.quaternions[left_to_right], axis=1)
    right_dots = np.sum(right.quaternions * left.quaternions[right_to_left], axis=1)
    dots = np.clip(np.abs(np.concatenate([left_dots, right_dots])), 0.0, 1.0)
    quaternion = 2.0 * np.arccos(dots)
    xyz_summary = _distance_summary(xyz, include_max=True)
    opacity_summary = _distance_summary(opacity, include_max=False)
    scale_summary = _distance_summary(scale, include_max=False)
    quaternion_summary = _distance_summary(quaternion, include_max=False)
    sh_summary = _distance_summary(sh, include_max=False)
    count_delta = abs(len(left.xyz) - len(right.xyz)) / max(
        len(left.xyz), len(right.xyz)
    )
    result = {
        "relative_count_delta": float(count_delta),
        "xyz_distance_mean_m": xyz_summary["mean"],
        "xyz_distance_p95_m": xyz_summary["p95"],
        "xyz_distance_max_m": xyz_summary["max"],
        "opacity_probability_abs_mean": opacity_summary["mean"],
        "opacity_probability_abs_p95": opacity_summary["p95"],
        "log_scale_vector_l2_mean": scale_summary["mean"],
        "log_scale_vector_l2_p95": scale_summary["p95"],
        "quaternion_angle_mean_rad": quaternion_summary["mean"],
        "quaternion_angle_p95_rad": quaternion_summary["p95"],
        "sh_vector_l2_mean": sh_summary["mean"],
        "sh_vector_l2_p95": sh_summary["p95"],
    }
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in result.values()),
        "pairwise geometry metrics are invalid",
    )
    return result


RenderArrays = tuple[np.ndarray, np.ndarray]
Renderer = Callable[[GaussianCloud, CanonicalCameras], RenderArrays]


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    _require(left.shape == right.shape and left.size > 0, "render shapes differ")
    _require(
        bool(np.all(np.isfinite(left))) and bool(np.all(np.isfinite(right))),
        "render is non-finite",
    )
    difference = left.astype(np.float64) - right.astype(np.float64)
    result = float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
    _require(math.isfinite(result) and result >= 0.0, "render RMSE is invalid")
    return result


def _fixed_gsplat_renderer(
    code_root: Path,
    physical_gpu_index: int,
) -> tuple[Renderer, Mapping[str, Any]]:
    hostname = _validate_execution_host()
    source = (code_root / "src").resolve(strict=True)
    source_value = os.fspath(source)
    _require(source_value in sys.path, "controlled analyzer source root is absent")
    runtime = importlib.import_module(
        "bayesian_phystwin.deform360_held_v8_gsplat_runtime"
    )
    expected_adapter = (code_root / RELATIVE_GSPLAT_ADAPTER_SOURCE).resolve(strict=True)
    _require(
        Path(str(runtime.__file__)).resolve(strict=True) == expected_adapter,
        "gsplat adapter escaped the code root",
    )
    smoke_record = _validate_gsplat_smoke(
        runtime.load_and_smoke_gsplat_runtime(),
        label="analyzer gsplat smoke",
        expected_physical_gpu_index=physical_gpu_index,
    )
    torch = importlib.import_module("torch")
    live_device = _validate_live_torch_device(torch)
    rendering = importlib.import_module("gsplat.rendering")
    rasterization = getattr(rendering, "rasterization", None)
    _require(callable(rasterization), "gsplat rasterization API is absent")

    def render(cloud: GaussianCloud, cameras: CanonicalCameras) -> RenderArrays:
        device = "cuda:0"
        dtype = torch.float32
        means = torch.as_tensor(cloud.xyz, dtype=dtype, device=device)
        # Match pinned Splatfacto: the CUDA rasterizer normalizes raw quaternions.
        quats = torch.as_tensor(cloud.raw_quaternions, dtype=dtype, device=device)
        log_scales = torch.as_tensor(cloud.log_scales, dtype=dtype, device=device)
        scales = torch.exp(log_scales)
        opacities = torch.sigmoid(
            torch.as_tensor(cloud.opacity_logits, dtype=dtype, device=device)
        )
        colors = torch.as_tensor(
            cloud.sh[:, :3, None].transpose(0, 2, 1), dtype=dtype, device=device
        )
        _require(
            bool(torch.isfinite(scales).all().item())
            and bool((scales > 0.0).all().item()),
            "activated Gaussian scales are invalid",
        )
        with torch.no_grad():
            render_rgb, alpha, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.as_tensor(cameras.viewmats, dtype=dtype, device=device),
                Ks=torch.as_tensor(cameras.intrinsics, dtype=dtype, device=device),
                width=cameras.width,
                height=cameras.height,
                packed=RENDER_CONTRACT["packed"],
                near_plane=RENDER_CONTRACT["near_plane"],
                far_plane=RENDER_CONTRACT["far_plane"],
                radius_clip=RENDER_CONTRACT["radius_clip"],
                eps2d=RENDER_CONTRACT["eps2d"],
                sh_degree=RENDER_CONTRACT["sh_degree"],
                tile_size=RENDER_CONTRACT["tile_size"],
                backgrounds=None,
                render_mode=RENDER_CONTRACT["render_mode"],
                sparse_grad=RENDER_CONTRACT["sparse_grad"],
                absgrad=RENDER_CONTRACT["absgrad"],
                rasterize_mode=RENDER_CONTRACT["rasterize_mode"],
                channel_chunk=RENDER_CONTRACT["channel_chunk"],
                distributed=RENDER_CONTRACT["distributed"],
                camera_model=RENDER_CONTRACT["camera_model"],
            )
        torch.cuda.synchronize(device)
        expected_rgb = (
            CANONICAL_CAMERA_COUNT,
            cameras.height,
            cameras.width,
            3,
        )
        expected_alpha = (
            CANONICAL_CAMERA_COUNT,
            cameras.height,
            cameras.width,
            1,
        )
        _require(tuple(render_rgb.shape) == expected_rgb, "gsplat RGB shape changed")
        _require(tuple(alpha.shape) == expected_alpha, "gsplat alpha shape changed")
        background = torch.tensor(INFERENCE_BACKGROUND_RGB, dtype=dtype, device=device)
        rgb = torch.clamp(render_rgb + (1.0 - alpha) * background, 0.0, 1.0)
        rgb_array = rgb.detach().cpu().numpy().astype(np.float32, copy=False)
        alpha_array = alpha.detach().cpu().numpy().astype(np.float32, copy=False)
        _require(
            bool(np.all(np.isfinite(rgb_array)))
            and bool(np.all(np.isfinite(alpha_array))),
            "gsplat render is non-finite",
        )
        return rgb_array, alpha_array

    return render, {
        "host": hostname,
        "physical_gpu_index": physical_gpu_index,
        "adapter_source": _bound_file(expected_adapter, label="gsplat adapter source"),
        "smoke_evidence": smoke_record,
        "live_device": live_device,
    }


def _pair_record(
    left: GaussianCloud,
    right: GaussianCloud,
    renders: Mapping[tuple[str, str], RenderArrays],
) -> dict[str, Any]:
    metrics = _pair_geometry_metrics(left, right)
    left_render = renders[(left.mode, left.pairing_id)]
    right_render = renders[(right.mode, right.pairing_id)]
    metrics["rgb_rmse"] = _rmse(left_render[0], right_render[0])
    metrics["alpha_rmse"] = _rmse(left_render[1], right_render[1])
    _require(set(metrics) == set(PAIR_METRIC_NAMES), "pair metric set changed")
    structured_equal = bool(
        left.schema == right.schema
        and left.vertices.dtype == right.vertices.dtype
        and left.vertices.shape == right.vertices.shape
        and np.array_equal(left.vertices, right.vertices)
    )
    return {
        "left": {"mode": left.mode, "pairing_id": left.pairing_id},
        "right": {"mode": right.mode, "pairing_id": right.pairing_id},
        "matched_pairing_id": left.pairing_id == right.pairing_id,
        "structured_array_exact": structured_equal,
        "file_sha256_exact": left.binding["sha256"] == right.binding["sha256"],
        "metrics": metrics,
    }


def _pair_groups(
    clouds: Mapping[str, Sequence[GaussianCloud]],
    renders: Mapping[tuple[str, str], RenderArrays],
) -> dict[str, list[dict[str, Any]]]:
    original = list(clouds["original"])
    wrapped = list(clouds["wrapped"])
    return {
        "within_original": [
            _pair_record(original[left], original[right], renders)
            for left in range(len(original))
            for right in range(left + 1, len(original))
        ],
        "within_wrapped": [
            _pair_record(wrapped[left], wrapped[right], renders)
            for left in range(len(wrapped))
            for right in range(left + 1, len(wrapped))
        ],
        "cross_mode": [
            _pair_record(left, right, renders) for left in original for right in wrapped
        ],
    }


def _metric_distribution(
    records: Sequence[Mapping[str, Any]], name: str
) -> dict[str, Any]:
    values = np.asarray(
        [record["metrics"][name] for record in records], dtype=np.float64
    )
    _require(
        values.size > 0 and bool(np.all(np.isfinite(values) & (values >= 0.0))),
        f"metric {name} is invalid",
    )
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "maximum": float(np.max(values)),
    }


def _evaluate_gate(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    counts = {name: len(records) for name, records in groups.items()}
    _require(
        counts["within_original"] >= MINIMUM_WITHIN_PAIRS_PER_MODE,
        "too few within-original pairs",
    )
    _require(
        counts["within_wrapped"] >= MINIMUM_WITHIN_PAIRS_PER_MODE,
        "too few within-wrapped pairs",
    )
    _require(counts["cross_mode"] >= MINIMUM_CROSS_PAIRS, "too few cross pairs")
    distributions: dict[str, dict[str, Any]] = {}
    per_metric: dict[str, Any] = {}
    for metric in PAIR_METRIC_NAMES:
        metric_distributions = {
            group: _metric_distribution(records, metric)
            for group, records in groups.items()
        }
        distributions[metric] = metric_distributions
        within_p95_limit = max(
            metric_distributions["within_original"]["p95"],
            metric_distributions["within_wrapped"]["p95"],
        )
        within_max_limit = max(
            metric_distributions["within_original"]["maximum"],
            metric_distributions["within_wrapped"]["maximum"],
        )
        median_passed = metric_distributions["cross_mode"]["median"] <= within_p95_limit
        p95_passed = metric_distributions["cross_mode"]["p95"] <= within_max_limit
        per_metric[metric] = {
            "cross_median": metric_distributions["cross_mode"]["median"],
            "within_p95_limit": within_p95_limit,
            "cross_median_condition_passed": median_passed,
            "cross_p95": metric_distributions["cross_mode"]["p95"],
            "within_max_limit": within_max_limit,
            "cross_p95_condition_passed": p95_passed,
            "passed": bool(median_passed and p95_passed),
        }
    gate = {
        "contract": dict(GATE_CONTRACT),
        "pair_counts": counts,
        "per_metric": per_metric,
        "all_metrics_finite_and_nonnegative": True,
        "passed": all(record["passed"] for record in per_metric.values()),
    }
    return gate, distributions


def _analyze_clouds(
    clouds: Mapping[str, Sequence[GaussianCloud]],
    cameras: CanonicalCameras,
    renderer: Renderer,
) -> dict[str, Any]:
    original = list(clouds["original"])
    wrapped = list(clouds["wrapped"])
    _require(
        len(original) >= MINIMUM_REPEATS_PER_MODE
        and len(wrapped) >= MINIMUM_REPEATS_PER_MODE,
        "too few clouds",
    )
    schemas = {cloud.schema for cloud in [*original, *wrapped]}
    _require(len(schemas) == 1, "Gaussian PLY schemas differ")
    renders: dict[tuple[str, str], RenderArrays] = {}
    render_calls: list[dict[str, str]] = []
    for cloud in [*original, *wrapped]:
        key = (cloud.mode, cloud.pairing_id)
        _require(key not in renders, "duplicate render key")
        rendered = renderer(cloud, cameras)
        _require(isinstance(rendered, tuple) and len(rendered) == 2, "bad renderer")
        renders[key] = rendered
        render_calls.append({"mode": cloud.mode, "pairing_id": cloud.pairing_id})
    _require(
        len(render_calls) == len(original) + len(wrapped),
        "a PLY was rendered more or less than once",
    )
    groups = _pair_groups({"original": original, "wrapped": wrapped}, renders)
    gate, distributions = _evaluate_gate(groups)
    matched_cross = [
        record for record in groups["cross_mode"] if record["matched_pairing_id"]
    ]
    _require(len(matched_cross) == len(original), "matched cross pairs are incomplete")
    exact_primary = all(record["structured_array_exact"] for record in matched_cross)
    exact_files = all(record["file_sha256_exact"] for record in matched_cross)
    accepted = bool(exact_primary or gate["passed"])
    basis = (
        "exact-structured-array-equality"
        if exact_primary
        else ("secondary-distributional-envelope" if gate["passed"] else "rejected")
    )
    return {
        "schema_validation": {
            "expected_field_count": 62,
            "expected_field_names": list(EXPECTED_PLY_FIELDS),
            "all_property_declarations_literal_float_f4": True,
            "inert_normal_fields": ["nx", "ny", "nz"],
            "all_inert_normal_values_exactly_zero": True,
            "inert_normal_fields_excluded_from_distribution_metrics": True,
            "identical_schema_across_all_plys": True,
            "all_source_and_derived_values_finite": True,
        },
        "render_execution": {
            "contract": dict(RENDER_CONTRACT),
            "calls": render_calls,
            "render_call_count": len(render_calls),
            "each_ply_rendered_exactly_once": True,
        },
        "pair_groups": groups,
        "metric_distributions": distributions,
        "secondary_distributional_gate": gate,
        "decision": {
            "exact_matched_structured_array_equality_primary_passed": exact_primary,
            "exact_matched_file_bytes_equal": exact_files,
            "secondary_distributional_equivalence_passed": gate["passed"],
            "accepted": accepted,
            "acceptance_basis": basis,
        },
    }


def _load_clouds(
    inputs: Mapping[str, Sequence[RepeatInput]],
) -> dict[str, list[GaussianCloud]]:
    result: dict[str, list[GaussianCloud]] = {}
    for mode in ("original", "wrapped"):
        result[mode] = []
        for record in inputs[mode]:
            payload = _read_regular_nofollow(record.path, label=f"{mode} Gaussian PLY")
            _require(
                len(payload) == record.binding["size_bytes"]
                and hashlib.sha256(payload).hexdigest() == record.binding["sha256"],
                f"{mode} PLY changed after manifest validation",
            )
            result[mode].append(
                _parse_gaussian_ply(
                    payload,
                    mode=mode,
                    pairing_id=record.pairing_id,
                    binding=record.binding,
                )
            )
    return result


def _validate_analyzer_runtime_record(
    value: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = dict(value)
    _require(
        set(runtime)
        == {
            "sys_executable",
            "sys_base_executable",
            "sys_prefix",
            "sys_base_prefix",
            "lexical_python",
            "resolved_python",
            "frozen_package_inventory",
            "frozen_runtime_tree_manifest",
            "verified_runtime_tree",
            "live_pip_freeze_all",
        },
        "analyzer runtime fields changed",
    )
    _require(
        runtime["frozen_package_inventory"]["sha256"]
        == expected["python_freeze_sha256"]
        and runtime["frozen_runtime_tree_manifest"]["sha256"]
        == expected["python_tree_manifest_sha256"],
        "runtime file binding differs from manifest",
    )
    verified_tree = runtime["verified_runtime_tree"]
    _require(
        isinstance(verified_tree, Mapping)
        and verified_tree.get("runtime_root") == os.fspath(PINNED_PYTHON_RUNTIME)
        and verified_tree.get("runtime_manifest_sha256")
        == PINNED_PYTHON_TREE_MANIFEST_SHA256
        and verified_tree.get("all_directories_and_regular_files_nonwritable") is True
        and verified_tree.get("all_entry_metadata_and_file_hashes_verified") is True,
        "actual analyzer runtime tree was not fully verified",
    )
    live_freeze = runtime["live_pip_freeze_all"]
    _require(
        isinstance(live_freeze, Mapping)
        and live_freeze.get("normalized_sha256") == PINNED_PYTHON_FREEZE_SHA256
        and live_freeze.get("equals_frozen_package_inventory") is True,
        "live analyzer package inventory changed",
    )
    return runtime


def _capture_run_state(
    *,
    manifest_file: Path,
    code: Path,
    generator_code_root: str | Path,
    deform360: Path,
    runtime_binding: Callable[[], Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, list[RepeatInput]],
    Path,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    manifest = _load_signed_json(manifest_file, label="repeat manifest")
    expected = _validate_manifest_environment(manifest)
    profile = str(expected["generator_profile"])
    physical_gpu_index = int(expected["physical_gpu_index"])
    code_git = _git_binding(code)
    _require(
        code_git == expected["analyzer_code"], "analyzer Git differs from manifest"
    )
    historical_generator = _historical_generator_binding(code)
    generator = _generator_checkout_binding(
        generator_code_root,
        profile=profile,
        analyzer_root=code,
        analyzer_git=code_git,
    )
    _require(generator == expected["generator_code"], "generator differs from manifest")
    deform360_git = _git_binding(deform360)
    _require(
        deform360_git["head"] == expected["deform360_git_head"]
        and deform360_git["tree"] == expected["deform360_git_tree"],
        "Deform360 source differs from manifest",
    )
    runtime = _validate_analyzer_runtime_record(
        dict(runtime_binding()), expected=expected
    )
    torch = importlib.import_module("torch")
    execution = _execution_binding(
        code, physical_gpu_index=physical_gpu_index, torch=torch
    )
    analyzer_source = (code / RELATIVE_ANALYZER_SOURCE).resolve(strict=True)
    analyzer_adapter = (code / RELATIVE_GSPLAT_ADAPTER_SOURCE).resolve(strict=True)
    analyzer_source_binding = _bound_file(analyzer_source, label="analyzer source")
    analyzer_adapter_binding = _bound_file(
        analyzer_adapter, label="analyzer gsplat adapter"
    )
    aot = _validated_pinned_aot_binding()
    validated = _validate_manifest(manifest)
    _require(isinstance(validated, tuple), "manifest input validation failed")
    inputs, transforms_path, validated_expected = validated
    _require(
        validated_expected == expected,
        "manifest environment changed between validation phases",
    )
    transitive_inputs: dict[str, list[dict[str, Any]]] = {}
    for mode in ("original", "wrapped"):
        transitive_inputs[mode] = []
        for record in inputs[mode]:
            inventory = _dataset_input_inventory(str(record.fit_evidence["dataset"]))
            transitive_inputs[mode].append(
                {
                    "pairing_id": record.pairing_id,
                    "ply": dict(record.binding),
                    "fit_evidence": {
                        **dict(record.fit_evidence_binding),
                        "artifact_sha256": record.fit_evidence["artifact_sha256"],
                    },
                    "dataset_input_inventory": inventory,
                }
            )
    state = {
        "manifest": {
            **_bound_file(manifest_file, label="repeat manifest"),
            "artifact_sha256": manifest["artifact_sha256"],
        },
        "canonical_source_dataset": _dataset_input_inventory(
            CANONICAL_PUBLIC_SOURCE_DATASET
        ),
        "canonical_transforms": _bound_file(
            transforms_path, label="canonical source transforms"
        ),
        "transitive_inputs": transitive_inputs,
        "source": {
            "analyzer_git": code_git,
            "generator": generator,
            "historical_generator_objects": historical_generator,
            "deform360_git": deform360_git,
            "analyzer_source": analyzer_source_binding,
            "analyzer_gsplat_adapter": analyzer_adapter_binding,
            "pinned_gsplat_aot": aot,
        },
        "runtime": runtime,
        "execution": execution,
    }
    return state, inputs, transforms_path, expected, manifest


def analyze(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    code_root: str | Path,
    generator_code_root: str | Path,
    deform360_root: str | Path,
    renderer_factory: Callable[[Path, int], tuple[Renderer, Mapping[str, Any]]] = (
        _fixed_gsplat_renderer
    ),
    runtime_binding: Callable[[], Mapping[str, Any]] = _runtime_binding,
) -> Path:
    output = _assert_nonheld_path(
        output_path, label="analysis output", must_exist=False
    )
    _require(not os.path.lexists(output), "analysis output already exists")
    code = _assert_nonheld_path(code_root, label="code root", must_exist=True)
    generator_root = _assert_nonheld_path(
        generator_code_root, label="generator code root", must_exist=True
    )
    deform360 = _assert_nonheld_path(
        deform360_root, label="Deform360 root", must_exist=True
    )
    canonical_source_root = _absolute(CANONICAL_PUBLIC_SOURCE_DATASET)
    _reject_output_within_roots(
        output,
        (code, generator_root, deform360, canonical_source_root),
        label="analysis output",
    )
    manifest_file = _assert_nonheld_path(
        manifest_path, label="repeat manifest", must_exist=True
    )
    bootstrap_manifest = _load_signed_json(manifest_file, label="repeat manifest")
    _reject_output_within_roots(
        output,
        _manifest_repeat_protected_roots(bootstrap_manifest),
        label="analysis output",
    )
    expected_source = (code / RELATIVE_ANALYZER_SOURCE).resolve(strict=True)
    _require(
        Path(__file__).resolve(strict=True) == expected_source,
        "analyzer source escaped the code root",
    )
    bootstrap_expected = bootstrap_manifest.get("expected_environment")
    _require(isinstance(bootstrap_expected, Mapping), "manifest environment is absent")
    bootstrap_profile = bootstrap_expected.get("generator_profile")
    _require(bootstrap_profile in FIT_QUALIFICATION_IDS, "manifest profile is invalid")
    _require(
        _strict_int(
            bootstrap_expected.get("physical_gpu_index"),
            label="manifest physical GPU",
        )
        == PROFILE_PHYSICAL_GPU_INDEX[str(bootstrap_profile)],
        "manifest physical GPU differs from profile",
    )
    _install_controlled_code_source(code)
    before_state, inputs, transforms_path, expected, manifest = _capture_run_state(
        manifest_file=manifest_file,
        code=code,
        generator_code_root=generator_root,
        deform360=deform360,
        runtime_binding=runtime_binding,
    )
    physical_gpu_index = int(expected["physical_gpu_index"])
    transform_bytes = _read_regular_nofollow(
        transforms_path, label="canonical transforms"
    )
    cameras = _load_canonical_cameras(transform_bytes)
    clouds = _load_clouds(inputs)
    renderer, gsplat_binding = renderer_factory(code, physical_gpu_index)
    _require(
        isinstance(gsplat_binding, Mapping)
        and set(gsplat_binding)
        == {
            "host",
            "physical_gpu_index",
            "adapter_source",
            "smoke_evidence",
            "live_device",
        }
        and gsplat_binding.get("host") == before_state["execution"]["host"]
        and gsplat_binding.get("physical_gpu_index") == physical_gpu_index
        and gsplat_binding.get("adapter_source")
        == before_state["source"]["analyzer_gsplat_adapter"]
        and gsplat_binding.get("live_device")
        == before_state["execution"]["live_device"],
        "analyzer gsplat binding differs from the validated run state",
    )
    _validate_gsplat_smoke(
        gsplat_binding.get("smoke_evidence"),
        label="analyzer gsplat smoke",
        expected_physical_gpu_index=physical_gpu_index,
    )
    analysis = _analyze_clouds(clouds, cameras, renderer)
    after_state, _, _, after_expected, after_manifest = _capture_run_state(
        manifest_file=manifest_file,
        code=code,
        generator_code_root=generator_root,
        deform360=deform360,
        runtime_binding=runtime_binding,
    )
    _require(after_expected == expected, "manifest environment changed during render")
    _require(after_manifest == manifest, "manifest changed during render")
    _require(after_state == before_state, "validated run state changed during render")
    evidence = _signed(
        {
            "schema_version": 1,
            "artifact_kind": RESULT_KIND,
            "analysis_id": ANALYSIS_ID,
            "development_only": True,
            "formal_path_accessed": False,
            "host": before_state["execution"]["host"],
            "generator_profile": expected["generator_profile"],
            "physical_gpu_index": physical_gpu_index,
            "input_manifest": before_state["manifest"],
            "source_bindings": before_state["source"],
            "runtime_binding": before_state["runtime"],
            "execution_binding": before_state["execution"],
            "gsplat_runtime": dict(gsplat_binding),
            "canonical_source_dataset": before_state["canonical_source_dataset"],
            "canonical_transforms": before_state["canonical_transforms"],
            "inputs": before_state["transitive_inputs"],
            "pre_post_render_stability": {
                "before": before_state,
                "after": after_state,
                "exact_equal": True,
                "analyzer_gsplat_smoke_executed_once_before_render": True,
                "adapter_and_aot_bytes_revalidated_after_render": True,
            },
            "statistical_limitations": list(STATISTICAL_LIMITATIONS),
            **analysis,
        }
    )
    return _write_new_json(output, evidence)


def _parse_pair_argument(value: Sequence[str]) -> tuple[str, Path, Path]:
    _require(len(value) == 3, "repeat must supply ID PLY FIT_EVIDENCE")
    pairing_id, path_value, evidence_value = value
    _require(
        pairing_id.isascii()
        and all(character.isalnum() or character in "-_." for character in pairing_id),
        "repeat pairing ID is invalid",
    )
    return pairing_id, Path(path_value), Path(evidence_value)


def _repeat_protected_roots(
    records: Iterable[tuple[str, Path, Path]],
) -> list[Path]:
    roots: list[Path] = []
    for pairing_id, ply_value, evidence_value in records:
        ply = _assert_nonheld_path(
            ply_value, label=f"repeat {pairing_id} PLY", must_exist=True
        )
        evidence_path = _assert_nonheld_path(
            evidence_value,
            label=f"repeat {pairing_id} fit evidence",
            must_exist=True,
        )
        evidence = _load_signed_json(
            evidence_path, label=f"repeat {pairing_id} fit evidence"
        )
        dataset_value = evidence.get("dataset")
        _require(
            isinstance(dataset_value, str) and dataset_value,
            f"repeat {pairing_id} dataset is absent",
        )
        dataset = _assert_nonheld_path(
            dataset_value,
            label=f"repeat {pairing_id} dataset",
            must_exist=True,
        )
        _require(dataset.is_dir(), f"repeat {pairing_id} dataset is not a directory")
        output_value = evidence.get("output")
        _require(
            isinstance(output_value, Mapping)
            and isinstance(output_value.get("path"), str)
            and bool(output_value.get("path")),
            f"repeat {pairing_id} fit output is absent",
        )
        fit_output = _assert_nonheld_path(
            str(output_value["path"]),
            label=f"repeat {pairing_id} fit output",
            must_exist=True,
        )
        roots.extend((ply.parent, evidence_path.parent, dataset, fit_output.parent))
    return roots


def _manifest_repeat_protected_roots(manifest: Mapping[str, Any]) -> list[Path]:
    modes = manifest.get("modes")
    _require(isinstance(modes, Mapping), "manifest modes are absent")
    _require(set(modes) == {"original", "wrapped"}, "manifest modes changed")
    roots: list[Path] = []
    for mode in ("original", "wrapped"):
        records = modes.get(mode)
        _require(isinstance(records, list), f"{mode} records are not a list")
        for index, record in enumerate(records):
            _require(isinstance(record, Mapping), f"{mode} repeat {index} is invalid")
            pairing_id = record.get("pairing_id")
            ply = record.get("ply")
            evidence = record.get("fit_evidence")
            inventory = record.get("dataset_input_inventory")
            _require(
                isinstance(pairing_id, str)
                and isinstance(ply, Mapping)
                and isinstance(ply.get("path"), str)
                and isinstance(evidence, Mapping)
                and isinstance(evidence.get("path"), str)
                and isinstance(inventory, Mapping)
                and isinstance(inventory.get("root"), str),
                f"{mode} repeat {index} path binding is invalid",
            )
            roots.extend(
                (
                    _absolute(str(ply["path"])).parent,
                    _absolute(str(evidence["path"])).parent,
                    _absolute(str(inventory["root"])),
                )
            )
    return roots


def prepare_manifest(
    *,
    original: Iterable[Sequence[str]],
    wrapped: Iterable[Sequence[str]],
    canonical_transforms: str | Path,
    output_path: str | Path,
    code_root: str | Path,
    generator_code_root: str | Path,
    deform360_root: str | Path,
    generator_profile: str,
) -> Path:
    output = _assert_nonheld_path(
        output_path, label="manifest output", must_exist=False
    )
    _require(not os.path.lexists(output), "manifest output already exists")
    _require(generator_profile in FIT_QUALIFICATION_IDS, "unknown generator profile")
    code = _assert_nonheld_path(code_root, label="analyzer code root", must_exist=True)
    generator_root = _assert_nonheld_path(
        generator_code_root, label="generator code root", must_exist=True
    )
    deform360 = _assert_nonheld_path(
        deform360_root, label="Deform360 root", must_exist=True
    )
    canonical_source_root = _assert_nonheld_path(
        CANONICAL_PUBLIC_SOURCE_DATASET,
        label="canonical public source dataset",
        must_exist=True,
    )
    _reject_output_within_roots(
        output,
        (code, generator_root, deform360, canonical_source_root),
        label="manifest output",
    )
    parsed_by_mode = {
        "original": [_parse_pair_argument(value) for value in original],
        "wrapped": [_parse_pair_argument(value) for value in wrapped],
    }
    _reject_output_within_roots(
        output,
        _repeat_protected_roots(
            [*parsed_by_mode["original"], *parsed_by_mode["wrapped"]]
        ),
        label="manifest output",
    )
    code_git = _git_binding(code)
    _historical_generator_binding(code)
    generator_code = _generator_checkout_binding(
        generator_root,
        profile=generator_profile,
        analyzer_root=code,
        analyzer_git=code_git,
    )
    deform360_git = _git_binding(deform360)
    _require(
        deform360_git["head"] == PINNED_DEFORM360_REVISION
        and deform360_git["tree"] == PINNED_DEFORM360_TREE,
        "Deform360 source is not the pinned tree",
    )
    _require(
        canonical_source_root == CANONICAL_PUBLIC_SOURCE_DATASET,
        "canonical public source dataset path is aliased",
    )
    canonical_source = _dataset_input_inventory(canonical_source_root)
    transforms_path = _assert_nonheld_path(
        canonical_transforms, label="canonical transforms", must_exist=True
    )
    _require(
        transforms_path == canonical_source_root / "transforms.json",
        "canonical transforms must be the exact public source transforms",
    )
    transforms = _bound_file(transforms_path, label="canonical transforms")
    transforms_descriptor = {
        key: transforms[key] for key in ("path", "size_bytes", "sha256")
    }
    normalized_transforms = _normalized_transforms_descriptor(
        _read_regular_nofollow(canonical_transforms, label="canonical transforms")
    )
    expected_adapter_path = Path(
        str(
            generator_code["sources"][
                "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
            ]["path"]
        )
    )
    modes: dict[str, list[dict[str, Any]]] = {}
    pairing_sets: dict[str, set[str]] = {}
    repeat_inodes: set[tuple[int, int]] = set()
    for mode in ("original", "wrapped"):
        parsed = parsed_by_mode[mode]
        _require(
            len(parsed) >= MINIMUM_REPEATS_PER_MODE,
            f"{mode} has fewer than {MINIMUM_REPEATS_PER_MODE} repeats",
        )
        identifiers = [identifier for identifier, _, _ in parsed]
        _require(len(identifiers) == len(set(identifiers)), f"duplicate {mode} ID")
        pairing_sets[mode] = set(identifiers)
        modes[mode] = []
        for identifier, path, evidence_path in sorted(parsed):
            binding = _bound_file(path, label=f"{mode} repeat PLY")
            path_stat = os.lstat(Path(binding["path"]))
            path_inode = (path_stat.st_dev, path_stat.st_ino)
            _require(
                path_inode not in repeat_inodes, "a repeat PLY/evidence inode is reused"
            )
            repeat_inodes.add(path_inode)
            ply_descriptor = {
                key: binding[key] for key in ("path", "size_bytes", "sha256")
            }
            evidence_binding = _bound_file(evidence_path, label=f"{mode} fit evidence")
            evidence_stat = os.lstat(Path(evidence_binding["path"]))
            evidence_inode = (evidence_stat.st_dev, evidence_stat.st_ino)
            _require(
                evidence_inode not in repeat_inodes,
                "a repeat PLY/evidence inode is reused",
            )
            repeat_inodes.add(evidence_inode)
            evidence_descriptor = {
                key: evidence_binding[key] for key in ("path", "size_bytes", "sha256")
            }
            _, evidence = _validate_fit_evidence(
                evidence_descriptor,
                mode=mode,
                ply_binding=binding,
                expected_adapter_path=expected_adapter_path,
                generator_profile=generator_profile,
                expected_physical_gpu_index=PROFILE_PHYSICAL_GPU_INDEX[
                    generator_profile
                ],
            )
            inventory = _dataset_input_inventory(str(evidence["dataset"]))
            _require(
                inventory["content_identity"] == canonical_source["content_identity"]
                and inventory["content_artifact_sha256"]
                == canonical_source["content_artifact_sha256"],
                f"{mode} repeat differs from the canonical public source after "
                "only absolute seed-path normalization",
            )
            modes[mode].append(
                {
                    "pairing_id": identifier,
                    "ply": ply_descriptor,
                    "fit_evidence": evidence_descriptor,
                    "dataset_input_inventory": inventory,
                }
            )
    _require(
        pairing_sets["original"] == pairing_sets["wrapped"],
        "original and wrapped pairing IDs differ",
    )
    total_repeats = sum(len(records) for records in modes.values())
    expected_inode_count = 2 * total_repeats
    _require(
        len(repeat_inodes) == expected_inode_count,
        "repeat PLY/evidence files are not all inode-distinct",
    )
    manifest = _signed(
        {
            "schema_version": 1,
            "artifact_kind": MANIFEST_KIND,
            "analysis_id": ANALYSIS_ID,
            "expected_environment": {
                "generator_profile": generator_profile,
                "physical_gpu_index": PROFILE_PHYSICAL_GPU_INDEX[generator_profile],
                "generator_code": generator_code,
                "analyzer_code": code_git,
                "deform360_git_head": deform360_git["head"],
                "deform360_git_tree": deform360_git["tree"],
                "python_freeze_sha256": PINNED_PYTHON_FREEZE_SHA256,
                "python_tree_manifest_sha256": PINNED_PYTHON_TREE_MANIFEST_SHA256,
            },
            "canonical_source_dataset": canonical_source,
            "canonical_transforms": {
                "raw_representative": transforms_descriptor,
                "normalized": normalized_transforms,
            },
            "modes": modes,
        }
    )
    return _write_new_json(output, manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument(
        "--original",
        action="append",
        nargs=3,
        required=True,
        metavar=("ID", "PLY", "FIT_EVIDENCE"),
    )
    prepare.add_argument(
        "--wrapped",
        action="append",
        nargs=3,
        required=True,
        metavar=("ID", "PLY", "FIT_EVIDENCE"),
    )
    prepare.add_argument("--canonical-transforms", type=Path, required=True)
    prepare.add_argument("--code-root", type=Path, required=True)
    prepare.add_argument("--generator-code-root", type=Path, required=True)
    prepare.add_argument(
        "--generator-profile",
        choices=tuple(FIT_QUALIFICATION_IDS),
        required=True,
    )
    prepare.add_argument("--deform360-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("analyze")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--code-root", type=Path, required=True)
    run.add_argument("--generator-code-root", type=Path, required=True)
    run.add_argument("--deform360-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "prepare-manifest":
        path = prepare_manifest(
            original=arguments.original,
            wrapped=arguments.wrapped,
            canonical_transforms=arguments.canonical_transforms,
            output_path=arguments.output,
            code_root=arguments.code_root,
            generator_code_root=arguments.generator_code_root,
            deform360_root=arguments.deform360_root,
            generator_profile=arguments.generator_profile,
        )
    else:
        path = analyze(
            arguments.manifest,
            arguments.output,
            code_root=arguments.code_root,
            generator_code_root=arguments.generator_code_root,
            deform360_root=arguments.deform360_root,
        )
    print(path)
    if arguments.command == "analyze":
        result = _load_signed_json(path, label="analysis result")
        decision = result.get("decision")
        _require(isinstance(decision, Mapping), "analysis result decision is absent")
        return 0 if decision.get("accepted") is True else 3
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by operator
    raise SystemExit(main())
