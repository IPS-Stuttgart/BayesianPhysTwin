#!/usr/bin/env python3
"""Seal a complete Deform360 lifecycle-qualification-v2 evidence closure.

This operator is deliberately only an integrity gate.  It never runs a fit,
re-renders a Gaussian cloud, or opens a formal-held numerical payload.  It
does independently bind every retained development input/output, validate the
signed fit, equivalence, and resource-soak evidence, recompute the declared
distributional gate and resource predicates, reject undeclared tree entries,
and make the completed qualification tree immutable.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import stat
import sys
from typing import Any


QUALIFICATION_ID = "deform360-nerfstudio-resource-lifecycle-qualification-v2"
QUALIFICATION_KIND = "Deform360ResourceLifecycleQualificationEvidenceV2"
ATTEMPT_KIND = "Deform360ResourceLifecycleQualificationAttemptV2"
COMPLETION_KIND = "Deform360ResourceLifecycleQualificationIntegrityCompletionV2"
MANIFEST_KIND = "Deform360ResourceLifecycleRepeatManifestV1"
RESULT_KIND = "Deform360ResourceLifecycleDistributionalEquivalenceV1"
FIT_KIND = "Deform360ResourceLifecycleFitChildEvidence"
SOAK_KIND = "Deform360ResourceLifecycleSoakChildEvidence"
ANALYSIS_ID = "deform360-resource-lifecycle-distributional-equivalence-v1"

EXPECTED_HOST = "workstation2"
BASE = Path("/mnt/corsair/florianpfaff")
ROOT_PREFIX = "bpt-resource-lifecycle-qualification-"
MAIN_NAME = "resource-lifecycle-qualification.json"
ATTEMPT_NAME = "qualification-attempt.json"
FORMAL_HELD_PARENT = BASE / "bpt-online-belief-v1"

PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
PINNED_PYTHON_RUNTIME = PINNED_PYTHON.parent.parent
PINNED_PYTHON_FREEZE = Path(f"{PINNED_PYTHON_RUNTIME}.freeze.sorted.txt")
PINNED_PYTHON_TREE_MANIFEST = Path(f"{PINNED_PYTHON_RUNTIME}.tree-manifest.json")
PINNED_PYTHON_BASE_PREFIX = "/usr"
PINNED_PYTHON_FREEZE_SHA256 = (
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
PINNED_PYTHON_TREE_MANIFEST_SHA256 = (
    "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61"
)
PINNED_PYTHON_TARGET = "/usr/bin/python3.12"
PINNED_PYTHON_TARGET_SHA256 = (
    "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7"
)
PINNED_NUMPY_VERSION = "1.26.4"
PINNED_NUMPY_SOURCE_RELATIVE = Path("lib/python3.12/site-packages/numpy/__init__.py")
PINNED_DEFORM360 = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
    "Deform360-processing-0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
)
PINNED_DEFORM360_HEAD = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
PINNED_DEFORM360_TREE = "c566ed29db7e0fd6a4cb768d840a4aa662864680"
PUBLIC_DATASET = Path(
    "/mnt/corsair/florianpfaff/deform360-reusable-sota-v1/"
    "processing-sam2-dev-smoke/004-rubber-band/episode_0001/"
    "splatfacto/.scratch_000000"
)
RELATIVE_ANALYZER_SOURCE = Path(
    "scripts/development/analyze_deform360_resource_lifecycle_equivalence.py"
)
RELATIVE_QUALIFIER_SOURCE = Path(
    "scripts/development/qualify_deform360_resource_lifecycle.py"
)
RELATIVE_WRAPPER_SOURCE = Path(
    "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
)
RELATIVE_GSPLAT_ADAPTER_SOURCE = Path(
    "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
)
ANALYZER_SOURCE_SHA256 = (
    "43056e39ff7ea5f760f18420784db0edbb75523031dba7f3a19eca0c6951c128"
)
GSPLAT_ADAPTER_SHA256 = (
    "2985de3b4e3f6bea7e98eb0e36148f52d8ee96bce027eb13bad98e87fd7f875c"
)

PINNED_TORCH_VERSION = "2.4.0+cu121"
PINNED_TORCH_CUDA_VERSION = "12.1"
PINNED_GPU_NAME = "NVIDIA RTX 6000 Ada Generation"
PINNED_GSPLAT_VERSION = "1.4.0"
PINNED_GSPLAT_EXTENSION_PATH = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v7-runtimes/"
    "gsplat-cuda-py312-cu121-"
    "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64/"
    "gsplat_cuda.so"
)
PINNED_GSPLAT_EXTENSION_SHA256 = (
    "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64"
)
PINNED_GSPLAT_SMOKE_CONTRACT_SHA256 = (
    "0c2786579530037e32e6b7e39291cbae9b06f9113828d602864f13d84d335962"
)

REPEAT_COUNT = 5
PAIRING_IDS = tuple(f"repeat-{index:03d}" for index in range(REPEAT_COUNT))
FIT_ITERATIONS = 250
FIT_SEED = 0
SOAK_FIT_COUNT = 243
SOAK_ITERATIONS = 1
SOAK_TRAINER_REINITIALIZATION_INTERVAL = 81
V8_PYCACHE_PREFIX = "/nonexistent/bpt-held-v8-pycache"
FIT_TIMEOUT_SECONDS = 3_600
ANALYZER_TIMEOUT_SECONDS = 86_400
SOAK_TIMEOUT_SECONDS = 86_400
FIRST_FIT_FD_GROWTH_LIMIT = 32
STEADY_FD_GROWTH_LIMIT = 4
STEADY_TASK_GROWTH_LIMIT = 4

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
GROUP_COUNTS = {"within_original": 10, "within_wrapped": 10, "cross_mode": 25}
GATE_CONTRACT: Mapping[str, Any] = {
    "contract_id": "deform360-empirical-pairwise-equivalence-envelope-v1",
    "metric_names": list(PAIR_METRIC_NAMES),
    "within_original_minimum_pair_count": 10,
    "within_wrapped_minimum_pair_count": 10,
    "cross_mode_minimum_pair_count": 25,
    "percentile_method": "linear",
    "per_metric_conditions": [
        "cross_mode_median <= max(within_original_p95, within_wrapped_p95)",
        "cross_mode_p95 <= max(within_original_max, within_wrapped_max)",
    ],
    "all_metrics_required": True,
    "exact_matched_structured_array_equality_is_primary": True,
    "distributional_gate_is_secondary": True,
}

REQUIRED_EXECUTION_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_MODULE_LOADING": "LAZY",
    "CUDA_VISIBLE_DEVICES": "1",
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
    "PYTHONPYCACHEPREFIX": V8_PYCACHE_PREFIX,
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
ROOT_CONSUMPTION_POLICY: Mapping[str, bool] = {
    "canonical_root_consumed_at_creation": True,
    "same_root_retry_permitted": False,
    "same_revision_retry_permitted": False,
    "in_place_reuse_permitted": False,
    "incomplete_root_sealable_or_replayable": False,
    "technical_fix_in_later_disclosed_revision_may_use_new_root": True,
    "replacement_requires_different_canonical_root": True,
    "replacement_may_change_frozen_analyzer_or_numerical_gate": False,
}
NO_GO_INTERPRETATION = (
    "admission-inconclusive; the frozen analyzer did not admit this single fresh "
    "cohort, which is not proof of wrapper inequivalence"
)

# Populated only after ``seal`` proves that this process is the exact isolated
# pinned interpreter.  Keeping NumPy out of module initialization ensures that
# no numerical code executes before that trust boundary is established.
np: Any | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _strict_int(value: object, *, label: str, minimum: int | None = None) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} is not a strict integer",
    )
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{label} is below its minimum")
    return result


def _finite(value: object, *, label: str, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is non-finite")
    if nonnegative:
        _require(result >= 0.0, f"{label} is negative")
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


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


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _artifact_sha256(result)
    return result


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_pinned_parent_runtime() -> None:
    flags = sys.flags
    _require(
        flags.isolated == 1
        and flags.ignore_environment == 1
        and flags.no_user_site == 1
        and flags.dont_write_bytecode == 1,
        "qualification sealer must use the pinned Python with -I -B",
    )
    base_executable_value = getattr(sys, "_base_executable", None)
    _require(
        isinstance(base_executable_value, str) and base_executable_value,
        "qualification sealer Python has no base executable",
    )
    _require(
        _absolute(sys.executable) == PINNED_PYTHON,
        "qualification sealer is not using the pinned Python launcher",
    )
    _require(
        _absolute(base_executable_value) == _absolute(PINNED_PYTHON_TARGET),
        "qualification sealer Python base executable changed",
    )
    _require(
        _absolute(sys.prefix) == PINNED_PYTHON_RUNTIME,
        "qualification sealer Python prefix changed",
    )
    _require(
        _absolute(sys.base_prefix) == _absolute(PINNED_PYTHON_BASE_PREFIX),
        "qualification sealer Python base prefix changed",
    )


def _validate_pinned_numpy(module: Any) -> None:
    source_value = getattr(module, "__file__", None)
    _require(
        getattr(module, "__version__", None) == PINNED_NUMPY_VERSION,
        "qualification sealer NumPy version changed",
    )
    _require(
        isinstance(source_value, str) and source_value,
        "qualification sealer NumPy source is absent",
    )
    source = _absolute(source_value)
    expected = _absolute(PINNED_PYTHON_RUNTIME / PINNED_NUMPY_SOURCE_RELATIVE)
    try:
        resolved_source = source.resolve(strict=True)
        resolved_expected = expected.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            "qualification sealer NumPy source is unavailable"
        ) from error
    _require(
        source == expected and resolved_source == resolved_expected,
        "qualification sealer NumPy source changed",
    )


def _load_pinned_numpy() -> None:
    global np

    _require(np is None, "qualification sealer NumPy was already initialized")
    preexisting = sorted(
        name for name in sys.modules if name == "numpy" or name.startswith("numpy.")
    )
    _require(not preexisting, "NumPy was imported before the sealer runtime preflight")
    modules_before = set(sys.modules)
    try:
        imported = importlib.import_module("numpy")
        _require(
            sys.modules.get("numpy") is imported,
            "qualification sealer NumPy module identity changed",
        )
        _validate_pinned_numpy(imported)
    except BaseException:
        for name in tuple(sys.modules):
            if name not in modules_before and (
                name == "numpy" or name.startswith("numpy.")
            ):
                sys.modules.pop(name, None)
        raise
    np = imported


def _require_pinned_runtime_and_load_numpy() -> None:
    _require_pinned_parent_runtime()
    _load_pinned_numpy()


def _numpy_module() -> Any:
    _require(np is not None, "qualification sealer NumPy is not initialized")
    return np


def _is_formal_held(path: Path) -> bool:
    try:
        relative = path.relative_to(FORMAL_HELD_PARENT)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0].startswith("held-")


def _stable_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_file(path: Path, *, role: str) -> tuple[dict[str, Any], bytes]:
    absolute = _absolute(path)
    _require(not _is_formal_held(absolute), f"{role} is inside a formal held root")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{role} is unavailable") from error
    _require(
        not _is_formal_held(resolved), f"{role} resolves inside a formal held root"
    )
    _require(resolved == absolute, f"{role} resolves through a symlink")
    before = os.lstat(absolute)
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"{role} is not a regular file",
    )
    _require(before.st_nlink == 1, f"{role} is hard-linked")
    descriptor = os.open(
        absolute,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        _require(_stable_state(opened) == _stable_state(before), f"{role} changed")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(absolute)
    _require(
        _stable_state(before) == _stable_state(after) == _stable_state(current),
        f"{role} changed while hashing",
    )
    return (
        {
            "path": os.fspath(absolute),
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
            "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
        },
        bytes(payload),
    )


def _stable_file(path: Path, *, role: str) -> dict[str, Any]:
    return _read_stable_file(path, role=role)[0]


def _load_signed(path: Path, *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record, payload = _read_stable_file(path, role=role)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} is not UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{role} is not a JSON object")
    _require(
        value.get("artifact_sha256") == _artifact_sha256(value),
        f"{role} signature changed",
    )
    return value, record


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_oid(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_relative_posix(value: object, *, role: str) -> str:
    _require(isinstance(value, str) and value, f"{role} is absent")
    _require(
        "\\" not in value and "\x00" not in value,
        f"{role} is not a canonical POSIX path",
    )
    candidate = PurePosixPath(value)
    _require(
        not candidate.is_absolute()
        and candidate.as_posix() == value
        and all(part not in {"", ".", ".."} for part in candidate.parts),
        f"{role} is not a canonical relative POSIX path",
    )
    return value


def _descendant(root: Path, value: object, *, role: str) -> Path:
    _require(isinstance(value, str) and value, f"{role} path is absent")
    path = _absolute(value)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{role} is outside the qualification root") from error
    _require(path != root, f"{role} is the qualification root")
    _require(path.resolve(strict=True) == path, f"{role} is aliased")
    return path


def _verify_record(
    value: object,
    *,
    role: str,
    root: Path | None = None,
    artifact_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(value, Mapping), f"{role} binding is absent")
    required = {"path", "size_bytes", "sha256"}
    allowed = required | {"mode_octal", "artifact_sha256"}
    _require(required <= set(value) <= allowed, f"{role} binding fields changed")
    path_value = value.get("path")
    path = (
        _descendant(root, path_value, role=role) if root else _absolute(str(path_value))
    )
    observed = _stable_file(path, role=role)
    _require(
        all(
            observed[key] == value.get(key) for key in ("path", "size_bytes", "sha256")
        ),
        f"{role} binding changed",
    )
    if "mode_octal" in value:
        _require(
            observed["mode_octal"] == value["mode_octal"],
            f"{role} mode changed before sealing",
        )
    if artifact_sha256 is not None:
        _require(
            value.get("artifact_sha256") == artifact_sha256,
            f"{role} artifact binding changed",
        )
    return path, observed


def _record_with_artifact(
    record: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
        "artifact_sha256": artifact["artifact_sha256"],
    }


def _validate_git_binding(value: object, *, role: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{role} is absent")
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
        f"{role} fields changed",
    )
    _require(
        isinstance(value.get("path"), str)
        and _valid_oid(value.get("head"))
        and _valid_oid(value.get("tree"))
        and value.get("clean") is True
        and value.get("ordinary_untracked_file_count") == 0
        and value.get("ignored_untracked_file_count") == 0,
        f"{role} identity changed",
    )
    return dict(value)


def _validate_source_record(
    value: object,
    *,
    role: str,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{role} is absent")
    _require(
        set(value)
        in (
            {"path", "size_bytes", "sha256", "mode_octal"},
            {"path", "size_bytes", "sha256", "mode_octal", "git_blob_oid"},
        ),
        f"{role} fields changed",
    )
    path, observed = _verify_record(
        {key: value[key] for key in ("path", "size_bytes", "sha256", "mode_octal")},
        role=role,
    )
    if expected_path is not None:
        _require(path == expected_path.resolve(strict=True), f"{role} path changed")
    if expected_sha256 is not None:
        _require(observed["sha256"] == expected_sha256, f"{role} source changed")
    if "git_blob_oid" in value:
        _require(_valid_oid(value.get("git_blob_oid")), f"{role} Git blob changed")
    return dict(value)


def _content_identity(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {"sha256": value["sha256"], "size_bytes": value["size_bytes"]}
        for name, value in sorted(records.items())
    }


def _dataset_input(
    dataset_root: Path,
    value: object,
    *,
    role: str,
    require_relative: bool,
) -> tuple[str, Path, dict[str, Any]]:
    _require(isinstance(value, str) and value, f"{role} path is absent")
    _require("\\" not in value and "\x00" not in value, f"{role} path is not POSIX")
    declared = Path(value)
    if declared.is_absolute():
        _require(not require_relative, f"{role} path must be relative")
        path = _absolute(declared)
        try:
            relative = path.relative_to(dataset_root)
        except ValueError as error:
            raise RuntimeError(f"{role} escapes the dataset") from error
    else:
        normalized = Path(os.path.normpath(value))
        _require(
            normalized.parts
            and normalized.parts[0] not in {"", ".", ".."}
            and ".." not in normalized.parts,
            f"{role} escapes the dataset",
        )
        relative = normalized
        path = dataset_root / relative
    name = _canonical_relative_posix(relative.as_posix(), role=f"{role} relative path")
    _require(path.resolve(strict=True) == path, f"{role} path is aliased")
    return name, path, _stable_file(path, role=role)


def _recompute_dataset_identity(dataset_root: Path, *, role: str) -> dict[str, Any]:
    """Derive a dataset closure directly from its transforms and referenced files."""

    root = _absolute(dataset_root)
    root_state = os.lstat(root)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and root.resolve(strict=True) == root,
        f"{role} root is not canonical",
    )
    transforms_path = root / "transforms.json"
    raw_transforms, transforms_payload = _read_stable_file(
        transforms_path, role=f"{role} transforms"
    )
    try:
        transforms = json.loads(transforms_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} transforms are invalid") from error
    _require(isinstance(transforms, dict), f"{role} transforms are not an object")
    seed_value = transforms.get("ply_file_path")
    _require(
        isinstance(seed_value, str) and Path(seed_value).is_absolute(),
        f"{role} seed declaration is not an absolute path",
    )
    seed_name, seed_path, seed_record = _dataset_input(
        root,
        seed_value,
        role=f"{role} seed PLY",
        require_relative=False,
    )
    frames = transforms.get("frames")
    _require(isinstance(frames, list) and frames, f"{role} frames are absent")
    references: list[tuple[str, str, Path, dict[str, Any]]] = [
        ("seed_ply", seed_name, seed_path, seed_record)
    ]
    frame_names: list[str] = []
    for index, frame in enumerate(frames):
        _require(isinstance(frame, Mapping), f"{role} frame {index} is not an object")
        name, path, record = _dataset_input(
            root,
            frame.get("file_path"),
            role=f"{role} frame {index}",
            require_relative=True,
        )
        frame_names.append(name)
        references.append(("frame_image", name, path, record))
    reference_names = [name for _, name, _, _ in references]
    _require(
        len(reference_names) == len(set(reference_names)),
        f"{role} input reference repeats",
    )
    references.sort(key=lambda item: (item[1], item[0]))
    records = {name: record for _, name, _, record in references}
    rows = [
        {
            "role": entry_role,
            "relative_path": name,
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for entry_role, name, _, record in references
    ]
    portable = json.loads(json.dumps(transforms, allow_nan=False))
    portable["ply_file_path"] = "<MATERIALIZED-SEED-PLY>"
    normalized_payload = _canonical_bytes(portable)
    content_identity = {
        "normalized_transforms": {
            "size_bytes": len(normalized_payload),
            "sha256": hashlib.sha256(normalized_payload).hexdigest(),
        },
        "referenced_files": rows,
    }
    return {
        "root": root,
        "raw_transforms": raw_transforms,
        "seed_declared_path": seed_value,
        "seed_relative_path": seed_name,
        "seed_path": seed_path,
        "frame_relative_paths": frame_names,
        "frame_count": len(frame_names),
        "referenced_records": records,
        "referenced_file_content": _content_identity(records),
        "referenced_rows": rows,
        "portable_transforms_sha256": hashlib.sha256(normalized_payload).hexdigest(),
        "content_identity": content_identity,
        "content_artifact_sha256": hashlib.sha256(
            _canonical_bytes(content_identity)
        ).hexdigest(),
    }


def _validate_dataset_audit(
    value: object,
    *,
    root: Path,
    expected_root: Path,
    role: str,
    canonical_dataset: Mapping[str, Any] | None = None,
) -> set[Path]:
    """Validate one qualifier materialization record and return retained files."""
    canonical = (
        _recompute_dataset_identity(PUBLIC_DATASET, role="canonical public dataset")
        if canonical_dataset is None
        else canonical_dataset
    )
    local = _recompute_dataset_identity(expected_root, role=f"{role} materialized")
    _require(isinstance(value, Mapping), f"{role} audit is absent")
    expected_fields = {
        "source_root",
        "destination_root",
        "source_transforms",
        "source_transforms_sha256",
        "materialized_transforms_sha256",
        "portable_transforms_sha256",
        "rewritten_field",
        "source_seed_ply_path",
        "materialized_seed_ply_path",
        "frame_count",
        "copied_regular_file_count",
        "source_records",
        "materialized_records",
        "referenced_source_content",
        "referenced_materialized_content",
        "referenced_source_materialized_content_equal",
        "unreferenced_outputs_copied",
    }
    _require(set(value) == expected_fields, f"{role} audit fields changed")
    _require(
        value.get("source_root") == os.fspath(PUBLIC_DATASET)
        and value.get("destination_root") == os.fspath(expected_root)
        and value.get("rewritten_field") == "ply_file_path"
        and value.get("referenced_source_materialized_content_equal") is True
        and value.get("unreferenced_outputs_copied") is False,
        f"{role} audit identity changed",
    )
    _require(
        expected_root.resolve(strict=True) == expected_root, f"{role} root is aliased"
    )
    source_records = value.get("source_records")
    materialized = value.get("materialized_records")
    _require(
        isinstance(source_records, Mapping)
        and isinstance(materialized, Mapping)
        and source_records
        and materialized,
        f"{role} input records are absent",
    )
    source_verified: dict[str, dict[str, Any]] = {}
    for name, record in source_records.items():
        name = _canonical_relative_posix(name, role=f"{role} source name")
        path, observed = _verify_record(record, role=f"{role} source {name}")
        _require(
            path == PUBLIC_DATASET / name,
            f"{role} source path changed: {name}",
        )
        source_verified[name] = observed
    canonical_records = canonical.get("referenced_records")
    _require(
        isinstance(canonical_records, Mapping)
        and set(source_verified) == set(canonical_records)
        and _content_identity(source_verified)
        == canonical.get("referenced_file_content"),
        f"{role} source reference set differs from the canonical public dataset",
    )
    source_transforms = value.get("source_transforms")
    transforms_path, transforms_record = _verify_record(
        source_transforms, role=f"{role} source transforms"
    )
    _require(
        transforms_path == PUBLIC_DATASET / "transforms.json"
        and transforms_record["sha256"] == value.get("source_transforms_sha256"),
        f"{role} source transforms changed",
    )
    _require(
        all(
            transforms_record[key] == canonical["raw_transforms"][key]
            for key in ("path", "size_bytes", "sha256")
        )
        and value.get("source_seed_ply_path") == os.fspath(canonical["seed_path"]),
        f"{role} source transforms or seed declaration differ from canonical",
    )
    local_verified: dict[str, dict[str, Any]] = {}
    retained: set[Path] = set()
    for name, record in materialized.items():
        name = _canonical_relative_posix(name, role=f"{role} materialized name")
        expected = expected_root / name
        path, observed = _verify_record(
            record, role=f"{role} materialized {name}", root=root
        )
        _require(path == expected, f"{role} materialized path changed: {name}")
        local_verified[name] = observed
        retained.add(path)
    _require(
        "transforms.json" in local_verified, f"{role} transforms were not retained"
    )
    materialized_without_transforms = dict(local_verified)
    transforms = materialized_without_transforms.pop("transforms.json")
    _require(
        set(source_verified)
        == set(materialized_without_transforms)
        == set(local["referenced_records"])
        and all(
            source_verified[name][key] == materialized_without_transforms[name][key]
            for name in source_verified
            for key in ("sha256", "size_bytes")
        ),
        f"{role} materialized content differs from source",
    )
    _require(
        value.get("referenced_source_content")
        == _content_identity(source_verified)
        == canonical.get("referenced_file_content")
        and value.get("referenced_materialized_content")
        == _content_identity(materialized_without_transforms)
        == local.get("referenced_file_content"),
        f"{role} declared content identity changed",
    )
    _require(
        _strict_int(value.get("frame_count"), label=f"{role} frame count", minimum=1)
        == canonical.get("frame_count")
        == local.get("frame_count")
        and _strict_int(
            value.get("frame_count"), label=f"{role} frame count", minimum=1
        )
        + 2
        == _strict_int(
            value.get("copied_regular_file_count"),
            label=f"{role} copied file count",
            minimum=2,
        )
        and len(local_verified) == value.get("copied_regular_file_count"),
        f"{role} copied input count changed",
    )
    transforms_payload = _read_stable_file(
        expected_root / "transforms.json", role=f"{role} transforms"
    )[1]
    _require(
        hashlib.sha256(transforms_payload).hexdigest()
        == value.get("materialized_transforms_sha256"),
        f"{role} materialized transforms digest changed",
    )
    try:
        transforms_json = json.loads(transforms_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} transforms are invalid") from error
    _require(isinstance(transforms_json, dict), f"{role} transforms are not an object")
    seed = transforms_json.get("ply_file_path")
    _require(
        seed == value.get("materialized_seed_ply_path") == os.fspath(local["seed_path"])
        and _descendant(root, seed, role=f"{role} seed") in retained,
        f"{role} seed binding changed",
    )
    portable = json.loads(json.dumps(transforms_json, allow_nan=False))
    portable["ply_file_path"] = "<MATERIALIZED-SEED-PLY>"
    _require(
        hashlib.sha256(_canonical_bytes(portable)).hexdigest()
        == value.get("portable_transforms_sha256")
        == local.get("portable_transforms_sha256")
        == canonical.get("portable_transforms_sha256"),
        f"{role} portable transforms differ from the canonical public dataset",
    )
    _require(
        transforms["sha256"] == value.get("materialized_transforms_sha256"),
        f"{role} transforms record changed",
    )
    return retained


def _validate_dataset_closure(
    value: object,
    *,
    root: Path,
    expected_root: Path,
    role: str,
    require_local: bool,
    canonical_dataset: Mapping[str, Any] | None = None,
) -> set[Path]:
    """Validate the analyzer's declared-input closure without trusting booleans."""
    canonical = (
        _recompute_dataset_identity(PUBLIC_DATASET, role="canonical public dataset")
        if canonical_dataset is None
        else canonical_dataset
    )
    local = _recompute_dataset_identity(expected_root, role=f"{role} inputs")
    _require(isinstance(value, Mapping), f"{role} closure is absent")
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "root",
        "raw_transforms",
        "transforms_relative_path",
        "seed_relative_path",
        "seed_reference",
        "frame_count",
        "regular_input_file_count",
        "referenced_input_bindings",
        "content_identity",
        "content_artifact_sha256",
        "generated_outputs_excluded",
        "symlinks_special_files_and_hardlink_aliases_accepted",
    }
    _require(set(value) == expected_fields, f"{role} closure fields changed")
    _require(
        value.get("schema_version") == 1
        and value.get("artifact_kind") == "Deform360DeclaredDatasetInputClosureV1"
        and value.get("root") == os.fspath(expected_root)
        and value.get("transforms_relative_path") == "transforms.json"
        and value.get("generated_outputs_excluded") is True
        and value.get("symlinks_special_files_and_hardlink_aliases_accepted") is False,
        f"{role} closure identity changed",
    )
    raw_path, raw_record = _verify_record(
        value.get("raw_transforms"),
        role=f"{role} raw transforms",
        root=root if require_local else None,
    )
    _require(
        raw_path == expected_root / "transforms.json"
        and all(
            raw_record[key] == local["raw_transforms"][key]
            for key in ("path", "size_bytes", "sha256")
        ),
        f"{role} transforms path changed",
    )
    bindings = value.get("referenced_input_bindings")
    _require(isinstance(bindings, list) and bindings, f"{role} bindings are absent")
    retained: set[Path] = {raw_path} if require_local else set()
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, binding in enumerate(bindings):
        _require(
            isinstance(binding, Mapping)
            and set(binding)
            == {"role", "relative_path", "path", "size_bytes", "sha256", "mode_octal"},
            f"{role} binding {index} fields changed",
        )
        name = _canonical_relative_posix(
            binding.get("relative_path"), role=f"{role} binding {index} path"
        )
        entry_role = binding.get("role")
        _require(
            entry_role in {"seed_ply", "frame_image"} and name not in names,
            f"{role} binding {index} identity changed",
        )
        names.add(name)
        path, observed = _verify_record(
            {
                key: binding[key]
                for key in ("path", "size_bytes", "sha256", "mode_octal")
            },
            role=f"{role} input {name}",
            root=root if require_local else None,
        )
        _require(path == expected_root / name, f"{role} input path changed")
        if require_local:
            retained.add(path)
        rows.append(
            {
                "role": entry_role,
                "relative_path": name,
                "size_bytes": observed["size_bytes"],
                "sha256": observed["sha256"],
            }
        )
    rows.sort(key=lambda item: (item["relative_path"], item["role"]))
    _require(
        len(rows) + 1
        == _strict_int(
            value.get("regular_input_file_count"),
            label=f"{role} input count",
            minimum=2,
        )
        and _strict_int(
            value.get("frame_count"), label=f"{role} frame count", minimum=1
        )
        == sum(row["role"] == "frame_image" for row in rows)
        == local.get("frame_count")
        == canonical.get("frame_count"),
        f"{role} declared counts changed",
    )
    normalized = value.get("content_identity")
    _require(isinstance(normalized, Mapping), f"{role} content identity is absent")
    expected_content = {
        "normalized_transforms": local["content_identity"]["normalized_transforms"],
        "referenced_files": rows,
    }
    transforms_payload = _read_stable_file(raw_path, role=f"{role} transforms payload")[
        1
    ]
    try:
        transforms_json = json.loads(transforms_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} transforms are invalid") from error
    _require(isinstance(transforms_json, dict), f"{role} transforms are not an object")
    portable = json.loads(json.dumps(transforms_json, allow_nan=False))
    _require(
        isinstance(portable.get("ply_file_path"), str),
        f"{role} seed declaration is absent",
    )
    portable["ply_file_path"] = "<MATERIALIZED-SEED-PLY>"
    normalized_payload = _canonical_bytes(portable)
    recomputed_normalized = {
        "size_bytes": len(normalized_payload),
        "sha256": hashlib.sha256(normalized_payload).hexdigest(),
    }
    _require(
        dict(normalized) == expected_content, f"{role} content identity rows changed"
    )
    _require(
        normalized.get("normalized_transforms") == recomputed_normalized,
        f"{role} normalized transforms changed",
    )
    _require(
        rows == local.get("referenced_rows")
        and dict(normalized) == local.get("content_identity")
        and dict(normalized) == canonical.get("content_identity"),
        f"{role} content differs from the canonical public dataset",
    )
    frames = transforms_json.get("frames")
    _require(
        isinstance(frames, list)
        and len(frames) == value.get("frame_count")
        and local.get("frame_relative_paths")
        == [
            binding["relative_path"]
            for binding in bindings
            if binding["role"] == "frame_image"
        ],
        f"{role} frame declarations differ from the closure",
    )
    _require(
        value.get("seed_relative_path") == local.get("seed_relative_path")
        and value.get("seed_relative_path") == canonical.get("seed_relative_path")
        and value.get("seed_relative_path")
        in {
            binding["relative_path"]
            for binding in bindings
            if binding["role"] == "seed_ply"
        }
        and value.get("seed_reference")
        == {
            "declared_path": local.get("seed_declared_path"),
            "canonical_absolute_alias_used": False,
        },
        f"{role} seed declaration differs from the closure",
    )
    _require(
        value.get("content_artifact_sha256")
        == hashlib.sha256(_canonical_bytes(expected_content)).hexdigest()
        == local.get("content_artifact_sha256")
        == canonical.get("content_artifact_sha256"),
        f"{role} content identity signature changed",
    )
    _require(raw_record["size_bytes"] > 0, f"{role} transforms are empty")
    if require_local:
        actual_files = {
            path
            for path in expected_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        _require(
            actual_files == retained,
            f"{role} contains undeclared inputs or generated outputs",
        )
    return retained


def _validate_resource_boundary(value: object, *, role: str) -> dict[str, int]:
    _require(isinstance(value, Mapping), f"{role} is absent")
    _require(
        set(value)
        == {
            "file_descriptor_count",
            "task_count",
            "rss_kib",
            "rlimit_nofile_soft",
            "rlimit_nofile_hard",
        },
        f"{role} fields changed",
    )
    result = {
        "file_descriptor_count": _strict_int(
            value["file_descriptor_count"], label=f"{role} FD", minimum=1
        ),
        "task_count": _strict_int(
            value["task_count"], label=f"{role} tasks", minimum=1
        ),
        "rss_kib": _strict_int(value["rss_kib"], label=f"{role} RSS", minimum=1),
        "rlimit_nofile_soft": _strict_int(
            value["rlimit_nofile_soft"], label=f"{role} soft NOFILE", minimum=1
        ),
        "rlimit_nofile_hard": _strict_int(
            value["rlimit_nofile_hard"], label=f"{role} hard NOFILE", minimum=1
        ),
    }
    _require(
        result["rlimit_nofile_hard"] >= result["rlimit_nofile_soft"],
        f"{role} NOFILE ordering changed",
    )
    return result


def _validate_global_snapshot(value: object, *, role: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{role} is absent")
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
        f"{role} fields changed",
    )
    for key in (
        "event_writers_object_id",
        "event_storage_object_id",
        "global_buffer_object_id",
        "profiler_object_id",
    ):
        _strict_int(value[key], label=f"{role} {key}", minimum=1)
    for key in ("event_writer_ids", "event_storage_ids", "profiler_ids"):
        _require(isinstance(value[key], list), f"{role} {key} changed")
        for index, item in enumerate(value[key]):
            _strict_int(item, label=f"{role} {key} {index}", minimum=1)
    _require(
        isinstance(value["global_buffer_items"], list), f"{role} global buffer changed"
    )
    for index, item in enumerate(value["global_buffer_items"]):
        _require(
            isinstance(item, list) and len(item) == 2 and isinstance(item[0], str),
            f"{role} global buffer {index} changed",
        )
        _strict_int(item[1], label=f"{role} global buffer {index}", minimum=1)
    profiler = value["pytorch_profiler_id"]
    _require(
        profiler is None
        or (
            isinstance(profiler, int)
            and not isinstance(profiler, bool)
            and profiler > 0
        ),
        f"{role} PyTorch profiler changed",
    )
    return dict(value)


def _validate_runtime(value: object, *, role: str) -> None:
    _require(isinstance(value, Mapping), f"{role} runtime is absent")
    _require(
        set(value)
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
        }
        and value.get("seed") == 0
        and value.get("python_random_seeded") is True
        and value.get("numpy_seeded") is True
        and value.get("torch_cpu_seeded") is True
        and value.get("torch_cuda_seeded") is True
        and value.get("torch_version") == PINNED_TORCH_VERSION
        and value.get("torch_cuda_version") == PINNED_TORCH_CUDA_VERSION
        and value.get("cuda_device_name") == PINNED_GPU_NAME
        and value.get("cuda_device_count") == 1
        and isinstance(value.get("python_version"), str)
        and str(value.get("python_version")).startswith("3.12."),
        f"{role} runtime changed",
    )


def _validate_smoke(value: object, *, role: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{role} smoke is absent")
    expected_fields = {
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
    }
    _require(
        set(value) == expected_fields
        and value.get("artifact_sha256") == _artifact_sha256(value),
        f"{role} smoke signature or fields changed",
    )
    _require(
        value.get("schema_version") == 1
        and value.get("artifact_kind") == "Deform360HeldGsplatRuntimeSmokeV1"
        and value.get("contract_sha256") == PINNED_GSPLAT_SMOKE_CONTRACT_SHA256
        and value.get("physical_gpu_index") == 1
        and value.get("logical_device") == "cuda:0"
        and value.get("gpu_name") == PINNED_GPU_NAME
        and value.get("compute_capability") == "8.9"
        and value.get("python_version") == "3.12"
        and value.get("torch_version") == PINNED_TORCH_VERSION
        and value.get("torch_cuda_version") == PINNED_TORCH_CUDA_VERSION
        and value.get("gsplat_version") == PINNED_GSPLAT_VERSION
        and value.get("extension_path") == os.fspath(PINNED_GSPLAT_EXTENSION_PATH)
        and value.get("extension_sha256") == PINNED_GSPLAT_EXTENSION_SHA256
        and value.get("extension_loaded_and_retained") is True
        and value.get("nvcc_visible") is False
        and value.get("ninja_visible") is False
        and value.get("target_or_outcome_path_accessed") is False,
        f"{role} frozen GPU runtime changed",
    )
    predicates = value.get("predicates")
    expected_predicates = {
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
    }
    _require(predicates == expected_predicates, f"{role} smoke predicates changed")
    return dict(value)


def _expected_environment(root: Path) -> dict[str, str]:
    return {**REQUIRED_EXECUTION_ENVIRONMENT, "TMPDIR": os.fspath(root / "tmp")}


def _python_prefix(source: Path) -> list[str]:
    return [
        os.fspath(PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={V8_PYCACHE_PREFIX}",
        os.fspath(source),
    ]


def _validate_invocation(
    value: object,
    *,
    expected_command: Sequence[str],
    expected_log: Path,
    root: Path,
    role: str,
    expected_return_code: int,
) -> Path:
    _require(isinstance(value, Mapping), f"{role} invocation is absent")
    _require(
        set(value)
        == {
            "command",
            "environment",
            "return_code",
            "timed_out",
            "timeout_error",
            "timeout_seconds",
            "log",
        },
        f"{role} invocation fields changed",
    )
    _require(
        value.get("command") == list(expected_command)
        and value.get("environment") == _expected_environment(root)
        and value.get("return_code") == expected_return_code
        and value.get("timed_out") is False
        and value.get("timeout_error") is None,
        f"{role} invocation changed",
    )
    expected_timeout = (
        FIT_TIMEOUT_SECONDS
        if role.startswith(("original ", "wrapped "))
        else SOAK_TIMEOUT_SECONDS
        if role == "soak"
        else ANALYZER_TIMEOUT_SECONDS
    )
    _require(
        value.get("timeout_seconds") == expected_timeout, f"{role} timeout changed"
    )
    log, _ = _verify_record(value.get("log"), role=f"{role} log", root=root)
    _require(log == expected_log, f"{role} log path changed")
    return log


def _fit_command(
    *, root: Path, code: Path, qualifier: Path, mode: str, pairing_id: str
) -> list[str]:
    fit_root = root / "ab" / mode / pairing_id
    return [
        *_python_prefix(qualifier),
        "_fit-child",
        "--code-root",
        os.fspath(code),
        "--deform360-repo",
        os.fspath(PINNED_DEFORM360),
        "--dataset",
        os.fspath(fit_root / "dataset"),
        "--output-dir",
        os.fspath(fit_root / "export"),
        "--result",
        os.fspath(fit_root / "fit-evidence.json"),
        "--iterations",
        str(FIT_ITERATIONS),
        "--seed",
        str(FIT_SEED),
        "--variant",
        mode,
    ]


def _validate_fit_child(
    evidence: object,
    *,
    root: Path,
    mode: str,
    pairing_id: str,
    dataset_root: Path,
    ply_path: Path,
    adapter_path: Path,
) -> None:
    _require(isinstance(evidence, Mapping), f"{mode} {pairing_id} child is absent")
    expected_fields = {
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
    }
    _require(
        set(evidence) == expected_fields
        and evidence.get("artifact_sha256") == _artifact_sha256(evidence),
        f"{mode} {pairing_id} child signature or fields changed",
    )
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_kind") == FIT_KIND
        and evidence.get("qualification_id") == QUALIFICATION_ID
        and evidence.get("variant") == mode
        and evidence.get("passed") is True
        and evidence.get("parameters")
        == {"iterations": FIT_ITERATIONS, "seed": FIT_SEED}
        and evidence.get("dataset") == os.fspath(dataset_root)
        and evidence.get("formal_held_path_supplied") is False,
        f"{mode} {pairing_id} child identity changed",
    )
    _validate_runtime(evidence.get("runtime"), role=f"{mode} {pairing_id}")
    gsplat = evidence.get("gsplat_runtime_smoke")
    _require(
        isinstance(gsplat, Mapping)
        and set(gsplat) == {"adapter_source", "evidence", "evidence_artifact_sha256"},
        f"{mode} {pairing_id} gsplat binding changed",
    )
    adapter = _validate_source_record(
        gsplat.get("adapter_source"),
        role=f"{mode} {pairing_id} adapter",
        expected_path=adapter_path,
        expected_sha256=GSPLAT_ADAPTER_SHA256,
    )
    _require(
        adapter["path"] == os.fspath(adapter_path),
        f"{mode} {pairing_id} adapter changed",
    )
    smoke = _validate_smoke(gsplat.get("evidence"), role=f"{mode} {pairing_id}")
    _require(
        gsplat.get("evidence_artifact_sha256") == smoke["artifact_sha256"],
        f"{mode} {pairing_id} smoke binding changed",
    )
    output, _ = _verify_record(
        evidence.get("output"), role=f"{mode} {pairing_id} PLY", root=root
    )
    _require(output == ply_path, f"{mode} {pairing_id} output path changed")
    boundary = evidence.get("resource_boundary")
    _require(
        isinstance(boundary, Mapping) and set(boundary) == {"before", "after"},
        f"{mode} {pairing_id} resource boundary changed",
    )
    before = _validate_resource_boundary(
        boundary["before"], role=f"{mode} {pairing_id} before"
    )
    after = _validate_resource_boundary(
        boundary["after"], role=f"{mode} {pairing_id} after"
    )
    _require(
        before["rlimit_nofile_soft"] == 1024
        and after["rlimit_nofile_soft"] == 1024
        and before["rlimit_nofile_hard"] == after["rlimit_nofile_hard"],
        f"{mode} {pairing_id} NOFILE changed",
    )
    global_state = evidence.get("global_state")
    _require(
        isinstance(global_state, Mapping)
        and set(global_state) == {"before", "after", "restored"},
        f"{mode} {pairing_id} global state changed",
    )
    before_globals = _validate_global_snapshot(
        global_state["before"], role=f"{mode} {pairing_id} globals before"
    )
    after_globals = _validate_global_snapshot(
        global_state["after"], role=f"{mode} {pairing_id} globals after"
    )
    if mode == "wrapped":
        _require(
            global_state.get("restored") is True and after_globals == before_globals,
            f"{mode} {pairing_id} globals were not restored",
        )
    _require(
        evidence.get("predicates")
        == {
            "output_created": True,
            "wrapped_fit_requires_global_restoration": True,
            "rlimit_nofile_soft_is_1024": True,
            "rlimit_nofile_unchanged": True,
            "gsplat_runtime_smoke_validated_and_retained": True,
        },
        f"{mode} {pairing_id} predicates changed",
    )


def _validate_attempt(
    attempt: Mapping[str, Any], *, root: Path, analyzer_source: Path
) -> dict[str, Any]:
    _require(
        set(attempt)
        == {
            "schema_version",
            "artifact_kind",
            "qualification_id",
            "state",
            "output_root",
            "code_revision",
            "generator_profile",
            "physical_gpu_index",
            "frozen_analyzer_source",
            "root_consumption_policy",
            "formal_held_path_supplied",
            "artifact_sha256",
        },
        "qualification attempt fields changed",
    )
    _require(
        attempt.get("schema_version") == 2
        and attempt.get("artifact_kind") == ATTEMPT_KIND
        and attempt.get("qualification_id") == QUALIFICATION_ID
        and attempt.get("state") == "canonical-root-consumed-at-creation"
        and attempt.get("output_root") == os.fspath(root)
        and _valid_oid(attempt.get("code_revision"))
        and attempt.get("generator_profile") == "same-as-analyzer"
        and attempt.get("physical_gpu_index") == 1
        and attempt.get("root_consumption_policy") == ROOT_CONSUMPTION_POLICY
        and attempt.get("formal_held_path_supplied") is False,
        "qualification attempt identity changed",
    )
    _validate_source_record(
        attempt.get("frozen_analyzer_source"),
        role="attempt frozen analyzer",
        expected_path=analyzer_source,
        expected_sha256=ANALYZER_SOURCE_SHA256,
    )
    return dict(attempt)


def _validate_python_binding(value: object, *, role: str) -> None:
    _require(isinstance(value, Mapping), f"{role} is absent")
    _require(
        set(value)
        == {
            "lexical_path",
            "lexical_mode_octal",
            "lexical_symlink_target",
            "resolved_executable",
            "pyvenv_config",
            "frozen_package_inventory",
            "frozen_runtime_tree_manifest",
            "pip_freeze_all",
        }
        and value.get("lexical_path") == os.fspath(PINNED_PYTHON)
        and value.get("lexical_mode_octal") == "0777"
        and value.get("lexical_symlink_target") == "/usr/bin/python3"
        and isinstance(value.get("resolved_executable"), Mapping)
        and value["resolved_executable"].get("path") == PINNED_PYTHON_TARGET
        and value["resolved_executable"].get("sha256") == PINNED_PYTHON_TARGET_SHA256,
        f"{role} changed",
    )
    _validate_source_record(
        value.get("resolved_executable"),
        role=f"{role} resolved executable",
        expected_path=Path(PINNED_PYTHON_TARGET),
        expected_sha256=PINNED_PYTHON_TARGET_SHA256,
    )
    _validate_source_record(
        value.get("pyvenv_config"),
        role=f"{role} pyvenv config",
        expected_path=PINNED_PYTHON_RUNTIME / "pyvenv.cfg",
    )
    freeze = _validate_source_record(
        value.get("frozen_package_inventory"),
        role=f"{role} frozen package inventory",
        expected_path=PINNED_PYTHON_FREEZE,
        expected_sha256=PINNED_PYTHON_FREEZE_SHA256,
    )
    tree = _validate_source_record(
        value.get("frozen_runtime_tree_manifest"),
        role=f"{role} frozen runtime tree manifest",
        expected_path=PINNED_PYTHON_TREE_MANIFEST,
        expected_sha256=PINNED_PYTHON_TREE_MANIFEST_SHA256,
    )
    _require(
        freeze.get("mode_octal") == "0400" and tree.get("mode_octal") == "0400",
        f"{role} frozen records are writable",
    )
    live = value.get("pip_freeze_all")
    _require(
        isinstance(live, Mapping)
        and set(live)
        == {
            "normalized_sha256",
            "normalized_line_count",
            "normalized_size_bytes",
            "equals_frozen_package_inventory",
        }
        and live.get("normalized_sha256") == PINNED_PYTHON_FREEZE_SHA256
        and _strict_int(
            live.get("normalized_line_count"),
            label=f"{role} freeze line count",
            minimum=1,
        )
        > 0
        and _strict_int(
            live.get("normalized_size_bytes"), label=f"{role} freeze size", minimum=1
        )
        > 0
        and live.get("equals_frozen_package_inventory") is True,
        f"{role} live package inventory changed",
    )


def _validate_aggregate_runtime(
    value: object, *, root: Path
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    _require(isinstance(value, Mapping), "aggregate runtime bindings are absent")
    expected_fields = {
        "python_path",
        "python",
        "python_after",
        "parent_python_process",
        "code",
        "code_after",
        "deform360",
        "deform360_after",
        "qualification_source",
        "wrapper_source",
        "analyzer_source",
    }
    _require(set(value) == expected_fields, "aggregate runtime binding fields changed")
    _require(
        value.get("python_path") == os.fspath(PINNED_PYTHON)
        and value.get("python") == value.get("python_after"),
        "aggregate Python changed",
    )
    _validate_python_binding(value.get("python"), role="aggregate Python")
    parent = value.get("parent_python_process")
    _require(
        parent
        == {
            "sys_executable": os.fspath(PINNED_PYTHON),
            "sys_base_executable": PINNED_PYTHON_TARGET,
            "sys_prefix": os.fspath(PINNED_PYTHON_RUNTIME),
            "sys_base_prefix": PINNED_PYTHON_BASE_PREFIX,
        },
        "aggregate parent Python process changed",
    )
    code = _validate_git_binding(value.get("code"), role="aggregate code")
    _require(
        code == value.get("code_after"), "aggregate code changed during qualification"
    )
    code_root = _absolute(code["path"])
    _require(
        root == BASE / f"{ROOT_PREFIX}{code['head']}",
        "qualification root does not bind its H1 revision",
    )
    deform = _validate_git_binding(value.get("deform360"), role="aggregate Deform360")
    _require(
        deform == value.get("deform360_after")
        and deform.get("path") == os.fspath(PINNED_DEFORM360)
        and deform.get("head") == PINNED_DEFORM360_HEAD
        and deform.get("tree") == PINNED_DEFORM360_TREE,
        "aggregate Deform360 binding changed",
    )
    qualifier = (code_root / RELATIVE_QUALIFIER_SOURCE).resolve(strict=True)
    wrapper = (code_root / RELATIVE_WRAPPER_SOURCE).resolve(strict=True)
    analyzer = (code_root / RELATIVE_ANALYZER_SOURCE).resolve(strict=True)
    adapter = (code_root / RELATIVE_GSPLAT_ADAPTER_SOURCE).resolve(strict=True)
    _validate_source_record(
        value.get("qualification_source"),
        role="qualification source",
        expected_path=qualifier,
    )
    _validate_source_record(
        value.get("wrapper_source"),
        role="resource wrapper source",
        expected_path=wrapper,
    )
    _validate_source_record(
        value.get("analyzer_source"),
        role="equivalence analyzer source",
        expected_path=analyzer,
        expected_sha256=ANALYZER_SOURCE_SHA256,
    )
    return code, qualifier, wrapper, analyzer, adapter


def _expected_admission(accepted: bool) -> dict[str, Any]:
    return {
        "decision": "admitted" if accepted else "inconclusive",
        "terminal": True,
        "analyzer_outcome": "accepted" if accepted else "scientific-no-go",
        "analyzer_no_go_interpretation": None if accepted else NO_GO_INTERPRETATION,
        "wrapper_inequivalence_proven": False,
        "retry_permitted": False,
        "in_place_reuse_permitted": False,
    }


def _canonical_parameters() -> dict[str, Any]:
    return {
        "dataset": os.fspath(PUBLIC_DATASET),
        "phase": "all",
        "cuda_device": 1,
        "seed": 0,
        "ab_iterations": FIT_ITERATIONS,
        "ab_repeat_count": REPEAT_COUNT,
        "soak_fit_count": SOAK_FIT_COUNT,
        "soak_iterations": SOAK_ITERATIONS,
        "first_fit_fd_growth_limit": FIRST_FIT_FD_GROWTH_LIMIT,
        "steady_fd_growth_limit": STEADY_FD_GROWTH_LIMIT,
        "steady_task_growth_limit": STEADY_TASK_GROWTH_LIMIT,
        "fit_timeout_seconds": FIT_TIMEOUT_SECONDS,
        "analyzer_timeout_seconds": ANALYZER_TIMEOUT_SECONDS,
        "soak_timeout_seconds": SOAK_TIMEOUT_SECONDS,
    }


def _parameters() -> dict[str, Any]:
    result = dict(_canonical_parameters())
    result.pop("dataset")
    result.pop("phase")
    return result


def _validate_main_identity(artifact: Mapping[str, Any], *, accepted: bool) -> None:
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "qualification_id",
        "status",
        "passed",
        "host",
        "phase",
        "generator_profile",
        "physical_gpu_index",
        "canonical_run_parameters",
        "parameters",
        "execution_order",
        "runtime_bindings",
        "source_dataset",
        "attempt",
        "root_consumption_policy",
        "materialized_datasets",
        "invocations",
        "ab",
        "soak",
        "cleanup_events",
        "admission",
        "predicates",
        "information_boundary",
        "artifact_sha256",
    }
    _require(set(artifact) == expected_fields, "qualification aggregate fields changed")
    _require(
        artifact.get("schema_version") == 2
        and artifact.get("artifact_kind") == QUALIFICATION_KIND
        and artifact.get("qualification_id") == QUALIFICATION_ID
        and artifact.get("status")
        == ("qualified" if accepted else "admission-inconclusive")
        and artifact.get("passed") is accepted
        and artifact.get("host") == EXPECTED_HOST
        and artifact.get("phase") == "all"
        and artifact.get("generator_profile") == "same-as-analyzer"
        and artifact.get("physical_gpu_index") == 1
        and artifact.get("canonical_run_parameters") == _canonical_parameters()
        and artifact.get("parameters") == _parameters()
        and artifact.get("execution_order")
        == [
            "fresh-five-original-and-five-wrapped-fits",
            "equivalence-analyzer",
            "243-fit-soak-only-after-analyzer-acceptance",
        ]
        and artifact.get("source_dataset") == os.fspath(PUBLIC_DATASET)
        and artifact.get("root_consumption_policy") == ROOT_CONSUMPTION_POLICY
        and artifact.get("admission") == _expected_admission(accepted),
        "qualification aggregate identity or terminal outcome changed",
    )
    _require(
        artifact.get("information_boundary")
        == {
            "formal_held_path_accepted": False,
            "formal_target_or_outcome_array_read": False,
            "development_dataset_only": True,
            "unreferenced_source_outputs_copied": False,
            "rlimit_nofile_changed": False,
        },
        "qualification information boundary changed",
    )


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    code: Mapping[str, Any],
    adapter_path: Path,
    aggregate_repeats: Mapping[tuple[str, str], Mapping[str, Any]],
    canonical_dataset: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], set[Path]]:
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
        }
        and manifest.get("schema_version") == 1
        and manifest.get("artifact_kind") == MANIFEST_KIND
        and manifest.get("analysis_id") == ANALYSIS_ID,
        "repeat manifest identity or fields changed",
    )
    expected = manifest.get("expected_environment")
    _require(isinstance(expected, Mapping), "repeat manifest environment is absent")
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
        }
        and expected.get("generator_profile") == "same-as-analyzer"
        and expected.get("physical_gpu_index") == 1
        and expected.get("deform360_git_head") == PINNED_DEFORM360_HEAD
        and expected.get("deform360_git_tree") == PINNED_DEFORM360_TREE
        and expected.get("python_freeze_sha256") == PINNED_PYTHON_FREEZE_SHA256
        and expected.get("python_tree_manifest_sha256")
        == PINNED_PYTHON_TREE_MANIFEST_SHA256,
        "repeat manifest frozen environment changed",
    )
    analyzer_git = _validate_git_binding(
        expected.get("analyzer_code"), role="manifest analyzer code"
    )
    _require(
        analyzer_git == dict(code), "manifest analyzer code differs from aggregate"
    )
    generator = expected.get("generator_code")
    _require(
        isinstance(generator, Mapping)
        and set(generator)
        == {"profile", "qualification_id", "physical_gpu_index", "git", "sources"}
        and generator.get("profile") == "same-as-analyzer"
        and generator.get("qualification_id") == QUALIFICATION_ID
        and generator.get("physical_gpu_index") == 1
        and generator.get("git") == analyzer_git,
        "manifest same-as-analyzer generator changed",
    )
    sources = generator.get("sources")
    expected_source_names = {
        os.fspath(RELATIVE_QUALIFIER_SOURCE),
        os.fspath(RELATIVE_WRAPPER_SOURCE.parent / "deform360_held_gsplat_runtime.py"),
        os.fspath(RELATIVE_WRAPPER_SOURCE),
        os.fspath(RELATIVE_GSPLAT_ADAPTER_SOURCE),
    }
    _require(
        isinstance(sources, Mapping) and set(sources) == expected_source_names,
        "manifest generator source set changed",
    )
    code_root = _absolute(code["path"])
    for name, record in sources.items():
        expected_path = (code_root / name).resolve(strict=True)
        expected_sha = (
            GSPLAT_ADAPTER_SHA256
            if name == os.fspath(RELATIVE_GSPLAT_ADAPTER_SOURCE)
            else None
        )
        _validate_source_record(
            record,
            role=f"manifest generator source {name}",
            expected_path=expected_path,
            expected_sha256=expected_sha,
        )

    canonical = manifest.get("canonical_source_dataset")
    _validate_dataset_closure(
        canonical,
        root=root,
        expected_root=PUBLIC_DATASET,
        role="canonical public dataset",
        require_local=False,
        canonical_dataset=canonical_dataset,
    )
    transforms = manifest.get("canonical_transforms")
    _require(
        isinstance(transforms, Mapping)
        and set(transforms) == {"raw_representative", "normalized"},
        "manifest canonical transforms changed",
    )
    raw_path, raw = _verify_record(
        transforms.get("raw_representative"), role="manifest canonical transforms"
    )
    _require(
        raw_path == PUBLIC_DATASET / "transforms.json"
        and isinstance(canonical, Mapping)
        and all(
            raw[key] == canonical["raw_transforms"][key]
            for key in ("path", "size_bytes", "sha256")
        ),
        "manifest canonical transforms binding changed",
    )
    _require(
        transforms.get("normalized")
        == canonical["content_identity"]["normalized_transforms"],
        "manifest normalized transforms changed",
    )

    modes = manifest.get("modes")
    _require(
        isinstance(modes, Mapping) and set(modes) == {"original", "wrapped"},
        "manifest modes changed",
    )
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    retained: set[Path] = set()
    for mode in ("original", "wrapped"):
        entries = modes.get(mode)
        _require(
            isinstance(entries, list) and len(entries) == REPEAT_COUNT,
            f"manifest {mode} repeat count changed",
        )
        _require(
            [entry.get("pairing_id") for entry in entries if isinstance(entry, Mapping)]
            == list(PAIRING_IDS),
            f"manifest {mode} pairing order changed",
        )
        for index, entry in enumerate(entries):
            pairing_id = PAIRING_IDS[index]
            _require(
                isinstance(entry, Mapping)
                and set(entry)
                == {"pairing_id", "ply", "fit_evidence", "dataset_input_inventory"},
                f"manifest {mode} {pairing_id} fields changed",
            )
            fit_root = root / "ab" / mode / pairing_id
            ply, _ = _verify_record(
                entry.get("ply"), role=f"manifest {mode} {pairing_id} PLY", root=root
            )
            evidence_path, _ = _verify_record(
                entry.get("fit_evidence"),
                role=f"manifest {mode} {pairing_id} evidence",
                root=root,
            )
            _require(
                ply == fit_root / "export/splat.ply"
                and evidence_path == fit_root / "fit-evidence.json",
                f"manifest {mode} {pairing_id} paths changed",
            )
            aggregate = aggregate_repeats[(mode, pairing_id)]
            child = aggregate["child_evidence"]
            _require(
                all(
                    entry["ply"][key] == child["output"][key]
                    for key in ("path", "size_bytes", "sha256")
                ),
                f"manifest {mode} {pairing_id} output differs from child",
            )
            loaded, loaded_record = _load_signed(
                evidence_path, role=f"manifest {mode} {pairing_id} child"
            )
            _require(
                loaded == child
                and all(
                    entry["fit_evidence"][key] == loaded_record[key]
                    for key in ("path", "size_bytes", "sha256")
                ),
                f"manifest {mode} {pairing_id} child binding changed",
            )
            dataset_root = fit_root / "dataset"
            retained |= _validate_dataset_closure(
                entry.get("dataset_input_inventory"),
                root=root,
                expected_root=dataset_root,
                role=f"manifest {mode} {pairing_id} dataset",
                require_local=True,
                canonical_dataset=canonical_dataset,
            )
            _validate_fit_child(
                child,
                root=root,
                mode=mode,
                pairing_id=pairing_id,
                dataset_root=dataset_root,
                ply_path=ply,
                adapter_path=adapter_path,
            )
            retained.update({ply, evidence_path})
            records[(mode, pairing_id)] = entry
    _require(
        set(records) == set(aggregate_repeats),
        "manifest and aggregate repeat sets differ",
    )
    return records, retained


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    numpy = _numpy_module()
    array = numpy.asarray(values, dtype=numpy.float64)
    _require(array.ndim == 1 and array.size > 0, "metric distribution is empty")
    return float(numpy.quantile(array, probability, method="linear"))


def _distribution(records: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [
        _finite(
            record["metrics"].get(metric),
            label=f"pair metric {metric}",
            nonnegative=True,
        )
        for record in records
    ]
    return {
        "count": len(values),
        "minimum": min(values),
        "median": _linear_quantile(values, 0.5),
        "p95": _linear_quantile(values, 0.95),
        "maximum": max(values),
    }


def _expected_pairs(group: str) -> list[tuple[str, str, str, str]]:
    if group == "within_original":
        return [
            ("original", PAIRING_IDS[left], "original", PAIRING_IDS[right])
            for left in range(REPEAT_COUNT)
            for right in range(left + 1, REPEAT_COUNT)
        ]
    if group == "within_wrapped":
        return [
            ("wrapped", PAIRING_IDS[left], "wrapped", PAIRING_IDS[right])
            for left in range(REPEAT_COUNT)
            for right in range(left + 1, REPEAT_COUNT)
        ]
    return [
        ("original", left, "wrapped", right)
        for left in PAIRING_IDS
        for right in PAIRING_IDS
    ]


def _recompute_equivalence(
    result: Mapping[str, Any], *, accepted: bool
) -> dict[str, Any]:
    groups = result.get("pair_groups")
    _require(
        isinstance(groups, Mapping) and set(groups) == set(GROUP_COUNTS),
        "equivalence pair groups changed",
    )
    for group, expected_count in GROUP_COUNTS.items():
        records = groups.get(group)
        expected_pairs = _expected_pairs(group)
        _require(
            isinstance(records, list) and len(records) == expected_count,
            f"equivalence {group} count changed",
        )
        for index, record in enumerate(records):
            _require(
                isinstance(record, Mapping)
                and set(record)
                == {
                    "left",
                    "right",
                    "matched_pairing_id",
                    "structured_array_exact",
                    "file_sha256_exact",
                    "metrics",
                },
                f"equivalence {group} pair {index} fields changed",
            )
            left_mode, left_id, right_mode, right_id = expected_pairs[index]
            _require(
                record.get("left") == {"mode": left_mode, "pairing_id": left_id}
                and record.get("right") == {"mode": right_mode, "pairing_id": right_id}
                and record.get("matched_pairing_id") is (left_id == right_id)
                and isinstance(record.get("structured_array_exact"), bool)
                and isinstance(record.get("file_sha256_exact"), bool)
                and isinstance(record.get("metrics"), Mapping)
                and set(record["metrics"]) == set(PAIR_METRIC_NAMES),
                f"equivalence {group} pair {index} identity changed",
            )
            for metric in PAIR_METRIC_NAMES:
                _finite(
                    record["metrics"][metric],
                    label=f"{group} {index} {metric}",
                    nonnegative=True,
                )

    distributions = {
        metric: {group: _distribution(groups[group], metric) for group in GROUP_COUNTS}
        for metric in PAIR_METRIC_NAMES
    }
    _require(
        result.get("metric_distributions") == distributions,
        "equivalence metric distributions were not independently reproduced",
    )
    per_metric: dict[str, Any] = {}
    for metric, distribution in distributions.items():
        within_p95 = max(
            distribution["within_original"]["p95"],
            distribution["within_wrapped"]["p95"],
        )
        within_max = max(
            distribution["within_original"]["maximum"],
            distribution["within_wrapped"]["maximum"],
        )
        cross_median = distribution["cross_mode"]["median"]
        cross_p95 = distribution["cross_mode"]["p95"]
        median_passed = cross_median <= within_p95
        p95_passed = cross_p95 <= within_max
        per_metric[metric] = {
            "cross_median": cross_median,
            "within_p95_limit": within_p95,
            "cross_median_condition_passed": median_passed,
            "cross_p95": cross_p95,
            "within_max_limit": within_max,
            "cross_p95_condition_passed": p95_passed,
            "passed": bool(median_passed and p95_passed),
        }
    secondary_passed = all(value["passed"] for value in per_metric.values())
    gate = {
        "contract": dict(GATE_CONTRACT),
        "pair_counts": dict(GROUP_COUNTS),
        "per_metric": per_metric,
        "all_metrics_finite_and_nonnegative": True,
        "passed": secondary_passed,
    }
    _require(
        result.get("secondary_distributional_gate") == gate,
        "equivalence secondary gate was not independently reproduced",
    )
    matched = [
        record for record in groups["cross_mode"] if record["matched_pairing_id"]
    ]
    _require(len(matched) == REPEAT_COUNT, "equivalence matched pairs are incomplete")
    exact_primary = all(record["structured_array_exact"] for record in matched)
    exact_files = all(record["file_sha256_exact"] for record in matched)
    recomputed_accepted = bool(exact_primary or secondary_passed)
    basis = (
        "exact-structured-array-equality"
        if exact_primary
        else ("secondary-distributional-envelope" if secondary_passed else "rejected")
    )
    decision = {
        "exact_matched_structured_array_equality_primary_passed": exact_primary,
        "exact_matched_file_bytes_equal": exact_files,
        "secondary_distributional_equivalence_passed": secondary_passed,
        "accepted": recomputed_accepted,
        "acceptance_basis": basis,
    }
    _require(
        result.get("decision") == decision and recomputed_accepted is accepted,
        "equivalence decision was not independently reproduced",
    )
    return decision


def _validate_result_runtime(result: Mapping[str, Any], *, root: Path) -> None:
    execution = result.get("execution_binding")
    _require(isinstance(execution, Mapping), "analyzer execution binding is absent")
    environment = execution.get("environment")
    required = dict(REQUIRED_EXECUTION_ENVIRONMENT)
    required.pop("CUDA_VISIBLE_DEVICES")
    _require(
        execution.get("host") == EXPECTED_HOST
        and execution.get("process_flags")
        == {
            "isolated": 1,
            "no_user_site": 1,
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "safe_path": True,
        }
        and isinstance(environment, Mapping)
        and environment.get("required") == required
        and environment.get("cuda_visible_devices") == "1"
        and environment.get("forbidden_absent") == list(FORBIDDEN_EXECUTION_ENVIRONMENT)
        and environment.get("tmpdir") == os.fspath(root / "tmp"),
        "analyzer process or sanitized environment changed",
    )
    live = execution.get("live_device")
    _require(
        isinstance(live, Mapping)
        and live.get("visible_cuda_device_count") == 1
        and live.get("logical_device") == "cuda:0"
        and live.get("gpu_name") == PINNED_GPU_NAME
        and live.get("compute_capability") == "8.9"
        and live.get("torch_version") == PINNED_TORCH_VERSION
        and live.get("torch_cuda_version") == PINNED_TORCH_CUDA_VERSION,
        "analyzer live GPU binding changed",
    )
    runtime = result.get("runtime_binding")
    _require(
        isinstance(runtime, Mapping)
        and runtime.get("frozen_package_inventory", {}).get("sha256")
        == PINNED_PYTHON_FREEZE_SHA256
        and runtime.get("frozen_runtime_tree_manifest", {}).get("sha256")
        == PINNED_PYTHON_TREE_MANIFEST_SHA256
        and runtime.get("live_pip_freeze_all", {}).get("normalized_sha256")
        == PINNED_PYTHON_FREEZE_SHA256
        and runtime.get("live_pip_freeze_all", {}).get(
            "equals_frozen_package_inventory"
        )
        is True,
        "analyzer pinned Python runtime changed",
    )


def _validate_result(
    result: Mapping[str, Any],
    *,
    root: Path,
    manifest: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    code: Mapping[str, Any],
    analyzer_path: Path,
    accepted: bool,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "analysis_id",
        "development_only",
        "formal_path_accessed",
        "host",
        "generator_profile",
        "physical_gpu_index",
        "input_manifest",
        "source_bindings",
        "runtime_binding",
        "execution_binding",
        "gsplat_runtime",
        "renderer_sys_path_restoration",
        "canonical_source_dataset",
        "canonical_transforms",
        "inputs",
        "pre_post_render_stability",
        "statistical_limitations",
        "schema_validation",
        "render_execution",
        "pair_groups",
        "metric_distributions",
        "secondary_distributional_gate",
        "decision",
        "artifact_sha256",
    }
    _require(set(result) == expected_fields, "analysis result fields changed")
    _require(
        result.get("schema_version") == 1
        and result.get("artifact_kind") == RESULT_KIND
        and result.get("analysis_id") == ANALYSIS_ID
        and result.get("development_only") is True
        and result.get("formal_path_accessed") is False
        and result.get("host") == EXPECTED_HOST
        and result.get("generator_profile") == "same-as-analyzer"
        and result.get("physical_gpu_index") == 1,
        "analysis result identity changed",
    )
    _require(
        all(
            result["input_manifest"].get(key) == manifest_record.get(key)
            for key in ("path", "size_bytes", "sha256")
        )
        and result["input_manifest"].get("artifact_sha256")
        == manifest.get("artifact_sha256"),
        "analysis result manifest binding changed",
    )
    source = result.get("source_bindings")
    _require(
        isinstance(source, Mapping) and source.get("analyzer_git") == dict(code),
        "analysis result code binding changed",
    )
    _validate_source_record(
        source.get("analyzer_source"),
        role="result analyzer source",
        expected_path=analyzer_path,
        expected_sha256=ANALYZER_SOURCE_SHA256,
    )
    _validate_result_runtime(result, root=root)
    gsplat = result.get("gsplat_runtime")
    _require(
        isinstance(gsplat, Mapping)
        and gsplat.get("host") == EXPECTED_HOST
        and gsplat.get("physical_gpu_index") == 1,
        "analysis gsplat execution binding changed",
    )
    _validate_smoke(gsplat.get("smoke_evidence"), role="analyzer")
    _require(
        result.get("renderer_sys_path_restoration", {}).get("restored_exactly") is True,
        "analyzer renderer import path was not restored",
    )
    _require(
        result.get("canonical_source_dataset")
        == manifest.get("canonical_source_dataset"),
        "result canonical dataset differs from manifest",
    )
    _require(
        all(
            result.get("canonical_transforms", {}).get(key)
            == manifest["canonical_transforms"]["raw_representative"].get(key)
            for key in ("path", "size_bytes", "sha256")
        ),
        "result canonical transforms differ from manifest",
    )
    result_inputs = result.get("inputs")
    manifest_modes = manifest.get("modes")
    _require(
        isinstance(result_inputs, Mapping)
        and isinstance(manifest_modes, Mapping)
        and set(result_inputs) == {"original", "wrapped"},
        "analysis transitive inputs are absent",
    )
    for mode in ("original", "wrapped"):
        _require(
            isinstance(result_inputs[mode], list)
            and len(result_inputs[mode]) == REPEAT_COUNT,
            f"analysis {mode} transitive input count changed",
        )
        for index, item in enumerate(result_inputs[mode]):
            declared = manifest_modes[mode][index]
            _require(
                isinstance(item, Mapping)
                and set(item)
                == {"pairing_id", "ply", "fit_evidence", "dataset_input_inventory"}
                and item.get("pairing_id") == declared.get("pairing_id")
                and all(
                    item["ply"].get(key) == declared["ply"].get(key)
                    for key in ("path", "size_bytes", "sha256")
                )
                and all(
                    item["fit_evidence"].get(key) == declared["fit_evidence"].get(key)
                    for key in ("path", "size_bytes", "sha256")
                )
                and item["fit_evidence"].get("artifact_sha256")
                == _load_signed(
                    Path(item["fit_evidence"]["path"]),
                    role=f"result {mode} fit evidence",
                )[0].get("artifact_sha256")
                and item.get("dataset_input_inventory")
                == declared.get("dataset_input_inventory"),
                f"analysis {mode} transitive input {index} differs from manifest",
            )
    _require(
        result.get("schema_validation")
        == {
            "expected_field_count": 62,
            "expected_field_names": result["schema_validation"].get(
                "expected_field_names"
            ),
            "all_property_declarations_literal_float_f4": True,
            "inert_normal_fields": ["nx", "ny", "nz"],
            "all_inert_normal_values_exactly_zero": True,
            "inert_normal_fields_excluded_from_distribution_metrics": True,
            "identical_schema_across_all_plys": True,
            "all_source_and_derived_values_finite": True,
        }
        and isinstance(result["schema_validation"].get("expected_field_names"), list)
        and len(result["schema_validation"]["expected_field_names"]) == 62,
        "analysis PLY schema validation changed",
    )
    render = result.get("render_execution")
    expected_calls = [
        {"mode": mode, "pairing_id": pairing_id}
        for mode in ("original", "wrapped")
        for pairing_id in PAIRING_IDS
    ]
    _require(
        isinstance(render, Mapping)
        and render.get("calls") == expected_calls
        and render.get("render_call_count") == 10
        and render.get("each_ply_rendered_exactly_once") is True
        and isinstance(render.get("contract"), Mapping)
        and render["contract"].get("one_batched_rasterization_call_per_ply") is True
        and render["contract"].get("camera_count") == 21
        and render["contract"].get("integer_downscale") == 4,
        "analysis render execution changed",
    )
    stability = result.get("pre_post_render_stability")
    _require(
        isinstance(stability, Mapping)
        and stability.get("before") == stability.get("after")
        and stability.get("exact_equal") is True
        and stability.get("analyzer_gsplat_smoke_executed_once_before_render") is True
        and stability.get("adapter_and_aot_bytes_revalidated_after_render") is True,
        "analysis pre/post state changed",
    )
    before_state = stability["before"]
    _require(
        isinstance(before_state, Mapping)
        and before_state.get("manifest") == result.get("input_manifest")
        and before_state.get("canonical_source_dataset")
        == result.get("canonical_source_dataset")
        and before_state.get("canonical_transforms")
        == result.get("canonical_transforms")
        and before_state.get("transitive_inputs") == result.get("inputs")
        and before_state.get("source") == result.get("source_bindings")
        and before_state.get("runtime") == result.get("runtime_binding")
        and before_state.get("execution") == result.get("execution_binding"),
        "analysis embedded pre/post state is not self-consistent",
    )
    return _recompute_equivalence(result, accepted=accepted)


def _recompute_soak_evaluation(
    before: Mapping[str, int], fits: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    fd_values = [int(fit["resource_boundary"]["file_descriptor_count"]) for fit in fits]
    task_values = [int(fit["resource_boundary"]["task_count"]) for fit in fits]
    rss_values = [int(fit["resource_boundary"]["rss_kib"]) for fit in fits]
    first = dict(fits[0]["resource_boundary"])
    reinitialization = [
        index for index, fit in enumerate(fits) if fit["trainer_reinitialized"]
    ]
    expected_reinitialization = list(
        range(0, SOAK_FIT_COUNT, SOAK_TRAINER_REINITIALIZATION_INTERVAL)
    )
    before_rlimit = (before["rlimit_nofile_soft"], before["rlimit_nofile_hard"])
    predicates = {
        "fit_count_exact": len(fits) == SOAK_FIT_COUNT
        and [fit["fit_index"] for fit in fits] == list(range(SOAK_FIT_COUNT)),
        "trainer_reinitialization_indices_exact": reinitialization
        == expected_reinitialization,
        "all_fits_created_outputs": all(fit["output_created"] is True for fit in fits),
        "dataset_outputs_created_after_every_fit": all(
            fit["dataset_outputs_created"] is True for fit in fits
        ),
        "cleanup_completed_after_every_fit": all(
            fit["cleanup_completed"] is True for fit in fits
        ),
        "output_ply_absent_after_every_fit": all(
            fit["output_ply_absent_after_cleanup"] is True for fit in fits
        ),
        "dataset_outputs_absent_after_every_fit": all(
            fit["dataset_outputs_absent_after_cleanup"] is True for fit in fits
        ),
        "resource_boundary_recorded_after_cleanup": all(
            fit["resource_boundary_stage"] == "after_cleanup" for fit in fits
        ),
        "globals_restored_after_every_fit": all(
            fit["globals_restored"] is True for fit in fits
        ),
        "first_fit_fd_growth_within_limit": fd_values[0]
        <= before["file_descriptor_count"] + FIRST_FIT_FD_GROWTH_LIMIT,
        "steady_fd_growth_within_limit": max(fd_values)
        <= fd_values[0] + STEADY_FD_GROWTH_LIMIT,
        "steady_task_growth_within_limit": max(task_values)
        <= task_values[0] + STEADY_TASK_GROWTH_LIMIT,
        "resource_counts_positive": min(fd_values) > 0
        and min(task_values) > 0
        and min(rss_values) > 0,
        "rlimit_nofile_soft_is_1024": before_rlimit[0] == 1024,
        "rlimit_nofile_unchanged": all(
            (
                fit["resource_boundary"]["rlimit_nofile_soft"],
                fit["resource_boundary"]["rlimit_nofile_hard"],
            )
            == before_rlimit
            for fit in fits
        ),
    }
    return {
        "passed": all(predicates.values()),
        "predicates": predicates,
        "reference": {"pre_fit": dict(before), "first_post_cleanup_fit": first},
        "observed": {
            "minimum_fd_count": min(fd_values),
            "maximum_fd_count": max(fd_values),
            "final_fd_count": fd_values[-1],
            "maximum_fd_growth_from_first_post_fit": max(fd_values) - fd_values[0],
            "minimum_task_count": min(task_values),
            "maximum_task_count": max(task_values),
            "final_task_count": task_values[-1],
            "maximum_task_growth_from_first_post_fit": max(task_values)
            - task_values[0],
            "minimum_rss_kib": min(rss_values),
            "maximum_rss_kib": max(rss_values),
            "final_rss_kib": rss_values[-1],
        },
        "limits": {
            "first_fit_fd_growth": FIRST_FIT_FD_GROWTH_LIMIT,
            "steady_fd_growth": STEADY_FD_GROWTH_LIMIT,
            "steady_task_growth": STEADY_TASK_GROWTH_LIMIT,
        },
        "trainer_reinitialization": {
            "interval": SOAK_TRAINER_REINITIALIZATION_INTERVAL,
            "expected_indices": expected_reinitialization,
            "observed_indices": reinitialization,
        },
    }


def _validate_soak_child(
    evidence: object, *, dataset_root: Path, output_root: Path, adapter_path: Path
) -> None:
    _require(isinstance(evidence, Mapping), "soak child is absent")
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "qualification_id",
        "passed",
        "parameters",
        "runtime",
        "gsplat_runtime_smoke",
        "dataset",
        "initial_global_state",
        "fits",
        "evaluation",
        "formal_held_path_supplied",
        "artifact_sha256",
    }
    _require(
        set(evidence) == expected_fields
        and evidence.get("artifact_sha256") == _artifact_sha256(evidence),
        "soak child signature or fields changed",
    )
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_kind") == SOAK_KIND
        and evidence.get("qualification_id") == QUALIFICATION_ID
        and evidence.get("passed") is True
        and evidence.get("parameters")
        == {
            "fit_count": SOAK_FIT_COUNT,
            "iterations_per_fit": SOAK_ITERATIONS,
            "seed": 0,
            "trainer_reinitialization_interval": SOAK_TRAINER_REINITIALIZATION_INTERVAL,
        }
        and evidence.get("dataset") == os.fspath(dataset_root)
        and evidence.get("formal_held_path_supplied") is False,
        "soak child identity changed",
    )
    _validate_runtime(evidence.get("runtime"), role="soak")
    gsplat = evidence.get("gsplat_runtime_smoke")
    _require(
        isinstance(gsplat, Mapping)
        and set(gsplat) == {"adapter_source", "evidence", "evidence_artifact_sha256"},
        "soak gsplat binding changed",
    )
    smoke = _validate_smoke(gsplat.get("evidence"), role="soak")
    _require(
        gsplat.get("evidence_artifact_sha256") == smoke["artifact_sha256"],
        "soak smoke binding changed",
    )
    _validate_source_record(
        gsplat.get("adapter_source"),
        role="soak gsplat adapter",
        expected_path=adapter_path,
        expected_sha256=GSPLAT_ADAPTER_SHA256,
    )
    initial = _validate_global_snapshot(
        evidence.get("initial_global_state"), role="soak initial globals"
    )
    fits = evidence.get("fits")
    _require(
        isinstance(fits, list) and len(fits) == SOAK_FIT_COUNT, "soak fit count changed"
    )
    expected_fit_fields = {
        "fit_index",
        "trainer_reinitialized",
        "output_created",
        "dataset_outputs_created",
        "output_size_bytes",
        "cleanup_completed",
        "cleanup",
        "output_ply_absent_after_cleanup",
        "dataset_outputs_absent_after_cleanup",
        "resource_boundary_stage",
        "resource_boundary",
        "globals_restored",
        "global_state",
    }
    for index, fit in enumerate(fits):
        _require(
            isinstance(fit, Mapping)
            and set(fit) == expected_fit_fields
            and fit.get("fit_index") == index,
            f"soak fit {index} fields or index changed",
        )
        _require(
            fit.get("trainer_reinitialized")
            is (index % SOAK_TRAINER_REINITIALIZATION_INTERVAL == 0)
            and fit.get("output_created") is True
            and fit.get("dataset_outputs_created") is True
            and _strict_int(
                fit.get("output_size_bytes"),
                label=f"soak fit {index} output bytes",
                minimum=1,
            )
            > 0
            and fit.get("cleanup_completed") is True
            and fit.get("output_ply_absent_after_cleanup") is True
            and fit.get("dataset_outputs_absent_after_cleanup") is True
            and fit.get("resource_boundary_stage") == "after_cleanup"
            and fit.get("globals_restored") is True,
            f"soak fit {index} lifecycle changed",
        )
        boundary = _validate_resource_boundary(
            fit.get("resource_boundary"), role=f"soak fit {index} boundary"
        )
        _require(
            boundary["rlimit_nofile_soft"] == 1024, f"soak fit {index} NOFILE changed"
        )
        _require(
            _validate_global_snapshot(
                fit.get("global_state"), role=f"soak fit {index} globals"
            )
            == initial,
            f"soak fit {index} globals were not restored",
        )
        cleanup = fit.get("cleanup")
        _require(
            isinstance(cleanup, Mapping)
            and set(cleanup) == {"output_ply", "dataset_outputs"},
            f"soak fit {index} cleanup fields changed",
        )
        _validate_removed_file(
            cleanup["output_ply"],
            expected_path=output_root / f"splat-{index:04d}.ply",
            expected_parent=output_root,
            expected_size=int(fit["output_size_bytes"]),
            role=f"soak fit {index} output",
        )
        _validate_removed_tree(
            cleanup["dataset_outputs"],
            expected_root=dataset_root / "outputs",
            expected_parent=dataset_root,
            role=f"soak fit {index} dataset outputs",
        )
    before = _validate_resource_boundary(
        evidence["evaluation"]["reference"]["pre_fit"], role="soak pre-fit boundary"
    )
    recomputed = _recompute_soak_evaluation(before, fits)
    _require(
        evidence.get("evaluation") == recomputed and recomputed["passed"] is True,
        "soak resource evaluation was not independently reproduced",
    )


def _soak_command(*, root: Path, code: Path, qualifier: Path) -> list[str]:
    soak = root / "soak"
    return [
        *_python_prefix(qualifier),
        "_soak-child",
        "--code-root",
        os.fspath(code),
        "--deform360-repo",
        os.fspath(PINNED_DEFORM360),
        "--dataset",
        os.fspath(soak / "dataset"),
        "--output-dir",
        os.fspath(soak / "export"),
        "--result",
        os.fspath(soak / "soak-evidence.json"),
        "--iterations",
        str(SOAK_ITERATIONS),
        "--seed",
        "0",
        "--fit-count",
        str(SOAK_FIT_COUNT),
        "--first-fd-growth-limit",
        str(FIRST_FIT_FD_GROWTH_LIMIT),
        "--steady-fd-growth-limit",
        str(STEADY_FD_GROWTH_LIMIT),
        "--steady-task-growth-limit",
        str(STEADY_TASK_GROWTH_LIMIT),
    ]


def _prepare_manifest_command(*, root: Path, code: Path, analyzer: Path) -> list[str]:
    command = [*_python_prefix(analyzer), "prepare-manifest"]
    for mode in ("original", "wrapped"):
        flag = "--original" if mode == "original" else "--wrapped"
        for pairing_id in PAIRING_IDS:
            fit = root / "ab" / mode / pairing_id
            command.extend(
                [
                    flag,
                    pairing_id,
                    os.fspath(fit / "export/splat.ply"),
                    os.fspath(fit / "fit-evidence.json"),
                ]
            )
    command.extend(
        [
            "--canonical-transforms",
            os.fspath(PUBLIC_DATASET / "transforms.json"),
            "--code-root",
            os.fspath(code),
            "--generator-code-root",
            os.fspath(code),
            "--generator-profile",
            "same-as-analyzer",
            "--deform360-root",
            os.fspath(PINNED_DEFORM360),
            "--output",
            os.fspath(root / "equivalence/repeat-manifest.json"),
        ]
    )
    return command


def _analyze_command(*, root: Path, code: Path, analyzer: Path) -> list[str]:
    return [
        *_python_prefix(analyzer),
        "analyze",
        "--manifest",
        os.fspath(root / "equivalence/repeat-manifest.json"),
        "--code-root",
        os.fspath(code),
        "--generator-code-root",
        os.fspath(code),
        "--deform360-root",
        os.fspath(PINNED_DEFORM360),
        "--output",
        os.fspath(root / "equivalence/analysis-result.json"),
    ]


def _validate_removed_tree(
    value: object,
    *,
    expected_root: Path,
    expected_parent: Path,
    role: str,
    recreated: bool = False,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{role} cleanup is absent")
    fields = {
        "bounded_parent",
        "pre_cleanup_inventory",
        "removed",
        "post_cleanup_absent",
    } | ({"recreated_empty"} if recreated else set())
    _require(set(value) == fields, f"{role} cleanup fields changed")
    inventory = value.get("pre_cleanup_inventory")
    _require(
        isinstance(inventory, Mapping)
        and set(inventory)
        == {"root", "entry_count", "regular_file_bytes", "inventory_sha256"}
        and inventory.get("root") == os.fspath(expected_root)
        and _strict_int(
            inventory.get("entry_count"), label=f"{role} entry count", minimum=0
        )
        >= 0
        and _strict_int(
            inventory.get("regular_file_bytes"), label=f"{role} bytes", minimum=0
        )
        >= 0
        and _valid_sha256(inventory.get("inventory_sha256"))
        and value.get("bounded_parent") == os.fspath(expected_parent)
        and value.get("removed") is True
        and value.get("post_cleanup_absent") is True
        and (not recreated or value.get("recreated_empty") is True),
        f"{role} cleanup changed",
    )
    return dict(value)


def _validate_removed_file(
    value: object,
    *,
    expected_path: Path,
    expected_parent: Path,
    expected_size: int,
    role: str,
) -> None:
    _require(
        isinstance(value, Mapping)
        and set(value)
        == {
            "bounded_parent",
            "pre_cleanup_binding",
            "pre_cleanup_link_count",
            "removed",
            "post_cleanup_absent",
        }
        and value.get("bounded_parent") == os.fspath(expected_parent)
        and value.get("pre_cleanup_link_count") == 1
        and value.get("removed") is True
        and value.get("post_cleanup_absent") is True,
        f"{role} cleanup changed",
    )
    binding = value.get("pre_cleanup_binding")
    _require(
        isinstance(binding, Mapping)
        and binding.get("path") == os.fspath(expected_path)
        and binding.get("size_bytes") == expected_size
        and _valid_sha256(binding.get("sha256")),
        f"{role} pre-clean binding changed",
    )


def _validate_repeat_cleanup(
    value: object, *, root: Path, dataset: Path, role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        isinstance(value, Mapping)
        and set(value)
        == {"generated_dataset_outputs", "qualification_temporary_cache"},
        f"{role} cleanup fields changed",
    )
    generated = _validate_removed_tree(
        value["generated_dataset_outputs"],
        expected_root=dataset / "outputs",
        expected_parent=dataset,
        role=f"{role} generated outputs",
    )
    temporary = _validate_removed_tree(
        value["qualification_temporary_cache"],
        expected_root=root / "tmp",
        expected_parent=root,
        role=f"{role} temporary cache",
        recreated=True,
    )
    return generated, temporary


def _ab_predicates(accepted: bool) -> dict[str, bool]:
    return {
        "repeat_count_exact": True,
        "pairing_ids_exact_and_shared": True,
        "all_fit_children_exited_zero": True,
        "all_fit_evidence_and_outputs_valid": True,
        "all_generated_outputs_and_caches_removed": True,
        "materialized_dataset_content_equal": True,
        "manifest_preparation_passed": True,
        "analyzer_exit_matches_signed_decision": True,
        "equivalence_analysis_accepted": accepted,
    }


def _validate_aggregate_repeats(
    ab: object,
    *,
    root: Path,
    code: Path,
    qualifier: Path,
    adapter: Path,
    datasets: Mapping[str, Any],
    accepted: bool,
    top_invocations: Mapping[str, Any],
    canonical_dataset: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    set[Path],
    list[dict[str, Any]],
]:
    _require(isinstance(ab, Mapping), "aggregate A/B evidence is absent")
    _require(
        set(ab)
        == {
            "passed",
            "repeat_count_per_mode",
            "pairing_ids",
            "repeats",
            "equivalence",
            "predicates",
        }
        and ab.get("passed") is accepted
        and ab.get("repeat_count_per_mode") == REPEAT_COUNT
        and ab.get("pairing_ids") == list(PAIRING_IDS),
        "aggregate A/B identity changed",
    )
    _require(
        ab.get("predicates") == _ab_predicates(accepted),
        "aggregate A/B predicates changed",
    )
    modes = ab.get("repeats")
    _require(
        isinstance(modes, Mapping) and set(modes) == {"original", "wrapped"},
        "aggregate A/B modes changed",
    )
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    retained: set[Path] = set()
    cleanup_events: list[dict[str, Any]] = []
    for mode in ("original", "wrapped"):
        entries = modes.get(mode)
        _require(
            isinstance(entries, list) and len(entries) == REPEAT_COUNT,
            f"aggregate {mode} repeat count changed",
        )
        for index, entry in enumerate(entries):
            pairing_id = PAIRING_IDS[index]
            _require(
                isinstance(entry, Mapping)
                and set(entry)
                == {
                    "pairing_id",
                    "dataset_key",
                    "invocation_key",
                    "invocation",
                    "child_evidence",
                    "child_evidence_record",
                    "child_evidence_validation",
                    "retained_output",
                    "cleanup",
                }
                and entry.get("pairing_id") == pairing_id
                and entry.get("child_evidence_validation")
                == {
                    "loaded_and_signature_valid": True,
                    "identity_and_output_binding_valid": True,
                    "artifact_sha256": entry.get("child_evidence", {}).get(
                        "artifact_sha256"
                    ),
                },
                f"aggregate {mode} {pairing_id} record changed",
            )
            fit_root = root / "ab" / mode / pairing_id
            log = _validate_invocation(
                entry.get("invocation"),
                expected_command=_fit_command(
                    root=root,
                    code=code,
                    qualifier=qualifier,
                    mode=mode,
                    pairing_id=pairing_id,
                ),
                expected_log=fit_root / "fit.log",
                root=root,
                role=f"{mode} {pairing_id}",
                expected_return_code=0,
            )
            dataset_key = f"ab_{mode}_{pairing_id.replace('-', '_')}"
            invocation_key = dataset_key
            _require(
                entry.get("dataset_key") == dataset_key
                and entry.get("invocation_key") == invocation_key,
                f"aggregate {mode} {pairing_id} keys changed",
            )
            _require(
                top_invocations.get(invocation_key) == entry.get("invocation"),
                f"aggregate {mode} {pairing_id} top invocation differs",
            )
            generated_cleanup, temporary_cleanup = _validate_repeat_cleanup(
                entry.get("cleanup"),
                root=root,
                dataset=fit_root / "dataset",
                role=f"{mode} {pairing_id}",
            )
            cleanup_events.extend((generated_cleanup, temporary_cleanup))
            evidence_path = fit_root / "fit-evidence.json"
            child, _ = _load_signed(evidence_path, role=f"{mode} {pairing_id} child")
            _require(
                child == entry.get("child_evidence"),
                f"aggregate {mode} {pairing_id} embedded child differs from file",
            )
            _require(
                dataset_key in datasets,
                f"aggregate {mode} {pairing_id} dataset audit missing",
            )
            dataset_root = fit_root / "dataset"
            retained |= _validate_dataset_audit(
                datasets[dataset_key],
                root=root,
                expected_root=dataset_root,
                role=f"{mode} {pairing_id}",
                canonical_dataset=canonical_dataset,
            )
            ply = fit_root / "export/splat.ply"
            _validate_fit_child(
                child,
                root=root,
                mode=mode,
                pairing_id=pairing_id,
                dataset_root=dataset_root,
                ply_path=ply,
                adapter_path=adapter,
            )
            child_record_path, child_record = _verify_record(
                entry.get("child_evidence_record"),
                role=f"aggregate {mode} {pairing_id} child record",
                root=root,
                artifact_sha256=child["artifact_sha256"],
            )
            output_path, output_record = _verify_record(
                entry.get("retained_output"),
                role=f"aggregate {mode} {pairing_id} retained output",
                root=root,
            )
            _require(
                child_record_path == evidence_path
                and output_path == ply
                and all(
                    child_record[key] == entry["child_evidence_record"][key]
                    for key in ("path", "size_bytes", "sha256")
                )
                and all(
                    output_record[key] == child["output"][key]
                    for key in ("path", "size_bytes", "sha256")
                ),
                f"aggregate {mode} {pairing_id} retained bindings changed",
            )
            _require(
                not os.path.lexists(dataset_root / "outputs"),
                f"{mode} {pairing_id} generated dataset outputs remain",
            )
            retained.update({log, evidence_path, ply})
            records[(mode, pairing_id)] = entry
    return records, retained, cleanup_events


def _validate_soak(
    value: object,
    *,
    root: Path,
    code: Path,
    qualifier: Path,
    adapter: Path,
    datasets: Mapping[str, Any],
    top_invocations: Mapping[str, Any],
    canonical_dataset: Mapping[str, Any],
) -> tuple[set[Path], list[dict[str, Any]]]:
    _require(isinstance(value, Mapping), "aggregate soak is absent")
    _require(
        set(value)
        == {
            "passed",
            "invocation",
            "child_evidence",
            "child_evidence_record",
            "child_evidence_validation",
            "cleanup",
        }
        and value.get("passed") is True
        and value.get("child_evidence_validation")
        == {
            "loaded_and_signature_valid": True,
            "identity_sequence_resource_and_cleanup_valid": True,
            "artifact_sha256": value.get("child_evidence", {}).get("artifact_sha256"),
        },
        "aggregate soak record changed",
    )
    soak_root = root / "soak"
    log = _validate_invocation(
        value.get("invocation"),
        expected_command=_soak_command(root=root, code=code, qualifier=qualifier),
        expected_log=soak_root / "soak.log",
        root=root,
        role="soak",
        expected_return_code=0,
    )
    _require(
        top_invocations.get("soak") == value.get("invocation"),
        "soak top invocation differs",
    )
    cleanup = value.get("cleanup")
    _require(
        isinstance(cleanup, Mapping)
        and set(cleanup)
        == {
            "generated_outputs_absent_after_every_fit",
            "empty_export_removed",
            "final_temporary_cache_removed",
        }
        and cleanup.get("generated_outputs_absent_after_every_fit") is True,
        "aggregate soak cleanup changed",
    )
    export_cleanup = _validate_removed_tree(
        cleanup["empty_export_removed"],
        expected_root=soak_root / "export",
        expected_parent=soak_root,
        role="soak export",
    )
    temp_cleanup = _validate_removed_tree(
        cleanup["final_temporary_cache_removed"],
        expected_root=root / "tmp",
        expected_parent=root,
        role="final qualification temporary cache",
    )
    evidence_path = soak_root / "soak-evidence.json"
    child, _ = _load_signed(evidence_path, role="soak child")
    _require(
        child == value.get("child_evidence"), "aggregate soak child differs from file"
    )
    child_path, child_record = _verify_record(
        value.get("child_evidence_record"),
        role="aggregate soak child record",
        root=root,
        artifact_sha256=child["artifact_sha256"],
    )
    _require(
        child_path == evidence_path
        and all(
            child_record[key] == value["child_evidence_record"][key]
            for key in ("path", "size_bytes", "sha256")
        ),
        "aggregate soak child binding changed",
    )
    _require("soak" in datasets, "soak dataset audit missing")
    retained = _validate_dataset_audit(
        datasets["soak"],
        root=root,
        expected_root=soak_root / "dataset",
        role="soak",
        canonical_dataset=canonical_dataset,
    )
    _validate_soak_child(
        child,
        dataset_root=soak_root / "dataset",
        output_root=soak_root / "export",
        adapter_path=adapter,
    )
    _require(
        not os.path.lexists(soak_root / "dataset/outputs")
        and not os.path.lexists(soak_root / "export"),
        "soak cleanup left generated outputs",
    )
    retained.update({log, evidence_path})
    return retained, [export_cleanup, temp_cleanup]


def _validate_equivalence(
    value: object,
    *,
    root: Path,
    code: Mapping[str, Any],
    analyzer: Path,
    adapter: Path,
    repeats: Mapping[tuple[str, str], Mapping[str, Any]],
    accepted: bool,
    top_invocations: Mapping[str, Any],
    canonical_dataset: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    set[Path],
    list[dict[str, Any]],
]:
    _require(isinstance(value, Mapping), "aggregate equivalence evidence is absent")
    _require(
        set(value)
        == {
            "passed",
            "manifest",
            "prepare_manifest_invocation",
            "result",
            "analysis_invocation",
            "decision",
            "cleanup",
        }
        and value.get("passed") is accepted,
        "aggregate equivalence record changed",
    )
    directory = root / "equivalence"
    manifest_path, manifest_record = _verify_record(
        value.get("manifest"), role="repeat manifest", root=root
    )
    _require(
        manifest_path == directory / "repeat-manifest.json",
        "repeat manifest path changed",
    )
    manifest, manifest_file = _load_signed(manifest_path, role="repeat manifest")
    _require(
        value["manifest"].get("artifact_sha256") == manifest.get("artifact_sha256")
        and all(
            manifest_file[key] == value["manifest"][key]
            for key in ("path", "size_bytes", "sha256")
        ),
        "repeat manifest aggregate binding changed",
    )
    _require(
        top_invocations.get("equivalence_prepare_manifest")
        == value.get("prepare_manifest_invocation"),
        "prepare-manifest top invocation differs",
    )
    prepare_log = _validate_invocation(
        value.get("prepare_manifest_invocation"),
        expected_command=_prepare_manifest_command(
            root=root, code=_absolute(code["path"]), analyzer=analyzer
        ),
        expected_log=directory / "prepare-manifest.log",
        root=root,
        role="prepare manifest",
        expected_return_code=0,
    )
    _, manifest_retained = _validate_manifest(
        manifest,
        root=root,
        code=code,
        adapter_path=adapter,
        aggregate_repeats=repeats,
        canonical_dataset=canonical_dataset,
    )
    result_path, result_record = _verify_record(
        value.get("result"), role="analysis result", root=root
    )
    _require(
        result_path == directory / "analysis-result.json",
        "analysis result path changed",
    )
    result, result_file = _load_signed(result_path, role="analysis result")
    _require(
        value["result"].get("artifact_sha256") == result.get("artifact_sha256")
        and all(
            result_file[key] == value["result"][key]
            for key in ("path", "size_bytes", "sha256")
        ),
        "analysis result aggregate binding changed",
    )
    analyze_log = _validate_invocation(
        value.get("analysis_invocation"),
        expected_command=_analyze_command(
            root=root, code=_absolute(code["path"]), analyzer=analyzer
        ),
        expected_log=directory / "analyze.log",
        root=root,
        role="analyzer",
        expected_return_code=0 if accepted else 3,
    )
    _require(
        top_invocations.get("equivalence_analyze") == value.get("analysis_invocation"),
        "analysis top invocation differs",
    )
    decision = _validate_result(
        result,
        root=root,
        manifest=manifest,
        manifest_record=manifest_record,
        code=code,
        analyzer_path=analyzer,
        accepted=accepted,
    )
    _require(
        value.get("decision") == decision, "aggregate equivalence decision changed"
    )
    cleanup = value.get("cleanup")
    _require(
        isinstance(cleanup, Mapping)
        and set(cleanup) == {"after_manifest", "after_analysis"},
        "aggregate equivalence cleanup fields changed",
    )
    manifest_cleanup = _validate_removed_tree(
        cleanup["after_manifest"],
        expected_root=root / "tmp",
        expected_parent=root,
        role="post-manifest temporary cache",
        recreated=True,
    )
    analysis_cleanup = _validate_removed_tree(
        cleanup["after_analysis"],
        expected_root=root / "tmp",
        expected_parent=root,
        role="post-analysis temporary cache",
        recreated=True,
    )
    retained = manifest_retained | {
        manifest_path,
        result_path,
        prepare_log,
        analyze_log,
    }
    return manifest, result, decision, retained, [manifest_cleanup, analysis_cleanup]


def _top_predicates(accepted: bool) -> dict[str, bool]:
    return {
        "formal_held_paths_rejected": True,
        "code_checkout_clean": True,
        "deform360_checkout_clean_and_pinned": True,
        "code_checkout_stable_across_qualification": True,
        "deform360_checkout_stable_across_qualification": True,
        "python_runtime_stable_across_qualification": True,
        "canonical_parent_process_is_pinned": True,
        "materialized_inputs_stable_across_qualification": True,
        "source_inputs_stable_across_qualification": True,
        "referenced_source_materialized_content_equal": True,
        "fresh_ten_fit_ab_cohort_passed": accepted,
        "equivalence_analyzer_accepted": accepted,
        "soak_started_only_after_analyzer_acceptance": True,
        "resource_soak_passed": accepted,
        "qualification_temporary_root_absent": True,
        "fresh_output_root_was_required": True,
        "analyzer_no_go_skips_soak_and_is_terminal": True,
        "retry_or_in_place_reuse_forbidden": True,
        "attempt_marker_stable_across_qualification": True,
        "frozen_analyzer_source_digest_exact": True,
    }


def _validate_qualification(
    artifact: Mapping[str, Any], *, root: Path, accepted: bool
) -> dict[str, Any]:
    _validate_main_identity(artifact, accepted=accepted)
    canonical_dataset = _recompute_dataset_identity(
        PUBLIC_DATASET, role="canonical public dataset before qualification validation"
    )
    runtime, qualifier, _wrapper, analyzer, adapter = _validate_aggregate_runtime(
        artifact.get("runtime_bindings"), root=root
    )
    marker_path, marker_bound = _verify_record(
        artifact.get("attempt"), role="qualification attempt", root=root
    )
    _require(marker_path == root / ATTEMPT_NAME, "qualification attempt path changed")
    attempt, marker_file = _load_signed(marker_path, role="qualification attempt")
    _require(
        artifact["attempt"].get("artifact_sha256") == attempt.get("artifact_sha256")
        and all(
            marker_file[key] == artifact["attempt"][key]
            for key in ("path", "size_bytes", "sha256")
        )
        and marker_bound["sha256"] == marker_file["sha256"],
        "qualification attempt aggregate binding changed",
    )
    _validate_attempt(attempt, root=root, analyzer_source=analyzer)
    _require(
        attempt.get("code_revision") == runtime["head"],
        "qualification attempt revision differs from aggregate",
    )

    datasets = artifact.get("materialized_datasets")
    _require(
        isinstance(datasets, Mapping), "aggregate materialized datasets are absent"
    )
    repeat_keys = {
        f"ab_{mode}_{pairing_id.replace('-', '_')}"
        for mode in ("original", "wrapped")
        for pairing_id in PAIRING_IDS
    }
    _require(
        set(datasets) == (repeat_keys | ({"soak"} if accepted else set())),
        "aggregate materialized dataset set changed",
    )
    top_invocations = artifact.get("invocations")
    _require(isinstance(top_invocations, Mapping), "aggregate invocation map is absent")
    expected_invocation_keys = (
        repeat_keys
        | {
            "equivalence_prepare_manifest",
            "equivalence_analyze",
        }
        | ({"soak"} if accepted else set())
    )
    _require(
        set(top_invocations) == expected_invocation_keys,
        "aggregate invocation set changed",
    )
    repeats, retained, cleanup_events = _validate_aggregate_repeats(
        artifact.get("ab"),
        root=root,
        code=_absolute(runtime["path"]),
        qualifier=qualifier,
        adapter=adapter,
        datasets=datasets,
        accepted=accepted,
        top_invocations=top_invocations,
        canonical_dataset=canonical_dataset,
    )
    manifest, result, decision, equivalence_retained, equivalence_cleanup = (
        _validate_equivalence(
            artifact["ab"].get("equivalence"),
            root=root,
            code=runtime,
            analyzer=analyzer,
            adapter=adapter,
            repeats=repeats,
            accepted=accepted,
            top_invocations=top_invocations,
            canonical_dataset=canonical_dataset,
        )
    )
    retained |= equivalence_retained
    cleanup_events.extend(equivalence_cleanup)
    if accepted:
        soak_retained, soak_cleanup = _validate_soak(
            artifact.get("soak"),
            root=root,
            code=_absolute(runtime["path"]),
            qualifier=qualifier,
            adapter=adapter,
            datasets=datasets,
            top_invocations=top_invocations,
            canonical_dataset=canonical_dataset,
        )
        retained |= soak_retained
        cleanup_events.extend(soak_cleanup)
    else:
        _require(
            artifact.get("soak") is None and not os.path.lexists(root / "soak"),
            "admission-inconclusive outcome improperly ran or retained soak",
        )
        no_go_cleanup = _validate_removed_tree(
            artifact.get("cleanup_events", [])[-1]
            if isinstance(artifact.get("cleanup_events"), list)
            and artifact.get("cleanup_events")
            else None,
            expected_root=root / "tmp",
            expected_parent=root,
            role="scientific no-go temporary cache",
        )
        cleanup_events.append(no_go_cleanup)
    _require(
        artifact.get("cleanup_events") == cleanup_events,
        "aggregate cleanup event sequence changed",
    )
    _require(
        artifact.get("predicates") == _top_predicates(accepted),
        "aggregate predicates changed",
    )
    _require(
        artifact.get("passed") is all(_top_predicates(accepted).values()),
        "aggregate passed value differs from predicates",
    )
    _require(
        not os.path.lexists(root / "tmp"), "qualification temporary directory remains"
    )
    _require(
        _recompute_dataset_identity(
            PUBLIC_DATASET,
            role="canonical public dataset after qualification validation",
        )
        == canonical_dataset,
        "canonical public dataset changed during qualification validation",
    )
    retained.update({root / MAIN_NAME, marker_path})
    return {
        "source_head": runtime["head"],
        "source_tree": runtime["tree"],
        "accepted": accepted,
        "attempt": attempt,
        "attempt_record": marker_file,
        "manifest": manifest,
        "result": result,
        "decision": decision,
        "analyzer_source": _stable_file(analyzer, role="final analyzer source binding"),
        "allowed_files": retained,
    }


def _walk_tree(root: Path) -> tuple[set[Path], set[Path]]:
    directories: set[Path] = set()
    files: set[Path] = set()
    for current, names, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        observed = os.lstat(current_path)
        _require(
            stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
            f"qualification directory is invalid: {current_path}",
        )
        if current_path != root:
            directories.add(current_path)
        names[:] = sorted(names)
        for name in names:
            child = current_path / name
            child_stat = os.lstat(child)
            _require(
                stat.S_ISDIR(child_stat.st_mode)
                and not stat.S_ISLNK(child_stat.st_mode),
                f"qualification descendant is not a real directory: {child}",
            )
        for name in sorted(filenames):
            child = current_path / name
            child_stat = os.lstat(child)
            _require(
                stat.S_ISREG(child_stat.st_mode)
                and not stat.S_ISLNK(child_stat.st_mode)
                and child_stat.st_nlink == 1,
                f"qualification file is linked or not regular: {child}",
            )
            files.add(child)
    return directories, files


def _require_allowlist(root: Path, allowed_files: Iterable[Path]) -> None:
    expected_files = {_absolute(path) for path in allowed_files}
    _require(
        all(path != root and root in path.parents for path in expected_files),
        "qualification allowlist escaped its root",
    )
    expected_directories: set[Path] = set()
    for path in expected_files:
        parent = path.parent
        while parent != root:
            expected_directories.add(parent)
            parent = parent.parent
    directories, files = _walk_tree(root)
    _require(
        files == expected_files, "qualification tree has missing or undeclared files"
    )
    _require(
        directories == expected_directories,
        "qualification tree has missing or undeclared directories",
    )


def _inventory(root: Path) -> dict[str, Any]:
    directories, files = _walk_tree(root)
    rows: list[dict[str, Any]] = [
        {"path": path.relative_to(root).as_posix(), "type": "directory"}
        for path in directories
    ]
    for path in files:
        record = _stable_file(path, role="qualification inventory file")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "type": "file",
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
        )
    rows.sort(key=lambda value: str(value["path"]))
    return {
        "entry_count": len(rows),
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
        "rows": rows,
    }


def _metadata_inventory_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    metadata = [
        {
            "path": row["path"],
            "type": row["type"],
            "mode_octal": "0500" if row["type"] == "directory" else "0400",
            **({"size_bytes": row["size_bytes"]} if row["type"] == "file" else {}),
        }
        for row in rows
    ]
    return hashlib.sha256(_canonical_bytes({"rows": metadata})).hexdigest()


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _open_verified_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    before = os.lstat(path)
    _require(
        stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"{role} is not a real directory",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_state(before) == _stable_state(opened) == _stable_state(current),
            f"{role} changed while opening",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, _inode_identity(before)


def _seal_tree(root: Path, *, expected_identity: tuple[int, int]) -> None:
    root_descriptor, root_identity = _open_verified_directory(
        root, role="qualification root before sealing"
    )
    _require(
        root_identity == expected_identity,
        "qualification root changed before sealing",
    )

    def seal_directory(descriptor: int, display: Path) -> None:
        for name in sorted(os.listdir(descriptor)):
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_display = display / name
            if stat.S_ISREG(before.st_mode):
                _require(
                    before.st_nlink == 1,
                    f"bad qualification file: {child_display}",
                )
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    _require(
                        _stable_state(os.fstat(child)) == _stable_state(before),
                        f"qualification file changed while opening: {child_display}",
                    )
                    os.fchmod(child, 0o400)
                    after = os.fstat(child)
                    current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    _require(
                        _inode_identity(after)
                        == _inode_identity(current)
                        == _inode_identity(before)
                        and stat.S_IMODE(after.st_mode) == 0o400
                        and stat.S_IMODE(current.st_mode) == 0o400,
                        f"qualification file changed while sealing: {child_display}",
                    )
                finally:
                    os.close(child)
            elif stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    _require(
                        _stable_state(os.fstat(child)) == _stable_state(before),
                        f"qualification directory changed while opening: {child_display}",
                    )
                    seal_directory(child, child_display)
                    os.fchmod(child, 0o500)
                    after = os.fstat(child)
                    current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    _require(
                        _inode_identity(after)
                        == _inode_identity(current)
                        == _inode_identity(before)
                        and stat.S_IMODE(after.st_mode) == 0o500
                        and stat.S_IMODE(current.st_mode) == 0o500,
                        f"qualification directory changed while sealing: {child_display}",
                    )
                finally:
                    os.close(child)
            else:
                raise RuntimeError(
                    f"qualification entry is linked or special: {child_display}"
                )
        os.fchmod(descriptor, 0o500)

    try:
        seal_directory(root_descriptor, root)
        _require(
            _inode_identity(os.fstat(root_descriptor)) == root_identity
            and stat.S_IMODE(os.fstat(root_descriptor).st_mode) == 0o500,
            "qualification root changed while sealing",
        )
    finally:
        os.close(root_descriptor)
    current_root = os.lstat(root)
    _require(
        _inode_identity(current_root) == root_identity
        and stat.S_ISDIR(current_root.st_mode)
        and not stat.S_ISLNK(current_root.st_mode)
        and stat.S_IMODE(current_root.st_mode) == 0o500,
        "qualification root changed while sealing",
    )


def _require_sealed_tree(root: Path) -> None:
    directories, files = _walk_tree(root)
    _require(
        stat.S_IMODE(os.lstat(root).st_mode) == 0o500,
        "qualification root is not sealed",
    )
    for path in directories:
        _require(
            stat.S_IMODE(os.lstat(path).st_mode) == 0o500,
            f"qualification directory is not sealed: {path}",
        )
    for path in files:
        _require(
            stat.S_IMODE(os.lstat(path).st_mode) == 0o400,
            f"qualification file is not sealed: {path}",
        )


def _require_directory_identity(
    path: Path, expected: tuple[int, int], *, role: str
) -> None:
    observed = os.lstat(path)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and _inode_identity(observed) == expected,
        f"{role} identity changed",
    )


def _unlink_if_identity(
    parent: int, name: str, expected_identity: tuple[int, int]
) -> bool:
    try:
        observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or _inode_identity(observed) != expected_identity
    ):
        return False
    os.unlink(name, dir_fd=parent)
    return True


def _write_completion(path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.name == Path(path.name).name, "qualification completion name changed")
    parent, parent_identity = _open_verified_directory(
        path.parent, role="qualification completion parent"
    )
    temporary_name = f".{path.name}.partial-{os.getpid()}"
    published = False
    keep_parent_open = False
    file_identity: tuple[int, int] | None = None
    try:
        for name, message in (
            (path.name, "qualification completion already exists"),
            (temporary_name, "qualification completion partial exists"),
        ):
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise RuntimeError(message)
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(artifact))
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o400)
            file_state = os.fstat(stream.fileno())
            _require(
                stat.S_ISREG(file_state.st_mode)
                and stat.S_IMODE(file_state.st_mode) == 0o400
                and file_state.st_nlink == 1,
                "qualification completion partial changed while writing",
            )
            file_identity = _inode_identity(file_state)
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )
        published = True
        _require(
            file_identity is not None
            and _inode_identity(
                os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            )
            == file_identity,
            "qualification completion changed while publishing",
        )
        _require(
            _unlink_if_identity(parent, temporary_name, file_identity),
            "qualification completion partial changed before unlink",
        )
        os.fsync(parent)
        _require_directory_identity(
            path.parent,
            parent_identity,
            role="qualification completion parent during publication",
        )
        published_state = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        path_state = os.lstat(path)
        _require(
            file_identity is not None
            and stat.S_ISREG(published_state.st_mode)
            and not stat.S_ISLNK(published_state.st_mode)
            and stat.S_IMODE(published_state.st_mode) == 0o400
            and published_state.st_nlink == 1
            and _inode_identity(published_state)
            == _inode_identity(path_state)
            == file_identity,
            "qualification completion changed after publication",
        )
        keep_parent_open = True
        return {
            "parent_descriptor": parent,
            "parent_identity": parent_identity,
            "file_identity": file_identity,
        }
    except BaseException:
        if file_identity is not None:
            _unlink_if_identity(parent, temporary_name, file_identity)
        if published and file_identity is not None:
            removed = _unlink_if_identity(parent, path.name, file_identity)
            if removed:
                try:
                    os.fsync(parent)
                except OSError:
                    pass
        else:
            try:
                os.stat(temporary_name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
        raise
    finally:
        if not keep_parent_open:
            os.close(parent)


def _remove_published_completion(path: Path, publication: Mapping[str, Any]) -> bool:
    try:
        parent = publication.get("parent_descriptor")
        parent_identity = publication.get("parent_identity")
        file_identity = publication.get("file_identity")
        _require(
            isinstance(parent, int)
            and isinstance(parent_identity, tuple)
            and len(parent_identity) == 2
            and isinstance(file_identity, tuple)
            and len(file_identity) == 2,
            "qualification completion publication token changed",
        )
        _require(
            _inode_identity(os.fstat(parent)) == parent_identity,
            "qualification completion publication directory changed",
        )
        removed = _unlink_if_identity(parent, path.name, file_identity)
        if removed:
            os.fsync(parent)
        return removed
    except (OSError, RuntimeError):
        return False


def _require_published_completion_identity(
    path: Path, publication: Mapping[str, Any]
) -> None:
    parent = publication.get("parent_descriptor")
    parent_identity = publication.get("parent_identity")
    file_identity = publication.get("file_identity")
    _require(
        isinstance(parent, int)
        and isinstance(parent_identity, tuple)
        and len(parent_identity) == 2
        and isinstance(file_identity, tuple)
        and len(file_identity) == 2,
        "qualification completion publication token changed",
    )
    _require(
        _inode_identity(os.fstat(parent)) == parent_identity,
        "qualification completion publication directory changed",
    )
    _require_directory_identity(
        path.parent,
        parent_identity,
        role="qualification completion parent after final validation",
    )
    held = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    visible = os.lstat(path)
    _require(
        stat.S_ISREG(held.st_mode)
        and not stat.S_ISLNK(held.st_mode)
        and stat.S_IMODE(held.st_mode) == 0o400
        and held.st_nlink == 1
        and _inode_identity(held) == _inode_identity(visible) == file_identity,
        "qualification completion identity changed after final validation",
    )


def seal(root: Path, completion: Path) -> dict[str, Any]:
    _require_pinned_runtime_and_load_numpy()
    root = _absolute(root)
    completion = _absolute(completion)
    _require(socket.gethostname() == EXPECTED_HOST, "qualification sealer host changed")
    _require(
        root.parent == BASE and root.name.startswith(ROOT_PREFIX),
        "non-canonical qualification root",
    )
    _require(
        completion == Path(f"{root}-integrity-completion.json"),
        "non-canonical qualification completion path",
    )
    _require(
        not _is_formal_held(root) and not _is_formal_held(completion),
        "formal held path refused",
    )
    root_state = os.lstat(root)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and root.resolve(strict=True) == root
        and completion.parent.resolve(strict=True) == BASE.resolve(strict=True),
        "qualification root is invalid",
    )
    root_identity = _inode_identity(root_state)
    _require(not os.path.lexists(completion), "qualification completion already exists")
    _require((root / ATTEMPT_NAME).is_file(), "qualification attempt marker is missing")
    _require(
        (root / MAIN_NAME).is_file(), "complete qualification aggregate is missing"
    )
    evidence, evidence_before = _load_signed(
        root / MAIN_NAME, role="qualification aggregate"
    )
    status = evidence.get("status")
    _require(
        status in {"qualified", "admission-inconclusive"},
        "qualification terminal outcome is incomplete",
    )
    accepted = status == "qualified"
    validated = _validate_qualification(evidence, root=root, accepted=accepted)
    allowed = validated["allowed_files"]
    _require_allowlist(root, allowed)
    before = _inventory(root)
    _seal_tree(root, expected_identity=root_identity)
    _require_sealed_tree(root)
    after = _inventory(root)
    _require(before == after, "qualification content changed while sealing")
    evidence_after = _stable_file(
        root / MAIN_NAME, role="sealed qualification aggregate"
    )
    _require(
        all(
            evidence_before[key] == evidence_after[key]
            for key in ("path", "size_bytes", "sha256")
        ),
        "qualification aggregate changed while sealing",
    )

    attempt_after = _stable_file(
        root / ATTEMPT_NAME, role="sealed qualification attempt"
    )
    manifest_path = Path(validated["result"]["input_manifest"]["path"])
    result_path = root / "equivalence/analysis-result.json"
    manifest_after = _stable_file(manifest_path, role="sealed repeat manifest")
    result_after = _stable_file(result_path, role="sealed analysis result")
    operator = _stable_file(
        Path(__file__).resolve(strict=True), role="qualification integrity sealer"
    )
    completion_artifact = _signed(
        {
            "schema_version": 2,
            "artifact_kind": COMPLETION_KIND,
            "qualification_id": QUALIFICATION_ID,
            "status": "qualification-integrity-complete",
            "passed": True,
            "terminal_outcome": status,
            "admission_eligible": accepted,
            "host": EXPECTED_HOST,
            "qualification_root": os.fspath(root),
            "qualification_root_mode_octal": "0500",
            "qualification_tree_fully_nonwritable": True,
            "root_consumption_policy": dict(ROOT_CONSUMPTION_POLICY),
            "qualification_attempt": _record_with_artifact(
                attempt_after, validated["attempt"]
            ),
            "qualification_evidence": _record_with_artifact(evidence_after, evidence),
            "repeat_manifest": _record_with_artifact(
                manifest_after, validated["manifest"]
            ),
            "equivalence_result": _record_with_artifact(
                result_after, validated["result"]
            ),
            "analyzer_source": validated["analyzer_source"],
            "equivalence_decision": validated["decision"],
            "sealed_content_inventory": {
                "entry_count": after["entry_count"],
                "inventory_sha256": after["inventory_sha256"],
                "metadata_inventory_sha256": _metadata_inventory_sha256(after["rows"]),
            },
            "source_code": {
                "source_head": validated["source_head"],
                "source_tree": validated["source_tree"],
            },
            "executed_integrity_sealer_source": operator,
            "information_boundary": {
                "formal_held_path_accessed": False,
                "formal_target_query_prediction_or_score_deserialized": False,
                "public_development_dataset_only": True,
                "scientific_method_selected_from_qualification": False,
            },
        }
    )
    publication: dict[str, Any] | None = None
    try:
        _require_directory_identity(
            root, root_identity, role="qualification root before completion"
        )
        publication = _write_completion(completion, completion_artifact)
        _require_directory_identity(
            root, root_identity, role="qualification root after completion"
        )
        observed_completion, _ = _load_signed(
            completion, role="qualification completion"
        )
        _require(
            observed_completion == completion_artifact,
            "qualification completion changed",
        )
        _require_sealed_tree(root)
        _require_allowlist(root, allowed)
        _require(
            _inventory(root) == after,
            "qualification content changed after completion publication",
        )
        final_evidence, final_evidence_record = _load_signed(
            root / MAIN_NAME, role="final qualification aggregate"
        )
        final_attempt, final_attempt_record = _load_signed(
            root / ATTEMPT_NAME, role="final qualification attempt"
        )
        final_manifest, final_manifest_record = _load_signed(
            manifest_path, role="final repeat manifest"
        )
        final_result, final_result_record = _load_signed(
            result_path, role="final analysis result"
        )
        _require(
            final_evidence == evidence
            and all(
                final_evidence_record[key] == evidence_after[key]
                for key in ("path", "size_bytes", "sha256")
            ),
            "qualification aggregate changed after publication",
        )
        _require(
            final_attempt == validated["attempt"]
            and all(
                final_attempt_record[key] == attempt_after[key]
                for key in ("path", "size_bytes", "sha256")
            ),
            "qualification attempt changed after publication",
        )
        _require(
            final_manifest == validated["manifest"]
            and all(
                final_manifest_record[key] == manifest_after[key]
                for key in ("path", "size_bytes", "sha256")
            ),
            "repeat manifest changed after publication",
        )
        _require(
            final_result == validated["result"]
            and all(
                final_result_record[key] == result_after[key]
                for key in ("path", "size_bytes", "sha256")
            ),
            "analysis result changed after publication",
        )
        _recompute_equivalence(final_result, accepted=accepted)
        if accepted:
            code_root = _absolute(evidence["runtime_bindings"]["code"]["path"])
            _validate_soak_child(
                evidence["soak"]["child_evidence"],
                dataset_root=root / "soak/dataset",
                output_root=root / "soak/export",
                adapter_path=(code_root / RELATIVE_GSPLAT_ADAPTER_SOURCE).resolve(
                    strict=True
                ),
            )
        _require_directory_identity(
            root, root_identity, role="qualification root after final validation"
        )
        _require_published_completion_identity(completion, publication)
    except BaseException:
        if publication is not None:
            _remove_published_completion(completion, publication)
        raise
    finally:
        if publication is not None:
            os.close(publication["parent_descriptor"])
    return completion_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--completion", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    completion = arguments.completion or Path(
        f"{_absolute(arguments.qualification_root)}-integrity-completion.json"
    )
    artifact = seal(arguments.qualification_root, completion)
    print(json.dumps(artifact, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
