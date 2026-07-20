#!/usr/bin/env python3
"""Prepare or verify the target-free held-v7 gsplat runtime evidence.

This operator runs only after the v7 method checkout has been made immutable
and before the calibration lock exists.  It never accepts a dataset, episode,
prediction, target, tactile, or outcome path.  Preparation performs two
operations:

* seal a canonical manifest around the already-frozen ``gsplat_cuda.so``; and
* run the deployed smoke once on each physical RTX 6000 Ada GPU in a fresh,
  isolated pinned-runtime process, with that GPU exposed as logical ``cuda:0``.

The resulting two signed device artifacts are nested unchanged in one signed
canonical evidence file.  Verification is read-only and repeats all byte,
mode, path, artifact-checksum, deployment-HEAD, and semantic checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from types import ModuleType
from typing import Any, Mapping, Sequence


_CANONICAL_SUPPLEMENT_ROOT = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v7-runtimes/"
    "gsplat-cuda-py312-cu121-"
    "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64"
)
_CANONICAL_EXTENSION = _CANONICAL_SUPPLEMENT_ROOT / "gsplat_cuda.so"
_CANONICAL_SUPPLEMENT_MANIFEST = (
    _CANONICAL_SUPPLEMENT_ROOT / "runtime-supplement-manifest.json"
)
_CANONICAL_HELD_V7_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v7")
_CANONICAL_SMOKE_EVIDENCE = (
    _CANONICAL_HELD_V7_ROOT / "gsplat-runtime-smoke-evidence.json"
)
_PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-"
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
_BASE_RUNTIME_ROOT = _PINNED_PYTHON.parents[1]
_BASE_RUNTIME_MANIFEST = Path(f"{_BASE_RUNTIME_ROOT}.tree-manifest.json")
_BASE_RUNTIME_MANIFEST_SHA256 = (
    "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61"
)
_BASE_RUNTIME_PIP_FREEZE_SHA256 = (
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
_EXTENSION_SHA256 = "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64"
_EXTENSION_SIZE_BYTES = 6_982_312
_MODULE_RELATIVE_PATH = Path("src/bayesian_phystwin/deform360_held_gsplat_runtime.py")
_GIT = Path("/usr/bin/git")
_PYCACHE_PREFIX = "/nonexistent/bpt-held-v7-pycache"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_WORKER_MARKER = "BPT_GSPLAT_SMOKE_JSON="
_EXPECTED_PREDICATES: Mapping[str, Any] = {
    "alpha_shape": [1, 16, 16, 1],
    "backward_complete": True,
    "cuda_synchronized": True,
    "forward_finite_nonempty_nonzero": True,
    "gradient_groups_finite_and_nonzero": [
        "colors",
        "means",
        "opacities",
        "quats",
        "scales",
    ],
    "positive_radius_count": 2,
    "render_shape": [1, 16, 16, 3],
}
_EXPECTED_SMOKE_KEYS = {
    "artifact_kind",
    "artifact_sha256",
    "compute_capability",
    "contract_sha256",
    "extension_loaded_and_retained",
    "extension_path",
    "extension_sha256",
    "gpu_name",
    "gsplat_version",
    "logical_device",
    "ninja_visible",
    "nvcc_visible",
    "physical_gpu_index",
    "predicates",
    "python_version",
    "schema_version",
    "target_or_outcome_path_accessed",
    "torch_cuda_version",
    "torch_version",
}
_WORKER_SOURCE = f"""\
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

code = Path(sys.argv[1])
expected_source_sha256 = sys.argv[2]
source = code / {os.fspath(_MODULE_RELATIVE_PATH)!r}
if not code.is_absolute() or code.resolve(strict=True) != code:
    raise SystemExit("deployed code is non-canonical")
if source.is_symlink() or source.resolve(strict=True) != source:
    raise SystemExit("deployed gsplat module is non-canonical")
if hashlib.sha256(source.read_bytes()).hexdigest() != expected_source_sha256:
    raise SystemExit("deployed gsplat module checksum changed")
sys.path.insert(0, os.fspath(code / "src"))
module = importlib.import_module(
    "bayesian_phystwin.deform360_held_gsplat_runtime"
)
if Path(module.__file__).resolve(strict=True) != source:
    raise SystemExit("loaded gsplat module came from another checkout")
evidence = module.load_and_smoke_gsplat_runtime()
print(
    {_WORKER_MARKER!r}
    + json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ),
    flush=True,
)
"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _absolute(path: Path) -> Path:
    value = Path(os.path.abspath(os.fspath(path)))
    _require(value.is_absolute(), "path is not absolute")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return {**unsigned, "artifact_sha256": _canonical_sha256(unsigned)}


def _canonical_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stable_identity(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _read_regular_nofollow(path: Path, role: str) -> bytes:
    absolute = _absolute(path)
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise ValueError(f"{role} is missing: {absolute}") from error
    _require(stat.S_ISREG(before.st_mode), f"{role} is not a regular file")
    _require(not stat.S_ISLNK(before.st_mode), f"{role} is a symlink")
    descriptor = os.open(
        absolute,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_identity(opened) == _stable_identity(before),
            f"{role} changed while opening",
        )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        final_fd = os.fstat(descriptor)
        final_path = os.lstat(absolute)
        _require(
            _stable_identity(final_fd) == _stable_identity(opened)
            and _stable_identity(final_path) == _stable_identity(opened),
            f"{role} changed while reading",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path, role: str) -> str:
    return hashlib.sha256(_read_regular_nofollow(path, role)).hexdigest()


def _load_json(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_nofollow(path, role)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{role} root is not an object")
    return value, payload


def _load_deployed_runtime_module(code: Path) -> tuple[ModuleType, Path, str]:
    source = code / _MODULE_RELATIVE_PATH
    _require(
        source.is_file()
        and not source.is_symlink()
        and source.resolve(strict=True) == source,
        "deployed gsplat runtime module is absent or aliased",
    )
    source_sha256 = _sha256_file(source, "deployed gsplat runtime module")
    name = "_deform360_held_v7_gsplat_runtime_for_preparation"
    spec = importlib.util.spec_from_file_location(name, source)
    _require(
        spec is not None and spec.loader is not None, "cannot load gsplat contract"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(
        Path(str(module.__file__)).resolve(strict=True) == source,
        "gsplat contract loaded from another source",
    )
    return module, source, source_sha256


def _deployed_head(code: Path) -> str:
    result = subprocess.run(
        [_GIT, "-C", code, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env={
            "HOME": "/home/florianpfaff",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    _require(result.returncode == 0, "cannot resolve deployed method HEAD")
    head = result.stdout.strip()
    _require(bool(_GIT_OBJECT.fullmatch(head)), "deployed method HEAD is invalid")
    _require(code.name == f"code-{head}", "deployed code directory does not bind HEAD")
    return head


def _validate_extension(extension_contract: Mapping[str, Any]) -> None:
    _require(
        extension_contract.get("canonical_path") == os.fspath(_CANONICAL_EXTENSION)
        and extension_contract.get("sha256") == _EXTENSION_SHA256
        and extension_contract.get("file_size_bytes") == _EXTENSION_SIZE_BYTES
        and extension_contract.get("file_mode_octal") == "0444"
        and extension_contract.get("parent_mode_octal") == "0555"
        and extension_contract.get("base_runtime_manifest_sha256")
        == _BASE_RUNTIME_MANIFEST_SHA256
        and extension_contract.get("base_runtime_pip_freeze_sha256")
        == _BASE_RUNTIME_PIP_FREEZE_SHA256,
        "deployed extension contract changed",
    )
    _require(
        _CANONICAL_EXTENSION.resolve(strict=True) == _CANONICAL_EXTENSION
        and not _CANONICAL_EXTENSION.is_symlink(),
        "frozen gsplat extension is absent or aliased",
    )
    observed = os.lstat(_CANONICAL_EXTENSION)
    _require(
        stat.S_ISREG(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o444
        and observed.st_size == _EXTENSION_SIZE_BYTES,
        "frozen gsplat extension mode, type, or size changed",
    )
    _require(
        _sha256_file(_CANONICAL_EXTENSION, "frozen gsplat extension")
        == _EXTENSION_SHA256,
        "frozen gsplat extension checksum changed",
    )
    _require(
        _sha256_file(_BASE_RUNTIME_MANIFEST, "base runtime manifest")
        == _BASE_RUNTIME_MANIFEST_SHA256,
        "base held runtime manifest checksum changed",
    )


def _expected_manifest(
    extension_contract: Mapping[str, Any], extension_contract_sha256: str
) -> dict[str, Any]:
    return _artifact(
        {
            "artifact_kind": "Deform360HeldGsplatRuntimeSupplementManifestV1",
            "extension_contract": dict(extension_contract),
            "extension_contract_sha256": extension_contract_sha256,
            "schema_version": 1,
        }
    )


def _validate_manifest(
    path: Path,
    *,
    extension_contract: Mapping[str, Any],
    extension_contract_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    _require(path == _CANONICAL_SUPPLEMENT_MANIFEST, "supplement path changed")
    _require(path.resolve(strict=True) == path, "supplement manifest is aliased")
    _require(
        stat.S_IMODE(os.lstat(path).st_mode) == 0o400,
        "supplement manifest mode is not 0400",
    )
    observed, payload = _load_json(path, "gsplat supplement manifest")
    expected = _expected_manifest(extension_contract, extension_contract_sha256)
    _require(observed == expected, "gsplat supplement manifest identity changed")
    _require(
        payload == _canonical_payload(expected), "supplement JSON is non-canonical"
    )
    _require(
        _CANONICAL_SUPPLEMENT_ROOT.resolve(strict=True) == _CANONICAL_SUPPLEMENT_ROOT
        and not _CANONICAL_SUPPLEMENT_ROOT.is_symlink()
        and stat.S_IMODE(os.lstat(_CANONICAL_SUPPLEMENT_ROOT).st_mode) == 0o555,
        "supplement root is absent, aliased, or writable",
    )
    _validate_extension(extension_contract)
    return observed, payload


def _write_exclusive(path: Path, payload: bytes, *, role: str) -> None:
    _require(not os.path.lexists(path), f"{role} already exists")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o400, follow_symlinks=False)


def _create_manifest(
    *, extension_contract: Mapping[str, Any], extension_contract_sha256: str
) -> tuple[dict[str, Any], bytes]:
    _require(
        _CANONICAL_SUPPLEMENT_ROOT.resolve(strict=True) == _CANONICAL_SUPPLEMENT_ROOT
        and not _CANONICAL_SUPPLEMENT_ROOT.is_symlink()
        and stat.S_IMODE(os.lstat(_CANONICAL_SUPPLEMENT_ROOT).st_mode) == 0o555,
        "supplement root is absent, aliased, or not sealed 0555",
    )
    _validate_extension(extension_contract)
    manifest = _expected_manifest(extension_contract, extension_contract_sha256)
    payload = _canonical_payload(manifest)
    try:
        os.chmod(_CANONICAL_SUPPLEMENT_ROOT, 0o755, follow_symlinks=False)
        _write_exclusive(
            _CANONICAL_SUPPLEMENT_MANIFEST,
            payload,
            role="gsplat supplement manifest",
        )
    finally:
        os.chmod(_CANONICAL_SUPPLEMENT_ROOT, 0o555, follow_symlinks=False)
    observed, observed_payload = _validate_manifest(
        _CANONICAL_SUPPLEMENT_MANIFEST,
        extension_contract=extension_contract,
        extension_contract_sha256=extension_contract_sha256,
    )
    _require(
        observed == manifest and observed_payload == payload,
        "supplement manifest changed after creation",
    )
    return observed, observed_payload


def _worker_environment(physical_gpu_index: int) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": str(physical_gpu_index),
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYNPUT_BACKEND": "dummy",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": _PYCACHE_PREFIX,
        "PYTHONSAFEPATH": "1",
        "PYOPENGL_PLATFORM": "egl",
        "TMPDIR": "/tmp",
        "USER": "florianpfaff",
        "WANDB_MODE": "disabled",
    }


def _run_device_smoke(
    *, code: Path, source_sha256: str, physical_gpu_index: int
) -> dict[str, Any]:
    result = subprocess.run(
        [
            _PINNED_PYTHON,
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={_PYCACHE_PREFIX}",
            "-c",
            _WORKER_SOURCE,
            os.fspath(code),
            source_sha256,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=_worker_environment(physical_gpu_index),
    )
    _require(
        result.returncode == 0,
        "gsplat smoke failed on physical GPU "
        f"{physical_gpu_index}: stdout={result.stdout[-2000:]!r}; "
        f"stderr={result.stderr[-4000:]!r}",
    )
    marked = [
        line[len(_WORKER_MARKER) :]
        for line in result.stdout.splitlines()
        if line.startswith(_WORKER_MARKER)
    ]
    _require(len(marked) == 1, "gsplat worker did not emit one evidence object")
    try:
        smoke = json.loads(marked[0])
    except json.JSONDecodeError as error:
        raise ValueError("gsplat worker evidence is invalid JSON") from error
    _require(isinstance(smoke, dict), "gsplat worker evidence is not an object")
    return smoke


def _validate_smoke(
    smoke: Mapping[str, Any],
    *,
    physical_gpu_index: int,
    smoke_contract_sha256: str,
) -> None:
    _require(
        set(smoke) == _EXPECTED_SMOKE_KEYS
        and smoke.get("artifact_kind") == "Deform360HeldGsplatRuntimeSmokeV1"
        and smoke.get("schema_version") == 1
        and smoke.get("artifact_sha256")
        == _canonical_sha256(
            {key: value for key, value in smoke.items() if key != "artifact_sha256"}
        )
        and smoke.get("contract_sha256") == smoke_contract_sha256
        and smoke.get("physical_gpu_index") == physical_gpu_index
        and smoke.get("logical_device") == "cuda:0"
        and smoke.get("gpu_name") == "NVIDIA RTX 6000 Ada Generation"
        and smoke.get("compute_capability") == "8.9"
        and smoke.get("python_version") == "3.12"
        and smoke.get("torch_version") == "2.4.0+cu121"
        and smoke.get("torch_cuda_version") == "12.1"
        and smoke.get("gsplat_version") == "1.4.0"
        and smoke.get("extension_path") == os.fspath(_CANONICAL_EXTENSION)
        and smoke.get("extension_sha256") == _EXTENSION_SHA256
        and smoke.get("extension_loaded_and_retained") is True
        and smoke.get("nvcc_visible") is False
        and smoke.get("ninja_visible") is False
        and smoke.get("target_or_outcome_path_accessed") is False
        and smoke.get("predicates") == _EXPECTED_PREDICATES,
        f"physical GPU {physical_gpu_index} smoke identity or predicates changed",
    )


def _expected_evidence(
    *,
    smoke_contract_sha256: str,
    method_head: str,
    source_sha256: str,
    manifest_payload: bytes,
    smokes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _artifact(
        {
            "artifact_kind": "Deform360HeldGsplatRuntimeSmokeEvidenceV1",
            "contract_sha256": smoke_contract_sha256,
            "deployed_method_head": method_head,
            "operator_source_sha256": source_sha256,
            "runtime_supplement_manifest_sha256": hashlib.sha256(
                manifest_payload
            ).hexdigest(),
            "schema_version": 1,
            "smokes": [dict(smoke) for smoke in smokes],
        }
    )


def _validate_evidence(
    path: Path,
    *,
    smoke_contract_sha256: str,
    method_head: str,
    source_sha256: str,
    manifest_payload: bytes,
) -> tuple[dict[str, Any], bytes]:
    _require(path == _CANONICAL_SMOKE_EVIDENCE, "smoke evidence path changed")
    _require(path.resolve(strict=True) == path, "smoke evidence is aliased")
    _require(
        stat.S_IMODE(os.lstat(path).st_mode) == 0o400,
        "smoke evidence mode is not 0400",
    )
    observed, payload = _load_json(path, "gsplat smoke evidence")
    smokes = observed.get("smokes")
    _require(
        isinstance(smokes, list) and len(smokes) == 2,
        "smoke evidence does not contain exactly two device artifacts",
    )
    for index, smoke in enumerate(smokes):
        _require(isinstance(smoke, Mapping), "nested smoke is not an object")
        _validate_smoke(
            smoke,
            physical_gpu_index=index,
            smoke_contract_sha256=smoke_contract_sha256,
        )
    expected = _expected_evidence(
        smoke_contract_sha256=smoke_contract_sha256,
        method_head=method_head,
        source_sha256=source_sha256,
        manifest_payload=manifest_payload,
        smokes=smokes,
    )
    _require(observed == expected, "aggregate smoke evidence identity changed")
    _require(payload == _canonical_payload(expected), "smoke evidence is non-canonical")
    return observed, payload


def _create_evidence(
    *,
    code: Path,
    smoke_contract_sha256: str,
    method_head: str,
    source_sha256: str,
    manifest_payload: bytes,
) -> tuple[dict[str, Any], bytes]:
    _require(
        _CANONICAL_HELD_V7_ROOT.resolve(strict=True) == _CANONICAL_HELD_V7_ROOT
        and not _CANONICAL_HELD_V7_ROOT.is_symlink(),
        "held-v7 root is absent or aliased",
    )
    with os.scandir(_CANONICAL_HELD_V7_ROOT) as scan:
        entries = sorted(scan, key=lambda entry: entry.name)
    _require(
        len(entries) == 1
        and entries[0].name == code.name
        and entries[0].is_dir(follow_symlinks=False),
        "pre-lock held-v7 root must contain only the deployed method checkout",
    )
    smokes: list[dict[str, Any]] = []
    for physical_gpu_index in (0, 1):
        smoke = _run_device_smoke(
            code=code,
            source_sha256=source_sha256,
            physical_gpu_index=physical_gpu_index,
        )
        _validate_smoke(
            smoke,
            physical_gpu_index=physical_gpu_index,
            smoke_contract_sha256=smoke_contract_sha256,
        )
        smokes.append(smoke)
    evidence = _expected_evidence(
        smoke_contract_sha256=smoke_contract_sha256,
        method_head=method_head,
        source_sha256=source_sha256,
        manifest_payload=manifest_payload,
        smokes=smokes,
    )
    payload = _canonical_payload(evidence)
    _write_exclusive(
        _CANONICAL_SMOKE_EVIDENCE,
        payload,
        role="gsplat runtime smoke evidence",
    )
    observed, observed_payload = _validate_evidence(
        _CANONICAL_SMOKE_EVIDENCE,
        smoke_contract_sha256=smoke_contract_sha256,
        method_head=method_head,
        source_sha256=source_sha256,
        manifest_payload=manifest_payload,
    )
    _require(
        observed == evidence and observed_payload == payload,
        "smoke evidence changed after creation",
    )
    return observed, observed_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployed-code", type=Path, required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify the two canonical artifacts without writing or running CUDA",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    os.umask(0o077)
    code = _absolute(arguments.deployed_code)
    _require(
        code.parent == _CANONICAL_HELD_V7_ROOT
        and code.resolve(strict=True) == code
        and not code.is_symlink(),
        "deployed code is outside the canonical held-v7 root or aliased",
    )
    method_head = _deployed_head(code)
    module, _source, source_sha256 = _load_deployed_runtime_module(code)
    extension_contract = dict(module.GSPLAT_CUDA_EXTENSION_CONTRACT)
    extension_contract_sha256 = str(module.GSPLAT_CUDA_EXTENSION_CONTRACT_SHA256)
    smoke_contract_sha256 = str(module.GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256)
    _require(
        bool(_SHA256.fullmatch(extension_contract_sha256))
        and extension_contract_sha256 == _canonical_sha256(extension_contract)
        and bool(_SHA256.fullmatch(smoke_contract_sha256)),
        "deployed gsplat contract checksum is invalid",
    )
    _validate_extension(extension_contract)
    if arguments.verify_existing:
        _manifest, manifest_payload = _validate_manifest(
            _CANONICAL_SUPPLEMENT_MANIFEST,
            extension_contract=extension_contract,
            extension_contract_sha256=extension_contract_sha256,
        )
        evidence, evidence_payload = _validate_evidence(
            _CANONICAL_SMOKE_EVIDENCE,
            smoke_contract_sha256=smoke_contract_sha256,
            method_head=method_head,
            source_sha256=source_sha256,
            manifest_payload=manifest_payload,
        )
    else:
        _manifest, manifest_payload = _create_manifest(
            extension_contract=extension_contract,
            extension_contract_sha256=extension_contract_sha256,
        )
        evidence, evidence_payload = _create_evidence(
            code=code,
            smoke_contract_sha256=smoke_contract_sha256,
            method_head=method_head,
            source_sha256=source_sha256,
            manifest_payload=manifest_payload,
        )
    print(
        json.dumps(
            {
                "deployed_method_head": method_head,
                "operator_source_sha256": source_sha256,
                "runtime_supplement_manifest_file_sha256": hashlib.sha256(
                    manifest_payload
                ).hexdigest(),
                "smoke_evidence_artifact_sha256": evidence["artifact_sha256"],
                "smoke_evidence_file_sha256": hashlib.sha256(
                    evidence_payload
                ).hexdigest(),
                "verified_existing": bool(arguments.verify_existing),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
