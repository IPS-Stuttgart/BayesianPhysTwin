#!/usr/bin/env python3
"""Prepare the prospective Deform360 held-v8 lock and immutable deployment.

The formal root must not exist when this operator starts.  A complete clean
Git deployment is staged on the same filesystem, the required disclosure and
calibration lock are written into a newly created root, and only then is the
already-validated deployment atomically renamed below that root.  Thus no
prediction, source acquisition, target reconstruction, or score can precede
the immutable calibration lock.

Run ``--preflight`` first.  It performs every read-only provenance check and
prints the exact prospective bindings without creating the formal root.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
from typing import Any


_HELD_BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
_HELD_ROOT = _HELD_BASE / "held-v8"
_LOCK_PATH = _HELD_ROOT / "calibration-lock.json"
_DISCLOSURE_PATH = _HELD_ROOT / "post-withdrawal-development-use-disclosure.json"
_V7_LOCK = _HELD_BASE / "held-v7" / "calibration-lock.json"
_V7_WITHDRAWAL = _HELD_BASE / "held-v7" / "v7-outcome-withdrawal-report.json"
_V7_RUNTIME_SMOKE = _HELD_BASE / "held-v7" / "gsplat-runtime-smoke-evidence.json"
_V8_ATTEMPT1_WITHDRAWAL_POINTER = (
    _HELD_BASE / "held-v8-attempt-1-withdrawal-pointer.json"
)
_V8_ATTEMPT1_WITHDRAWAL_REPORT = (
    _HELD_BASE
    / "held-v8-attempt-1-withdrawn-preoutcome"
    / "execution-withdrawal-preoutcome.json"
)
_V8_ATTEMPT2_ARCHIVE = _HELD_BASE / "held-v8-attempt-2-withdrawn-preoutcome"
_V8_ATTEMPT2_WITHDRAWAL_POINTER = (
    _HELD_BASE / "held-v8-attempt-2-withdrawal-pointer.json"
)
_V8_ATTEMPT2_WITHDRAWAL_REPORT = (
    _V8_ATTEMPT2_ARCHIVE / "execution-withdrawal-preoutcome-attempt2.json"
)
_V8_ATTEMPT2_INTEGRITY_COMPLETION = (
    _HELD_BASE / "held-v8-attempt-2-withdrawal-integrity-completion.json"
)
_V8_ATTEMPT2_MANIFEST_SCALE_DIAGNOSTIC = (
    _V8_ATTEMPT2_ARCHIVE / "prewithdrawal-072-manifest-scale-diagnostic.json"
)
_V8_ATTEMPT2_ADMISSION_DIAGNOSTIC = (
    _V8_ATTEMPT2_ARCHIVE / "prewithdrawal-072-admission-compatibility-diagnostic.json"
)
_V8_ATTEMPT2_FAILURE_LOG = (
    _V8_ATTEMPT2_ARCHIVE
    / "calibration"
    / "logs"
    / "072-cotton-clohesline-ep0003.physical.failed.log"
)
_V8_ADMISSION_REPLAY_ROOT = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v8-admission-wrapper-scratch-20260722"
)
_V8_ADMISSION_REPLAY_REPORT = (
    _V8_ADMISSION_REPLAY_ROOT / "metadata-only-replay-report.json"
)
_V8_ADMISSION_REPLAY_CODE_BINDING = (
    _V8_ADMISSION_REPLAY_ROOT / "metadata-only-replay-code-binding.json"
)
_OPEN27_DECISION = (
    _HELD_BASE
    / "runs"
    / "deform360-query-field-open27-v1-development"
    / "decision.json"
)
_RIGID_RESIDUAL_DECISION = (
    _HELD_BASE
    / "runs"
    / "deform360-rigid-residual-open27-v1-development"
    / "decision.json"
)
_GSPLAT_SUPPLEMENT = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v7-runtimes/"
    "gsplat-cuda-py312-cu121-"
    "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64/"
    "runtime-supplement-manifest.json"
)
_RUNTIME_ROOT = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
_PINNED_PYTHON = _RUNTIME_ROOT / "bin" / "python"
_PYTHON_FREEZE = Path(f"{_RUNTIME_ROOT}.freeze.sorted.txt")
_PYTHON_TREE_MANIFEST = Path(f"{_RUNTIME_ROOT}.tree-manifest.json")
_SEMANTIC_MODEL_LOCK = Path(
    "/mnt/corsair/florianpfaff/bpt-framezero-field-dev-20260720/"
    "scratch_siglip2_model_lock.json"
)
_ALLTRACKER_CHECKPOINT = Path("/mnt/corsair/florianpfaff/model-cache/alltracker.pth")
_SAM2_CHECKPOINT = Path(
    "/mnt/lexar4tb/datasets/deform360/sam2-2b90b9f5/checkpoints/sam2.1_hiera_small.pt"
)
_DEFORM360_CODE = Path("/mnt/lexar4tb/datasets/deform360/code")

_EXPECTED_EXTERNAL_FILES: Mapping[str, tuple[Path, str, int | None]] = {
    "v7_calibration_lock_file": (
        _V7_LOCK,
        "b464d7cfda3b4ad94f57ffd46267b3b50d8dc65e2ff8dfec2befc7953718aca7",
        0o400,
    ),
    "v7_withdrawal_report": (
        _V7_WITHDRAWAL,
        "7bcab7169fc2addad8e56b7bb5ca9086b5249e9a744e18b9d51a7f395098c1a3",
        0o400,
    ),
    "v7_gsplat_runtime_smoke_evidence": (
        _V7_RUNTIME_SMOKE,
        "c5f0218962e1c18748f52d423c11804864e2695a719f00ff63452cebdbde029c",
        0o400,
    ),
    "v8_attempt1_preoutcome_withdrawal_pointer": (
        _V8_ATTEMPT1_WITHDRAWAL_POINTER,
        "f7af6d1adf8541fd015cbe5336da97e013777c1bb711deaa01d9a84a49c81daa",
        0o400,
    ),
    "v8_attempt1_preoutcome_withdrawal_report": (
        _V8_ATTEMPT1_WITHDRAWAL_REPORT,
        "c04a6e7a95d958950ea7e7c05e7e2b98ee4516c01f03e9284f85ccccf0f6873b",
        0o400,
    ),
    "v8_attempt2_preoutcome_withdrawal_pointer": (
        _V8_ATTEMPT2_WITHDRAWAL_POINTER,
        "007d3fbde0dc93dc350661aafdd5d08d1398aa8d1f164e17bf295521fc40463a",
        0o400,
    ),
    "v8_attempt2_preoutcome_withdrawal_report": (
        _V8_ATTEMPT2_WITHDRAWAL_REPORT,
        "5830f9bfe8d29d5a09f64afbcaeabadc3acb7c8fdf820c1aeb68a6601055a895",
        0o400,
    ),
    "v8_attempt2_withdrawal_integrity_completion": (
        _V8_ATTEMPT2_INTEGRITY_COMPLETION,
        "21e7695af5f610193502ecb6e7e6c647d853bde34daa1c5f362e990dffdf56a7",
        0o400,
    ),
    "v8_attempt2_manifest_scale_diagnostic": (
        _V8_ATTEMPT2_MANIFEST_SCALE_DIAGNOSTIC,
        "3166d488258f1f62535c87813bbd895c9e4ba9855d43fa4393b8795f85c78973",
        0o400,
    ),
    "v8_attempt2_admission_compatibility_diagnostic": (
        _V8_ATTEMPT2_ADMISSION_DIAGNOSTIC,
        "ba45b56d1e127099d7ef1a910d199cc0f6c9dd698b7f785828163bc28904e2fb",
        0o400,
    ),
    "v8_attempt2_failure_log": (
        _V8_ATTEMPT2_FAILURE_LOG,
        "e296021c5b647d5e26cbf8cecd2e3fc46ebed97026a2564224a54f0fcd156b1c",
        0o400,
    ),
    "v8_external_admission_metadata_only_replay": (
        _V8_ADMISSION_REPLAY_REPORT,
        "dc4ec1d5f913bd0dd6d10116783d98a7d9ef88ac9a7c74d778329687f6ff052b",
        0o400,
    ),
    "v8_external_admission_replay_code_binding": (
        _V8_ADMISSION_REPLAY_CODE_BINDING,
        "0015a7e9b7f2b7a7241dc405e27d96d31911980fb781cd569d227e066f595209",
        0o400,
    ),
    "gsplat_runtime_supplement_manifest": (
        _GSPLAT_SUPPLEMENT,
        "87532ef68494442e2ab54885abbd760b7331ea8a83fa72110ea93589a60b1eee",
        0o400,
    ),
    "pinned_python_freeze": (
        _PYTHON_FREEZE,
        "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004",
        0o400,
    ),
    "pinned_python_tree_manifest": (
        _PYTHON_TREE_MANIFEST,
        "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61",
        0o400,
    ),
    "open27_development_decision": (
        _OPEN27_DECISION,
        "110b3c1831898ff6b333f35236401761222f85eafac1dcbcea7b7183d5b434bd",
        0o400,
    ),
    "rigid_residual_rejection_decision": (
        _RIGID_RESIDUAL_DECISION,
        "b72faf6f7d4551622d6abbbd9521f05e46da7ef8cf4e9e17b161896889c7a2fa",
        0o400,
    ),
    "semantic_model_lock": (
        _SEMANTIC_MODEL_LOCK,
        "e5696dc4650194fe2d773a7c5a197862e9d87dda6d7ee5cc45401d5b71f55239",
        0o400,
    ),
    "alltracker_checkpoint": (
        _ALLTRACKER_CHECKPOINT,
        "ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d",
        None,
    ),
    "sam2_checkpoint": (
        _SAM2_CHECKPOINT,
        "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        None,
    ),
}

_EXPECTED_EXTERNAL_ARTIFACT_SHA256: Mapping[str, str] = {
    "v8_attempt2_preoutcome_withdrawal_pointer": (
        "9063011657b955902d1cf7d85a4253eee65caa430a41edae2709a18032baf99c"
    ),
    "v8_attempt2_preoutcome_withdrawal_report": (
        "457c6a64c0208b91ee5eb0f8038d22ae7eda743e29fb60a4bcb4ef1a2861b147"
    ),
    "v8_attempt2_withdrawal_integrity_completion": (
        "eb3a6c092a84dd95f516770d9837711a4f5b1eb58a28fee84c6df0bddb4999b0"
    ),
    "v8_attempt2_manifest_scale_diagnostic": (
        "96f7edc666cda3cf84c6121623028c290b577ceec62cc104a41780b7bb6560ce"
    ),
    "v8_attempt2_admission_compatibility_diagnostic": (
        "e659ceb9b4120c9a2e0c2bf33cbc8478bfc0157ed9b4f9415c3ebef194ea3f80"
    ),
    "v8_external_admission_metadata_only_replay": (
        "1788c212d91d97accb7a6ae2996888ccd879281587f774196e244e66c7c2e8f1"
    ),
    "v8_external_admission_replay_code_binding": (
        "8b27e19b2535ce079a5b38cc1ddd6a693d06bb47ef30eefa8d02ced36e2046d6"
    ),
}

_LOCAL_BINDING_FILES: Mapping[str, str] = {
    "held_v8_lock_preparer_source": "scripts/held/prepare_deform360_v8_lock.py",
    "held_v8_attempt2_withdrawal_operator_source": (
        "scripts/held/seal_deform360_v8_attempt2_withdrawal.py"
    ),
    "held_v8_attempt2_withdrawal_integrity_completion_operator_source": (
        "scripts/held/seal_deform360_v8_attempt2_withdrawal_completion.py"
    ),
    "held_v8_disclosure_sealer_source": (
        "scripts/held/seal_deform360_v8_post_withdrawal_disclosure.py"
    ),
    "held_v8_replacement_source_acquisition_launcher_source": (
        "scripts/held/run_deform360_v8_replacement_source.py"
    ),
    "held_v8_calibration_case_runner_source": (
        "scripts/held/run_deform360_v8_calibration_case.sh"
    ),
    "held_v8_confirmation_case_runner_source": (
        "scripts/held/run_deform360_v8_confirmation_case.sh"
    ),
    "held_v8_common_case_runner_source": (
        "scripts/held/run_deform360_v8_case_common.sh"
    ),
    "held_v8_calibration_shard_runner_source": (
        "scripts/held/run_deform360_v8_calibration_shard.sh"
    ),
    "held_v8_confirmation_shard_runner_source": (
        "scripts/held/run_deform360_v8_confirmation_shard.sh"
    ),
    "held_v8_calibration_outcome_driver_source": (
        "scripts/held/run_deform360_v8_calibration_outcomes.py"
    ),
    "held_v8_confirmation_outcome_driver_source": (
        "scripts/held/run_deform360_v8_confirmation_outcomes.py"
    ),
    "held_v8_x0_query_worker_source": ("scripts/held/run_deform360_v8_x0_query.py"),
    "held_v8_protocol_source": ("src/bayesian_phystwin/deform360_held_v8_protocol.py"),
    "held_v8_replacement_source_operator_source": (
        "src/bayesian_phystwin/deform360_held_v8_replacement_source.py"
    ),
    "held_v8_builder_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v8_builders.py"
    ),
    "held_v8_outcome_driver_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_driver.py"
    ),
    "held_v8_outcome_reconstruction_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_reconstruction.py"
    ),
    "held_v8_gsplat_runtime_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    ),
    "held_v8_query_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_query_artifacts.py"
    ),
    "held_v8_outcome_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_artifacts.py"
    ),
    "held_v8_scoring_source": ("src/bayesian_phystwin/deform360_held_v8_scoring.py"),
    "held_v8_score_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_score_artifacts.py"
    ),
    "held_v8_frozen_query_field_source": (
        "src/bayesian_phystwin/deform360_frozen_query_field.py"
    ),
    # This intentionally overrides the inherited v7 identity: v8 keeps the
    # exhaustive optimizer and changes only its audit serialization schema.
    "frame_zero_builder_source": (
        "src/bayesian_phystwin/deform360_frame_zero_assets.py"
    ),
    "held_official_reconstruction_numerical_source": (
        "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
    ),
    "held_gsplat_runtime_source": (
        "src/bayesian_phystwin/deform360_held_gsplat_runtime.py"
    ),
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _read_file(
    path: str | Path, *, role: str, required_mode: int | None = None
) -> tuple[Path, bytes, os.stat_result]:
    source = _absolute(path)
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"{role} is not a regular file",
    )
    _require(source.resolve() == source, f"{role} has a symlinked ancestor")
    if required_mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == required_mode,
            f"{role} mode changed",
        )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{role} changed while opening",
        )
        digest = hashlib.sha256()
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(source)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    _require(
        identity
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        == (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ),
        f"{role} changed while hashing",
    )
    _require(digest.digest() == hashlib.sha256(payload).digest(), "hash state changed")
    return source, bytes(payload), after


def _sha256_file(
    path: str | Path, *, role: str, required_mode: int | None = None
) -> str:
    _, payload, _ = _read_file(path, role=role, required_mode=required_mode)
    return hashlib.sha256(payload).hexdigest()


def _run_git(
    root: Path, arguments: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if check and completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed


def _parse_git_tree(raw: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        header, separator, path_bytes = encoded.partition(b"\t")
        _require(bool(separator) and bool(path_bytes), "malformed Git tree record")
        fields = header.split(b" ")
        _require(len(fields) == 3, "malformed Git tree header")
        mode, kind, object_id = (field.decode("ascii") for field in fields)
        path = path_bytes.decode("utf-8")
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and len(object_id) in {40, 64}
            and all(character in "0123456789abcdef" for character in object_id),
            f"unsupported tracked entry: {path}",
        )
        _require(
            path and not path.startswith("/") and ".." not in Path(path).parts,
            "unsafe tracked path",
        )
        records.append(
            {"mode": mode, "type": kind, "object_id": object_id, "path": path}
        )
    _require(bool(records), "Git tree is empty")
    _require(
        [record["path"] for record in records]
        == sorted(record["path"] for record in records),
        "Git tree paths are not sorted",
    )
    return records


def _validate_repository(root: str | Path) -> dict[str, Any]:
    code = _absolute(root)
    observed = os.lstat(code)
    _require(
        stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        "source code is not a real directory",
    )
    _require(code.resolve() == code, "source code has a symlinked ancestor")
    _require((code / ".git").is_dir(), "source code is not a non-bare Git repository")
    top = _run_git(code, ["rev-parse", "--show-toplevel"]).stdout.decode().strip()
    _require(top == str(code), "source Git top level changed")
    head = _run_git(code, ["rev-parse", "HEAD"]).stdout.decode().strip().lower()
    _require(
        len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head),
        "source HEAD is invalid",
    )
    _require(
        _run_git(code, ["status", "--porcelain=v1", "--untracked-files=all"]).stdout
        == b"",
        "source worktree is not completely clean",
    )
    _require(
        _run_git(code, ["rev-parse", "--is-shallow-repository"]).stdout.decode().strip()
        == "false",
        "source repository is shallow",
    )
    _run_git(code, ["fsck", "--full", "--no-dangling"])
    records = _parse_git_tree(_run_git(code, ["ls-tree", "-r", "-z", "HEAD"]).stdout)
    _require(
        all((code / record["path"]).is_file() for record in records),
        "source tracked file is absent",
    )
    return {
        "root": code,
        "head": head,
        "head_text_sha256": _sha256_text(head),
        "tree_records": records,
        "tree_sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
    }


def _require_deployed_read_only(code: Path) -> None:
    for root, directories, files in os.walk(code, followlinks=False):
        root_path = Path(root)
        for name in [*directories, *files]:
            path = root_path / name
            observed = os.lstat(path)
            _require(
                not stat.S_ISLNK(observed.st_mode), "deployment contains a symlink"
            )
            _require(observed.st_mode & 0o222 == 0, f"deployment is writable: {path}")


def _make_read_only(code: Path) -> None:
    paths: list[Path] = []
    for root, directories, files in os.walk(code, topdown=False, followlinks=False):
        root_path = Path(root)
        paths.extend(root_path / name for name in files)
        paths.extend(root_path / name for name in directories)
    paths.append(code)
    for path in paths:
        observed = os.lstat(path)
        _require(not stat.S_ISLNK(observed.st_mode), "deployment contains a symlink")
        if stat.S_ISDIR(observed.st_mode):
            mode = 0o555
        elif stat.S_ISREG(observed.st_mode):
            mode = 0o555 if observed.st_mode & 0o111 else 0o444
        else:
            raise ValueError(f"deployment contains a special file: {path}")
        os.chmod(path, mode, follow_symlinks=False)
    _require_deployed_read_only(code)


def _clone_staged_deployment(source: Path, head: str, stage: Path) -> dict[str, Any]:
    _require(not os.path.lexists(stage), "deployment stage already exists")
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    completed = subprocess.run(
        ["git", "clone", "--no-hardlinks", "--no-local", str(source), str(stage)],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    _require(
        completed.returncode == 0,
        "failed to clone independent deployment: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )
    try:
        _run_git(stage, ["checkout", "--detach", head])
        _run_git(stage, ["remote", "remove", "origin"])
        observed = _validate_repository(stage)
        _require(observed["head"] == head, "staged deployment HEAD changed")
        _make_read_only(stage)
        return observed
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _external_bindings() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, (
        path,
        expected_sha256,
        required_mode,
    ) in _EXPECTED_EXTERNAL_FILES.items():
        observed = _sha256_file(
            path, role=name.replace("_", " "), required_mode=required_mode
        )
        _require(observed == expected_sha256, f"{name} SHA-256 changed")
        result[name] = observed
        expected_artifact = _EXPECTED_EXTERNAL_ARTIFACT_SHA256.get(name)
        if expected_artifact is not None:
            _, payload, _ = _read_file(
                path,
                role=f"{name.replace('_', ' ')} artifact",
                required_mode=required_mode,
            )
            try:
                artifact = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{name} is not canonical JSON") from error
            _require(
                isinstance(artifact, dict)
                and artifact.get("artifact_sha256") == expected_artifact,
                f"{name} artifact SHA-256 field changed",
            )
            unsigned = dict(artifact)
            unsigned.pop("artifact_sha256")
            _require(
                hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
                == expected_artifact,
                f"{name} canonical artifact SHA-256 changed",
            )
            result[f"{name}_artifact"] = expected_artifact
    result["pinned_python_executable_target"] = _validate_pinned_python()
    return result


def _inherited_v7_bindings() -> dict[str, str]:
    """Load numerical/runtime pins from the exact sealed v7 parent lock.

    These are source and runtime identities, not v7 predictions or outcomes.
    The v8-specific method/tree bindings are overlaid later.
    """

    _, payload, _ = _read_file(
        _V7_LOCK, role="sealed v7 calibration lock", required_mode=0o400
    )
    _require(
        hashlib.sha256(payload).hexdigest()
        == "b464d7cfda3b4ad94f57ffd46267b3b50d8dc65e2ff8dfec2befc7953718aca7",
        "sealed v7 calibration lock changed before inheritance",
    )
    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sealed v7 calibration lock is not JSON") from error
    _require(
        isinstance(artifact, dict)
        and artifact.get("protocol_id") == "deform360-held-online-belief-v7",
        "sealed parent lock is not exact held v7",
    )
    raw = artifact.get("immutable_bindings")
    _require(isinstance(raw, dict) and bool(raw), "v7 immutable bindings are absent")
    bindings = {str(key): str(value) for key, value in raw.items()}
    _require(
        all(
            key
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for key, value in bindings.items()
        ),
        "v7 immutable binding is not named SHA-256",
    )
    return dict(sorted(bindings.items()))


def _validate_pinned_python() -> str:
    """Validate the exact venv launcher symlink and its system interpreter."""

    observed = os.lstat(_PINNED_PYTHON)
    _require(
        stat.S_ISLNK(observed.st_mode)
        and os.readlink(_PINNED_PYTHON) == "/usr/bin/python3",
        "pinned Python launcher symlink changed",
    )
    target = _PINNED_PYTHON.resolve(strict=True)
    _require(
        target == Path("/usr/bin/python3.12")
        and target.is_file()
        and not target.is_symlink()
        and os.access(target, os.X_OK),
        "pinned Python target changed or is not executable",
    )
    digest = _sha256_file(target, role="pinned Python executable target")
    _require(
        digest == "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7",
        "pinned Python executable target SHA-256 changed",
    )
    return digest


def _local_file_bindings(code: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative in _LOCAL_BINDING_FILES.items():
        result[name] = _sha256_file(code / relative, role=name.replace("_", " "))
    return result


def _validate_attempt2_operator_source_lineage(
    local_bindings: Mapping[str, str],
) -> None:
    _, payload, _ = _read_file(
        _V8_ATTEMPT2_INTEGRITY_COMPLETION,
        role="attempt-2 withdrawal integrity completion",
        required_mode=0o400,
    )
    try:
        completion = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("attempt-2 integrity completion is not JSON") from error
    records = completion.get("operator_source_bindings")
    _require(isinstance(records, Mapping), "attempt-2 operator bindings are absent")
    expected = {
        "held_v8_attempt2_withdrawal_operator_source": ("attempt2_withdrawal_operator"),
        "held_v8_attempt2_withdrawal_integrity_completion_operator_source": (
            "attempt2_integrity_completion_operator"
        ),
    }
    for local_name, completion_name in expected.items():
        record = records.get(completion_name)
        _require(
            isinstance(record, Mapping)
            and record.get("sha256") == local_bindings.get(local_name),
            f"{local_name} differs from the executed operator source",
        )


def _validate_admission_replay_source_lineage(
    local_bindings: Mapping[str, str],
) -> None:
    _, payload, _ = _read_file(
        _V8_ADMISSION_REPLAY_CODE_BINDING,
        role="v8 admission replay code binding",
        required_mode=0o400,
    )
    try:
        replay = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v8 admission replay code binding is not JSON") from error
    tested = replay.get("local_worktree_at_replay")
    _require(isinstance(tested, Mapping), "replay-tested source binding is absent")
    for local_name, replay_name in (
        ("held_v8_builder_adapter_source", "adapter_source_sha256"),
        ("held_v8_protocol_source", "protocol_source_sha256"),
    ):
        _require(
            tested.get(replay_name) == local_bindings.get(local_name),
            f"{local_name} differs from the real pinned-upstream replay",
        )


def _import_v8_modules(code: Path) -> tuple[Any, Any]:
    source_root = code / "src"
    sys.path.insert(0, str(source_root))
    try:
        from bayesian_phystwin import deform360_held_v8_protocol as protocol
        from bayesian_phystwin import (
            deform360_held_v8_replacement_source as replacement,
        )
    finally:
        sys.path.pop(0)
    for module, label in ((protocol, "protocol"), (replacement, "replacement")):
        module_path = Path(module.__file__).resolve()
        _require(
            module_path.is_relative_to(source_root),
            f"{label} module imported outside the clean source tree",
        )
    return protocol, replacement


def _processing_revision() -> str:
    code = _absolute(_DEFORM360_CODE)
    _require(code.is_dir() and code.resolve() == code, "Deform360 code is absent")
    revision = _run_git(code, ["rev-parse", "HEAD"]).stdout.decode().strip().lower()
    return revision


def prospective_bindings(
    source_code: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    provenance = _validate_repository(source_code)
    code = provenance["root"]
    _require(
        Path(__file__).resolve()
        == code / "scripts" / "held" / "prepare_deform360_v8_lock.py",
        "lock preparer is not the tracked source operator",
    )
    external = _external_bindings()
    inherited = _inherited_v7_bindings()
    bindings = dict(inherited)
    bindings["v7_inherited_immutable_bindings_contract"] = hashlib.sha256(
        _canonical_bytes(inherited)
    ).hexdigest()
    bindings.update(external)
    local_bindings = _local_file_bindings(code)
    _validate_attempt2_operator_source_lineage(local_bindings)
    _validate_admission_replay_source_lineage(local_bindings)
    bindings.update(local_bindings)
    protocol, replacement = _import_v8_modules(code)
    processing_revision = _processing_revision()
    _require(
        processing_revision == replacement.PROCESSING_CODE_REVISION,
        "Deform360 processing revision changed",
    )
    bindings.update(
        {
            "method_deployed_snapshot_tree": provenance["tree_sha256"],
            "method_head_text_sha256": provenance["head_text_sha256"],
            "replacement_source_inventory_contract": protocol.held_contract_sha256(
                replacement.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
            ),
            "replacement_automatic_twin_admission_contract": (
                protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256
            ),
            "frame_zero_exact_eight_subset_bounded_audit_contract": (
                protocol.frame_zero_assets.EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256
            ),
            "frozen_query_field_contract": protocol.held_contract_sha256(
                protocol.FROZEN_FIELD_CONTRACT
            ),
            "primary_method_contract": protocol.held_contract_sha256(
                protocol.PRIMARY_METHOD
            ),
            "deform360_processing_head_text_sha256": _sha256_text(processing_revision),
            "hf_dataset_revision_text_sha256": _sha256_text(
                replacement.HF_DATASET_REVISION
            ),
        }
    )
    _require(
        all(
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in bindings.values()
        ),
        "prospective binding is not SHA-256",
    )
    return dict(sorted(bindings.items())), provenance


def _disclosure_environment() -> dict[str, str]:
    return {
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": "/tmp",
        "USER": "florianpfaff",
    }


def _seal_disclosure(source_code: Path) -> None:
    operator = (
        source_code
        / "scripts"
        / "held"
        / "seal_deform360_v8_post_withdrawal_disclosure.py"
    )
    completed = subprocess.run(
        [str(_PINNED_PYTHON), "-I", "-B", str(operator)],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=source_code,
        env=_disclosure_environment(),
    )
    _require(
        completed.returncode == 0,
        "disclosure sealer failed: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )


def create_lock_and_deployment(source_code: str | Path) -> dict[str, Any]:
    _require(
        socket.gethostname() == "workstation2", "formal lock must run on workstation2"
    )
    _require(not os.path.lexists(_HELD_ROOT), "formal held-v8 root is not fresh")
    bindings, provenance = prospective_bindings(source_code)
    source = provenance["root"]
    head = provenance["head"]
    stage = _HELD_BASE / f".held-v8-code-stage-{head}"
    destination = _HELD_ROOT / f"code-{head}"
    staged = _clone_staged_deployment(source, head, stage)
    _require(
        staged["tree_sha256"] == provenance["tree_sha256"],
        "staged deployment tree differs from source",
    )
    protocol, _replacement = _import_v8_modules(source)
    capability = protocol.prepare_fresh_held_root(_HELD_ROOT)
    deployment_moved = False
    try:
        _seal_disclosure(source)
        disclosure_sha256 = _sha256_file(
            _DISCLOSURE_PATH,
            role="post-withdrawal disclosure",
            required_mode=0o400,
        )
        bindings["post_withdrawal_development_use_disclosure"] = disclosure_sha256
        final_external = _external_bindings()
        _require(
            all(bindings.get(name) == value for name, value in final_external.items()),
            "external immutable input changed during lock preparation",
        )
        lock = protocol.create_calibration_protocol_lock(
            _LOCK_PATH,
            held_root=_HELD_ROOT,
            fresh_root_capability=capability,
            immutable_bindings=bindings,
            v7_withdrawal_report_path=_V7_WITHDRAWAL,
            post_withdrawal_disclosure_path=_DISCLOSURE_PATH,
            development_decision_path=_OPEN27_DECISION,
        )
        _require(not os.path.lexists(destination), "deployment destination exists")
        # The Corsair filesystem refuses to rename a directory whose own
        # owner-write bit is absent, even though POSIX rename ordinarily only
        # requires write permission on the two parents.  Descendants remain
        # immutable; expose the staging root bit only for the atomic move and
        # remove it again before any deployed validation or execution.
        os.chmod(stage, 0o755, follow_symlinks=False)
        os.rename(stage, destination)
        deployment_moved = True
        os.chmod(destination, 0o555, follow_symlinks=False)
        _require_deployed_read_only(destination)
        deployed = _validate_repository(destination)
        _require(
            deployed["head"] == head
            and deployed["tree_sha256"] == bindings["method_deployed_snapshot_tree"],
            "deployed repository differs after atomic move",
        )
        validated = protocol.validate_protocol_lock(_LOCK_PATH)
        _require(validated == lock, "calibration lock changed after deployment")
        return {
            "operation": "created_held_v8_calibration_lock_and_deployment",
            "protocol_id": lock["protocol_id"],
            "lock_path": str(_LOCK_PATH),
            "lock_file_sha256": _sha256_file(
                _LOCK_PATH, role="calibration lock", required_mode=0o400
            ),
            "lock_artifact_sha256": lock["artifact_sha256"],
            "deployed_code": str(destination),
            "deployed_head": head,
            "deployed_tree_sha256": deployed["tree_sha256"],
            "binding_count": len(bindings),
            "formal_root_was_absent": True,
        }
    except BaseException:
        if not deployment_moved and os.path.lexists(stage):
            # The stage is outside the formal root and contains no outcome.
            # It is safe to remove; the formal root is deliberately retained
            # as incident evidence and must never be silently retried as v8.
            for root, directories, files in os.walk(stage, topdown=False):
                for name in files:
                    os.chmod(Path(root) / name, 0o600, follow_symlinks=False)
                for name in directories:
                    os.chmod(Path(root) / name, 0o700, follow_symlinks=False)
            os.chmod(stage, 0o700, follow_symlinks=False)
            shutil.rmtree(stage)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-code", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--create", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    _require(
        sys.flags.isolated == 1 and sys.dont_write_bytecode,
        "run lock preparation with Python -I -B",
    )
    if arguments.preflight:
        bindings, provenance = prospective_bindings(arguments.source_code)
        result = {
            "operation": "preflight_only",
            "formal_root_absent": not os.path.lexists(_HELD_ROOT),
            "source_head": provenance["head"],
            "source_tree_sha256": provenance["tree_sha256"],
            "prospective_binding_count": len(bindings) + 1,
            "prospective_bindings": bindings,
        }
    else:
        result = create_lock_and_deployment(arguments.source_code)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
