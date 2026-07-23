#!/usr/bin/env python3
"""Replay the v8.1 object-072 admission wrapper without outcome access.

This development-only operator reads the already sealed attempt-2
``prediction_only_input.pkl`` and executes only the automatic-twin builder.
It never loads an official target, an x0 query, a queried prediction, a score,
or a confirmation artifact.  A second invocation with the wrong episode must
be rejected before producing numerical outputs.

The resulting report and source binding are inputs to the attempt-5 lock
preparer.  Run this operator from an exact clean Git checkout on workstation2
with the pinned interpreter and no ``PYTHONPATH`` dependency::

    /mnt/corsair/florianpfaff/bpt-held-v5-runtimes/\
bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/\
bin/python -I -B /absolute/checkout/scripts/held/\
replay_deform360_v81_external_admission.py

Only standard-library modules are imported until the checkout has passed its
cleanliness and Git-rewrite preflight.  The exact checked ``src`` directory is
then installed for the two deferred project imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
from typing import Any

ROOT = Path(
    "/mnt/corsair/florianpfaff/"
    "bpt-held-v8.1-attempt-5-admission-wrapper-scratch-20260722"
)
QUALIFICATION_BASE = Path("/mnt/corsair/florianpfaff")
QUALIFICATION_ROOT_PREFIX = "bpt-resource-lifecycle-qualification-"
SOURCE_INPUT = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/"
    "held-v8-attempt-2-withdrawn-preoutcome/calibration/cases/"
    "072-cotton-clohesline-ep0003/physical/prediction_only_input.pkl"
)
SOURCE_INPUT_SHA256 = "2f783d15426759a0928fcb6cb8a98fa61b38d582a46ec006d296d53b439ae015"
SOURCE_INPUT_SIZE_BYTES = 19_261_048
PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
PINNED_PYTHON_RUNTIME = PINNED_PYTHON.parent.parent
PINNED_PYTHON_TARGET = Path("/usr/bin/python3.12")
PINNED_PYTHON_BASE_PREFIX = Path("/usr")
PINNED_PYTHON_TARGET_SHA256 = (
    "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7"
)
PYTHON_RUNTIME_FREEZE = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
    ".freeze.sorted.txt"
)
PYTHON_RUNTIME_TREE_MANIFEST = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
    ".tree-manifest.json"
)
UPSTREAM = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "Bayesian-PhysTwin-upstream-58ab4808e59d"
)
UPSTREAM_HEAD = "58ab4808e59da811dd1a2c66ac628fe4ea2faeab"
UPSTREAM_TREE = "2b35d539be7a17b2de2c644b46c267b16ce26bf0"
UPSTREAM_AUTOMATIC_TWIN_BUILDER = (
    UPSTREAM / "scripts/remote/build_deform360_automatic_episode_twin.py"
)
UPSTREAM_DENSE_PANEL_AUTHORIZER = (
    UPSTREAM / "src/causal4d_public/deform360_dense_reusable_panel.py"
)
DEFORM360_HEAD = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
DEFORM360_TREE = "c566ed29db7e0fd6a4cb768d840a4aa662864680"
DEFORM360 = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
    f"Deform360-processing-{DEFORM360_HEAD}"
)
CASE_NAME = "072-cotton-clohesline-ep0003"
OBJECT_ID = "072-cotton-clohesline"
EPISODE_ID = 3
WRONG_EPISODE_ID = 4
REPORT_NAME = "metadata-only-replay-report.json"
CODE_BINDING_NAME = "metadata-only-replay-code-binding.json"
CROSS_AUTHORIZATION_REJECTION_MARKER = (
    "outside the exact v8 external calibration admission"
)
CHILD_CWD = Path("/home/florianpfaff")
CHILD_TIMEOUT_SECONDS = 1_800
SUCCESS_OUTPUT_NAMES = frozenset(
    {
        "episode_graph.npz",
        "simulator_final_data.pkl",
        "state_artifact.npz",
        "twin_summary.json",
    }
)
SUCCESS_INFORMATION_BOUNDARY = {
    "contact_conditioned_action_result_sha256": None,
    "contact_conditioned_action_used": False,
    "future_object_tracks_present": False,
    "future_robot_action_available": True,
    "object_observation_frames_used": [0],
    "post_initial_object_observation_used": False,
    "prediction_only_input_required": True,
    "simulator_residual_used": False,
    "target_access": False,
}
OPERATOR_RELATIVE = Path("scripts/held/replay_deform360_v81_external_admission.py")
PROTOCOL_RELATIVE = Path("src/bayesian_phystwin/deform360_held_v8_protocol.py")
ADAPTER_RELATIVE = Path("src/bayesian_phystwin/deform360_held_v8_builders.py")
PROTOCOL_MODULE = "bayesian_phystwin.deform360_held_v8_protocol"
ADAPTER_MODULE = "bayesian_phystwin.deform360_held_v8_builders"

# These are populated only by ``_load_verified_source_modules`` after the
# stdlib-only source preflight succeeds.  Keeping the names explicit preserves
# narrow unit-test seams without permitting an import at operator load time.
builders: Any | None = None
protocol: Any | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_isolated_launch() -> None:
    _require(
        sys.flags.isolated == 1
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.flags.dont_write_bytecode == 1,
        "replay must be launched with the pinned Python using -I -B",
    )
    executable = Path(os.path.abspath(sys.executable))
    base_executable_value = getattr(sys, "_base_executable", None)
    _require(
        isinstance(base_executable_value, str) and base_executable_value,
        "replay Python has no base executable",
    )
    base_executable = Path(os.path.abspath(base_executable_value))
    prefix = Path(os.path.abspath(sys.prefix))
    base_prefix = Path(os.path.abspath(sys.base_prefix))
    _require(
        executable == PINNED_PYTHON,
        "replay is not executing through the pinned Python launcher",
    )
    _require(
        base_executable == PINNED_PYTHON_TARGET,
        "replay Python base executable changed",
    )
    _require(prefix == PINNED_PYTHON_RUNTIME, "replay Python prefix changed")
    _require(
        base_prefix == PINNED_PYTHON_BASE_PREFIX,
        "replay Python base prefix changed",
    )


def _source_modules() -> tuple[Any, Any]:
    _require(
        builders is not None and protocol is not None,
        "verified replay source modules have not been loaded",
    )
    return builders, protocol


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _read_regular(path: Path, *, role: str, mode: int | None = None) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(absolute)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and absolute.resolve() == absolute,
        f"{role} is not a canonical regular file",
    )
    if mode is not None:
        _require(stat.S_IMODE(before.st_mode) == mode, f"{role} mode changed")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{role} changed while opening",
        )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"{role} changed while reading",
    )
    return b"".join(chunks)


def _file_record(
    path: Path, *, role: str, reported_path: Path | None = None
) -> dict[str, Any]:
    payload = _read_regular(path, role=role)
    return {
        "path": str(path if reported_path is None else reported_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                _require(written > 0, "short write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _run_git(root: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            *arguments,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/home/florianpfaff",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    _require(
        result.returncode == 0,
        f"git {' '.join(arguments)} failed: {result.stderr.strip()}",
    )
    return result.stdout.strip()


def _require_ordinary_git_index(root: Path, *, role: str) -> None:
    tracked = _run_git(root, ["ls-files", "-v"])
    lines = tracked.splitlines()
    _require(lines, f"{role} repository has no tracked files")
    _require(
        all(line.startswith("H ") and len(line) > 2 for line in lines),
        f"{role} repository contains non-ordinary index flags",
    )


def _require_no_git_rewrite_state(root: Path, *, role: str) -> None:
    replacement_refs = _run_git(
        root,
        ["for-each-ref", "--format=%(refname)", "refs/replace"],
    )
    _require(
        replacement_refs == "",
        f"{role} repository contains replacement refs",
    )
    git_directory = Path(_run_git(root, ["rev-parse", "--absolute-git-dir"]))
    git_state = os.lstat(git_directory)
    _require(
        git_directory.is_absolute()
        and stat.S_ISDIR(git_state.st_mode)
        and not stat.S_ISLNK(git_state.st_mode)
        and git_directory.resolve(strict=True) == git_directory,
        f"{role} Git directory is not canonical",
    )
    _require(
        not os.path.lexists(git_directory / "info/grafts"),
        f"{role} repository contains a grafts file",
    )


def _repository_binding(
    root: Path, *, expected_head: str, expected_tree: str, role: str
) -> dict[str, Any]:
    observed = os.lstat(root)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and root.resolve() == root
        and observed.st_mode & 0o222 == 0,
        f"{role} is not a canonical directory",
    )
    for current, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            entry = os.lstat(Path(current) / name)
            _require(
                stat.S_ISDIR(entry.st_mode)
                and not stat.S_ISLNK(entry.st_mode)
                and entry.st_mode & 0o222 == 0,
                f"{role} contains an unsafe directory",
            )
        for name in files:
            entry = os.lstat(Path(current) / name)
            _require(
                stat.S_ISREG(entry.st_mode)
                and not stat.S_ISLNK(entry.st_mode)
                and entry.st_mode & 0o222 == 0,
                f"{role} contains an unsafe file",
            )
    prefix = ["-c", "core.fileMode=false"]
    _require_no_git_rewrite_state(root, role=role)
    _require_ordinary_git_index(root, role=role)
    head = _run_git(root, [*prefix, "rev-parse", "HEAD"]).lower()
    tree = _run_git(root, [*prefix, "rev-parse", "HEAD^{tree}"]).lower()
    _require(
        head == expected_head
        and tree == expected_tree
        and _run_git(
            root,
            [*prefix, "status", "--porcelain=v1", "--untracked-files=all"],
        )
        == ""
        and _run_git(
            root,
            [
                *prefix,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
            ],
        )
        == ""
        and _run_git(root, [*prefix, "rev-parse", "--is-shallow-repository"])
        == "false",
        f"{role} repository identity changed",
    )
    _run_git(root, [*prefix, "fsck", "--full", "--no-dangling"])
    return {
        "repository_root": str(root),
        "git_head": head,
        "git_tree": tree,
        "clean_tracked_and_untracked": True,
        "ignored_files_absent": True,
        "fully_nonwritable": True,
    }


def _runtime_binding() -> dict[str, Any]:
    launcher = os.lstat(PINNED_PYTHON)
    _require(
        stat.S_ISLNK(launcher.st_mode)
        and os.readlink(PINNED_PYTHON) == "/usr/bin/python3"
        and PINNED_PYTHON.resolve(strict=True) == PINNED_PYTHON_TARGET,
        "pinned Python launcher identity changed",
    )
    target = _file_record(PINNED_PYTHON_TARGET, role="pinned Python target")
    freeze = _file_record(PYTHON_RUNTIME_FREEZE, role="pinned Python freeze")
    tree = _file_record(
        PYTHON_RUNTIME_TREE_MANIFEST,
        role="pinned Python tree manifest",
    )
    _require(
        target["sha256"] == PINNED_PYTHON_TARGET_SHA256
        and freeze["sha256"]
        == "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
        and tree["sha256"]
        == "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61",
        "pinned Python runtime identity changed",
    )
    upstream = _repository_binding(
        UPSTREAM,
        expected_head=UPSTREAM_HEAD,
        expected_tree=UPSTREAM_TREE,
        role="pinned upstream",
    )
    builder = _file_record(
        UPSTREAM_AUTOMATIC_TWIN_BUILDER,
        role="pinned upstream automatic-twin builder",
    )
    authorizer = _file_record(
        UPSTREAM_DENSE_PANEL_AUTHORIZER,
        role="pinned upstream dense-panel authorizer",
    )
    _require(
        builder["sha256"]
        == "dd43bfeaa0ddb53252e3b2d9c907c147379b2cce6b4c5d5dfa14f310fdacfa9a"
        and authorizer["sha256"]
        == "0861831b9ab3cf6d64833efe533073f4f444f2315c04057377f243efffd8b17e",
        "pinned upstream numerical source changed",
    )
    return {
        "python": {
            "launcher_path": str(PINNED_PYTHON),
            "launcher_target": "/usr/bin/python3",
            "executable_target": target,
            "environment_freeze": freeze,
            "tree_manifest": tree,
        },
        "upstream": {
            **upstream,
            "automatic_twin_builder": builder,
            "dense_panel_authorizer": authorizer,
        },
        "deform360": _repository_binding(
            DEFORM360,
            expected_head=DEFORM360_HEAD,
            expected_tree=DEFORM360_TREE,
            role="pinned Deform360 processing snapshot",
        ),
    }


def _operator_source_layout(operator_path: Path | None = None) -> tuple[Path, Path]:
    operator = Path(
        os.path.abspath(os.fspath(__file__ if operator_path is None else operator_path))
    )
    operator_state = os.lstat(operator)
    _require(
        stat.S_ISREG(operator_state.st_mode)
        and not stat.S_ISLNK(operator_state.st_mode)
        and operator_state.st_nlink == 1
        and operator.resolve(strict=True) == operator,
        "admission replay operator is not a canonical single-link regular file",
    )
    _require(
        len(operator.parents) >= 3,
        "admission replay operator path is too shallow",
    )
    code = operator.parents[2]
    code_state = os.lstat(code)
    _require(
        stat.S_ISDIR(code_state.st_mode)
        and not stat.S_ISLNK(code_state.st_mode)
        and code.resolve(strict=True) == code
        and operator == code / OPERATOR_RELATIVE,
        "operator is outside the exact code-root location",
    )
    return code, operator


def _preflight_source_checkout(*, operator_path: Path | None = None) -> dict[str, Any]:
    """Validate the source checkout without importing any project module."""

    code, operator = _operator_source_layout(operator_path)
    top_level = Path(_run_git(code, ["rev-parse", "--show-toplevel"]))
    _require(
        top_level.is_absolute() and top_level.resolve(strict=True) == code,
        "operator code root is not the exact Git top level",
    )
    _require_no_git_rewrite_state(code, role="source checkout")
    _require_ordinary_git_index(code, role="source checkout")
    status = _run_git(code, ["status", "--porcelain=v1", "--untracked-files=all"])
    ordinary_untracked = _run_git(code, ["ls-files", "--others", "--exclude-standard"])
    ignored_untracked = _run_git(
        code,
        ["ls-files", "--others", "--ignored", "--exclude-standard"],
    )
    _require(
        status == "" and ordinary_untracked == "",
        "source checkout is dirty",
    )
    _require(ignored_untracked == "", "source checkout contains ignored files")
    _require(
        _run_git(code, ["rev-parse", "--is-shallow-repository"]) == "false",
        "source checkout is shallow",
    )
    _run_git(code, ["fsck", "--full", "--no-dangling"])
    head = _run_git(code, ["rev-parse", "HEAD"]).lower()
    tree = _run_git(code, ["rev-parse", "HEAD^{tree}"]).lower()
    _require(
        len(head) == 40
        and all(character in "0123456789abcdef" for character in head)
        and len(tree) == 40
        and all(character in "0123456789abcdef" for character in tree),
        "source checkout Git identity is invalid",
    )
    protocol_path = code / PROTOCOL_RELATIVE
    adapter_path = code / ADAPTER_RELATIVE
    source_root = code / "src"
    source_root_state = os.lstat(source_root)
    _require(
        stat.S_ISDIR(source_root_state.st_mode)
        and not stat.S_ISLNK(source_root_state.st_mode)
        and source_root.resolve(strict=True) == source_root,
        "source checkout src directory is not canonical",
    )
    return {
        "code_root": code,
        "source_root": source_root,
        "operator_path": operator,
        "protocol_path": protocol_path,
        "adapter_path": adapter_path,
        "git_head": head,
        "git_tree": tree,
        "protocol_source_sha256": hashlib.sha256(
            _read_regular(protocol_path, role="v8.1 protocol source")
        ).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(
            _read_regular(adapter_path, role="v8.1 adapter source")
        ).hexdigest(),
        "replay_operator_source_sha256": hashlib.sha256(
            _read_regular(operator, role="admission replay operator")
        ).hexdigest(),
    }


def _load_verified_source_modules(preflight: Mapping[str, Any]) -> None:
    """Import exact project modules only after the stdlib preflight passes."""

    global builders, protocol
    _require(
        builders is None and protocol is None,
        "verified replay source modules were already loaded",
    )
    polluted = sorted(
        name
        for name in sys.modules
        if name == "bayesian_phystwin" or name.startswith("bayesian_phystwin.")
    )
    _require(not polluted, "bayesian_phystwin was imported before source preflight")
    source_root = preflight.get("source_root")
    _require(isinstance(source_root, Path), "source preflight root is invalid")
    source_entry = str(source_root)
    _require(
        source_entry not in sys.path and source_entry not in sys.path_importer_cache,
        "verified source path was active before source preflight",
    )
    sys.path.insert(0, source_entry)
    try:
        imported_protocol = importlib.import_module(PROTOCOL_MODULE)
        imported_builders = importlib.import_module(ADAPTER_MODULE)
    finally:
        _require(
            sys.path and sys.path[0] == source_entry,
            "project import changed the verified source-path position",
        )
        del sys.path[0]
        sys.path_importer_cache.pop(source_entry, None)

    protocol_path = preflight.get("protocol_path")
    adapter_path = preflight.get("adapter_path")
    _require(
        isinstance(protocol_path, Path)
        and isinstance(adapter_path, Path)
        and Path(imported_protocol.__file__).resolve(strict=True) == protocol_path
        and Path(imported_builders.__file__).resolve(strict=True) == adapter_path,
        "imported v8.1 source is outside the exact checkout",
    )
    refreshed = _preflight_source_checkout(operator_path=preflight.get("operator_path"))
    _require(refreshed == dict(preflight), "source checkout changed during import")
    protocol = imported_protocol
    builders = imported_builders


def _bootstrap_source_modules(*, operator_path: Path | None = None) -> dict[str, Any]:
    preflight = _preflight_source_checkout(operator_path=operator_path)
    _load_verified_source_modules(preflight)
    return preflight


def _source_binding(preflight: Mapping[str, Any]) -> dict[str, Any]:
    builders_module, protocol_module = _source_modules()
    protocol_path = preflight.get("protocol_path")
    adapter_path = preflight.get("adapter_path")
    operator_path = preflight.get("operator_path")
    _require(
        isinstance(protocol_path, Path)
        and isinstance(adapter_path, Path)
        and isinstance(operator_path, Path)
        and Path(protocol_module.__file__).resolve(strict=True) == protocol_path
        and Path(builders_module.__file__).resolve(strict=True) == adapter_path,
        "imported v8.1 source is outside the exact checkout",
    )
    return {
        "git_head": preflight["git_head"],
        "protocol_source_sha256": preflight["protocol_source_sha256"],
        "adapter_source_sha256": preflight["adapter_source_sha256"],
        "replay_operator_source_sha256": preflight["replay_operator_source_sha256"],
        "exact_child_bootstrap_sha256": hashlib.sha256(
            builders_module._V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP.encode("utf-8")
        ).hexdigest(),
        "clean_tracked_and_untracked": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
        "uncommitted_correction_present": False,
        "external_runtime": _runtime_binding(),
    }


def _qualification_lineage(source_binding: Mapping[str, Any]) -> dict[str, Any]:
    _, protocol_module = _source_modules()
    head = source_binding.get("git_head")
    _require(
        isinstance(head, str)
        and len(head) == 40
        and all(character in "0123456789abcdef" for character in head),
        "replay source HEAD is invalid",
    )
    root = QUALIFICATION_BASE / f"{QUALIFICATION_ROOT_PREFIX}{head}"
    evidence = root / "resource-lifecycle-qualification.json"
    completion = Path(f"{root}-integrity-completion.json")
    lineage = protocol_module.validate_resource_lifecycle_qualification_lineage(
        evidence_path=evidence,
        completion_path=completion,
        verify_content_inventory=True,
        require_admission=True,
    )
    integrity = lineage.get("resource_lifecycle_qualification_integrity")
    _require(
        isinstance(integrity, Mapping)
        and integrity.get("source_head") == head
        and integrity.get("terminal_outcome") == "qualified"
        and integrity.get("admission_eligible") is True
        and integrity.get("generator_profile") == "same-as-analyzer"
        and integrity.get("physical_gpu_index") == 1
        and integrity.get("analyzer_source_sha256")
        == protocol_module.RESOURCE_LIFECYCLE_ANALYZER_SOURCE_SHA256,
        "replay qualification does not admit this exact H1 source",
    )
    return lineage


def _arguments(root: Path, *, episode_id: int) -> list[str]:
    return [
        "--repo",
        str(UPSTREAM),
        "--object-id",
        OBJECT_ID,
        "--episode-id",
        str(episode_id),
        "--phase",
        "calibration",
        "--episode-final-data",
        str(SOURCE_INPUT),
        "--episode-graph",
        str(root / "episode_graph.npz"),
        "--simulator-final-data",
        str(root / "simulator_final_data.pkl"),
        "--state-artifact",
        str(root / "state_artifact.npz"),
        "--summary",
        str(root / "twin_summary.json"),
        "--prediction-only-input",
        "--canonical-node-count",
        "1024",
        "--source-admission-passed",
    ]


def _command(root: Path, *, episode_id: int) -> list[str]:
    builders_module, _ = _source_modules()
    script = UPSTREAM / "scripts/remote/build_deform360_automatic_episode_twin.py"
    baseline = builders_module._V7_PHYSICAL_ISOLATED_RUNPY_COMMAND(
        PINNED_PYTHON,
        script,
        import_roots=(UPSTREAM / "src", DEFORM360),
        arguments=_arguments(root, episode_id=episode_id),
    )
    frozen = builders_module.physical._ISOLATED_RUNPY_BOOTSTRAP
    _require(baseline.count(frozen) == 1, "frozen child bootstrap changed")
    return [
        builders_module._V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP
        if value == frozen
        else value
        for value in baseline
    ]


def _environment() -> dict[str, str]:
    # Match the original successful admission replay's ``env -i`` boundary.
    # In particular, do not inherit Python, CUDA, model-cache, or scheduler
    # variables from the SSH session that launches this operator.
    return {
        "HOME": "/home/florianpfaff",
        "USER": "florianpfaff",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
    }


def _run_child(root: Path, *, episode_id: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        _command(root, episode_id=episode_id),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=CHILD_CWD,
        env=_environment(),
        timeout=CHILD_TIMEOUT_SECONDS,
    )


def _require_exact_replay_tree(root: Path, *, reports_written: bool) -> None:
    root_files = set(SUCCESS_OUTPUT_NAMES) | {"stdout.log", "stderr.log"}
    if reports_written:
        root_files |= {REPORT_NAME, CODE_BINDING_NAME}
    expected_root = root_files | {"cross-auth"}
    observed_root = {entry.name for entry in os.scandir(root)}
    _require(observed_root == expected_root, "admission replay root entries changed")
    for name in sorted(root_files):
        observed = os.lstat(root / name)
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode)
            and observed.st_nlink == 1,
            f"admission replay root entry is not a regular file: {name}",
        )
    cross = root / "cross-auth"
    cross_state = os.lstat(cross)
    _require(
        stat.S_ISDIR(cross_state.st_mode) and not stat.S_ISLNK(cross_state.st_mode),
        "cross-authorization replay entry is not a directory",
    )
    expected_cross = {"stdout.log", "stderr.log"}
    observed_cross = {entry.name for entry in os.scandir(cross)}
    _require(
        observed_cross == expected_cross,
        "cross-authorization replay entries changed",
    )
    for name in sorted(expected_cross):
        observed = os.lstat(cross / name)
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode)
            and observed.st_nlink == 1,
            f"cross-authorization entry is not a regular file: {name}",
        )


def _validate_cross_authorization_rejection(
    result: subprocess.CompletedProcess[bytes],
    root: Path,
    numerical_names: Sequence[str],
) -> None:
    numerical_outputs = [name for name in numerical_names if (root / name).exists()]
    _require(
        result.returncode == 1
        and CROSS_AUTHORIZATION_REJECTION_MARKER.encode("utf-8") in result.stderr
        and not numerical_outputs,
        "wrong-case admission was not rejected by the exact authorization gate",
    )


def _validate_successful_replay(outputs: Mapping[str, Path]) -> dict[str, Any]:
    builders_module, _ = _source_modules()
    summary = builders_module.physical._load_json(outputs["twin_summary.json"])
    boundary = summary.get("information_boundary")
    metrics = summary.get("state_metrics")
    _require(
        summary.get("schema_version") == 1
        and summary.get("artifact_kind") == "Deform360AutomaticEpisodeTwin"
        and summary.get("protocol_id")
        == builders_module.V8_EXTERNAL_ADMISSION_PROTOCOL_ID
        and summary.get("protocol_config_sha256")
        == builders_module.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
        and summary.get("object_id") == OBJECT_ID
        and int(summary.get("episode_id", -1)) == EPISODE_ID
        and summary.get("phase") == "calibration"
        and summary.get("passed") is True
        and summary.get("result_sha256")
        == builders_module.physical._upstream_result_sha256(summary),
        "successful replay summary identity changed",
    )
    _require(
        boundary == SUCCESS_INFORMATION_BOUNDARY,
        "successful replay crossed its observation boundary",
    )
    _require(
        isinstance(metrics, Mapping)
        and metrics.get("passed") is True
        and metrics.get("finite") is True,
        "successful replay did not pass state admission",
    )
    _require(
        isinstance(summary.get("graph"), Mapping)
        and isinstance(summary.get("capacity_diagnostic"), Mapping)
        and isinstance(summary.get("prediction_input_validation"), Mapping),
        "successful replay diagnostics are incomplete",
    )
    expected_outputs = {
        "episode_graph": builders_module.physical.sha256_file(
            outputs["episode_graph.npz"]
        ),
        "simulator_final_data": builders_module.physical.sha256_file(
            outputs["simulator_final_data.pkl"]
        ),
        "state_artifact": builders_module.physical.sha256_file(
            outputs["state_artifact.npz"]
        ),
    }
    _require(
        summary.get("input_sha256", {}).get("episode_final_data") == SOURCE_INPUT_SHA256
        and summary.get("output_sha256") == expected_outputs,
        "successful replay input/output binding changed",
    )
    return summary


def _seal_tree(root: Path, *, seal_root: bool) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in files:
            os.chmod(current_path / name, 0o400, follow_symlinks=False)
        for name in directories:
            os.chmod(current_path / name, 0o500, follow_symlinks=False)
    if seal_root:
        os.chmod(root, 0o500, follow_symlinks=False)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _streaming_file_record(path: Path) -> dict[str, Any]:
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1,
        f"replay inventory file is invalid: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size)
            == (opened.st_dev, opened.st_ino, opened.st_size),
            f"replay inventory file changed while opening: {path}",
        )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
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
        f"replay inventory file changed while hashing: {path}",
    )
    return {"size_bytes": before.st_size, "sha256": digest.hexdigest()}


def _replay_content_inventory(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(directories)
        for name in directories:
            child = current_path / name
            observed = os.lstat(child)
            _require(
                stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
                f"replay inventory directory is invalid: {child}",
            )
            rows.append(
                {"path": child.relative_to(root).as_posix(), "type": "directory"}
            )
        for name in sorted(files):
            child = current_path / name
            rows.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "type": "file",
                    **_streaming_file_record(child),
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "entry_count": len(rows),
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
    }


def _require_sealed_replay_tree(root: Path) -> None:
    root_state = os.lstat(root)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and stat.S_IMODE(root_state.st_mode) == 0o500
        and root.resolve() == root,
        "published replay root is not sealed",
    )
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in directories:
            observed = os.lstat(current_path / name)
            _require(
                stat.S_ISDIR(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o500,
                "published replay directory is not sealed",
            )
        for name in files:
            observed = os.lstat(current_path / name)
            _require(
                stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o400
                and observed.st_nlink == 1,
                "published replay file is not sealed",
            )


def main() -> int:
    _require_isolated_launch()
    preflight = _bootstrap_source_modules()
    builders_module, protocol_module = _source_modules()
    _require(socket.gethostname() == "workstation2", "replay must run on workstation2")
    _require(
        protocol_module.PROTOCOL_ID == "deform360-held-online-belief-v8.1",
        "protocol identity changed",
    )
    _require(protocol_module.EXECUTION_ATTEMPT == 5, "execution attempt changed")
    _require(not os.path.lexists(ROOT), "admission replay root already exists")
    stage = ROOT.parent / f".{ROOT.name}.stage-{os.getpid()}"
    _require(not os.path.lexists(stage), "admission replay stage already exists")
    source_binding = _source_binding(preflight)
    qualification_lineage = _qualification_lineage(source_binding)
    source_payload = _read_regular(
        SOURCE_INPUT, role="prediction-only input", mode=0o400
    )
    _require(
        len(source_payload) == SOURCE_INPUT_SIZE_BYTES
        and hashlib.sha256(source_payload).hexdigest() == SOURCE_INPUT_SHA256,
        "prediction-only input changed",
    )
    stage.mkdir(mode=0o700)
    cross = stage / "cross-auth"
    cross.mkdir(mode=0o700)
    try:
        success = _run_child(stage, episode_id=EPISODE_ID)
        _write_new(stage / "stdout.log", success.stdout)
        _write_new(stage / "stderr.log", success.stderr)
        _require(success.returncode == 0, "exact-case admission replay failed")
        output_paths = {name: stage / name for name in sorted(SUCCESS_OUTPUT_NAMES)}
        _require(
            all(path.is_file() for path in output_paths.values()),
            "replay output is absent",
        )
        validated = _validate_successful_replay(output_paths)

        rejected = _run_child(cross, episode_id=WRONG_EPISODE_ID)
        _write_new(cross / "stdout.log", rejected.stdout)
        _write_new(cross / "stderr.log", rejected.stderr)
        numerical_names = tuple(output_paths)
        _validate_cross_authorization_rejection(rejected, cross, numerical_names)
        _require_exact_replay_tree(stage, reports_written=False)

        final_outputs = {
            name: _file_record(
                path,
                role=f"replay output {name}",
                reported_path=ROOT / name,
            )
            for name, path in output_paths.items()
        }
        report = _artifact(
            {
                "schema_version": 1,
                "artifact_kind": "Deform360HeldV81ExternalAdmissionMetadataOnlyReplay",
                "protocol_id": protocol_module.PROTOCOL_ID,
                "execution_attempt": protocol_module.EXECUTION_ATTEMPT,
                "case_name": CASE_NAME,
                "role": "calibration",
                "development_replay_only": True,
                "formal_outcome_evidence": False,
                "resource_lifecycle_qualification": qualification_lineage,
                "source_evidence": {
                    "archived_attempt": 2,
                    "prediction_only_input": {
                        "path": str(SOURCE_INPUT),
                        "sha256": SOURCE_INPUT_SHA256,
                        "size_bytes": SOURCE_INPUT_SIZE_BYTES,
                    },
                    "future_object_observation_used": False,
                    "source_used_for_numerical_replay": "prediction_only_input_only",
                },
                "admission": {
                    "protocol_id": builders_module.V8_EXTERNAL_ADMISSION_PROTOCOL_ID,
                    "contract_sha256": (
                        builders_module.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
                    ),
                    "exact_case_only": True,
                    "target_access": False,
                },
                "successful_replay": {
                    "exit_code": success.returncode,
                    "hook_restoration_guard_completed": True,
                    "summary_result_sha256": validated["result_sha256"],
                    "validator_result_sha256": validated["result_sha256"],
                    "graph": validated["graph"],
                    "capacity_diagnostic": validated["capacity_diagnostic"],
                    "prediction_input_validation": validated[
                        "prediction_input_validation"
                    ],
                    "state_metrics": validated["state_metrics"],
                    "information_boundary": validated["information_boundary"],
                    "outputs": final_outputs,
                    "stdout_log": _file_record(
                        stage / "stdout.log",
                        role="successful replay stdout",
                        reported_path=ROOT / "stdout.log",
                    ),
                    "stderr_log": _file_record(
                        stage / "stderr.log",
                        role="successful replay stderr",
                        reported_path=ROOT / "stderr.log",
                    ),
                },
                "cross_authorization_rejection": {
                    "attempted_case_name": f"{OBJECT_ID}-ep{WRONG_EPISODE_ID:04d}",
                    "exit_code": rejected.returncode,
                    "rejected": True,
                    "numerical_output_count": 0,
                    "stderr_marker": CROSS_AUTHORIZATION_REJECTION_MARKER,
                    "stderr_marker_present": True,
                    "stdout_log": _file_record(
                        cross / "stdout.log",
                        role="cross-authorization stdout",
                        reported_path=ROOT / "cross-auth/stdout.log",
                    ),
                    "stderr_log": _file_record(
                        cross / "stderr.log",
                        role="cross-authorization stderr",
                        reported_path=ROOT / "cross-auth/stderr.log",
                    ),
                },
                "information_boundary": {
                    "official_target_created": False,
                    "official_target_read": False,
                    "query_created": False,
                    "query_read": False,
                    "score_created": False,
                    "score_read": False,
                    "outcome_created": False,
                    "outcome_read": False,
                    "confirmation_accessed": False,
                },
                "local_source_at_replay": source_binding,
            }
        )
        report_path = stage / REPORT_NAME
        _write_new(report_path, _json_bytes(report))
        report_record = _file_record(
            report_path,
            role="admission replay report",
            reported_path=ROOT / REPORT_NAME,
        )
        code_binding = _artifact(
            {
                "schema_version": 1,
                "artifact_kind": "Deform360HeldV81ExternalAdmissionReplayCodeBinding",
                "protocol_id": protocol_module.PROTOCOL_ID,
                "execution_attempt": protocol_module.EXECUTION_ATTEMPT,
                "admission_contract_sha256": (
                    builders_module.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
                ),
                "formal_outcome_evidence": False,
                "target_query_score_or_outcome_accessed": False,
                "local_worktree_at_replay": source_binding,
                "replay_report": {
                    **report_record,
                    "artifact_sha256": report["artifact_sha256"],
                },
            }
        )
        code_binding_path = stage / CODE_BINDING_NAME
        _write_new(code_binding_path, _json_bytes(code_binding))
        code_binding_record = _file_record(
            code_binding_path,
            role="admission replay code binding",
            reported_path=ROOT / CODE_BINDING_NAME,
        )
        _require_exact_replay_tree(stage, reports_written=True)
        expected_inventory = _replay_content_inventory(stage)
        _seal_tree(stage, seal_root=False)
        _rename_noreplace(stage, ROOT)
        os.chmod(ROOT, 0o500, follow_symlinks=False)
        _require_exact_replay_tree(ROOT, reports_written=True)
        _require_sealed_replay_tree(ROOT)
        _require(
            _replay_content_inventory(ROOT) == expected_inventory,
            "published replay content inventory changed",
        )
        _fsync_directory(ROOT.parent)
        result = {
            "root": str(ROOT),
            "report_file_sha256": report_record["sha256"],
            "report_artifact_sha256": report["artifact_sha256"],
            "code_binding_file_sha256": code_binding_record["sha256"],
            "code_binding_artifact_sha256": code_binding["artifact_sha256"],
            "source_head": source_binding["git_head"],
            "adapter_source_sha256": source_binding["adapter_source_sha256"],
            "protocol_source_sha256": source_binding["protocol_source_sha256"],
            "exact_child_bootstrap_sha256": source_binding[
                "exact_child_bootstrap_sha256"
            ],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BaseException:
        if stage.exists():
            for current, directories, files in os.walk(stage, topdown=False):
                current_path = Path(current)
                for name in files:
                    os.chmod(current_path / name, 0o600, follow_symlinks=False)
                for name in directories:
                    os.chmod(current_path / name, 0o700, follow_symlinks=False)
            os.chmod(stage, 0o700, follow_symlinks=False)
            shutil.rmtree(stage)
        raise


if __name__ == "__main__":
    sys.exit(main())
