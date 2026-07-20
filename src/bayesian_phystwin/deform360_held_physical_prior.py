"""Prediction-only Deform360 physical priors for the held protocol.

The numerical twin and Warp rollout are deliberately reused from the frozen
``reusable-trust-fresh-code/Bayesian-PhysTwin`` tree.  This module adds the
missing security boundary around that code:

* the case must be in the exact calibration or confirmation whitelist;
* the only object input is a validated :class:`Deform360HeldFrameZeroBundle`;
* the aligned realized end-effector kinematics are selected without reading
  object motion;
* the upstream source files, official PhysTwin revision, and Warp config are
  checked before a subprocess is started; and
* driven and zero-action trajectories are converted into the frozen
  graph-support prediction and hashed before any outcome operation.

No function in this module accepts an outcome path.  The resulting four files
are inputs to :func:`bayesian_phystwin.deform360_held_protocol.create_physical_prior_seal`.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from pathlib import Path
import pickle
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_held_protocol import (
    FRAME_COUNT,
    PROTOCOL_ID,
    create_physical_prior_seal,
    held_artifact_sha256,
    held_contract_sha256,
    load_held_protocol_lock,
    validate_frame_zero_bundle_manifest,
)
from .deform360_robot_kinematics import (
    ROBOT_KINEMATICS_WINDOW_CONTRACT,
    ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
    ROBOT_KINEMATICS_WINDOW_POLICY_ID,
    Deform360RobotKinematics,
    artifact_sha256 as robot_artifact_sha256,
    load_robot_kinematics_archive,
    robot_kinematics_array_records,
    select_robot_kinematics_window,
    slice_robot_kinematics,
    validate_selected_robot_kinematics_bundle,
)


ARTIFACT_KIND = "Deform360HeldPhysicalPrediction"
PREDICTION_INPUT_KIND = "Deform360PredictionOnlyInput"
SCHEMA_VERSION = 1

OFFICIAL_PHYSTWIN_REVISION = "2b6630528141b9cba5a7677c8b88b2129b4a8390"
OFFICIAL_REAL_CONFIG_SHA256 = (
    "a40a5ec2f5c978c1290810f20ed56db7cab99dc0c227adfe6b7434dfc95ead48"
)
LENGTH_SCALE_M = 0.12
ACTION_RESPONSE = 0.9
AUTONOMOUS_DRIFT_RESPONSE = 0.0
CANONICAL_NODE_COUNT = 1024
MINIMUM_NODE_COUNT = 128
HELD_PYCACHE_PREFIX = "/nonexistent/bpt-held-v7-pycache"

HELD_PYTHON_RUNTIME = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-"
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
HELD_PYTHON_RUNTIME_MANIFEST = Path(f"{HELD_PYTHON_RUNTIME}.tree-manifest.json")
HELD_PYTHON_RUNTIME_MANIFEST_KIND = "Deform360HeldPythonRuntimeTreeManifestV1"
HELD_PYTHON_RUNTIME_SYMLINKS = {
    "bin/python": "/usr/bin/python3",
    "bin/python3": "python",
    "bin/python3.12": "python",
}

_GIT_EXECUTABLE = Path("/usr/bin/git")
_PYTHON_IMPORTABLE_SUFFIXES = (".py", ".pyc", ".pyo", ".pyd", ".so")
_QQTT_IMPORTED_PROVENANCE = {
    "qqtt": "qqtt/__init__.py",
    "qqtt.engine.trainer_warp": "qqtt/engine/trainer_warp.py",
    "qqtt.model.diff_simulator.spring_mass_warp": (
        "qqtt/model/diff_simulator/spring_mass_warp.py"
    ),
    "qqtt.utils": "qqtt/utils/__init__.py",
}

# The interpreter starts in isolated mode before these roots are admitted.  The
# numerical script may terminate through ``SystemExit(0)``; retain control long
# enough to prove that its critical official-PhysTwin imports came from the
# validated checkout rather than a working directory or startup hook.
_ISOLATED_RUNPY_BOOTSTRAP = """\
import os
import runpy
import sys

root_count = int(sys.argv[1])
roots = sys.argv[2 : 2 + root_count]
script_index = 2 + root_count
script = sys.argv[script_index]
provenance_root = sys.argv[script_index + 1]
script_arguments = sys.argv[script_index + 2 :]
if not roots or any(not os.path.isabs(root) for root in roots):
    raise RuntimeError("isolated bootstrap received a non-absolute import root")
if not os.path.isabs(script):
    raise RuntimeError("isolated bootstrap received a non-absolute script")
sys.path[:0] = roots
sys.argv = [script, *script_arguments]
exit_status = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as error:
    if error.code is None:
        exit_status = 0
    elif isinstance(error.code, int):
        exit_status = error.code
    else:
        print(error.code, file=sys.stderr)
        exit_status = 1
if exit_status:
    raise SystemExit(exit_status)
if provenance_root:
    expected = {
        "qqtt": "qqtt/__init__.py",
        "qqtt.engine.trainer_warp": "qqtt/engine/trainer_warp.py",
        "qqtt.model.diff_simulator.spring_mass_warp":
            "qqtt/model/diff_simulator/spring_mass_warp.py",
        "qqtt.utils": "qqtt/utils/__init__.py",
    }
    for module_name, relative_path in expected.items():
        module = sys.modules.get(module_name)
        observed = getattr(module, "__file__", None)
        locked = os.path.join(provenance_root, *relative_path.split("/"))
        if (
            not isinstance(observed, str)
            or os.path.realpath(observed) != os.path.realpath(locked)
        ):
            raise RuntimeError(
                f"official PhysTwin import provenance changed: {module_name}"
            )
"""

PHYSICAL_MODE_WARP_TWIN = "warp_twin"
PHYSICAL_MODE_PERSISTENCE_FALLBACK = "persistence_fallback"
PERSISTENCE_FALLBACK_REASON = "automatic_twin_source_admission_failed"
AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE = 2
AUTOMATIC_TWIN_PROTOCOL_ID = "deform360-dense-reusable-panel-v1"
AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256 = (
    "1a78b8d74679ebf65768cc5078b34d034a2fcac55f7e0c0a00e50e1967a1c9bd"
)
AUTOMATIC_TWIN_ADMISSION_THRESHOLDS = {
    "maximum_supported_distance_m": 0.02,
    "minimum_observed_target_fraction": 0.95,
    "minimum_effective_target_reliability": 0.70,
    "maximum_p99_relative_edge_strain": 0.50,
    "maximum_bridge_relative_edge_strain": 0.50,
    "maximum_contact_anchor_error_m": 0.015,
}

WARP_DYNAMICS = {
    "init_spring_y": 10_000.0,
    "drag_damping": 10.0,
    "dashpot_damping": 100.0,
    "controller_radius_m": 0.03,
    "controller_max_neighbours": 1,
    "canonical_controller_patch_size": 16,
    "support_dynamics": "official-ground",
}

UPSTREAM_FILE_SHA256 = {
    "scripts/remote/build_deform360_automatic_episode_twin.py": (
        "dd43bfeaa0ddb53252e3b2d9c907c147379b2cce6b4c5d5dfa14f310fdacfa9a"
    ),
    "scripts/remote/run_deform360_official_phystwin_smoke.py": (
        "e7bf6a6c06e074ac3cdefe259c1cf5eecf8cd905dae1b710a81107ab166ca535"
    ),
    "src/causal4d_public/deform360_reusable_graph.py": (
        "97b93e32c5009f5783b2f36be7e03d4acda33f0608c9694797e8e5c72d3dd8a5"
    ),
    "src/causal4d_public/deform360_partial_graph_state.py": (
        "81536d81ce4cfd0e61074d2f4096b3160624b6afa2e1dda1d0dab16c113192a3"
    ),
    "src/causal4d_public/deform360_dense_reusable_panel.py": (
        "0861831b9ab3cf6d64833efe533073f4f444f2315c04057377f243efffd8b17e"
    ),
    "src/causal4d_public/deform360_action_support.py": (
        "132283722400ac102ec84e9b7d21974edcdac0ff750168d70860cd89c8446783"
    ),
    "src/causal4d_public/deform360_contact_conditioned_action.py": (
        "1d4e2bbd4389d8d7055d0803f3feda3ea540d45123e0aa3f646bccf2cfa6c57e"
    ),
    "src/causal4d_public/deform360_dense_source.py": (
        "6c9ffa0043302079acf303f23af9e9ebb895f0aa8cf03930effe8936a879bb29"
    ),
    "src/bayesian_phystwin/phystwin_graph.py": (
        "f6f1ef8d3a1fb95fc069a550ae7db12d6b32efe80582f479efb411452062b6fb"
    ),
    "configs/causal4d_public/deform360_dense_reusable_panel_v1.json": (
        "8a90705dd38c6c90b042ed8f450e2bc7e3cffc54b965765b004d0385999d40ea"
    ),
    "configs/causal4d_public/deform360_independent_source_split_v1.json": (
        "c150b2c8ea3947fe2ffe359c5da45d321b5086cd67141c2da9f912aac154ff4a"
    ),
}

UPSTREAM_LOCK_BINDING_BY_PATH = {
    "scripts/remote/build_deform360_automatic_episode_twin.py": (
        "upstream_automatic_twin_builder"
    ),
    "scripts/remote/run_deform360_official_phystwin_smoke.py": (
        "upstream_official_phystwin_smoke"
    ),
    "src/causal4d_public/deform360_reusable_graph.py": (
        "upstream_reusable_graph_source"
    ),
    "src/causal4d_public/deform360_partial_graph_state.py": (
        "upstream_partial_graph_state_source"
    ),
    "src/causal4d_public/deform360_dense_reusable_panel.py": (
        "upstream_dense_reusable_panel_source"
    ),
    "src/causal4d_public/deform360_action_support.py": (
        "upstream_action_support_source"
    ),
    "src/causal4d_public/deform360_contact_conditioned_action.py": (
        "upstream_contact_conditioned_action_source"
    ),
    "src/causal4d_public/deform360_dense_source.py": "upstream_dense_source",
    "src/bayesian_phystwin/phystwin_graph.py": "upstream_phystwin_graph_source",
    "configs/causal4d_public/deform360_dense_reusable_panel_v1.json": (
        "upstream_dense_reusable_panel_config"
    ),
    "configs/causal4d_public/deform360_independent_source_split_v1.json": (
        "upstream_independent_source_split_config"
    ),
}

UPSTREAM_RUNTIME_BUNDLE_CONTRACT = {
    "artifact_kind": "Deform360HeldUpstreamRuntimeBundleV1",
    "files": [
        {"path": path, "sha256": UPSTREAM_FILE_SHA256[path]}
        for path in sorted(UPSTREAM_FILE_SHA256)
    ],
}

HELD_PHYSICAL_NUMERIC_CONTRACT = {
    "contract_id": "deform360-held-physical-prior-v3",
    "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
    "official_real_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
    "length_scale_m": LENGTH_SCALE_M,
    "action_response": ACTION_RESPONSE,
    "autonomous_drift_response": AUTONOMOUS_DRIFT_RESPONSE,
    "canonical_node_count": CANONICAL_NODE_COUNT,
    "minimum_node_count": MINIMUM_NODE_COUNT,
    "automatic_twin_admission_thresholds": AUTOMATIC_TWIN_ADMISSION_THRESHOLDS,
    "persistence_fallback": {
        "physical_mode": PHYSICAL_MODE_PERSISTENCE_FALLBACK,
        "reason": PERSISTENCE_FALLBACK_REASON,
        "required_automatic_twin_exit_code": AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE,
        "requires_valid_checksummed_inadmissible_twin": True,
        "warp_attempted": False,
        "prediction_equals_persistence": True,
        "driven_equals_persistence": True,
        "zero_action_equals_persistence": True,
        "action_support": "all_zero",
        "physical_admitted": False,
    },
    "warp_dynamics": WARP_DYNAMICS,
    "robot_kinematics": {
        "policy_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
        "contract_sha256": ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
        "trajectory_semantics": ROBOT_KINEMATICS_WINDOW_CONTRACT[
            "trajectory_semantics"
        ],
        "controller_source": "T_worlds absolute end-effector pose and openings",
    },
    "upstream_file_sha256": UPSTREAM_FILE_SHA256,
}

FRAME_ZERO_ARRAYS = frozenset(
    {
        "frame_indices",
        "camera_names",
        "rgb_frame0",
        "mask_frame0",
        "depth_frame0_m",
        "depth_valid_frame0",
        "intrinsics",
        "camera_to_world",
        "projection_world_to_pixel",
        "object_points_world_m",
        "object_colors_rgb",
        "object_color_support_count",
        "visual_hull_points_world_m",
    }
)

PHYSICAL_ARCHIVE_ARRAYS = frozenset(
    {
        "prediction_m",
        "persistence_m",
        "driven_readout_m",
        "zero_action_readout_m",
        "action_support",
        "frame_zero_points_m",
    }
)

_FINGER_BASE_LEFT = np.array([-0.04246242, 0.0835, 0.0097])
_FINGER_BASE_RIGHT = np.array([0.04246242, 0.0835, 0.0107])
_TAXEL_X_M = 0.007
_TAXEL_Y0_M = -0.056
_TAXEL_Y_STEP_M = -0.002
_TAXEL_Z_PITCH_M = 0.025 / 12.0
_TAXEL_ROWS = 12
_TAXEL_COLS = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash an array exactly like the frozen independent-source code."""

    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination.resolve()


def _bound_file(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(
        source.is_file() and not source.is_symlink(), f"missing regular file: {source}"
    )
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _validate_bound_file(
    record: Mapping[str, Any], *, label: str, allow_metadata: bool = False
) -> Path:
    _require(isinstance(record, Mapping), f"{label} binding is missing")
    path = record.get("path")
    _require(isinstance(path, str) and bool(path), f"{label} path is missing")
    observed = _bound_file(path)
    bound_record = {key: record.get(key) for key in observed}
    _require(observed == bound_record, f"{label} binding changed")
    if not allow_metadata:
        _require(set(record) == set(observed), f"{label} binding has unexpected fields")
    return Path(observed["path"])


class _LoggedCommandError(RuntimeError):
    """A subprocess failure with its exact exit status and elapsed runtime."""

    def __init__(
        self, message: str, *, returncode: int, elapsed_seconds: float
    ) -> None:
        super().__init__(message)
        self.returncode = int(returncode)
        self.elapsed_seconds = float(elapsed_seconds)


def _sorted_pip_freeze_sha256(stdout: bytes) -> str:
    lines = stdout.splitlines()
    canonical = b"\n".join(sorted(lines)) + b"\n"
    return hashlib.sha256(canonical).hexdigest()


def _stable_file_identity(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _open_regular_nofollow(path: Path) -> tuple[int, os.stat_result]:
    _require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is unavailable")
    try:
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise ValueError(f"cannot open locked regular file: {path}") from error
    try:
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"locked path is not a file: {path}")
        _require(
            _stable_file_identity(opened) == _stable_file_identity(before),
            f"locked file changed while opening: {path}",
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, opened


def _finish_regular_nofollow(
    path: Path,
    descriptor: int,
    opened: os.stat_result,
) -> None:
    try:
        final_fd = os.fstat(descriptor)
        final_path = os.lstat(path)
        _require(
            _stable_file_identity(final_fd) == _stable_file_identity(opened),
            f"locked file changed while reading: {path}",
        )
        _require(
            _stable_file_identity(final_path) == _stable_file_identity(opened),
            f"locked file was replaced while reading: {path}",
        )
    finally:
        os.close(descriptor)


def _read_regular_nofollow(path: Path) -> bytes:
    descriptor, opened = _open_regular_nofollow(path)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        _finish_regular_nofollow(path, descriptor, opened)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return b"".join(chunks)


def _sha256_regular_nofollow(
    path: Path,
    *,
    expected: os.stat_result,
) -> str:
    descriptor, opened = _open_regular_nofollow(path)
    try:
        _require(
            _stable_file_identity(opened) == _stable_file_identity(expected),
            f"runtime entry changed before hashing: {path}",
        )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        _finish_regular_nofollow(path, descriptor, opened)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return digest.hexdigest()


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate runtime-manifest key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite runtime-manifest value: {value}")


def _runtime_entry_paths(root: Path) -> list[str]:
    paths: list[str] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as scan:
                children = sorted(scan, key=lambda item: os.fsencode(item.name))
        except OSError as error:
            raise ValueError(f"cannot scan held Python runtime: {directory}") from error
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
                    f"unsupported held Python runtime entry: {relative}",
                )

    visit(root)
    return sorted(paths, key=os.fsencode)


def _validate_runtime_manifest_path(path: str) -> None:
    _require(path != "", "runtime-manifest entry path is empty")
    _require("\x00" not in path, "runtime-manifest entry path contains NUL")
    _require("\\" not in path, "runtime-manifest entry path is not POSIX")
    _require(not path.startswith("/"), "runtime-manifest entry path is absolute")
    parts = path.split("/")
    _require(
        all(part not in {"", ".", ".."} for part in parts),
        "runtime-manifest entry path is not canonical",
    )


def _validate_held_python_runtime_tree(
    immutable_bindings: Mapping[str, Any],
) -> dict[str, str]:
    root = _canonical_directory(
        HELD_PYTHON_RUNTIME,
        label="held Python runtime root",
    )
    root_stat = os.lstat(root)
    _require(
        stat.S_IMODE(root_stat.st_mode) == 0o555,
        "held Python runtime root mode differs from 0555",
    )
    manifest_path = HELD_PYTHON_RUNTIME_MANIFEST
    expected_manifest_path = root.parent / f"{root.name}.tree-manifest.json"
    _require(
        manifest_path == expected_manifest_path,
        "held Python runtime manifest is not the exact sibling path",
    )
    _require(
        manifest_path.parent.resolve(strict=True) == manifest_path.parent,
        "held Python runtime manifest has aliased ancestry",
    )
    try:
        manifest_stat = os.lstat(manifest_path)
    except FileNotFoundError as error:
        raise ValueError("held Python runtime manifest is missing") from error
    _require(
        stat.S_ISREG(manifest_stat.st_mode),
        "held Python runtime manifest is not a regular file",
    )
    _require(
        stat.S_IMODE(manifest_stat.st_mode) == 0o400,
        "held Python runtime manifest mode differs from 0400",
    )
    raw_manifest = _read_regular_nofollow(manifest_path)
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    locked_manifest_sha256 = immutable_bindings.get("held_frozen_runtime_manifest")
    _require(
        isinstance(locked_manifest_sha256, str)
        and len(locked_manifest_sha256) == 64
        and all(
            character in "0123456789abcdef" for character in locked_manifest_sha256
        ),
        "held Python runtime manifest binding is invalid",
    )
    _require(
        manifest_sha256 == locked_manifest_sha256,
        "held Python runtime manifest differs from the immutable lock",
    )
    try:
        manifest = json.loads(
            raw_manifest,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("held Python runtime manifest is invalid JSON") from error
    _require(isinstance(manifest, Mapping), "runtime manifest is not an object")
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
        "held Python runtime manifest fields changed",
    )
    canonical_manifest = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )
    _require(
        raw_manifest == canonical_manifest,
        "held Python runtime manifest is not canonical JSON",
    )
    _require(
        manifest["artifact_kind"] == HELD_PYTHON_RUNTIME_MANIFEST_KIND,
        "held Python runtime manifest kind changed",
    )
    _require(
        manifest["root_path"] == str(root),
        "held Python runtime manifest root changed",
    )
    locked_freeze = immutable_bindings.get("python_pip_freeze_sorted")
    _require(
        manifest["python_pip_freeze_sorted_sha256"] == locked_freeze,
        "held Python runtime manifest freeze identity changed",
    )
    entries = manifest["entries"]
    _require(isinstance(entries, list), "runtime-manifest entries are not a list")
    entry_paths: list[str] = []
    observed_counts = {"directory": 0, "file": 0, "symlink": 0}
    total_regular_file_bytes = 0
    observed_symlinks: dict[str, str] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), "runtime-manifest entry is not an object")
        path = entry.get("path")
        entry_type = entry.get("type")
        mode = entry.get("mode")
        _require(isinstance(path, str), "runtime-manifest entry path is invalid")
        _validate_runtime_manifest_path(path)
        _require(
            isinstance(mode, str)
            and len(mode) == 4
            and all(character in "01234567" for character in mode),
            "runtime-manifest entry mode is invalid",
        )
        _require(
            entry_type in observed_counts,
            "runtime-manifest entry type is invalid",
        )
        if entry_type == "directory":
            _require(
                set(entry) == {"path", "mode", "type"},
                "runtime-manifest directory fields changed",
            )
        elif entry_type == "file":
            _require(
                set(entry) == {"path", "mode", "type", "size", "sha256"},
                "runtime-manifest file fields changed",
            )
            _require(
                isinstance(entry["size"], int)
                and not isinstance(entry["size"], bool)
                and entry["size"] >= 0,
                "runtime-manifest file size is invalid",
            )
            _require(
                isinstance(entry["sha256"], str)
                and len(entry["sha256"]) == 64
                and all(
                    character in "0123456789abcdef" for character in entry["sha256"]
                ),
                "runtime-manifest file checksum is invalid",
            )
        else:
            _require(
                set(entry) == {"path", "mode", "type", "target"},
                "runtime-manifest symlink fields changed",
            )
            _require(
                isinstance(entry["target"], str) and bool(entry["target"]),
                "runtime-manifest symlink target is invalid",
            )
        entry_paths.append(path)
    _require(
        entry_paths == sorted(entry_paths, key=os.fsencode)
        and len(entry_paths) == len(set(entry_paths)),
        "runtime-manifest entry paths are unsorted or duplicated",
    )
    _require(
        _runtime_entry_paths(root) == entry_paths,
        "held Python runtime paths differ from the manifest",
    )
    for entry in entries:
        path = str(entry["path"])
        runtime_path = root.joinpath(*path.split("/"))
        observed = os.lstat(runtime_path)
        observed_mode = stat.S_IMODE(observed.st_mode)
        _require(
            format(observed_mode, "04o") == entry["mode"],
            f"held Python runtime mode changed: {path}",
        )
        entry_type = str(entry["type"])
        if entry_type == "directory":
            _require(
                stat.S_ISDIR(observed.st_mode),
                f"held Python runtime directory changed type: {path}",
            )
            _require(
                observed_mode & 0o222 == 0,
                f"held Python runtime directory is writable: {path}",
            )
        elif entry_type == "file":
            _require(
                stat.S_ISREG(observed.st_mode),
                f"held Python runtime file changed type: {path}",
            )
            _require(
                observed_mode & 0o222 == 0,
                f"held Python runtime file is writable: {path}",
            )
            _require(
                observed.st_size == entry["size"],
                f"held Python runtime file size changed: {path}",
            )
            _require(
                _sha256_regular_nofollow(runtime_path, expected=observed)
                == entry["sha256"],
                f"held Python runtime file checksum changed: {path}",
            )
            total_regular_file_bytes += observed.st_size
        else:
            _require(
                stat.S_ISLNK(observed.st_mode),
                f"held Python runtime symlink changed type: {path}",
            )
            target = os.readlink(runtime_path)
            _require(
                target == entry["target"],
                f"held Python runtime symlink target changed: {path}",
            )
            observed_symlinks[path] = target
        observed_counts[entry_type] += 1
    _require(
        observed_symlinks == HELD_PYTHON_RUNTIME_SYMLINKS,
        "held Python runtime symlink policy changed",
    )
    counts = manifest["entry_counts"]
    _require(
        isinstance(counts, Mapping)
        and set(counts) == set(observed_counts)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts.values()
        )
        and dict(counts) == observed_counts,
        "held Python runtime entry counts changed",
    )
    _require(
        isinstance(manifest["total_regular_file_bytes"], int)
        and not isinstance(manifest["total_regular_file_bytes"], bool)
        and manifest["total_regular_file_bytes"] == total_regular_file_bytes,
        "held Python runtime byte count changed",
    )
    canonical_entries = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    tree_sha256 = hashlib.sha256(canonical_entries).hexdigest()
    _require(
        tree_sha256 == manifest["tree_sha256"],
        "held Python runtime tree checksum changed",
    )
    return {
        "runtime_root": str(root),
        "runtime_manifest_path": str(manifest_path),
        "runtime_manifest_sha256": manifest_sha256,
        "runtime_tree_sha256": tree_sha256,
    }


def _canonical_directory(path: str | Path, *, label: str) -> Path:
    supplied = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    try:
        observed = os.lstat(supplied)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {supplied}") from error
    _require(stat.S_ISDIR(observed.st_mode), f"{label} is not a directory")
    _require(
        supplied.resolve(strict=True) == supplied,
        f"{label} is a symlink or has aliased ancestry",
    )
    return supplied


def _git_environment(root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("GIT_CONFIG_KEY_") or key.startswith("GIT_CONFIG_VALUE_"):
            environment.pop(key)
    for key in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_COUNT",
        "GIT_DIR",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CEILING_DIRECTORIES": os.fspath(root.parent),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _run_git(root: Path, arguments: Sequence[str]) -> bytes:
    _require(
        _GIT_EXECUTABLE.is_file() and not _GIT_EXECUTABLE.is_symlink(),
        "pinned Git executable is unavailable",
    )
    completed = subprocess.run(
        [
            os.fspath(_GIT_EXECUTABLE),
            "--no-replace-objects",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "core.excludesFile=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.untrackedCache=false",
            *arguments,
        ],
        cwd=root,
        env=_git_environment(root),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    _require(
        completed.returncode == 0,
        "official PhysTwin Git provenance command failed: "
        f"git {' '.join(arguments)}: "
        f"{completed.stderr.decode('utf-8', 'replace').strip()}",
    )
    return completed.stdout


def _parse_git_tree(raw: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, path_bytes = entry.split(b"\t", 1)
            mode_bytes, kind, object_bytes, size_bytes = metadata.split(b" ", 3)
            relative = path_bytes.decode("utf-8")
            mode = mode_bytes.decode("ascii")
            object_id = object_bytes.decode("ascii")
            size = int(size_bytes)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("official PhysTwin Git tree is malformed") from error
        parts = relative.split("/")
        _require(
            relative
            and not relative.startswith("/")
            and all(part not in {"", ".", ".."} for part in parts),
            "official PhysTwin Git tree contains an unsafe path",
        )
        _require(kind == b"blob", f"non-blob official PhysTwin entry: {relative}")
        _require(
            mode in {"100644", "100755"},
            f"unsafe official PhysTwin Git mode: {relative}",
        )
        _require(
            len(object_id) in {40, 64}
            and all(character in "0123456789abcdef" for character in object_id),
            "invalid official PhysTwin Git object id",
        )
        _require(size >= 0, "invalid official PhysTwin Git blob size")
        records.append(
            {
                "git_object": object_id,
                "mode": mode,
                "path": relative,
                "size_bytes": size,
            }
        )
    _require(bool(records), "official PhysTwin Git tree is empty")
    paths = [str(record["path"]) for record in records]
    _require(paths == sorted(paths), "official PhysTwin Git tree order changed")
    _require(len(paths) == len(set(paths)), "official PhysTwin Git paths repeat")
    return records


def _git_blob_digest(path: Path, *, size: int, algorithm: str) -> str:
    before = os.lstat(path)
    _require(stat.S_ISREG(before.st_mode), f"tracked path is not regular: {path}")
    _require(before.st_size == size, f"tracked file size changed: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.new(algorithm)
    digest.update(f"blob {size}\0".encode("ascii"))
    try:
        after = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
            f"tracked file changed while opening: {path}",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _reject_untracked_importables_and_symlinks(
    root: Path, tracked_paths: frozenset[str]
) -> None:
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            if directory == root and entry.name == ".git":
                continue
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            observed = entry.stat(follow_symlinks=False)
            _require(
                not stat.S_ISLNK(observed.st_mode),
                f"official PhysTwin worktree contains a symlink: {relative}",
            )
            if stat.S_ISDIR(observed.st_mode):
                stack.append(path)
                continue
            _require(
                stat.S_ISREG(observed.st_mode),
                f"official PhysTwin worktree contains a special file: {relative}",
            )
            if relative in tracked_paths:
                continue
            importable = entry.name in {"sitecustomize.py", "usercustomize.py"} or (
                entry.name.endswith(_PYTHON_IMPORTABLE_SUFFIXES)
            )
            _require(
                not importable,
                f"official PhysTwin has an untracked importable file: {relative}",
            )


def _validate_official_phystwin_worktree(
    repository: str | Path,
    immutable_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    root = _canonical_directory(repository, label="official PhysTwin repository")
    top_level = Path(
        _run_git(root, ("rev-parse", "--show-toplevel")).decode("utf-8").strip()
    )
    _require(top_level == root, "official PhysTwin Git top level changed")
    revision = _run_git(root, ("rev-parse", "--verify", "HEAD")).decode("ascii").strip()
    _require(
        revision == OFFICIAL_PHYSTWIN_REVISION,
        "official PhysTwin revision changed",
    )
    _require(
        hashlib.sha256(revision.encode("ascii")).hexdigest()
        == immutable_bindings.get("official_phystwin_revision_literal"),
        "official PhysTwin revision differs from the immutable lock",
    )
    status = _run_git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=no"),
    )
    _require(status == b"", "official PhysTwin tracked worktree is dirty")
    object_format = (
        _run_git(root, ("rev-parse", "--show-object-format")).decode("ascii").strip()
    )
    _require(
        object_format in {"sha1", "sha256"},
        "unsupported official PhysTwin Git object format",
    )
    binding_tree_lines = _run_git(
        root,
        ("ls-tree", "-r", "--full-tree", "HEAD"),
    ).splitlines()
    binding_tree_manifest = b"".join(
        line + b"\n" for line in sorted(binding_tree_lines)
    )
    tree_sha256 = hashlib.sha256(binding_tree_manifest).hexdigest()
    _require(
        tree_sha256 == immutable_bindings.get("official_phystwin_git_tree_manifest"),
        "official PhysTwin Git tree differs from the immutable lock",
    )
    tree = _parse_git_tree(
        _run_git(root, ("ls-tree", "-r", "-l", "-z", "--full-tree", "HEAD"))
    )
    commit_payload = _run_git(root, ("cat-file", "commit", "HEAD"))
    _require(
        commit_payload.startswith(b"tree "),
        "official PhysTwin commit object is malformed",
    )
    commit_sha256 = hashlib.sha256(commit_payload).hexdigest()
    _require(
        commit_sha256 == immutable_bindings.get("official_phystwin_commit_object"),
        "official PhysTwin commit object differs from the immutable lock",
    )
    tracked_paths = frozenset(str(record["path"]) for record in tree)
    for record in tree:
        relative = str(record["path"])
        path = root / relative
        _require(
            _git_blob_digest(
                path,
                size=int(record["size_bytes"]),
                algorithm=object_format,
            )
            == record["git_object"],
            f"tracked official PhysTwin bytes changed: {relative}",
        )
    _reject_untracked_importables_and_symlinks(root, tracked_paths)
    for module_name, relative in _QQTT_IMPORTED_PROVENANCE.items():
        _require(
            relative in tracked_paths and (root / relative).is_file(),
            f"official PhysTwin lacks locked import source: {module_name}",
        )
    return {
        "repository_root": str(root),
        "revision": revision,
        "revision_literal_sha256": hashlib.sha256(revision.encode("ascii")).hexdigest(),
        "commit_object_sha256": commit_sha256,
        "git_tree_manifest_sha256": tree_sha256,
        "tracked_file_count": len(tree),
        "qqtt_imported_provenance": {
            name: str(root / relative)
            for name, relative in _QQTT_IMPORTED_PROVENANCE.items()
        },
    }


def _isolated_runpy_command(
    python: str | Path,
    script: str | Path,
    *,
    import_roots: Sequence[str | Path],
    arguments: Sequence[str],
    provenance_root: str | Path | None = None,
) -> list[str]:
    roots = [os.path.abspath(os.fspath(root)) for root in import_roots]
    return [
        os.fspath(python),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={HELD_PYCACHE_PREFIX}",
        "-c",
        _ISOLATED_RUNPY_BOOTSTRAP,
        str(len(roots)),
        *roots,
        os.path.abspath(os.fspath(script)),
        "" if provenance_root is None else os.path.abspath(os.fspath(provenance_root)),
        *map(str, arguments),
    ]


def validate_python_runtime(
    python: str | Path,
    immutable_bindings: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the frozen venv and interpreter without resolving its entry point.

    A virtualenv's ``bin/python`` is commonly a symlink.  Executing the resolved
    target bypasses Python's virtualenv discovery, so the supplied absolute path
    is retained for every subprocess.  Only the executable-byte checksum follows
    the symlink to its resolved regular file.
    """

    provided = os.fspath(python)
    _require(
        isinstance(provided, str)
        and os.path.isabs(provided)
        and os.path.abspath(provided) == provided,
        "supplied Python interpreter path is not exact and absolute",
    )
    supplied = Path(provided)
    _require(
        supplied == HELD_PYTHON_RUNTIME / "bin/python",
        "supplied Python interpreter is outside the frozen runtime",
    )
    runtime = _validate_held_python_runtime_tree(immutable_bindings)
    _require(supplied.is_file(), "supplied Python interpreter is missing")
    resolved = supplied.resolve(strict=True)
    _require(resolved.is_file(), "resolved Python interpreter is not a file")
    executable_sha256 = sha256_file(resolved)
    _require(
        executable_sha256 == immutable_bindings.get("python_executable"),
        "Python executable bytes differ from the immutable lock",
    )
    completed = subprocess.run(
        [
            str(supplied),
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={HELD_PYCACHE_PREFIX}",
            "-m",
            "pip",
            "freeze",
            "--all",
        ],
        check=True,
        capture_output=True,
    )
    freeze_sha256 = _sorted_pip_freeze_sha256(completed.stdout)
    _require(
        freeze_sha256 == immutable_bindings.get("python_pip_freeze_sorted"),
        "Python pip freeze differs from the immutable lock",
    )
    return {
        **runtime,
        "supplied_python_path": str(supplied),
        "resolved_python_path": str(resolved),
        "python_executable_sha256": executable_sha256,
        "python_pip_freeze_sorted_sha256": freeze_sha256,
    }


def validate_upstream_runtime(
    upstream_repo: str | Path,
    official_phystwin_repo: str | Path,
    official_config: str | Path,
    immutable_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail before computation if any frozen numerical implementation moved."""

    upstream = _canonical_directory(upstream_repo, label="frozen upstream repository")
    official = _canonical_directory(
        official_phystwin_repo,
        label="official PhysTwin repository",
    )
    config = Path(os.path.abspath(os.fspath(official_config)))
    _require(
        config.is_file()
        and not config.is_symlink()
        and config.resolve(strict=True) == config,
        "official PhysTwin config is missing, linked, or aliased",
    )
    observed_files: dict[str, str] = {}
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = upstream / relative
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.resolve(strict=True) == path,
            f"missing or linked frozen upstream file: {relative}",
        )
        observed = sha256_file(path)
        _require(observed == expected, f"frozen upstream file changed: {relative}")
        observed_files[relative] = observed
    config_sha256 = sha256_file(config)
    _require(
        config_sha256 == OFFICIAL_REAL_CONFIG_SHA256,
        "official PhysTwin real.yaml changed",
    )
    _require(
        config_sha256 == immutable_bindings.get("official_phystwin_real_config"),
        "official PhysTwin config differs from the immutable lock",
    )
    official_worktree = _validate_official_phystwin_worktree(
        official,
        immutable_bindings,
    )
    return {
        "upstream_repository_root": str(upstream),
        "official_phystwin_repository_root": str(official),
        "official_phystwin_revision": official_worktree["revision"],
        "official_config_path": str(config),
        "official_config_sha256": config_sha256,
        "official_phystwin_worktree": official_worktree,
        "upstream_file_sha256": observed_files,
    }


def _load_frame_zero_geometry(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    case_name: str,
    role: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = validate_frame_zero_bundle_manifest(
        manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    bundle_path = _validate_bound_file(manifest["bundle"], label="frame-zero bundle")
    with np.load(bundle_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == FRAME_ZERO_ARRAYS,
            "frame-zero bundle array set changed",
        )
        frame_indices = np.asarray(stored["frame_indices"])
        points = np.asarray(stored["object_points_world_m"], dtype=np.float32)
        colors = np.asarray(stored["object_colors_rgb"])
        camera_names = np.asarray(stored["camera_names"])
        rgb = np.asarray(stored["rgb_frame0"])
        masks = np.asarray(stored["mask_frame0"])
        depth = np.asarray(stored["depth_frame0_m"])
        valid = np.asarray(stored["depth_valid_frame0"])
        intrinsics = np.asarray(stored["intrinsics"])
        camera_to_world = np.asarray(stored["camera_to_world"])
        projections = np.asarray(stored["projection_world_to_pixel"])
    _require(
        np.array_equal(frame_indices, np.array([0])), "bundle contains a nonzero frame"
    )
    _require(
        points.ndim == 2 and points.shape[1] == 3, "invalid frame-zero object points"
    )
    _require(len(points) >= MINIMUM_NODE_COUNT, "frame-zero point count is below 128")
    _require(colors.shape == points.shape, "frame-zero point colors differ from points")
    _require(np.all(np.isfinite(points)), "frame-zero object points are non-finite")
    _require(np.all(np.isfinite(colors)), "frame-zero object colors are non-finite")
    camera_count = len(camera_names)
    _require(camera_count >= 2, "frame-zero bundle has fewer than two cameras")
    _require(
        rgb.ndim == 4 and rgb.shape[0] == camera_count, "invalid frame-zero RGB stack"
    )
    _require(masks.shape == rgb.shape[:3], "frame-zero masks differ from RGB")
    _require(
        depth.shape == masks.shape and valid.shape == masks.shape, "invalid depth stack"
    )
    _require(intrinsics.shape == (camera_count, 3, 3), "invalid camera intrinsics")
    _require(camera_to_world.shape == (camera_count, 4, 4), "invalid camera poses")
    _require(projections.shape == (camera_count, 3, 4), "invalid camera projections")
    _require(np.all(np.isfinite(intrinsics)), "non-finite camera intrinsics")
    _require(np.all(np.isfinite(camera_to_world)), "non-finite camera poses")
    _require(np.all(np.isfinite(projections)), "non-finite camera projections")
    if colors.dtype.kind in "ui":
        colors = colors.astype(np.float32) / float(np.iinfo(colors.dtype).max)
    else:
        colors = colors.astype(np.float32)
    _require(
        float(np.min(colors)) >= -1e-6 and float(np.max(colors)) <= 1.0 + 1e-6,
        "frame-zero colors must lie in [0,1]",
    )
    return manifest, points, np.clip(colors, 0.0, 1.0)


def _taxel_grid_root_frame(joint: float) -> np.ndarray:
    rows, columns = np.meshgrid(
        np.arange(_TAXEL_ROWS), np.arange(_TAXEL_COLS), indexing="ij"
    )
    rows = rows.reshape(-1).astype(np.float64)
    columns = columns.reshape(-1).astype(np.float64)
    y_root = -(_TAXEL_Y0_M + _TAXEL_Y_STEP_M * columns)
    z_root = -_TAXEL_Z_PITCH_M * (11.5 - rows)
    left = np.stack(
        (
            np.full_like(y_root, _FINGER_BASE_LEFT[0] + joint + _TAXEL_X_M),
            _FINGER_BASE_LEFT[1] + y_root,
            _FINGER_BASE_LEFT[2] + z_root,
        ),
        axis=1,
    )
    right = np.stack(
        (
            np.full_like(y_root, _FINGER_BASE_RIGHT[0] - joint - _TAXEL_X_M),
            _FINGER_BASE_RIGHT[1] + y_root,
            _FINGER_BASE_RIGHT[2] + z_root,
        ),
        axis=1,
    )
    interleaved = np.empty((2 * _TAXEL_ROWS * _TAXEL_COLS, 3), dtype=np.float64)
    interleaved[0::2] = left
    interleaved[1::2] = right
    return interleaved


def _gripper_taxel_points(opening_m: float, world_from_eef: np.ndarray) -> np.ndarray:
    clamped = float(np.clip(opening_m, 0.04, 0.112))
    normalized = (clamped - 0.04) / (0.112 - 0.04)
    joint = 0.038 - normalized * (0.038 - 0.005)
    points = _taxel_grid_root_frame(joint)
    pose = np.asarray(world_from_eef, dtype=np.float64)
    return points @ pose[:3, :3].T + pose[:3, 3]


def _validated_expected_range(
    value: Sequence[int] | None,
    *,
    expected_length: int,
    label: str,
) -> list[int] | None:
    if value is None:
        return None
    _require(
        isinstance(value, (list, tuple)) and len(value) == 2,
        f"{label} is not a two-element range",
    )
    start, stop = value
    _require(
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(stop, int)
        and not isinstance(stop, bool)
        and start >= 0
        and stop - start == expected_length,
        f"{label} has the wrong extent",
    )
    return [start, stop]


def _controller_trajectory_from_state(
    state: Deform360RobotKinematics,
    *,
    frame_count: int,
) -> np.ndarray:
    """Reproduce the taxel cloud from validated absolute EEF kinematics."""

    _require(
        state.frame_count == frame_count,
        "selected robot kinematics are not the prediction frame count",
    )
    poses = state.T_worlds
    openings = state.openings
    controllers: list[np.ndarray] = []
    for frame in range(frame_count):
        blocks: list[np.ndarray] = []
        for gripper in range(state.gripper_count):
            pose = poses[frame, gripper] if state.bimanual else poses[frame]
            opening = openings[frame, gripper] if state.bimanual else openings[frame]
            blocks.append(_gripper_taxel_points(float(opening), pose))
        controllers.append(np.concatenate(blocks, axis=0))
    trajectory = np.stack(controllers).astype(np.float32)
    _require(np.all(np.isfinite(trajectory)), "controller trajectory is non-finite")
    return trajectory


def load_controller_trajectory(
    robot_path: str | Path,
    *,
    frame_count: int = FRAME_COUNT,
    source_robot_path: str | Path | None = None,
    expected_selected_raw_frame_range: Sequence[int] | None = None,
    expected_prediction_raw_frame_range: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load strict realized EEF kinematics and reproduce the taxel cloud.

    A 76-frame input is explicitly a preselected prediction slice.  When its
    raw source is supplied, this function independently recomputes the shared
    81-frame selector and proves that all five selected archive fields are the
    exact corresponding 76-frame source slice.  Longer inputs are raw episode
    archives and are selected directly with the same shared contract.
    """

    _require(frame_count >= 2, "prediction frame count is invalid")
    selected_range = _validated_expected_range(
        expected_selected_raw_frame_range,
        expected_length=frame_count + 5,
        label="expected selected raw frame range",
    )
    prediction_range = _validated_expected_range(
        expected_prediction_raw_frame_range,
        expected_length=frame_count,
        label="expected prediction raw frame range",
    )
    state = load_robot_kinematics_archive(robot_path)

    if state.frame_count == frame_count:
        if source_robot_path is None:
            _require(
                selected_range is None and prediction_range is None,
                "raw ranges require the source robot archive",
            )
            selected_state = state
            audit: dict[str, Any] = {
                "policy_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
                "contract_sha256": ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
                "trajectory_semantics": ROBOT_KINEMATICS_WINDOW_CONTRACT[
                    "trajectory_semantics"
                ],
                "input_mode": "preselected_exact_prediction_slice",
                "selection_performed": False,
                "source_robot_frame_count": None,
                "selected_raw_frame_range_half_open": None,
                "prediction_raw_frame_range_half_open": None,
                "selected_prediction_frame_range_half_open": [0, frame_count],
                "prediction_frame_count": frame_count,
                "bimanual": state.bimanual,
                "gripper_count": state.gripper_count,
                "exact_source_slice_verified": False,
                "selected_array_records": robot_kinematics_array_records(state),
            }
        else:
            source_state = load_robot_kinematics_archive(source_robot_path)
            source_audit = select_robot_kinematics_window(
                source_state,
                window_length_frames=frame_count + 5,
                prediction_frame_count=frame_count,
            )
            source_selected_range = source_audit["selected_raw_frame_range_half_open"]
            source_prediction_range = source_audit[
                "prediction_raw_frame_range_half_open"
            ]
            if selected_range is not None:
                _require(
                    selected_range == source_selected_range,
                    "manifest selected raw range disagrees with the shared selector",
                )
            if prediction_range is not None:
                _require(
                    prediction_range == source_prediction_range,
                    "manifest prediction raw range disagrees with the shared selector",
                )
            source_slice_audit = validate_selected_robot_kinematics_bundle(
                state,
                source_state=source_state,
                prediction_start_frame=int(source_prediction_range[0]),
                prediction_frame_count=frame_count,
            )
            selected_state = state
            audit = dict(source_audit)
            audit.pop("artifact_sha256", None)
            audit.update(
                {
                    "input_mode": "preselected_exact_prediction_slice",
                    "selection_performed": True,
                    "selected_prediction_frame_range_half_open": [0, frame_count],
                    "exact_source_slice_verified": True,
                    "selected_array_records": robot_kinematics_array_records(state),
                    "selected_bundle_validation": source_slice_audit,
                }
            )
    else:
        _require(
            source_robot_path is None,
            "raw robot input must not also name a source archive",
        )
        source_audit = select_robot_kinematics_window(
            state,
            window_length_frames=frame_count + 5,
            prediction_frame_count=frame_count,
        )
        if selected_range is not None:
            _require(
                selected_range == source_audit["selected_raw_frame_range_half_open"],
                "expected selected raw range disagrees with the shared selector",
            )
        if prediction_range is not None:
            _require(
                prediction_range
                == source_audit["prediction_raw_frame_range_half_open"],
                "expected prediction raw range disagrees with the shared selector",
            )
        prediction_start = int(source_audit["prediction_raw_frame_range_half_open"][0])
        selected_state = slice_robot_kinematics(
            state,
            start_frame=prediction_start,
            frame_count=frame_count,
        )
        audit = dict(source_audit)
        audit.pop("artifact_sha256", None)
        audit.update(
            {
                "input_mode": "raw_episode_robot_kinematics",
                "selection_performed": True,
                "selected_prediction_frame_range_half_open": [0, frame_count],
                "exact_source_slice_verified": True,
                "selected_array_records": robot_kinematics_array_records(
                    selected_state
                ),
            }
        )

    trajectory = _controller_trajectory_from_state(
        selected_state,
        frame_count=frame_count,
    )
    audit["controller_point_count"] = int(trajectory.shape[1])
    audit["controller_trajectory_sha256"] = sha256_array(trajectory)
    audit["artifact_sha256"] = robot_artifact_sha256(audit)
    return trajectory, audit


def _validate_controller_kinematics_audit(
    audit: object,
    *,
    require_raw_source: bool,
    controller_trajectory: np.ndarray | None = None,
) -> dict[str, Any]:
    _require(isinstance(audit, Mapping), "robot kinematics audit is missing")
    value = dict(audit)
    _require(
        value.get("policy_id") == ROBOT_KINEMATICS_WINDOW_POLICY_ID
        and value.get("contract_sha256") == ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256
        and value.get("trajectory_semantics")
        == ROBOT_KINEMATICS_WINDOW_CONTRACT["trajectory_semantics"],
        "robot kinematics contract changed",
    )
    _require(
        value.get("prediction_frame_count") == FRAME_COUNT
        and value.get("selected_prediction_frame_range_half_open") == [0, FRAME_COUNT],
        "robot kinematics prediction range changed",
    )
    if require_raw_source:
        selected_range = value.get("selected_raw_frame_range_half_open")
        prediction_range = value.get("prediction_raw_frame_range_half_open")
        _require(
            isinstance(selected_range, list)
            and len(selected_range) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in selected_range
            )
            and isinstance(prediction_range, list)
            and len(prediction_range) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in prediction_range
            )
            and selected_range[0] == prediction_range[0]
            and selected_range[1] - selected_range[0] == FRAME_COUNT + 5
            and prediction_range[1] - prediction_range[0] == FRAME_COUNT
            and value.get("selection_performed") is True
            and value.get("exact_source_slice_verified") is True,
            "robot kinematics audit lacks a verified raw source range",
        )
    _require(
        value.get("artifact_sha256") == robot_artifact_sha256(value),
        "robot kinematics audit checksum changed",
    )
    if controller_trajectory is not None:
        controllers = np.asarray(controller_trajectory)
        _require(
            value.get("controller_point_count") == controllers.shape[1]
            and value.get("controller_trajectory_sha256") == sha256_array(controllers),
            "robot kinematics audit does not bind the controller trajectory",
        )
    return value


def build_prediction_only_artifacts(
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    *,
    case_name: str,
    role: str = "calibration",
) -> dict[str, Any]:
    """Create the exact constant-object PhysTwin contract from one frame."""

    manifest, points, colors = _load_frame_zero_geometry(
        frame_zero_manifest_path,
        lock_path,
        case_name=case_name,
        role=role,
    )
    alignment = manifest.get("action_alignment", {})
    _require(isinstance(alignment, Mapping), "frame-zero action alignment is missing")
    staging_range = alignment.get("selected_raw_frame_range_half_open")
    prediction_range = alignment.get("prediction_raw_frame_range_half_open")
    _require(
        isinstance(staging_range, list)
        and len(staging_range) == 2
        and int(staging_range[1]) - int(staging_range[0]) == FRAME_COUNT + 5,
        "frame-zero action alignment is not the frozen 81-frame window",
    )
    _require(
        isinstance(prediction_range, list)
        and len(prediction_range) == 2
        and int(prediction_range[0]) == int(staging_range[0])
        and int(prediction_range[1]) - int(prediction_range[0]) == FRAME_COUNT,
        "frame-zero selected action is not the frozen 76-frame window",
    )
    robot_path = _validate_bound_file(
        alignment.get("selected_action_bundle", {}),
        label="selected realized robot kinematics",
    )
    action_inputs = manifest.get("action_inputs", {})
    _require(isinstance(action_inputs, Mapping), "frame-zero robot inputs are missing")
    source_robot_path = _validate_bound_file(
        action_inputs.get("robot_trajectory", {}),
        label="source realized robot kinematics",
    )
    controllers, robot_kinematics_window = load_controller_trajectory(
        robot_path,
        source_robot_path=source_robot_path,
        expected_selected_raw_frame_range=staging_range,
        expected_prediction_raw_frame_range=prediction_range,
    )
    robot_kinematics_window = _validate_controller_kinematics_audit(
        robot_kinematics_window,
        require_raw_source=True,
        controller_trajectory=controllers,
    )
    object_points = np.repeat(points[None], FRAME_COUNT, axis=0).astype(np.float32)
    object_colors = np.repeat(colors[None], FRAME_COUNT, axis=0).astype(np.float32)
    observed = np.ones(object_points.shape[:2], dtype=bool)
    marker = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": manifest["object_id"],
        "episode_id": int(manifest["episode_id"]),
        "role": role,
        "object_observation_frames_used": [0],
        "known_future_realized_robot_kinematics_used": True,
        # Compatibility alias for the original upstream input marker.
        "known_future_robot_trajectory_used": True,
        "future_object_observations_present": False,
        "future_tactile_used": False,
        "frame_zero_manifest_artifact_sha256": manifest["artifact_sha256"],
        "frame_zero_bundle_sha256": manifest["bundle"]["sha256"],
        "source_robot_trajectory_sha256": manifest["action_inputs"]["robot_trajectory"][
            "sha256"
        ],
        "selected_robot_trajectory_sha256": sha256_file(robot_path),
        "robot_kinematics_window": robot_kinematics_window,
        # Compatibility alias retained for already-frozen upstream consumers.
        "action_window": robot_kinematics_window,
    }
    payload = {
        "object_points": object_points,
        "object_colors": object_colors,
        "object_visibilities": observed,
        "object_motions_valid": observed.copy(),
        "controller_points": controllers,
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
        "prediction_only_input": marker,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PREDICTION_INPUT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": manifest["object_id"],
        "episode_id": int(manifest["episode_id"]),
        "role": role,
        "frame_count": FRAME_COUNT,
        "point_count": len(points),
        "controller_point_count": int(controllers.shape[1]),
        "frame_zero_points_sha256": sha256_array(points),
        "frame_zero_colors_sha256": sha256_array(colors),
        "controller_trajectory_sha256": sha256_array(controllers),
        "robot_kinematics_window": robot_kinematics_window,
        "action_window": robot_kinematics_window,
        "input_sha256": {
            "held_lock": sha256_file(lock_path),
            "frame_zero_manifest": sha256_file(frame_zero_manifest_path),
            "frame_zero_bundle": manifest["bundle"]["sha256"],
            "robot_trajectory": manifest["action_inputs"]["robot_trajectory"]["sha256"],
            "robot_metadata": manifest["action_inputs"]["robot_metadata"]["sha256"],
            "selected_robot_trajectory": sha256_file(robot_path),
        },
        "output_sha256": sha256_file(destination),
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_aligned_robot_kinematics_read": True,
            # Compatibility alias in the held protocol v1/v2 schema.
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
        },
        "passed": True,
    }
    summary["artifact_sha256"] = held_artifact_sha256(summary)
    _write_json(summary_path, summary)
    return summary


def _run_logged(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    log_path: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise _LoggedCommandError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n"
            + "\n".join(tail),
            returncode=completed.returncode,
            elapsed_seconds=elapsed,
        )
    return elapsed


def _expected_warp_overrides() -> dict[str, Any]:
    return {
        "controller_max_neighbours": WARP_DYNAMICS["controller_max_neighbours"],
        "controller_radius": WARP_DYNAMICS["controller_radius_m"],
        "dashpot_damping": WARP_DYNAMICS["dashpot_damping"],
        "drag_damping": WARP_DYNAMICS["drag_damping"],
        "init_spring_Y": WARP_DYNAMICS["init_spring_y"],
    }


def _load_prediction_pickle(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("rb") as stream:
        value = pickle.load(stream)
    _require(isinstance(value, Mapping), "prediction-only pickle is not a mapping")
    return value


def _upstream_result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _metric_matches(observed: Any, expected: float) -> bool:
    return isinstance(observed, (int, float)) and np.isclose(
        float(observed), float(expected), rtol=1e-12, atol=1e-12
    )


def _validate_inadmissible_automatic_twin(
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    state_path: str | Path,
    twin_summary_path: str | Path,
    *,
    case_name: str,
    object_id: str,
    episode_id: int,
    role: str,
) -> dict[str, Any]:
    """Validate the one upstream exit-2 result eligible for persistence fallback."""

    prediction_path = Path(prediction_data_path).resolve()
    simulator_path = Path(simulator_data_path).resolve()
    graph_file = Path(graph_path).resolve()
    state_file = Path(state_path).resolve()
    summary_file = Path(twin_summary_path).resolve()
    for label, path in (
        ("prediction-only input", prediction_path),
        ("automatic-twin simulator data", simulator_path),
        ("automatic-twin graph", graph_file),
        ("automatic-twin state", state_file),
        ("automatic-twin summary", summary_file),
    ):
        _require(path.is_file() and not path.is_symlink(), f"missing {label}")

    prediction = _load_prediction_pickle(prediction_path)
    points = np.asarray(prediction.get("object_points"))
    colors = np.asarray(prediction.get("object_colors"))
    controllers = np.asarray(prediction.get("controller_points"))
    _require(
        points.ndim == 3
        and points.shape[0] == FRAME_COUNT
        and points.shape[2] == 3
        and colors.shape == points.shape
        and controllers.ndim == 3
        and controllers.shape[0] == FRAME_COUNT
        and controllers.shape[2] == 3,
        "inadmissible twin used an invalid prediction-only input",
    )
    _require(
        np.array_equal(points, np.repeat(points[:1], FRAME_COUNT, axis=0)),
        "inadmissible twin input contains future object geometry",
    )

    with np.load(graph_file, allow_pickle=False) as stored:
        expected_graph_arrays = {
            "vertices",
            "colors",
            "source_indices",
            "springs",
            "rest_lengths",
            "masses",
            "bridge_spring_count",
            "observed_node_count",
            "latent_node_count",
            "contact_anchor_indices",
            "contact_chain_spring_count",
            "reusable_graph_sha256",
        }
        _require(
            set(stored.files) == expected_graph_arrays,
            "inadmissible twin graph array set changed",
        )
        vertices = np.asarray(stored["vertices"], dtype=np.float64)
        springs = np.asarray(stored["springs"], dtype=np.int64)
        rest_lengths = np.asarray(stored["rest_lengths"], dtype=np.float64)
        bridge_count = int(np.asarray(stored["bridge_spring_count"]).item())
        observed_count = int(np.asarray(stored["observed_node_count"]).item())
        latent_count = int(np.asarray(stored["latent_node_count"]).item())
        anchors = np.asarray(stored["contact_anchor_indices"], dtype=np.int64)
        contact_chain_count = int(
            np.asarray(stored["contact_chain_spring_count"]).item()
        )
        graph_sha256 = str(np.asarray(stored["reusable_graph_sha256"]).item())
    effective_count = min(CANONICAL_NODE_COUNT, points.shape[1])
    _require(
        vertices.ndim == 2
        and vertices.shape[1] == 3
        and springs.ndim == 2
        and springs.shape[1] == 2
        and rest_lengths.shape == (len(springs),)
        and observed_count == effective_count
        and latent_count == len(vertices) - observed_count
        and 0 <= bridge_count <= len(springs)
        and 0 <= contact_chain_count <= bridge_count
        and np.all(np.isfinite(vertices))
        and np.all(np.isfinite(rest_lengths))
        and np.all(rest_lengths > 0.0)
        and np.all((springs >= 0) & (springs < len(vertices)))
        and np.all((anchors >= 0) & (anchors < len(vertices))),
        "inadmissible twin graph failed structural validation",
    )

    with np.load(state_file, allow_pickle=False) as stored:
        expected_state_arrays = {
            "vertices",
            "readout_weights",
            "readout_covariance_m2",
            "target_prior_reliability",
            "state_covariance_m2",
            "source_to_target_distance_m",
            "target_to_source_distance_m",
            "relative_edge_strain",
            "canonical_graph_sha256",
            "state_frame",
        }
        _require(
            set(stored.files) == expected_state_arrays,
            "inadmissible twin state array set changed",
        )
        state_vertices = np.asarray(stored["vertices"], dtype=np.float64)
        weights = np.asarray(stored["readout_weights"], dtype=np.float64)
        readout_covariance = np.asarray(
            stored["readout_covariance_m2"], dtype=np.float64
        )
        reliability = np.asarray(stored["target_prior_reliability"], dtype=np.float64)
        state_covariance = np.asarray(stored["state_covariance_m2"], dtype=np.float64)
        source_distance = np.asarray(
            stored["source_to_target_distance_m"], dtype=np.float64
        )
        target_distance = np.asarray(
            stored["target_to_source_distance_m"], dtype=np.float64
        )
        strain = np.asarray(stored["relative_edge_strain"], dtype=np.float64)
        state_graph_sha256 = str(np.asarray(stored["canonical_graph_sha256"]).item())
        state_frame = int(np.asarray(stored["state_frame"]).item())
    point_count = points.shape[1]
    _require(
        state_vertices.shape == vertices.shape
        and np.array_equal(state_vertices, vertices)
        and weights.shape == (point_count, len(vertices))
        and readout_covariance.shape == (point_count, 3, 3)
        and reliability.shape == (point_count,)
        and state_covariance.shape == (len(vertices), 3, 3)
        and source_distance.shape == (len(vertices),)
        and target_distance.shape == (point_count,)
        and strain.shape == (len(springs),)
        and state_graph_sha256 == graph_sha256
        and state_frame == 0
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6)
        and np.all(np.isfinite(reliability))
        and np.all((0.0 <= reliability) & (reliability <= 1.0))
        and np.all(np.isfinite(source_distance))
        and np.all(np.isfinite(target_distance))
        and np.all(np.isfinite(strain)),
        "inadmissible twin state failed structural validation",
    )

    twin = _load_json(summary_file)
    expected_summary_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "protocol_config_sha256",
        "object_id",
        "episode_id",
        "phase",
        "graph_mode",
        "capacity_diagnostic",
        "graph",
        "state_metrics",
        "input_sha256",
        "output_sha256",
        "information_boundary",
        "prediction_input_validation",
        "sota_input_validation",
        "passed",
        "claim_boundary",
        "result_sha256",
    }
    _require(
        set(twin) == expected_summary_keys,
        "inadmissible automatic-twin summary schema changed",
    )
    _require(
        twin.get("schema_version") == 1
        and twin.get("artifact_kind") == "Deform360AutomaticEpisodeTwin"
        and twin.get("protocol_id") == AUTOMATIC_TWIN_PROTOCOL_ID
        and twin.get("protocol_config_sha256") == AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
        and twin.get("object_id") == object_id
        and int(twin.get("episode_id", -1)) == episode_id
        and twin.get("phase") == ("calibration" if role == "calibration" else "source")
        and twin.get("graph_mode") == "episode_specific_frame_zero_control"
        and twin.get("passed") is False
        and twin.get("sota_input_validation") is None
        and twin.get("result_sha256") == _upstream_result_sha256(twin),
        "automatic twin is not a checksummed explicit inadmissible result",
    )
    capacity = twin.get("capacity_diagnostic", {})
    _require(
        capacity
        == {
            "configured_canonical_node_count": 192,
            "requested_canonical_node_count": CANONICAL_NODE_COUNT,
            "effective_canonical_node_count": effective_count,
            "source_only_override": effective_count != 192,
            "capacity_is_a_maximum": True,
        },
        "inadmissible twin used another graph capacity",
    )
    graph = twin.get("graph", {})
    _require(
        graph
        == {
            "schema_version": 1,
            "artifact_kind": "Deform360CanonicalReusableGraph",
            "path": str(graph_file),
            "reusable_graph_sha256": graph_sha256,
            "node_count": len(vertices),
            "object_spring_count": len(springs),
            "bridge_spring_count": bridge_count,
            "observed_node_count": observed_count,
            "latent_node_count": latent_count,
            "contact_anchor_count": len(anchors),
            "contact_chain_spring_count": contact_chain_count,
        },
        "inadmissible twin summary binds another graph",
    )
    _require(
        twin.get("input_sha256")
        == {
            "episode_final_data": sha256_file(prediction_path),
            "development_observations": None,
            "contact_conditioned_action": None,
        }
        and twin.get("output_sha256")
        == {
            "episode_graph": sha256_file(graph_file),
            "simulator_final_data": sha256_file(simulator_path),
            "state_artifact": sha256_file(state_file),
        },
        "inadmissible twin input/output checksums changed",
    )
    _require(
        twin.get("prediction_input_validation")
        == {
            "frame_count": FRAME_COUNT,
            "point_count": point_count,
            "controller_point_count": controllers.shape[1],
            "frame_zero_points_sha256": sha256_array(points[0]),
            "controller_trajectory_sha256": sha256_array(controllers),
        },
        "inadmissible twin prediction-input validation changed",
    )
    _require(
        twin.get("information_boundary")
        == {
            "object_observation_frames_used": [0],
            "future_robot_action_available": True,
            "post_initial_object_observation_used": False,
            "simulator_residual_used": False,
            "target_access": False,
            "prediction_only_input_required": True,
            "future_object_tracks_present": False,
            "contact_conditioned_action_used": False,
            "contact_conditioned_action_result_sha256": None,
        },
        "inadmissible twin crossed the prediction-only boundary",
    )

    metrics = twin.get("state_metrics", {})
    expected_metric_keys = {
        "passed",
        "finite",
        "symmetric_chamfer_m",
        "source_to_target_p95_m",
        "target_to_source_p95_m",
        "observed_target_fraction",
        "canonical_supported_fraction",
        "effective_target_reliability",
        "initial_readout_rmse_m",
        "p99_absolute_relative_edge_strain",
        "maximum_absolute_relative_edge_strain",
        "maximum_bridge_absolute_relative_edge_strain",
        "maximum_contact_anchor_error_m",
    }
    _require(
        isinstance(metrics, Mapping)
        and set(metrics) == expected_metric_keys
        and metrics.get("passed") is False,
        "automatic twin lacks explicit failed state metrics",
    )
    finite = bool(
        np.all(np.isfinite(state_vertices))
        and np.all(np.isfinite(readout_covariance))
        and np.all(np.isfinite(state_covariance))
    )
    readout = weights @ state_vertices
    bridge_strain = strain[-bridge_count:] if bridge_count else np.empty(0)
    expected_metrics = {
        "symmetric_chamfer_m": 0.5
        * (float(np.mean(source_distance)) + float(np.mean(target_distance))),
        "source_to_target_p95_m": float(np.quantile(source_distance, 0.95)),
        "target_to_source_p95_m": float(np.quantile(target_distance, 0.95)),
        "observed_target_fraction": float(
            np.mean(
                target_distance
                <= AUTOMATIC_TWIN_ADMISSION_THRESHOLDS["maximum_supported_distance_m"]
            )
        ),
        "canonical_supported_fraction": float(
            np.mean(
                source_distance
                <= AUTOMATIC_TWIN_ADMISSION_THRESHOLDS["maximum_supported_distance_m"]
            )
        ),
        "effective_target_reliability": float(np.mean(reliability)),
        "initial_readout_rmse_m": float(np.sqrt(np.mean((readout - points[0]) ** 2))),
        "p99_absolute_relative_edge_strain": float(np.quantile(strain, 0.99)),
        "maximum_absolute_relative_edge_strain": float(np.max(strain)),
        "maximum_bridge_absolute_relative_edge_strain": float(
            np.max(bridge_strain, initial=0.0)
        ),
    }
    _require(metrics.get("finite") is finite, "automatic twin finite metric changed")
    for name, expected in expected_metrics.items():
        _require(
            _metric_matches(metrics.get(name), expected),
            f"automatic twin metric changed: {name}",
        )
    contact_error = metrics.get("maximum_contact_anchor_error_m")
    _require(
        isinstance(contact_error, (int, float))
        and np.isfinite(float(contact_error))
        and float(contact_error) >= 0.0,
        "automatic twin contact metric is invalid",
    )
    threshold = AUTOMATIC_TWIN_ADMISSION_THRESHOLDS
    recomputed_pass = bool(
        finite
        and float(metrics["observed_target_fraction"])
        >= threshold["minimum_observed_target_fraction"]
        and float(metrics["effective_target_reliability"])
        >= threshold["minimum_effective_target_reliability"]
        and float(metrics["p99_absolute_relative_edge_strain"])
        <= threshold["maximum_p99_relative_edge_strain"]
        and float(metrics["maximum_bridge_absolute_relative_edge_strain"])
        <= threshold["maximum_bridge_relative_edge_strain"]
        and float(contact_error) <= threshold["maximum_contact_anchor_error_m"]
    )
    _require(not recomputed_pass, "automatic twin metrics actually pass admission")
    _require(
        isinstance(twin.get("claim_boundary"), str)
        and "frame-zero episode-twin control" in twin["claim_boundary"],
        "automatic twin claim boundary changed",
    )
    return twin


def _graph_contact_distances(
    vertex_count: int,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(vertex_count)]
    for edge, length in zip(springs, rest_lengths):
        first, second = map(int, edge)
        distance = float(length)
        _require(
            0 <= first < vertex_count and 0 <= second < vertex_count,
            "invalid graph edge",
        )
        _require(np.isfinite(distance) and distance >= 0.0, "invalid graph rest length")
        adjacency[first].append((second, distance))
        adjacency[second].append((first, distance))
    distances = np.full(vertex_count, np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for anchor_value in anchors:
        anchor = int(anchor_value)
        _require(0 <= anchor < vertex_count, "invalid contact anchor")
        distances[anchor] = 0.0
        heapq.heappush(queue, (0.0, anchor))
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, edge_length in adjacency[node]:
            proposed = distance + edge_length
            if proposed < distances[neighbour]:
                distances[neighbour] = proposed
                heapq.heappush(queue, (proposed, neighbour))
    _require(
        np.all(np.isfinite(distances)), "contact does not reach the material graph"
    )
    return distances


def build_physical_prediction_archive(
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    readout_path: str | Path,
    twin_summary_path: str | Path,
    driven_result_path: str | Path,
    zero_result_path: str | Path,
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    case_name: str,
    role: str,
    runtime_provenance: Mapping[str, Any],
    stage_runtime_seconds: Mapping[str, float],
) -> dict[str, Any]:
    """Port the frozen driven-minus-zero graph-support sealer for held cases."""

    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    prediction_data = _load_prediction_pickle(prediction_data_path)
    points = np.asarray(prediction_data["object_points"], dtype=np.float64)
    controllers = np.asarray(prediction_data["controller_points"])
    marker = prediction_data.get("prediction_only_input", {})
    _require(
        points.ndim == 3 and points.shape[0] == FRAME_COUNT, "invalid prediction points"
    )
    _require(
        controllers.ndim == 3 and len(controllers) == FRAME_COUNT, "invalid controllers"
    )
    _require(
        np.array_equal(points, np.repeat(points[:1], FRAME_COUNT, axis=0)),
        "prediction input contains changing future object geometry",
    )
    _require(
        marker.get("object_observation_frames_used") == [0], "prediction marker changed"
    )
    _require(
        marker.get("future_object_observations_present") is False,
        "future object data present",
    )
    _require(marker.get("future_tactile_used") is False, "future tactile present")
    _require(
        marker.get("known_future_realized_robot_kinematics_used") is True,
        "prediction marker lacks realized robot kinematics",
    )
    robot_kinematics_window = _validate_controller_kinematics_audit(
        marker.get("robot_kinematics_window"),
        require_raw_source=True,
        controller_trajectory=controllers,
    )
    twin = _load_json(twin_summary_path)
    _require(twin.get("passed") is True, "automatic twin failed admission")
    _require(twin.get("object_id") == frame_zero["object_id"], "twin object changed")
    _require(
        int(twin.get("episode_id", -1)) == int(frame_zero["episode_id"]),
        "twin episode changed",
    )
    boundary = twin.get("information_boundary", {})
    _require(boundary.get("target_access") is False, "automatic twin accessed target")
    _require(
        boundary.get("post_initial_object_observation_used") is False,
        "twin used future object",
    )
    _require(
        twin.get("input_sha256", {}).get("episode_final_data")
        == sha256_file(prediction_data_path),
        "twin used another prediction input",
    )
    _require(
        twin.get("output_sha256", {}).get("simulator_final_data")
        == sha256_file(simulator_data_path),
        "simulator data changed after twin construction",
    )
    with np.load(graph_path, allow_pickle=False) as graph:
        vertices = np.asarray(graph["vertices"], dtype=np.float64)
        springs = np.asarray(graph["springs"], dtype=np.int64)
        rest_lengths = np.asarray(graph["rest_lengths"], dtype=np.float64)
        anchors = np.asarray(graph["contact_anchor_indices"], dtype=np.int64)
        observed_nodes = int(np.asarray(graph["observed_node_count"]).item())
        graph_semantic_sha256 = str(np.asarray(graph["reusable_graph_sha256"]).item())
    _require(
        MINIMUM_NODE_COUNT <= observed_nodes <= CANONICAL_NODE_COUNT,
        "observed graph capacity is outside the frozen range",
    )
    with np.load(readout_path, allow_pickle=False) as state:
        weights = np.asarray(state["readout_weights"], dtype=np.float64)
        state_graph_sha256 = str(np.asarray(state["canonical_graph_sha256"]).item())
    _require(state_graph_sha256 == graph_semantic_sha256, "readout uses another graph")
    _require(weights.shape == (points.shape[1], len(vertices)), "readout shape changed")
    _require(
        np.all(np.isfinite(weights)) and np.all(weights >= 0.0),
        "invalid readout weights",
    )
    _require(
        np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6), "readout is not convex"
    )

    trajectories: dict[str, np.ndarray] = {}
    result_files = {
        "driven": Path(driven_result_path),
        "zero_action": Path(zero_result_path),
    }
    expected_scales = {"driven": 1.0, "zero_action": 0.0}
    for label, result_file in result_files.items():
        result = _load_json(result_file)
        _require(result.get("passed") is True, f"{label} Warp rollout failed")
        _require(
            "external_target_scoring" not in result, f"{label} rollout read a target"
        )
        _require(
            result.get("data_sha256") == sha256_file(simulator_data_path),
            "Warp data changed",
        )
        _require(
            result.get("official_phystwin_revision") == OFFICIAL_PHYSTWIN_REVISION,
            "Warp used another PhysTwin revision",
        )
        _require(
            result.get("config_sha256") == OFFICIAL_REAL_CONFIG_SHA256,
            "Warp config changed",
        )
        _require(
            result.get("config_overrides") == _expected_warp_overrides(),
            "Warp overrides changed",
        )
        _require(
            result.get("support_dynamics", {}).get("mode")
            == WARP_DYNAMICS["support_dynamics"],
            "Warp support dynamics changed",
        )
        graph_record = result.get("canonical_reusable_graph", {})
        _require(
            graph_record.get("file_sha256") == sha256_file(graph_path),
            "Warp graph file changed",
        )
        _require(
            graph_record.get("reusable_graph_sha256") == graph_semantic_sha256,
            "Warp graph changed",
        )
        _require(
            int(graph_record.get("controller_patch_size_per_anchor", -1))
            == WARP_DYNAMICS["canonical_controller_patch_size"],
            "Warp controller patch changed",
        )
        _require(
            float(
                result.get("realized_actuation", {}).get(
                    "controller_displacement_scale", -1.0
                )
            )
            == expected_scales[label],
            f"{label} action scale changed",
        )
        trajectory_path = result_file.with_name("official_phystwin_trajectory.npz")
        _require(
            result.get("trajectory_sha256") == sha256_file(trajectory_path),
            "trajectory changed",
        )
        with np.load(trajectory_path, allow_pickle=False) as trajectory_file:
            trajectory = np.asarray(trajectory_file["vertices"], dtype=np.float64)
        _require(
            trajectory.ndim == 3
            and trajectory.shape[0] == FRAME_COUNT
            and trajectory.shape[1] >= len(vertices)
            and trajectory.shape[2] == 3
            and np.all(np.isfinite(trajectory)),
            f"invalid {label} trajectory",
        )
        trajectories[label] = trajectory[:, : len(vertices)]

    distances = _graph_contact_distances(len(vertices), springs, rest_lengths, anchors)
    node_support = np.exp(-distances / LENGTH_SCALE_M)
    support = np.clip(weights @ node_support, 0.0, 1.0)
    driven_readout = np.einsum(
        "mn,tnc->tmc", weights, trajectories["driven"], optimize=True
    )
    zero_readout = np.einsum(
        "mn,tnc->tmc", weights, trajectories["zero_action"], optimize=True
    )
    initial = points[0]
    offset = initial - zero_readout[0]
    driven_readout += offset[None]
    zero_readout += offset[None]
    prediction = initial[None] + ACTION_RESPONSE * support[None, :, None] * (
        driven_readout - zero_readout
    )
    persistence = np.repeat(initial[None], FRAME_COUNT, axis=0)
    _require(np.all(np.isfinite(prediction)), "physical prediction is non-finite")
    arrays = {
        "prediction_m": prediction.astype(np.float32),
        "persistence_m": persistence.astype(np.float32),
        "driven_readout_m": driven_readout.astype(np.float32),
        "zero_action_readout_m": zero_readout.astype(np.float32),
        "action_support": support.astype(np.float32),
        "frame_zero_points_m": initial.astype(np.float32),
    }
    archive = Path(archive_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **arrays)
    input_paths = {
        "prediction_only_input": prediction_data_path,
        "simulator_final_data": simulator_data_path,
        "episode_graph": graph_path,
        "state_artifact": readout_path,
        "twin_summary": twin_summary_path,
        "driven_result": driven_result_path,
        "zero_action_result": zero_result_path,
        "driven_trajectory": Path(driven_result_path).with_name(
            "official_phystwin_trajectory.npz"
        ),
        "zero_action_trajectory": Path(zero_result_path).with_name(
            "official_phystwin_trajectory.npz"
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": frame_zero["object_id"],
        "episode_id": int(frame_zero["episode_id"]),
        "role": role,
        "physical_mode": PHYSICAL_MODE_WARP_TWIN,
        "physical_admitted": True,
        "fallback_diagnostics": None,
        "frozen_predictor": {
            "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
            "official_real_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
            "length_scale_m": LENGTH_SCALE_M,
            "action_response": ACTION_RESPONSE,
            "autonomous_drift_response": AUTONOMOUS_DRIFT_RESPONSE,
            "frame_count": FRAME_COUNT,
            "observed_graph_node_count": observed_nodes,
            "total_graph_node_count": len(vertices),
            "point_count": points.shape[1],
            "warp_dynamics": dict(WARP_DYNAMICS),
        },
        "physical_prediction_archive": {
            **_bound_file(archive),
            "array_sha256": {
                name: sha256_array(value) for name, value in arrays.items()
            },
        },
        "input_files": {name: _bound_file(path) for name, path in input_paths.items()},
        "held_lock_sha256": sha256_file(lock_path),
        "frame_zero_manifest_sha256": sha256_file(frame_zero_manifest_path),
        "frame_zero_manifest_artifact_sha256": frame_zero["artifact_sha256"],
        "robot_kinematics_window": robot_kinematics_window,
        "runtime_provenance": dict(runtime_provenance),
        "stage_runtime_seconds": {
            key: float(value) for key, value in stage_runtime_seconds.items()
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_aligned_robot_kinematics_read": True,
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "future_object_visibility_read": False,
            "future_tactile_read": False,
            "external_target_scoring_in_warp": False,
            "outcome_created": False,
            "outcome_read": False,
            "physical_prediction_hashed_before_outcome": True,
        },
        "passed": True,
    }
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    _write_json(manifest_path, manifest)
    return validate_physical_prediction_manifest(manifest_path, verify_archive=True)


def build_persistence_fallback_archive(
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    state_path: str | Path,
    twin_summary_path: str | Path,
    automatic_twin_log_path: str | Path,
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    case_name: str,
    role: str,
    automatic_twin_exit_code: int,
    runtime_provenance: Mapping[str, Any],
    stage_runtime_seconds: Mapping[str, float],
) -> dict[str, Any]:
    """Seal persistence only after a genuine automatic-twin admission rejection."""

    _require(
        automatic_twin_exit_code == AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE,
        "persistence fallback requires the frozen inadmissible exit code",
    )
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    twin = _validate_inadmissible_automatic_twin(
        prediction_data_path,
        simulator_data_path,
        graph_path,
        state_path,
        twin_summary_path,
        case_name=case_name,
        object_id=str(frame_zero["object_id"]),
        episode_id=int(frame_zero["episode_id"]),
        role=role,
    )
    prediction_data = _load_prediction_pickle(prediction_data_path)
    points = np.asarray(prediction_data["object_points"], dtype=np.float32)
    controllers = np.asarray(prediction_data["controller_points"])
    marker = prediction_data.get("prediction_only_input", {})
    _require(
        marker.get("known_future_realized_robot_kinematics_used") is True,
        "prediction marker lacks realized robot kinematics",
    )
    robot_kinematics_window = _validate_controller_kinematics_audit(
        marker.get("robot_kinematics_window"),
        require_raw_source=True,
        controller_trajectory=controllers,
    )
    _require(
        points.shape[0] == FRAME_COUNT
        and np.array_equal(points, np.repeat(points[:1], FRAME_COUNT, axis=0)),
        "persistence fallback input contains changing object geometry",
    )
    persistence = np.repeat(points[:1], FRAME_COUNT, axis=0).astype(
        np.float32, copy=False
    )
    zeros = np.zeros(points.shape[1], dtype=np.float32)
    arrays = {
        "prediction_m": persistence.copy(),
        "persistence_m": persistence.copy(),
        "driven_readout_m": persistence.copy(),
        "zero_action_readout_m": persistence.copy(),
        "action_support": zeros,
        "frame_zero_points_m": points[0].copy(),
    }
    archive = Path(archive_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **arrays)
    with np.load(graph_path, allow_pickle=False) as graph:
        observed_nodes = int(np.asarray(graph["observed_node_count"]).item())
        total_nodes = len(np.asarray(graph["vertices"]))
    input_paths = {
        "prediction_only_input": prediction_data_path,
        "simulator_final_data": simulator_data_path,
        "episode_graph": graph_path,
        "state_artifact": state_path,
        "twin_summary": twin_summary_path,
        "automatic_twin_log": automatic_twin_log_path,
    }
    summary_record = _bound_file(twin_summary_path)
    log_record = _bound_file(automatic_twin_log_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": frame_zero["object_id"],
        "episode_id": int(frame_zero["episode_id"]),
        "role": role,
        "physical_mode": PHYSICAL_MODE_PERSISTENCE_FALLBACK,
        "physical_admitted": False,
        "fallback_diagnostics": {
            "reason": PERSISTENCE_FALLBACK_REASON,
            "automatic_twin_exit_code": automatic_twin_exit_code,
            "automatic_twin_result_sha256": twin["result_sha256"],
            "automatic_twin_summary_sha256": summary_record["sha256"],
            "automatic_twin_log_sha256": log_record["sha256"],
            "automatic_twin_state_metrics": dict(twin["state_metrics"]),
            "warp_attempted": False,
        },
        "frozen_predictor": {
            "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
            "official_real_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
            "length_scale_m": LENGTH_SCALE_M,
            "action_response": ACTION_RESPONSE,
            "autonomous_drift_response": AUTONOMOUS_DRIFT_RESPONSE,
            "frame_count": FRAME_COUNT,
            "observed_graph_node_count": observed_nodes,
            "total_graph_node_count": total_nodes,
            "point_count": points.shape[1],
            "warp_dynamics": dict(WARP_DYNAMICS),
        },
        "physical_prediction_archive": {
            **_bound_file(archive),
            "array_sha256": {
                name: sha256_array(value) for name, value in arrays.items()
            },
        },
        "input_files": {name: _bound_file(path) for name, path in input_paths.items()},
        "held_lock_sha256": sha256_file(lock_path),
        "frame_zero_manifest_sha256": sha256_file(frame_zero_manifest_path),
        "frame_zero_manifest_artifact_sha256": frame_zero["artifact_sha256"],
        "robot_kinematics_window": robot_kinematics_window,
        "runtime_provenance": dict(runtime_provenance),
        "stage_runtime_seconds": {
            key: float(value) for key, value in stage_runtime_seconds.items()
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_aligned_robot_kinematics_read": True,
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "future_object_visibility_read": False,
            "future_tactile_read": False,
            "external_target_scoring_in_warp": False,
            "outcome_created": False,
            "outcome_read": False,
            "physical_prediction_hashed_before_outcome": True,
        },
        "passed": True,
    }
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    _write_json(manifest_path, manifest)
    return validate_physical_prediction_manifest(manifest_path, verify_archive=True)


def validate_physical_prediction_manifest(
    manifest_path: str | Path,
    *,
    verify_archive: bool = False,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "unsupported prediction schema",
    )
    _require(
        manifest.get("artifact_kind") == ARTIFACT_KIND,
        "unsupported prediction artifact",
    )
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "prediction protocol changed")
    _require(manifest.get("passed") is True, "physical prediction did not pass")
    mode = manifest.get("physical_mode")
    _require(
        mode in {PHYSICAL_MODE_WARP_TWIN, PHYSICAL_MODE_PERSISTENCE_FALLBACK},
        "physical prediction mode changed",
    )
    admitted = manifest.get("physical_admitted")
    _require(
        admitted is (mode == PHYSICAL_MODE_WARP_TWIN),
        "physical admission flag disagrees with prediction mode",
    )
    fallback = manifest.get("fallback_diagnostics")
    if mode == PHYSICAL_MODE_WARP_TWIN:
        _require(fallback is None, "Warp-twin prediction carries fallback diagnostics")
    else:
        _require(
            isinstance(fallback, Mapping)
            and set(fallback)
            == {
                "reason",
                "automatic_twin_exit_code",
                "automatic_twin_result_sha256",
                "automatic_twin_summary_sha256",
                "automatic_twin_log_sha256",
                "automatic_twin_state_metrics",
                "warp_attempted",
            }
            and fallback.get("reason") == PERSISTENCE_FALLBACK_REASON
            and fallback.get("automatic_twin_exit_code")
            == AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE
            and fallback.get("warp_attempted") is False,
            "persistence fallback diagnostics changed",
        )
    frozen = manifest.get("frozen_predictor", {})
    _require(
        frozen.get("official_phystwin_revision") == OFFICIAL_PHYSTWIN_REVISION,
        "revision changed",
    )
    _require(
        frozen.get("official_real_config_sha256") == OFFICIAL_REAL_CONFIG_SHA256,
        "config changed",
    )
    _require(
        float(frozen.get("length_scale_m", -1.0)) == LENGTH_SCALE_M,
        "length scale changed",
    )
    _require(
        float(frozen.get("action_response", -1.0)) == ACTION_RESPONSE,
        "action response changed",
    )
    _require(
        float(frozen.get("autonomous_drift_response", -1.0))
        == AUTONOMOUS_DRIFT_RESPONSE,
        "autonomous drift changed",
    )
    _require(int(frozen.get("frame_count", -1)) == FRAME_COUNT, "frame count changed")
    _require(frozen.get("warp_dynamics") == WARP_DYNAMICS, "Warp dynamics changed")
    _require(
        isinstance(frozen.get("point_count"), int)
        and int(frozen.get("point_count")) >= MINIMUM_NODE_COUNT
        and MINIMUM_NODE_COUNT
        <= int(frozen.get("observed_graph_node_count", -1))
        <= CANONICAL_NODE_COUNT
        and int(frozen.get("total_graph_node_count", -1))
        >= int(frozen.get("observed_graph_node_count", -1)),
        "physical graph capacity changed",
    )
    archive_record = manifest.get("physical_prediction_archive", {})
    archive_path = _validate_bound_file(
        archive_record,
        label="physical prediction archive",
        allow_metadata=True,
    )
    inputs = manifest.get("input_files", {})
    expected_input_roles = (
        {
            "prediction_only_input",
            "simulator_final_data",
            "episode_graph",
            "state_artifact",
            "twin_summary",
            "driven_result",
            "zero_action_result",
            "driven_trajectory",
            "zero_action_trajectory",
        }
        if mode == PHYSICAL_MODE_WARP_TWIN
        else {
            "prediction_only_input",
            "simulator_final_data",
            "episode_graph",
            "state_artifact",
            "twin_summary",
            "automatic_twin_log",
        }
    )
    _require(
        isinstance(inputs, Mapping) and set(inputs) == expected_input_roles,
        "prediction input roles changed",
    )
    for label, record in inputs.items():
        _validate_bound_file(record, label=str(label))
    _validate_controller_kinematics_audit(
        manifest.get("robot_kinematics_window"),
        require_raw_source=True,
    )
    if mode == PHYSICAL_MODE_PERSISTENCE_FALLBACK:
        twin = _validate_inadmissible_automatic_twin(
            inputs["prediction_only_input"]["path"],
            inputs["simulator_final_data"]["path"],
            inputs["episode_graph"]["path"],
            inputs["state_artifact"]["path"],
            inputs["twin_summary"]["path"],
            case_name=str(manifest.get("case_name")),
            object_id=str(manifest.get("object_id")),
            episode_id=int(manifest.get("episode_id", -1)),
            role=str(manifest.get("role")),
        )
        _require(
            fallback.get("automatic_twin_result_sha256") == twin["result_sha256"]
            and fallback.get("automatic_twin_summary_sha256")
            == inputs["twin_summary"]["sha256"]
            and fallback.get("automatic_twin_log_sha256")
            == inputs["automatic_twin_log"]["sha256"]
            and fallback.get("automatic_twin_state_metrics") == twin["state_metrics"],
            "persistence fallback diagnostics do not bind the failed twin",
        )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("known_future_aligned_robot_kinematics_read") is True
        and boundary.get("known_future_robot_action_read") is True
        and boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("future_object_visibility_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("external_target_scoring_in_warp") is False
        and boundary.get("outcome_created") is False
        and boundary.get("outcome_read") is False
        and boundary.get("physical_prediction_hashed_before_outcome") is True,
        "physical prediction crossed its information boundary",
    )
    _require(
        manifest.get("artifact_sha256") == held_artifact_sha256(manifest),
        "physical prediction manifest checksum changed",
    )
    if verify_archive:
        expected = archive_record.get("array_sha256")
        _require(
            isinstance(expected, Mapping) and set(expected) == PHYSICAL_ARCHIVE_ARRAYS,
            "archive array checksums are missing or changed",
        )
        with np.load(archive_path, allow_pickle=False) as stored:
            _require(
                set(stored.files) == PHYSICAL_ARCHIVE_ARRAYS,
                "prediction archive array set changed",
            )
            for name in stored.files:
                _require(
                    sha256_array(stored[name]) == expected[name],
                    f"{name} checksum changed",
                )
            if mode == PHYSICAL_MODE_PERSISTENCE_FALLBACK:
                persistence = np.asarray(stored["persistence_m"])
                frame_zero = np.asarray(stored["frame_zero_points_m"])
                _require(
                    persistence.dtype == np.dtype(np.float32)
                    and persistence.shape
                    == (FRAME_COUNT, int(frozen["point_count"]), 3)
                    and frame_zero.dtype == np.dtype(np.float32)
                    and frame_zero.shape == (int(frozen["point_count"]), 3)
                    and np.array_equal(
                        persistence,
                        np.repeat(frame_zero[None], FRAME_COUNT, axis=0),
                    )
                    and np.array_equal(stored["prediction_m"], persistence)
                    and np.array_equal(stored["driven_readout_m"], persistence)
                    and np.array_equal(stored["zero_action_readout_m"], persistence)
                    and stored["action_support"].dtype == np.dtype(np.float32)
                    and stored["action_support"].shape == (int(frozen["point_count"]),)
                    and np.count_nonzero(stored["action_support"]) == 0,
                    "persistence fallback arrays changed",
                )
    return manifest


def run_held_physical_prior(
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    output_dir: str | Path,
    *,
    case_name: str,
    role: str = "calibration",
    upstream_repo: str | Path,
    official_phystwin_repo: str | Path,
    official_config: str | Path,
    deform360_repo: str | Path,
    python: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run and seal one prediction-only held/calibration physical forecast."""

    # Both authorization and all source/config hashes are checked before a GPU process.
    lock = load_held_protocol_lock(lock_path)
    _require(
        lock["immutable_bindings"]["held_physical_numeric_contract"]
        == held_contract_sha256(HELD_PHYSICAL_NUMERIC_CONTRACT),
        "physical numeric contract differs from the immutable lock",
    )
    _require(
        lock["immutable_bindings"]["upstream_runtime_bundle_tree"]
        == held_contract_sha256(UPSTREAM_RUNTIME_BUNDLE_CONTRACT),
        "upstream runtime bundle differs from the immutable lock",
    )
    for relative_path, binding_key in UPSTREAM_LOCK_BINDING_BY_PATH.items():
        _require(
            lock["immutable_bindings"][binding_key]
            == UPSTREAM_FILE_SHA256[relative_path],
            f"upstream source binding changed: {relative_path}",
        )
    python_runtime = validate_python_runtime(python, lock["immutable_bindings"])
    validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    provenance = validate_upstream_runtime(
        upstream_repo,
        official_phystwin_repo,
        official_config,
        lock["immutable_bindings"],
    )
    validated_runtime = dict(provenance)
    provenance["python_runtime"] = python_runtime
    deform360 = _canonical_directory(
        deform360_repo,
        label="Deform360 import repository",
    )
    provenance["deform360_import_root"] = str(deform360)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prediction_data = root / "prediction_only_input.pkl"
    prediction_summary = root / "prediction_only_input.json"
    build_prediction_only_artifacts(
        frame_zero_manifest_path,
        lock_path,
        prediction_data,
        prediction_summary,
        case_name=case_name,
        role=role,
    )
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path, lock_path
    )
    upstream = Path(provenance["upstream_repository_root"])
    official = Path(provenance["official_phystwin_repository_root"])
    config = Path(provenance["official_config_path"])
    # Do not resolve this path: executing a venv symlink through its base
    # interpreter silently disables the virtualenv's prefix/site-packages.
    python_path = Path(python_runtime["supplied_python_path"])
    _require(
        not os.path.lexists(HELD_PYCACHE_PREFIX),
        "reserved held pycache prefix exists",
    )
    env = dict(os.environ)
    for variable in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
    ):
        env.pop(variable, None)
    env.update(
        {
            "PYNPUT_BACKEND": "dummy",
            "PYOPENGL_PLATFORM": "egl",
            "WANDB_MODE": "disabled",
        }
    )
    graph_path = root / "episode_graph.npz"
    simulator_data = root / "simulator_final_data.pkl"
    state_path = root / "state_artifact.npz"
    twin_summary = root / "twin_summary.json"
    runtimes: dict[str, float] = {}
    import_roots = (upstream / "src", deform360)
    twin_script = upstream / "scripts/remote/build_deform360_automatic_episode_twin.py"
    twin_arguments = [
        "--repo",
        str(upstream),
        "--object-id",
        str(frame_zero["object_id"]),
        "--episode-id",
        str(frame_zero["episode_id"]),
        "--phase",
        "calibration" if role == "calibration" else "source",
        "--episode-final-data",
        str(prediction_data),
        "--episode-graph",
        str(graph_path),
        "--simulator-final-data",
        str(simulator_data),
        "--state-artifact",
        str(state_path),
        "--summary",
        str(twin_summary),
        "--prediction-only-input",
        "--canonical-node-count",
        str(CANONICAL_NODE_COUNT),
    ]
    if role == "calibration":
        twin_arguments.append("--source-admission-passed")
    twin_command = _isolated_runpy_command(
        python_path,
        twin_script,
        import_roots=import_roots,
        arguments=twin_arguments,
    )
    automatic_twin_log = root / "logs/automatic_twin.log"
    try:
        runtimes["automatic_twin"] = _run_logged(
            twin_command,
            env=env,
            log_path=automatic_twin_log,
        )
    except _LoggedCommandError as error:
        if error.returncode != AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE:
            raise
        runtimes["automatic_twin"] = error.elapsed_seconds
        _require(
            validate_upstream_runtime(
                upstream,
                official,
                config,
                lock["immutable_bindings"],
            )
            == validated_runtime,
            "validated runtime changed during automatic-twin execution",
        )
        prediction_archive = root / "prediction.npz"
        physical_manifest = root / "physical_prediction_manifest.json"
        seal_started = time.perf_counter()
        prediction_manifest = build_persistence_fallback_archive(
            prediction_data,
            simulator_data,
            graph_path,
            state_path,
            twin_summary,
            automatic_twin_log,
            prediction_archive,
            physical_manifest,
            frame_zero_manifest_path=frame_zero_manifest_path,
            lock_path=lock_path,
            case_name=case_name,
            role=role,
            automatic_twin_exit_code=error.returncode,
            runtime_provenance=provenance,
            stage_runtime_seconds=runtimes,
        )
        runtimes["prediction_seal"] = time.perf_counter() - seal_started
        physical_seal_path = root / "physical_prior_seal.json"
        physical_seal = create_physical_prior_seal(
            physical_seal_path,
            lock_path,
            frame_zero_manifest_path,
            {
                "prediction_only_input": prediction_data,
                "prediction_only_summary": prediction_summary,
                "physical_prediction_archive": prediction_archive,
                "physical_prediction_manifest": physical_manifest,
            },
            case_name=case_name,
            role=role,
        )
        return {
            "case_name": case_name,
            "role": role,
            "physical_prediction_manifest": prediction_manifest,
            "physical_prior_seal": physical_seal,
            "runtime_seconds": runtimes,
        }

    smoke_script = upstream / "scripts/remote/run_deform360_official_phystwin_smoke.py"
    split_path = (
        upstream / "configs/causal4d_public/deform360_independent_source_split_v1.json"
    )
    result_paths: dict[str, Path] = {}
    for label, scale in (("driven", 1.0), ("zero_action", 0.0)):
        rollout_dir = root / f"warp_{label}"
        smoke_arguments = [
            "--official-phystwin-repo",
            str(official),
            "--data",
            str(simulator_data),
            "--config",
            str(config),
            "--split-json",
            str(split_path),
            "--output-dir",
            str(rollout_dir),
            "--canonical-reusable-graph",
            str(graph_path),
            "--device",
            device,
            "--controller-radius-m",
            str(WARP_DYNAMICS["controller_radius_m"]),
            "--controller-max-neighbours",
            str(WARP_DYNAMICS["controller_max_neighbours"]),
            "--canonical-controller-patch-size",
            str(WARP_DYNAMICS["canonical_controller_patch_size"]),
            "--init-spring-y",
            str(WARP_DYNAMICS["init_spring_y"]),
            "--drag-damping",
            str(WARP_DYNAMICS["drag_damping"]),
            "--dashpot-damping",
            str(WARP_DYNAMICS["dashpot_damping"]),
            "--controller-displacement-scale",
            str(scale),
            "--support-dynamics",
            str(WARP_DYNAMICS["support_dynamics"]),
            "--report-edge-strain",
        ]
        command = _isolated_runpy_command(
            python_path,
            smoke_script,
            import_roots=import_roots,
            arguments=smoke_arguments,
            provenance_root=official,
        )
        runtimes[f"warp_{label}"] = _run_logged(
            command,
            env=env,
            log_path=root / f"logs/warp_{label}.log",
        )
        result_paths[label] = rollout_dir / "official_phystwin_smoke.json"

    _require(
        validate_upstream_runtime(
            upstream,
            official,
            config,
            lock["immutable_bindings"],
        )
        == validated_runtime,
        "validated runtime changed during official PhysTwin execution",
    )

    prediction_archive = root / "prediction.npz"
    physical_manifest = root / "physical_prediction_manifest.json"
    seal_started = time.perf_counter()
    prediction_manifest = build_physical_prediction_archive(
        prediction_data,
        simulator_data,
        graph_path,
        state_path,
        twin_summary,
        result_paths["driven"],
        result_paths["zero_action"],
        prediction_archive,
        physical_manifest,
        frame_zero_manifest_path=frame_zero_manifest_path,
        lock_path=lock_path,
        case_name=case_name,
        role=role,
        runtime_provenance=provenance,
        stage_runtime_seconds=runtimes,
    )
    runtimes["prediction_seal"] = time.perf_counter() - seal_started
    physical_seal_path = root / "physical_prior_seal.json"
    physical_seal = create_physical_prior_seal(
        physical_seal_path,
        lock_path,
        frame_zero_manifest_path,
        {
            "prediction_only_input": prediction_data,
            "prediction_only_summary": prediction_summary,
            "physical_prediction_archive": prediction_archive,
            "physical_prediction_manifest": physical_manifest,
        },
        case_name=case_name,
        role=role,
    )
    return {
        "case_name": case_name,
        "role": role,
        "physical_prediction_manifest": prediction_manifest,
        "physical_prior_seal": physical_seal,
        "runtime_seconds": runtimes,
    }


__all__ = [
    "ACTION_RESPONSE",
    "ARTIFACT_KIND",
    "AUTONOMOUS_DRIFT_RESPONSE",
    "CANONICAL_NODE_COUNT",
    "HELD_PHYSICAL_NUMERIC_CONTRACT",
    "HELD_PYCACHE_PREFIX",
    "HELD_PYTHON_RUNTIME",
    "HELD_PYTHON_RUNTIME_MANIFEST",
    "LENGTH_SCALE_M",
    "OFFICIAL_PHYSTWIN_REVISION",
    "OFFICIAL_REAL_CONFIG_SHA256",
    "PHYSICAL_MODE_PERSISTENCE_FALLBACK",
    "PHYSICAL_MODE_WARP_TWIN",
    "UPSTREAM_FILE_SHA256",
    "UPSTREAM_LOCK_BINDING_BY_PATH",
    "UPSTREAM_RUNTIME_BUNDLE_CONTRACT",
    "WARP_DYNAMICS",
    "build_persistence_fallback_archive",
    "build_physical_prediction_archive",
    "build_prediction_only_artifacts",
    "load_controller_trajectory",
    "run_held_physical_prior",
    "sha256_array",
    "sha256_file",
    "validate_physical_prediction_manifest",
    "validate_python_runtime",
    "validate_upstream_runtime",
]
