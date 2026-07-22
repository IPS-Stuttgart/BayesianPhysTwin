#!/usr/bin/env python3
"""Replay the v8.1 object-072 admission wrapper without outcome access.

This development-only operator reads the already sealed attempt-2
``prediction_only_input.pkl`` and executes only the automatic-twin builder.
It never loads an official target, an x0 query, a queried prediction, a score,
or a confirmation artifact.  A second invocation with the wrong episode must
be rejected before producing numerical outputs.

The resulting report and source binding are inputs to the attempt-4 lock
preparer.  Run this operator from an exact clean Git checkout on workstation2.
"""

from __future__ import annotations

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

from bayesian_phystwin import deform360_held_v8_builders as builders
from bayesian_phystwin import deform360_held_v8_protocol as protocol


ROOT = Path(
    "/mnt/corsair/florianpfaff/"
    "bpt-held-v8.1-attempt-4-admission-wrapper-scratch-20260722"
)
SOURCE_INPUT = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/"
    "held-v8-attempt-2-withdrawn-preoutcome/calibration/cases/"
    "072-cotton-clohesline-ep0003/physical/prediction_only_input.pkl"
)
SOURCE_INPUT_SHA256 = (
    "2f783d15426759a0928fcb6cb8a98fa61b38d582a46ec006d296d53b439ae015"
)
SOURCE_INPUT_SIZE_BYTES = 19_261_048
PINNED_PYTHON = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004/"
    "bin/python"
)
UPSTREAM = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "Bayesian-PhysTwin-upstream-58ab4808e59d"
)
DEFORM360 = Path("/mnt/lexar4tb/datasets/deform360/code")
CASE_NAME = "072-cotton-clohesline-ep0003"
OBJECT_ID = "072-cotton-clohesline"
EPISODE_ID = 3
WRONG_EPISODE_ID = 4
REPORT_NAME = "metadata-only-replay-report.json"
CODE_BINDING_NAME = "metadata-only-replay-code-binding.json"


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
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_git(root: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
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


def _source_binding() -> dict[str, Any]:
    code = Path(__file__).resolve().parents[2]
    _require(Path(__file__).resolve().is_relative_to(code), "operator is outside code")
    _require(_run_git(code, ["status", "--porcelain=v1", "--untracked-files=all"]) == "", "source checkout is dirty")
    head = _run_git(code, ["rev-parse", "HEAD"])
    protocol_path = code / "src/bayesian_phystwin/deform360_held_v8_protocol.py"
    adapter_path = code / "src/bayesian_phystwin/deform360_held_v8_builders.py"
    operator_path = Path(__file__).resolve()
    bindings = {
        "git_head": head,
        "protocol_source_sha256": hashlib.sha256(
            _read_regular(protocol_path, role="v8.1 protocol source")
        ).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(
            _read_regular(adapter_path, role="v8.1 adapter source")
        ).hexdigest(),
        "replay_operator_source_sha256": hashlib.sha256(
            _read_regular(operator_path, role="admission replay operator")
        ).hexdigest(),
        "exact_child_bootstrap_sha256": hashlib.sha256(
            builders._V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP.encode("utf-8")
        ).hexdigest(),
        "uncommitted_correction_present": False,
    }
    _require(
        Path(protocol.__file__).resolve() == protocol_path
        and Path(builders.__file__).resolve() == adapter_path,
        "imported v8.1 source is outside the exact checkout",
    )
    return bindings


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
    script = UPSTREAM / "scripts/remote/build_deform360_automatic_episode_twin.py"
    baseline = builders._V7_PHYSICAL_ISOLATED_RUNPY_COMMAND(
        PINNED_PYTHON,
        script,
        import_roots=(UPSTREAM / "src", DEFORM360),
        arguments=_arguments(root, episode_id=episode_id),
    )
    frozen = builders.physical._ISOLATED_RUNPY_BOOTSTRAP
    _require(baseline.count(frozen) == 1, "frozen child bootstrap changed")
    return [
        builders._V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP if value == frozen else value
        for value in baseline
    ]


def _environment() -> dict[str, str]:
    result = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
    ):
        result.pop(name, None)
    result.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "PYNPUT_BACKEND": "dummy",
            "PYOPENGL_PLATFORM": "egl",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    return result


def _run_child(root: Path, *, episode_id: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        _command(root, episode_id=episode_id),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_environment(),
    )


def _validate_successful_replay(outputs: Mapping[str, Path]) -> dict[str, Any]:
    summary = builders.physical._load_json(outputs["twin_summary.json"])
    boundary = summary.get("information_boundary")
    metrics = summary.get("state_metrics")
    _require(
        summary.get("schema_version") == 1
        and summary.get("artifact_kind") == "Deform360AutomaticEpisodeTwin"
        and summary.get("protocol_id")
        == builders.V8_EXTERNAL_ADMISSION_PROTOCOL_ID
        and summary.get("protocol_config_sha256")
        == builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
        and summary.get("object_id") == OBJECT_ID
        and int(summary.get("episode_id", -1)) == EPISODE_ID
        and summary.get("phase") == "calibration"
        and summary.get("passed") is True
        and summary.get("result_sha256")
        == builders.physical._upstream_result_sha256(summary),
        "successful replay summary identity changed",
    )
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("object_observation_frames_used") == [0]
        and boundary.get("post_initial_object_observation_used") is False
        and boundary.get("target_access") is False
        and boundary.get("prediction_only_input_required") is True
        and boundary.get("future_object_tracks_present") is False,
        "successful replay crossed its observation boundary",
    )
    _require(
        isinstance(metrics, Mapping)
        and metrics.get("passed") is True
        and metrics.get("finite") is True,
        "successful replay did not pass state admission",
    )
    expected_outputs = {
        "episode_graph": builders.physical.sha256_file(outputs["episode_graph.npz"]),
        "simulator_final_data": builders.physical.sha256_file(
            outputs["simulator_final_data.pkl"]
        ),
        "state_artifact": builders.physical.sha256_file(outputs["state_artifact.npz"]),
    }
    _require(
        summary.get("input_sha256", {}).get("episode_final_data")
        == SOURCE_INPUT_SHA256
        and summary.get("output_sha256") == expected_outputs,
        "successful replay input/output binding changed",
    )
    return summary


def _seal_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in files:
            os.chmod(current_path / name, 0o400, follow_symlinks=False)
        for name in directories:
            os.chmod(current_path / name, 0o500, follow_symlinks=False)
    os.chmod(root, 0o500, follow_symlinks=False)


def main() -> int:
    _require(socket.gethostname() == "workstation2", "replay must run on workstation2")
    _require(protocol.PROTOCOL_ID == "deform360-held-online-belief-v8.1", "protocol identity changed")
    _require(protocol.EXECUTION_ATTEMPT == 4, "execution attempt changed")
    _require(not os.path.lexists(ROOT), "admission replay root already exists")
    stage = ROOT.parent / f".{ROOT.name}.stage-{os.getpid()}"
    _require(not os.path.lexists(stage), "admission replay stage already exists")
    source_payload = _read_regular(SOURCE_INPUT, role="prediction-only input", mode=0o400)
    _require(
        len(source_payload) == SOURCE_INPUT_SIZE_BYTES
        and hashlib.sha256(source_payload).hexdigest() == SOURCE_INPUT_SHA256,
        "prediction-only input changed",
    )
    source_binding = _source_binding()
    stage.mkdir(mode=0o700)
    cross = stage / "cross-auth"
    cross.mkdir(mode=0o700)
    try:
        success = _run_child(stage, episode_id=EPISODE_ID)
        _write_new(stage / "stdout.log", success.stdout)
        _write_new(stage / "stderr.log", success.stderr)
        _require(success.returncode == 0, "exact-case admission replay failed")
        output_paths = {
            name: stage / name
            for name in (
                "episode_graph.npz",
                "simulator_final_data.pkl",
                "state_artifact.npz",
                "twin_summary.json",
            )
        }
        _require(all(path.is_file() for path in output_paths.values()), "replay output is absent")
        validated = _validate_successful_replay(output_paths)

        rejected = _run_child(cross, episode_id=WRONG_EPISODE_ID)
        _write_new(cross / "stdout.log", rejected.stdout)
        _write_new(cross / "stderr.log", rejected.stderr)
        numerical_names = tuple(output_paths)
        cross_outputs = [name for name in numerical_names if (cross / name).exists()]
        _require(rejected.returncode != 0 and not cross_outputs, "wrong-case admission was not rejected cleanly")

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
                "protocol_id": protocol.PROTOCOL_ID,
                "execution_attempt": protocol.EXECUTION_ATTEMPT,
                "case_name": CASE_NAME,
                "role": "calibration",
                "development_replay_only": True,
                "formal_outcome_evidence": False,
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
                    "protocol_id": builders.V8_EXTERNAL_ADMISSION_PROTOCOL_ID,
                    "contract_sha256": builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256,
                    "exact_case_only": True,
                    "target_access": False,
                },
                "successful_replay": {
                    "exit_code": success.returncode,
                    "hook_restoration_guard_completed": True,
                    "validator_result_sha256": validated["result_sha256"],
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
                "protocol_id": protocol.PROTOCOL_ID,
                "execution_attempt": protocol.EXECUTION_ATTEMPT,
                "admission_contract_sha256": builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256,
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
        os.rename(stage, ROOT)
        _seal_tree(ROOT)
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
