#!/usr/bin/env python3
"""Materialize one registered Deform360 source episode without target access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, TextIO

OFFICIAL_REVISION = "d8522a4403b766aeb387510c04e89032a56fdf35"
SOURCE_OBJECT = "038-mat-cloth"
SOURCE_EPISODE = 3
TARGET_EPISODE = 9
MARKER_NAME = ".bpt-source-processing-v1.json"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(
    path: Path, *, hash_limit: int = 64 * 1024 * 1024
) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= hash_limit:
        result["sha256"] = sha256(path)
    return result


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    environment: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def append_environment(handle: TextIO, name: str, value: Any) -> None:
    handle.write(f"{name}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--official-source", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    return parser.parse_args()


def validate_request(
    request: dict[str, Any], raw_root: Path, processed_root: Path
) -> tuple[Path, Path]:
    raw_object = raw_root / "raw" / SOURCE_OBJECT
    output_object = processed_root / "cross-action-source-v1" / SOURCE_OBJECT
    required = {
        "schema": "deform360-source-pilot-materialization-request-v2",
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "raw_object": str(raw_object),
        "output_object": str(output_object),
        "source_object": SOURCE_OBJECT,
        "source_episode": SOURCE_EPISODE,
        "runner_label": "gpuserver4090",
        "official_processing_revision": OFFICIAL_REVISION,
        "stages": ["undistort", "tactile", "robot"],
        "source_only": True,
        "target_future_authorized": False,
        "persistent_processed_write_authorized": True,
    }
    for key, expected in required.items():
        if request.get(key) != expected:
            raise ValueError(f"invalid request field {key!r}")
    allowed = set(required) | {"request_id", "expected_source_revision"}
    if set(request) != allowed:
        raise ValueError("request keys differ from the registered contract")
    if not isinstance(request.get("request_id"), str) or not request["request_id"]:
        raise ValueError("request_id must be a nonempty string")
    revision = request.get("expected_source_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("expected_source_revision must be a full commit SHA")
    return raw_object, output_object


def source_camera_plan(raw_object: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for camera in sorted(
        path
        for path in raw_object.iterdir()
        if path.is_dir()
        and path.name.startswith("brics-odroid-")
        and "tactile" not in path.name
    ):
        videos = sorted(camera.glob("*.mp4"))
        timestamps = {path.stem: path for path in camera.glob("*.txt")}
        paired = [path for path in videos if path.stem in timestamps]
        if SOURCE_EPISODE >= len(paired):
            continue
        video = paired[SOURCE_EPISODE]
        timestamp = timestamps[video.stem]
        total_bytes += video.stat().st_size
        rows.append(
            {
                "camera": camera.name,
                "video": str(video),
                "video_size_bytes": video.stat().st_size,
                "timestamps": str(timestamp),
                "timestamp_size_bytes": timestamp.stat().st_size,
            }
        )
    return rows, total_bytes


def prepare_output(
    raw_object: Path,
    output_object: Path,
    metadata_bytes: bytes,
) -> dict[str, Any]:
    marker_payload = {
        "schema": "bayesian-phystwin/deform360-source-processing-marker-v1",
        "source_object": SOURCE_OBJECT,
        "source_episode": SOURCE_EPISODE,
        "raw_object": str(raw_object),
        "output_object": str(output_object),
        "official_processing_revision": OFFICIAL_REVISION,
        "raw_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "target_future_authorized": False,
    }
    marker = output_object / MARKER_NAME
    if output_object.exists():
        if not marker.is_file():
            raise RuntimeError("output exists without the exact custody marker")
        if json.loads(marker.read_text(encoding="utf-8")) != marker_payload:
            raise RuntimeError("existing output custody marker differs")
    else:
        output_object.mkdir(parents=True)
        write_json(marker, marker_payload)
    target_dir = output_object / f"episode_{TARGET_EPISODE:04d}"
    if target_dir.exists():
        raise RuntimeError("target output directory already exists in source custody root")
    return marker_payload


def inventory_outputs(output_object: Path) -> dict[str, Any]:
    episode = output_object / f"episode_{SOURCE_EPISODE:04d}"
    target = output_object / f"episode_{TARGET_EPISODE:04d}"
    cameras: list[dict[str, Any]] = []
    tactile: list[dict[str, Any]] = []
    if episode.is_dir():
        for directory in sorted(path for path in episode.iterdir() if path.is_dir()):
            video = directory / "undistorted.mp4"
            if video.is_file():
                cameras.append(
                    {
                        "camera": directory.name,
                        "video_size_bytes": video.stat().st_size,
                        "metadata": file_record(directory / "metadata.json"),
                        "alignment": file_record(directory / "alignment.json"),
                        "timestamps": file_record(
                            directory / "aligned_timestamps.txt"
                        ),
                    }
                )
            synced = directory / "synced_tactile.npy"
            if synced.is_file():
                tactile.append(
                    {
                        "sensor": directory.name,
                        "size_bytes": synced.stat().st_size,
                        "metadata": file_record(directory / "metadata.json"),
                        "alignment": file_record(directory / "alignment.json"),
                    }
                )
    return {
        "episode_dir_exists": episode.is_dir(),
        "episode_alignment": file_record(episode / "alignment.json"),
        "intrinsics": file_record(episode / "undistorted_intrinsics.npy"),
        "extrinsics": file_record(episode / "extrinsics.npy"),
        "camera_count": len(cameras),
        "cameras": cameras,
        "tactile_sensor_count": len(tactile),
        "tactile": tactile,
        "robot": file_record(episode / "robot" / "robot.npz"),
        "robot_metadata": file_record(episode / "robot" / "robot.meta.json"),
        "target_output_dir_exists": target.exists(),
    }


def write_report(result: dict[str, Any], evidence_root: Path) -> None:
    outputs = result["outputs"]
    lines = [
        "# Deform360 source pilot materialization v2",
        "",
        f"Decision: `{result['decision']}`",
        f"Aligned cameras: `{outputs['camera_count']}`",
        f"Aligned tactile sensors: `{outputs['tactile_sensor_count']}`",
        f"Robot state present: `{outputs['robot'] is not None}`",
        f"Target output directory created: `{outputs['target_output_dir_exists']}`",
        "",
        "Only source object `038-mat-cloth`, episode `3`, was requested.",
        "No target episode was opened or scored.",
        "",
    ]
    (evidence_root / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve(strict=True)
    processed_root = args.processed_root.expanduser().resolve(strict=True)
    official_source = args.official_source.expanduser().resolve(strict=True)
    evidence_root = args.evidence_root.expanduser().resolve()
    if evidence_root.exists():
        raise FileExistsError(f"evidence directory exists: {evidence_root}")
    evidence_root.mkdir(parents=True)

    request = json.loads(args.request.read_text(encoding="utf-8"))
    raw_object, output_object = validate_request(request, raw_root, processed_root)
    if not raw_object.is_dir() or not os.access(raw_object, os.R_OK):
        raise RuntimeError("registered raw source object is unavailable")
    if not os.access(processed_root, os.W_OK):
        raise RuntimeError("processed root is not writable")
    if not output_object.is_relative_to(processed_root):
        raise RuntimeError("output object escapes processed root")
    if git_revision(official_source) != OFFICIAL_REVISION:
        raise RuntimeError("official Deform360 checkout revision differs")

    metadata_path = raw_object / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    sequence = metadata.get("sequences", {}).get(str(SOURCE_EPISODE))
    if not isinstance(sequence, dict):
        raise RuntimeError("source episode is absent from metadata.json")

    camera_plan, source_video_bytes = source_camera_plan(raw_object)
    if len(camera_plan) < 8:
        raise RuntimeError(
            f"source episode has only {len(camera_plan)} exact camera pairs"
        )
    free_before = shutil.disk_usage(processed_root).free
    required_bytes = max(50 * 1024**3, 2 * source_video_bytes + 10 * 1024**3)
    if free_before < required_bytes:
        raise RuntimeError(f"insufficient free space: {free_before} < {required_bytes}")
    marker = prepare_output(raw_object, output_object, metadata_bytes)

    runtime_root = evidence_root.parent / f"{evidence_root.name}-venv"
    if runtime_root.exists():
        raise FileExistsError(f"runtime directory exists: {runtime_root}")
    plan = {
        **marker,
        "request": request,
        "camera_pair_count": len(camera_plan),
        "source_video_bytes": source_video_bytes,
        "processed_free_bytes_before": free_before,
        "required_free_bytes": required_bytes,
        "camera_pairs": camera_plan,
        "runtime_root": str(runtime_root),
    }
    write_json(evidence_root / "plan.json", plan)

    environment_path = evidence_root / "environment.txt"
    with environment_path.open("w", encoding="utf-8") as handle:
        append_environment(handle, "repository", os.environ.get("GITHUB_REPOSITORY"))
        append_environment(handle, "revision", os.environ.get("GITHUB_SHA"))
        append_environment(handle, "runner_name", os.environ.get("RUNNER_NAME"))
        append_environment(handle, "runner_os", os.environ.get("RUNNER_OS"))
        append_environment(handle, "runner_arch", os.environ.get("RUNNER_ARCH"))
        append_environment(handle, "required_runner_label", "gpuserver4090")
        append_environment(handle, "official_revision", OFFICIAL_REVISION)
        append_environment(handle, "python", platform.python_version())
        append_environment(handle, "raw_object", raw_object)
        append_environment(handle, "output_object", output_object)
        append_environment(handle, "source_episode", SOURCE_EPISODE)

    stage_codes: dict[str, int | None] = {
        "runtime": None,
        "undistort": None,
        "tactile": None,
        "robot": None,
    }
    errors: list[str] = []
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(runtime_root)
        runtime_python = runtime_root / "bin" / "python"
        runtime_bin = runtime_root / "bin"
        install_log = evidence_root / "runtime-bootstrap.log"
        commands = [
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-e",
                str(official_source),
            ],
            [str(runtime_python), "-m", "pip", "check"],
            [
                str(runtime_python),
                "-c",
                (
                    "import cv2, deform360, numpy; "
                    "assert hasattr(cv2, 'aruco'); "
                    "print(deform360.__file__); "
                    "print(numpy.__version__); print(cv2.__version__)"
                ),
            ],
        ]
        install_log.write_text("", encoding="utf-8")
        for command in commands:
            temporary = evidence_root / ".install-part.log"
            code = run_logged(command, temporary)
            with install_log.open("a", encoding="utf-8") as destination:
                destination.write(temporary.read_text(encoding="utf-8"))
            temporary.unlink(missing_ok=True)
            if code != 0:
                stage_codes["runtime"] = code
                raise RuntimeError(f"runtime bootstrap failed with exit code {code}")
        stage_codes["runtime"] = 0

        common_environment = dict(os.environ)
        common_environment["PYTHONHASHSEED"] = "0"
        common_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        common_environment["OPENBLAS_NUM_THREADS"] = "1"
        common_environment["OMP_NUM_THREADS"] = "1"
        common_environment["MKL_NUM_THREADS"] = "1"
        common_environment["NUMEXPR_NUM_THREADS"] = "1"

        stage_codes["undistort"] = run_logged(
            [
                str(runtime_bin / "deform360-undistort"),
                "--object-dir",
                str(raw_object),
                "--output-dir",
                str(output_object),
                "--episodes",
                str(SOURCE_EPISODE),
                "--no-overwrite",
            ],
            evidence_root / "undistort.log",
            environment=common_environment,
        )
        if stage_codes["undistort"] == 0:
            stage_codes["tactile"] = run_logged(
                [
                    str(runtime_bin / "deform360-process-tactile"),
                    "--object-dir",
                    str(raw_object),
                    "--aligned-dir",
                    str(output_object),
                    "--episodes",
                    str(SOURCE_EPISODE),
                    "--no-overwrite",
                ],
                evidence_root / "tactile.log",
                environment=common_environment,
            )
            robot_driver = evidence_root / "robot_driver.py"
            robot_driver.write_text(
                "\n".join(
                    [
                        "import json",
                        "from pathlib import Path",
                        "from deform360.processing.robot_stage import process_robot_episode",
                        f"raw = Path({str(raw_object)!r})",
                        f"aligned = Path({str(output_object)!r})",
                        f"episode = {SOURCE_EPISODE}",
                        "metadata = json.loads((raw / 'metadata.json').read_text())",
                        "sequence = metadata.get('sequences', {}).get(str(episode), {})",
                        "bimanual = sequence.get('bimanual', 'no') == 'yes'",
                        "result = process_robot_episode(",
                        "    aligned, episode, bimanual=bimanual, cameras=None,",
                        "    seed=0, overwrite=False, plot=False,",
                        ")",
                        "print(result)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            stage_codes["robot"] = run_logged(
                [str(runtime_python), str(robot_driver)],
                evidence_root / "robot.log",
                environment=common_environment,
            )
        else:
            errors.append("undistortion failed; tactile and robot stages were skipped")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        outputs = inventory_outputs(output_object)
        target_closed = outputs["target_output_dir_exists"] is False
        base_complete = (
            stage_codes["runtime"] == 0
            and stage_codes["undistort"] == 0
            and stage_codes["robot"] == 0
            and int(outputs["camera_count"]) >= 8
            and outputs["robot"] is not None
            and target_closed
        )
        tactile_complete = stage_codes["tactile"] == 0
        if base_complete and tactile_complete:
            decision = "source-base-and-tactile-materialization-complete"
        elif base_complete:
            decision = "source-base-materialization-complete-tactile-unavailable"
        else:
            decision = "source-base-materialization-incomplete"
        result = {
            "schema": "deform360-source-pilot-materialization-result-v2",
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "revision": os.environ.get("GITHUB_SHA"),
            "workflow_run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
            "workflow_run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "required_runner_label": "gpuserver4090",
            "official_processing_revision": OFFICIAL_REVISION,
            "source_object": SOURCE_OBJECT,
            "source_episode": SOURCE_EPISODE,
            "raw_object": str(raw_object),
            "output_object": str(output_object),
            "stage_exit_codes": stage_codes,
            "outputs": outputs,
            "errors": errors,
            "processed_free_bytes_after": shutil.disk_usage(processed_root).free,
            "decision": decision,
            "information_boundary": {
                "source_episode_payload_opened": stage_codes["undistort"] is not None,
                "persistent_source_outputs_written": bool(outputs["episode_dir_exists"]),
                "target_episode_requested_by_driver": False,
                "target_output_directory_created": bool(
                    outputs["target_output_dir_exists"]
                ),
                "target_numeric_payload_opened": False,
                "target_scoring_performed": False,
                "fresh_confirmation_authorized": False,
                "paper_claim_authorized": False,
            },
            "claim_boundary": (
                "Source-only preprocessing evidence for one public Deform360 "
                "episode. Not geometry reconstruction, Prob4D qualification, "
                "BayesianPhysTwin benefit, cross-action transport, Causal4D "
                "decision value, calibration, or a paper claim."
            ),
        }
        canonical = json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
        write_json(evidence_root / "result.json", result)
        write_report(result, evidence_root)
        print(
            json.dumps(
                {
                    "decision": decision,
                    "stage_exit_codes": stage_codes,
                    "camera_count": outputs["camera_count"],
                    "tactile_sensor_count": outputs["tactile_sensor_count"],
                    "robot_present": outputs["robot"] is not None,
                    "target_output_directory_created": outputs[
                        "target_output_dir_exists"
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
    return 0 if base_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
