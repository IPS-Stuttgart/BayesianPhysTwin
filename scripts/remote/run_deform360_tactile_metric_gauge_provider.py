#!/usr/bin/env python3
"""Run frozen supplemental MotionCrafter jobs for the tactile gauge smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_tactile_metric_gauge import (
    load_tactile_metric_gauge_lock,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean(root: Path, *, name: str) -> str:
    head = _git_head(root)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError(f"{name} checkout is dirty")
    return head


def _require_revision(root: Path, expected: str, *, name: str) -> None:
    head = _require_clean(root, name=name)
    if head != expected:
        raise ValueError(f"{name} checkout is at {head}, expected {expected}")


def _require_ancestor(root: Path, expected: str, *, name: str) -> str:
    head = _require_clean(root, name=name)
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, "HEAD"],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{name} does not contain frozen revision {expected}")
    return head


def _safe_member(root: Path, relative: str) -> Path:
    candidate = (root.resolve() / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes root: {relative}") from error
    return candidate


def _model_source(source: Mapping[str, Any]) -> tuple[str, str | None]:
    if source.get("kind") != "huggingface_revision":
        raise ValueError("supplement runner requires portable Hugging Face sources")
    return str(source["repository"]), str(source["revision"])


def _build_pinned_runner(model_set: Any, runner_type: Any, config: Any) -> Any:
    """Build inference through the inspected model set's pinned adapter."""

    return runner_type(config, adapter_factory=model_set.adapter_factory())


def _validate_source(processed_root: Path, job: Mapping[str, Any]) -> Path:
    record = job["source_video"]
    path = _safe_member(processed_root, str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError("source video byte count changed")
    if _sha256(path) != record["sha256"]:
        raise ValueError("source video digest changed")
    return path


def _validate_prediction(
    path: Path,
    *,
    job: Mapping[str, Any],
    configuration: Mapping[str, Any],
    verify_manifest: Any,
) -> dict[str, Any]:
    verification = verify_manifest(path, verify_hashes=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("prediction manifest lacks config")
    expected = {
        "model_type": configuration["model_type"],
        "height": configuration["height"],
        "width": configuration["width"],
        "window_size": configuration["window_size"],
        "overlap": configuration["overlap"],
        "num_inference_steps": configuration["num_inference_steps"],
        "guidance_scale": configuration["guidance_scale"],
        "decode_chunk_size": configuration["decode_chunk_size"],
        "seed": configuration["seed"],
        "seed_policy": configuration["seed_policy"],
        "low_memory_usage": configuration["low_memory_usage"],
        "frame_start": job["source_frame_start"],
        "frame_stop": job["source_frame_stop_exclusive"],
        "frame_stride": configuration["frame_stride"],
        "model_source_set_sha256": configuration["model_source_set_sha256"],
    }
    if {key: config.get(key) for key in expected} != expected:
        raise ValueError("supplement prediction configuration changed")
    schedule = payload.get("stochastic_seed_schedule")
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("calls") != job["seed_schedule"]
    ):
        raise ValueError("supplement prediction seed schedule changed")
    expected_windows = [
        {
            "window_id": item["window_id"],
            "path": f"windows/{item['window_id']}.npz",
            "start_frame": item["source_frame_start"],
            "stop_frame": item["source_frame_stop_exclusive"],
        }
        for item in job["windows"]
    ]
    if payload.get("overlap_windows") != expected_windows:
        raise ValueError("supplement prediction windows changed")
    return {
        "job_id": job["job_id"],
        "prediction_manifest": str(path.resolve()),
        "prediction_manifest_sha256": _sha256(path),
        "verification": verification,
    }


def _worker_command(
    args: argparse.Namespace,
    *,
    job_id: str,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--lock",
        str(args.lock.resolve()),
        "--processed-root",
        str(args.processed_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--prob4d-root",
        str(args.prob4d_root.resolve()),
        "--motioncrafter-root",
        str(args.motioncrafter_root.resolve()),
        "--cache-dir",
        str(args.cache_dir.resolve()),
        "--repository-root",
        str(args.repository_root.resolve()),
        "--worker-job-id",
        job_id,
    ]
    if resume:
        command.append("--resume")
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prob4d-root", type=Path, required=True)
    parser.add_argument("--motioncrafter-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-job-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    lock = load_tactile_metric_gauge_lock(args.lock)
    implementation = lock["implementation"]
    assert isinstance(implementation, Mapping)
    runner_source = Path(__file__).resolve()
    if _sha256(runner_source) != implementation["runner_source_sha256"]:
        raise ValueError("executed runner differs from the frozen source")
    runtime_revision = _require_ancestor(
        args.repository_root.resolve(),
        str(implementation["revision"]),
        name="Bayesian-PhysTwin",
    )
    provider = lock["provider"]
    assert isinstance(provider, Mapping)
    provider_lock = provider["provider_lock"]
    motioncrafter = provider["motioncrafter"]
    assert isinstance(provider_lock, Mapping)
    assert isinstance(motioncrafter, Mapping)
    _require_revision(
        args.prob4d_root.resolve(),
        str(provider_lock["provider_revision"]),
        name="Prob4D",
    )
    _require_revision(
        args.motioncrafter_root.resolve(),
        str(motioncrafter["revision"]),
        name="MotionCrafter",
    )
    jobs = list(lock["supplemental_jobs"])
    if args.worker_job_id is not None:
        jobs = [job for job in jobs if job["job_id"] == args.worker_job_id]
        if len(jobs) != 1:
            raise ValueError("worker job is not unique")

    if args.worker_job_id is None:
        completed: list[dict[str, Any]] = []
        args.output_root.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            output = _safe_member(args.output_root, str(job["output_relative_path"]))
            existing = output.exists() and any(output.iterdir())
            if existing and not args.resume:
                raise ValueError(f"supplement output exists: {output}")
            subprocess.run(
                _worker_command(args, job_id=str(job["job_id"]), resume=existing),
                check=True,
            )
            manifest = output / "predictions.json"
            completed.append(
                {
                    "job_id": job["job_id"],
                    "prediction_manifest": str(manifest.resolve()),
                    "prediction_manifest_sha256": _sha256(manifest),
                }
            )
            report_descriptor = {
                "schema": "bayesian-phystwin.deform360-tactile-gauge-provider-run",
                "schema_version": 1,
                "lock_id": lock["artifact_id"],
                "runtime_revision": runtime_revision,
                "status": "complete" if len(completed) == len(jobs) else "incomplete",
                "requested_job_count": len(jobs),
                "completed_job_count": len(completed),
                "completed_jobs": completed,
                "information_boundary": lock["information_boundary"],
                "claim_boundary": lock["claim_boundary"],
            }
            report = {"run_id": content_id(report_descriptor), **report_descriptor}
            write_atomic_json(
                report,
                args.output_root / "run_report.json",
                overwrite=True,
            )
            print(f"completed {len(completed)}/{len(jobs)} {job['camera']}", flush=True)
        return 0

    sys.path.insert(0, str(args.prob4d_root.resolve() / "src"))
    from prob4d.motioncrafter_integrity import (  # noqa: PLC0415
        verify_motioncrafter_prediction_manifest,
    )
    from prob4d.motioncrafter_models import (  # noqa: PLC0415
        PinnedMotionCrafterModelSet,
    )
    from prob4d.motioncrafter_runner import SafeMotionCrafterRunner  # noqa: PLC0415

    model_manifest = motioncrafter["model_set_manifest"]
    sources = model_manifest["sources"]
    unet_reference, unet_revision = _model_source(sources["unet"])
    vae_reference, vae_revision = _model_source(sources["vae"])
    image_reference, image_revision = _model_source(sources["image_vae"])
    base_reference, base_revision = _model_source(sources["base_pipeline"])
    model_set = PinnedMotionCrafterModelSet.inspect(
        model_type=str(model_manifest["model_type"]),
        unet_reference=unet_reference,
        unet_revision=unet_revision,
        vae_reference=vae_reference,
        vae_revision=vae_revision,
        image_vae_reference=image_reference,
        image_vae_revision=image_revision,
        base_pipeline_reference=base_reference,
        base_pipeline_revision=base_revision,
    )
    if model_set.set_sha256 != motioncrafter["model_set_id"]:
        raise ValueError("runtime MotionCrafter model set changed")
    configuration = provider["run_configuration"]
    job = jobs[0]
    video = _validate_source(args.processed_root.resolve(), job)
    output = _safe_member(args.output_root.resolve(), str(job["output_relative_path"]))
    config = model_set.build_config(
        upstream_root=args.motioncrafter_root.resolve(),
        video_path=video,
        output_directory=output,
        cache_directory=str(args.cache_dir.resolve()),
        height=int(configuration["height"]),
        width=int(configuration["width"]),
        window_size=int(configuration["window_size"]),
        overlap=int(configuration["overlap"]),
        num_inference_steps=int(configuration["num_inference_steps"]),
        guidance_scale=float(configuration["guidance_scale"]),
        decode_chunk_size=int(configuration["decode_chunk_size"]),
        seed=int(configuration["seed"]),
        seed_policy=str(configuration["seed_policy"]),
        low_memory_usage=bool(configuration["low_memory_usage"]),
        frame_start=int(job["source_frame_start"]),
        frame_stop=int(job["source_frame_stop_exclusive"]),
        frame_stride=int(configuration["frame_stride"]),
    )
    existing = output.exists() and any(output.iterdir())
    if existing and not args.resume:
        raise ValueError(f"supplement output exists: {output}")
    prediction = _build_pinned_runner(
        model_set,
        SafeMotionCrafterRunner,
        config,
    ).run(resume=existing)
    result = _validate_prediction(
        prediction,
        job=job,
        configuration=configuration,
        verify_manifest=verify_motioncrafter_prediction_manifest,
    )
    print(
        f"worker completed {job['camera']} {result['prediction_manifest_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
