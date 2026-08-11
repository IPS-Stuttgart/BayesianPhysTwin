#!/usr/bin/env python3
"""Run the frozen public Deform360 v5.1 MotionCrafter source jobs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_joint_sparse_camera_recovery_v5_2 import (
    RECOVERY_PROVIDER_SCHEMA,
    load_deform360_joint_sparse_motioncrafter_execution_plan_v5_2,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
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


def _require_clean_revision(root: Path, expected: str, *, name: str) -> None:
    head = _git_head(root)
    if head != expected:
        raise ValueError(f"{name} checkout is at {head}, expected {expected}")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError(f"{name} checkout is dirty")


def _safe_member(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"artifact path escapes root: {relative}") from error
    return candidate


def _require_bound_file(path: Path, binding: Mapping[str, Any], *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is missing")
    if _sha256(path) != binding["file_sha256"]:
        raise ValueError(f"{name} digest changed")


def _model_source(source: Mapping[str, Any]) -> tuple[str, str | None]:
    kind = source.get("kind")
    if kind == "huggingface_revision":
        return str(source["repository"]), str(source["revision"])
    if kind == "local_snapshot":
        raise ValueError("portable source plan cannot select a runtime-local snapshot")
    raise ValueError(f"unsupported model source kind {kind!r}")


class _SharedPinnedAdapterFactory:
    """Reuse one immutable model set while swapping per-video run metadata."""

    def __init__(self, model_set: Any) -> None:
        self._factory = model_set.adapter_factory()
        self._adapter: Any | None = None
        self._static_configuration: dict[str, object] | None = None

    @staticmethod
    def _static(config: Any) -> dict[str, object]:
        return {
            key: getattr(config, key)
            for key in (
                "model_type",
                "unet_path",
                "vae_path",
                "base_pipeline_path",
                "cache_directory",
                "height",
                "width",
                "window_size",
                "overlap",
                "num_inference_steps",
                "guidance_scale",
                "decode_chunk_size",
                "seed",
                "seed_policy",
                "low_memory_usage",
                "frame_stride",
                "model_source_set_sha256",
            )
        }

    def __call__(self, config: Any) -> Any:
        current = self._static(config)
        if self._adapter is None:
            self._adapter = self._factory(config)
            self._static_configuration = current
        elif current != self._static_configuration:
            raise ValueError("shared MotionCrafter model configuration changed")
        else:
            self._adapter.config = config
        return self._adapter


def _release_job_memory() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_with_memory_barrier(runner: Any, *, resume: bool) -> Path:
    try:
        return runner.run(resume=resume)
    finally:
        _release_job_memory()


def _isolated_worker_command(
    args: argparse.Namespace,
    *,
    runner_source: Path,
    job_id: str,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(runner_source),
        "--plan",
        str(args.plan.resolve()),
        "--source-execution-lock",
        str(args.source_execution_lock.resolve()),
        "--prepared-source-inventory",
        str(args.prepared_source_inventory.resolve()),
        "--camera-roster-manifest",
        str(args.camera_roster_manifest.resolve()),
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
    for option, path in (
        ("--base-provider-plan", args.base_provider_plan),
        ("--camera-recovery-preflight", args.camera_recovery_preflight),
        ("--camera-recovery-amendment", args.camera_recovery_amendment),
    ):
        if path is not None:
            command.extend((option, str(path.resolve())))
    return command


def _validate_source(root: Path, job: Mapping[str, Any]) -> Path:
    source = job["source_video"]
    path = _safe_member(root, str(source["path"]))
    if not path.is_file():
        raise ValueError(f"source video is missing: {source['path']}")
    if path.stat().st_size != int(source["bytes"]):
        raise ValueError(f"source video byte count changed: {source['path']}")
    if _sha256(path) != source["sha256"]:
        raise ValueError(f"source video digest changed: {source['path']}")
    return path


def _validate_prediction(
    path: Path,
    *,
    job: Mapping[str, Any],
    configuration: Mapping[str, Any],
    verify_motioncrafter_prediction_manifest: Any,
) -> dict[str, object]:
    verification = verify_motioncrafter_prediction_manifest(path, verify_hashes=True)
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
        raise ValueError("prediction configuration changed")
    schedule = payload.get("stochastic_seed_schedule")
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("calls") != job["seed_schedule"]
    ):
        raise ValueError("prediction seed schedule changed")
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
        raise ValueError("prediction overlap windows changed")
    return {
        "prediction_manifest": str(path.resolve()),
        "prediction_manifest_sha256": _sha256(path),
        "verification": verification,
    }


def _run_report(
    *,
    plan: Mapping[str, Any],
    completed: list[dict[str, object]],
    requested_job_count: int,
    shard_index: int,
    shard_count: int,
    smoke_only: bool,
    runtime_revision: str,
) -> dict[str, Any]:
    descriptor = {
        "schema": ("bayesian-phystwin.deform360-joint-sparse-motioncrafter-source-run"),
        "schema_version": 1,
        "source_plan_sha256": plan["manifest_sha256"],
        "runtime_revision": runtime_revision,
        "mode": "smoke" if smoke_only else "complete" if shard_count == 1 else "shard",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "status": "complete" if len(completed) == requested_job_count else "incomplete",
        "requested_job_count": requested_job_count,
        "completed_job_count": len(completed),
        "completed_jobs": completed,
        "information_boundary": {
            "provider_outputs_opened_for_integrity": True,
            "development_suffix_opened": False,
            "future_object_observations_used": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "human_approval_required": False,
            "new_measurements_required": False,
        },
        "claim_boundary": (
            "Public source-prefix provider inference and integrity only. No "
            "development suffix, prediction score, confirmation payload, or "
            "target outcome was opened."
        ),
    }
    return {"run_sha256": content_id(descriptor), **descriptor}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-execution-lock", type=Path, required=True)
    parser.add_argument("--prepared-source-inventory", type=Path, required=True)
    parser.add_argument("--camera-roster-manifest", type=Path, required=True)
    parser.add_argument("--base-provider-plan", type=Path)
    parser.add_argument("--camera-recovery-preflight", type=Path)
    parser.add_argument("--camera-recovery-amendment", type=Path)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prob4d-root", type=Path, required=True)
    parser.add_argument("--motioncrafter-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--worker-job-id", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    if args.worker_job_id is not None and (
        args.smoke_only or args.shard_count != 1 or args.shard_index != 0
    ):
        raise ValueError("worker mode cannot be combined with smoke or sharding")

    plan = load_deform360_joint_sparse_motioncrafter_execution_plan_v5_2(args.plan)
    implementation = plan["implementation"]
    runner_source = Path(__file__).resolve()
    if _sha256(runner_source) != implementation["runner_source_sha256"]:
        raise ValueError("executed runner differs from the frozen source plan")
    _require_bound_file(
        args.source_execution_lock.resolve(),
        plan["source_execution_lock"],
        name="source execution lock",
    )
    _require_bound_file(
        args.prepared_source_inventory.resolve(),
        plan["prepared_source_inventory"],
        name="prepared source inventory",
    )
    _require_bound_file(
        args.camera_roster_manifest.resolve(),
        plan["camera_roster_source"],
        name="camera-roster manifest",
    )
    recovery_paths = (
        args.base_provider_plan,
        args.camera_recovery_preflight,
        args.camera_recovery_amendment,
    )
    if plan["schema"] == RECOVERY_PROVIDER_SCHEMA:
        if any(path is None for path in recovery_paths):
            raise ValueError("recovery plan requires all recovery provenance paths")
        _require_bound_file(
            args.base_provider_plan.resolve(),
            plan["base_provider_plan"],
            name="base provider plan",
        )
        _require_bound_file(
            args.camera_recovery_preflight.resolve(),
            plan["camera_recovery_preflight"],
            name="camera recovery preflight",
        )
        _require_bound_file(
            args.camera_recovery_amendment.resolve(),
            plan["camera_recovery_amendment"],
            name="camera recovery amendment",
        )
    elif any(path is not None for path in recovery_paths):
        raise ValueError("base provider plan cannot accept recovery provenance paths")
    _require_clean_revision(
        args.repository_root.resolve(),
        str(implementation["revision"]),
        name="BayesianPhysTwin",
    )
    _require_clean_revision(
        args.prob4d_root.resolve(),
        str(plan["provider_lock"]["provider_revision"]),
        name="Prob4D",
    )
    _require_clean_revision(
        args.motioncrafter_root.resolve(),
        str(plan["motioncrafter"]["revision"]),
        name="MotionCrafter",
    )

    sys.path.insert(0, str(args.prob4d_root.resolve() / "src"))
    from prob4d.motioncrafter_integrity import (  # noqa: PLC0415
        verify_motioncrafter_prediction_manifest,
    )
    from prob4d.motioncrafter_models import (  # noqa: PLC0415
        PinnedMotionCrafterModelSet,
    )
    from prob4d.motioncrafter_runner import SafeMotionCrafterRunner  # noqa: PLC0415

    model_manifest = plan["motioncrafter"]["model_set_manifest"]
    sources = model_manifest["sources"]
    unet_reference, unet_revision = _model_source(sources["unet"])
    vae_reference, vae_revision = _model_source(sources["vae"])
    image_vae_reference, image_vae_revision = _model_source(sources["image_vae"])
    base_reference, base_revision = _model_source(sources["base_pipeline"])
    model_set = PinnedMotionCrafterModelSet.inspect(
        model_type=str(model_manifest["model_type"]),
        unet_reference=unet_reference,
        unet_revision=unet_revision,
        vae_reference=vae_reference,
        vae_revision=vae_revision,
        image_vae_reference=image_vae_reference,
        image_vae_revision=image_vae_revision,
        base_pipeline_reference=base_reference,
        base_pipeline_revision=base_revision,
    )
    if model_set.set_sha256 != plan["motioncrafter"]["model_set_id"]:
        raise ValueError("runtime MotionCrafter model set differs from the plan")

    configuration = plan["run_configuration"]
    jobs = list(plan["jobs"])
    if args.worker_job_id is not None:
        jobs = [job for job in jobs if job["job_id"] == args.worker_job_id]
        if len(jobs) != 1:
            raise ValueError("worker job is not unique in the source plan")
    elif args.smoke_only:
        jobs = [job for job in jobs if job["job_id"] == plan["smoke_job_id"]]
    else:
        jobs = [
            job
            for index, job in enumerate(jobs)
            if index % args.shard_count == args.shard_index
        ]
    shared_factory = (
        _SharedPinnedAdapterFactory(model_set)
        if args.worker_job_id is not None
        else None
    )
    completed: list[dict[str, object]] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_name = (
        "run_report.json"
        if args.shard_count == 1
        else f"run_report.shard-{args.shard_index:02d}-of-{args.shard_count:02d}.json"
    )
    report_path = args.output_root.resolve() / report_name

    for job in jobs:
        video_path = _validate_source(args.processed_root.resolve(), job)
        output_directory = _safe_member(
            args.output_root.resolve(), str(job["output_relative_path"])
        )
        config = model_set.build_config(
            upstream_root=args.motioncrafter_root.resolve(),
            video_path=video_path,
            output_directory=output_directory,
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
        has_existing = output_directory.exists() and any(output_directory.iterdir())
        if has_existing and not args.resume:
            raise ValueError(
                f"job output already exists; rerun with --resume: {output_directory}"
            )
        prediction_path = output_directory / "predictions.json"
        if args.worker_job_id is not None:
            if shared_factory is None:
                raise AssertionError("worker lacks a pinned adapter factory")
            runner = SafeMotionCrafterRunner(config, adapter_factory=shared_factory)
            prediction_path = _run_with_memory_barrier(runner, resume=has_existing)
        elif not prediction_path.is_file():
            subprocess.run(
                _isolated_worker_command(
                    args,
                    runner_source=runner_source,
                    job_id=str(job["job_id"]),
                    resume=has_existing,
                ),
                check=True,
            )
        result = _validate_prediction(
            prediction_path,
            job=job,
            configuration=configuration,
            verify_motioncrafter_prediction_manifest=(
                verify_motioncrafter_prediction_manifest
            ),
        )
        completed.append({"job_id": job["job_id"], **result})
        if args.worker_job_id is not None:
            print(f"worker completed {job['job_id']}", flush=True)
            continue
        write_atomic_json(
            _run_report(
                plan=plan,
                completed=completed,
                requested_job_count=len(jobs),
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                smoke_only=args.smoke_only,
                runtime_revision=_git_head(args.repository_root.resolve()),
            ),
            report_path,
            overwrite=True,
        )
        print(f"completed {len(completed)}/{len(jobs)} {job['job_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
