#!/usr/bin/env python3
"""Run the frozen official-Hub MotionCrafter calibration jobs."""

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
from bayesian_phystwin.deform360_official_hub_motioncrafter_jobs import (
    load_deform360_motioncrafter_job_manifest,
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


def _require_ancestor(root: Path, ancestor: str, *, name: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{name} implementation revision is not in runtime history")
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


def _model_source(source: Mapping[str, Any]) -> tuple[str, str | None]:
    kind = source.get("kind")
    if kind == "huggingface_revision":
        return str(source["repository"]), str(source["revision"])
    if kind == "local_snapshot":
        raise ValueError("portable job manifest cannot select a runtime-local snapshot")
    raise ValueError(f"unsupported model source kind {kind!r}")


class _SharedPinnedAdapterFactory:
    """Reuse one immutable model set while swapping only per-video run metadata."""

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
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"prediction configuration changed: {actual!r}")
    schedule = payload.get("stochastic_seed_schedule")
    if not isinstance(schedule, Mapping) or schedule.get("calls") != job["seed_schedule"]:
        raise ValueError("prediction seed schedule changed")
    overlap = payload.get("overlap_windows")
    expected_windows = [
        {
            "window_id": item["window_id"],
            "path": f"windows/{item['window_id']}.npz",
            "start_frame": item["source_frame_start"],
            "stop_frame": item["source_frame_stop_exclusive"],
        }
        for item in job["windows"]
    ]
    if overlap != expected_windows:
        raise ValueError("prediction overlap windows changed")
    return {
        "prediction_manifest": str(path.resolve()),
        "prediction_manifest_sha256": _sha256(path),
        "verification": verification,
    }


def _run_report(
    *,
    job_manifest: Mapping[str, Any],
    completed: list[dict[str, object]],
    requested_job_count: int,
    smoke_only: bool,
    runtime_revision: str,
) -> dict[str, Any]:
    descriptor = {
        "schema": "bayesian-phystwin.deform360-motioncrafter-calibration-run",
        "schema_version": 1,
        "job_manifest_sha256": job_manifest["manifest_sha256"],
        "runtime_revision": runtime_revision,
        "mode": "smoke" if smoke_only else "complete-cohort",
        "status": (
            "smoke_complete"
            if smoke_only and len(completed) == requested_job_count
            else "complete"
            if len(completed) == requested_job_count
            else "incomplete"
        ),
        "requested_job_count": requested_job_count,
        "completed_job_count": len(completed),
        "completed_jobs": completed,
        "information_boundary": {
            "calibration_provider_outputs_opened_for_integrity": True,
            "calibration_scores_opened": False,
            "calibration_policy_fit": False,
            "future_frames_used_for_prediction": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        },
        "claim_boundary": (
            "Provider inference and integrity only. No prediction score, calibration "
            "fit, confirmation payload, or target outcome was opened."
        ),
    }
    return {"run_sha256": content_id(descriptor), **descriptor}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prob4d-root", type=Path, required=True)
    parser.add_argument("--motioncrafter-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    job_manifest = load_deform360_motioncrafter_job_manifest(args.job_manifest)
    implementation = job_manifest["implementation"]
    runner_source = Path(__file__).resolve()
    if _sha256(runner_source) != implementation["runner_source_sha256"]:
        raise ValueError("executed runner source differs from the frozen job manifest")
    _require_ancestor(
        args.repository_root.resolve(),
        str(implementation["revision"]),
        name="Bayesian-PhysTwin",
    )
    _require_clean_revision(
        args.prob4d_root.resolve(),
        str(job_manifest["provider_lock"]["provider_revision"]),
        name="Prob4D",
    )
    _require_clean_revision(
        args.motioncrafter_root.resolve(),
        str(job_manifest["motioncrafter"]["revision"]),
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

    model_manifest = job_manifest["motioncrafter"]["model_set_manifest"]
    sources = model_manifest["sources"]
    unet_reference, unet_revision = _model_source(sources["unet"])
    vae_reference, vae_revision = _model_source(sources["vae"])
    base_reference, base_revision = _model_source(sources["base_pipeline"])
    model_set = PinnedMotionCrafterModelSet.inspect(
        model_type=str(model_manifest["model_type"]),
        unet_reference=unet_reference,
        unet_revision=unet_revision,
        vae_reference=vae_reference,
        vae_revision=vae_revision,
        base_pipeline_reference=base_reference,
        base_pipeline_revision=base_revision,
    )
    if model_set.set_sha256 != job_manifest["motioncrafter"]["model_set_id"]:
        raise ValueError("runtime MotionCrafter model set differs from the frozen set")

    configuration = job_manifest["run_configuration"]
    jobs = list(job_manifest["jobs"])
    if args.smoke_only:
        jobs = [job for job in jobs if job["job_id"] == job_manifest["smoke_job_id"]]
        if len(jobs) != 1:
            raise ValueError("frozen smoke job is not unique")
    shared_factory = _SharedPinnedAdapterFactory(model_set)
    completed: list[dict[str, object]] = []
    report_path = args.output_root.resolve() / "run_report.json"
    args.output_root.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        video_path = _validate_source(args.processed_root.resolve(), job)
        output_directory = _safe_member(
            args.output_root.resolve(),
            str(job["output_relative_path"]),
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
        prediction_path = SafeMotionCrafterRunner(
            config,
            adapter_factory=shared_factory,
        ).run(resume=has_existing)
        result = _validate_prediction(
            prediction_path,
            job=job,
            configuration=configuration,
            verify_motioncrafter_prediction_manifest=(
                verify_motioncrafter_prediction_manifest
            ),
        )
        completed.append({"job_id": job["job_id"], **result})
        write_atomic_json(
            _run_report(
                job_manifest=job_manifest,
                completed=completed,
                requested_job_count=len(jobs),
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
