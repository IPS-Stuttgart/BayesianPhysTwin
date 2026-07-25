#!/usr/bin/env python3
"""Run exactly one official Deform360 reconstruction in a child process."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one process-isolated official Deform360 reconstruction"
    )
    parser.add_argument("--lock", required=True)
    parser.add_argument(
        "--role", required=True, choices=("calibration", "confirmation")
    )
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--online-prediction-seal", required=True)
    parser.add_argument("--aligned-episode", required=True)
    parser.add_argument("--reconstruction-output-dir", required=True)
    parser.add_argument("--result-archive", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--cohort-barrier-sha256", required=True)
    parser.add_argument("--deform360-repo", required=True)
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--cotracker-repo", required=True)
    parser.add_argument("--cotracker-checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ffmpeg", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("isolated reconstruction is not bound to physical GPU 0")
    if not (
        sys.flags.isolated == 1
        and sys.flags.no_user_site == 1
        and sys.flags.ignore_environment == 1
        and sys.dont_write_bytecode
    ):
        raise RuntimeError("isolated reconstruction Python flags changed")
    source = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(source))
    from bayesian_phystwin import deform360_held_v83_gsplat_runtime as gsplat
    from bayesian_phystwin import deform360_held_v83_viser_guard as viser_guard

    gsplat_runtime_smoke = gsplat.load_and_smoke_gsplat_runtime()
    viser_process_guard = viser_guard.install_viser_process_churn_guard()
    from bayesian_phystwin import deform360_case_process_isolation as isolation
    from bayesian_phystwin import deform360_held_outcome_reconstruction as numerical
    from bayesian_phystwin import deform360_held_v8_outcome_driver as driver
    from bayesian_phystwin import deform360_held_v8_outcome_reconstruction as adapter

    if dict(os.environ) != driver._normalized_environment():
        raise RuntimeError("isolated reconstruction environment changed")
    backend = numerical.PinnedOfficialPipelineBackend(
        deform360_repo=arguments.deform360_repo,
        sam2_repository=arguments.sam2_repository,
        sam2_checkpoint=arguments.sam2_checkpoint,
        cotracker_repo=arguments.cotracker_repo,
        cotracker_checkpoint=arguments.cotracker_checkpoint,
        device=arguments.device,
        ffmpeg=arguments.ffmpeg,
        splat_trainer_mode=numerical.PROCESS_ISOLATED_PINNED_TRAINER_MODE,
    )
    reconstruction = adapter.reconstruct_fresh_official_target(
        lock_path=arguments.lock,
        role=arguments.role,
        case_name=arguments.case_name,
        online_prediction_seal_path=arguments.online_prediction_seal,
        aligned_episode_dir=arguments.aligned_episode,
        output_dir=arguments.reconstruction_output_dir,
        cohort_barrier_sha256=arguments.cohort_barrier_sha256,
        backend=backend,
    )
    reconstruction = dict(reconstruction)
    provenance = dict(reconstruction.get("provenance", {}))
    provenance["isolated_gsplat_runtime_smoke"] = dict(gsplat_runtime_smoke)
    provenance["isolated_viser_process_churn_guard"] = dict(
        viser_process_guard
    )
    reconstruction["provenance"] = provenance
    isolation.write_isolated_reconstruction_result(
        arguments.result_archive,
        arguments.result_manifest,
        case_name=arguments.case_name,
        role=arguments.role,
        lock_path=arguments.lock,
        cohort_barrier_sha256=arguments.cohort_barrier_sha256,
        reconstruction=reconstruction,
        worker_source_path=Path(__file__).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
