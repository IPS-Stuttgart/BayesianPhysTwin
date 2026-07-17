#!/usr/bin/env python3
"""Run resumable prediction-first Deform360 independent-source shards."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _valid_file_hash(path: Path, expected: object) -> bool:
    return (
        path.is_file() and isinstance(expected, str) and _sha256_file(path) == expected
    )


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {log_path.stem}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        print("\n".join(tail), file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command)
    print(f"DONE {log_path.stem}", flush=True)


def _reconstruction_command(
    python: Path,
    bpt_site_packages: Path,
    script: Path,
    arguments: Sequence[str],
) -> list[str]:
    wrapper = (
        "import runpy,sys; "
        "extra=sys.argv.pop(1); script=sys.argv.pop(1); "
        "sys.path.append(extra); sys.argv[0]=script; "
        "runpy.run_path(script,run_name='__main__')"
    )
    return [
        str(python),
        "-c",
        wrapper,
        str(bpt_site_packages),
        str(script),
        *arguments,
    ]


def _prediction_seal_valid(
    seal_path: Path,
    *,
    lock_path: Path,
    object_id: str,
    episode_id: int,
    bpt_python: Path,
    bpt_env: dict[str, str],
) -> bool:
    if not seal_path.is_file():
        return False
    code = (
        "import json,sys; "
        "from causal4d_public.deform360_independent_source import "
        "validate_independent_source_prediction_seal as validate; "
        "p=json.load(open(sys.argv[1])); validate(p,verify_archive=True); "
        "assert p['object_id']==sys.argv[2] and int(p['episode_id'])==int(sys.argv[3]); "
        "assert p['lock_sha256']==__import__('hashlib').sha256(open(sys.argv[4],'rb').read()).hexdigest()"
    )
    completed = subprocess.run(
        [
            str(bpt_python),
            "-c",
            code,
            str(seal_path),
            object_id,
            str(episode_id),
            str(lock_path),
        ],
        env=bpt_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _stage_valid(episode_dir: Path, object_id: str, episode_id: int) -> bool:
    manifest_path = episode_dir / "dense_source_smoke.manifest.json"
    alignment_path = episode_dir / "action_aligned_source_staging.json"
    if not manifest_path.is_file() or not alignment_path.is_file():
        return False
    manifest = _load_json(manifest_path)
    alignment = _load_json(alignment_path)
    return bool(
        manifest.get("source_only") is True
        and manifest.get("target_episode_accessed") is False
        and manifest.get("calibration_episode_accessed") is False
        and manifest.get("object_id") == object_id
        and int(manifest.get("episode_index", -1)) == episode_id
        and alignment.get("object_id") == object_id
        and int(alignment.get("episode_id", -1)) == episode_id
        and alignment.get("source_only") is True
        and alignment.get("target_action_read") is False
        and alignment.get("target_observation_read") is False
        and alignment.get("target_future_read") is False
        and len(alignment.get("selected_raw_frame_range_half_open", ())) == 2
        and int(alignment["selected_raw_frame_range_half_open"][1])
        - int(alignment["selected_raw_frame_range_half_open"][0])
        == 81
    )


def _reconstruction_valid(
    metadata_path: Path,
    *,
    frame_zero_only: bool,
    prediction_result_sha256: str | None = None,
) -> bool:
    if not metadata_path.is_file():
        return False
    payload = _load_json(metadata_path)
    expected_frames = [0] if frame_zero_only else list(range(81))
    if (
        payload.get("frame_zero_only") is not frame_zero_only
        or sorted(int(value) for value in payload.get("outputs", {})) != expected_frames
    ):
        return False
    if not frame_zero_only:
        if (
            payload.get("prediction_seal", {}).get("result_sha256")
            != prediction_result_sha256
            or payload.get("information_boundary", {}).get(
                "prediction_seal_verified_before_future_reconstruction"
            )
            is not True
        ):
            return False
    return all(
        _valid_file_hash(Path(record["path"]), record.get("sha256"))
        for record in payload["outputs"].values()
    )


def _summary_valid(
    path: Path,
    *,
    object_id: str,
    episode_id: int,
    output_key: str | None = None,
    output_path: Path | None = None,
) -> bool:
    if not path.is_file():
        return False
    payload = _load_json(path)
    if (
        payload.get("passed") is not True
        or payload.get("object_id") != object_id
        or int(payload.get("episode_id", -1)) != episode_id
    ):
        return False
    if output_key is not None and output_path is not None:
        expected = payload.get("output_sha256", {}).get(output_key)
        return _valid_file_hash(output_path, expected)
    return True


def _warp_result_valid(path: Path, displacement_scale: float) -> bool:
    if not path.is_file():
        return False
    payload = _load_json(path)
    return bool(
        payload.get("passed") is True
        and payload.get("official_phystwin_revision")
        == "2b6630528141b9cba5a7677c8b88b2129b4a8390"
        and payload.get("support_dynamics", {}).get("mode") == "official-ground"
        and float(
            payload.get("realized_actuation", {}).get(
                "controller_displacement_scale", -1.0
            )
        )
        == displacement_scale
    )


def _outcome_valid(
    outcome_path: Path,
    target_path: Path,
    *,
    object_id: str,
    episode_id: int,
    prediction_result_sha256: str,
) -> bool:
    if not outcome_path.is_file() or not target_path.is_file():
        return False
    payload = _load_json(outcome_path)
    return bool(
        payload.get("object_id") == object_id
        and int(payload.get("episode_id", -1)) == episode_id
        and payload.get("prediction_seal_sha256") == prediction_result_sha256
        and _valid_file_hash(
            target_path, payload.get("output_sha256", {}).get("target_data")
        )
        and payload.get("information_boundary", {}).get("future_tactile_read") is False
        and payload.get("information_boundary", {}).get("target_outcome_read") is False
    )


def _evaluation_valid(
    evaluation_path: Path,
    target_path: Path,
    *,
    object_id: str,
    episode_id: int,
    prediction_result_sha256: str,
) -> bool:
    if not evaluation_path.is_file() or not target_path.is_file():
        return False
    payload = _load_json(evaluation_path)
    return bool(
        payload.get("object_id") == object_id
        and int(payload.get("episode_id", -1)) == episode_id
        and payload.get("prediction_seal_sha256") == prediction_result_sha256
        and payload.get("target_data_sha256") == _sha256_file(target_path)
        and payload.get("information_boundary", {}).get("target_outcome_read") is False
    )


def _run_episode(
    args: argparse.Namespace,
    *,
    object_id: str,
    episode_id: int,
) -> None:
    episode_slug = f"{object_id}-ep{episode_id:04d}"
    aligned_root = args.stage_root / episode_slug
    episode_dir = aligned_root / "episode_0000"
    result_root = args.result_root / episode_slug
    log_root = result_root / "logs"
    result_root.mkdir(parents=True, exist_ok=True)

    common_path = os.pathsep.join((str(args.repo / "src"), str(args.deform360_repo)))
    bpt_env = dict(os.environ)
    bpt_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": common_path,
            "PYNPUT_BACKEND": "dummy",
            "PYOPENGL_PLATFORM": "egl",
            "WANDB_MODE": "disabled",
        }
    )
    reconstruction_env = dict(os.environ)
    reconstruction_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "CUDA_HOME": str(args.cuda_home),
            "PATH": os.pathsep.join(
                (
                    str(args.cuda_home / "bin"),
                    str(args.bpt_python.parent),
                    os.environ.get("PATH", ""),
                )
            ),
            "TORCH_EXTENSIONS_DIR": str(args.torch_extensions_dir),
            "PYTHONPATH": os.pathsep.join(
                (
                    str(args.runtime_overlay),
                    str(args.repo / "src"),
                    str(args.deform360_repo),
                    str(args.cotracker_repo),
                )
            ),
        }
    )

    seal_path = result_root / "prediction_seal.json"
    seal_valid = _prediction_seal_valid(
        seal_path,
        lock_path=args.lock,
        object_id=object_id,
        episode_id=episode_id,
        bpt_python=args.bpt_python,
        bpt_env=bpt_env,
    )
    if not seal_valid:
        if not _stage_valid(episode_dir, object_id, episode_id):
            _run(
                [
                    str(args.bpt_python),
                    str(
                        args.repo
                        / "scripts/remote/stage_deform360_dense_source_smoke.py"
                    ),
                    str(args.source_aligned_root),
                    str(
                        args.observation_root
                        / object_id
                        / f"episode_{episode_id:04d}"
                        / "sampled_masks.npz"
                    ),
                    str(aligned_root),
                    "--protocol",
                    str(args.replication_protocol),
                    "--object-id",
                    object_id,
                    "--episode",
                    str(episode_id),
                    "--dense-panel-config",
                    str(args.dense_panel_config),
                    "--action-aligned",
                    "--sam2-repository",
                    str(args.sam2_repo),
                    "--checkpoint",
                    str(args.sam2_checkpoint),
                    "--device",
                    "cuda",
                    "--allow-automatic-initial-mask-fallback",
                    "--overwrite",
                ],
                env=bpt_env,
                log_path=log_root / "01_stage.log",
            )
        frame_zero_meta = (
            episode_dir / "strict_hull_reconstruction_frame_zero.meta.json"
        )
        if not _reconstruction_valid(frame_zero_meta, frame_zero_only=True):
            _run(
                _reconstruction_command(
                    args.reconstruction_python,
                    args.bpt_site_packages,
                    args.repo
                    / "scripts/remote/run_deform360_strict_hull_reconstruction.py",
                    [
                        "--aligned-dir",
                        str(aligned_root),
                        "--episode",
                        "0",
                        "--frame-zero-only",
                    ],
                ),
                env=reconstruction_env,
                log_path=log_root / "02_frame_zero_reconstruction.log",
            )

        prediction_data = result_root / "prediction_only_input.pkl"
        prediction_summary = result_root / "prediction_only_input.json"
        if (
            not _summary_valid(
                prediction_summary,
                object_id=object_id,
                episode_id=episode_id,
                output_key=None,
                output_path=prediction_data,
            )
            or not prediction_data.is_file()
        ):
            _run(
                [
                    str(args.bpt_python),
                    str(
                        args.repo
                        / "scripts/remote/build_deform360_prediction_only_input.py"
                    ),
                    "--lock",
                    str(args.lock),
                    "--object-id",
                    object_id,
                    "--episode-id",
                    str(episode_id),
                    "--episode-dir",
                    str(episode_dir),
                    "--output",
                    str(prediction_data),
                    "--summary",
                    str(prediction_summary),
                ],
                env=bpt_env,
                log_path=log_root / "03_prediction_input.log",
            )

        episode_graph = result_root / "episode_graph.npz"
        simulator_data = result_root / "simulator_final_data.pkl"
        state_artifact = result_root / "state_artifact.npz"
        twin_summary = result_root / "twin_summary.json"
        if not _summary_valid(
            twin_summary,
            object_id=object_id,
            episode_id=episode_id,
            output_key="simulator_final_data",
            output_path=simulator_data,
        ):
            _run(
                [
                    str(args.bpt_python),
                    str(
                        args.repo
                        / "scripts/remote/build_deform360_automatic_episode_twin.py"
                    ),
                    "--repo",
                    str(args.repo),
                    "--object-id",
                    object_id,
                    "--episode-id",
                    str(episode_id),
                    "--phase",
                    "source",
                    "--episode-final-data",
                    str(prediction_data),
                    "--episode-graph",
                    str(episode_graph),
                    "--simulator-final-data",
                    str(simulator_data),
                    "--state-artifact",
                    str(state_artifact),
                    "--summary",
                    str(twin_summary),
                    "--prediction-only-input",
                    "--canonical-node-count",
                    "384",
                ],
                env=bpt_env,
                log_path=log_root / "04_automatic_twin.log",
            )

        def run_warp(label: str, scale: float) -> Path:
            output_dir = result_root / label
            result_path = output_dir / "official_phystwin_smoke.json"
            if _warp_result_valid(result_path, scale):
                return result_path
            if output_dir.exists():
                suffix = 1
                while (result_root / f"{label}_retry{suffix}").exists():
                    suffix += 1
                output_dir = result_root / f"{label}_retry{suffix}"
                result_path = output_dir / "official_phystwin_smoke.json"
            _run(
                [
                    str(args.bpt_python),
                    str(
                        args.repo
                        / "scripts/remote/run_deform360_official_phystwin_smoke.py"
                    ),
                    "--official-phystwin-repo",
                    str(args.official_phystwin_repo),
                    "--data",
                    str(simulator_data),
                    "--config",
                    str(args.official_phystwin_config),
                    "--split-json",
                    str(args.score_split),
                    "--output-dir",
                    str(output_dir),
                    "--canonical-reusable-graph",
                    str(episode_graph),
                    "--device",
                    "cuda:0",
                    "--controller-radius-m",
                    "0.03",
                    "--controller-max-neighbours",
                    "1",
                    "--canonical-controller-patch-size",
                    "16",
                    "--init-spring-y",
                    "10000",
                    "--drag-damping",
                    "10",
                    "--dashpot-damping",
                    "100",
                    "--controller-displacement-scale",
                    str(scale),
                    "--support-dynamics",
                    "official-ground",
                    "--report-edge-strain",
                ],
                env=bpt_env,
                log_path=log_root / f"05_warp_{label}.log",
            )
            return result_path

        driven_result = run_warp("warp_driven", 1.0)
        zero_result = run_warp("warp_zero", 0.0)
        prediction_archive = result_root / "prediction.npz"
        _run(
            [
                str(args.bpt_python),
                str(
                    args.repo
                    / "scripts/remote/seal_deform360_independent_source_prediction.py"
                ),
                "--lock",
                str(args.lock),
                "--object-id",
                object_id,
                "--episode-id",
                str(episode_id),
                "--prediction-data",
                str(prediction_data),
                "--simulator-data",
                str(simulator_data),
                "--graph",
                str(episode_graph),
                "--readout",
                str(state_artifact),
                "--twin-summary",
                str(twin_summary),
                "--driven-result",
                str(driven_result),
                "--zero-result",
                str(zero_result),
                "--prediction-archive",
                str(prediction_archive),
                "--output",
                str(seal_path),
            ],
            env=bpt_env,
            log_path=log_root / "06_prediction_seal.log",
        )
        seal_valid = _prediction_seal_valid(
            seal_path,
            lock_path=args.lock,
            object_id=object_id,
            episode_id=episode_id,
            bpt_python=args.bpt_python,
            bpt_env=bpt_env,
        )
        if not seal_valid:
            raise RuntimeError("new prediction seal did not validate")

    print(f"SEALED {object_id}/{episode_id}", flush=True)
    if args.phase == "prediction":
        return

    seal = _load_json(seal_path)
    prediction_result_sha256 = str(seal["result_sha256"])
    full_meta = episode_dir / "strict_hull_reconstruction_full.meta.json"
    if not _reconstruction_valid(
        full_meta,
        frame_zero_only=False,
        prediction_result_sha256=prediction_result_sha256,
    ):
        _run(
            _reconstruction_command(
                args.reconstruction_python,
                args.bpt_site_packages,
                args.repo
                / "scripts/remote/run_deform360_strict_hull_reconstruction.py",
                [
                    "--aligned-dir",
                    str(aligned_root),
                    "--episode",
                    "0",
                    "--resume",
                    "--prediction-seal",
                    str(seal_path),
                ],
            ),
            env=reconstruction_env,
            log_path=log_root / "07_full_reconstruction.log",
        )
    if args.phase == "reconstruction":
        print(f"RECONSTRUCTED {object_id}/{episode_id}", flush=True)
        return

    target_data = result_root / "target_data.pkl"
    outcome_path = result_root / "outcome.json"
    if not _outcome_valid(
        outcome_path,
        target_data,
        object_id=object_id,
        episode_id=episode_id,
        prediction_result_sha256=prediction_result_sha256,
    ):
        _run(
            _reconstruction_command(
                args.reconstruction_python,
                args.bpt_site_packages,
                args.repo
                / "scripts/remote/build_deform360_independent_source_outcome.py",
                [
                    "--repo",
                    str(args.repo),
                    "--deform360-repo",
                    str(args.deform360_repo),
                    "--lock",
                    str(args.lock),
                    "--prediction-seal",
                    str(seal_path),
                    "--aligned-dir",
                    str(aligned_root),
                    "--staged-episode",
                    "0",
                    "--checkpoint",
                    str(args.cotracker_checkpoint),
                    "--output-data",
                    str(target_data),
                    "--output",
                    str(outcome_path),
                    "--reuse-validated-stages",
                ],
            ),
            env=reconstruction_env,
            log_path=log_root / "08_outcome.log",
        )

    evaluation_path = result_root / "evaluation.json"
    if not _evaluation_valid(
        evaluation_path,
        target_data,
        object_id=object_id,
        episode_id=episode_id,
        prediction_result_sha256=prediction_result_sha256,
    ):
        evaluation_env = dict(os.environ)
        evaluation_env["PYTHONPATH"] = str(args.repo / "src")
        _run(
            [
                str(args.reconstruction_python),
                str(
                    args.repo
                    / "scripts/remote/evaluate_deform360_independent_source_prediction.py"
                ),
                "--lock",
                str(args.lock),
                "--prediction-seal",
                str(seal_path),
                "--target-data",
                str(target_data),
                "--output",
                str(evaluation_path),
            ],
            env=evaluation_env,
            log_path=log_root / "09_evaluation.log",
        )
    print(f"SCORED {object_id}/{episode_id}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prediction", "reconstruction", "outcome", "all"),
        required=True,
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--object-id")
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/mnt/corsair/florianpfaff/deform360-reusable-twin-v1"),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path(
            "/mnt/lexar4tb/datasets/deform360/graph-action-support-independent-source-v1"
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "/mnt/corsair/florianpfaff/deform360-dense-reusable-panel-v1/independent-source-v1"
        ),
    )
    parser.add_argument(
        "--source-aligned-root",
        type=Path,
        default=Path(
            "/mnt/lexar4tb/datasets/deform360/data-7fea8e2/replication-v1/aligned"
        ),
    )
    parser.add_argument(
        "--observation-root",
        type=Path,
        default=Path(
            "/mnt/lexar4tb/datasets/deform360/data-7fea8e2/replication-v1/observations"
        ),
    )
    parser.add_argument(
        "--deform360-repo",
        type=Path,
        default=Path("/mnt/lexar4tb/datasets/deform360/code"),
    )
    parser.add_argument(
        "--official-phystwin-repo",
        type=Path,
        default=Path("/home/florianpfaff/PhysTwin-upstream"),
    )
    parser.add_argument(
        "--official-phystwin-config",
        type=Path,
        default=Path("/home/florianpfaff/PhysTwin-upstream/configs/real.yaml"),
    )
    parser.add_argument(
        "--bpt-python",
        type=Path,
        default=Path("/home/florianpfaff/.venvs/bpt-gpu/bin/python"),
    )
    parser.add_argument(
        "--reconstruction-python",
        type=Path,
        default=Path(
            "/home/florianpfaff/codex-runs/pymegdec-source-inner-panel-20260606/.venv/bin/python"
        ),
    )
    parser.add_argument(
        "--bpt-site-packages",
        type=Path,
        default=Path("/home/florianpfaff/.venvs/bpt-gpu/lib/python3.12/site-packages"),
    )
    parser.add_argument(
        "--runtime-overlay",
        type=Path,
        default=Path("/mnt/corsair/florianpfaff/deform360-runtime-cu130"),
    )
    parser.add_argument(
        "--torch-extensions-dir",
        type=Path,
        default=Path("/home/florianpfaff/.cache/torch_extensions/py312_cu130"),
    )
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-12.9"))
    parser.add_argument(
        "--sam2-repo",
        type=Path,
        default=Path("/mnt/lexar4tb/datasets/deform360/sam2-2b90b9f5"),
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path(
            "/mnt/lexar4tb/datasets/deform360/sam2-2b90b9f5/checkpoints/sam2.1_hiera_small.pt"
        ),
    )
    parser.add_argument(
        "--cotracker-repo",
        type=Path,
        default=Path("/home/florianpfaff/co-tracker"),
    )
    parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        default=Path(
            "/home/florianpfaff/.cache/torch/hub/checkpoints/scaled_offline.pth"
        ),
    )
    args = parser.parse_args()
    args.lock = (
        args.repo
        / "configs/causal4d_public/deform360_graph_action_support_independent_source_v1.json"
    )
    args.replication_protocol = (
        args.repo / "configs/causal4d_public/deform360_replication_v1.json"
    )
    args.dense_panel_config = (
        args.repo / "configs/causal4d_public/deform360_dense_reusable_panel_v1.json"
    )
    args.score_split = (
        args.repo / "configs/causal4d_public/deform360_independent_source_split_v1.json"
    )
    if (args.object_id is None) != (args.episode_id is None):
        parser.error("--object-id and --episode-id must be provided together")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    return args


def main() -> int:
    args = _parse_args()
    lock = _load_json(args.lock)
    episodes = [
        (str(object_id), int(episode_id))
        for object_id, episode_ids in lock["independent_source_panel"][
            "episodes_by_object"
        ].items()
        for episode_id in episode_ids
    ]
    if args.object_id is not None:
        requested = (args.object_id, int(args.episode_id))
        if requested not in episodes:
            raise ValueError(
                f"episode is outside the independent source lock: {requested}"
            )
        episodes = [requested]
    else:
        episodes = [
            episode
            for index, episode in enumerate(episodes)
            if index % args.shard_count == args.shard_index
        ]
    failures = []
    for object_id, episode_id in episodes:
        print(f"BEGIN {object_id}/{episode_id} phase={args.phase}", flush=True)
        try:
            episode_lock_dir = args.result_root / f"{object_id}-ep{episode_id:04d}"
            episode_lock_dir.mkdir(parents=True, exist_ok=True)
            with (episode_lock_dir / ".panel_runner.lock").open("w") as lock_stream:
                fcntl.flock(lock_stream, fcntl.LOCK_EX)
                _run_episode(args, object_id=object_id, episode_id=episode_id)
        except Exception as error:
            failures.append(
                {
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                f"FAILED {object_id}/{episode_id}: {error}", file=sys.stderr, flush=True
            )
            if not args.continue_on_error:
                raise
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
