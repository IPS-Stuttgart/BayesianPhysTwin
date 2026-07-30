#!/usr/bin/env python3
"""Run the frozen public-PGND source competence experiment."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_pgnd_source import (
    PGNDMetricTransform,
    build_pgnd_gripper_actions,
    evaluate_pgnd_source_prediction,
    interpolate_model_steps,
    physically_supported_contact_trajectory,
    prepare_pgnd_source_input,
    sha256_file,
    verify_clean_git_checkout,
    verify_pgnd_assets,
)
from bayesian_phystwin.phystwin_pgrd_adapter import (
    deterministic_farthest_point_sample,
)


def _write_json(value: object, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare(args: argparse.Namespace) -> None:
    summary = prepare_pgnd_source_input(
        final_data_path=args.final_data,
        physical_trajectory_path=args.physical_trajectory,
        split_path=args.split,
        output_path=args.output,
    )
    _write_json(summary, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _load_pgnd_runtime(
    checkout: Path,
    checkpoint: Path,
    config: Path,
    device: str,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    import torch
    import warp as wp
    import yaml
    from omegaconf import OmegaConf

    sys.path.insert(0, str(checkout))
    from pgnd.material import PGNDModel
    from pgnd.sim import (
        CacheDiffSimWithFrictionBatch,
        CollidersBatch,
        StaticsBatch,
    )

    with config.open("r", encoding="utf-8") as handle:
        cfg = OmegaConf.create(yaml.load(handle, Loader=yaml.CLoader))
    cfg.sim.num_steps = 1
    cfg.sim.gripper_forcing = False
    cfg.sim.uniform = True
    cfg.gpus = [0]
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({"fast_math": False})
    torch_device = torch.device(device)
    model = PGNDModel(cfg).to(torch_device)
    state = torch.load(checkpoint, map_location=torch_device)
    model.load_state_dict(state["material"])
    model.requires_grad_(False)
    model.eval()
    return (
        torch,
        wp,
        cfg,
        model,
        CacheDiffSimWithFrictionBatch,
        StaticsBatch,
        CollidersBatch,
    )


def _rollout_once(
    *,
    carrier: dict[str, np.ndarray],
    checkout: Path,
    checkpoint: Path,
    config: Path,
    device: str,
) -> dict[str, np.ndarray]:
    (
        torch,
        wp,
        cfg,
        model,
        CacheDiffSimWithFrictionBatch,
        StaticsBatch,
        CollidersBatch,
    ) = _load_pgnd_runtime(checkout, checkpoint, config, device)

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    physical = np.asarray(carrier["physical_trajectory"], dtype=np.float64)
    controller = np.asarray(carrier["controller_points"], dtype=np.float64)
    initialization_frame = int(carrier["initialization_frame"])
    history_frames = tuple(int(value) for value in carrier["history_frames"])
    prediction_frames = tuple(int(value) for value in carrier["prediction_frames"])
    stride = prediction_frames[0] - initialization_frame
    dt_s = stride / 30.0
    sample_indices = deterministic_farthest_point_sample(
        physical[initialization_frame],
        int(cfg.sim.n_particles),
    )
    original_surface_count = int(carrier["num_surface_points"])
    sample_indices = np.concatenate(
        [
            sample_indices[sample_indices < original_surface_count],
            sample_indices[sample_indices >= original_surface_count],
        ]
    )
    sampled_surface_count = int(
        np.count_nonzero(sample_indices < original_surface_count)
    )
    if sampled_surface_count == 0:
        raise ValueError("PGND sample contains no original surface nodes")
    sampled_physical = physical[:, sample_indices]
    transform = PGNDMetricTransform.fit(
        sampled_physical[initialization_frame],
        grid_spacing_m=float(cfg.sim.num_grids[-1]),
        clip_bound_cells=float(cfg.model.clip_bound),
    )
    x_model = transform.positions_to_model(
        sampled_physical[initialization_frame]
    ).astype(np.float32)
    history_model = transform.positions_to_model(
        sampled_physical[np.asarray(history_frames)]
    ).astype(np.float32)
    previous_world = sampled_physical[initialization_frame - stride]
    velocity_world = (sampled_physical[initialization_frame] - previous_world) / dt_s
    velocity_model = transform.velocities_to_model(velocity_world).astype(np.float32)
    history_velocity_world = np.stack(
        [
            (sampled_physical[frame] - sampled_physical[frame - stride]) / dt_s
            for frame in history_frames
        ],
        axis=0,
    )
    history_velocity_model = transform.velocities_to_model(
        history_velocity_world
    ).astype(np.float32)

    action_frames = (initialization_frame,) + prediction_frames
    supported_contact_world, contact_indices = physically_supported_contact_trajectory(
        controller[np.asarray(action_frames)],
        physical[np.asarray(action_frames)],
    )
    contact_model = transform.positions_to_model(supported_contact_world)
    gripper_actions = build_pgnd_gripper_actions(
        contact_model,
        dt_s=dt_s,
        radius_m=float(cfg.model.gripper_radius),
    )

    torch_device = torch.device(device)
    wp_device = wp.get_device(device)
    x = torch.from_numpy(x_model[None]).to(torch_device)
    v = torch.from_numpy(velocity_model[None]).to(torch_device)
    x_history = (
        torch.from_numpy(history_model.transpose(1, 0, 2).reshape(1, len(x_model), -1))
        .to(torch_device)
        .contiguous()
    )
    v_history = (
        torch.from_numpy(
            history_velocity_model.transpose(1, 0, 2).reshape(1, len(x_model), -1)
        )
        .to(torch_device)
        .contiguous()
    )
    enabled = torch.ones((1, len(x_model)), dtype=torch.bool, device=torch_device)
    actions = torch.from_numpy(gripper_actions).to(torch_device)
    friction = torch.tensor(
        [[float(cfg.model.friction.value)]],
        dtype=torch.float32,
        device=torch_device,
    )

    sim = CacheDiffSimWithFrictionBatch(
        cfg,
        len(prediction_frames),
        1,
        wp_device,
        requires_grad=False,
    )
    statics = StaticsBatch()
    statics.init(shape=(1, len(x_model)), device=wp_device)
    statics.update_clip_bound(
        torch.tensor([float(cfg.model.clip_bound)], dtype=torch.float32)
    )
    statics.update_enabled(torch.ones((1, len(x_model)), dtype=torch.bool))
    colliders = CollidersBatch()
    colliders.init(shape=(1, 1), device=wp_device)
    colliders.initialize_grippers(actions[0:1])

    predictions_model = []
    with torch.no_grad():
        for step in range(len(prediction_frames)):
            # PGND applies the action row at the start of each transition. Its
            # velocity carries the gripper from this center to the next center.
            colliders.update_grippers(actions[step : step + 1])
            grid_velocity = model(x, v, x_history, v_history, enabled)
            if not torch.isfinite(grid_velocity).all():
                raise ValueError("PGND produced a non-finite grid velocity")
            x_next, v_next = sim(
                statics,
                colliders,
                step,
                x,
                v,
                friction,
                grid_velocity,
            )
            if not torch.isfinite(x_next).all() or not torch.isfinite(v_next).all():
                raise ValueError("PGND produced a non-finite state")
            predictions_model.append(x_next.detach().cpu().numpy()[0])
            x_history = torch.cat(
                [
                    x_history.reshape(1, len(x_model), -1, 3)[:, :, 1:],
                    x_next[:, :, None].detach(),
                ],
                dim=2,
            ).reshape(1, len(x_model), -1)
            v_history = torch.cat(
                [
                    v_history.reshape(1, len(x_model), -1, 3)[:, :, 1:],
                    v_next[:, :, None].detach(),
                ],
                dim=2,
            ).reshape(1, len(x_model), -1)
            x, v = x_next, v_next
    torch.cuda.synchronize()
    predictions_world = transform.positions_to_world(
        np.asarray(predictions_model)
    ).astype(np.float32)
    full_candidate = interpolate_model_steps(
        physical_prefix=sampled_physical,
        model_prediction_frames=prediction_frames,
        model_predictions=predictions_world,
        initialization_frame=initialization_frame,
        frame_count=physical.shape[0],
    ).astype(np.float32)
    return {
        "candidate_trajectory": full_candidate,
        "equal_support_physical": sampled_physical.astype(np.float32),
        "sample_indices": sample_indices.astype(np.int64),
        "num_surface_points": np.asarray(sampled_surface_count, dtype=np.int64),
        "contact_indices": contact_indices.astype(np.int64),
        "contact_world_m": supported_contact_world.astype(np.float32),
        "prediction_frames": np.asarray(prediction_frames, dtype=np.int64),
        "initialization_frame": np.asarray(initialization_frame, dtype=np.int64),
        "rotation_model_from_world": transform.rotation_model_from_world.astype(
            np.float64
        ),
        "translation_model": transform.translation_model.astype(np.float64),
    }


def _predict(args: argparse.Namespace) -> None:
    implementation = verify_clean_git_checkout(Path(__file__).resolve().parents[2])
    checkout = Path(args.pgnd_checkout).resolve()
    checkpoint = Path(args.pgnd_checkpoint).resolve()
    config = Path(args.pgnd_config).resolve()
    provenance = verify_pgnd_assets(checkout, checkpoint, config)
    with np.load(args.input) as archive:
        carrier = {name: archive[name] for name in archive.files}
    first = _rollout_once(
        carrier=carrier,
        checkout=checkout,
        checkpoint=checkpoint,
        config=config,
        device=args.device,
    )
    second = _rollout_once(
        carrier=carrier,
        checkout=checkout,
        checkpoint=checkpoint,
        config=config,
        device=args.device,
    )
    replay_max_abs_m = float(
        np.max(np.abs(first["candidate_trajectory"] - second["candidate_trajectory"]))
    )
    if replay_max_abs_m != 0.0:
        raise ValueError(
            f"PGND replay is not bit-exact: max difference {replay_max_abs_m}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **first)
    seal = {
        "schema_version": 1,
        "status": "prediction_sealed",
        "claim_boundary": (
            "PGND prediction used only the sealed physical-prior/action carrier. "
            "Future object observations and manual tracks were not loaded."
        ),
        "input": {
            "path": str(Path(args.input).resolve()),
            "sha256": sha256_file(args.input),
        },
        "implementation": implementation,
        "pgnd": provenance,
        "output": {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
        },
        "replay_max_abs_position_m": replay_max_abs_m,
        "device": args.device,
    }
    _write_json(seal, args.seal)
    print(json.dumps(seal, indent=2, sort_keys=True))


def _evaluate(args: argparse.Namespace) -> None:
    implementation = verify_clean_git_checkout(Path(__file__).resolve().parents[2])
    seal = json.loads(Path(args.seal).read_text(encoding="utf-8"))
    if sha256_file(args.prediction) != seal["output"]["sha256"]:
        raise ValueError("prediction bytes changed after sealing")
    with np.load(args.prediction) as archive:
        prediction = {name: archive[name] for name in archive.files}
    with Path(args.physical_trajectory).open("rb") as handle:
        full_physical = np.asarray(pickle.load(handle), dtype=np.float32)
    with Path(args.final_data).open("rb") as handle:
        final_data = pickle.load(handle)
    with Path(args.gt_track).open("rb") as handle:
        gt_track = np.asarray(pickle.load(handle), dtype=np.float32)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    endpoint_frame = int(split["train"][1]) - 1
    current = prediction["equal_support_physical"][endpoint_frame]
    persistence = prediction["equal_support_physical"].copy()
    persistence[endpoint_frame + 1 :] = current
    evaluation = evaluate_pgnd_source_prediction(
        candidate_trajectory=prediction["candidate_trajectory"],
        equal_support_physical=prediction["equal_support_physical"],
        equal_support_surface_count=int(prediction["num_surface_points"]),
        full_physical=full_physical,
        persistence_trajectory=persistence,
        final_data=final_data,
        gt_track_3d=gt_track,
        train_end_exclusive=int(split["train"][1]),
        test_end_exclusive=int(split["test"][1]),
        required_relative_improvement=args.required_improvement,
    )
    summary = {
        "schema_version": 1,
        "status": "source_smoke_evaluated",
        "claim_boundary": (
            "Already-open single-case source competence result. It is not an "
            "independent benchmark, confirmation, or SOTA claim."
        ),
        "prediction_seal": {
            "path": str(Path(args.seal).resolve()),
            "sha256": sha256_file(args.seal),
        },
        "evaluation_implementation": implementation,
        "outcome_inputs": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "physical_trajectory": args.physical_trajectory,
                "final_data": args.final_data,
                "gt_track": args.gt_track,
                "split": args.split,
            }.items()
        },
        "evaluation": evaluation,
    }
    _write_json(summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--final-data", required=True)
    prepare.add_argument("--physical-trajectory", required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--summary", required=True)
    prepare.set_defaults(handler=_prepare)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--input", required=True)
    predict.add_argument("--pgnd-checkout", required=True)
    predict.add_argument("--pgnd-checkpoint", required=True)
    predict.add_argument("--pgnd-config", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--seal", required=True)
    predict.add_argument("--device", default="cuda:0")
    predict.set_defaults(handler=_predict)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--seal", required=True)
    evaluate.add_argument("--physical-trajectory", required=True)
    evaluate.add_argument("--final-data", required=True)
    evaluate.add_argument("--gt-track", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--required-improvement", type=float, default=0.02)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
