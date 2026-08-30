from __future__ import annotations

from .core import (
    INTERNAL,
    Any,
    Array,
    Mapping,
    Path,
    Sequence,
    _protocol_part,
    _write_json,
    cast,
    deserialize_deform_local_residual_model,
    json,
    math,
    np,
    serialize_deform_local_residual_model,
    sha256_file,
    source_runtime,
    time,
)


def _save_model(path: Path, model: Mapping[str, object]) -> None:
    payload = serialize_deform_local_residual_model(model)
    if "coefficient_covariance_full" in model:
        payload["coefficient_covariance_full"] = np.asarray(
            model["coefficient_covariance_full"]
        )
    if "residual_covariance_full" in model:
        payload["residual_covariance_full"] = np.asarray(
            model["residual_covariance_full"]
        )
    np.savez_compressed(path, **payload)


def _load_full_model(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        model = deserialize_deform_local_residual_model(archive)
        model["coefficient_covariance_full"] = np.asarray(
            archive["coefficient_covariance_full"]
        )
        model["residual_covariance_full"] = np.asarray(
            archive["residual_covariance_full"]
        )
    return model


def _train_physical(
    trajectories: Mapping[str, Array],
    names: Sequence[str],
    *,
    protocol: Mapping[str, object],
    upstream_root: Path,
    output_root: Path,
    dlo: str,
    device: str,
    torch: Any,
    smoke: bool = False,
) -> dict[str, object]:
    training = _protocol_part(protocol, "physical_training")
    output_root.mkdir(parents=True, exist_ok=True)
    modules = source_runtime._load_upstream(upstream_root)
    seed = int(cast(Any, training["seed"]))
    source_runtime._seed_everything(torch, seed)
    model_function, model = source_runtime._build_dlo_model(
        modules,
        torch,
        device,
        dlo_type=dlo,
        node_count=int(cast(Any, _protocol_part(protocol, "data")["node_count"])),
        pbd_iterations=int(cast(Any, training["pbd_iterations"])),
    )
    optimizer = source_runtime._official_optimizer(torch, model)
    ordered_names = list(names)
    orientations = source_runtime._precompute_material_u0(
        dict(trajectories),
        modules=modules,
        model_function=model_function,
        torch=torch,
        device=device,
    )
    registered_updates = int(cast(Any, training["total_updates"]))
    maximum_extra = int(cast(Any, training["maximum_compute_matched_updates"]))
    requested_updates = 1 if smoke else registered_updates
    trajectory_indices, start_indices = source_runtime._make_schedule(
        fit_names=ordered_names,
        updates=registered_updates + maximum_extra,
        batch_size=int(cast(Any, training["batch_size"])),
        frame_count=int(cast(Any, _protocol_part(protocol, "data")["frame_count"])),
        horizon=int(cast(Any, training["unroll_horizon_frames"])),
        seed=seed,
    )
    schedule_path = output_root / "window_schedule.npz"
    np.savez_compressed(
        schedule_path,
        fit_names=np.asarray(ordered_names),
        trajectory_indices=trajectory_indices,
        start_indices=start_indices,
    )
    losses: list[dict[str, object]] = []
    started = time.perf_counter()
    for update_index in range(requested_updates):
        batch = source_runtime._assemble_batch(
            dict(trajectories),
            orientations,
            ordered_names,
            trajectory_indices[update_index],
            start_indices[update_index],
            horizon=int(cast(Any, training["unroll_horizon_frames"])),
            torch=torch,
            device=device,
        )
        update_started = time.perf_counter()
        loss = source_runtime._train_update(
            modules=modules,
            model_function=model_function,
            model=model,
            optimizer=optimizer,
            batch=batch,
            torch=torch,
            device=device,
        )
        torch.cuda.synchronize(device)
        update = update_index + 1
        losses.append(
            {
                "update": update,
                "position_l1_m": float(loss),
                "seconds": time.perf_counter() - update_started,
            }
        )
        if update == 1 or update % 50 == 0:
            progress = {
                "dlo": dlo,
                "completed_updates": update,
                "requested_updates": requested_updates,
                "latest_position_l1_m": float(loss),
                "elapsed_seconds": time.perf_counter() - started,
                "target_eval_read": False,
            }
            _write_json(output_root / "progress.json", progress, immutable=False)
            print(json.dumps(progress, sort_keys=True), flush=True)
    checkpoint_path = output_root / f"physical_update_{requested_updates:04d}.pt"
    torch.save(
        {
            "dlo": dlo,
            "update": requested_updates,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "schedule_sha256": sha256_file(schedule_path),
            "target_eval_read": False,
        },
        checkpoint_path,
    )
    return {
        "modules": modules,
        "model_function": model_function,
        "model": model,
        "optimizer": optimizer,
        "orientations": orientations,
        "trajectory_indices": trajectory_indices,
        "start_indices": start_indices,
        "state": model.state_dict(),
        "checkpoint_path": checkpoint_path,
        "schedule_path": schedule_path,
        "losses": losses,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _continue_compute_matched(
    trained: Mapping[str, object],
    trajectories: Mapping[str, Array],
    names: Sequence[str],
    *,
    protocol: Mapping[str, object],
    local_wall_seconds: float,
    output_root: Path,
    device: str,
    torch: Any,
) -> dict[str, object]:
    losses = cast(list[Mapping[str, object]], trained["losses"])
    recent = np.asarray(
        [float(cast(Any, record["seconds"])) for record in losses[-100:]],
        dtype=np.float64,
    )
    median = float(np.median(recent))
    maximum = int(
        cast(
            Any,
            _protocol_part(protocol, "physical_training")[
                "maximum_compute_matched_updates"
            ],
        )
    )
    additional = int(math.ceil(local_wall_seconds / median))
    if not 1 <= additional <= maximum:
        raise RuntimeError("compute-matched continuation is outside frozen bounds")
    training = _protocol_part(protocol, "physical_training")
    registered = int(cast(Any, training["total_updates"]))
    modules = trained["modules"]
    model_function = trained["model_function"]
    model = trained["model"]
    optimizer = trained["optimizer"]
    orientations = cast(Mapping[str, Any], trained["orientations"])
    trajectory_indices = np.asarray(trained["trajectory_indices"])
    start_indices = np.asarray(trained["start_indices"])
    ordered_names = list(names)
    for offset in range(additional):
        index = registered + offset
        batch = source_runtime._assemble_batch(
            dict(trajectories),
            dict(orientations),
            ordered_names,
            trajectory_indices[index],
            start_indices[index],
            horizon=int(cast(Any, training["unroll_horizon_frames"])),
            torch=torch,
            device=device,
        )
        source_runtime._train_update(
            modules=modules,
            model_function=model_function,
            model=model,
            optimizer=optimizer,
            batch=batch,
            torch=torch,
            device=device,
        )
    torch.cuda.synchronize(device)
    path = output_root / f"physical_compute_matched_update_{registered + additional}.pt"
    torch.save(
        {
            "update": registered + additional,
            "additional_updates": additional,
            "model_state_dict": model.state_dict(),
            "target_eval_read": False,
        },
        path,
    )
    return {
        "checkpoint_path": path,
        "state": model.state_dict(),
        "additional_updates": additional,
        "local_residual_wall_seconds": local_wall_seconds,
        "median_update_seconds_6301_6400": median,
        "start_update": registered,
        "end_update": registered + additional,
    }


def _point_summary(
    candidate: Array,
    baseline: Array,
    target: Array,
    names: Sequence[str],
) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.shape != target.shape:
        raise ValueError("point-score arrays do not align")
    if candidate.ndim != 4 or candidate.shape[0] != len(names):
        raise ValueError("point-score trajectory identities do not align")

    def case_l1(values: Array, nodes: slice) -> Array:
        return np.mean(
            np.abs(values[:, :, nodes] - target[:, :, nodes]),
            axis=(1, 2, 3),
        )

    candidate_error = case_l1(candidate, slice(None))
    baseline_error = case_l1(baseline, slice(None))
    candidate_free_error = case_l1(candidate, INTERNAL)
    baseline_free_error = case_l1(baseline, INTERNAL)
    if np.any(baseline_error <= 0.0) or np.any(baseline_free_error <= 0.0):
        raise ValueError("point-score baseline errors must be positive")
    ratios = candidate_error / baseline_error
    free_ratios = candidate_free_error / baseline_free_error
    horizon = candidate.shape[1]
    thirds = (0, horizon // 3, 2 * horizon // 3, horizon)

    def horizon_l1(nodes: slice) -> list[float]:
        return [
            float(
                np.mean(
                    np.abs(
                        candidate[:, thirds[index] : thirds[index + 1], nodes]
                        - target[:, thirds[index] : thirds[index + 1], nodes]
                    )
                )
            )
            for index in range(3)
        ]

    return {
        "metric": "official-mean-coordinate-l1-all-nodes",
        "candidate_mean_l1_m": float(np.mean(candidate_error)),
        "baseline_mean_l1_m": float(np.mean(baseline_error)),
        "relative_improvement": float(
            1.0 - np.mean(candidate_error) / np.mean(baseline_error)
        ),
        "wins": int(np.count_nonzero(candidate_error < baseline_error)),
        "ties": int(np.count_nonzero(candidate_error == baseline_error)),
        "worst_candidate_to_baseline_ratio": float(np.max(ratios)),
        "case_names": list(names),
        "candidate_case_l1_m": candidate_error.tolist(),
        "baseline_case_l1_m": baseline_error.tolist(),
        "case_ratios": ratios.tolist(),
        "candidate_early_middle_late_l1_m": horizon_l1(slice(None)),
        "free_node_diagnostic": {
            "metric": "mean-coordinate-l1-free-nodes-only",
            "candidate_mean_l1_m": float(np.mean(candidate_free_error)),
            "baseline_mean_l1_m": float(np.mean(baseline_free_error)),
            "relative_improvement": float(
                1.0 - np.mean(candidate_free_error) / np.mean(baseline_free_error)
            ),
            "wins": int(np.count_nonzero(candidate_free_error < baseline_free_error)),
            "ties": int(np.count_nonzero(candidate_free_error == baseline_free_error)),
            "worst_candidate_to_baseline_ratio": float(np.max(free_ratios)),
            "candidate_case_l1_m": candidate_free_error.tolist(),
            "baseline_case_l1_m": baseline_free_error.tolist(),
            "case_ratios": free_ratios.tolist(),
            "candidate_early_middle_late_l1_m": horizon_l1(INTERNAL),
        },
    }


def _source_gate(
    candidate: Array,
    baseline: Array,
    target: Array,
    names: Sequence[str],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    summary = _point_summary(candidate, baseline, target, names)
    gate = _protocol_part(protocol, "source_gate")
    passed = (
        float(cast(Any, summary["relative_improvement"]))
        >= float(cast(Any, gate["minimum_relative_improvement"]))
        and int(cast(Any, summary["wins"])) >= int(cast(Any, gate["minimum_case_wins"]))
        and float(cast(Any, summary["worst_candidate_to_baseline_ratio"]))
        <= float(cast(Any, gate["maximum_case_ratio"]))
    )
    return {
        **summary,
        "passed": passed,
        "thresholds": dict(gate),
    }


def _target_gate(
    summary: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    target = _protocol_part(protocol, "target_evaluation")
    passed = (
        float(cast(Any, summary["relative_improvement"]))
        >= float(cast(Any, target["minimum_relative_improvement"]))
        and int(cast(Any, summary["wins"]))
        >= int(cast(Any, target["minimum_case_wins"]))
        and float(cast(Any, summary["worst_candidate_to_baseline_ratio"]))
        <= float(cast(Any, target["maximum_case_ratio"]))
    )
    return {
        **dict(summary),
        "passed": passed,
        "thresholds": {
            "minimum_relative_improvement": target["minimum_relative_improvement"],
            "minimum_case_wins": target["minimum_case_wins"],
            "maximum_case_ratio": target["maximum_case_ratio"],
        },
    }
