from __future__ import annotations

from .core import (
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    Any,
    Mapping,
    Path,
    Sequence,
    _assert_upstream_and_initialization,
    _file_manifest,
    _identity,
    _load_named_from_manifest,
    _mapping,
    _paths,
    _protocol_part,
    _read_json,
    _setup_torch,
    _verified_file,
    _write_json,
    argparse,
    augment_deform_local_residual_full_covariance,
    build_deform_bayesian_covariance_ablation_v1,
    cast,
    deform_bayesian_covariance_archive_key,
    fit_deform_local_residual,
    json,
    local_runtime,
    np,
    posterior_runtime,
    sha256_file,
    time,
)
from .model import _continue_compute_matched, _save_model, _train_physical
from .source import _validate_source_result


def _validate_authorization(
    path: Path,
    *,
    protocol_path: Path,
    source_result_path: Path,
    dlo: str,
) -> dict[str, object]:
    authorization = _read_json(path)
    source_results = _mapping(
        authorization.get("source_results"), label="authorization source results"
    )
    if (
        authorization.get("contract") != "deform-dlo45-target-authorization-v1"
        or authorization.get("target_authorized") is not True
        or authorization.get("joint_prediction_seal_required") is not True
        or authorization.get("target_eval_read") is not False
        or authorization.get("target_outcomes_scored") is not False
        or authorization.get("target_retries") is not False
        or authorization.get("case_replacement") is not False
        or _mapping(authorization.get("protocol"), label="authorization protocol").get(
            "sha256"
        )
        != sha256_file(protocol_path)
        or _mapping(source_results.get(dlo), label=f"{dlo} authorized source").get(
            "sha256"
        )
        != sha256_file(source_result_path)
    ):
        raise ValueError("target authorization differs")
    return authorization


def _predict(args: argparse.Namespace, protocol: Mapping[str, object]) -> int:
    if args.dlo is None or args.source_result is None or args.authorization is None:
        raise ValueError("predict requires --dlo, --source-result, and --authorization")
    dlo = args.dlo
    source_result_path = args.source_result.resolve()
    source_result = _validate_source_result(
        source_result_path,
        dlo=dlo,
        protocol_path=args.protocol.resolve(),
    )
    _validate_authorization(
        args.authorization.resolve(),
        protocol_path=args.protocol.resolve(),
        source_result_path=source_result_path,
        dlo=dlo,
    )
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"target output root is not empty: {output_root}")
    output_root.mkdir(parents=True)
    _write_json(
        output_root / "custody_state.json",
        {
            "stage": "alltrain-source-fit",
            "dlo": dlo,
            "target_eval_enumerated": False,
            "target_eval_read": False,
            "target_outcomes_scored": False,
            "retry_authorized": False,
        },
        immutable=False,
    )
    _assert_upstream_and_initialization(protocol, args.upstream_root.resolve(), dlo)
    source_manifest_path = _verified_file(
        source_result.get("source_manifest"), label=f"{dlo} source manifest"
    )
    source_manifest = _read_json(source_manifest_path)
    all_names = list(cast(Sequence[str], source_manifest.get("ordered_names", ())))
    data = _protocol_part(protocol, "data")
    all_trajectories = _load_named_from_manifest(
        source_manifest,
        all_names,
        frame_count=int(cast(Any, data["frame_count"])),
        node_count=int(cast(Any, data["node_count"])),
    )
    torch = _setup_torch(protocol, args.device)
    trained = _train_physical(
        all_trajectories,
        all_names,
        protocol=protocol,
        upstream_root=args.upstream_root.resolve(),
        output_root=output_root / "alltrain",
        dlo=dlo,
        device=args.device,
        torch=torch,
    )
    checkpoint_bundle = torch.load(
        cast(Path, trained["checkpoint_path"]),
        map_location="cpu",
        weights_only=True,
    )
    state = cast(Mapping[str, Any], checkpoint_bundle["model_state_dict"])
    rollout = posterior_runtime._evaluate_state(
        dict(state),
        all_trajectories,
        modules=trained["modules"],
        torch=torch,
        device=args.device,
        dlo_type=dlo,
        node_count=12,
    )
    initial, action = local_runtime._causal_inputs(all_trajectories, all_names)
    residual = _protocol_part(protocol, "local_residual")
    local_started = time.perf_counter()
    local_model = fit_deform_local_residual(
        initial,
        action,
        np.asarray(rollout["predictions"]),
        np.asarray(rollout["targets"]),
        all_names,
        ridge=float(cast(Any, residual["ridge"])),
        variance_floor_m2=float(cast(Any, residual["coordinate_variance_floor_m2"])),
    )
    local_wall_seconds = time.perf_counter() - local_started
    full_model = augment_deform_local_residual_full_covariance(
        local_model,
        initial,
        action,
        np.asarray(rollout["predictions"]),
        np.asarray(rollout["targets"]),
        all_names,
    )
    full_model_path = output_root / "full_covariance_model.npz"
    _save_model(full_model_path, full_model)
    compute = _continue_compute_matched(
        trained,
        all_trajectories,
        all_names,
        protocol=protocol,
        local_wall_seconds=local_wall_seconds,
        output_root=output_root,
        device=args.device,
        torch=torch,
    )
    variance_scale = float(
        cast(
            Any,
            _mapping(
                source_result.get("covariance_calibration"),
                label="source covariance calibration",
            )["variance_scale"],
        )
    )
    method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo45-alltrain-method-seal-v1",
        "dlo": dlo,
        "protocol": _identity(args.protocol.resolve()),
        "source_result": _identity(source_result_path),
        "target_authorization": _identity(args.authorization.resolve()),
        "physical_checkpoint": _identity(cast(Path, trained["checkpoint_path"]), update=6400),
        "compute_matched_checkpoint": _identity(
            cast(Path, compute["checkpoint_path"]), update=compute["end_update"]
        ),
        "compute_match": {
            key: value
            for key, value in compute.items()
            if key not in {"state", "checkpoint_path"}
        },
        "window_schedule": _identity(cast(Path, trained["schedule_path"])),
        "full_covariance_model": _identity(full_model_path),
        "ridge": residual["ridge"],
        "shrinkage": residual["shrinkage"],
        "variance_scale": variance_scale,
        "bayesian_ablation_distributions": list(
            DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        ),
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "target_eval_enumerated": False,
        "target_eval_read": False,
        "target_outcomes_scored": False,
    }
    method_seal_path = output_root / "method_seal.json"
    _write_json(method_seal_path, method_seal)

    eval_paths = _paths(args.dataset_root.resolve(), dlo, "eval")
    eval_manifest = {
        "schema_version": 1,
        "contract": "deform-dlo45-eval-manifest-v1",
        "dlo": dlo,
        "partition": f"{dlo}/eval",
        "trajectory_policy": "all-fourteen-sorted-once-no-replacement",
        "trajectories": _file_manifest(eval_paths, hash_payload=True),
        "ordered_names": [path.name for path in eval_paths],
        "method_seal": _identity(method_seal_path),
        "target_eval_read": True,
        "target_outcomes_scored": False,
    }
    eval_manifest_path = output_root / "eval_manifest.json"
    _write_json(eval_manifest_path, eval_manifest)
    _write_json(
        output_root / "custody_state.json",
        {
            "stage": "target-prediction",
            "dlo": dlo,
            "target_eval_enumerated": True,
            "target_eval_read": True,
            "target_outcomes_scored": False,
            "retry_authorized": False,
        },
        immutable=False,
    )
    eval_trajectories = _load_named_from_manifest(
        eval_manifest,
        cast(Sequence[str], eval_manifest["ordered_names"]),
        frame_count=500,
        node_count=12,
    )
    eval_names = list(eval_trajectories)
    physical_rollout = posterior_runtime._evaluate_state(
        dict(state),
        eval_trajectories,
        modules=trained["modules"],
        torch=torch,
        device=args.device,
        dlo_type=dlo,
        node_count=12,
    )
    compute_bundle = torch.load(
        cast(Path, compute["checkpoint_path"]),
        map_location="cpu",
        weights_only=True,
    )
    compute_rollout = posterior_runtime._evaluate_state(
        dict(cast(Mapping[str, Any], compute_bundle["model_state_dict"])),
        eval_trajectories,
        modules=trained["modules"],
        torch=torch,
        device=args.device,
        dlo_type=dlo,
        node_count=12,
    )
    eval_initial, eval_action = local_runtime._causal_inputs(eval_trajectories, eval_names)
    bayesian = build_deform_bayesian_covariance_ablation_v1(
        full_model,
        eval_initial,
        eval_action,
        np.asarray(physical_rollout["predictions"]),
        shrinkage=float(cast(Any, residual["shrinkage"])),
        variance_scale=variance_scale,
    )
    primary = bayesian["calibrated-full-coordinate-covariance-v1"]
    prediction_path = output_root / "target_predictions.npz"
    payload = {
        "names": np.asarray(eval_names),
        "physical": np.asarray(physical_rollout["predictions"]),
        "compute_matched_physical": np.asarray(compute_rollout["predictions"]),
        "persistence": np.asarray(physical_rollout["persistence"]),
        "candidate": np.asarray(primary["predictions"]),
    }
    payload.update(
        {
            deform_bayesian_covariance_archive_key(label): np.asarray(
                prediction["coordinate_covariance_m2"]
            )
            for label, prediction in bayesian.items()
        }
    )
    np.savez_compressed(prediction_path, **cast(dict[str, Any], payload))
    prediction_seal = {
        "schema_version": 1,
        "contract": "deform-dlo45-target-prediction-seal-v1",
        "dlo": dlo,
        "protocol": _identity(args.protocol.resolve()),
        "source_result": _identity(source_result_path),
        "target_authorization": _identity(args.authorization.resolve()),
        "method_seal": _identity(method_seal_path),
        "eval_manifest": _identity(eval_manifest_path),
        "predictions": _identity(prediction_path),
        "target_case_count": 14,
        "bayesian_ablation_distributions": list(
            DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        ),
        "point_mean_count": 1,
        "target_eval_read": True,
        "target_outcomes_scored": False,
        "retry_authorized": False,
        "case_replacement": False,
    }
    prediction_seal_path = output_root / "prediction_seal.json"
    _write_json(prediction_seal_path, prediction_seal)
    print(json.dumps(prediction_seal, indent=2, sort_keys=True))
    return 0


def _validate_prediction_seal(
    path: Path,
    *,
    dlo: str,
    protocol_path: Path,
) -> dict[str, object]:
    seal = _read_json(path)
    if (
        seal.get("contract") != "deform-dlo45-target-prediction-seal-v1"
        or seal.get("dlo") != dlo
        or int(cast(Any, seal.get("target_case_count", -1))) != 14
        or seal.get("point_mean_count") != 1
        or seal.get("target_eval_read") is not True
        or seal.get("target_outcomes_scored") is not False
        or seal.get("retry_authorized") is not False
        or seal.get("case_replacement") is not False
        or _mapping(seal.get("protocol"), label="seal protocol").get("sha256")
        != sha256_file(protocol_path)
    ):
        raise ValueError(f"{dlo} prediction seal differs")
    for key in ("method_seal", "eval_manifest", "predictions", "target_authorization"):
        _verified_file(seal.get(key), label=f"{dlo} {key}")
    return seal
