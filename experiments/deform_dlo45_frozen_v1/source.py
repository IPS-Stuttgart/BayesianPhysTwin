from __future__ import annotations

from .core import (
    DLOS,
    Any,
    Array,
    Mapping,
    Path,
    _assert_upstream_and_initialization,
    _file_manifest,
    _identity,
    _load_named_from_manifest,
    _load_paths,
    _mapping,
    _partition_names,
    _paths,
    _protocol_part,
    _read_json,
    _setup_torch,
    _verified_file,
    _write_json,
    argparse,
    augment_deform_local_residual_full_covariance,
    build_deform_bayesian_covariance_ablation_v1,
    calibrate_deform_full_covariance,
    cast,
    deform_bayesian_covariance_archive_key,
    evaluate_deform_predictive_distribution,
    fit_deform_local_residual,
    hashlib,
    json,
    local_runtime,
    np,
    os,
    posterior_runtime,
    sha256_file,
    source_runtime,
    sys,
)
from .model import _save_model, _source_gate, _train_physical


def _inventory(args: argparse.Namespace, protocol: Mapping[str, object]) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    torch = _setup_torch(protocol, args.device)
    records: dict[str, object] = {}
    for index, dlo in enumerate(DLOS):
        train_paths = _paths(args.dataset_root.resolve(), dlo, "train")
        eval_paths = _paths(args.dataset_root.resolve(), dlo, "eval")
        provenance = _assert_upstream_and_initialization(
            protocol, args.upstream_root.resolve(), dlo
        )
        record: dict[str, object] = {
            "train_count": len(train_paths),
            "eval_count": len(eval_paths),
            "train_names_sha256": hashlib.sha256(
                "\n".join(path.name for path in train_paths).encode()
            ).hexdigest(),
            "eval_names_sha256": hashlib.sha256(
                "\n".join(path.name for path in eval_paths).encode()
            ).hexdigest(),
            "train_bytes": sum(path.stat().st_size for path in train_paths),
            "eval_bytes": sum(path.stat().st_size for path in eval_paths),
            **provenance,
        }
        if args.smoke:
            subset = _load_paths(
                train_paths[:4],
                frame_count=500,
                node_count=12,
            )
            smoke_root = output_root / f"{dlo.lower()}-smoke"
            trained = _train_physical(
                subset,
                list(subset),
                protocol=protocol,
                upstream_root=args.upstream_root.resolve(),
                output_root=smoke_root,
                dlo=dlo,
                device=f"cuda:{index}",
                torch=torch,
                smoke=True,
            )
            record["one_update_smoke"] = {
                "checkpoint": _identity(cast(Path, trained["checkpoint_path"])),
                "loss": cast(list[Mapping[str, object]], trained["losses"])[0],
            }
        records[dlo] = record
    result = {
        "schema_version": 1,
        "contract": "deform-dlo45-inventory-v1",
        "protocol": _identity(args.protocol.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "dlos": records,
        "trajectory_payloads_hashed": False,
        "eval_trajectories_deserialized": False,
        "target_outcomes_scored": False,
        "next_stage": "source-qualification",
    }
    _write_json(output_root / "inventory.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _source(args: argparse.Namespace, protocol: Mapping[str, object]) -> int:
    if args.dlo is None:
        raise ValueError("source command requires --dlo")
    dlo = args.dlo
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"source output root is not empty: {output_root}")
    output_root.mkdir(parents=True)
    _assert_upstream_and_initialization(protocol, args.upstream_root.resolve(), dlo)
    data_root = args.dataset_root.resolve()
    for target_dlo in DLOS:
        source_runtime._install_eval_read_guard(data_root / target_dlo / "eval")
    train_paths = _paths(data_root, dlo, "train")
    names = [path.name for path in train_paths]
    partitions = _partition_names(names, dlo=dlo, protocol=protocol)
    manifest = {
        "schema_version": 1,
        "contract": "deform-dlo45-source-manifest-v1",
        "dlo": dlo,
        "partition": f"{dlo}/train",
        "trajectories": _file_manifest(train_paths, hash_payload=True),
        "ordered_names": names,
        "partitions": {key: list(value) for key, value in partitions.items()},
        "eval_enumerated": False,
        "eval_read": False,
    }
    manifest_path = output_root / "source_manifest.json"
    _write_json(manifest_path, manifest)
    torch = _setup_torch(protocol, args.device)
    data = _protocol_part(protocol, "data")
    development_names = list(partitions["fit"] + partitions["calibration"])
    development = _load_named_from_manifest(
        manifest,
        development_names,
        frame_count=int(cast(Any, data["frame_count"])),
        node_count=int(cast(Any, data["node_count"])),
    )
    fit_names = list(partitions["fit"])
    calibration_names = list(partitions["calibration"])
    fit = {name: development[name] for name in fit_names}
    calibration = {name: development[name] for name in calibration_names}
    trained = _train_physical(
        fit,
        fit_names,
        protocol=protocol,
        upstream_root=args.upstream_root.resolve(),
        output_root=output_root / "physical",
        dlo=dlo,
        device=args.device,
        torch=torch,
    )
    state = cast(Mapping[str, Any], trained["state"])
    fit_rollout = posterior_runtime._evaluate_state(
        dict(state),
        fit,
        modules=trained["modules"],
        torch=torch,
        device=args.device,
        dlo_type=dlo,
        node_count=12,
    )
    fit_initial, fit_action = local_runtime._causal_inputs(fit, fit_names)
    residual = _protocol_part(protocol, "local_residual")
    local_model = fit_deform_local_residual(
        fit_initial,
        fit_action,
        np.asarray(fit_rollout["predictions"]),
        np.asarray(fit_rollout["targets"]),
        fit_names,
        ridge=float(cast(Any, residual["ridge"])),
        variance_floor_m2=float(cast(Any, residual["coordinate_variance_floor_m2"])),
    )
    full_model = augment_deform_local_residual_full_covariance(
        local_model,
        fit_initial,
        fit_action,
        np.asarray(fit_rollout["predictions"]),
        np.asarray(fit_rollout["targets"]),
        fit_names,
    )
    model_path = output_root / "full_covariance_model.npz"
    _save_model(model_path, full_model)
    calibration_rollout = posterior_runtime._evaluate_state(
        dict(state),
        calibration,
        modules=trained["modules"],
        torch=torch,
        device=args.device,
        dlo_type=dlo,
        node_count=12,
    )
    calibration_initial, calibration_action = local_runtime._causal_inputs(
        calibration, calibration_names
    )
    raw_calibration = build_deform_bayesian_covariance_ablation_v1(
        full_model,
        calibration_initial,
        calibration_action,
        np.asarray(calibration_rollout["predictions"]),
        shrinkage=float(cast(Any, residual["shrinkage"])),
        variance_scale=1.0,
    )["trajectory-clustered-full-coordinate-covariance-v1"]
    calibration_record = calibrate_deform_full_covariance(
        np.asarray(raw_calibration["predictions"]),
        np.asarray(calibration_rollout["targets"]),
        np.asarray(raw_calibration["coordinate_covariance_m2"]),
    )
    calibration_json = {
        **calibration_record,
        "trajectory_scores": np.asarray(
            calibration_record["trajectory_scores"]
        ).tolist(),
        "source_test_opened": False,
        "target_eval_read": False,
    }
    calibration_path = output_root / "covariance_calibration.json"
    _write_json(calibration_path, calibration_json)
    method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo45-source-method-seal-v1",
        "dlo": dlo,
        "protocol": _identity(args.protocol.resolve()),
        "source_manifest": _identity(manifest_path),
        "physical_checkpoint": _identity(
            cast(Path, trained["checkpoint_path"]), update=6400
        ),
        "window_schedule": _identity(cast(Path, trained["schedule_path"])),
        "full_covariance_model": _identity(model_path),
        "covariance_calibration": _identity(calibration_path),
        "source_test_opened": False,
        "target_eval_read": False,
        "target_selection": False,
        "target_calibration": False,
    }
    method_seal_path = output_root / "method_seal.json"
    _write_json(method_seal_path, method_seal)

    source_test_names = list(partitions["source_test"])
    source_test = _load_named_from_manifest(
        manifest,
        source_test_names,
        frame_count=500,
        node_count=12,
    )
    source_rollout = posterior_runtime._evaluate_state(
        dict(state),
        source_test,
        modules=trained["modules"],
        torch=torch,
        device=args.device,
        dlo_type=dlo,
        node_count=12,
    )
    source_initial, source_action = local_runtime._causal_inputs(
        source_test, source_test_names
    )
    bayesian = build_deform_bayesian_covariance_ablation_v1(
        full_model,
        source_initial,
        source_action,
        np.asarray(source_rollout["predictions"]),
        shrinkage=float(cast(Any, residual["shrinkage"])),
        variance_scale=float(cast(Any, calibration_record["variance_scale"])),
    )
    candidate = bayesian["trajectory-clustered-full-coordinate-covariance-v1"]
    prediction_path = output_root / "source_predictions.npz"
    payload: dict[str, Array] = {
        "names": np.asarray(source_test_names),
        "physical": np.asarray(source_rollout["predictions"]),
        "candidate": np.asarray(candidate["predictions"]),
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
        "contract": "deform-dlo45-source-prediction-seal-v1",
        "dlo": dlo,
        "method_seal": _identity(method_seal_path),
        "predictions": _identity(prediction_path),
        "source_test_case_count": 8,
        "source_outcomes_scored": False,
        "target_eval_read": False,
    }
    prediction_seal_path = output_root / "prediction_seal.json"
    _write_json(prediction_seal_path, prediction_seal)

    targets = np.asarray(source_rollout["targets"])
    gate = _source_gate(
        np.asarray(candidate["predictions"]),
        np.asarray(source_rollout["predictions"]),
        targets,
        source_test_names,
        protocol,
    )
    distributions = {
        label: evaluate_deform_predictive_distribution(
            np.asarray(prediction["predictions"]),
            targets,
            np.asarray(prediction["coordinate_covariance_m2"]),
        )
        for label, prediction in bayesian.items()
    }
    result = {
        "schema_version": 1,
        "contract": "deform-dlo45-source-result-v1",
        "dlo": dlo,
        "protocol": _identity(args.protocol.resolve()),
        "source_manifest": _identity(manifest_path),
        "method_seal": _identity(method_seal_path),
        "prediction_seal": _identity(prediction_seal_path),
        "source_gate": gate,
        "covariance_calibration": calibration_json,
        "bayesian_distributions": distributions,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
        "source_test_opened": True,
        "target_eval_enumerated": False,
        "target_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
    }
    result_path = output_root / "source_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _validate_source_result(
    path: Path,
    *,
    dlo: str,
    protocol_path: Path,
) -> dict[str, object]:
    result = _read_json(path)
    if (
        result.get("contract") != "deform-dlo45-source-result-v1"
        or result.get("dlo") != dlo
        or result.get("source_test_opened") is not True
        or result.get("target_eval_read") is not False
        or result.get("target_authorized") is not False
        or result.get("retry_authorized") is not False
        or _mapping(result.get("protocol"), label="source protocol").get("sha256")
        != sha256_file(protocol_path)
        or _mapping(result.get("source_gate"), label="source gate").get("passed")
        is not True
    ):
        raise ValueError(f"{dlo} source result did not authorize target prediction")
    _verified_file(result.get("method_seal"), label=f"{dlo} source method seal")
    _verified_file(result.get("prediction_seal"), label=f"{dlo} source prediction seal")
    return result


def _authorize(args: argparse.Namespace, protocol: Mapping[str, object]) -> int:
    if (
        args.source_result_dlo4 is None
        or args.source_result_dlo5 is None
        or args.request is None
    ):
        raise ValueError("authorize requires both source results and --request")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = {
        "DLO4": _validate_source_result(
            args.source_result_dlo4.resolve(),
            dlo="DLO4",
            protocol_path=args.protocol.resolve(),
        ),
        "DLO5": _validate_source_result(
            args.source_result_dlo5.resolve(),
            dlo="DLO5",
            protocol_path=args.protocol.resolve(),
        ),
    }
    request = _read_json(args.request.resolve())
    if (
        request.get("schema") != "deform-dlo45-evaluation-request-v1"
        or request.get("mode") != "evaluate"
        or request.get("dataset_root") != str(args.dataset_root.resolve())
        or request.get("upstream_root") != str(args.upstream_root.resolve())
        or request.get("runner_label") != "gpuserver4090"
        or request.get("devices") != ["cuda:0", "cuda:1"]
        or request.get("authorize_target_scoring") is not True
        or request.get("paper_claim_authorized") is not False
        or request.get("evidence_class")
        != (
            "fresh frozen-procedure DLO4/DLO5 replication; "
            "joint prediction seal before scoring; no retry"
        )
    ):
        raise ValueError("evaluation request differs from the frozen contract")
    authorization = {
        "schema_version": 1,
        "contract": "deform-dlo45-target-authorization-v1",
        "protocol": _identity(args.protocol.resolve()),
        "request": _identity(args.request.resolve()),
        "source_results": {
            dlo: _identity(
                args.source_result_dlo4.resolve()
                if dlo == "DLO4"
                else args.source_result_dlo5.resolve()
            )
            for dlo in DLOS
        },
        "source_gates": {
            dlo: _mapping(result.get("source_gate"), label=f"{dlo} source gate")
            for dlo, result in results.items()
        },
        "target_authorized": True,
        "joint_prediction_seal_required": True,
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "case_replacement": False,
        "target_eval_enumerated": False,
        "target_eval_read": False,
        "target_outcomes_scored": False,
    }
    path = output_root / "target_authorization.json"
    _write_json(path, authorization)
    print(json.dumps(authorization, indent=2, sort_keys=True))
    return 0
