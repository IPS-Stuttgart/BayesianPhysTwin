from __future__ import annotations

from .core import (
    Any,
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    DLOS,
    Mapping,
    Path,
    Sequence,
    _identity,
    _load_named_from_manifest,
    _mapping,
    _read_json,
    _verified_file,
    _write_json,
    argparse,
    cast,
    deform_bayesian_covariance_archive_key,
    evaluate_deform_predictive_distribution,
    json,
    np,
    sha256_file,
)
from .model import (
    _point_summary,
    _target_gate,
)
from .predict import (
    _validate_prediction_seal,
)


def _seal(args: argparse.Namespace, protocol: Mapping[str, object]) -> int:
    del protocol
    if args.prediction_seal_dlo4 is None or args.prediction_seal_dlo5 is None:
        raise ValueError("seal requires both DLO prediction seals")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seals = {
        "DLO4": _validate_prediction_seal(
            args.prediction_seal_dlo4.resolve(),
            dlo="DLO4",
            protocol_path=args.protocol.resolve(),
        ),
        "DLO5": _validate_prediction_seal(
            args.prediction_seal_dlo5.resolve(),
            dlo="DLO5",
            protocol_path=args.protocol.resolve(),
        ),
    }
    authorization_sha = {
        _mapping(seal.get("target_authorization"), label=f"{dlo} authorization").get(
            "sha256"
        )
        for dlo, seal in seals.items()
    }
    if len(authorization_sha) != 1:
        raise ValueError("DLO4 and DLO5 predictions use different authorizations")
    joint = {
        "schema_version": 1,
        "contract": "deform-dlo45-joint-prediction-seal-v1",
        "protocol": _identity(args.protocol.resolve()),
        "prediction_seals": {
            "DLO4": _identity(args.prediction_seal_dlo4.resolve()),
            "DLO5": _identity(args.prediction_seal_dlo5.resolve()),
        },
        "target_authorization_sha256": next(iter(authorization_sha)),
        "datasets": list(DLOS),
        "total_target_cases": 28,
        "both_datasets_predicted_before_any_scoring": True,
        "target_outcomes_scored": False,
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "case_replacement": False,
    }
    path = output_root / "joint_prediction_seal.json"
    _write_json(path, joint)
    print(json.dumps(joint, indent=2, sort_keys=True))
    return 0


def _score_one(
    seal_path: Path,
    *,
    dlo: str,
    protocol: Mapping[str, object],
    protocol_path: Path,
) -> dict[str, object]:
    seal = _validate_prediction_seal(seal_path, dlo=dlo, protocol_path=protocol_path)
    predictions_path = _verified_file(seal.get("predictions"), label=f"{dlo} predictions")
    eval_manifest_path = _verified_file(
        seal.get("eval_manifest"), label=f"{dlo} eval manifest"
    )
    eval_manifest = _read_json(eval_manifest_path)
    eval_names = list(cast(Sequence[str], eval_manifest["ordered_names"]))
    trajectories = _load_named_from_manifest(
        eval_manifest,
        eval_names,
        frame_count=500,
        node_count=12,
    )
    targets = np.stack([trajectories[name][2:] for name in eval_names])
    with np.load(predictions_path, allow_pickle=False) as archive:
        names = [str(value) for value in np.asarray(archive["names"]).tolist()]
        if names != eval_names:
            raise ValueError(f"{dlo} prediction order differs from eval manifest")
        candidate = np.asarray(archive["candidate"])
        physical = np.asarray(archive["physical"])
        compute = np.asarray(archive["compute_matched_physical"])
        persistence = np.asarray(archive["persistence"])
        primary = _target_gate(
            _point_summary(candidate, physical, targets, eval_names),
            protocol,
        )
        compute_summary = _point_summary(candidate, compute, targets, eval_names)
        persistence_summary = _point_summary(
            candidate, persistence, targets, eval_names
        )
        distributions = {
            label: evaluate_deform_predictive_distribution(
                candidate,
                targets,
                np.asarray(archive[deform_bayesian_covariance_archive_key(label)]),
            )
            for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        }
    return {
        "dlo": dlo,
        "primary_vs_matching_physical": primary,
        "candidate_vs_compute_matched_physical": compute_summary,
        "candidate_vs_persistence": persistence_summary,
        "bayesian_distributions": distributions,
        "primary_distribution": "calibrated-full-coordinate-covariance-v1",
        "distribution_selection": "none",
        "target_outcomes_used_for_distribution_construction": False,
        "target_outcomes_used_for_distribution_selection": False,
    }


def _score(args: argparse.Namespace, protocol: Mapping[str, object]) -> int:
    if args.joint_seal is None:
        raise ValueError("score requires --joint-seal")
    joint_path = args.joint_seal.resolve()
    joint = _read_json(joint_path)
    if (
        joint.get("contract") != "deform-dlo45-joint-prediction-seal-v1"
        or joint.get("both_datasets_predicted_before_any_scoring") is not True
        or joint.get("target_outcomes_scored") is not False
        or joint.get("target_retries") is not False
        or joint.get("case_replacement") is not False
        or _mapping(joint.get("protocol"), label="joint protocol").get("sha256")
        != sha256_file(args.protocol.resolve())
    ):
        raise ValueError("joint prediction seal differs")
    seal_identities = _mapping(
        joint.get("prediction_seals"), label="joint prediction seals"
    )
    seal_paths = {
        dlo: _verified_file(seal_identities.get(dlo), label=f"{dlo} joint seal entry")
        for dlo in DLOS
    }
    results = {
        dlo: _score_one(
            path,
            dlo=dlo,
            protocol=protocol,
            protocol_path=args.protocol.resolve(),
        )
        for dlo, path in seal_paths.items()
    }
    both_pass = all(
        _mapping(
            result.get("primary_vs_matching_physical"),
            label=f"{dlo} primary result",
        ).get("passed")
        is True
        for dlo, result in results.items()
    )
    pooled_candidate = np.mean(
        [
            float(
                cast(
                    Any,
                    _mapping(
                        result["primary_vs_matching_physical"],
                        label=f"{dlo} primary",
                    )["candidate_mean_l1_m"],
                )
            )
            for dlo, result in results.items()
        ]
    )
    pooled_baseline = np.mean(
        [
            float(
                cast(
                    Any,
                    _mapping(
                        result["primary_vs_matching_physical"],
                        label=f"{dlo} primary",
                    )["baseline_mean_l1_m"],
                )
            )
            for dlo, result in results.items()
        ]
    )
    output = {
        "schema_version": 1,
        "contract": "deform-dlo45-frozen-transfer-result-v1",
        "decision": (
            "both-fresh-dlos-pass-frozen-primary-gate"
            if both_pass
            else "one-or-more-fresh-dlos-fail-frozen-primary-gate"
        ),
        "protocol": _identity(args.protocol.resolve()),
        "joint_prediction_seal": _identity(joint_path),
        "results": results,
        "both_primary_gates_passed": both_pass,
        "equal_dlo_mean_candidate_l1_m": float(pooled_candidate),
        "equal_dlo_mean_physical_l1_m": float(pooled_baseline),
        "equal_dlo_relative_improvement": float(
            1.0 - pooled_candidate / pooled_baseline
        ),
        "total_candidate_wins": sum(
            int(
                cast(
                    Any,
                    _mapping(
                        result["primary_vs_matching_physical"],
                        label=f"{dlo} primary",
                    )["wins"],
                )
            )
            for dlo, result in results.items()
        ),
        "target_case_count": 28,
        "both_datasets_predicted_before_any_scoring": True,
        "target_outcomes_scored": True,
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "case_replacement": False,
        "prob4d_used": False,
        "paper_claim_authorized": False,
        "claim_boundary": (
            "Fresh frozen-procedure replication on the exact released DLO4 and "
            "DLO5 operators only; not zero-shot arbitrary-object generalization, "
            "not a Prob4D result, and not a safety or universal-SOTA claim."
        ),
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "result.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _failure_receipt(args: argparse.Namespace, error: BaseException) -> None:
    try:
        root = args.output_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        state_path = root / "custody_state.json"
        state = _read_json(state_path) if state_path.is_file() else {}
        receipt = {
            "schema_version": 1,
            "contract": "deform-dlo45-terminal-failure-v1",
            "command": args.command,
            "dlo": args.dlo,
            "exception_type": type(error).__name__,
            "message": str(error),
            "custody_state": state,
            "retry_authorized": False,
            "case_replacement": False,
        }
        _write_json(root / "failure_receipt.json", receipt)
    except Exception:
        pass
