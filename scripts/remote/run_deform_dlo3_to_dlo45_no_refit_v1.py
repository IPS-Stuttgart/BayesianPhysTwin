#!/usr/bin/env python3
"""Apply sealed DLO3 residual coefficients to DLO4/DLO5 without refitting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.deform_dlo_cross_object_transfer_v1 import (
    DLOS,
    SEEDS,
    evaluate_cross_object_transfer,
    feature_support_summary,
    load_cross_object_transfer_protocol,
)
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deform_causal_inputs,
    deserialize_deform_local_residual_model,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file
from experiments.deform_dlo45_frozen_v1.core import _load_named_from_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dlo45-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), default="run")
    return parser.parse_args()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identity(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _verified_file(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    raw_path = identity.get("path", identity.get("repository_path", ""))
    path = Path(str(raw_path)).resolve(strict=True)
    if (
        path.stat().st_size != int(cast(Any, identity.get("size_bytes", -1)))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity differs")
    return path


def _load_model(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return deserialize_deform_local_residual_model(archive)


def _validate_parent_result(
    path: Path,
    *,
    protocol_sha256: str,
    joint_seal_path: Path,
) -> dict[str, object]:
    result = _read_json(path)
    if (
        result.get("schema_version") != 1
        or result.get("contract") != "deform-dlo45-frozen-transfer-result-v1"
        or _mapping(result.get("protocol"), label="parent result protocol").get(
            "sha256"
        )
        != protocol_sha256
        or _mapping(
            result.get("joint_prediction_seal"),
            label="parent result joint seal",
        ).get("sha256")
        != sha256_file(joint_seal_path)
        or result.get("both_datasets_predicted_before_any_scoring") is not True
        or result.get("target_outcomes_scored") is not True
        or result.get("target_selection") is not False
        or result.get("target_calibration") is not False
        or result.get("target_retries") is not False
        or result.get("case_replacement") is not False
        or result.get("prob4d_used") is not False
        or result.get("paper_claim_authorized") is not False
        or int(cast(Any, result.get("target_case_count", -1))) != 28
    ):
        raise ValueError("parent DLO4/DLO5 result differs")
    if set(_mapping(result.get("results"), label="parent DLO results")) != set(
        DLOS
    ):
        raise ValueError("parent DLO result set differs")
    return result


def _validate_joint_seal(
    path: Path,
    *,
    protocol_sha256: str,
) -> dict[str, object]:
    seal = _read_json(path)
    if (
        seal.get("schema_version") != 1
        or seal.get("contract") != "deform-dlo45-joint-prediction-seal-v1"
        or _mapping(seal.get("protocol"), label="joint protocol").get("sha256")
        != protocol_sha256
        or tuple(
            str(value)
            for value in cast(Sequence[object], seal.get("datasets", ()))
        )
        != DLOS
        or int(cast(Any, seal.get("total_target_cases", -1))) != 28
        or seal.get("both_datasets_predicted_before_any_scoring") is not True
        or seal.get("target_outcomes_scored") is not False
        or seal.get("target_selection") is not False
        or seal.get("target_calibration") is not False
        or seal.get("target_retries") is not False
        or seal.get("case_replacement") is not False
    ):
        raise ValueError("parent DLO4/DLO5 joint seal differs")
    if set(
        _mapping(seal.get("prediction_seals"), label="joint prediction seals")
    ) != set(DLOS):
        raise ValueError("joint prediction-seal set differs")
    return seal


def _validate_prediction_seal(
    path: Path,
    *,
    dlo: str,
    protocol_sha256: str,
) -> dict[str, object]:
    seal = _read_json(path)
    if (
        seal.get("schema_version") != 1
        or seal.get("contract") != "deform-dlo45-target-prediction-seal-v1"
        or seal.get("dlo") != dlo
        or int(cast(Any, seal.get("target_case_count", -1))) != 14
        or int(cast(Any, seal.get("point_mean_count", -1))) != 1
        or _mapping(seal.get("protocol"), label=f"{dlo} protocol").get("sha256")
        != protocol_sha256
        or seal.get("target_eval_read") is not True
        or seal.get("target_outcomes_scored") is not False
        or seal.get("retry_authorized") is not False
        or seal.get("case_replacement") is not False
    ):
        raise ValueError(f"{dlo} prediction seal differs")
    for key in ("method_seal", "eval_manifest", "predictions", "target_authorization"):
        _verified_file(seal.get(key), label=f"{dlo} {key}")
    return seal


def _load_eval_panel(
    manifest_path: Path,
    *,
    dlo: str,
    frame_count: int,
    node_count: int,
) -> tuple[list[str], np.ndarray]:
    manifest = _read_json(manifest_path)
    names = manifest.get("ordered_names")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("contract") != "deform-dlo45-eval-manifest-v1"
        or manifest.get("dlo") != dlo
        or manifest.get("partition") != f"{dlo}/eval"
        or manifest.get("trajectory_policy")
        != "all-fourteen-sorted-once-no-replacement"
        or not isinstance(names, list)
        or len(names) != 14
        or len(set(names)) != 14
        or not all(isinstance(name, str) and name for name in names)
        or manifest.get("target_eval_read") is not True
        or manifest.get("target_outcomes_scored") is not False
    ):
        raise ValueError(f"{dlo} eval manifest differs")
    trajectories = _load_named_from_manifest(
        manifest,
        cast(Sequence[str], names),
        frame_count=frame_count,
        node_count=node_count,
    )
    for name in names:
        identity = _mapping(
            _mapping(manifest["trajectories"], label="eval trajectories")[name],
            label=f"{dlo} trajectory {name}",
        )
        if f"/{dlo}/eval/" not in Path(str(identity["path"])).as_posix():
            raise ValueError(f"{dlo} eval trajectory left its partition")
    return cast(list[str], names), np.stack([trajectories[name] for name in names])


def _load_prediction_archive(
    path: Path,
    *,
    expected_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = [str(value) for value in np.asarray(archive["names"]).tolist()]
        if names != list(expected_names):
            raise ValueError("target prediction order differs from eval manifest")
        physical = np.asarray(archive["physical"], dtype=np.float64)
        object_specific = np.asarray(archive["candidate"], dtype=np.float64)
    if (
        physical.ndim != 4
        or object_specific.shape != physical.shape
        or not np.isfinite(physical).all()
        or not np.isfinite(object_specific).all()
    ):
        raise ValueError("target prediction archive contains invalid point arrays")
    return physical, object_specific


def _verify_parent_point_summary(
    observed: Mapping[str, object],
    parent: Mapping[str, object],
    *,
    dlo: str,
) -> None:
    comparisons = (
        ("candidate_mean_l1_m", "candidate_mean_l1_m"),
        ("baseline_mean_l1_m", "baseline_mean_l1_m"),
        ("relative_improvement", "relative_improvement"),
        ("maximum_case_ratio", "worst_candidate_to_baseline_ratio"),
    )
    for observed_key, parent_key in comparisons:
        if not math.isclose(
            float(cast(Any, observed[observed_key])),
            float(cast(Any, parent[parent_key])),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{dlo} parent {parent_key} does not reproduce")
    observed_names = [
        str(case["name"])
        for case in cast(Sequence[Mapping[str, object]], observed["cases"])
    ]
    if (
        int(cast(Any, observed["wins"])) != int(cast(Any, parent["wins"]))
        or observed_names
        != [str(value) for value in cast(Sequence[object], parent["case_names"])]
    ):
        raise ValueError(f"{dlo} parent point summary does not reproduce")


def _write_csv(path: Path, result: Mapping[str, object]) -> None:
    dlo_results = _mapping(result["results"], label="DLO results")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "dlo",
                "name",
                "matching_object_physical_l1_m",
                "matching_object_fitted_residual_l1_m",
                "dlo3_seed_42_no_refit_l1_m",
                "dlo3_seed_43_no_refit_l1_m",
                "dlo3_seed_44_no_refit_l1_m",
                "dlo3_equal_seed_no_refit_l1_m",
            ],
        )
        writer.writeheader()
        for dlo in DLOS:
            dlo_result = _mapping(dlo_results[dlo], label=f"{dlo} result")
            primary = _mapping(
                dlo_result["primary_vs_matching_physical"],
                label=f"{dlo} primary",
            )
            specific = _mapping(
                dlo_result["matching_object_residual_vs_physical"],
                label=f"{dlo} object-specific",
            )
            seeds = _mapping(
                dlo_result["individual_seed_vs_matching_physical"],
                label=f"{dlo} seeds",
            )
            primary_cases = cast(Sequence[Mapping[str, object]], primary["cases"])
            specific_cases = cast(Sequence[Mapping[str, object]], specific["cases"])
            seed_cases = {
                seed: cast(
                    Sequence[Mapping[str, object]],
                    _mapping(seeds[str(seed)], label=f"{dlo} seed {seed}")["cases"],
                )
                for seed in SEEDS
            }
            for index, case in enumerate(primary_cases):
                writer.writerow(
                    {
                        "dlo": dlo,
                        "name": case["name"],
                        "matching_object_physical_l1_m": case["baseline_l1_m"],
                        "matching_object_fitted_residual_l1_m": specific_cases[index][
                            "candidate_l1_m"
                        ],
                        "dlo3_seed_42_no_refit_l1_m": seed_cases[42][index][
                            "candidate_l1_m"
                        ],
                        "dlo3_seed_43_no_refit_l1_m": seed_cases[43][index][
                            "candidate_l1_m"
                        ],
                        "dlo3_seed_44_no_refit_l1_m": seed_cases[44][index][
                            "candidate_l1_m"
                        ],
                        "dlo3_equal_seed_no_refit_l1_m": case["candidate_l1_m"],
                    }
                )


def _write_report(path: Path, result: Mapping[str, object]) -> None:
    dlo_results = _mapping(result["results"], label="DLO results")
    equal_dlo = _mapping(result["equal_dlo_summary"], label="equal-DLO summary")
    parent = _mapping(result["parent_dlo45_result"], label="parent result")
    support = _mapping(result["feature_support"], label="feature support")
    interval = cast(
        Sequence[float],
        equal_dlo["stratified_trajectory_bootstrap_95_interval_m"],
    )
    lines = [
        "# DLO3 residual coefficient transfer to DLO4 and DLO5",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Parent prospective decision: **{parent['decision']}**",
        "- Registration class: outcome-blind pre-score secondary diagnostic",
        "",
        "## Per-DLO no-refit results",
        "",
        (
            "| DLO | Physical (mm) | Matching residual (mm) | "
            "DLO3 transfer (mm) | Improvement | Wins | "
            "Seed models improving | Supported |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for dlo in DLOS:
        dlo_result = _mapping(dlo_results[dlo], label=f"{dlo} result")
        methods = _mapping(dlo_result["methods"], label=f"{dlo} methods")
        primary = _mapping(
            dlo_result["primary_vs_matching_physical"],
            label=f"{dlo} primary",
        )
        gate = _mapping(dlo_result["promotion_gate"], label=f"{dlo} gate")
        lines.append(
            "| {dlo} | {physical:.4f} | {specific:.4f} | {transfer:.4f} | "
            "{improvement:.2f}% | {wins}/14 | {seeds}/3 | {supported} |".format(
                dlo=dlo,
                physical=1000.0
                * float(cast(Any, methods["matching_object_physical"])),
                specific=1000.0
                * float(cast(Any, methods["matching_object_fitted_residual"])),
                transfer=1000.0
                * float(cast(Any, methods["dlo3_equal_seed_no_refit_residual"])),
                improvement=100.0
                * float(cast(Any, primary["relative_improvement"])),
                wins=primary["wins"],
                seeds=gate["improving_seed_models"],
                supported="yes" if gate["supported"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Equal-DLO aggregate",
            "",
            (
                "- Relative improvement over matching physical models: "
                "**{:.2f}%**.".format(
                    100.0 * float(cast(Any, equal_dlo["relative_improvement"]))
                )
            ),
            (
                "- Trajectory wins/ties/losses: "
                f"**{equal_dlo['wins']}/{equal_dlo['ties']}/{equal_dlo['losses']}**."
            ),
            (
                "- Stratified trajectory-bootstrap 95% interval for transfer minus "
                f"physical: **[{1000.0 * interval[0]:.4f}, "
                f"{1000.0 * interval[1]:.4f}] mm**."
            ),
            "",
            "## DLO3 feature-support shift",
            "",
            "| DLO | Seed | |z| > 3 | |z| > 5 | |z| > 10 | max |z| |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dlo in DLOS:
        dlo_support = _mapping(support[dlo], label=f"{dlo} feature support")
        for seed in SEEDS:
            record = _mapping(
                dlo_support[str(seed)],
                label=f"{dlo} seed {seed} support",
            )
            lines.append(
                (
                    "| {dlo} | {seed} | {gt3:.4f} | {gt5:.4f} | "
                    "{gt10:.4f} | {maximum:.3f} |"
                ).format(
                    dlo=dlo,
                    seed=seed,
                    gt3=float(cast(Any, record["fraction_absolute_z_gt_3"])),
                    gt5=float(cast(Any, record["fraction_absolute_z_gt_5"])),
                    gt10=float(cast(Any, record["fraction_absolute_z_gt_10"])),
                    maximum=float(cast(Any, record["maximum_absolute_z"])),
                )
            )
    lines.extend(
        [
            "",
            (
                "The feature-support diagnostic is descriptive and cannot alter "
                "the registered promotion gate."
            ),
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_cross_object_transfer_protocol(protocol_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    parent_contract = _mapping(protocol["parent_dlo45"], label="parent DLO45")
    parent_root = args.dlo45_root.resolve(strict=True)
    if parent_root != Path(str(parent_contract["run_root"])).resolve():
        raise ValueError("DLO4/DLO5 parent run root differs from protocol")
    parent_protocol_path = _verified_file(
        parent_contract["protocol"],
        label="parent DLO4/DLO5 protocol",
    )
    parent_protocol_sha256 = sha256_file(parent_protocol_path)
    score_result_path = (parent_root / "score" / "result.json").resolve(strict=True)
    joint_seal_path = (
        parent_root / "joint" / "joint_prediction_seal.json"
    ).resolve(strict=True)
    prediction_seal_paths = {
        "DLO4": (
            parent_root / "dlo4-target" / "prediction_seal.json"
        ).resolve(strict=True),
        "DLO5": (
            parent_root / "dlo5-target" / "prediction_seal.json"
        ).resolve(strict=True),
    }
    raw_models = cast(
        Sequence[Mapping[str, object]],
        protocol["dlo3_local_residual_models"],
    )
    model_paths = {
        int(cast(Any, identity["seed"])): _verified_file(
            identity,
            label=f"DLO3 seed {identity['seed']} residual model",
        )
        for identity in raw_models
    }
    parent_identities = {
        "score_result": _identity(score_result_path),
        "joint_prediction_seal": _identity(joint_seal_path),
        "prediction_seals": {
            dlo: _identity(path) for dlo, path in prediction_seal_paths.items()
        },
    }
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo3-to-dlo45-no-refit-preflight-v1",
        "mode": args.mode,
        "protocol": _identity(protocol_path),
        "implementation_sha": os.environ.get("IMPLEMENTATION_SHA"),
        "parent_run_root": str(parent_root),
        "parent_dlo45_protocol": _identity(parent_protocol_path),
        "parent_artifacts": parent_identities,
        "dlo3_local_residual_models": {
            str(seed): _identity(path) for seed, path in model_paths.items()
        },
        "parent_target_scores_semantically_read": False,
        "target_prediction_arrays_loaded": False,
        "target_trajectory_payload_deserialized": False,
        "dlo3_official_evaluation_read": False,
        "target_retry_authorized": False,
        "paper_claim_authorized": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    evaluation = _mapping(protocol["evaluation"], label="evaluation")
    gate = _mapping(protocol["promotion_gate"], label="promotion gate")
    method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-to-dlo45-no-refit-method-seal-v1",
        "protocol": _identity(protocol_path),
        "implementation_sha": os.environ.get("IMPLEMENTATION_SHA"),
        "parent_run_root": str(parent_root),
        "parent_dlo45_protocol": _identity(parent_protocol_path),
        "parent_artifacts": parent_identities,
        "dlo3_local_residual_models": {
            str(seed): _identity(path) for seed, path in model_paths.items()
        },
        "primary_arm": evaluation["primary_arm"],
        "seed_aggregation": evaluation["seed_aggregation"],
        "shrinkage": evaluation["shrinkage"],
        "promotion_gate": dict(gate),
        "registration_classification": (
            "outcome-blind-pre-score-secondary-diagnostic"
        ),
        "parent_target_scores_semantically_read": False,
        "target_prediction_arrays_loaded": False,
        "target_trajectory_payload_deserialized": False,
        "target_dependent_selection": False,
        "target_retry_authorized": False,
        "paper_claim_authorized": False,
    }
    method_seal_path = output_root / "method_seal.json"
    _write_json(method_seal_path, method_seal)

    parent_result = _validate_parent_result(
        score_result_path,
        protocol_sha256=parent_protocol_sha256,
        joint_seal_path=joint_seal_path,
    )
    joint_seal = _validate_joint_seal(
        joint_seal_path,
        protocol_sha256=parent_protocol_sha256,
    )
    joint_prediction_seals = _mapping(
        joint_seal["prediction_seals"],
        label="joint prediction seals",
    )
    prediction_seals = {}
    for dlo in DLOS:
        expected_identity = _mapping(
            joint_prediction_seals[dlo],
            label=f"joint {dlo} prediction seal",
        )
        if (
            Path(str(expected_identity["path"])).resolve() != prediction_seal_paths[dlo]
            or expected_identity.get("sha256")
            != sha256_file(prediction_seal_paths[dlo])
        ):
            raise ValueError(f"joint {dlo} prediction-seal identity differs")
        prediction_seals[dlo] = _validate_prediction_seal(
            prediction_seal_paths[dlo],
            dlo=dlo,
            protocol_sha256=parent_protocol_sha256,
        )

    data = _mapping(protocol["data"], label="data")
    frame_count = int(cast(Any, data["frame_count"]))
    node_count = int(cast(Any, data["node_count"]))
    names_by_dlo: dict[str, list[str]] = {}
    truth_by_dlo: dict[str, np.ndarray] = {}
    physical_by_dlo: dict[str, np.ndarray] = {}
    object_specific_by_dlo: dict[str, np.ndarray] = {}
    initial_by_dlo: dict[str, np.ndarray] = {}
    action_by_dlo: dict[str, np.ndarray] = {}
    for dlo in DLOS:
        seal = prediction_seals[dlo]
        manifest_path = _verified_file(
            seal["eval_manifest"],
            label=f"{dlo} eval manifest",
        )
        prediction_path = _verified_file(
            seal["predictions"],
            label=f"{dlo} predictions",
        )
        names, trajectories = _load_eval_panel(
            manifest_path,
            dlo=dlo,
            frame_count=frame_count,
            node_count=node_count,
        )
        physical, object_specific = _load_prediction_archive(
            prediction_path,
            expected_names=names,
        )
        truth = trajectories[:, 2:]
        if physical.shape != truth.shape:
            raise ValueError(f"{dlo} target predictions and truth differ")
        initial, action = deform_causal_inputs(trajectories)
        names_by_dlo[dlo] = names
        truth_by_dlo[dlo] = truth
        physical_by_dlo[dlo] = physical
        object_specific_by_dlo[dlo] = object_specific
        initial_by_dlo[dlo] = initial
        action_by_dlo[dlo] = action

    models = {seed: _load_model(path) for seed, path in model_paths.items()}
    shrinkage = float(cast(Any, evaluation["shrinkage"]))
    transferred_by_dlo = {
        dlo: {
            seed: predict_deform_local_residual(
                models[seed],
                initial_by_dlo[dlo],
                action_by_dlo[dlo],
                physical_by_dlo[dlo],
                shrinkage=shrinkage,
            )["predictions"]
            for seed in SEEDS
        }
        for dlo in DLOS
    }
    support = {
        dlo: {
            str(seed): feature_support_summary(
                models[seed],
                initial_by_dlo[dlo],
                action_by_dlo[dlo],
                physical_by_dlo[dlo],
            )
            for seed in SEEDS
        }
        for dlo in DLOS
    }
    result = evaluate_cross_object_transfer(
        names_by_dlo=names_by_dlo,
        truth_by_dlo=truth_by_dlo,
        physical_by_dlo=physical_by_dlo,
        object_specific_by_dlo=object_specific_by_dlo,
        transferred_by_dlo=transferred_by_dlo,
        protocol=protocol,
    )

    parent_results = _mapping(parent_result["results"], label="parent DLO results")
    result_dlos = _mapping(result["results"], label="cross-object DLO results")
    parent_primary_summary = {}
    for dlo in DLOS:
        parent_dlo = _mapping(parent_results[dlo], label=f"parent {dlo} result")
        parent_primary = _mapping(
            parent_dlo["primary_vs_matching_physical"],
            label=f"parent {dlo} primary",
        )
        observed_primary = _mapping(
            _mapping(result_dlos[dlo], label=f"{dlo} result")[
                "matching_object_residual_vs_physical"
            ],
            label=f"observed {dlo} object-specific summary",
        )
        _verify_parent_point_summary(observed_primary, parent_primary, dlo=dlo)
        parent_primary_summary[dlo] = dict(parent_primary)

    result.update(
        {
            "protocol": _identity(protocol_path),
            "method_seal": _identity(method_seal_path),
            "implementation_sha": os.environ.get("IMPLEMENTATION_SHA"),
            "parent_dlo45_result": {
                "identity": _identity(score_result_path),
                "decision": parent_result["decision"],
                "both_primary_gates_passed": parent_result[
                    "both_primary_gates_passed"
                ],
                "equal_dlo_relative_improvement": parent_result[
                    "equal_dlo_relative_improvement"
                ],
                "total_candidate_wins": parent_result["total_candidate_wins"],
                "primary_summaries": parent_primary_summary,
            },
            "parent_joint_prediction_seal": _identity(joint_seal_path),
            "feature_support": support,
            "parent_target_scores_read_after_method_seal": True,
            "target_prediction_arrays_loaded_after_method_seal": True,
            "target_trajectory_payload_deserialized_after_method_seal": True,
        }
    )
    _write_json(output_root / "result.json", result)
    _write_csv(output_root / "trajectory-results.csv", result)
    _write_report(output_root / "report.md", result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "parent_decision": _mapping(
                    result["parent_dlo45_result"],
                    label="parent result",
                )["decision"],
                "equal_dlo_summary": result["equal_dlo_summary"],
                "per_dlo": {
                    dlo: _mapping(result["results"], label="DLO results")[dlo][
                        "promotion_gate"
                    ]
                    for dlo in DLOS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
