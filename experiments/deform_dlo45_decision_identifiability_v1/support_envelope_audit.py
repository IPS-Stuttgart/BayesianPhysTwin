"""Trajectory-conformal support enlargement for the DEFORM decision certificate.

The parent finite-action result remains unchanged. This retrospective audit
uses only the parent's source-test trajectories to calibrate, separately for
DLO4 and DLO5, a complete-trajectory regret-excess radius. The target stage
then evaluates fixed operational tolerances on the official evaluation
trajectories without target tuning or retries.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.support_robust_decision_certificate_v1 import (
    SUPPORT_ROBUST_DECISION_CERTIFICATE_CLAIM_BOUNDARY,
    split_conformal_trajectory_envelope,
    support_robust_action_decision,
    trajectory_policy_regret_excess,
)

from ._common import (
    ATOL,
    DLOS,
    canonical_sha256,
    load_protocol,
    partition_names,
    read_json,
    sha256_file,
    trajectory_paths,
    write_json,
)
from ._model import build_pool, fit_model
from .gate_audit import WindowRecord, _window_records

CONTRACT = "deform-dlo45-support-robust-decision-envelope-v1"
PARENT_CONTRACT = "deform-dlo45-decision-identifiability-v1"


def load_envelope_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    parent_source = value.get("parent_source_artifact")
    parent_target = value.get("parent_target_artifact")
    dataset = value.get("dataset")
    conformal = value.get("conformal")
    operational = value.get("operational")
    bootstrap = value.get("bootstrap")
    if (
        value.get("contract") != CONTRACT
        or value.get("schema_version") != 1
        or value.get("parent_contract") != PARENT_CONTRACT
        or value.get("parent_workflow_run_id") != 33473378340
        or not all(
            isinstance(item, dict)
            for item in (
                parent_source,
                parent_target,
                dataset,
                conformal,
                operational,
                bootstrap,
            )
        )
    ):
        raise ValueError("invalid support-envelope protocol header")
    assert isinstance(parent_source, dict)
    assert isinstance(parent_target, dict)
    assert isinstance(dataset, dict)
    assert isinstance(conformal, dict)
    assert isinstance(operational, dict)
    assert isinstance(bootstrap, dict)
    if (
        parent_source.get("id") != 9787311310
        or parent_source.get("source_model_sha256")
        != "a43aed43cd563ee47358e48cab84829dc7eebc77d97725721a11b228f3b6b7f0"
        or parent_target.get("id") != 9787322207
        or parent_target.get("eval_manifest_sha256")
        != "93f7799700f56a2a3da08bc647114dc5aa258f866af21e9eb3ad196816952ffb"
        or tuple(dataset.get("dlos", ())) != DLOS
        or dataset.get("calibration_partition") != "source_test"
        or dataset.get("target_partition") != "eval"
        or conformal.get("calibration_unit") != "complete_trajectory"
        or conformal.get("base_policy") != "exact-finite-support-certificate-v1"
        or conformal.get("pool_across_dlos") is not False
        or conformal.get("insufficient_rank_behavior")
        != "positive-infinity-and-exact-fallback"
        or operational.get("fallback_action_index") != 0
        or operational.get("target_tuning") is not False
        or operational.get("target_retries") is not False
        or bootstrap.get("unit") != "complete_trajectory"
    ):
        raise ValueError("invalid frozen support-envelope protocol")
    alpha = float(conformal["miscoverage"])
    fixed_epsilon = float(operational["fixed_regret_tolerance"])
    retention = float(operational["source_retention_fraction"])
    grid = tuple(float(item) for item in operational["report_regret_tolerance_grid"])
    if (
        not 0.0 < alpha < 1.0
        or fixed_epsilon < 0.0
        or not 0.0 < retention <= 1.0
        or not grid
        or any(item < 0.0 for item in grid)
        or tuple(sorted(set(grid))) != grid
        or int(bootstrap["replicates"]) < 1000
    ):
        raise ValueError("invalid support-envelope numeric protocol")
    return value


def _parent_files(parent_source_dir: Path) -> tuple[Path, Path, Path]:
    candidates = [parent_source_dir, parent_source_dir / "dlo45-decision-source"]
    for root in candidates:
        model = root / "source_model.npz"
        result = root / "source_result.json"
        seal = root / "source_seal.json"
        if model.is_file() and result.is_file() and seal.is_file():
            return model, result, seal
    raise FileNotFoundError("parent source artifact is missing model/result/seal")


def validate_parent_source(
    parent_source_dir: Path,
    parent_protocol_path: Path,
    envelope_protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    model_path, result_path, seal_path = _parent_files(parent_source_dir)
    source_result = read_json(result_path)
    source_seal = read_json(seal_path)
    load_protocol(parent_protocol_path)
    expected_model_sha = str(
        envelope_protocol["parent_source_artifact"]["source_model_sha256"]
    )
    if (
        source_result.get("contract") != PARENT_CONTRACT
        or source_result.get("stage") != "source-result"
        or source_result.get("all_source_gates_passed") is not True
        or source_result.get("target_data_read") is not False
        or source_seal.get("contract") != PARENT_CONTRACT
        or source_seal.get("stage") != "source-seal"
        or source_seal.get("source_model_sha256") != expected_model_sha
        or sha256_file(model_path) != expected_model_sha
        or sha256_file(result_path) != source_seal.get("source_result_sha256")
        or source_result.get("protocol_sha256") != source_seal.get("protocol_sha256")
        or canonical_sha256(json.loads(parent_protocol_path.read_text(encoding="utf-8")))
        != source_seal.get("protocol_sha256")
    ):
        raise ValueError("parent source evidence validation failed")
    return model_path, source_result


def _source_test_model_and_paths(
    dataset_root: Path,
    dlo: str,
    parent_protocol_path: Path,
    source_result: Mapping[str, Any],
):
    protocol = load_protocol(parent_protocol_path)
    train = trajectory_paths(dataset_root, dlo, "train")
    names = tuple(path.name for path in train)
    split = partition_names(names, dlo, protocol)
    dlo_result = source_result["dlos"][dlo]
    if not isinstance(dlo_result, dict):
        raise ValueError(f"{dlo}: malformed parent source result")
    recorded_partition = {
        name: tuple(values) for name, values in dlo_result["partition"].items()
    }
    if recorded_partition != split:
        raise ValueError(f"{dlo}: source partition differs from the parent result")
    settings = dlo_result["selected_settings"]
    if not isinstance(settings, dict):
        raise ValueError(f"{dlo}: malformed selected settings")
    refit_names = split["fit"] + split["calibration"]
    features, residuals, _ = build_pool(train, refit_names, protocol)
    model = fit_model(
        features,
        residuals,
        cluster_count=int(settings["cluster_count"]),
        neighbors=int(settings["neighbors"]),
        temperature_scale=float(settings["temperature_scale"]),
        regret_tolerance=float(settings["regret_tolerance"]),
        protocol=protocol,
    )
    wanted = set(split["source_test"])
    paths = tuple(path for path in train if path.name in wanted)
    if len(paths) != protocol.source_test_count:
        raise ValueError(f"{dlo}: incomplete source-test partition")
    return protocol, model, paths


def group_records(records: Sequence[WindowRecord]) -> dict[str, list[WindowRecord]]:
    grouped: dict[str, list[WindowRecord]] = defaultdict(list)
    for record in records:
        grouped[record.trajectory].append(record)
    return dict(sorted(grouped.items()))


def trajectory_score(records: Sequence[WindowRecord]) -> dict[str, Any]:
    losses = np.stack([record.normalized_regret for record in records])
    bounds = np.stack(
        [record.decision.decision.worst_case_regret for record in records]
    )
    actions = np.asarray(
        [record.decision.decision.certificate_action for record in records],
        dtype=np.int64,
    )
    score = trajectory_policy_regret_excess(
        losses,
        bounds,
        actions,
        fallback_action_index=0,
    )
    return {
        "trajectory": records[0].trajectory,
        "decision_count": score.decision_count,
        "base_nonfallback_count": score.nonfallback_count,
        "score": score.score,
        "maximum_realized_regret": float(np.max(score.realized_regret)),
        "maximum_selected_support_bound": float(
            np.max(score.finite_support_regret_bound[score.nonfallback_mask], initial=0.0)
        ),
    }


def retention_epsilon(
    records: Sequence[WindowRecord],
    *,
    radius: float,
    retention_fraction: float,
) -> float:
    if not np.isfinite(radius):
        return float("inf")
    bounds = sorted(
        float(record.certificate_source_regret_bound)
        for record in records
        if record.decision.decision.certificate_action != 0
    )
    if not bounds:
        return float("inf")
    needed = max(1, int(math.ceil(retention_fraction * len(bounds))))
    return float(radius + bounds[needed - 1])


def policy_actions(
    records: Sequence[WindowRecord],
    *,
    radius: float,
    epsilon: float,
) -> list[int]:
    result: list[int] = []
    for record in records:
        base = record.decision.decision.certificate_action
        bound = float(record.decision.decision.worst_case_regret[base])
        decision = support_robust_action_decision(
            base_selected_action_index=base,
            fallback_action_index=0,
            action_count=len(record.physical_mse),
            finite_support_regret_bound=bound,
            conformal_radius=radius,
            operational_regret_tolerance=epsilon,
        )
        result.append(decision.returned_action_index)
    return result


def summarize_actions(
    records: Sequence[WindowRecord],
    actions: Sequence[int],
    *,
    epsilon: float | None,
) -> dict[str, Any]:
    if len(records) != len(actions) or not records:
        raise ValueError("records/actions mismatch")
    method_mse = np.asarray(
        [
            record.physical_mse[action]
            for record, action in zip(records, actions, strict=True)
        ],
        dtype=np.float64,
    )
    fallback_mse = np.asarray(
        [record.fallback_mse for record in records], dtype=np.float64
    )
    regrets = np.asarray(
        [
            record.normalized_regret[action]
            for record, action in zip(records, actions, strict=True)
        ],
        dtype=np.float64,
    )
    nonfallback = np.asarray(actions, dtype=np.int64) != 0
    harmful = method_mse > fallback_mse + ATOL
    method_rmse = math.sqrt(float(np.mean(method_mse)))
    fallback_rmse = math.sqrt(float(np.mean(fallback_mse)))
    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_trajectory[record.trajectory].append(index)
    trajectory_violation = []
    for indices in by_trajectory.values():
        selected = [index for index in indices if nonfallback[index]]
        trajectory_violation.append(
            bool(
                epsilon is not None
                and selected
                and np.max(regrets[selected]) > epsilon + ATOL
            )
        )
    return {
        "decision_count": len(records),
        "nonfallback_count": int(np.count_nonzero(nonfallback)),
        "nonfallback_fraction": float(np.mean(nonfallback)),
        "action_counts": np.bincount(
            np.asarray(actions, dtype=np.int64),
            minlength=len(records[0].physical_mse),
        ).tolist(),
        "rmse_mm": 1000.0 * method_rmse,
        "fallback_rmse_mm": 1000.0 * fallback_rmse,
        "rmse_reduction_fraction": 1.0 - method_rmse / max(fallback_rmse, ATOL),
        "harmful_nonfallback_count": int(np.count_nonzero(harmful & nonfallback)),
        "maximum_selected_realized_regret": float(
            np.max(regrets[nonfallback], initial=0.0)
        ),
        "p95_selected_realized_regret": float(
            np.quantile(regrets[nonfallback], 0.95) if np.any(nonfallback) else 0.0
        ),
        "trajectory_any_regret_violation_count": int(
            np.count_nonzero(trajectory_violation)
        ),
        "trajectory_count": len(by_trajectory),
    }


def base_certificate_actions(records: Sequence[WindowRecord]) -> list[int]:
    return [record.decision.decision.certificate_action for record in records]


def run_source(
    *,
    dataset_root: Path,
    parent_source_dir: Path,
    parent_protocol_path: Path,
    envelope_protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    envelope_protocol = load_envelope_protocol(envelope_protocol_path)
    _, source_result = validate_parent_source(
        parent_source_dir,
        parent_protocol_path,
        envelope_protocol,
    )
    alpha = float(envelope_protocol["conformal"]["miscoverage"])
    fixed_epsilon = float(
        envelope_protocol["operational"]["fixed_regret_tolerance"]
    )
    retention = float(
        envelope_protocol["operational"]["source_retention_fraction"]
    )
    per_trajectory: list[dict[str, Any]] = []
    dlos: dict[str, Any] = {}
    for dlo in DLOS:
        protocol, model, paths = _source_test_model_and_paths(
            dataset_root,
            dlo,
            parent_protocol_path,
            source_result,
        )
        records = _window_records(paths, model, protocol, dlo)
        grouped = group_records(records)
        scores = [trajectory_score(items) for items in grouped.values()]
        per_trajectory.extend({"dlo": dlo, **item} for item in scores)
        envelope = split_conformal_trajectory_envelope(
            [item["score"] for item in scores],
            miscoverage=alpha,
        )
        retained_epsilon = retention_epsilon(
            records,
            radius=envelope.radius,
            retention_fraction=retention,
        )
        dlos[dlo] = {
            "calibration_trajectory_count": len(scores),
            "calibration_decision_count": len(records),
            "base_certificate": summarize_actions(
                records,
                base_certificate_actions(records),
                epsilon=None,
            ),
            "envelope": envelope.summary(),
            "fixed_epsilon": fixed_epsilon,
            "fixed_epsilon_policy": summarize_actions(
                records,
                policy_actions(records, radius=envelope.radius, epsilon=fixed_epsilon),
                epsilon=fixed_epsilon,
            ),
            "source_retention_epsilon": retained_epsilon,
            "source_retention_policy": summarize_actions(
                records,
                policy_actions(
                    records,
                    radius=envelope.radius,
                    epsilon=retained_epsilon,
                ),
                epsilon=retained_epsilon,
            ),
            "trajectory_manifest": {
                path.name: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in paths
            },
        }

    result = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "source-envelope",
        "parent_contract": PARENT_CONTRACT,
        "protocol_sha256": canonical_sha256(envelope_protocol),
        "parent_protocol_sha256": source_result["protocol_sha256"],
        "target_data_read": False,
        "dlos": dlos,
        "claim_boundary": envelope_protocol["claim_boundary"],
        "generic_claim_boundary": SUPPORT_ROBUST_DECISION_CERTIFICATE_CLAIM_BOUNDARY,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "source_envelope.json"
    write_json(result_path, result)
    with (output_dir / "source_per_trajectory.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        for item in per_trajectory:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    seal = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "source-envelope-seal",
        "source_envelope_sha256": sha256_file(result_path),
        "source_per_trajectory_sha256": sha256_file(
            output_dir / "source_per_trajectory.jsonl"
        ),
        "protocol_sha256": result["protocol_sha256"],
    }
    write_json(output_dir / "source_envelope_seal.json", seal)
    return result


def _find_parent_target_result(parent_target_dir: Path) -> Path:
    candidates = (
        parent_target_dir / "target_result.json",
        parent_target_dir / "dlo45-decision-target" / "target_result.json",
        parent_target_dir
        / "dlo45-decision-result"
        / "dlo45-decision-target"
        / "target_result.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    recursive = tuple(parent_target_dir.rglob("target_result.json"))
    if len(recursive) == 1:
        return recursive[0]
    raise FileNotFoundError("parent target artifact is missing target_result.json")


def validate_source_envelope(source_dir: Path) -> dict[str, Any]:
    result_path = source_dir / "source_envelope.json"
    score_path = source_dir / "source_per_trajectory.jsonl"
    seal = read_json(source_dir / "source_envelope_seal.json")
    result = read_json(result_path)
    if (
        result.get("contract") != CONTRACT
        or result.get("stage") != "source-envelope"
        or result.get("target_data_read") is not False
        or seal.get("contract") != CONTRACT
        or seal.get("stage") != "source-envelope-seal"
        or seal.get("source_envelope_sha256") != sha256_file(result_path)
        or seal.get("source_per_trajectory_sha256") != sha256_file(score_path)
        or seal.get("protocol_sha256") != result.get("protocol_sha256")
    ):
        raise ValueError("source envelope seal validation failed")
    return result


def _target_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    return {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in paths
    }


def _validate_target_manifest(
    local: Mapping[str, Mapping[str, Any]],
    parent_target_result: Mapping[str, Any],
    dlo: str,
) -> None:
    parent_manifest = parent_target_result["eval_manifest"][dlo]
    if local != parent_manifest:
        raise ValueError(f"{dlo}: local eval files differ from parent target manifest")


def run_target(
    *,
    dataset_root: Path,
    parent_source_dir: Path,
    parent_target_dir: Path,
    parent_protocol_path: Path,
    envelope_protocol_path: Path,
    source_envelope_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    envelope_protocol = load_envelope_protocol(envelope_protocol_path)
    model_path, source_result = validate_parent_source(
        parent_source_dir,
        parent_protocol_path,
        envelope_protocol,
    )
    source_envelope = validate_source_envelope(source_envelope_dir)
    if source_envelope["protocol_sha256"] != canonical_sha256(envelope_protocol):
        raise ValueError("source envelope does not match the frozen protocol")
    parent_target_result = read_json(_find_parent_target_result(parent_target_dir))
    expected_eval_hash = str(
        envelope_protocol["parent_target_artifact"]["eval_manifest_sha256"]
    )
    if (
        parent_target_result.get("contract") != PARENT_CONTRACT
        or parent_target_result.get("stage") != "target-result"
        or parent_target_result.get("target_tuning") is not False
        or parent_target_result.get("target_retries") is not False
        or parent_target_result.get("source_model_sha256") != sha256_file(model_path)
        or parent_target_result.get("eval_manifest_sha256") != expected_eval_hash
        or canonical_sha256(parent_target_result["eval_manifest"]) != expected_eval_hash
    ):
        raise ValueError("parent target manifest validation failed")

    protocol = load_protocol(parent_protocol_path)
    fixed_epsilon = float(
        envelope_protocol["operational"]["fixed_regret_tolerance"]
    )
    epsilon_grid = tuple(
        float(item)
        for item in envelope_protocol["operational"]["report_regret_tolerance_grid"]
    )
    dlos: dict[str, Any] = {}
    per_decision: list[dict[str, Any]] = []
    aggregate_records: list[WindowRecord] = []
    aggregate_policy_actions: dict[str, list[int]] = defaultdict(list)

    for dlo in DLOS:
        _, target_model, _ = _source_test_model_and_paths(
            dataset_root,
            dlo,
            parent_protocol_path,
            source_result,
        )
        paths = trajectory_paths(dataset_root, dlo, "eval")
        local_manifest = _target_manifest(paths)
        _validate_target_manifest(local_manifest, parent_target_result, dlo)
        records = _window_records(paths, target_model, protocol, dlo)
        aggregate_records.extend(records)
        source_dlo = source_envelope["dlos"][dlo]
        radius = float(source_dlo["envelope"]["radius"])
        retained_epsilon = float(source_dlo["source_retention_epsilon"])
        base_actions = base_certificate_actions(records)
        fixed_actions = policy_actions(records, radius=radius, epsilon=fixed_epsilon)
        retained_actions = policy_actions(
            records,
            radius=radius,
            epsilon=retained_epsilon,
        )
        curve = {
            str(epsilon): summarize_actions(
                records,
                policy_actions(records, radius=radius, epsilon=epsilon),
                epsilon=epsilon,
            )
            for epsilon in epsilon_grid
        }
        target_scores = [
            trajectory_score(items) for items in group_records(records).values()
        ]
        empirical_coverage = float(
            np.mean([item["score"] <= radius + ATOL for item in target_scores])
        )
        dlos[dlo] = {
            "target_trajectory_count": len(paths),
            "target_decision_count": len(records),
            "envelope_radius": radius,
            "nominal_trajectory_coverage": 1.0
            - float(envelope_protocol["conformal"]["miscoverage"]),
            "empirical_base_policy_trajectory_coverage": empirical_coverage,
            "base_certificate": summarize_actions(records, base_actions, epsilon=None),
            "fixed_epsilon": fixed_epsilon,
            "fixed_epsilon_policy": summarize_actions(
                records,
                fixed_actions,
                epsilon=fixed_epsilon,
            ),
            "source_retention_epsilon": retained_epsilon,
            "source_retention_policy": summarize_actions(
                records,
                retained_actions,
                epsilon=retained_epsilon,
            ),
            "regret_tolerance_curve": curve,
            "target_trajectory_scores": target_scores,
            "eval_manifest": local_manifest,
        }
        aggregate_policy_actions["base_certificate"].extend(base_actions)
        aggregate_policy_actions["fixed_epsilon"].extend(fixed_actions)
        aggregate_policy_actions["source_retention"].extend(retained_actions)

        for record, base, fixed, retained in zip(
            records,
            base_actions,
            fixed_actions,
            retained_actions,
            strict=True,
        ):
            per_decision.append(
                {
                    "dlo": dlo,
                    "stable_id": record.stable_id,
                    "trajectory": record.trajectory,
                    "current_frame": record.current_frame,
                    "base_action": base,
                    "fixed_epsilon_action": fixed,
                    "source_retention_action": retained,
                    "source_support_regret_bound": record.certificate_source_regret_bound,
                    "base_realized_regret": record.certificate_realized_regret,
                    "base_regret_excess": record.certificate_regret_excess,
                    "fixed_epsilon": fixed_epsilon,
                    "source_retention_epsilon": retained_epsilon,
                    "conformal_radius": radius,
                }
            )

    aggregate = {
        name: summarize_actions(
            aggregate_records,
            actions,
            epsilon=(fixed_epsilon if name == "fixed_epsilon" else None),
        )
        for name, actions in aggregate_policy_actions.items()
    }
    result = {
        "contract": CONTRACT,
        "schema_version": 1,
        "stage": "target-audit",
        "parent_contract": PARENT_CONTRACT,
        "protocol_sha256": canonical_sha256(envelope_protocol),
        "source_envelope_sha256": sha256_file(
            source_envelope_dir / "source_envelope.json"
        ),
        "target_tuning": False,
        "target_retries": False,
        "dlos": dlos,
        "aggregate": aggregate,
        "claim_boundary": envelope_protocol["claim_boundary"],
        "generic_claim_boundary": SUPPORT_ROBUST_DECISION_CERTIFICATE_CLAIM_BOUNDARY,
        "interpretation": (
            "The conformal envelope calibrates trajectory-level regret excess beyond the "
            "registered finite physical support. The same parent source-test model, fitted "
            "without the calibration trajectories, is used for calibration and target "
            "evaluation. This is retrospective within-DLO evidence."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "target_audit.json", result)
    with (output_dir / "target_per_decision.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        for item in per_decision:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("source", "target"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--dataset-root", type=Path, required=True)
        sub.add_argument("--parent-source-dir", type=Path, required=True)
        sub.add_argument("--parent-protocol", type=Path, required=True)
        sub.add_argument("--envelope-protocol", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        if name == "target":
            sub.add_argument("--parent-target-dir", type=Path, required=True)
            sub.add_argument("--source-envelope-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "source":
        result = run_source(
            dataset_root=args.dataset_root,
            parent_source_dir=args.parent_source_dir,
            parent_protocol_path=args.parent_protocol,
            envelope_protocol_path=args.envelope_protocol,
            output_dir=args.output_dir,
        )
    else:
        result = run_target(
            dataset_root=args.dataset_root,
            parent_source_dir=args.parent_source_dir,
            parent_target_dir=args.parent_target_dir,
            parent_protocol_path=args.parent_protocol,
            envelope_protocol_path=args.envelope_protocol,
            source_envelope_dir=args.source_envelope_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
