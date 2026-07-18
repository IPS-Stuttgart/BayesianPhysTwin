"""Future-blind family gate for teacher-centered part-pair spring refits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .matphys_part_family_gate import choose_part_family


PART_PAIR_SOURCE_GATE_CONTRACT = "phystwin-part-pair-source-gate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _metric_pair(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a metric object")
    metrics = {
        name: float(value[name])
        for name in ("chamfer_distance_m", "track_error_m")
    }
    if not all(np.isfinite(metric) and metric > 0.0 for metric in metrics.values()):
        raise ValueError(f"{label} metrics must be finite and positive")
    return metrics


def _identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": _sha256(path)}


def _validate_run_config(
    config: Mapping[str, object],
    method: Mapping[str, object],
    *,
    train_end: int,
    fit_end: int,
) -> None:
    expected = {
        "variant": method["observation_variant"],
        "train_end_frame": train_end,
        "fit_end_frame": fit_end,
        "epochs": int(method["epochs"]),
        "learning_rate": float(method["learning_rate"]),
        "spring_scale_weight_decay": float(method["spring_scale_weight_decay"]),
        "early_stopping_patience": int(method["early_stopping_patience"]),
        "selection_metric": method["selection_metric"],
        "spring_parameterization": "part_pair",
        "deterministic_spring_forces": bool(method["deterministic_spring_forces"]),
        "optimize_collision": False,
        "dashpot_log_scale": 0.0,
        "drag_log_scale": 0.0,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"run config {key} changed: {config.get(key)!r} != {expected_value!r}"
            )


def _summarize_scales(parameters: object) -> dict[str, object]:
    if not isinstance(parameters, Mapping):
        raise ValueError("run summary omits parameter diagnostics")
    groups = parameters.get("group_log_scales")
    if not isinstance(groups, Mapping):
        raise ValueError("run summary omits part-pair group scales")
    raw_pairs = groups.get("part_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("part-pair run contains no object groups")
    within: list[float] = []
    cross: list[float] = []
    records: list[dict[str, object]] = []
    for raw in raw_pairs:
        if not isinstance(raw, Mapping):
            raise ValueError("part-pair scale record must be an object")
        parts = tuple(int(value) for value in raw["parts"])
        if len(parts) != 2:
            raise ValueError("part-pair scale must identify two endpoint parts")
        scale = float(raw["log_scale"])
        if not np.isfinite(scale):
            raise ValueError("part-pair scale must be finite")
        (within if parts[0] == parts[1] else cross).append(scale)
        records.append(
            {
                "parts": list(parts),
                "log_scale": scale,
                "spring_count": int(raw["spring_count"]),
            }
        )
    all_scales = np.asarray(within + cross, dtype=float)
    return {
        "controller_log_scale": float(groups["controller"]),
        "part_pairs": records,
        "within_part_mean_log_scale": float(np.mean(within)) if within else None,
        "cross_part_mean_log_scale": float(np.mean(cross)) if cross else None,
        "object_log_scale_std": float(np.std(all_scales)),
        "object_log_scale_max_abs": float(np.max(np.abs(all_scales))),
    }


def run_part_pair_source_gate(
    source_root: str | Path,
    output_path: str | Path,
    source_protocol: str | Path,
) -> dict[str, object]:
    """Apply the locked prefix-only gate to completed headless refit summaries."""

    root = Path(source_root).resolve()
    output = Path(output_path).resolve()
    protocol_path = Path(source_protocol).resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported source protocol schema")
    if protocol.get("protocol_name") != "phystwin-part-pair-source-v1":
        raise ValueError("source protocol is not the locked part-pair protocol")
    locked_protocol = root / "locked_protocol.json"
    if _sha256(locked_protocol) != _sha256(protocol_path):
        raise ValueError("source-root protocol differs from the requested lock")

    cases = protocol.get("source_cases")
    method = protocol.get("method")
    evidence = protocol.get("evidence_boundary")
    family_gate = protocol.get("family_gate")
    acceptance = protocol.get("source_acceptance")
    if not isinstance(cases, list) or not cases:
        raise ValueError("source protocol contains no cases")
    if not all(
        isinstance(value, Mapping)
        for value in (method, evidence, family_gate, acceptance)
    ):
        raise ValueError("source protocol omits gate configuration")

    minimum_improvement = float(family_gate["minimum_relative_score_improvement"])
    maximum_regression = float(family_gate["maximum_per_metric_regression"])
    fit_fraction = float(evidence["fit_fraction_of_released_prefix"])
    case_results: dict[str, object] = {}
    code_commits: set[str] = set()
    official_commits: set[str] = set()
    for case in cases:
        case_root = root / str(case)
        prefix_manifest_path = case_root / "prefix" / "manifest.json"
        summary_path = case_root / "learned" / "summary.json"
        manifest = _load_json(prefix_manifest_path)
        summary = _load_json(summary_path)
        if manifest.get("contract") != "phystwin-observation-prefix-plus-hold-v1":
            raise ValueError(f"{case}: unsupported prefix contract")
        train_end = int(manifest["prefix_end_frame"])
        if int(manifest["hold_frame_index"]) != train_end:
            raise ValueError(f"{case}: hold sentinel moved")
        if int(manifest["output_frame_count"]) != train_end + 1:
            raise ValueError(f"{case}: prefix payload frame count changed")
        fit_end = int(np.floor(fit_fraction * train_end))
        config = summary.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"{case}: run summary omits config")
        _validate_run_config(config, method, train_end=train_end, fit_end=fit_end)

        inputs = summary.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError(f"{case}: run summary omits input identities")
        for summary_key, manifest_key in (
            ("final_data", "final_data"),
            ("gt_track_3d", "gt_track_3d"),
        ):
            summary_identity = inputs.get(summary_key)
            expected_identity = manifest["outputs"][manifest_key]
            if not isinstance(summary_identity, Mapping):
                raise ValueError(f"{case}: missing {summary_key} identity")
            if summary_identity.get("sha256") != expected_identity["sha256"]:
                raise ValueError(f"{case}: {summary_key} does not match prefix artifact")

        baseline = summary.get("baseline_official_evaluation")
        candidate = summary.get("official_evaluation")
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError(f"{case}: run summary omits official validation")
        teacher_metrics = _metric_pair(
            baseline.get("validation"), label=f"{case}.teacher"
        )
        candidate_metrics = _metric_pair(
            candidate.get("validation"), label=f"{case}.candidate"
        )
        decision = choose_part_family(
            teacher_metrics,
            candidate_metrics,
            minimum_relative_score_improvement=minimum_improvement,
            maximum_metric_regression=maximum_regression,
        )
        selected_metrics = (
            candidate_metrics if decision["learned_accepted"] else teacher_metrics
        )
        selected_name = (
            "trajectory.pkl"
            if decision["learned_accepted"]
            else "baseline_trajectory.pkl"
        )
        identity_parity = summary.get("selected_baseline_trajectory_parity")
        if not isinstance(identity_parity, Mapping):
            raise ValueError(f"{case}: missing internal identity parity")
        code_commits.add(str(summary["code_commit"]))
        official_commits.add(str(summary["official_commit"]))
        case_results[str(case)] = {
            "fit_interval": [0, fit_end],
            "validation_interval": [fit_end, train_end],
            "teacher_validation_metrics": teacher_metrics,
            "candidate_validation_metrics": candidate_metrics,
            "selected_validation_metrics": selected_metrics,
            "decision": decision,
            "runner_selected_epoch": int(summary["selection"]["selected_epoch"]),
            "internal_identity_vector_rmse_m": float(identity_parity["vector_rmse_m"]),
            "released_replay_vector_rmse_m": float(
                summary["released_baseline_trajectory_parity"]["vector_rmse_m"]
            ),
            "spring_scale_diagnostic": _summarize_scales(summary.get("parameters")),
            "artifacts": {
                "prefix_manifest": _identity(prefix_manifest_path),
                "run_summary": _identity(summary_path),
                "selected_validation_trajectory": _identity(
                    case_root / "learned" / selected_name
                ),
            },
        }

    learned_count = sum(
        int(result["decision"]["learned_accepted"])
        for result in case_results.values()
    )
    aggregate: dict[str, dict[str, float]] = {}
    for family, key in (
        ("teacher", "teacher_validation_metrics"),
        ("candidate", "candidate_validation_metrics"),
        ("selected", "selected_validation_metrics"),
    ):
        aggregate[family] = {
            metric: float(
                np.mean([result[key][metric] for result in case_results.values()])
            )
            for metric in ("chamfer_distance_m", "track_error_m")
        }
    aggregate_both_improved = all(
        aggregate["selected"][metric] < aggregate["teacher"][metric]
        for metric in ("chamfer_distance_m", "track_error_m")
    )
    required_count = int(acceptance["minimum_learned_case_count"])
    source_gate_passed = bool(
        learned_count >= required_count and aggregate_both_improved
    )
    result = {
        "schema_version": 1,
        "contract": PART_PAIR_SOURCE_GATE_CONTRACT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_protocol": _identity(protocol_path),
        "code_commits": sorted(code_commits),
        "official_commits": sorted(official_commits),
        "learned_acceptance_count": learned_count,
        "teacher_fallback_count": len(cases) - learned_count,
        "required_learned_case_count": required_count,
        "aggregate_validation_metrics": aggregate,
        "aggregate_both_metrics_improved": aggregate_both_improved,
        "source_gate_passed": source_gate_passed,
        "future_metrics_opened": False,
        "case_results": case_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(output)
    return result
