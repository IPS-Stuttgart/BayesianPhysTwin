#!/usr/bin/env python3
"""Evaluate a causal tactile guard on already-open Deform360 source panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_tactile_regret_guard import (  # noqa: E402
    TACTILE_REGRET_FEATURE_NAMES,
    TactileRegretGuardModel,
    fit_object_balanced_tactile_regret_guard,
    tactile_benefit_scores,
)

UPDATE_INTERVALS = ((19, 38), (38, 57), (57, 76))
IDENTITY_METRIC = "hidden_identity_rmse_m"
CHAMFER_METRIC = "hidden_symmetric_chamfer_m"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    stripped = dict(payload)
    stripped.pop("artifact_sha256", None)
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _protocol_sha256(payload: dict[str, Any]) -> str:
    stripped = dict(payload)
    stripped.pop("protocol_sha256", None)
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _tactile_by_case(
    *payloads: dict[str, Any],
) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for payload in payloads:
        _require(
            payload.get("artifact_kind") == "Deform360CausalTactileFeatureAudit",
            "unexpected tactile feature artifact",
        )
        _require(
            payload.get("artifact_sha256") == _canonical_sha256(payload),
            "tactile feature artifact checksum changed",
        )
        boundary = payload.get("information_boundary", {})
        _require(
            boundary.get("opened_source_only") is True
            and boundary.get("future_tactile_used_for_update") is False
            and boundary.get("each_update_uses_tactile_at_or_before_update") is True
            and boundary.get("held_v8_read") is False,
            "tactile feature artifact crossed its information boundary",
        )
        for case in payload.get("cases", []):
            case_name = str(case["case"])
            _require(case_name not in output, "tactile source case repeated")
            updates = {
                int(row["update_frame"]): dict(row) for row in case["updates"]
            }
            _require(
                tuple(updates) == tuple(start for start, _ in UPDATE_INTERVALS),
                "tactile update schedule changed",
            )
            output[case_name] = updates
    return output


def _feature_vector(update: dict[str, Any]) -> np.ndarray:
    sensor_ratio = np.asarray(update["sensor_energy_over_initial"], dtype=np.float64)
    active_taxels = np.asarray(update["sensor_active_taxels"], dtype=np.float64)
    expanded = {
        **update,
        "sensor_ratio_min": float(np.min(sensor_ratio)),
        "sensor_ratio_max": float(np.max(sensor_ratio)),
        "sensor_ratio_std": float(np.std(sensor_ratio)),
        "active_taxel_mean": float(np.mean(active_taxels)),
        "active_taxel_std": float(np.std(active_taxels)),
    }
    values = np.asarray(
        [float(expanded[name]) for name in TACTILE_REGRET_FEATURE_NAMES],
        dtype=np.float64,
    )
    _require(np.all(np.isfinite(values)), "stored tactile feature is non-finite")
    return values


def _source_rows(
    source_result: dict[str, Any],
    tactile: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for case in source_result["source_interval_diagnostics"]:
        case_name = str(case["case"])
        for interval in case["intervals"]:
            frame = int(interval["frame"])
            rows.append(
                {
                    "panel": "open27",
                    "object": str(case["object_id"]),
                    "case": case_name,
                    "frame": frame,
                    "feature": _feature_vector(tactile[case_name][frame]),
                    "maximum_regret_m": float(interval["maximum_metric_regret_m"]),
                    "baseline_identity_m": float(
                        interval["baseline_hidden_identity_rmse_m"]
                    ),
                    "candidate_identity_m": float(
                        interval["candidate_hidden_identity_rmse_m"]
                    ),
                    "baseline_chamfer_m": float(
                        interval["baseline_hidden_symmetric_chamfer_m"]
                    ),
                    "candidate_chamfer_m": float(
                        interval["candidate_hidden_symmetric_chamfer_m"]
                    ),
                }
            )
    return rows


def _mean_interval_metric(
    score: dict[str, Any],
    metric: str,
    *,
    update: int,
    stop: int,
) -> float:
    frames = np.asarray(score["scored_frames"], dtype=np.int64)
    values = np.asarray(score["by_frame"][metric], dtype=np.float64)
    _require(frames.shape == values.shape, "stress score frames differ")
    selected = (frames > update) & (frames < stop)
    _require(np.any(selected), "stress interval has no scored frame")
    return float(np.mean(values[selected]))


def _stress_rows(
    source_result: dict[str, Any],
    tactile: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for case in source_result["opened_stress"]["cases"]:
        case_name = str(case["case"])
        baseline = case["scores"]["selected_raw_backbone"]
        candidate = case["scores"]["dual_backbone_pairwise_consensus_rbf"]
        for update, stop in UPDATE_INTERVALS:
            baseline_identity = _mean_interval_metric(
                baseline,
                IDENTITY_METRIC,
                update=update,
                stop=stop,
            )
            candidate_identity = _mean_interval_metric(
                candidate,
                IDENTITY_METRIC,
                update=update,
                stop=stop,
            )
            baseline_chamfer = _mean_interval_metric(
                baseline,
                CHAMFER_METRIC,
                update=update,
                stop=stop,
            )
            candidate_chamfer = _mean_interval_metric(
                candidate,
                CHAMFER_METRIC,
                update=update,
                stop=stop,
            )
            rows.append(
                {
                    "panel": "stress12",
                    "object": str(case["object_id"]),
                    "case": case_name,
                    "frame": update,
                    "feature": _feature_vector(tactile[case_name][update]),
                    "maximum_regret_m": max(
                        candidate_identity - baseline_identity,
                        candidate_chamfer - baseline_chamfer,
                    ),
                    "baseline_identity_m": baseline_identity,
                    "candidate_identity_m": candidate_identity,
                    "baseline_chamfer_m": baseline_chamfer,
                    "candidate_chamfer_m": candidate_chamfer,
                }
            )
    return rows


def _model_dict(model: TactileRegretGuardModel) -> dict[str, Any]:
    return {
        "feature_names": list(TACTILE_REGRET_FEATURE_NAMES),
        "feature_center": list(model.feature_center),
        "feature_scale": list(model.feature_scale),
        "coefficients": list(model.coefficients),
        "ridge_penalty": model.ridge_penalty,
        "admission_threshold": model.admission_threshold,
        "source_object_count": model.source_object_count,
        "source_row_count": model.source_row_count,
        "score_semantics": "linear source-fitted benefit score; not a probability",
    }


def _cross_fitted_decisions(
    rows: list[dict[str, Any]],
    *,
    ridge_penalty: float,
    admission_threshold: float,
) -> list[dict[str, Any]]:
    decisions = []
    objects = sorted({str(row["object"]) for row in rows})
    for held_object in objects:
        train = [row for row in rows if row["object"] != held_object]
        test = [row for row in rows if row["object"] == held_object]
        model = fit_object_balanced_tactile_regret_guard(
            np.asarray([row["feature"] for row in train]),
            np.asarray([row["maximum_regret_m"] for row in train]),
            [str(row["object"]) for row in train],
            ridge_penalty=ridge_penalty,
            admission_threshold=admission_threshold,
        )
        scores = tactile_benefit_scores(
            np.asarray([row["feature"] for row in test]),
            model,
        )
        for row, score in zip(test, scores, strict=True):
            accepted = bool(score >= admission_threshold)
            decisions.append(
                {
                    **{key: value for key, value in row.items() if key != "feature"},
                    "benefit_score": float(score),
                    "admission_threshold": admission_threshold,
                    "candidate_accepted": accepted,
                    "bit_exact_baseline_fallback": not accepted,
                    "held_object_excluded_from_fit": True,
                    "fit_source_object_count": model.source_object_count,
                }
            )
    return sorted(decisions, key=lambda row: (row["case"], row["frame"]))


def _panel_summary(
    decisions: list[dict[str, Any]],
    panel: str | None,
) -> dict[str, Any]:
    selected = (
        decisions if panel is None else [row for row in decisions if row["panel"] == panel]
    )
    _require(selected, "metric panel is empty")
    case_rows = []
    for case_name in sorted({str(row["case"]) for row in selected}):
        intervals = [row for row in selected if row["case"] == case_name]
        _require(len(intervals) == len(UPDATE_INTERVALS), "case interval count changed")
        record: dict[str, Any] = {
            "case": case_name,
            "object": str(intervals[0]["object"]),
            "accepted_update_count": int(
                sum(bool(row["candidate_accepted"]) for row in intervals)
            ),
        }
        for metric in ("identity", "chamfer"):
            baseline = np.asarray(
                [float(row[f"baseline_{metric}_m"]) for row in intervals]
            )
            candidate = np.asarray(
                [float(row[f"candidate_{metric}_m"]) for row in intervals]
            )
            accepted = np.asarray(
                [bool(row["candidate_accepted"]) for row in intervals]
            )
            guarded = np.where(accepted, candidate, baseline)
            record[f"baseline_{metric}_m"] = float(np.mean(baseline))
            record[f"candidate_{metric}_m"] = float(np.mean(candidate))
            record[f"guarded_{metric}_m"] = float(np.mean(guarded))
        case_rows.append(record)

    objects = sorted({str(row["object"]) for row in case_rows})

    def object_balanced(name: str) -> float:
        return float(
            np.mean(
                [
                    np.mean(
                        [float(row[name]) for row in case_rows if row["object"] == obj]
                    )
                    for obj in objects
                ]
            )
        )

    metrics: dict[str, Any] = {}
    for metric in ("identity", "chamfer"):
        baseline = object_balanced(f"baseline_{metric}_m")
        candidate = object_balanced(f"candidate_{metric}_m")
        guarded = object_balanced(f"guarded_{metric}_m")
        metrics[metric] = {
            "baseline_m": baseline,
            "candidate_m": candidate,
            "guarded_m": guarded,
            "candidate_relative_percent": 100.0 * (candidate / baseline - 1.0),
            "guarded_relative_percent": 100.0 * (guarded / baseline - 1.0),
            "guarded_case_wins": int(
                sum(
                    row[f"guarded_{metric}_m"] < row[f"baseline_{metric}_m"]
                    for row in case_rows
                )
            ),
            "guarded_case_ties": int(
                sum(
                    row[f"guarded_{metric}_m"] == row[f"baseline_{metric}_m"]
                    for row in case_rows
                )
            ),
            "guarded_case_regressions": int(
                sum(
                    row[f"guarded_{metric}_m"] > row[f"baseline_{metric}_m"]
                    for row in case_rows
                )
            ),
        }
    return {
        "object_count": len(objects),
        "case_count": len(case_rows),
        "update_count": len(selected),
        "accepted_update_count": int(
            sum(bool(row["candidate_accepted"]) for row in selected)
        ),
        "accepted_beneficial_update_count": int(
            sum(
                bool(row["candidate_accepted"])
                and float(row["maximum_regret_m"]) < 0.0
                for row in selected
            )
        ),
        "accepted_regressive_update_count": int(
            sum(
                bool(row["candidate_accepted"])
                and float(row["maximum_regret_m"]) > 0.0
                for row in selected
            )
        ),
        "metrics": metrics,
        "cases": case_rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--open27-tactile", type=Path, required=True)
    parser.add_argument("--stress12-tactile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument("--admission-threshold", type=float, default=0.7)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_paths = {
        "protocol": args.protocol.resolve(),
        "source_result": args.source_result.resolve(),
        "open27_tactile": args.open27_tactile.resolve(),
        "stress12_tactile": args.stress12_tactile.resolve(),
    }
    protocol = _read_json(input_paths["protocol"])
    _require(
        protocol.get("protocol_id")
        == "deform360-tactile-regret-guard-source-v1"
        and protocol.get("protocol_sha256") == _protocol_sha256(protocol),
        "tactile source protocol changed",
    )
    implementation = protocol["implementation"]
    _require(
        _file_sha256(
            REPOSITORY_ROOT
            / "src/bayesian_phystwin/deform360_tactile_regret_guard.py"
        )
        == implementation["tactile_guard_module_sha256"]
        and _file_sha256(Path(__file__).resolve())
        == implementation["diagnostic_runner_sha256"],
        "tactile source implementation changed",
    )
    method = protocol["method"]
    _require(
        float(method["ridge_penalty"]) == args.ridge_penalty
        and float(method["admission_threshold"]) == args.admission_threshold
        and tuple(method["feature_names"]) == TACTILE_REGRET_FEATURE_NAMES,
        "tactile source method arguments changed",
    )
    bound_inputs = protocol["inputs"]
    _require(
        _file_sha256(input_paths["source_result"])
        == bound_inputs["pairwise_source_result_sha256"]
        and _file_sha256(input_paths["open27_tactile"])
        == bound_inputs["open27_tactile_features_sha256"]
        and _file_sha256(input_paths["stress12_tactile"])
        == bound_inputs["stress12_tactile_features_sha256"],
        "tactile source input changed",
    )
    source_result = _read_json(input_paths["source_result"])
    open27_tactile = _read_json(input_paths["open27_tactile"])
    stress12_tactile = _read_json(input_paths["stress12_tactile"])
    tactile = _tactile_by_case(open27_tactile, stress12_tactile)
    rows = _source_rows(source_result, tactile) + _stress_rows(source_result, tactile)
    _require(
        len(rows) == 117 and len({row["object"] for row in rows}) == 17,
        "opened tactile source panel changed",
    )
    decisions = _cross_fitted_decisions(
        rows,
        ridge_penalty=args.ridge_penalty,
        admission_threshold=args.admission_threshold,
    )
    final_model = fit_object_balanced_tactile_regret_guard(
        np.asarray([row["feature"] for row in rows]),
        np.asarray([row["maximum_regret_m"] for row in rows]),
        [str(row["object"]) for row in rows],
        ridge_penalty=args.ridge_penalty,
        admission_threshold=args.admission_threshold,
    )
    open27 = _panel_summary(decisions, "open27")
    stress12 = _panel_summary(decisions, "stress12")
    combined = _panel_summary(decisions, None)
    gates = {
        "open27_identity_improvement_at_least_2_percent": bool(
            open27["metrics"]["identity"]["guarded_relative_percent"] <= -2.0
        ),
        "open27_chamfer_improvement_at_least_2_percent": bool(
            open27["metrics"]["chamfer"]["guarded_relative_percent"] <= -2.0
        ),
        "stress12_zero_regressive_updates": bool(
            stress12["accepted_regressive_update_count"] == 0
        ),
        "combined_zero_regressive_updates": bool(
            combined["accepted_regressive_update_count"] == 0
        ),
        "at_least_three_beneficial_updates_admitted": bool(
            combined["accepted_beneficial_update_count"] >= 3
        ),
    }
    payload = {
        "artifact_kind": "Deform360TactileRegretGuardSourceDiagnostic",
        "claim_boundary": (
            "Post-open, object-cross-fitted source-development evidence. It does not "
            "establish prospective accuracy or state of the art."
        ),
        "method": {
            "candidate_arm": "dual_backbone_pairwise_consensus_rbf",
            "baseline_arm": "selected_raw_backbone",
            "guard_arm": "dual_backbone_pairwise_tactile_guarded",
            "raw_baseline_subtracted_tactile": True,
            "episode_wide_tactile_normalization_used": False,
            "history_frame_count": 3,
            "initial_reference_frame_count": 6,
            "update_frames": [start for start, _ in UPDATE_INTERVALS],
            "ridge_penalty": args.ridge_penalty,
            "admission_threshold": args.admission_threshold,
            "object_balanced_source_loss": True,
            "held_object_excluded_from_each_fit": True,
            "rejected_interval_is_bit_exact_baseline": True,
        },
        "inputs": {
            name: {"path": _display_path(path), "file_sha256": _file_sha256(path)}
            for name, path in input_paths.items()
        },
        "information_boundary": {
            "opened_source_only": True,
            "future_tactile_read": False,
            "target_argument_accepted": False,
            "held_v8_read": False,
            "source_outcomes_used_to_fit_guard": True,
        },
        "cross_fitted": {
            "open27": open27,
            "stress12": stress12,
            "combined": combined,
            "decisions": decisions,
        },
        "full_source_model_for_future_lock": _model_dict(final_model),
        "advancement_gates": gates,
        "all_advancement_gates_passed": bool(all(gates.values())),
        "recommendation": (
            "Lock a genuinely fresh, non-overlapping prospective object cohort."
            if all(gates.values())
            else "Do not advance this tactile guard."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact_sha256": payload["artifact_sha256"],
        "all_advancement_gates_passed": payload["all_advancement_gates_passed"],
        "open27_identity_percent": open27["metrics"]["identity"]["guarded_relative_percent"],
        "open27_chamfer_percent": open27["metrics"]["chamfer"]["guarded_relative_percent"],
        "stress12_regressive_updates": stress12["accepted_regressive_update_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
