#!/usr/bin/env python3
"""Evaluate the causal-support union guard on already-open source panels."""

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

from bayesian_phystwin.deform360_causal_support_guard import (  # noqa: E402
    CAUSAL_SUPPORT_FEATURE_NAMES,
    CausalSupportGuardModel,
    causal_support_decisions,
    causal_support_feature_vector,
    fit_causal_support_guard,
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
        for case in payload["cases"]:
            case_name = str(case["case"])
            _require(case_name not in output, "duplicate tactile case")
            updates = {
                int(row["update_frame"]): dict(row) for row in case["updates"]
            }
            _require(
                tuple(updates) == tuple(start for start, _ in UPDATE_INTERVALS),
                "tactile update schedule changed",
            )
            output[case_name] = updates
    return output


def _tactile_features(update: dict[str, Any]) -> dict[str, float]:
    ratios = np.asarray(update["sensor_energy_over_initial"], dtype=np.float64)
    _require(ratios.ndim == 1 and np.all(np.isfinite(ratios)), "invalid tactile ratio")
    return {"sensor_ratio_max": float(np.max(ratios))}


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


def _row(
    *,
    panel: str,
    object_id: str,
    case: str,
    frame: int,
    pairwise_features: dict[str, Any],
    tactile_update: dict[str, Any],
    baseline_identity_m: float,
    candidate_identity_m: float,
    baseline_chamfer_m: float,
    candidate_chamfer_m: float,
) -> dict[str, Any]:
    maximum_regret = max(
        candidate_identity_m - baseline_identity_m,
        candidate_chamfer_m - baseline_chamfer_m,
    )
    candidate_nontrivial = not (
        candidate_identity_m == baseline_identity_m
        and candidate_chamfer_m == baseline_chamfer_m
    )
    return {
        "panel": panel,
        "object": object_id,
        "case": case,
        "frame": frame,
        "feature": causal_support_feature_vector(
            _tactile_features(tactile_update),
            pairwise_features,
        ),
        "maximum_regret_m": float(maximum_regret),
        "candidate_nontrivial": candidate_nontrivial,
        "baseline_identity_m": baseline_identity_m,
        "candidate_identity_m": candidate_identity_m,
        "baseline_chamfer_m": baseline_chamfer_m,
        "candidate_chamfer_m": candidate_chamfer_m,
    }


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
                _row(
                    panel="open27",
                    object_id=str(case["object_id"]),
                    case=case_name,
                    frame=frame,
                    pairwise_features=dict(interval["features"]),
                    tactile_update=tactile[case_name][frame],
                    baseline_identity_m=float(
                        interval["baseline_hidden_identity_rmse_m"]
                    ),
                    candidate_identity_m=float(
                        interval["candidate_hidden_identity_rmse_m"]
                    ),
                    baseline_chamfer_m=float(
                        interval["baseline_hidden_symmetric_chamfer_m"]
                    ),
                    candidate_chamfer_m=float(
                        interval["candidate_hidden_symmetric_chamfer_m"]
                    ),
                )
            )
    return rows


def _stress_rows(
    source_result: dict[str, Any],
    tactile: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for case in source_result["opened_stress"]["cases"]:
        case_name = str(case["case"])
        baseline = case["scores"]["selected_raw_backbone"]
        candidate = case["scores"]["dual_backbone_pairwise_consensus_rbf"]
        feature_by_frame = {
            int(update["frame"]): dict(features)
            for update, features in zip(
                case["candidate_report"]["updates"],
                case["feature_diagnostics"],
                strict=True,
            )
        }
        for update, stop in UPDATE_INTERVALS:
            rows.append(
                _row(
                    panel="stress12",
                    object_id=str(case["object_id"]),
                    case=case_name,
                    frame=update,
                    pairwise_features=feature_by_frame[update],
                    tactile_update=tactile[case_name][update],
                    baseline_identity_m=_mean_interval_metric(
                        baseline,
                        IDENTITY_METRIC,
                        update=update,
                        stop=stop,
                    ),
                    candidate_identity_m=_mean_interval_metric(
                        candidate,
                        IDENTITY_METRIC,
                        update=update,
                        stop=stop,
                    ),
                    baseline_chamfer_m=_mean_interval_metric(
                        baseline,
                        CHAMFER_METRIC,
                        update=update,
                        stop=stop,
                    ),
                    candidate_chamfer_m=_mean_interval_metric(
                        candidate,
                        CHAMFER_METRIC,
                        update=update,
                        stop=stop,
                    ),
                )
            )
    return rows


def _model_dict(model: CausalSupportGuardModel) -> dict[str, Any]:
    return {
        "feature_names": list(CAUSAL_SUPPORT_FEATURE_NAMES),
        "source_object_count": model.source_object_count,
        "source_row_count": model.source_row_count,
        "source_informative_row_count": model.source_informative_row_count,
        "regret_tolerance_m": model.regret_tolerance_m,
        "routes": [
            {
                "name": route.name,
                "feature_name": route.feature_name,
                "direction": route.direction,
                "evidence_kind": route.evidence_kind,
                "threshold": route.threshold,
                "enabled": route.enabled,
                "source_beneficial_admission_count": (
                    route.source_beneficial_admission_count
                ),
                "source_regressive_admission_count": (
                    route.source_regressive_admission_count
                ),
            }
            for route in model.routes
        ],
    }


def _fit(rows: list[dict[str, Any]]) -> CausalSupportGuardModel:
    return fit_causal_support_guard(
        np.asarray([row["feature"] for row in rows]),
        np.asarray([row["maximum_regret_m"] for row in rows]),
        [str(row["object"]) for row in rows],
        candidate_nontrivial=[bool(row["candidate_nontrivial"]) for row in rows],
    )


def _cross_fitted_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    for held_object in sorted({str(row["object"]) for row in rows}):
        training = [row for row in rows if row["object"] != held_object]
        held = [row for row in rows if row["object"] == held_object]
        model = _fit(training)
        support = causal_support_decisions(
            np.asarray([row["feature"] for row in held]),
            model,
        )
        for row, decision in zip(held, support, strict=True):
            accepted = bool(
                decision["support_available"] and row["candidate_nontrivial"]
            )
            decisions.append(
                {
                    **{key: value for key, value in row.items() if key != "feature"},
                    "candidate_accepted": accepted,
                    "support_available": bool(decision["support_available"]),
                    "admitting_routes": decision["admitting_routes"],
                    "routes": decision["routes"],
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
        decisions
        if panel is None
        else [row for row in decisions if row["panel"] == panel]
    )
    _require(bool(selected), "metric panel is empty")
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
        "admitted_object_count": len(
            {
                str(row["object"])
                for row in selected
                if bool(row["candidate_accepted"])
            }
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        "protocol": args.protocol.resolve(),
        "source_result": args.source_result.resolve(),
        "open27_tactile": args.open27_tactile.resolve(),
        "stress12_tactile": args.stress12_tactile.resolve(),
    }
    protocol = _read_json(paths["protocol"])
    _require(
        protocol.get("protocol_id") == "deform360-causal-support-guard-source-v1"
        and protocol.get("protocol_sha256") == _protocol_sha256(protocol),
        "causal-support source protocol changed",
    )
    implementation = protocol["implementation"]
    _require(
        _file_sha256(
            REPOSITORY_ROOT
            / "src/bayesian_phystwin/deform360_causal_support_guard.py"
        )
        == implementation["guard_module_sha256"]
        and _file_sha256(Path(__file__).resolve())
        == implementation["diagnostic_runner_sha256"],
        "causal-support source implementation changed",
    )
    inputs = protocol["inputs"]
    _require(
        _file_sha256(paths["source_result"])
        == inputs["pairwise_source_result_sha256"]
        and _file_sha256(paths["open27_tactile"])
        == inputs["open27_tactile_features_sha256"]
        and _file_sha256(paths["stress12_tactile"])
        == inputs["stress12_tactile_features_sha256"],
        "causal-support source input changed",
    )
    _require(
        tuple(protocol["method"]["feature_names"])
        == CAUSAL_SUPPORT_FEATURE_NAMES,
        "causal-support source features changed",
    )

    source_result = _read_json(paths["source_result"])
    tactile = _tactile_by_case(
        _read_json(paths["open27_tactile"]),
        _read_json(paths["stress12_tactile"]),
    )
    rows = _source_rows(source_result, tactile) + _stress_rows(source_result, tactile)
    _require(
        len(rows) == 117 and len({row["object"] for row in rows}) == 17,
        "causal-support opened source panel changed",
    )
    decisions = _cross_fitted_decisions(rows)
    final_model = _fit(rows)
    open27 = _panel_summary(decisions, "open27")
    stress12 = _panel_summary(decisions, "stress12")
    combined = _panel_summary(decisions, None)
    joint_case_wins = int(
        sum(
            case["guarded_identity_m"] < case["baseline_identity_m"]
            and case["guarded_chamfer_m"] < case["baseline_chamfer_m"]
            for case in combined["cases"]
        )
    )
    gates = {
        "open27_identity_improvement_at_least_4_percent": bool(
            open27["metrics"]["identity"]["guarded_relative_percent"] <= -4.0
        ),
        "open27_chamfer_improvement_at_least_4_percent": bool(
            open27["metrics"]["chamfer"]["guarded_relative_percent"] <= -4.0
        ),
        "combined_zero_regressive_updates": bool(
            combined["accepted_regressive_update_count"] == 0
        ),
        "stress12_zero_regressive_updates": bool(
            stress12["accepted_regressive_update_count"] == 0
        ),
        "at_least_ten_beneficial_updates_admitted": bool(
            combined["accepted_beneficial_update_count"] >= 10
        ),
        "at_least_five_joint_case_wins": bool(joint_case_wins >= 5),
        "at_least_five_objects_with_admission": bool(
            combined["admitted_object_count"] >= 5
        ),
    }
    payload = {
        "artifact_kind": "Deform360CausalSupportGuardSourceDiagnostic",
        "claim_boundary": (
            "Post-open, object-cross-fitted source-development evidence. The three "
            "routes were chosen after examining opened source evidence, so this "
            "result cannot establish prospective accuracy or state of the art."
        ),
        "method": {
            "candidate_arm": "dual_backbone_pairwise_consensus_rbf",
            "baseline_arm": "selected_raw_backbone",
            "guard_arm": "dual_backbone_causal_support_union_guarded",
            "feature_names": list(CAUSAL_SUPPORT_FEATURE_NAMES),
            "route_union": "logical_or",
            "held_object_excluded_from_each_fit": True,
            "route_objective": (
                "maximize beneficial admissions with zero regressive source "
                "admissions; then minimize admissions and choose stricter threshold"
            ),
            "candidate_noop_is_not_an_admission": True,
            "rejected_interval_is_bit_exact_baseline": True,
        },
        "inputs": {
            name: {"path": _display_path(path), "file_sha256": _file_sha256(path)}
            for name, path in paths.items()
        },
        "information_boundary": {
            "opened_source_only": True,
            "source_outcomes_used_to_select_routes_and_thresholds": True,
            "future_tactile_read": False,
            "future_camera_observation_read": False,
            "target_argument_accepted": False,
            "held_v8_read": False,
        },
        "cross_fitted": {
            "open27": open27,
            "stress12": stress12,
            "combined": combined,
            "joint_case_wins": joint_case_wins,
            "decisions": decisions,
        },
        "full_source_model_for_future_lock": _model_dict(final_model),
        "development_advancement_checks": gates,
        "all_development_advancement_checks_passed": bool(all(gates.values())),
        "recommendation": (
            "Write and lock a genuinely fresh, non-overlapping prospective protocol."
            if all(gates.values())
            else "Do not advance this causal-support guard."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "artifact_sha256": payload["artifact_sha256"],
                "all_development_advancement_checks_passed": payload[
                    "all_development_advancement_checks_passed"
                ],
                "combined_beneficial_updates": combined[
                    "accepted_beneficial_update_count"
                ],
                "combined_regressive_updates": combined[
                    "accepted_regressive_update_count"
                ],
                "open27_identity_percent": open27["metrics"]["identity"][
                    "guarded_relative_percent"
                ],
                "open27_chamfer_percent": open27["metrics"]["chamfer"][
                    "guarded_relative_percent"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
