#!/usr/bin/env python3
"""Evaluate the v2 cross-modal support guard on opened source panels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_cross_modal_support_guard import (  # noqa: E402
    CROSS_MODAL_SUPPORT_FEATURE_NAMES,
    CrossModalSupportGuardModel,
    cross_modal_support_decisions,
    fit_cross_modal_support_guard,
)

_SOURCE_V1_SPEC = importlib.util.spec_from_file_location(
    "_deform360_causal_support_guard_source_v1",
    REPOSITORY_ROOT / "scripts/remote/diagnose_deform360_causal_support_guard.py",
)
if _SOURCE_V1_SPEC is None or _SOURCE_V1_SPEC.loader is None:
    raise RuntimeError("could not load causal-support source v1 helpers")
source_v1 = importlib.util.module_from_spec(_SOURCE_V1_SPEC)
_SOURCE_V1_SPEC.loader.exec_module(source_v1)


def _extra_feature_lookup(
    source_result: dict[str, Any],
    tactile: dict[str, dict[int, dict[str, Any]]],
) -> dict[tuple[str, int], tuple[float, float]]:
    coherence: dict[tuple[str, int], float] = {}
    for case in source_result["source_interval_diagnostics"]:
        for interval in case["intervals"]:
            key = (str(case["case"]), int(interval["frame"]))
            coherence[key] = float(interval["features"]["correction_coherence"])
    for case in source_result["opened_stress"]["cases"]:
        for update, features in zip(
            case["candidate_report"]["updates"],
            case["feature_diagnostics"],
            strict=True,
        ):
            key = (str(case["case"]), int(update["frame"]))
            coherence[key] = float(features["correction_coherence"])

    output = {}
    for case_name, updates in tactile.items():
        for frame, tactile_update in updates.items():
            key = (case_name, frame)
            source_v1._require(key in coherence, "cross-modal coherence row missing")
            output[key] = (
                float(
                    tactile_update[
                        "cumulative_energy_change_from_frame0_fraction"
                    ]
                ),
                coherence[key],
            )
    return output


def _rows(
    source_result: dict[str, Any],
    tactile: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = source_v1._source_rows(source_result, tactile) + source_v1._stress_rows(
        source_result,
        tactile,
    )
    extra = _extra_feature_lookup(source_result, tactile)
    for row in rows:
        key = (str(row["case"]), int(row["frame"]))
        row["feature"] = np.concatenate(
            (np.asarray(row["feature"], dtype=np.float64), np.asarray(extra[key]))
        )
        row["candidate_nontrivial"] = not (
            float(row["baseline_identity_m"]) == float(row["candidate_identity_m"])
            and float(row["baseline_chamfer_m"])
            == float(row["candidate_chamfer_m"])
        )
    source_v1._require(
        all(
            np.asarray(row["feature"]).shape
            == (len(CROSS_MODAL_SUPPORT_FEATURE_NAMES),)
            for row in rows
        ),
        "cross-modal source feature shape changed",
    )
    return rows


def _fit(rows: list[dict[str, Any]]) -> CrossModalSupportGuardModel:
    return fit_cross_modal_support_guard(
        np.asarray([row["feature"] for row in rows]),
        np.asarray([row["maximum_regret_m"] for row in rows]),
        [str(row["object"]) for row in rows],
        candidate_nontrivial=[bool(row["candidate_nontrivial"]) for row in rows],
    )


def _model_dict(model: CrossModalSupportGuardModel) -> dict[str, Any]:
    cross_modal = model.stable_tactile_coherent_correction
    return {
        "feature_names": list(CROSS_MODAL_SUPPORT_FEATURE_NAMES),
        "source_object_count": model.source_object_count,
        "source_row_count": model.source_row_count,
        "source_informative_row_count": model.source_informative_row_count,
        "regret_tolerance_m": model.regret_tolerance_m,
        "causal_support": source_v1._model_dict(model.causal_support),
        "stable_tactile_coherent_correction": {
            "name": "stable_tactile_coherent_correction",
            "evidence_kind": "cross-modal-regime-support",
            "maximum_cumulative_energy_change": (
                cross_modal.maximum_cumulative_energy_change
            ),
            "minimum_correction_coherence": (
                cross_modal.minimum_correction_coherence
            ),
            "enabled": cross_modal.enabled,
            "source_beneficial_admission_count": (
                cross_modal.source_beneficial_admission_count
            ),
            "source_regressive_admission_count": (
                cross_modal.source_regressive_admission_count
            ),
        },
    }


def _cross_fitted(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = []
    fold_models = []
    for held_object in sorted({str(row["object"]) for row in rows}):
        training = [row for row in rows if row["object"] != held_object]
        held = [row for row in rows if row["object"] == held_object]
        model = _fit(training)
        fold_models.append(
            {
                "held_object": held_object,
                "held_object_excluded_from_fit": True,
                "model": _model_dict(model),
            }
        )
        support = cross_modal_support_decisions(
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
                    "causal_support_routes": decision["causal_support_routes"],
                    "cross_modal_route": decision["cross_modal_route"],
                    "bit_exact_baseline_fallback": not accepted,
                    "held_object_excluded_from_fit": True,
                    "fit_source_object_count": model.source_object_count,
                }
            )
    return (
        sorted(decisions, key=lambda row: (row["case"], row["frame"])),
        fold_models,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-v1-result", type=Path, required=True)
    parser.add_argument("--open27-tactile", type=Path, required=True)
    parser.add_argument("--stress12-tactile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        "protocol": args.protocol.resolve(),
        "source_result": args.source_result.resolve(),
        "source_v1_result": args.source_v1_result.resolve(),
        "open27_tactile": args.open27_tactile.resolve(),
        "stress12_tactile": args.stress12_tactile.resolve(),
    }
    protocol = source_v1._read_json(paths["protocol"])
    source_v1._require(
        protocol.get("protocol_id")
        == "deform360-cross-modal-support-guard-source-v2"
        and protocol.get("protocol_sha256") == source_v1._protocol_sha256(protocol),
        "cross-modal source protocol changed",
    )
    implementation = protocol["implementation"]
    implementation_paths = {
        "cross_modal_guard_module_sha256": REPOSITORY_ROOT
        / "src/bayesian_phystwin/deform360_cross_modal_support_guard.py",
        "causal_support_guard_module_sha256": REPOSITORY_ROOT
        / "src/bayesian_phystwin/deform360_causal_support_guard.py",
        "diagnostic_runner_sha256": Path(__file__).resolve(),
        "source_v1_runner_sha256": REPOSITORY_ROOT
        / "scripts/remote/diagnose_deform360_causal_support_guard.py",
    }
    source_v1._require(
        all(
            source_v1._file_sha256(path) == implementation[name]
            for name, path in implementation_paths.items()
        ),
        "cross-modal source implementation changed",
    )
    inputs = protocol["inputs"]
    source_v1._require(
        source_v1._file_sha256(paths["source_result"])
        == inputs["pairwise_source_result_sha256"]
        and source_v1._file_sha256(paths["source_v1_result"])
        == inputs["causal_support_source_v1_result_sha256"]
        and source_v1._file_sha256(paths["open27_tactile"])
        == inputs["open27_tactile_features_sha256"]
        and source_v1._file_sha256(paths["stress12_tactile"])
        == inputs["stress12_tactile_features_sha256"],
        "cross-modal source input changed",
    )
    source_v1_result = source_v1._read_json(paths["source_v1_result"])
    source_v1._require(
        source_v1_result.get("artifact_sha256")
        == source_v1._canonical_sha256(source_v1_result)
        and source_v1_result.get("all_development_advancement_checks_passed")
        is False
        and source_v1_result["development_advancement_checks"].get(
            "at_least_five_objects_with_admission"
        )
        is False,
        "causal-support source v1 predecessor changed",
    )
    source_v1._require(
        tuple(protocol["method"]["feature_names"])
        == CROSS_MODAL_SUPPORT_FEATURE_NAMES,
        "cross-modal source features changed",
    )

    source_result = source_v1._read_json(paths["source_result"])
    tactile = source_v1._tactile_by_case(
        source_v1._read_json(paths["open27_tactile"]),
        source_v1._read_json(paths["stress12_tactile"]),
    )
    rows = _rows(source_result, tactile)
    source_v1._require(
        len(rows) == 117 and len({row["object"] for row in rows}) == 17,
        "cross-modal opened source panel changed",
    )
    decisions, fold_models = _cross_fitted(rows)
    final_model = _fit(rows)
    open27 = source_v1._panel_summary(decisions, "open27")
    stress12 = source_v1._panel_summary(decisions, "stress12")
    combined = source_v1._panel_summary(decisions, None)
    joint_case_wins = int(
        sum(
            case["guarded_identity_m"] < case["baseline_identity_m"]
            and case["guarded_chamfer_m"] < case["baseline_chamfer_m"]
            for case in combined["cases"]
        )
    )
    gates = {
        "open27_identity_improvement_at_least_5_percent": bool(
            open27["metrics"]["identity"]["guarded_relative_percent"] <= -5.0
        ),
        "open27_chamfer_improvement_at_least_5_percent": bool(
            open27["metrics"]["chamfer"]["guarded_relative_percent"] <= -5.0
        ),
        "combined_zero_regressive_updates": bool(
            combined["accepted_regressive_update_count"] == 0
        ),
        "stress12_zero_regressive_updates": bool(
            stress12["accepted_regressive_update_count"] == 0
        ),
        "at_least_twelve_beneficial_updates_admitted": bool(
            combined["accepted_beneficial_update_count"] >= 12
        ),
        "at_least_six_joint_case_wins": bool(joint_case_wins >= 6),
        "at_least_five_objects_with_admission": bool(
            combined["admitted_object_count"] >= 5
        ),
    }
    payload = {
        "artifact_kind": "Deform360CrossModalSupportGuardSourceDiagnostic",
        "claim_boundary": (
            "Post-open, object-cross-fitted source-development evidence. The "
            "cross-modal route was identified after v1 failed its breadth check; "
            "this result cannot establish prospective accuracy or state of the art."
        ),
        "predecessor": {
            "artifact_sha256": source_v1_result["artifact_sha256"],
            "status": "failed_nontrivial_admission_breadth",
            "preserved_without_relabeling_noops": True,
        },
        "method": {
            "candidate_arm": "dual_backbone_pairwise_consensus_rbf",
            "baseline_arm": "selected_raw_backbone",
            "guard_arm": "dual_backbone_cross_modal_support_union_guarded",
            "feature_names": list(CROSS_MODAL_SUPPORT_FEATURE_NAMES),
            "v1_routes_unchanged": True,
            "new_route": "stable_tactile_coherent_correction",
            "new_route_operator": "logical_and",
            "all_routes_operator": "logical_or",
            "held_object_excluded_from_each_fit": True,
            "candidate_noop_is_not_an_admission": True,
            "rejected_interval_is_bit_exact_baseline": True,
        },
        "inputs": {
            name: {
                "path": source_v1._display_path(path),
                "file_sha256": source_v1._file_sha256(path),
            }
            for name, path in paths.items()
        },
        "information_boundary": {
            "opened_source_only": True,
            "source_outcomes_used_to_select_cross_modal_route_and_thresholds": True,
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
            "fold_models": fold_models,
            "decisions": decisions,
        },
        "full_source_model_for_future_lock": _model_dict(final_model),
        "development_advancement_checks": gates,
        "all_development_advancement_checks_passed": bool(all(gates.values())),
        "recommendation": (
            "Write and lock a genuinely fresh, non-overlapping prospective protocol."
            if all(gates.values())
            else "Do not advance this cross-modal support guard."
        ),
    }
    payload["artifact_sha256"] = source_v1._canonical_sha256(payload)
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
                "admitted_object_count": combined["admitted_object_count"],
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
