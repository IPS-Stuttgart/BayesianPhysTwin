"""Cross-material real-cloth screen for selective simulator competence.

This study reuses the maintained Tracking Cloth spring-mesh implementation. It
is deliberately retrospective because the twisting outcomes were opened in
workflow run 33302686759. Every held-out material is nevertheless scored by a
gate fitted only on the other three materials, and every rejection returns the
registered persistence loss exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tracking_cloth_deformation_v1.data import (
    Case,
    Inputs,
    audit_dataset,
    digest,
    infer_source_scale,
    input_view,
    object_digest,
    read_prefix,
    scoring_view,
    write_json,
)
from experiments.tracking_cloth_deformation_v1.model import (
    masks,
    means,
    predict,
    source_weights,
    squared_error,
)

HERE = Path(__file__).resolve().parent
BASE_HERE = HERE.parent / "tracking_cloth_deformation_v1"
POLICIES = (
    "always_fallback",
    "always_candidate",
    "motion_gate",
    "query_horizon_gate",
    "oracle",
)


@dataclass(frozen=True)
class RecordPrediction:
    case: Case
    inputs: Inputs
    truth: np.ndarray
    candidate: np.ndarray
    map_physics: np.ndarray
    last_residual: np.ndarray
    nominal_physics: np.ndarray
    persistence: np.ndarray
    bank: np.ndarray
    weights: np.ndarray


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _implementation() -> dict[str, str]:
    files = sorted(HERE.glob("*.py")) + [
        BASE_HERE / "data.py",
        BASE_HERE / "model.py",
    ]
    return {str(path.relative_to(HERE.parents[1])): digest(path) for path in files}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _free_indices(inputs: Inputs) -> np.ndarray:
    corners = set(inputs.corners.tolist())
    return np.asarray(
        [i for i in range(len(inputs.order)) if i not in corners],
        dtype=int,
    )


def _grid_shape(marker_count: int) -> tuple[int, int]:
    if marker_count == 20:
        return 5, 4
    if marker_count == 12:
        return 4, 3
    raise ValueError(f"Unsupported marker count: {marker_count}")


def query_value(
    positions: np.ndarray,
    inputs: Inputs,
    query: str,
) -> np.ndarray:
    """Map a trajectory to a registered physical query time series."""
    free = _free_indices(inputs)
    if query == "free_marker_shape":
        return positions[:, free]
    if query == "free_marker_centroid":
        return positions[:, free].mean(axis=1)
    rows, cols = _grid_shape(len(inputs.order))
    bottom = np.arange((rows - 1) * cols, rows * cols)
    if query == "bottom_edge_centroid":
        return positions[:, bottom].mean(axis=1)
    if query == "shape_radius":
        values = positions[:, free]
        centre = values.mean(axis=1, keepdims=True)
        return np.sqrt(np.mean(np.sum((values - centre) ** 2, axis=2), axis=1))
    raise ValueError(f"Unknown query: {query}")


def _future_indices(inputs: Inputs, horizon_seconds: float) -> np.ndarray:
    start = float(inputs.times[inputs.cutoff])
    dt = float(np.median(np.diff(inputs.times)))
    selected = np.flatnonzero(
        (inputs.times > start + 1e-10)
        & (inputs.times <= start + horizon_seconds + 0.51 * dt)
    )
    if not len(selected):
        raise ValueError(f"No forecast samples for horizon {horizon_seconds}")
    return selected


def query_rmse_mm(
    prediction: np.ndarray,
    truth: np.ndarray,
    inputs: Inputs,
    query: str,
    horizon_seconds: float,
) -> float:
    selected = _future_indices(inputs, horizon_seconds)
    p = query_value(prediction, inputs, query)[selected]
    y = query_value(truth, inputs, query)[selected]
    error = p - y
    if query == "free_marker_shape":
        valid = np.isfinite(y).all(axis=2)
        squared = np.sum(error**2, axis=2)[valid]
    elif error.ndim == 2:
        valid = np.isfinite(y).all(axis=1)
        squared = np.sum(error**2, axis=1)[valid]
    else:
        valid = np.isfinite(y)
        squared = (error**2)[valid]
    if not squared.size:
        raise ValueError(f"No valid samples for {query} at {horizon_seconds} s")
    return 1000.0 * float(np.sqrt(np.mean(squared)))


def query_disagreement_mm(
    first: np.ndarray,
    second: np.ndarray,
    inputs: Inputs,
    query: str,
    horizon_seconds: float,
) -> float:
    return query_rmse_mm(first, second, inputs, query, horizon_seconds)


def query_ensemble_spread_mm(
    bank: np.ndarray,
    weights: np.ndarray,
    inputs: Inputs,
    query: str,
    horizon_seconds: float,
) -> float:
    selected = _future_indices(inputs, horizon_seconds)
    values = np.stack([query_value(member, inputs, query)[selected] for member in bank])
    mean = np.einsum("k,k...->...", weights, values)
    delta = values - mean
    if query == "free_marker_shape":
        squared = np.sum(delta**2, axis=3)
    elif query in {"free_marker_centroid", "bottom_edge_centroid"}:
        squared = np.sum(delta**2, axis=2)
    else:
        squared = delta**2
    weighted = np.einsum("k,k...->...", weights, squared)
    return 1000.0 * float(np.sqrt(np.mean(weighted)))


def initial_diameter_mm(inputs: Inputs) -> float:
    initial = inputs.prefix[0]
    distances = np.linalg.norm(initial[:, None] - initial[None, :], axis=2)
    value = 1000.0 * float(np.max(distances))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Invalid initial cloth diameter")
    return value


def _source_predictions(
    cases: list[Case],
    base_protocol: dict[str, Any],
    scale: float,
) -> tuple[list[RecordPrediction], dict[str, np.ndarray]]:
    by_specimen: dict[str, list[tuple[Case, Inputs, Any, np.ndarray]]] = defaultdict(
        list
    )
    for case in cases:
        if case.motion != "shake":
            continue
        inputs = input_view(case, base_protocol, scale)
        prediction = predict(inputs, base_protocol)
        truth = scoring_view(case, inputs)
        by_specimen[case.specimen].append((case, inputs, prediction, truth))

    records: list[RecordPrediction] = []
    full_weights: dict[str, np.ndarray] = {}
    for specimen, values in sorted(by_specimen.items()):
        values = sorted(values, key=lambda item: item[0].path.name)
        if len(values) != 4:
            raise ValueError(f"{specimen}: expected four shaking conditions")
        losses = np.asarray(
            [
                [
                    squared_error(member, truth, masks(inputs, truth))
                    for member in prediction.bank
                ]
                for _, inputs, prediction, truth in values
            ]
        )
        full_weights[specimen] = source_weights(
            losses, base_protocol["measurement_floor_m"]
        )
        for held, (case, inputs, prediction, truth) in enumerate(values):
            weights = source_weights(
                np.delete(losses, held, axis=0),
                base_protocol["measurement_floor_m"],
            )
            arm_means = means(prediction, weights, base_protocol)
            records.append(
                RecordPrediction(
                    case=case,
                    inputs=inputs,
                    truth=truth,
                    candidate=arm_means["bayesian_physics"],
                    map_physics=arm_means["map_physics"],
                    last_residual=arm_means["last_residual"],
                    nominal_physics=arm_means["nominal_physics"],
                    persistence=arm_means["persistence"],
                    bank=prediction.bank,
                    weights=weights,
                )
            )
    if len(records) != 32 or len(full_weights) != 8:
        raise ValueError("Incomplete shaking source roster")
    return records, full_weights


def _twist_predictions(
    cases: list[Case],
    base_protocol: dict[str, Any],
    scale: float,
    full_weights: dict[str, np.ndarray],
) -> list[RecordPrediction]:
    records: list[RecordPrediction] = []
    for case in sorted(cases, key=lambda item: item.path.name):
        if case.motion != "twist":
            continue
        inputs = input_view(case, base_protocol, scale)
        prediction = predict(inputs, base_protocol)
        weights = full_weights[case.specimen]
        arm_means = means(prediction, weights, base_protocol)
        records.append(
            RecordPrediction(
                case=case,
                inputs=inputs,
                truth=scoring_view(case, inputs),
                candidate=arm_means["bayesian_physics"],
                map_physics=arm_means["map_physics"],
                last_residual=arm_means["last_residual"],
                nominal_physics=arm_means["nominal_physics"],
                persistence=arm_means["persistence"],
                bank=prediction.bank,
                weights=weights,
            )
        )
    if len(records) != 32:
        raise ValueError("Incomplete twisting target roster")
    return records


def score_records(
    records: Iterable[RecordPrediction],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        diameter = initial_diameter_mm(record.inputs)
        harm_margin = protocol["practical_harm_fraction_of_initial_diameter"] * diameter
        for query in protocol["queries"]:
            for horizon in protocol["horizons_seconds"]:
                horizon_seconds = float(horizon)
                candidate = query_rmse_mm(
                    record.candidate,
                    record.truth,
                    record.inputs,
                    query,
                    horizon_seconds,
                )
                fallback = query_rmse_mm(
                    record.persistence,
                    record.truth,
                    record.inputs,
                    query,
                    horizon_seconds,
                )
                row = {
                    "recording": record.case.path.name,
                    "specimen": record.case.specimen,
                    "material": record.case.material,
                    "size": record.case.size,
                    "motion": record.case.motion,
                    "speed": record.case.speed,
                    "grasp": record.case.grasp,
                    "query": query,
                    "horizon_seconds": horizon_seconds,
                    "candidate_loss_mm": candidate,
                    "fallback_loss_mm": fallback,
                    "map_loss_mm": query_rmse_mm(
                        record.map_physics,
                        record.truth,
                        record.inputs,
                        query,
                        horizon_seconds,
                    ),
                    "last_residual_loss_mm": query_rmse_mm(
                        record.last_residual,
                        record.truth,
                        record.inputs,
                        query,
                        horizon_seconds,
                    ),
                    "nominal_loss_mm": query_rmse_mm(
                        record.nominal_physics,
                        record.truth,
                        record.inputs,
                        query,
                        horizon_seconds,
                    ),
                    "candidate_minus_fallback_mm": candidate - fallback,
                    "candidate_fallback_disagreement_mm": query_disagreement_mm(
                        record.candidate,
                        record.persistence,
                        record.inputs,
                        query,
                        horizon_seconds,
                    ),
                    "ensemble_spread_mm": query_ensemble_spread_mm(
                        record.bank,
                        record.weights,
                        record.inputs,
                        query,
                        horizon_seconds,
                    ),
                    "initial_diameter_mm": diameter,
                    "practical_harm_margin_mm": harm_margin,
                    "strict_regression": candidate > fallback,
                    "practical_harm": candidate > fallback + harm_margin,
                }
                rows.append(row)
    expected = 64 * len(protocol["queries"]) * len(protocol["horizons_seconds"])
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} query cases, found {len(rows)}")
    return rows


def _context_key(row: dict[str, Any], policy: str) -> tuple[Any, ...]:
    if policy == "motion_gate":
        return (row["motion"],)
    if policy == "query_horizon_gate":
        return (row["motion"], row["query"], float(row["horizon_seconds"]))
    raise ValueError(f"No learned context for policy {policy}")


def _fit_context_decisions(
    rows: list[dict[str, Any]],
    heldout_material: str,
    policy: str,
    protocol: dict[str, Any],
) -> dict[tuple[Any, ...], bool]:
    training = [row for row in rows if row["material"] != heldout_material]
    contexts = sorted({_context_key(row, policy) for row in training})
    decisions: dict[tuple[Any, ...], bool] = {}
    minimum_gain = float(protocol["primary_gate"]["minimum_relative_gain"])
    maximum_harm = float(
        protocol["primary_gate"]["maximum_training_practical_harm_fraction"]
    )
    for context in contexts:
        subset = [row for row in training if _context_key(row, policy) == context]
        materials = sorted({row["material"] for row in subset})
        if len(materials) != 3:
            raise ValueError("Each outer fold must train on exactly three materials")
        material_regret = [
            float(
                np.mean(
                    [
                        row["candidate_minus_fallback_mm"]
                        for row in subset
                        if row["material"] == material
                    ]
                )
            )
            for material in materials
        ]
        mean_regret = float(
            np.mean([row["candidate_minus_fallback_mm"] for row in subset])
        )
        mean_fallback = float(np.mean([row["fallback_loss_mm"] for row in subset]))
        harm_fraction = float(np.mean([bool(row["practical_harm"]) for row in subset]))
        decisions[context] = bool(
            mean_regret <= -minimum_gain * mean_fallback
            and max(material_regret) <= 0.0
            and harm_fraction <= maximum_harm
        )
    return decisions


def cross_material_policy_rows(
    rows: list[dict[str, Any]],
    policy: str,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    materials = protocol["materials"]
    for heldout in materials:
        test = [row for row in rows if row["material"] == heldout]
        decisions = (
            _fit_context_decisions(rows, heldout, policy, protocol)
            if policy in {"motion_gate", "query_horizon_gate"}
            else {}
        )
        for row in test:
            if policy == "always_fallback":
                accepted = False
            elif policy == "always_candidate":
                accepted = True
            elif policy == "oracle":
                accepted = row["candidate_loss_mm"] < row["fallback_loss_mm"]
            else:
                accepted = decisions[_context_key(row, policy)]
            selected = (
                float(row["candidate_loss_mm"])
                if accepted
                else float(row["fallback_loss_mm"])
            )
            fallback = float(row["fallback_loss_mm"])
            output.append(
                {
                    **row,
                    "policy": policy,
                    "heldout_material": heldout,
                    "accepted": accepted,
                    "selected_loss_mm": selected,
                    "selected_minus_fallback_mm": selected - fallback,
                    "selected_practical_harm": bool(
                        accepted
                        and selected > fallback + float(row["practical_harm_margin_mm"])
                    ),
                    "exact_fallback": bool(accepted or selected == fallback),
                }
            )
    if len(output) != len(rows):
        raise ValueError("Cross-material policy lost or duplicated query cases")
    return output


def _policy_summary(
    policy_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    materials = protocol["materials"]
    material_rows = []
    for material in materials:
        subset = [row for row in policy_rows if row["material"] == material]
        material_rows.append(
            {
                "material": material,
                "selected_minus_fallback_mm": float(
                    np.mean([row["selected_minus_fallback_mm"] for row in subset])
                ),
                "coverage": float(np.mean([row["accepted"] for row in subset])),
            }
        )
    material_regrets = np.asarray(
        [row["selected_minus_fallback_mm"] for row in material_rows]
    )
    rng = np.random.default_rng(
        int(protocol["bootstrap_seed"]) + POLICIES.index(policy_rows[0]["policy"])
    )
    draws = rng.integers(
        0,
        len(materials),
        size=(int(protocol["bootstrap_repetitions"]), len(materials)),
    )
    bootstrap = material_regrets[draws].mean(axis=1)
    accepted = [row for row in policy_rows if row["accepted"]]
    return {
        "policy": policy_rows[0]["policy"],
        "query_cases": len(policy_rows),
        "selected_coverage": float(np.mean([row["accepted"] for row in policy_rows])),
        "selected_minus_fallback_mm": float(
            np.mean([row["selected_minus_fallback_mm"] for row in policy_rows])
        ),
        "equal_material_selected_minus_fallback_mm": float(material_regrets.mean()),
        "material_bootstrap_95_interval_mm": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "heldout_materials_nonpositive": int(np.sum(material_regrets <= 0)),
        "heldout_materials_negative": int(np.sum(material_regrets < 0)),
        "accepted_query_cases": len(accepted),
        "accepted_strict_regression_fraction": (
            float(np.mean([row["strict_regression"] for row in accepted]))
            if accepted
            else 0.0
        ),
        "accepted_practical_harm_fraction": (
            float(np.mean([row["selected_practical_harm"] for row in accepted]))
            if accepted
            else 0.0
        ),
        "exact_fallback_violations": int(
            np.sum([not row["exact_fallback"] for row in policy_rows])
        ),
        "material_results": material_rows,
    }


def summarize(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_policy_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for policy in POLICIES:
        policy_rows = cross_material_policy_rows(rows, policy, protocol)
        all_policy_rows.extend(policy_rows)
        summaries[policy] = _policy_summary(policy_rows, protocol)

    primary = summaries["query_horizon_gate"]
    always = summaries["always_candidate"]
    required = protocol["positive_feasibility_gate"]
    criteria = {
        "minimum_selected_coverage": (
            primary["selected_coverage"] >= float(required["minimum_selected_coverage"])
        ),
        "negative_equal_material_regret": (
            primary["equal_material_selected_minus_fallback_mm"] < 0
        ),
        "negative_material_bootstrap_upper_95": (
            primary["material_bootstrap_95_interval_mm"][1] < 0
        ),
        "all_heldout_material_mean_regrets_nonpositive": (
            primary["heldout_materials_nonpositive"] == len(protocol["materials"])
        ),
        "zero_exact_fallback_violations": (primary["exact_fallback_violations"] == 0),
        "lower_practical_harm_rate_than_always_candidate": (
            primary["accepted_practical_harm_fraction"]
            < always["accepted_practical_harm_fraction"]
        ),
    }
    decision = (
        "retrospective-positive-feasibility"
        if all(criteria.values())
        else "retrospective-mixed-or-negative"
    )

    context_table = []
    primary_rows = [
        row for row in all_policy_rows if row["policy"] == "query_horizon_gate"
    ]
    keys = sorted(
        {
            (row["motion"], row["query"], float(row["horizon_seconds"]))
            for row in primary_rows
        }
    )
    for motion, query, horizon in keys:
        subset = [
            row
            for row in primary_rows
            if row["motion"] == motion
            and row["query"] == query
            and float(row["horizon_seconds"]) == horizon
        ]
        context_table.append(
            {
                "motion": motion,
                "query": query,
                "horizon_seconds": horizon,
                "selected_coverage": float(
                    np.mean([row["accepted"] for row in subset])
                ),
                "selected_minus_fallback_mm": float(
                    np.mean([row["selected_minus_fallback_mm"] for row in subset])
                ),
                "always_candidate_minus_fallback_mm": float(
                    np.mean([row["candidate_minus_fallback_mm"] for row in subset])
                ),
                "accepted_practical_harm_fraction": (
                    float(
                        np.mean(
                            [
                                row["selected_practical_harm"]
                                for row in subset
                                if row["accepted"]
                            ]
                        )
                    )
                    if any(row["accepted"] for row in subset)
                    else 0.0
                ),
            }
        )
    result = {
        "schema": "bayesian-phystwin.tracking-cloth-selective-twin-result.v1",
        "schema_version": 1,
        "decision": decision,
        "criteria": criteria,
        "primary_policy": "query_horizon_gate",
        "policy_summaries": summaries,
        "context_results": context_table,
        "claim_boundary": protocol["claim_boundary"],
        "information_boundary": protocol["information_boundary"],
    }
    result["result_id"] = object_digest(result)
    return all_policy_rows, result


def _report(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    primary = result["policy_summaries"]["query_horizon_gate"]
    motion = result["policy_summaries"]["motion_gate"]
    always = result["policy_summaries"]["always_candidate"]
    fallback = result["policy_summaries"]["always_fallback"]
    lines = [
        "# Selective digital twin on public real cloth",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "This is a retrospective, leave-one-material-out feasibility screen.",
        "The 32 twisting outcomes were previously opened in workflow 33302686759.",
        "No fresh-confirmation or deployment-safety claim is authorized.",
        "",
        "## Main comparison",
        "",
        "| Policy | Coverage | Selected - fallback [mm] | "
        "Practical harm among accepted | Material-bootstrap 95% interval [mm] |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in (fallback, always, motion, primary):
        interval = summary["material_bootstrap_95_interval_mm"]
        lines.append(
            f"| {summary['policy']} | {100 * summary['selected_coverage']:.2f}% "
            f"| {summary['equal_material_selected_minus_fallback_mm']:.4f} "
            f"| {100 * summary['accepted_practical_harm_fraction']:.2f}% "
            f"| [{interval[0]:.4f}, {interval[1]:.4f}] |"
        )
    lines.extend(
        [
            "",
            "Negative regret favors the selected policy. Rejected cases use the",
            "persistence loss exactly. The independent outer unit is material;",
            "query/horizon rows are repeated task views, not independent specimens.",
            "",
            "## Registered criteria",
            "",
        ]
    )
    for name, passed in result["criteria"].items():
        lines.append(f"- `{name}`: **{'pass' if passed else 'fail'}**")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            result["claim_boundary"],
            "",
            f"Scored query cases: {len(rows)}.",
        ]
    )
    return "\n".join(lines) + "\n"


def execute(dataset_root: Path, output: Path, protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(
        (BASE_HERE / "protocol.json").read_text(encoding="utf-8")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "base_protocol.json", base_protocol)

    cases, inventory = audit_dataset(dataset_root, base_protocol)
    write_json(output / "dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(
        inventory["included_license_text"], encoding="utf-8"
    )
    source_cases = [case for case in cases if case.motion == "shake"]
    scales = [
        infer_source_scale(
            case,
            read_prefix(case, base_protocol["prefix_seconds"])[1],
        )
        for case in source_cases
    ]
    if len(set(scales)) != 1:
        raise ValueError("Source recordings disagree about coordinate units")
    scale = scales[0]

    source_records, weights = _source_predictions(cases, base_protocol, scale)
    twist_records = _twist_predictions(cases, base_protocol, scale, weights)
    score_rows = score_records(source_records + twist_records, protocol)
    policy_rows, result = summarize(score_rows, protocol)

    _write_csv(output / "query_cases.csv", score_rows)
    _write_csv(output / "policy_cases.csv", policy_rows)
    write_json(output / "result.json", result)
    (output / "report.md").write_text(_report(result, score_rows), encoding="utf-8")
    manifest = {
        "schema": "bayesian-phystwin.tracking-cloth-selective-twin-run.v1",
        "schema_version": 1,
        "created_at": now(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "implementation_sha256": _implementation(),
        "protocol_id": object_digest(protocol),
        "base_protocol_id": object_digest(base_protocol),
        "inventory_id": inventory["inventory_id"],
        "result_id": result["result_id"],
        "raw_trajectory_upload": False,
        "paper_claim_authorized": False,
    }
    write_json(output / "run_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "protocol.json",
    )
    args = parser.parse_args()
    try:
        execute(args.dataset_root, args.output, args.protocol)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "failure.json",
            {
                "schema": "bayesian-phystwin.tracking-cloth-selective-twin-failure.v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_conclusion": None,
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
