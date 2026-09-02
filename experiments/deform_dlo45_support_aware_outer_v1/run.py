"""Retrospective support-aware outer gate for DEFORM DLO4/DLO5.

The inner finite-action certificate is exact only on its represented finite
support. This experiment asks whether support misspecification can be detected
from pre-outcome diagnostics of that certificate. Model fitting and threshold
selection use only leave-one-complete-trajectory-out source reconstructions.
Held outcomes are used only for descriptive evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.deform_dlo45_decision_identifiability_v1._common import (
    ATOL,
    DLOS,
    Model,
    Protocol,
    load_protocol,
    window_starts,
)
from experiments.deform_dlo45_decision_identifiability_v1._model import fit_model
from experiments.deform_dlo45_decision_identifiability_v1.gate_audit import diagnose

CONTRACT = "deform-dlo45-support-aware-outer-certificate-v1"
RESULT_SCHEMA = "bayesian-phystwin/support-aware-outer-certificate-prototype-v1"
SOURCE_SETTINGS = {
    "cluster_count": 16,
    "neighbors": 16,
    "temperature_scale": 1.0,
    "regret_tolerance": 0.05,
}
SCORE_NAMES = (
    "quotient_concentration",
    "maximum_quotient_mass",
    "maximum_kernel_weight",
    "expected_fallback_advantage",
    "expected_action_gap",
    "hypothesis_action_agreement",
    "negative_residual_disagreement",
    "negative_unsupported_specificity",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_outer_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("contract") != CONTRACT or value.get("schema_version") != 1:
        raise ValueError("unsupported support-aware outer protocol")
    source = value.get("source_reconstruction")
    model = value.get("outer_model")
    operating = value.get("source_operating_point")
    held = value.get("held_evaluation")
    if not all(isinstance(item, dict) for item in (source, model, operating, held)):
        raise ValueError("support-aware protocol sections must be JSON objects")
    assert isinstance(source, dict)
    assert isinstance(model, dict)
    assert isinstance(operating, dict)
    assert isinstance(held, dict)
    if (
        source.get("official_training_trajectory_count") != 112
        or source.get("windows_per_trajectory") != 19
        or source.get("parent_settings") != SOURCE_SETTINGS
        or model.get("held_outcome_features_forbidden") is not True
        or held.get("target_outcomes_used_for_model_selection") is not False
        or held.get("target_outcomes_used_for_threshold_selection") is not False
    ):
        raise ValueError("support-aware information boundary changed")
    return value


def source_record(
    *,
    stable_id: str,
    dlo: str,
    trajectory: str,
    current_frame: int,
    feature: np.ndarray,
    actual_residual: np.ndarray,
    model: Model,
    protocol: Protocol,
) -> dict[str, object]:
    diagnostic = diagnose(feature, model, protocol)
    decision = diagnostic.decision
    certificate_action = decision.certificate_action
    candidate_action = decision.jeffrey_action
    actions = model.action_scales[:, None] * decision.correction[None, :]
    normalized_mse = np.mean(
        np.square(actual_residual[None, :] - actions),
        axis=1,
    )
    best = float(np.min(normalized_mse))
    denominator = max(float(normalized_mse[0]), model.loss_floor)
    normalized_regret = (normalized_mse - best) / denominator
    source_bound = float(decision.worst_case_regret[certificate_action])
    realized_regret = float(normalized_regret[certificate_action])
    return {
        "stable_id": stable_id,
        "dlo": dlo,
        "trajectory": trajectory,
        "current_frame": current_frame,
        "certificate_action": certificate_action,
        "candidate_action": candidate_action,
        "certificate_source_regret_bound": source_bound,
        "certificate_realized_regret": realized_regret,
        "certificate_regret_excess": realized_regret - source_bound,
        "fallback_realized_regret": float(normalized_regret[0]),
        "certificate_harmful_vs_fallback": bool(
            normalized_mse[certificate_action] > normalized_mse[0] + ATOL
        ),
        "registered_worst_case_regret_by_action": (
            decision.worst_case_regret.tolist()
        ),
        "scores": {name: float(diagnostic.scores[name]) for name in SCORE_NAMES},
    }


def reconstruct_source_panel(
    source_dir: Path,
    parent_protocol: Protocol,
) -> list[dict[str, object]]:
    source_result = read_json(source_dir / "source_result.json")
    starts = window_starts(parent_protocol)
    if len(starts) != 19:
        raise ValueError("parent window roster changed")
    rows: list[dict[str, object]] = []
    with np.load(source_dir / "source_model.npz", allow_pickle=False) as archive:
        for dlo in DLOS:
            prefix = dlo.lower()
            features = np.asarray(archive[f"{prefix}_features"], dtype=np.float64)
            residuals = np.asarray(archive[f"{prefix}_residuals"], dtype=np.float64)
            names = sorted(source_result["train_manifest"][dlo].keys())
            if len(names) != 56 or len(features) != 56 * len(starts):
                raise ValueError(f"{dlo}: parent source roster changed")
            for trajectory_index, trajectory in enumerate(names):
                first = trajectory_index * len(starts)
                stop = first + len(starts)
                keep = np.ones(len(features), dtype=bool)
                keep[first:stop] = False
                model = fit_model(
                    features[keep],
                    residuals[keep],
                    cluster_count=SOURCE_SETTINGS["cluster_count"],
                    neighbors=SOURCE_SETTINGS["neighbors"],
                    temperature_scale=SOURCE_SETTINGS["temperature_scale"],
                    regret_tolerance=SOURCE_SETTINGS["regret_tolerance"],
                    protocol=parent_protocol,
                )
                for local_index, current_frame in enumerate(starts):
                    row_index = first + local_index
                    rows.append(
                        source_record(
                            stable_id=f"{dlo}/{trajectory}/{current_frame}",
                            dlo=dlo,
                            trajectory=trajectory,
                            current_frame=current_frame,
                            feature=features[row_index],
                            actual_residual=residuals[row_index],
                            model=model,
                            protocol=parent_protocol,
                        )
                    )
    if len(rows) != 2128:
        raise ValueError(f"expected 2128 source rows, got {len(rows)}")
    return rows


def load_held_panel(target_dir: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in (target_dir / "per_decision.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    target_audit = read_json(target_dir / "target_audit.json")
    for dlo in DLOS:
        indices = [index for index, row in enumerate(rows) if row["dlo"] == dlo]
        fallback = target_audit["dlos"][dlo]["point_diagnostics"]["fallback"][
            "regret_values"
        ]
        if len(indices) != len(fallback):
            raise ValueError(f"{dlo}: held fallback roster changed")
        for index, value in zip(indices, fallback, strict=True):
            rows[index]["fallback_realized_regret"] = float(value)
    if len(rows) != 532:
        raise ValueError(f"expected 532 held rows, got {len(rows)}")
    return rows


def feature_matrix(rows: Sequence[dict[str, object]]) -> np.ndarray:
    result: list[list[float]] = []
    for row in rows:
        scores = row["scores"]
        if not isinstance(scores, dict):
            raise ValueError("missing pre-outcome score dictionary")
        worst = row["registered_worst_case_regret_by_action"]
        if not isinstance(worst, list) or len(worst) != 3:
            raise ValueError("registered regret vector changed")
        values = [float(row["certificate_source_regret_bound"])]
        values.extend(float(scores[name]) for name in SCORE_NAMES)
        values.extend(
            (
                float(row["current_frame"]) / 454.0,
                float(row["dlo"] == "DLO5"),
                float(int(row["certificate_action"]) == 2),
                float(int(row["candidate_action"]) == 2),
            )
        )
        values.extend(float(value) for value in worst)
        result.append(values)
    return np.asarray(result, dtype=np.float64)


def source_groups(rows: Sequence[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [f"{row['dlo']}/{row['trajectory']}" for row in rows],
        dtype=object,
    )


def support_violation_labels(rows: Sequence[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [float(row["certificate_regret_excess"]) > 0.0 for row in rows],
        dtype=np.int64,
    )


def select_source_threshold(
    probabilities: np.ndarray,
    violations: np.ndarray,
    groups: Sequence[str],
    *,
    risk_cap: float,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    unique_groups = sorted(set(groups))
    group_index = {name: index for index, name in enumerate(unique_groups)}
    row_group = np.asarray([group_index[name] for name in groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    multiplicity = np.empty((repetitions, len(unique_groups)), dtype=np.int16)
    for repetition in range(repetitions):
        draw = rng.integers(0, len(unique_groups), len(unique_groups))
        multiplicity[repetition] = np.bincount(draw, minlength=len(unique_groups))

    order = np.argsort(probabilities, kind="stable")
    selected_by_group = np.zeros(len(unique_groups), dtype=np.int64)
    violations_by_group = np.zeros(len(unique_groups), dtype=np.int64)
    best: dict[str, float | int] | None = None
    position = 0
    while position < len(order):
        threshold = float(probabilities[order[position]])
        end = position
        while end < len(order) and probabilities[order[end]] == threshold:
            row = int(order[end])
            group = int(row_group[row])
            selected_by_group[group] += 1
            violations_by_group[group] += int(violations[row])
            end += 1
        selected = int(np.sum(selected_by_group))
        if selected >= 10:
            bootstrap_selected = multiplicity @ selected_by_group
            bootstrap_violations = multiplicity @ violations_by_group
            ratio = np.divide(
                bootstrap_violations,
                bootstrap_selected,
                out=np.zeros(repetitions, dtype=np.float64),
                where=bootstrap_selected > 0,
            )
            upper = float(np.quantile(ratio, 0.95))
            if upper <= risk_cap:
                mask = probabilities <= threshold
                best = {
                    "threshold": threshold,
                    "source_selected_count": selected,
                    "source_selected_fraction": selected / len(probabilities),
                    "source_empirical_violation_fraction": float(
                        np.mean(violations[mask])
                    ),
                    "source_block_bootstrap_upper_095": upper,
                    "source_block_bootstrap_upper_0975": float(
                        np.quantile(ratio, 0.975)
                    ),
                }
        position = end
    if best is None:
        raise RuntimeError("no source operating point satisfies the risk cap")
    return best


def policy_metrics(
    rows: Sequence[dict[str, object]],
    selected: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, object]:
    if len(rows) != len(selected):
        raise ValueError("policy roster length mismatch")
    selected_indices = np.flatnonzero(selected)
    source_excess = np.asarray(
        [float(row["certificate_regret_excess"]) for row in rows],
        dtype=np.float64,
    )
    realized_regret = np.asarray(
        [float(row["certificate_realized_regret"]) for row in rows],
        dtype=np.float64,
    )
    fallback_regret = np.asarray(
        [float(row["fallback_realized_regret"]) for row in rows],
        dtype=np.float64,
    )
    harmful = np.asarray(
        [bool(row["certificate_harmful_vs_fallback"]) for row in rows],
        dtype=bool,
    )
    policy_regret = np.where(selected, realized_regret, fallback_regret)
    count = len(selected_indices)
    return {
        "decision_count": len(rows),
        "selected_count": count,
        "overall_coverage": count / len(rows),
        "support_bound_violation_count": int(
            np.count_nonzero(source_excess[selected_indices] > 0.0)
        ),
        "support_bound_violation_fraction_selected": (
            float(np.mean(source_excess[selected_indices] > 0.0)) if count else 0.0
        ),
        "regret_tolerance_violation_count": int(
            np.count_nonzero(realized_regret[selected_indices] > tolerance)
        ),
        "regret_tolerance_violation_fraction_selected": (
            float(np.mean(realized_regret[selected_indices] > tolerance))
            if count
            else 0.0
        ),
        "harmful_count": int(np.count_nonzero(harmful[selected_indices])),
        "harmful_fraction_selected": (
            float(np.mean(harmful[selected_indices])) if count else 0.0
        ),
        "mean_normalized_regret": float(np.mean(policy_regret)),
        "fallback_mean_normalized_regret": float(np.mean(fallback_regret)),
        "normalized_regret_reduction_vs_fallback": float(
            1.0 - np.mean(policy_regret) / np.mean(fallback_regret)
        ),
    }


def trajectory_maximum_conformal_inflation(
    source_rows: Sequence[dict[str, object]],
    *,
    coverage: float,
) -> float:
    grouped: dict[str, list[float]] = {}
    for row in source_rows:
        group = f"{row['dlo']}/{row['trajectory']}"
        grouped.setdefault(group, []).append(
            max(float(row["certificate_regret_excess"]), 0.0)
        )
    maxima = np.asarray([max(values) for values in grouped.values()], dtype=np.float64)
    rank = min(len(maxima), math.ceil((len(maxima) + 1) * coverage))
    return float(np.sort(maxima)[rank - 1])


def run(
    *,
    source_dir: Path,
    target_dir: Path,
    output: Path,
    outer_protocol_path: Path,
    parent_protocol_path: Path,
) -> dict[str, object]:
    outer = load_outer_protocol(outer_protocol_path)
    parent = load_protocol(parent_protocol_path)
    source_rows = reconstruct_source_panel(source_dir, parent)
    held_rows = load_held_panel(target_dir)
    source_nonfallback = [row for row in source_rows if int(row["certificate_action"]) != 0]
    held_nonfallback = [row for row in held_rows if int(row["certificate_action"]) != 0]
    if len(source_nonfallback) != 332 or len(held_nonfallback) != 82:
        raise ValueError("inner certificate nonfallback roster changed")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    except ImportError as error:  # pragma: no cover - runtime dependency boundary
        raise RuntimeError("scikit-learn is required for this experiment") from error

    x_source = feature_matrix(source_nonfallback)
    x_held = feature_matrix(held_nonfallback)
    y_source = support_violation_labels(source_nonfallback)
    y_held = support_violation_labels(held_nonfallback)
    groups = source_groups(source_nonfallback)
    model_config = outer["outer_model"]
    operating = outer["source_operating_point"]
    estimator = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(
            C=float(model_config["logistic_c"]),
            max_iter=20000,
            class_weight=str(model_config["class_weight"]),
            random_state=0,
        ),
    )
    cv = GroupKFold(n_splits=int(model_config["grouped_cross_validation_folds"]))
    source_probability = cross_val_predict(
        estimator,
        x_source,
        y_source,
        groups=groups,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    estimator.fit(x_source, y_source)
    held_probability = estimator.predict_proba(x_held)[:, 1]
    source_auc = float(roc_auc_score(y_source, source_probability))
    held_auc = float(roc_auc_score(y_held, held_probability))
    threshold = select_source_threshold(
        source_probability,
        y_source,
        groups.astype(str).tolist(),
        risk_cap=float(operating["risk_cap"]),
        repetitions=int(operating["bootstrap_repetitions"]),
        seed=int(operating["bootstrap_seed"]),
    )

    inner_selected = np.asarray(
        [int(row["certificate_action"]) != 0 for row in held_rows],
        dtype=bool,
    )
    outer_selected = np.zeros(len(held_rows), dtype=bool)
    held_indices = np.flatnonzero(inner_selected)
    outer_selected[held_indices] = held_probability <= float(threshold["threshold"])
    inner_metrics = policy_metrics(
        held_rows,
        inner_selected,
        tolerance=SOURCE_SETTINGS["regret_tolerance"],
    )
    outer_metrics = policy_metrics(
        held_rows,
        outer_selected,
        tolerance=SOURCE_SETTINGS["regret_tolerance"],
    )
    outer_metrics["fraction_of_inner_nonfallback_retained"] = (
        int(outer_metrics["selected_count"]) / int(inner_metrics["selected_count"])
    )

    by_dlo: dict[str, object] = {}
    for dlo in DLOS:
        mask = np.asarray([row["dlo"] == dlo for row in held_rows], dtype=bool)
        local_rows = [row for row, keep in zip(held_rows, mask, strict=True) if keep]
        by_dlo[dlo] = policy_metrics(
            local_rows,
            outer_selected[mask],
            tolerance=SOURCE_SETTINGS["regret_tolerance"],
        )

    conformal = trajectory_maximum_conformal_inflation(
        source_nonfallback,
        coverage=float(outer["negative_control"]["coverage"]),
    )
    held_bounds = np.asarray(
        [float(row["certificate_source_regret_bound"]) for row in held_nonfallback],
        dtype=np.float64,
    )
    conformal_accepted = int(
        np.count_nonzero(
            held_bounds + conformal <= SOURCE_SETTINGS["regret_tolerance"]
        )
    )
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "retrospective-prototype-complete",
        "source": {
            "complete_trajectory_count": 112,
            "decision_count": len(source_rows),
            "inner_nonfallback_count": len(source_nonfallback),
            "grouped_cross_validated_auc": source_auc,
            **threshold,
        },
        "held_discrimination_auc_descriptive_only": held_auc,
        "held": {
            "inner_finite_support_certificate": inner_metrics,
            "outer_support_gate_plus_inner_certificate": outer_metrics,
            "outer_by_dlo": by_dlo,
        },
        "strict_static_conformal_negative_control": {
            "coverage": float(outer["negative_control"]["coverage"]),
            "regret_inflation": conformal,
            "held_inner_nonfallback_count": len(held_nonfallback),
            "held_accepted_count": conformal_accepted,
        },
        "claim_boundary": outer["claim_boundary"],
    }
    result["result_id"] = canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "result.json", result)
    report = (
        "# DEFORM support-aware outer certificate\n\n"
        f"Source grouped-CV AUC: **{source_auc:.4f}**.  \n"
        f"Held descriptive AUC: **{held_auc:.4f}**.  \n\n"
        "| Policy | Selected | Support-bound violation | Tolerance violation | "
        "Harm | Regret gain |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| Inner | {inner_metrics['selected_count']} | "
        f"{100*float(inner_metrics['support_bound_violation_fraction_selected']):.2f}% | "
        f"{100*float(inner_metrics['regret_tolerance_violation_fraction_selected']):.2f}% | "
        f"{100*float(inner_metrics['harmful_fraction_selected']):.2f}% | "
        f"{100*float(inner_metrics['normalized_regret_reduction_vs_fallback']):.2f}% |\n"
        f"| Outer + inner | {outer_metrics['selected_count']} | "
        f"{100*float(outer_metrics['support_bound_violation_fraction_selected']):.2f}% | "
        f"{100*float(outer_metrics['regret_tolerance_violation_fraction_selected']):.2f}% | "
        f"{100*float(outer_metrics['harmful_fraction_selected']):.2f}% | "
        f"{100*float(outer_metrics['normalized_regret_reduction_vs_fallback']):.2f}% |\n\n"
        f"Static 90% trajectory-maximum conformal inflation: **{conformal:.4f}**, "
        f"held accepted **{conformal_accepted}/{len(held_nonfallback)}**.\n\n"
        f"Result ID: `{result['result_id']}`\n\n{outer['claim_boundary']}\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=here / "protocol.json")
    parser.add_argument(
        "--parent-protocol",
        type=Path,
        default=(
            here.parent
            / "deform_dlo45_decision_identifiability_v1"
            / "protocol.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        output=args.output,
        outer_protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
