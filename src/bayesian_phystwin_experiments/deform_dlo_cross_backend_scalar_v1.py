"""Cross-validated scalar transport for a fixed cross-backend residual field.

This diagnostic is a deliberately weaker fallback to exact no-refit coefficient
transfer.  The high-dimensional DEFORM residual field remains unchanged.  For
each held-out trajectory, one non-negative scalar amplitude is estimated from
the other complete trajectories and then applied to the held-out trajectory.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.deform_dlo_cross_backend_transfer_v1 import (
    paired_point_summary,
)

SCHEMA_VERSION = 1
CONTRACT = "deform-dlo3-cross-backend-scalar-transport-v1"
RESULT_CONTRACT = "deform-dlo3-cross-backend-scalar-transport-result-v1"


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_array(value: object, *, ndim: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {ndim}-D array")
    return array


def load_cross_backend_scalar_protocol(path: str | Path) -> dict[str, object]:
    """Load the frozen one-scalar cross-backend transport protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scalar transport protocol must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported scalar transport protocol schema")
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported scalar transport protocol contract")

    parent = _mapping(payload.get("parent"), label="parent")
    if (
        parent.get("protocol") != "protocols/deform_dlo3_cross_backend_transfer_v1.json"
        or parent.get("direct_arm") != "equal-seed-no-refit-transfer"
        or parent.get("direct_shrinkage") != 0.25
        or parent.get("seed_models") != [42, 43, 44]
    ):
        raise ValueError("scalar transport parent changed")

    panel = _mapping(payload.get("source_panel"), label="source panel")
    if (
        panel.get("dataset") != "DEFORM-DLO3"
        or panel.get("partition") != "train/source_test"
        or int(cast(Any, panel.get("trajectory_count", -1))) != 8
        or int(cast(Any, panel.get("frame_count", -1))) != 500
        or int(cast(Any, panel.get("node_count", -1))) != 12
        or panel.get("source_outcomes_previously_opened") is not True
        or panel.get("official_evaluation_read") is not False
    ):
        raise ValueError("scalar transport source panel changed")

    transport = _mapping(payload.get("transport"), label="transport")
    if (
        transport.get("operator")
        != "leave-one-complete-trajectory-out-pooled-least-squares-scalar-v1"
        or transport.get("intercept") is not False
        or transport.get("correction_field_refit") is not False
        or transport.get("fold_specific_high_dimensional_parameters") is not False
        or float(cast(Any, transport.get("minimum_scalar", math.nan))) != 0.0
        or float(cast(Any, transport.get("maximum_scalar", math.nan))) != 4.0
        or transport.get("selection_metric")
        != "coordinate-l2-on-seven-training-trajectories"
        or transport.get("evaluation_metric") != "mean-coordinate-l1-m-all-nodes"
    ):
        raise ValueError("scalar transport operator changed")

    gate = _mapping(payload.get("promotion_gate"), label="promotion gate")
    if (
        float(cast(Any, gate.get("minimum_relative_improvement", math.nan))) != 0.01
        or int(cast(Any, gate.get("minimum_case_wins", -1))) != 6
        or float(cast(Any, gate.get("maximum_case_ratio", math.nan))) != 1.10
        or int(cast(Any, gate.get("minimum_positive_alignment_cases", -1))) != 6
        or float(cast(Any, gate.get("minimum_median_alignment", math.nan))) != 0.05
        or int(cast(Any, gate.get("minimum_positive_fold_scalars", -1))) != 6
    ):
        raise ValueError("scalar transport promotion gate changed")

    evaluation = _mapping(payload.get("evaluation"), label="evaluation")
    if (
        int(cast(Any, evaluation.get("bootstrap_repetitions", -1))) != 10000
        or int(cast(Any, evaluation.get("bootstrap_seed", -1))) != 20260901
        or evaluation.get("bootstrap_interpretation")
        != "descriptive-fixed-fold-predictions-over-complete-trajectories"
    ):
        raise ValueError("scalar transport evaluation changed")

    boundary = _mapping(payload.get("information_boundary"), label="boundary")
    if (
        boundary.get("source_test_labels_used_in_other_folds") is not True
        or boundary.get("same_trajectory_label_used_for_its_scalar") is not False
        or boundary.get("pyelastica_high_dimensional_refit") is not False
        or boundary.get("deform_refit") is not False
        or boundary.get("dlo3_official_evaluation_read") is not False
        or boundary.get("dlo4_or_dlo5_read") is not False
        or boundary.get("held_v8_access") is not False
        or boundary.get("paper_claim_authorized") is not False
    ):
        raise ValueError("scalar transport information boundary changed")

    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def leave_one_trajectory_out_scalar_transport(
    direct_prediction: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    *,
    minimum_scalar: float,
    maximum_scalar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one scalar on all other trajectories and predict each held-out one."""

    direct = _finite_array(direct_prediction, ndim=4, label="direct prediction")
    base = _finite_array(baseline, ndim=4, label="baseline")
    target = _finite_array(truth, ndim=4, label="truth")
    if direct.shape != base.shape or direct.shape != target.shape:
        raise ValueError("scalar transport arrays differ in shape")
    if direct.shape[0] < 3:
        raise ValueError("scalar transport requires at least three trajectories")
    if (
        not math.isfinite(minimum_scalar)
        or not math.isfinite(maximum_scalar)
        or minimum_scalar < 0.0
        or maximum_scalar <= minimum_scalar
    ):
        raise ValueError("scalar transport bounds are invalid")

    correction = direct - base
    residual = target - base
    predictions = np.empty_like(base)
    scalars = np.empty(direct.shape[0], dtype=np.float64)
    for held_out in range(direct.shape[0]):
        train = np.arange(direct.shape[0]) != held_out
        train_correction = correction[train]
        train_residual = residual[train]
        denominator = float(np.sum(train_correction * train_correction))
        numerator = float(np.sum(train_correction * train_residual))
        unconstrained = numerator / denominator if denominator > 0.0 else 0.0
        scalar = float(np.clip(unconstrained, minimum_scalar, maximum_scalar))
        scalars[held_out] = scalar
        predictions[held_out] = base[held_out] + scalar * correction[held_out]
    return predictions, scalars


def trajectory_alignment(
    direct_prediction: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Return per-trajectory cosine alignment of transferred and true residuals."""

    direct = _finite_array(direct_prediction, ndim=4, label="direct prediction")
    base = _finite_array(baseline, ndim=4, label="baseline")
    target = _finite_array(truth, ndim=4, label="truth")
    if direct.shape != base.shape or direct.shape != target.shape:
        raise ValueError("alignment arrays differ in shape")
    correction = (direct - base).reshape(direct.shape[0], -1)
    residual = (target - base).reshape(direct.shape[0], -1)
    numerator = np.sum(correction * residual, axis=1)
    denominator = np.linalg.norm(correction, axis=1) * np.linalg.norm(residual, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _point_gate(
    summary: Mapping[str, object],
    gate: Mapping[str, object],
) -> bool:
    return (
        float(cast(Any, summary["relative_improvement"]))
        >= float(cast(Any, gate["minimum_relative_improvement"]))
        and int(cast(Any, summary["wins"])) >= int(cast(Any, gate["minimum_case_wins"]))
        and float(cast(Any, summary["maximum_case_ratio"]))
        <= float(cast(Any, gate["maximum_case_ratio"]))
    )


def evaluate_cross_backend_scalar_transport(
    *,
    names: Sequence[str],
    truth: np.ndarray,
    baseline: np.ndarray,
    direct_prediction: np.ndarray,
    pyelastica_specific_candidate: np.ndarray,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate exact transfer and its frozen one-scalar fallback."""

    if len(names) != 8 or len(set(names)) != len(names):
        raise ValueError("scalar transport requires eight unique trajectory names")
    target = _finite_array(truth, ndim=4, label="truth")
    base = _finite_array(baseline, ndim=4, label="baseline")
    direct = _finite_array(direct_prediction, ndim=4, label="direct prediction")
    specific = _finite_array(
        pyelastica_specific_candidate,
        ndim=4,
        label="PyElastica-specific candidate",
    )
    if len({target.shape, base.shape, direct.shape, specific.shape}) != 1:
        raise ValueError("scalar transport evaluation arrays differ in shape")
    if target.shape[0] != len(names):
        raise ValueError("trajectory names and arrays do not align")

    transport = _mapping(protocol.get("transport"), label="transport")
    scalar_prediction, scalars = leave_one_trajectory_out_scalar_transport(
        direct,
        base,
        target,
        minimum_scalar=float(cast(Any, transport["minimum_scalar"])),
        maximum_scalar=float(cast(Any, transport["maximum_scalar"])),
    )
    evaluation = _mapping(protocol.get("evaluation"), label="evaluation")
    repetitions = int(cast(Any, evaluation["bootstrap_repetitions"]))
    seed = int(cast(Any, evaluation["bootstrap_seed"]))
    direct_summary = paired_point_summary(
        direct,
        base,
        target,
        names,
        repetitions=repetitions,
        seed=seed,
    )
    scalar_summary = paired_point_summary(
        scalar_prediction,
        base,
        target,
        names,
        repetitions=repetitions,
        seed=seed + 1,
    )
    specific_summary = paired_point_summary(
        specific,
        base,
        target,
        names,
        repetitions=repetitions,
        seed=seed + 2,
    )
    scalar_vs_specific = paired_point_summary(
        scalar_prediction,
        specific,
        target,
        names,
        repetitions=repetitions,
        seed=seed + 3,
    )

    alignment = trajectory_alignment(direct, base, target)
    gate = _mapping(protocol.get("promotion_gate"), label="promotion gate")
    positive_alignment = int(np.sum(alignment > 0.0))
    median_alignment = float(np.median(alignment))
    positive_scalars = int(np.sum(scalars > 0.0))
    directional_passed = (
        positive_alignment >= int(cast(Any, gate["minimum_positive_alignment_cases"]))
        and median_alignment >= float(cast(Any, gate["minimum_median_alignment"]))
        and positive_scalars >= int(cast(Any, gate["minimum_positive_fold_scalars"]))
    )
    scalar_point_passed = _point_gate(scalar_summary, gate)
    direct_point_passed = _point_gate(direct_summary, gate)
    shared_geometry_supported = scalar_point_passed and directional_passed

    baseline_mean = float(cast(Any, scalar_summary["baseline_mean_l1_m"]))
    scalar_mean = float(cast(Any, scalar_summary["candidate_mean_l1_m"]))
    specific_mean = float(cast(Any, specific_summary["candidate_mean_l1_m"]))
    specific_gain = baseline_mean - specific_mean
    retained = (
        (baseline_mean - scalar_mean) / specific_gain if specific_gain > 0.0 else None
    )

    cases = []
    scalar_cases = cast(Sequence[Mapping[str, object]], scalar_summary["cases"])
    direct_cases = cast(Sequence[Mapping[str, object]], direct_summary["cases"])
    for index, name in enumerate(names):
        cases.append(
            {
                "name": str(name),
                "fold_scalar": float(scalars[index]),
                "alignment_cosine": float(alignment[index]),
                "baseline_l1_m": float(cast(Any, scalar_cases[index]["baseline_l1_m"])),
                "direct_l1_m": float(cast(Any, direct_cases[index]["candidate_l1_m"])),
                "scalar_l1_m": float(cast(Any, scalar_cases[index]["candidate_l1_m"])),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RESULT_CONTRACT,
        "decision": (
            "cross-backend-shared-residual-geometry-supported"
            if shared_geometry_supported
            else "cross-backend-shared-residual-geometry-not-supported"
        ),
        "claim_ladder": {
            "exact_no_refit_point_transfer_supported": direct_point_passed,
            "one_scalar_cross_validated_point_transfer_supported": scalar_point_passed,
            "directional_alignment_supported": directional_passed,
            "shared_residual_geometry_supported": shared_geometry_supported,
        },
        "methods": {
            "raw_pyelastica": baseline_mean,
            "direct_equal_seed_no_refit": float(
                cast(Any, direct_summary["candidate_mean_l1_m"])
            ),
            "leave_one_trajectory_out_one_scalar": scalar_mean,
            "pyelastica_specific_high_dimensional_refit": specific_mean,
        },
        "direct_vs_raw_pyelastica": direct_summary,
        "scalar_vs_raw_pyelastica": scalar_summary,
        "pyelastica_specific_vs_raw_pyelastica": specific_summary,
        "scalar_vs_pyelastica_specific": scalar_vs_specific,
        "directional_alignment": {
            "trajectory_cosines": [float(value) for value in alignment],
            "positive_cases": positive_alignment,
            "minimum_positive_cases": int(
                cast(Any, gate["minimum_positive_alignment_cases"])
            ),
            "median_cosine": median_alignment,
            "minimum_median_cosine": float(cast(Any, gate["minimum_median_alignment"])),
            "passed": directional_passed,
        },
        "fold_scalars": {
            "values": [float(value) for value in scalars],
            "positive_count": positive_scalars,
            "minimum_positive_count": int(
                cast(Any, gate["minimum_positive_fold_scalars"])
            ),
            "minimum": float(np.min(scalars)),
            "median": float(np.median(scalars)),
            "maximum": float(np.max(scalars)),
        },
        "promotion_gate": {
            "scalar_point_passed": scalar_point_passed,
            "directional_passed": directional_passed,
            "supported": shared_geometry_supported,
            "minimum_relative_improvement": float(
                cast(Any, gate["minimum_relative_improvement"])
            ),
            "minimum_case_wins": int(cast(Any, gate["minimum_case_wins"])),
            "maximum_case_ratio": float(cast(Any, gate["maximum_case_ratio"])),
        },
        "backend_specific_gain_retained_fraction": (
            None if retained is None else float(retained)
        ),
        "cases": cases,
        "information_boundary": {
            "retrospective_source_only": True,
            "source_test_labels_used_in_other_folds": True,
            "same_trajectory_label_used_for_its_scalar": False,
            "pyelastica_high_dimensional_refit": False,
            "deform_refit": False,
            "dlo3_official_evaluation_read": False,
            "dlo4_or_dlo5_read": False,
            "held_v8_access": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "A positive result supports a shared cross-backend residual direction "
            "whose amplitude can be recalibrated with one scalar fitted on other "
            "complete source trajectories. It is weaker than exact no-refit "
            "coefficient transfer and is not fresh target confirmation, arbitrary-"
            "backend transfer, zero-shot object generalization, safety, or state "
            "of the art."
        ),
    }
