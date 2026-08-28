"""Opt-in hard positional boundaries for a frozen DEFORM source screen."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np

from .deform_state_restart import RestartConfig, prediction_metrics

SCHEMA = "deform-kinematic-boundary-source-v1"
ARMS = ("incumbent", "paired", "hard_baseline", "hard_paired")
PRIMARY = "hard_paired"
CLAMPS = (0, 1, 10, 11)
HORIZONS = {"all": (0, 120), "early": (0, 40), "middle": (40, 80), "late": (80, 120)}


def config_for_source() -> RestartConfig:
    return RestartConfig(seed=260929)


def hard_position_projection(
    model: Any,
    updated_vertices: Any,
    m_restEdgeL: Any,
    m_pmass: Any,
    clamped_index: Any,
    iterative_times: int = 10,
    mode: str = "pytorch",
) -> Any:
    """Native edge order/algebra, except an edge with two fixed ends is skipped.

    Prescribed boundary segments may disagree with their rest lengths. This
    variant honors positions, not inextensibility of those prescribed segments.
    """
    if mode != "pytorch" or str(model.device) != "cpu":
        raise ValueError("only the registered CPU PyTorch projection is supported")
    if type(iterative_times) is not int or iterative_times < 1:
        raise ValueError("positive integer projection iterations required")
    batch, nodes, axes = updated_vertices.shape
    if (
        axes != 3
        or nodes != model.n_vert
        or m_restEdgeL.shape != (batch, nodes - 1)
        or m_pmass.shape not in ((1, nodes), (batch, nodes))
        or clamped_index.shape != (nodes,)
    ):
        raise ValueError("projection geometry or mass shapes differ")
    for value in (updated_vertices, m_restEdgeL, m_pmass, clamped_index):
        if value.device.type != "cpu" or not value.isfinite().all():
            raise ValueError("finite CPU projection inputs required")
    if (
        not (m_restEdgeL > 0).all()
        or not (m_pmass > 0).all()
        or not ((clamped_index == 0) | (clamped_index == 1)).all()
    ):
        raise ValueError(
            "positive rest lengths/masses and binary boundary mask required"
        )
    mask = clamped_index.bool()
    anchors = updated_vertices[:, mask].clone()
    zero = updated_vertices.new_zeros((1,))
    for _ in range(iterative_times):
        for i in range(nodes - 1):
            if bool(mask[i]) and bool(mask[i + 1]):
                continue
            edge = updated_vertices[:, i + 1] - updated_vertices[:, i]
            correction = 1 - 2 * m_restEdgeL[:, i] * m_restEdgeL[:, i] / (
                m_restEdgeL[:, i] * m_restEdgeL[:, i] + (edge * edge).sum(dim=1)
            )
            if bool(mask[i]):
                left, right = zero, -correction
            elif bool(mask[i + 1]):
                left, right = correction, zero
            else:
                mass = m_pmass[:, i] + m_pmass[:, i + 1]
                left = correction * m_pmass[:, i + 1] / mass
                right = -correction * m_pmass[:, i] / mass
            updated_vertices[:, i] = updated_vertices[:, i] + left.unsqueeze(1) * edge
            updated_vertices[:, i + 1] = (
                updated_vertices[:, i + 1] + right.unsqueeze(1) * edge
            )
    # Assigning an exact zero increment can change signed-zero bytes.
    updated_vertices[:, mask] = anchors
    return updated_vertices


def install_hard_position_projection(model: Any, *, enabled: bool = False) -> bool:
    """Change one model instance only; leave upstream files/classes untouched."""
    if type(enabled) is not bool:
        raise ValueError("explicit boolean boundary mode required")
    if not enabled:
        return False
    if str(model.device) != "cpu" or not callable(
        getattr(model, "applyInternalConstraintsIteration", None)
    ):
        raise ValueError("registered native CPU model required")
    if getattr(model, "_bpt_hard_position_projection", False):
        raise ValueError("hard boundary mode is already installed")
    model.applyInternalConstraintsIteration = MethodType(
        hard_position_projection, model
    )
    model._bpt_hard_position_projection = True
    return True


def hard_boundary_readout(
    incumbent: np.ndarray,
    archived_native: np.ndarray,
    hard_native: np.ndarray,
    *,
    enabled: bool = False,
) -> np.ndarray:
    if type(enabled) is not bool:
        raise ValueError("explicit boolean readout mode required")
    if not enabled:
        return incumbent
    shape = incumbent.shape
    if (
        len(shape) != 4
        or shape[-2:] != (12, 3)
        or archived_native.shape != shape
        or hard_native.shape != shape
        or not all(
            np.isfinite(x).all() for x in (incumbent, archived_native, hard_native)
        )
    ):
        raise ValueError("finite, aligned DLO2 prediction arrays required")
    offset = incumbent.astype(float) - archived_native.astype(float)
    if np.count_nonzero(offset[:, :, CLAMPS]):
        raise ValueError("the frozen readout cannot change boundary nodes")
    return hard_native.astype(float) + offset


def score_predictions(
    names: list[str], predictions: dict[str, np.ndarray], truth: np.ndarray
) -> dict[str, Any]:
    if (
        names != sorted(names)
        or len(names) != 14
        or len(set(names)) != 14
        or names.count("103.pkl") != 1
        or set(predictions) != set(ARMS)
        or truth.shape != (14, 120, 12, 3)
        or any(v.shape != truth.shape for v in predictions.values())
    ):
        raise ValueError("registered source roster/arm/time/identity contract required")
    cases: dict[str, Any] = {}
    for i, name in enumerate(names):
        if name == "103.pkl":
            continue
        cases[name] = {
            arm: {
                horizon: prediction_metrics(
                    value[i, start:end][:, (3, 5, 7, 9)],
                    truth[i, start:end][:, (3, 5, 7, 9)],
                )
                for horizon, (start, end) in HORIZONS.items()
            }
            for arm, value in predictions.items()
        }
    means = {
        arm: {
            horizon: {
                metric: float(
                    np.mean([c[arm][horizon][metric] for c in cases.values()])
                )
                for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm")
            }
            for horizon in HORIZONS
        }
        for arm in ARMS
    }
    delta = np.array(
        [
            c[PRIMARY]["all"]["point_rmse_mm"] - c["paired"]["all"]["point_rmse_mm"]
            for c in cases.values()
        ]
    )
    draws = np.random.default_rng(260929).integers(0, 13, (10000, 13))
    interval = np.asarray(np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]))
    wins = sum(
        all(
            c[PRIMARY]["all"][m] < c["paired"]["all"][m]
            for m in ("coordinate_l1_mm", "point_rmse_mm")
        )
        for c in cases.values()
    )
    worst = max(
        c[PRIMARY]["all"]["point_rmse_mm"]
        / max(c["paired"]["all"]["point_rmse_mm"], 1e-12)
        for c in cases.values()
    )
    a, b, c = means[PRIMARY], means["paired"], means["hard_baseline"]
    checks = {
        "two_percent_l1_gain": a["all"]["coordinate_l1_mm"]
        <= 0.98 * b["all"]["coordinate_l1_mm"],
        "two_percent_rmse_gain": a["all"]["point_rmse_mm"]
        <= 0.98 * b["all"]["point_rmse_mm"],
        "late_rmse_nonincreasing": a["late"]["point_rmse_mm"]
        <= b["late"]["point_rmse_mm"],
        "eight_of_thirteen_joint_wins": wins >= 8,
        "worst_rmse_ratio_at_most_1_05": worst <= 1.05,
        "rmse_bootstrap_upper_below_zero": bool(interval[1] < 0),
        "sparse_update_improves_hard_baseline": all(
            a["all"][m] < c["all"][m] for m in ("coordinate_l1_mm", "point_rmse_mm")
        ),
    }
    return {
        "case_metrics": cases,
        "decision": {
            "means": means,
            "checks": checks,
            "passed": all(checks.values()),
            "primary_joint_wins": wins,
            "primary_worst_case_rmse_ratio": worst,
            "paired_rmse_difference_95pct_mm": interval.tolist(),
            "transfer_authorized": False,
            "incumbent_modified": False,
        },
    }
