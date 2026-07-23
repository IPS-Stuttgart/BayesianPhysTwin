"""Target-free covariance-gated 4-to-8-view recursive RBF prediction.

This module contains only the predictor used by the development diagnostic.
It never accepts a target trajectory or an outcome.  At each causal update it
first tests a four-view triangulation-covariance diagnostic, then an eight-view
diagnostic.  When neither has enough supported centers at the frozen routing
threshold, the complete forecast interval falls back to the physical prior and
neither recursive discrepancy state is updated.  The covariance score is used
only for target-free routing.  It is not interpreted as a calibrated safety
probability.

The camera count is the number of distinct dynamic RGB prefixes activated
after frame-zero all-camera planning.  A four-view prefix is cached before an
eight-view escalation, so overlapping cameras are counted once.  It is not a
claim about the number of sensors required for frame-zero reconstruction or
camera selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_held_online_prefix import (
    FRAME_COUNT,
    HELD_RBF_CONFIG,
    MINIMUM_SELECTOR_SUPPORT,
    UPDATE_FRAMES,
    _symmetric_set_chamfer_m,
)
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


ADAPTIVE_COVARIANCE_PROTOCOL_ID = (
    "deform360-target-free-adaptive-covariance-selected-rbf-v1-development"
)


@dataclass(frozen=True)
class AdaptiveCovarianceRbfConfig:
    """Frozen target-free routing choices for the development candidate."""

    camera_budgets: tuple[int, ...] = (4, 8)
    covariance_quantile: float = 0.90
    maximum_normalized_covariance_dispersion: float = 0.015
    minimum_valid_covariance_centers: int = 8

    def __post_init__(self) -> None:
        if self.camera_budgets != (4, 8):
            raise ValueError("the frozen adaptive route must be exactly 4 then 8")
        if not 0.0 < self.covariance_quantile <= 1.0:
            raise ValueError("covariance_quantile must lie in (0, 1]")
        if (
            not np.isfinite(self.maximum_normalized_covariance_dispersion)
            or self.maximum_normalized_covariance_dispersion <= 0.0
        ):
            raise ValueError("covariance dispersion threshold must be positive")
        if self.minimum_valid_covariance_centers < MINIMUM_SELECTOR_SUPPORT:
            raise ValueError("covariance support cannot be lower than selector support")


FROZEN_ADAPTIVE_COVARIANCE_CONFIG = AdaptiveCovarianceRbfConfig()


def _object_bbox_diagonal_m(frame_zero_points_m: np.ndarray) -> float:
    points = np.asarray(frame_zero_points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("frame_zero_points_m must have shape (N, 3)")
    if not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("frame-zero geometry must be nonempty and finite")
    diagonal = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("frame-zero object bounding-box diagonal is not positive")
    return diagonal


def normalized_covariance_dispersion(
    measurement_covariance_m2: np.ndarray,
    measurement_covariance_valid: np.ndarray,
    center_ids: np.ndarray,
    frame_index: int,
    frame_zero_points_m: np.ndarray,
    *,
    quantile: float = 0.90,
) -> dict[str, Any]:
    """Summarize target-free triangulation covariance as a routing score."""

    covariance = np.asarray(measurement_covariance_m2, dtype=float)
    valid = np.asarray(measurement_covariance_valid, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    if covariance.ndim != 4 or covariance.shape[2:] != (3, 3):
        raise ValueError("measurement covariance must have shape (T, N, 3, 3)")
    if valid.shape != covariance.shape[:2]:
        raise ValueError("measurement covariance validity shape changed")
    if not 0 <= frame_index < len(covariance):
        raise ValueError("frame_index is outside the covariance trajectory")
    if (
        centers.ndim != 1
        or len(np.unique(centers)) != len(centers)
        or np.any(centers < 0)
        or np.any(centers >= covariance.shape[1])
    ):
        raise ValueError("invalid center IDs")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")

    selected_ids = centers[valid[frame_index, centers]]
    selected_covariance = covariance[frame_index, selected_ids]
    if len(selected_covariance):
        if not np.all(np.isfinite(selected_covariance)):
            raise ValueError("valid measurement covariance is non-finite")
        if not np.allclose(
            selected_covariance,
            np.swapaxes(selected_covariance, 1, 2),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("valid measurement covariance is not symmetric")
        eigenvalues = np.linalg.eigvalsh(selected_covariance)
        if np.any(eigenvalues < -1e-12):
            raise ValueError(
                "valid measurement covariance is not positive semidefinite"
            )
        radial_standard_deviation_m = np.sqrt(
            np.maximum(np.trace(selected_covariance, axis1=1, axis2=2), 0.0)
        )
        radial_quantile_m = float(np.quantile(radial_standard_deviation_m, quantile))
        normalized = radial_quantile_m / _object_bbox_diagonal_m(frame_zero_points_m)
    else:
        radial_standard_deviation_m = np.empty(0, dtype=float)
        radial_quantile_m = None
        normalized = None

    return {
        "frame": int(frame_index),
        "valid_covariance_center_count": int(len(selected_ids)),
        "valid_covariance_center_ids": selected_ids.tolist(),
        "covariance_quantile": float(quantile),
        "radial_standard_deviation_quantile_m": radial_quantile_m,
        "frame_zero_bbox_diagonal_m": _object_bbox_diagonal_m(frame_zero_points_m),
        "normalized_covariance_dispersion": (
            None if normalized is None else float(normalized)
        ),
        "probabilistic_calibration_claimed": False,
    }


def _validate_budget_arrays(
    reference_shape: tuple[int, ...],
    center_ids: np.ndarray,
    camera_budgets: tuple[int, ...],
    selected_cameras_by_budget: Mapping[int, Sequence[str]],
    measurement_m_by_budget: Mapping[int, np.ndarray],
    measurement_validity_by_budget: Mapping[int, np.ndarray],
    measurement_covariance_m2_by_budget: Mapping[int, np.ndarray],
    measurement_covariance_valid_by_budget: Mapping[int, np.ndarray],
) -> None:
    expected = set(camera_budgets)
    mappings: tuple[tuple[str, Mapping[int, np.ndarray]], ...] = (
        ("measurement", measurement_m_by_budget),
        ("measurement validity", measurement_validity_by_budget),
        ("measurement covariance", measurement_covariance_m2_by_budget),
        (
            "measurement covariance validity",
            measurement_covariance_valid_by_budget,
        ),
    )
    if set(selected_cameras_by_budget) != expected:
        raise ValueError("selected-camera budgets changed")
    for label, values in mappings:
        if set(values) != expected:
            raise ValueError(f"{label} budgets changed")
    for budget in camera_budgets:
        cameras = tuple(selected_cameras_by_budget[budget])
        if (
            len(cameras) != budget
            or len(set(cameras)) != len(cameras)
            or any(not isinstance(camera, str) or not camera for camera in cameras)
        ):
            raise ValueError(f"{budget}-view selected cameras are invalid")
        measurement = np.asarray(measurement_m_by_budget[budget])
        measurement_validity = np.asarray(measurement_validity_by_budget[budget])
        covariance = np.asarray(measurement_covariance_m2_by_budget[budget])
        covariance_valid = np.asarray(measurement_covariance_valid_by_budget[budget])
        if measurement.shape != reference_shape:
            raise ValueError(f"{budget}-view measurement shape changed")
        if measurement_validity.shape != reference_shape[:2]:
            raise ValueError(f"{budget}-view measurement-validity shape changed")
        if covariance.shape != (*reference_shape[:2], 3, 3):
            raise ValueError(f"{budget}-view measurement covariance shape changed")
        if covariance_valid.shape != reference_shape[:2]:
            raise ValueError(
                f"{budget}-view measurement covariance-validity shape changed"
            )
        if np.any(
            np.asarray(covariance_valid, dtype=bool)
            & ~np.asarray(measurement_validity, dtype=bool)
        ):
            raise ValueError(f"{budget}-view covariance claims an invalid measurement")
        if np.any(center_ids >= reference_shape[1]):
            raise ValueError("center ID exceeds one or more budget arrays")
    if tuple(selected_cameras_by_budget[8])[:4] != tuple(selected_cameras_by_budget[4]):
        raise ValueError("four-view cameras are not the ordered eight-view prefix")


def predict_adaptive_covariance_selected_backbone_rbf(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    selected_cameras_by_budget: Mapping[int, Sequence[str]],
    measurement_m_by_budget: Mapping[int, np.ndarray],
    measurement_validity_by_budget: Mapping[int, np.ndarray],
    measurement_covariance_m2_by_budget: Mapping[int, np.ndarray],
    measurement_covariance_valid_by_budget: Mapping[int, np.ndarray],
    *,
    center_ids: np.ndarray,
    config: AdaptiveCovarianceRbfConfig | None = None,
    rbf_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Predict with a target-free 4-view, 8-view, physical-fallback cascade."""

    routing = config or FROZEN_ADAPTIVE_COVARIANCE_CONFIG
    belief_config = rbf_config or HELD_RBF_CONFIG
    if routing != FROZEN_ADAPTIVE_COVARIANCE_CONFIG:
        raise ValueError("adaptive covariance configuration changed")
    if belief_config != HELD_RBF_CONFIG:
        raise ValueError("held RBF configuration changed")

    prior_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    prior = np.asarray(prior_input, dtype=float)
    persistence = np.asarray(persistence_input, dtype=float)
    centers = np.asarray(center_ids, dtype=np.int64)
    if prior.shape != persistence.shape:
        raise ValueError("physical and persistence shapes differ")
    if prior.ndim != 3 or prior.shape[0] != FRAME_COUNT or prior.shape[2] != 3:
        raise ValueError("physical prior must have shape (76, N, 3)")
    if not np.all(np.isfinite(prior)) or not np.all(np.isfinite(persistence)):
        raise ValueError("backbone trajectories must be finite")
    if (
        centers.ndim != 1
        or len(centers) == 0
        or len(np.unique(centers)) != len(centers)
        or np.any(centers < 0)
        or np.any(centers >= prior.shape[1])
    ):
        raise ValueError("invalid center IDs")
    if not np.array_equal(prior[0], persistence[0]):
        raise ValueError("backbone frame-zero material identities differ")

    _validate_budget_arrays(
        prior.shape,
        centers,
        routing.camera_budgets,
        selected_cameras_by_budget,
        measurement_m_by_budget,
        measurement_validity_by_budget,
        measurement_covariance_m2_by_budget,
        measurement_covariance_valid_by_budget,
    )

    output_dtype = prior_input.dtype
    selected_raw = prior_input.copy()
    prediction = prior_input.copy()
    backbones = {"physical_prior": prior, "persistence": persistence}
    states = {
        name: initialize_recursive_rbf_belief(
            centers,
            trajectory[0, centers],
            trajectory[0],
            config=belief_config,
        )
        for name, trajectory in backbones.items()
    }
    updates: list[dict[str, Any]] = []

    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(prior)
        )
        budget_diagnostics: dict[str, dict[str, Any]] = {}
        selected_budget: int | None = None
        activated_cameras: list[str] = []
        for budget in routing.camera_budgets:
            activated_cameras = list(selected_cameras_by_budget[budget])
            reliability = normalized_covariance_dispersion(
                measurement_covariance_m2_by_budget[budget],
                measurement_covariance_valid_by_budget[budget],
                centers,
                update,
                prior[0],
                quantile=routing.covariance_quantile,
            )
            normalized = reliability["normalized_covariance_dispersion"]
            reliable = (
                reliability["valid_covariance_center_count"]
                >= routing.minimum_valid_covariance_centers
                and normalized is not None
                and normalized <= routing.maximum_normalized_covariance_dispersion
            )
            budget_diagnostics[str(budget)] = {
                **reliability,
                "reliable": bool(reliable),
            }
            if reliable:
                selected_budget = budget
                break

        if selected_budget is None:
            selected_raw[update + 1 : stop] = prior_input[update + 1 : stop]
            prediction[update + 1 : stop] = prior_input[update + 1 : stop]
            if not np.array_equal(
                prediction[update + 1 : stop],
                prior_input[update + 1 : stop],
            ):
                raise AssertionError("physical fallback is not bit-exact")
            updates.append(
                {
                    "frame": int(update),
                    "stop_frame_exclusive": int(stop),
                    "route": "physical_prior_fallback",
                    "selected_camera_budget": None,
                    "tracked_camera_count": int(len(activated_cameras)),
                    "tracked_cameras": list(activated_cameras),
                    "selected_backbone": "physical_prior",
                    "rbf_correction_applied": False,
                    "state_updated": False,
                    "budget_diagnostics": budget_diagnostics,
                }
            )
            continue

        measurement = np.asarray(measurement_m_by_budget[selected_budget], dtype=float)
        measurement_validity = np.asarray(
            measurement_validity_by_budget[selected_budget], dtype=bool
        )
        available = (
            measurement_validity[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        available_ids = centers[available]
        if len(available_ids) < MINIMUM_SELECTOR_SUPPORT:
            raise AssertionError("reliable covariance route lacks selector support")
        observed = measurement[update, available_ids]
        chamfer = {
            name: _symmetric_set_chamfer_m(
                trajectory[update, available_ids],
                observed,
            )
            for name, trajectory in backbones.items()
        }
        selected_name = min(
            ("physical_prior", "persistence"),
            key=lambda name: (
                chamfer[name],
                0 if name == "physical_prior" else 1,
            ),
        )
        selected = backbones[selected_name]
        selected_raw[update + 1 : stop] = selected[update + 1 : stop]
        prediction[update + 1 : stop] = selected[update + 1 : stop]
        for backbone_name, trajectory in backbones.items():
            residual = np.full((len(centers), 3), np.nan, dtype=float)
            residual[available] = observed - trajectory[update, available_ids]
            posterior, _ = update_recursive_rbf_belief(
                states[backbone_name],
                update,
                trajectory[update, centers],
                residual,
                available,
                config=belief_config,
            )
            states[backbone_name] = posterior
        posterior = states[selected_name]
        for frame in range(update + 1, stop):
            decoded = decode_recursive_rbf_belief(
                posterior,
                selected[update],
                forecast_frames=frame - update,
                config=belief_config,
            )
            prediction[frame] = (selected[frame].astype(float) + decoded.mean_m).astype(
                output_dtype, copy=False
            )
        updates.append(
            {
                "frame": int(update),
                "stop_frame_exclusive": int(stop),
                "route": f"{selected_budget}_view_rbf",
                "selected_camera_budget": int(selected_budget),
                "tracked_camera_count": int(len(activated_cameras)),
                "tracked_cameras": list(activated_cameras),
                "available_center_count": int(len(available_ids)),
                "selected_backbone": selected_name,
                "current_observation_chamfer_m": chamfer,
                "rbf_correction_applied": True,
                "state_updated": True,
                "budget_diagnostics": budget_diagnostics,
            }
        )

    return (
        prediction,
        selected_raw,
        {
            "protocol_id": ADAPTIVE_COVARIANCE_PROTOCOL_ID,
            "config": asdict(routing),
            "rbf_config": asdict(belief_config),
            "fallback": {
                "trajectory": "physical_prior",
                "rbf_state_update": False,
                "bit_exact": True,
            },
            "camera_budget_semantics": (
                "distinct dynamic RGB prefixes activated after all-view "
                "frame-zero planning; lower-budget prefixes are cached"
            ),
            "calibration_boundary": (
                "covariance is an outcome-free routing score, not a calibrated "
                "coverage or safety probability"
            ),
            "updates": updates,
        },
    )


__all__ = [
    "ADAPTIVE_COVARIANCE_PROTOCOL_ID",
    "AdaptiveCovarianceRbfConfig",
    "FROZEN_ADAPTIVE_COVARIANCE_CONFIG",
    "normalized_covariance_dispersion",
    "predict_adaptive_covariance_selected_backbone_rbf",
]
