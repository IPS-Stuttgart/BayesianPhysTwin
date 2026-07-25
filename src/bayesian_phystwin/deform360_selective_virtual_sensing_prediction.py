"""Target-free primary predictor for selective Deform360 virtual sensing."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from .cpd_registration import NonrigidCpdConfig, fit_nonrigid_cpd
from .deform360_raw_pairwise_correspondence_diagnostic import (
    CPD_ARM,
    MINIMUM_SELECTOR_SUPPORT,
    PERSISTENCE_CLIQUE_RBF_ARM,
    UNGATED_RBF_ARM,
    _corrected_frame,
)
from .phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
    detect_pairwise_consensus_correspondences,
)
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


def _validate_prediction_inputs(
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    center_ids: np.ndarray,
    update_frames: tuple[int, ...],
) -> None:
    if persistence_m.ndim != 3 or persistence_m.shape[2] != 3:
        raise ValueError("persistence must have shape (T, N, 3)")
    if measurement_m.shape != persistence_m.shape:
        raise ValueError("measurement must match persistence shape")
    for name, mask in (
        ("measurement_visibility", measurement_visibility),
        ("measurement_validity", measurement_validity),
    ):
        if mask.shape != persistence_m.shape[:2]:
            raise ValueError(f"{name} must have shape (T, N)")
    if center_ids.ndim != 1 or len(center_ids) != len(np.unique(center_ids)):
        raise ValueError("center_ids must be a unique vector")
    if np.any(center_ids < 0) or np.any(center_ids >= persistence_m.shape[1]):
        raise ValueError("centre ID exceeds persistence trajectory")
    if tuple(sorted(set(update_frames))) != update_frames:
        raise ValueError("update frames must be strictly increasing")
    if not update_frames or update_frames[-1] >= len(persistence_m):
        raise ValueError("update frame exceeds persistence trajectory")


def predict_persistence_pairwise_rbf_arrays(
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_frames: Sequence[int] = (19, 38, 57),
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Predict a full trajectory without accepting any target-like input."""

    persistence_input = np.asarray(persistence_m)
    persistence = np.asarray(persistence_input, dtype=float)
    measurement = np.asarray(measurement_m, dtype=float)
    measurement_visible = np.asarray(measurement_visibility, dtype=bool)
    measurement_valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    updates_tuple = tuple(int(frame) for frame in update_frames)
    _validate_prediction_inputs(
        persistence,
        measurement,
        measurement_visible,
        measurement_valid,
        centers,
        updates_tuple,
    )
    gate_cfg = gate_config or PairwiseCorrespondenceGateConfig()
    belief_cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    state = initialize_recursive_rbf_belief(
        centers,
        persistence[0, centers],
        persistence[0],
        config=belief_cfg,
    )
    prediction = persistence_input.copy()
    output_dtype = persistence_input.dtype
    update_reports: list[dict[str, Any]] = []

    for update_index, update in enumerate(updates_tuple):
        stop = (
            updates_tuple[update_index + 1]
            if update_index + 1 < len(updates_tuple)
            else len(persistence)
        )
        prediction[update + 1 : stop] = persistence_input[update + 1 : stop]
        available = (
            measurement_visible[update, centers]
            & measurement_valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        residual = np.full((len(centers), 3), np.nan, dtype=float)
        available_ids = centers[available]
        residual[available] = (
            measurement[update, available_ids]
            - persistence[update, available_ids]
        )
        gate = detect_pairwise_consensus_correspondences(
            persistence[update, centers],
            measurement[update, centers],
            available,
            material_ids=centers,
            config=gate_cfg,
        )
        support_sufficient = len(available_ids) >= MINIMUM_SELECTOR_SUPPORT
        accepted = support_sufficient and gate.accepted
        if accepted:
            state, belief_reliability = update_recursive_rbf_belief(
                state,
                update,
                persistence[update, centers],
                residual,
                gate.inlier_mask.copy(),
                config=belief_cfg,
            )
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    state,
                    persistence[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                prediction[frame] = _corrected_frame(
                    persistence[frame], decoded.mean_m, dtype=output_dtype
                )
        else:
            belief_reliability = None
            if not np.array_equal(
                prediction[update + 1 : stop],
                persistence_input[update + 1 : stop],
            ):
                raise AssertionError("abstention did not preserve persistence")
        update_reports.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "available_center_count": int(len(available_ids)),
                "selector_support_sufficient": support_sufficient,
                "accepted": accepted,
                "decision": (
                    gate.decision
                    if support_sufficient
                    else "insufficient_selector_support"
                ),
                "inlier_count": gate.inlier_count,
                "inlier_fraction": gate.inlier_fraction,
                "compatible_pair_fraction": gate.compatible_pair_fraction,
                "bit_exact_persistence_fallback": bool(
                    not accepted
                    and np.array_equal(
                        prediction[update + 1 : stop],
                        persistence_input[update + 1 : stop],
                    )
                ),
                "belief_reliability": (
                    None
                    if belief_reliability is None
                    else {
                        "minimum": float(np.min(belief_reliability)),
                        "mean": float(np.mean(belief_reliability)),
                        "maximum": float(np.max(belief_reliability)),
                    }
                ),
            }
        )

    report = {
        "arm": PERSISTENCE_CLIQUE_RBF_ARM,
        "center_ids": centers.tolist(),
        "update_frames": list(updates_tuple),
        "gate_config": asdict(gate_cfg),
        "belief_config": asdict(belief_cfg),
        "updates": update_reports,
        "information_boundary": {
            "target_argument_accepted": False,
            "physical_prior_argument_accepted": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
        },
    }
    return report, prediction


def predict_persistence_control_arrays(
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_frames: Sequence[int] = (19, 38, 57),
    belief_config: RecursiveRbfBeliefConfig | None = None,
    cpd_config: NonrigidCpdConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the frozen ungated-RBF and unordered-CPD controls target-free."""

    persistence_input = np.asarray(persistence_m)
    persistence = np.asarray(persistence_input, dtype=float)
    measurement = np.asarray(measurement_m, dtype=float)
    measurement_visible = np.asarray(measurement_visibility, dtype=bool)
    measurement_valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    updates_tuple = tuple(int(frame) for frame in update_frames)
    _validate_prediction_inputs(
        persistence,
        measurement,
        measurement_visible,
        measurement_valid,
        centers,
        updates_tuple,
    )
    belief_cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    cpd_cfg = cpd_config or NonrigidCpdConfig()
    belief = initialize_recursive_rbf_belief(
        centers,
        persistence[0, centers],
        persistence[0],
        config=belief_cfg,
    )
    ungated = persistence_input.copy()
    cpd = persistence_input.copy()
    output_dtype = persistence_input.dtype
    updates: list[dict[str, Any]] = []

    for update_index, update in enumerate(updates_tuple):
        stop = (
            updates_tuple[update_index + 1]
            if update_index + 1 < len(updates_tuple)
            else len(persistence)
        )
        ungated[update + 1 : stop] = persistence_input[update + 1 : stop]
        cpd[update + 1 : stop] = persistence_input[update + 1 : stop]
        available = (
            measurement_visible[update, centers]
            & measurement_valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        available_ids = centers[available]
        supported = len(available_ids) >= MINIMUM_SELECTOR_SUPPORT
        cpd_transform = None
        cpd_error = None
        if supported:
            residual = np.full((len(centers), 3), np.nan, dtype=float)
            residual[available] = (
                measurement[update, available_ids]
                - persistence[update, available_ids]
            )
            belief, reliability = update_recursive_rbf_belief(
                belief,
                update,
                persistence[update, centers],
                residual,
                available.copy(),
                config=belief_cfg,
            )
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    belief,
                    persistence[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                ungated[frame] = _corrected_frame(
                    persistence[frame], decoded.mean_m, dtype=output_dtype
                )
            try:
                cpd_transform = fit_nonrigid_cpd(
                    persistence[update, available_ids],
                    measurement[update, available_ids],
                    config=cpd_cfg,
                )
                for frame in range(update + 1, stop):
                    cpd[frame] = cpd_transform.transform(
                        persistence[frame]
                    ).astype(output_dtype, copy=False)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                cpd_error = f"{type(error).__name__}: {error}"
        else:
            reliability = None
        if cpd_transform is None and not np.array_equal(
            cpd[update + 1 : stop], persistence_input[update + 1 : stop]
        ):
            raise AssertionError("CPD failure did not preserve persistence")
        updates.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "available_center_count": int(len(available_ids)),
                "selector_support_sufficient": supported,
                "ungated_rbf_updated": supported,
                "ungated_mean_reliability": (
                    None if reliability is None else float(np.mean(reliability))
                ),
                "cpd_fit_performed": cpd_transform is not None,
                "cpd_fit_error": cpd_error,
                "cpd_effective_correspondence_count": (
                    None
                    if cpd_transform is None
                    else cpd_transform.effective_correspondence_count
                ),
                "cpd_bit_exact_persistence_fallback": bool(
                    cpd_transform is None
                    and np.array_equal(
                        cpd[update + 1 : stop],
                        persistence_input[update + 1 : stop],
                    )
                ),
            }
        )

    report = {
        "arms": [UNGATED_RBF_ARM, CPD_ARM],
        "center_ids": centers.tolist(),
        "update_frames": list(updates_tuple),
        "belief_config": asdict(belief_cfg),
        "cpd_config": asdict(cpd_cfg),
        "updates": updates,
        "information_boundary": {
            "target_argument_accepted": False,
            "physical_prior_argument_accepted": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
        },
    }
    return report, {UNGATED_RBF_ARM: ungated, CPD_ARM: cpd}


__all__ = [
    "predict_persistence_control_arrays",
    "predict_persistence_pairwise_rbf_arrays",
]
