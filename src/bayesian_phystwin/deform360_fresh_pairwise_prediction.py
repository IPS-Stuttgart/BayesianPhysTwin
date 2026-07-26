"""Target-free frozen pairwise-belief prediction for fresh Deform360 objects.

This module extracts the prediction part of the already frozen open-27
diagnostic. It never accepts a target trajectory. The selected physical or
persistence backbone is chosen from the current sparse RGB-prefix observation,
then the frozen pairwise-consensus gate either admits the recursive RBF update
or preserves that selected backbone bit exactly.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
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


PROTOCOL_ID = "deform360-fresh-pairwise-belief-v1"
UPDATE_FRAMES = (19, 38, 57)
MINIMUM_SELECTOR_SUPPORT = 3
PHYSICAL_ARM = "physical_prior"
PERSISTENCE_ARM = "persistence"
SELECTED_RAW_ARM = "selected_raw_backbone_persistence_insufficient_default"
CANDIDATE_ARM = "raw_selected_backbone_full_blend_rbf_pairwise_clique"


def _corrected_frame(
    backbone_frame_m: np.ndarray,
    correction_m: np.ndarray,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    return (
        np.asarray(backbone_frame_m, dtype=float)
        + np.asarray(correction_m, dtype=float)
    ).astype(dtype, copy=False)


def predict_fresh_pairwise_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_frames: Sequence[int] = UPDATE_FRAMES,
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Produce the frozen candidate without accepting any future outcome."""

    prior_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    prior = np.asarray(prior_input, dtype=float)
    persistence = np.asarray(persistence_input, dtype=float)
    measurement = np.asarray(measurement_m, dtype=float)
    measurement_visible = np.asarray(measurement_visibility, dtype=bool)
    measurement_valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    updates_requested = tuple(int(frame) for frame in update_frames)
    if prior.shape != persistence.shape:
        raise ValueError("physical prior and persistence shapes differ")
    if prior.ndim != 3 or prior.shape[2] != 3:
        raise ValueError("trajectories must have shape (T, N, 3)")
    if measurement.shape != prior.shape:
        raise ValueError("measurement must match the trajectory shape")
    for name, mask in (
        ("measurement_visibility", measurement_visible),
        ("measurement_validity", measurement_valid),
    ):
        if mask.shape != prior.shape[:2]:
            raise ValueError(f"{name} must have shape (T, N)")
    if centers.ndim != 1 or len(centers) != len(np.unique(centers)):
        raise ValueError("center_ids must be a unique vector")
    if np.any(centers < 0) or np.any(centers >= prior.shape[1]):
        raise ValueError("centre ID exceeds trajectory")
    if (
        not updates_requested
        or tuple(sorted(set(updates_requested))) != updates_requested
        or updates_requested[0] < 0
        or updates_requested[-1] >= len(prior)
    ):
        raise ValueError("update frames must be increasing and inside the trajectory")
    if not np.array_equal(prior_input[0], persistence_input[0]):
        raise ValueError("physical and persistence frame-zero identities differ")

    gate_cfg = gate_config or PairwiseCorrespondenceGateConfig()
    belief_cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    backbones = {PHYSICAL_ARM: prior, PERSISTENCE_ARM: persistence}
    states = {
        name: initialize_recursive_rbf_belief(
            centers,
            backbone[0, centers],
            backbone[0],
            config=belief_cfg,
        )
        for name, backbone in backbones.items()
    }
    output_dtype = prior_input.dtype
    selected_trajectory = prior_input.copy()
    candidate_trajectory = prior_input.copy()
    update_records: list[dict[str, Any]] = []

    for update_index, update in enumerate(updates_requested):
        stop = (
            updates_requested[update_index + 1]
            if update_index + 1 < len(updates_requested)
            else len(prior)
        )
        available = (
            measurement_visible[update, centers]
            & measurement_valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        available_ids = centers[available]
        observed = measurement[update, available_ids]
        selector_support = len(available_ids) >= MINIMUM_SELECTOR_SUPPORT
        if selector_support:
            chamfer = {
                name: _symmetric_set_chamfer_m(
                    backbone[update, available_ids], observed
                )
                for name, backbone in backbones.items()
            }
            selected_name = min(
                (PHYSICAL_ARM, PERSISTENCE_ARM),
                key=lambda name: (
                    chamfer[name],
                    0 if name == PHYSICAL_ARM else 1,
                ),
            )
        else:
            chamfer = (
                {
                    name: _symmetric_set_chamfer_m(
                        backbone[update, available_ids], observed
                    )
                    for name, backbone in backbones.items()
                }
                if len(available_ids)
                else {PHYSICAL_ARM: None, PERSISTENCE_ARM: None}
            )
            selected_name = PERSISTENCE_ARM
        selected = backbones[selected_name]
        selected_trajectory[update + 1 : stop] = selected[update + 1 : stop]
        candidate_trajectory[update + 1 : stop] = selected[update + 1 : stop]

        gates = {}
        residuals: dict[str, np.ndarray] = {}
        for backbone_name, backbone in backbones.items():
            residual = np.full((len(centers), 3), np.nan, dtype=float)
            residual[available] = observed - backbone[update, available_ids]
            residuals[backbone_name] = residual
            gates[backbone_name] = detect_pairwise_consensus_correspondences(
                backbone[update, centers],
                measurement[update, centers],
                available,
                material_ids=centers,
                config=gate_cfg,
            )

        if selector_support:
            for backbone_name, backbone in backbones.items():
                gate = gates[backbone_name]
                if gate.accepted:
                    states[backbone_name], _ = update_recursive_rbf_belief(
                        states[backbone_name],
                        update,
                        backbone[update, centers],
                        residuals[backbone_name],
                        gate.inlier_mask.copy(),
                        config=belief_cfg,
                    )
            selected_gate = gates[selected_name]
            if selected_gate.accepted:
                for frame in range(update + 1, stop):
                    decoded = decode_recursive_rbf_belief(
                        states[selected_name],
                        selected[update],
                        forecast_frames=frame - update,
                        config=belief_cfg,
                    )
                    candidate_trajectory[frame] = _corrected_frame(
                        selected[frame], decoded.mean_m, dtype=output_dtype
                    )
        else:
            selected_gate = gates[selected_name]

        accepted = bool(selector_support and selected_gate.accepted)
        exact_fallback = bool(
            accepted
            or np.array_equal(
                candidate_trajectory[update + 1 : stop],
                selected[update + 1 : stop],
            )
        )
        if not exact_fallback:
            raise AssertionError("pairwise rejection did not preserve the backbone")
        update_records.append(
            {
                "frame": int(update),
                "interval_end_exclusive": int(stop),
                "available_center_count": int(len(available_ids)),
                "selector_support_sufficient": selector_support,
                "selected_backbone": selected_name,
                "selector_decision": (
                    "current_observation_chamfer"
                    if selector_support
                    else "insufficient_support_persistence_default"
                ),
                "current_observation_chamfer_m": chamfer,
                "selected_pairwise_gate": {
                    "accepted": accepted,
                    "decision": (
                        selected_gate.decision
                        if selector_support
                        else "insufficient_selector_support"
                    ),
                    "inlier_count": selected_gate.inlier_count,
                    "inlier_fraction": selected_gate.inlier_fraction,
                    "compatible_pair_fraction": (
                        selected_gate.compatible_pair_fraction
                    ),
                    "bit_exact_raw_fallback": bool(not accepted and exact_fallback),
                },
            }
        )

    arrays = {
        PHYSICAL_ARM: prior_input.copy(),
        PERSISTENCE_ARM: persistence_input.copy(),
        SELECTED_RAW_ARM: selected_trajectory,
        CANDIDATE_ARM: candidate_trajectory,
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "center_ids": centers.tolist(),
        "update_frames": list(updates_requested),
        "gate_config": asdict(gate_cfg),
        "belief_config": asdict(belief_cfg),
        "updates": update_records,
        "method_contract": {
            "selector_minimum_support": MINIMUM_SELECTOR_SUPPORT,
            "selector_insufficient_default": PERSISTENCE_ARM,
            "backbone_states": "separate physical and persistence RBF states",
            "clique_rejection": "bit-exact selected raw backbone",
            "target_argument": False,
        },
        "information_boundary": {
            "future_target_read": False,
            "outcome_manifest_read": False,
            "prediction_depends_on": (
                "sealed physical and persistence backbones plus causal RGB-prefix "
                "measurements"
            ),
        },
    }
    return report, arrays


__all__ = [
    "CANDIDATE_ARM",
    "MINIMUM_SELECTOR_SUPPORT",
    "PERSISTENCE_ARM",
    "PHYSICAL_ARM",
    "PROTOCOL_ID",
    "SELECTED_RAW_ARM",
    "UPDATE_FRAMES",
    "predict_fresh_pairwise_arrays",
]
