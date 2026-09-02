"""Exact source-grid decisions for the Tracking Cloth action audit."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    ActSenseFallbackCertificateV1,
    act_sense_fallback_certificate,
)


def _probe_binary_outcomes(features: np.ndarray) -> tuple[np.ndarray, list[float]]:
    outcomes: list[np.ndarray] = []
    thresholds: list[float] = []
    for row in features:
        ordered = np.sort(row)
        middle = ordered.size // 2
        threshold = float((ordered[middle - 1] + ordered[middle]) / 2.0)
        binary = (row > threshold).astype(np.int64)
        # A constant source feature is an uninformative one-outcome probe. Do
        # not invent a binary split from row order merely to manufacture
        # decision information.
        outcomes.append(binary)
        thresholds.append(threshold)
    return np.asarray(outcomes, dtype=np.int64), thresholds


def _fallback_action(losses: np.ndarray) -> int:
    mean_loss = np.mean(losses, axis=0)
    return int(np.flatnonzero(np.isclose(mean_loss, np.min(mean_loss)))[0])


def _resolve_source_actions(
    certificate: ActSenseFallbackCertificateV1,
    probe_outcomes: np.ndarray,
    block_indices: np.ndarray,
) -> np.ndarray:
    if certificate.output_mode in {"act", "fallback"}:
        return np.full(
            block_indices.size,
            certificate.terminal_action(),
            dtype=np.int64,
        )
    probe_index = certificate.selected_probe_index
    if probe_index is None:
        raise RuntimeError("sensing output is missing its probe")
    return np.asarray(
        [
            certificate.terminal_action(int(probe_outcomes[probe_index, index]))
            for index in block_indices
        ],
        dtype=np.int64,
    )


def decision_grid(
    blocks: list[tuple[str, int]],
    losses: np.ndarray,
    probe_outcomes: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    materials = list(protocol["materials"])
    actions = list(protocol["interactions"])
    material_index = {material: index for index, material in enumerate(materials)}
    classes = np.asarray(
        [material_index[material] for material, _ in blocks],
        dtype=np.int64,
    )
    prior = np.full(len(blocks), 1.0 / len(blocks), dtype=np.float64)
    fallback = _fallback_action(losses)
    loss_scale = float(max(np.quantile(losses, 0.9), 1e-12))
    normalized_losses = losses / loss_scale

    records: list[dict[str, Any]] = []
    for probe_cost in protocol["probe_cost_grid"]:
        for tolerance in protocol["regret_tolerance_grid"]:
            outputs: list[dict[str, Any]] = []
            selected_losses: list[float] = []
            oracle_losses: list[float] = []
            fallback_losses: list[float] = []
            for material in materials:
                quotient = np.zeros(len(materials), dtype=np.float64)
                quotient[material_index[material]] = 1.0
                certificate = act_sense_fallback_certificate(
                    prior,
                    quotient,
                    classes,
                    normalized_losses,
                    probe_outcomes,
                    np.full(
                        len(actions),
                        float(probe_cost),
                        dtype=np.float64,
                    ),
                    fallback_action_index=fallback,
                    regret_tolerance=float(tolerance),
                    probe_names=actions,
                    max_plan_count=int(protocol["max_plan_count"]),
                )
                block_indices = np.flatnonzero(classes == material_index[material])
                chosen = _resolve_source_actions(
                    certificate,
                    probe_outcomes,
                    block_indices,
                )
                row_indices = block_indices
                selected = losses[row_indices, chosen]
                selected_losses.extend(float(value) for value in selected)
                oracle_losses.extend(
                    float(value)
                    for value in np.min(losses[row_indices], axis=1)
                )
                fallback_losses.extend(
                    float(value) for value in losses[row_indices, fallback]
                )
                outputs.append(
                    {
                        "material": material,
                        "mode": certificate.output_mode,
                        "selected_probe": (
                            None
                            if certificate.selected_probe_index is None
                            else actions[certificate.selected_probe_index]
                        ),
                        "worst_case_regret": (
                            certificate.plan_certificate.minimax_worst_case_regret
                        ),
                        "chosen_actions_by_source_repetition": [
                            actions[int(value)] for value in chosen
                        ],
                    }
                )
            modes = Counter(item["mode"] for item in outputs)
            records.append(
                {
                    "probe_cost": float(probe_cost),
                    "regret_tolerance": float(tolerance),
                    "mode_counts": {
                        mode: int(modes.get(mode, 0))
                        for mode in ("act", "sense", "fallback")
                    },
                    "mean_source_loss": float(np.mean(selected_losses)),
                    "mean_fallback_loss": float(np.mean(fallback_losses)),
                    "mean_oracle_loss": float(np.mean(oracle_losses)),
                    "relative_gain_vs_fallback": float(
                        (np.mean(fallback_losses) - np.mean(selected_losses))
                        / max(np.mean(fallback_losses), 1e-12)
                    ),
                    "outputs": outputs,
                }
            )

    candidates = [
        item
        for item in records
        if item["mode_counts"]["sense"] > 0
        and item["relative_gain_vs_fallback"] > 0.0
    ]
    if candidates:
        selected = min(
            candidates,
            key=lambda item: (
                -item["relative_gain_vs_fallback"],
                item["mode_counts"]["fallback"],
                item["probe_cost"],
                item["regret_tolerance"],
            ),
        )
        selected_setting = {
            key: selected[key]
            for key in (
                "probe_cost",
                "regret_tolerance",
                "mode_counts",
                "mean_source_loss",
                "mean_fallback_loss",
                "mean_oracle_loss",
                "relative_gain_vs_fallback",
                "outputs",
            )
        }
    else:
        selected_setting = {
            "status": "no-source-setting-combines-sensing-and-fallback-gain"
        }
    return records, {
        "fallback_action": actions[fallback],
        "loss_scale": loss_scale,
        "selected_source_setting": selected_setting,
    }
