#!/usr/bin/env python3
"""Hierarchical diagnosis of transferable and nontransferable model discrepancy.

The module deliberately operates on a compact, provider-neutral residual-panel
contract.  A simulator/data adapter supplies the physical-model residual and
feature blocks; this module performs source-only inference, grouped validation,
mechanism attribution, and exact-fallback auditing.

NPZ contract
------------
Required arrays:

``y``
    Physical-model residual, shape ``(samples, output_dimensions)``.
``trajectory_id``
    Complete-trajectory identity, shape ``(samples,)``.
``object_id``
    Physical-object identity, shape ``(samples,)``.
``backend_id``
    Simulator-backend identity, shape ``(samples,)``.
``block__<group>``
    One or more two-dimensional feature blocks.  The registered groups are
    ``shared_physics``, ``object``, ``backend``, ``contact``, ``actuation``, and
    ``sensor``.

A neighbouring JSON manifest may override transfer eligibility and records the
feature semantics/calibration identities.  No target outcome is needed to
select groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

EPS = 1.0e-12
REGISTERED_GROUPS = (
    "shared_physics",
    "object",
    "backend",
    "contact",
    "actuation",
    "sensor",
)
DEFAULT_TRANSFER_ELIGIBILITY = {
    "shared_physics": True,
    "object": False,
    "backend": False,
    "contact": False,
    "actuation": False,
    "sensor": False,
}


class ContractError(ValueError):
    """Raised when an input would violate the registered evidence contract."""


@dataclass(frozen=True)
class ResidualPanel:
    """Provider-neutral residual and feature panel."""

    y: np.ndarray
    blocks: Mapping[str, np.ndarray]
    trajectory_id: np.ndarray
    object_id: np.ndarray
    backend_id: np.ndarray
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        y = np.asarray(self.y)
        if y.ndim != 2 or y.shape[0] < 4 or y.shape[1] < 1:
            raise ContractError("y must have shape (samples>=4, dimensions>=1)")
        if not np.all(np.isfinite(y)):
            raise ContractError("y contains non-finite values")
        n = y.shape[0]
        for name, values in (
            ("trajectory_id", self.trajectory_id),
            ("object_id", self.object_id),
            ("backend_id", self.backend_id),
        ):
            values = np.asarray(values)
            if values.ndim != 1 or values.shape[0] != n:
                raise ContractError(f"{name} must have shape ({n},)")
        if "shared_physics" not in self.blocks:
            raise ContractError("shared_physics feature block is required")
        unknown = sorted(set(self.blocks) - set(REGISTERED_GROUPS))
        if unknown:
            raise ContractError(f"unregistered feature groups: {unknown}")
        for name, block in self.blocks.items():
            block = np.asarray(block)
            if block.ndim != 2 or block.shape[0] != n or block.shape[1] < 1:
                raise ContractError(
                    f"block {name!r} must have shape ({n}, features>=1)"
                )
            if not np.all(np.isfinite(block)):
                raise ContractError(f"block {name!r} contains non-finite values")
        if len(np.unique(self.trajectory_id.astype(str))) < 2:
            raise ContractError("at least two complete trajectories are required")


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale > 1.0e-10, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


@dataclass(frozen=True)
class FittedARD:
    group_order: tuple[str, ...]
    slices: Mapping[str, slice]
    x_standardizers: Mapping[str, Standardizer]
    y_mean: np.ndarray
    coefficient_mean: np.ndarray
    posterior_covariance: np.ndarray
    noise_precision: float
    alpha: Mapping[str, float]
    effective_df_per_output: Mapping[str, float]
    iterations: int
    converged: bool

    def predict(
        self,
        blocks: Mapping[str, np.ndarray],
        active_groups: Iterable[str] | None = None,
    ) -> np.ndarray:
        active = set(self.group_order if active_groups is None else active_groups)
        n: int | None = None
        parts: list[np.ndarray] = []
        for name in self.group_order:
            if name not in blocks:
                raise ContractError(f"prediction panel lacks block {name!r}")
            block = np.asarray(blocks[name], dtype=float)
            if n is None:
                n = block.shape[0]
            if block.ndim != 2 or block.shape[0] != n:
                raise ContractError(f"invalid prediction block {name!r}")
            expected = self.slices[name].stop - self.slices[name].start
            if block.shape[1] != expected:
                raise ContractError(
                    f"block {name!r} has {block.shape[1]} features; expected {expected}"
                )
            transformed = self.x_standardizers[name].transform(block)
            if name not in active:
                transformed = np.zeros_like(transformed)
            parts.append(transformed)
        if n is None:
            raise ContractError("no prediction blocks supplied")
        x = np.concatenate(parts, axis=1)
        return self.y_mean + x @ self.coefficient_mean

    def predictive_variance(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        parts = [
            self.x_standardizers[name].transform(np.asarray(blocks[name], dtype=float))
            for name in self.group_order
        ]
        x = np.concatenate(parts, axis=1)
        epistemic = np.einsum("ni,ij,nj->n", x, self.posterior_covariance, x)
        return np.maximum(1.0 / self.noise_precision + epistemic, EPS)


def _stable_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix, rcond=1.0e-10)


def fit_group_ard(
    panel: ResidualPanel,
    *,
    groups: Sequence[str] | None = None,
    max_iterations: int = 250,
    tolerance: float = 1.0e-6,
) -> FittedARD:
    """Fit group-ARD Bayesian linear regression to source residuals.

    Outputs are conditionally independent but share group precisions.  Group
    effective degrees of freedom and leave-group-out losses are subsequently
    used as mechanism-diagnosis evidence.  All standardization is fitted on
    the supplied source panel only.
    """

    panel.validate()
    selected = tuple(groups or [g for g in REGISTERED_GROUPS if g in panel.blocks])
    if not selected or "shared_physics" not in selected:
        raise ContractError("selected groups must contain shared_physics")
    if len(set(selected)) != len(selected):
        raise ContractError("selected groups contain duplicates")

    standardizers: dict[str, Standardizer] = {}
    slices: dict[str, slice] = {}
    parts: list[np.ndarray] = []
    offset = 0
    for name in selected:
        if name not in panel.blocks:
            raise ContractError(f"selected group {name!r} is absent")
        raw = np.asarray(panel.blocks[name], dtype=float)
        standardizer = Standardizer.fit(raw)
        transformed = standardizer.transform(raw)
        standardizers[name] = standardizer
        slices[name] = slice(offset, offset + transformed.shape[1])
        offset += transformed.shape[1]
        parts.append(transformed)

    x = np.concatenate(parts, axis=1)
    y_mean = np.mean(panel.y, axis=0)
    y = np.asarray(panel.y, dtype=float) - y_mean
    n, p = x.shape
    output_dim = y.shape[1]

    alpha = {name: 1.0 for name in selected}
    initial_variance = float(np.mean(np.square(y)))
    beta = 1.0 / max(initial_variance, 1.0e-8)
    xtx = x.T @ x
    xty = x.T @ y
    coefficient = np.zeros((p, output_dim), dtype=float)
    covariance = np.eye(p, dtype=float)
    converged = False

    for iteration in range(1, max_iterations + 1):
        diagonal = np.empty(p, dtype=float)
        for name in selected:
            diagonal[slices[name]] = alpha[name]
        precision = beta * xtx + np.diag(diagonal)
        covariance = _stable_inverse(precision)
        coefficient_new = beta * covariance @ xty

        gamma: dict[str, float] = {}
        alpha_new: dict[str, float] = {}
        for name in selected:
            group_slice = slices[name]
            width = group_slice.stop - group_slice.start
            trace = float(np.trace(covariance[group_slice, group_slice]))
            gamma_value = float(np.clip(width - alpha[name] * trace, 0.0, width))
            energy = float(np.sum(np.square(coefficient_new[group_slice, :])))
            alpha_value = output_dim * gamma_value / max(energy, EPS)
            gamma[name] = gamma_value
            alpha_new[name] = float(np.clip(alpha_value, 1.0e-9, 1.0e12))

        residual = y - x @ coefficient_new
        effective_parameters = output_dim * sum(gamma.values())
        beta_numerator = max(n * output_dim - effective_parameters, 1.0)
        beta_new = beta_numerator / max(float(np.sum(np.square(residual))), EPS)
        beta_new = float(np.clip(beta_new, 1.0e-12, 1.0e12))

        relative_changes = [
            abs(math.log(alpha_new[name]) - math.log(alpha[name]))
            for name in selected
        ]
        relative_changes.append(abs(math.log(beta_new) - math.log(beta)))
        coefficient = coefficient_new
        alpha = alpha_new
        beta = beta_new
        if max(relative_changes) < tolerance:
            converged = True
            break

    effective_df = {}
    for name in selected:
        group_slice = slices[name]
        width = group_slice.stop - group_slice.start
        trace = float(np.trace(covariance[group_slice, group_slice]))
        effective_df[name] = float(
            np.clip(width - alpha[name] * trace, 0.0, width)
        )

    return FittedARD(
        group_order=selected,
        slices=slices,
        x_standardizers=standardizers,
        y_mean=y_mean,
        coefficient_mean=coefficient,
        posterior_covariance=covariance,
        noise_precision=beta,
        alpha=alpha,
        effective_df_per_output=effective_df,
        iterations=iteration,
        converged=converged,
    )


def _subset(panel: ResidualPanel, mask: np.ndarray) -> ResidualPanel:
    return ResidualPanel(
        y=np.asarray(panel.y)[mask],
        blocks={name: np.asarray(block)[mask] for name, block in panel.blocks.items()},
        trajectory_id=np.asarray(panel.trajectory_id)[mask],
        object_id=np.asarray(panel.object_id)[mask],
        backend_id=np.asarray(panel.backend_id)[mask],
        metadata=panel.metadata,
    )


def _rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - prediction))))


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(y - prediction)))


def _gaussian_log_score(
    y: np.ndarray, prediction: np.ndarray, variance: np.ndarray
) -> float:
    variance = np.asarray(variance, dtype=float).reshape(-1, 1)
    values = 0.5 * (
        np.log(2.0 * np.pi * variance) + np.square(y - prediction) / variance
    )
    return float(np.mean(values))


def source_grouped_predictions(
    panel: ResidualPanel,
    *,
    active_groups: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-complete-trajectory-out predictions and variances."""

    panel.validate()
    trajectory = np.asarray(panel.trajectory_id).astype(str)
    unique = np.unique(trajectory)
    prediction = np.zeros_like(panel.y, dtype=float)
    variance = np.zeros(panel.y.shape[0], dtype=float)
    for held_out in unique:
        test_mask = trajectory == held_out
        train_mask = ~test_mask
        if np.sum(train_mask) < 4:
            raise ContractError(f"too few samples after holding out {held_out!r}")
        train = _subset(panel, train_mask)
        model = fit_group_ard(train)
        prediction[test_mask] = model.predict(
            {name: np.asarray(block)[test_mask] for name, block in panel.blocks.items()},
            active_groups=active_groups,
        )
        variance[test_mask] = model.predictive_variance(
            {name: np.asarray(block)[test_mask] for name, block in panel.blocks.items()}
        )
    return prediction, variance


def group_diagnostics(panel: ResidualPanel, model: FittedARD) -> dict[str, Any]:
    """Return source-only mechanism attribution diagnostics."""

    full_prediction = model.predict(panel.blocks)
    full_rmse = _rmse(panel.y, full_prediction)
    full_variance = model.predictive_variance(panel.blocks)
    full_score = _gaussian_log_score(panel.y, full_prediction, full_variance)
    diagnostics: dict[str, Any] = {}
    for name in model.group_order:
        active = [candidate for candidate in model.group_order if candidate != name]
        ablated = model.predict(panel.blocks, active_groups=active)
        group_slice = model.slices[name]
        coefficient_energy = float(
            np.sum(np.square(model.coefficient_mean[group_slice, :]))
        )
        ablated_rmse = _rmse(panel.y, ablated)
        ablated_score = _gaussian_log_score(panel.y, ablated, full_variance)
        diagnostics[name] = {
            "alpha": model.alpha[name],
            "effective_df_per_output": model.effective_df_per_output[name],
            "coefficient_energy": coefficient_energy,
            "full_fit_rmse_increase_when_removed": ablated_rmse - full_rmse,
            "negative_log_score_increase_when_removed": ablated_score - full_score,
            "source_supported": bool(
                model.effective_df_per_output[name] >= 0.10
                and ablated_rmse > full_rmse + 1.0e-12
            ),
        }
    return diagnostics


def bootstrap_diagnosis_frequency(
    panel: ResidualPanel,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap complete trajectories and count supported diagnoses."""

    if repetitions < 1:
        raise ContractError("bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    trajectory = np.asarray(panel.trajectory_id).astype(str)
    unique = np.unique(trajectory)
    counts = {name: 0 for name in panel.blocks}
    successful = 0
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices: list[int] = []
        synthetic_ids: list[str] = []
        for copy_index, value in enumerate(sampled):
            selected = np.flatnonzero(trajectory == value)
            indices.extend(selected.tolist())
            synthetic_ids.extend([f"boot-{copy_index}:{value}"] * len(selected))
        mask_indices = np.asarray(indices, dtype=int)
        boot = ResidualPanel(
            y=np.asarray(panel.y)[mask_indices],
            blocks={
                name: np.asarray(block)[mask_indices]
                for name, block in panel.blocks.items()
            },
            trajectory_id=np.asarray(synthetic_ids),
            object_id=np.asarray(panel.object_id)[mask_indices],
            backend_id=np.asarray(panel.backend_id)[mask_indices],
            metadata=panel.metadata,
        )
        try:
            model = fit_group_ard(boot)
            diagnostic = group_diagnostics(boot, model)
        except (ContractError, np.linalg.LinAlgError, FloatingPointError):
            continue
        successful += 1
        for name, value in diagnostic.items():
            counts[name] += int(value["source_supported"])
    if successful == 0:
        raise ContractError("all trajectory-bootstrap fits failed")
    return {name: counts[name] / successful for name in counts}


def _per_trajectory_metrics(
    panel: ResidualPanel,
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    trajectory = np.asarray(panel.trajectory_id).astype(str)
    rows: list[dict[str, Any]] = []
    for value in np.unique(trajectory):
        mask = trajectory == value
        row: dict[str, Any] = {
            "trajectory_id": value,
            "object_id": str(np.asarray(panel.object_id)[mask][0]),
            "backend_id": str(np.asarray(panel.backend_id)[mask][0]),
        }
        for name, prediction in predictions.items():
            row[f"{name}_rmse"] = _rmse(panel.y[mask], prediction[mask])
            row[f"{name}_mae"] = _mae(panel.y[mask], prediction[mask])
        rows.append(row)
    return rows


def evaluate_source_panel(
    panel: ResidualPanel,
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Run the registered source-only diagnosis and transferability checks."""

    panel.validate()
    model = fit_group_ard(panel)
    diagnostics = group_diagnostics(panel, model)
    bootstrap_frequency = bootstrap_diagnosis_frequency(
        panel, repetitions=bootstrap_repetitions, seed=seed
    )

    transferable = [
        name
        for name in model.group_order
        if bool(
            panel.metadata.get("transfer_eligibility", {}).get(
                name, DEFAULT_TRANSFER_ELIGIBILITY[name]
            )
        )
        and diagnostics[name]["source_supported"]
        and bootstrap_frequency.get(name, 0.0) >= float(
            panel.metadata.get("minimum_bootstrap_diagnosis_frequency", 0.70)
        )
    ]
    if not transferable:
        transferable = []

    all_groups = list(model.group_order)
    nontransferable_supported = [
        name
        for name in all_groups
        if name not in transferable and diagnostics[name]["source_supported"]
    ]
    wrong_group = max(
        nontransferable_supported,
        key=lambda name: diagnostics[name]["negative_log_score_increase_when_removed"],
        default=None,
    )

    predictions: dict[str, np.ndarray] = {
        "physical": np.zeros_like(panel.y),
    }
    variances: dict[str, np.ndarray] = {}
    for arm, active in (
        ("shared_physics", ["shared_physics"]),
        ("all_components", all_groups),
        ("diagnosis_guided", transferable),
        ("wrong_diagnosis", [] if wrong_group is None else [wrong_group]),
    ):
        if active:
            prediction, variance = source_grouped_predictions(
                panel, active_groups=active
            )
        else:
            prediction = np.zeros_like(panel.y)
            variance = np.full(panel.y.shape[0], np.var(panel.y) + EPS)
        predictions[arm] = prediction
        variances[arm] = variance

    aggregate = {}
    physical_rmse = _rmse(panel.y, predictions["physical"])
    for name, prediction in predictions.items():
        rmse = _rmse(panel.y, prediction)
        aggregate[name] = {
            "rmse": rmse,
            "mae": _mae(panel.y, prediction),
            "relative_improvement_vs_physical": (
                (physical_rmse - rmse) / max(physical_rmse, EPS)
            ),
        }
        if name in variances:
            aggregate[name]["negative_log_score"] = _gaussian_log_score(
                panel.y, prediction, variances[name]
            )

    per_trajectory = _per_trajectory_metrics(panel, predictions)
    fallback_hash = hashlib.sha256(
        np.ascontiguousarray(predictions["physical"]).view(np.uint8)
    ).hexdigest()
    rejected_hash = hashlib.sha256(
        np.ascontiguousarray(np.zeros_like(panel.y)).view(np.uint8)
    ).hexdigest()

    gate_cfg = panel.metadata.get("source_gate", {})
    minimum_improvement = float(
        gate_cfg.get("shared_vs_physical_min_relative_improvement", 0.01)
    )
    minimum_frequency = float(
        gate_cfg.get("minimum_source_bootstrap_shared_diagnosis_frequency", 0.70)
    )
    trajectory_wins = sum(
        row["diagnosis_guided_rmse"] < row["physical_rmse"]
        for row in per_trajectory
    )
    worst_ratio = max(
        row["diagnosis_guided_rmse"] / max(row["physical_rmse"], EPS)
        for row in per_trajectory
    )
    gate = {
        "shared_improvement_pass": bool(
            aggregate["shared_physics"]["relative_improvement_vs_physical"]
            >= minimum_improvement
        ),
        "shared_bootstrap_frequency_pass": bool(
            bootstrap_frequency.get("shared_physics", 0.0) >= minimum_frequency
        ),
        "trajectory_win_fraction": trajectory_wins / len(per_trajectory),
        "trajectory_win_fraction_pass": bool(
            trajectory_wins / len(per_trajectory)
            >= float(gate_cfg.get("minimum_complete_trajectory_win_fraction", 0.60))
        ),
        "worst_trajectory_ratio": worst_ratio,
        "worst_trajectory_pass": bool(
            worst_ratio
            <= float(gate_cfg.get("maximum_worst_trajectory_ratio_vs_physical", 1.10))
        ),
        "fallback_identity_violations": int(fallback_hash != rejected_hash),
    }
    gate["passed"] = bool(
        all(
            gate[key]
            for key in (
                "shared_improvement_pass",
                "shared_bootstrap_frequency_pass",
                "trajectory_win_fraction_pass",
                "worst_trajectory_pass",
            )
        )
        and gate["fallback_identity_violations"] == 0
    )

    return {
        "schema": "bayesian-phystwin.hierarchical-missing-physics-diagnosis-result",
        "schema_version": 1,
        "panel_metadata": dict(panel.metadata),
        "sample_count": int(panel.y.shape[0]),
        "output_dimensions": int(panel.y.shape[1]),
        "trajectory_count": int(len(np.unique(panel.trajectory_id.astype(str)))),
        "object_count": int(len(np.unique(panel.object_id.astype(str)))),
        "backend_count": int(len(np.unique(panel.backend_id.astype(str)))),
        "model": {
            "groups": list(model.group_order),
            "iterations": model.iterations,
            "converged": model.converged,
            "noise_precision": model.noise_precision,
        },
        "group_diagnostics": diagnostics,
        "bootstrap_diagnosis_frequency": bootstrap_frequency,
        "transferable_groups_selected_source_only": transferable,
        "wrong_diagnosis_group": wrong_group,
        "aggregate": aggregate,
        "per_trajectory": per_trajectory,
        "source_gate": gate,
        "fallback_sha256": fallback_hash,
        "information_boundary": {
            "target_outcomes_read": False,
            "target_group_selection": False,
            "target_hyperparameter_selection": False,
            "frames_treated_as_independent_for_claim": False,
        },
    }


def load_panel(npz_path: Path, manifest_path: Path | None = None) -> ResidualPanel:
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"y", "trajectory_id", "object_id", "backend_id"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ContractError(f"NPZ lacks required arrays: {missing}")
        blocks = {
            key.removeprefix("block__"): np.asarray(archive[key], dtype=float)
            for key in archive.files
            if key.startswith("block__")
        }
        metadata: Mapping[str, Any] = {}
        if manifest_path is not None:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        panel = ResidualPanel(
            y=np.asarray(archive["y"], dtype=float),
            blocks=blocks,
            trajectory_id=np.asarray(archive["trajectory_id"]).astype(str),
            object_id=np.asarray(archive["object_id"]).astype(str),
            backend_id=np.asarray(archive["backend_id"]).astype(str),
            metadata=metadata,
        )
    panel.validate()
    return panel


def make_synthetic_panel(seed: int = 7) -> ResidualPanel:
    """Create a source-only falsification panel with known group structure."""

    rng = np.random.default_rng(seed)
    trajectories = []
    objects = []
    backends = []
    block_rows = {name: [] for name in REGISTERED_GROUPS}
    y_rows = []
    samples_per_trajectory = 24
    for object_index, object_name in enumerate(("DLO2", "DLO3", "DLO6")):
        for backend_index, backend_name in enumerate(("deform", "alternate")):
            for trajectory_index in range(4):
                trajectory_name = (
                    f"{object_name}:{backend_name}:trajectory-{trajectory_index}"
                )
                phase = rng.uniform(-np.pi, np.pi)
                for sample_index in range(samples_per_trajectory):
                    t = sample_index / (samples_per_trajectory - 1)
                    shared = np.array(
                        [
                            math.sin(2.0 * np.pi * t + phase),
                            math.cos(2.0 * np.pi * t + phase),
                            t - 0.5,
                        ]
                    )
                    object_block = np.zeros(3)
                    object_block[object_index] = shared[0]
                    backend_block = np.zeros(2)
                    backend_block[backend_index] = shared[1]
                    contact = np.array([float(t > 0.55), max(t - 0.55, 0.0)])
                    actuation = np.array([math.sin(np.pi * t), t])
                    sensor = np.array([1.0, (-1.0) ** sample_index])
                    signal = np.array(
                        [
                            0.75 * shared[0] - 0.35 * shared[1] + 0.20 * shared[2],
                            -0.55 * shared[0] + 0.25 * shared[1],
                        ]
                    )
                    signal += np.array(
                        [0.12 * object_block[object_index], -0.08 * object_block[object_index]]
                    )
                    signal += np.array(
                        [0.10 * backend_block[backend_index], 0.07 * backend_block[backend_index]]
                    )
                    signal += np.array([0.05 * contact[0], -0.04 * contact[1]])
                    signal += rng.normal(scale=0.08, size=2)
                    y_rows.append(signal)
                    trajectories.append(trajectory_name)
                    objects.append(object_name)
                    backends.append(backend_name)
                    for name, value in (
                        ("shared_physics", shared),
                        ("object", object_block),
                        ("backend", backend_block),
                        ("contact", contact),
                        ("actuation", actuation),
                        ("sensor", sensor),
                    ):
                        block_rows[name].append(value)
    return ResidualPanel(
        y=np.asarray(y_rows),
        blocks={name: np.asarray(rows) for name, rows in block_rows.items()},
        trajectory_id=np.asarray(trajectories),
        object_id=np.asarray(objects),
        backend_id=np.asarray(backends),
        metadata={
            "panel_id": "synthetic-mechanism-falsification-v1",
            "transfer_eligibility": DEFAULT_TRANSFER_ELIGIBILITY,
            "minimum_bootstrap_diagnosis_frequency": 0.60,
            "source_gate": {
                "shared_vs_physical_min_relative_improvement": 0.20,
                "minimum_source_bootstrap_shared_diagnosis_frequency": 0.60,
                "minimum_complete_trajectory_win_fraction": 0.80,
                "maximum_worst_trajectory_ratio_vs_physical": 0.95,
            },
        },
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--synthetic-smoke", action="store_true")
    arguments = parser.parse_args()

    if arguments.synthetic_smoke:
        if arguments.panel is not None or arguments.manifest is not None:
            raise SystemExit("--synthetic-smoke cannot be combined with panel inputs")
        panel = make_synthetic_panel(arguments.seed)
    else:
        if arguments.panel is None:
            raise SystemExit("--panel is required unless --synthetic-smoke is used")
        panel = load_panel(arguments.panel, arguments.manifest)

    result = evaluate_source_panel(
        panel,
        bootstrap_repetitions=arguments.bootstrap,
        seed=arguments.seed,
    )
    result["result_id"] = _canonical_hash(result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "source_gate": result["source_gate"],
                "transferable_groups_selected_source_only": result[
                    "transferable_groups_selected_source_only"
                ],
                "bootstrap_diagnosis_frequency": result[
                    "bootstrap_diagnosis_frequency"
                ],
                "aggregate": result["aggregate"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
