"""Combine per-interaction PhysTwin grids under shared object stiffness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis) if axis is not None else result.reshape(())


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    normalized = np.asarray(log_weights, dtype=float) - float(
        _logsumexp(np.asarray(log_weights, dtype=float))
    )
    return np.exp(normalized)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, probability, side="left")]])


def _grid_summary(
    object_grid: np.ndarray,
    controller_grid: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    object_weights = np.sum(weights, axis=1)
    controller_weights = np.sum(weights, axis=0)
    object_mean = float(np.sum(object_weights * object_grid))
    controller_mean = float(np.sum(controller_weights * controller_grid))
    return {
        "effective_grid_points": float(1.0 / np.sum(np.square(weights))),
        "object_log_scale": {
            "mean": object_mean,
            "std": float(
                np.sqrt(np.sum(object_weights * np.square(object_grid - object_mean)))
            ),
            "q05": _weighted_quantile(object_grid, object_weights, 0.05),
            "q95": _weighted_quantile(object_grid, object_weights, 0.95),
        },
        "controller_log_scale": {
            "mean": controller_mean,
            "std": float(
                np.sqrt(
                    np.sum(
                        controller_weights
                        * np.square(controller_grid - controller_mean)
                    )
                )
            ),
            "q05": _weighted_quantile(controller_grid, controller_weights, 0.05),
            "q95": _weighted_quantile(controller_grid, controller_weights, 0.95),
        },
    }


def combine_joint_profile_files(
    profile_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    object_prior_std: float = 0.15,
    controller_prior_std: float = 0.50,
    likelihood_temperatures: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Share one object-scale correction and marginalize trial controllers."""

    if len(profile_paths) < 2:
        raise ValueError("joint profiling requires at least two interactions")
    if object_prior_std <= 0.0 or controller_prior_std <= 0.0:
        raise ValueError("prior standard deviations must be positive")
    temperatures = dict(likelihood_temperatures or {})
    unknown_temperatures = set(temperatures) - set(profile_paths)
    if unknown_temperatures:
        raise ValueError(
            "temperatures contain unknown cases: "
            + ", ".join(sorted(unknown_temperatures))
        )

    loaded: dict[str, dict[str, np.ndarray | float | str]] = {}
    reference_object_grid = None
    reference_controller_grid = None
    for case_name, raw_path in profile_paths.items():
        path = Path(raw_path)
        with np.load(path) as archive:
            required = {"object_log_scales", "controller_log_scales", "log_likelihood"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"{case_name} profile is missing: {', '.join(sorted(missing))}"
                )
            object_grid = np.asarray(archive["object_log_scales"], dtype=float)
            controller_grid = np.asarray(archive["controller_log_scales"], dtype=float)
            log_likelihood = np.asarray(archive["log_likelihood"], dtype=float)
        if log_likelihood.shape != (len(object_grid), len(controller_grid)):
            raise ValueError(f"{case_name} likelihood shape does not match its grids")
        if not np.all(np.isfinite(log_likelihood)):
            raise ValueError(f"{case_name} likelihood contains non-finite values")
        if reference_object_grid is None:
            reference_object_grid = object_grid
            reference_controller_grid = controller_grid
        elif not (
            np.array_equal(reference_object_grid, object_grid)
            and np.array_equal(reference_controller_grid, controller_grid)
        ):
            raise ValueError("all interactions must use identical profile grids")
        temperature = float(temperatures.get(case_name, 1.0))
        if temperature <= 0.0:
            raise ValueError("likelihood temperatures must be positive")
        loaded[case_name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "log_likelihood": log_likelihood,
            "temperature": temperature,
        }

    assert reference_object_grid is not None
    assert reference_controller_grid is not None
    object_grid = reference_object_grid
    controller_grid = reference_controller_grid
    object_log_prior = -0.5 * np.square(object_grid / object_prior_std)
    controller_log_prior = -0.5 * np.square(
        controller_grid / controller_prior_std
    )
    case_conditionals: dict[str, np.ndarray] = {}
    case_object_evidence: dict[str, np.ndarray] = {}
    for case_name, values in loaded.items():
        likelihood = np.asarray(values["log_likelihood"], dtype=float)
        conditional = (
            likelihood / float(values["temperature"])
            + controller_log_prior[None, :]
        )
        case_conditionals[case_name] = conditional
        case_object_evidence[case_name] = _logsumexp(conditional, axis=1)

    shared_object_log_weight = object_log_prior.copy()
    for evidence in case_object_evidence.values():
        shared_object_log_weight += evidence
    shared_object_weights = _normalize_log_weights(shared_object_log_weight)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    case_summaries: dict[str, object] = {}
    output_paths: dict[str, str] = {}
    for case_name, conditional in case_conditionals.items():
        other_evidence = np.zeros_like(object_grid)
        for other_name, evidence in case_object_evidence.items():
            if other_name != case_name:
                other_evidence += evidence
        joint_log_weight = (
            object_log_prior[:, None]
            + conditional
            + other_evidence[:, None]
        )
        posterior_weights = _normalize_log_weights(joint_log_weight)
        path = output / f"{case_name}.npz"
        np.savez_compressed(
            path,
            object_log_scales=object_grid,
            controller_log_scales=controller_grid,
            log_likelihood=np.asarray(loaded[case_name]["log_likelihood"]),
            posterior_weights=posterior_weights,
            shared_object_weights=shared_object_weights,
            likelihood_temperature=np.asarray(loaded[case_name]["temperature"]),
        )
        output_paths[case_name] = str(path.resolve())
        case_summaries[case_name] = _grid_summary(
            object_grid,
            controller_grid,
            posterior_weights,
        )

    object_mean = float(np.sum(shared_object_weights * object_grid))
    summary = {
        "schema_version": 1,
        "contract": (
            "one shared relative object-spring log-scale correction; "
            "independent trial-specific controller log scales"
        ),
        "object_prior_std": object_prior_std,
        "controller_prior_std": controller_prior_std,
        "inputs": {
            case_name: {
                "path": values["path"],
                "sha256": values["sha256"],
                "likelihood_temperature": values["temperature"],
            }
            for case_name, values in loaded.items()
        },
        "shared_object_log_scale": {
            "mean": object_mean,
            "std": float(
                np.sqrt(
                    np.sum(
                        shared_object_weights * np.square(object_grid - object_mean)
                    )
                )
            ),
            "q05": _weighted_quantile(object_grid, shared_object_weights, 0.05),
            "q95": _weighted_quantile(object_grid, shared_object_weights, 0.95),
            "effective_grid_points": float(
                1.0 / np.sum(np.square(shared_object_weights))
            ),
        },
        "cases": case_summaries,
        "outputs": output_paths,
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
