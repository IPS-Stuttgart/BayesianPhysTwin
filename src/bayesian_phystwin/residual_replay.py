"""Replay exported PhysTwin residuals through reliability-aware likelihoods."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calibration import binary_calibration_metrics
from .pseudo_measurements import (
    PseudoMeasurementBatch,
    ReliabilityConfig,
    measurement_variance,
    reliability_weighted_loss,
    score_reliability,
)
from .robust_likelihood import (
    RobustLikelihoodConfig,
    robust_mixture_likelihood,
)


SCORE_COLUMNS = (
    "residual_norm",
    "prior_reliability",
    "posterior_inlier_probability",
    "inflated_variance_mean",
    "robust_negative_log_likelihood",
)


@dataclass(frozen=True)
class ResidualColumnSchema:
    observed: tuple[str, ...]
    predicted: tuple[str, ...]
    variance: tuple[str, ...]
    variance_mode: str
    inlier_label: str | None


@dataclass(frozen=True)
class ResidualReplayResult:
    """Serializable summary and row-level scores from one residual export."""

    summary: dict[str, Any]
    input_fieldnames: tuple[str, ...]
    scored_rows: tuple[dict[str, str], ...]

    def write_summary_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_scored_csv(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.input_fieldnames)
        fieldnames.extend(name for name in SCORE_COLUMNS if name not in fieldnames)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.scored_rows)


@dataclass(frozen=True)
class _LoadedResiduals:
    batch: PseudoMeasurementBatch
    rows: tuple[dict[str, str], ...]
    fieldnames: tuple[str, ...]
    schema: ResidualColumnSchema
    inlier_target: np.ndarray | None


def _sort_suffixes(suffixes: Sequence[str]) -> list[str]:
    axis_order = {"x": 0, "y": 1, "z": 2, "u": 0, "v": 1, "w": 2}

    def key(value: str) -> tuple[int, int | str]:
        if value in axis_order:
            return 0, axis_order[value]
        if value.isdigit():
            return 1, int(value)
        return 2, value

    return sorted(suffixes, key=key)


def _detect_vector_columns(fieldnames: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    for observed_prefix, predicted_prefix in (
        ("observed_", "predicted_"),
        ("obs_", "pred_"),
    ):
        suffixes = [
            name[len(observed_prefix) :]
            for name in fieldnames
            if name.startswith(observed_prefix)
            and name != observed_prefix
            and predicted_prefix + name[len(observed_prefix) :] in fieldnames
        ]
        if suffixes:
            ordered = _sort_suffixes(suffixes)
            return (
                [observed_prefix + suffix for suffix in ordered],
                [predicted_prefix + suffix for suffix in ordered],
                ordered,
            )
    raise ValueError(
        "could not detect residual vectors; expected matched observed_*/predicted_* "
        "or obs_*/pred_* columns"
    )


def _parse_float(row: dict[str, str], column: str, row_number: int) -> float:
    value = row.get(column, "")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: {column} must be numeric, got {value!r}") from error
    if not np.isfinite(parsed):
        raise ValueError(f"row {row_number}: {column} must be finite")
    return parsed


def _parse_optional_float(
    row: dict[str, str],
    column: str,
    row_number: int,
    default: float,
) -> float:
    if not row.get(column, "").strip():
        return default
    return _parse_float(row, column, row_number)


def _parse_bool(value: str, *, row_number: int, column: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(
        f"row {row_number}: {column} must be a boolean (0/1 or false/true), got {value!r}"
    )


def _load_residual_csv(path: str | Path, *, default_variance: float) -> _LoadedResiduals:
    input_path = Path(path)
    if default_variance <= 0.0:
        raise ValueError("default_variance must be positive")

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("residual CSV must have a header")
        fieldnames = tuple(reader.fieldnames)
        rows = tuple(dict(row) for row in reader)

    if not rows:
        raise ValueError("residual CSV must contain at least one measurement")

    observed_columns, predicted_columns, suffixes = _detect_vector_columns(fieldnames)
    if "variance" in fieldnames:
        variance_columns = ["variance"]
        variance_mode = "scalar"
    elif all(f"variance_{suffix}" in fieldnames for suffix in suffixes):
        variance_columns = [f"variance_{suffix}" for suffix in suffixes]
        variance_mode = "coordinate"
    elif all(f"var_{suffix}" in fieldnames for suffix in suffixes):
        variance_columns = [f"var_{suffix}" for suffix in suffixes]
        variance_mode = "coordinate"
    else:
        variance_columns = []
        variance_mode = "default"

    observed: list[list[float]] = []
    predicted: list[list[float]] = []
    variance: list[float] | list[list[float]] = []
    confidence: list[float] | None = [] if "confidence" in fieldnames else None
    occluded: list[bool] | None = [] if "occluded" in fieldnames else None
    boundary: list[float] | None = [] if "boundary_distance" in fieldnames else None
    flow: list[float] | None = [] if "flow_inconsistency" in fieldnames else None

    label_column: str | None = None
    invert_label = False
    if "is_inlier" in fieldnames:
        label_column = "is_inlier"
    elif "is_corrupted" in fieldnames:
        label_column = "is_corrupted"
        invert_label = True
    targets: list[bool] | None = [] if label_column else None

    for row_number, row in enumerate(rows, start=2):
        observed.append([_parse_float(row, column, row_number) for column in observed_columns])
        predicted.append([_parse_float(row, column, row_number) for column in predicted_columns])
        if variance_mode == "scalar":
            variance.append(_parse_float(row, variance_columns[0], row_number))
        elif variance_mode == "coordinate":
            variance.append([_parse_float(row, column, row_number) for column in variance_columns])

        if confidence is not None:
            confidence.append(_parse_optional_float(row, "confidence", row_number, 1.0))
        if occluded is not None:
            raw_occluded = row.get("occluded", "").strip()
            occluded.append(
                False
                if not raw_occluded
                else _parse_bool(raw_occluded, row_number=row_number, column="occluded")
            )
        if boundary is not None:
            boundary.append(
                _parse_optional_float(row, "boundary_distance", row_number, np.inf)
            )
        if flow is not None:
            flow.append(_parse_optional_float(row, "flow_inconsistency", row_number, 0.0))
        if targets is not None and label_column is not None:
            target = _parse_bool(
                row.get(label_column, ""),
                row_number=row_number,
                column=label_column,
            )
            targets.append(not target if invert_label else target)

    batch_variance: float | list[float] | list[list[float]]
    batch_variance = default_variance if variance_mode == "default" else variance
    batch = PseudoMeasurementBatch(
        observed=observed,
        predicted=predicted,
        variance=batch_variance,
        confidence=confidence,
        occluded=occluded,
        boundary_distance=boundary,
        flow_inconsistency=flow,
    )
    schema = ResidualColumnSchema(
        observed=tuple(observed_columns),
        predicted=tuple(predicted_columns),
        variance=tuple(variance_columns),
        variance_mode=variance_mode,
        inlier_label=label_column,
    )
    return _LoadedResiduals(
        batch=batch,
        rows=rows,
        fieldnames=fieldnames,
        schema=schema,
        inlier_target=None if targets is None else np.asarray(targets, dtype=bool),
    )


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.sum(np.square(weights)))
    if denominator == 0.0:
        return 0.0
    total = float(np.sum(weights))
    return total * total / denominator


def _per_frame_summary(
    rows: Sequence[dict[str, str]],
    residual_norm: np.ndarray,
    prior: np.ndarray,
    posterior: np.ndarray,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["frame"], []).append(index)

    summaries: list[dict[str, Any]] = []
    for frame, indexes in groups.items():
        selected = np.asarray(indexes, dtype=int)
        summaries.append(
            {
                "frame": frame,
                "count": int(selected.size),
                "mean_residual_norm": float(np.mean(residual_norm[selected])),
                "mean_prior_reliability": float(np.mean(prior[selected])),
                "mean_posterior_inlier_probability": float(np.mean(posterior[selected])),
                "prior_effective_sample_size": _effective_sample_size(prior[selected]),
                "posterior_effective_sample_size": _effective_sample_size(posterior[selected]),
            }
        )
    return summaries


def replay_residual_csv(
    path: str | Path,
    *,
    reliability_config: ReliabilityConfig | None = None,
    likelihood_config: RobustLikelihoodConfig | None = None,
    default_variance: float = 1e-4,
    calibration_bins: int = 10,
) -> ResidualReplayResult:
    """Score one canonical residual CSV and return paper-ready summaries."""

    reliability_cfg = reliability_config or ReliabilityConfig()
    likelihood_cfg = likelihood_config or RobustLikelihoodConfig()
    loaded = _load_residual_csv(path, default_variance=default_variance)
    observed, predicted = loaded.batch.arrays()
    variance = measurement_variance(loaded.batch)
    reliability = score_reliability(loaded.batch, reliability_cfg)
    likelihood = robust_mixture_likelihood(
        loaded.batch,
        prior_reliability=reliability.weights,
        config=likelihood_cfg,
    )

    residual = observed - predicted
    unweighted_mahalanobis = np.sum(np.square(residual) / variance, axis=1)
    posterior = likelihood.posterior_inlier_probability
    summary: dict[str, Any] = {
        "schema_version": 1,
        "input_csv": str(Path(path).resolve()),
        "measurement_count": int(observed.shape[0]),
        "measurement_dimension": int(observed.shape[1]),
        "columns": asdict(loaded.schema),
        "config": {
            "reliability": asdict(reliability_cfg),
            "robust_likelihood": asdict(likelihood_cfg),
            "default_variance": default_variance,
        },
        "residual_norm": _distribution_summary(reliability.residual_norm),
        "prior_reliability": _distribution_summary(reliability.weights),
        "posterior_inlier_probability": _distribution_summary(posterior),
        "effective_sample_size": {
            "prior": reliability.effective_sample_size,
            "posterior": _effective_sample_size(posterior),
        },
        "objectives": {
            "mean_unweighted_mahalanobis": float(np.mean(unweighted_mahalanobis)),
            "mean_reliability_weighted_mahalanobis": reliability_weighted_loss(
                loaded.batch,
                reliability_cfg,
            ),
            "mean_unweighted_gaussian_nll": float(np.mean(-likelihood.log_inlier_density)),
            "mean_robust_mixture_nll": likelihood.mean_negative_log_likelihood,
        },
    }

    if loaded.inlier_target is not None:
        summary["labels"] = {
            "inlier_count": int(np.sum(loaded.inlier_target)),
            "outlier_count": int(np.sum(~loaded.inlier_target)),
        }
        summary["calibration"] = {
            "prior_reliability": binary_calibration_metrics(
                reliability.weights,
                loaded.inlier_target,
                n_bins=calibration_bins,
            ).as_dict(),
            "posterior_inlier_probability": binary_calibration_metrics(
                posterior,
                loaded.inlier_target,
                n_bins=calibration_bins,
            ).as_dict(),
        }

    if "frame" in loaded.fieldnames:
        summary["per_frame"] = _per_frame_summary(
            loaded.rows,
            reliability.residual_norm,
            reliability.weights,
            posterior,
        )

    scored_rows: list[dict[str, str]] = []
    for index, row in enumerate(loaded.rows):
        scored = dict(row)
        scored.update(
            {
                "residual_norm": f"{reliability.residual_norm[index]:.12g}",
                "prior_reliability": f"{reliability.weights[index]:.12g}",
                "posterior_inlier_probability": f"{posterior[index]:.12g}",
                "inflated_variance_mean": (
                    f"{np.mean(reliability.inflated_variance[index]):.12g}"
                ),
                "robust_negative_log_likelihood": (
                    f"{likelihood.negative_log_likelihood[index]:.12g}"
                ),
            }
        )
        scored_rows.append(scored)

    return ResidualReplayResult(
        summary=summary,
        input_fieldnames=loaded.fieldnames,
        scored_rows=tuple(scored_rows),
    )
