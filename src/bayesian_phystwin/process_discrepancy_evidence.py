"""Evaluation and source-frozen evidence for process discrepancy candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .process_discrepancy import (
    PROCESS_DISCREPANCY_SCHEMA_VERSION,
    DynamicsConsistentForceBasis,
    StableLatentForceProcess,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_data(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _trajectory_metrics(
    reference_m: np.ndarray,
    candidate_m: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    residual = candidate_m - reference_m
    selected = valid & np.all(np.isfinite(residual), axis=2)
    _require(np.any(selected), "trajectory comparison has no valid entries")
    vectors = residual[selected]
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(vectors)))),
        "vector_rmse_m": float(np.sqrt(np.mean(np.square(norms)))),
        "mean_vector_error_m": float(np.mean(norms)),
        "maximum_vector_error_m": float(np.max(norms, initial=0.0)),
    }


def compare_process_discrepancy_rollouts(
    reference_m: np.ndarray,
    baseline_m: np.ndarray,
    readout_only_m: np.ndarray,
    process_discrepancy_m: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare a process candidate against both required controls."""

    reference = np.asarray(reference_m, dtype=np.float64)
    baseline = np.asarray(baseline_m, dtype=np.float64)
    readout = np.asarray(readout_only_m, dtype=np.float64)
    process = np.asarray(process_discrepancy_m, dtype=np.float64)
    _require(
        reference.ndim == 3
        and reference.shape[2] == 3
        and baseline.shape == readout.shape == process.shape == reference.shape,
        "all trajectories must have matching shape (frame, node, 3)",
    )
    if valid is None:
        support = np.ones(reference.shape[:2], dtype=bool)
    else:
        support = np.asarray(valid, dtype=bool).copy()
        _require(
            support.shape == reference.shape[:2],
            "valid mask has the wrong shape",
        )
    support &= np.all(np.isfinite(reference), axis=2)
    _require(np.any(support), "trajectory comparison has no valid reference entries")
    for name, candidate in (
        ("baseline", baseline),
        ("readout_only", readout),
        ("process_discrepancy", process),
    ):
        _require(
            np.all(np.isfinite(candidate[support])),
            f"{name} contains non-finite values on the common support",
        )
    metrics = {
        "baseline": _trajectory_metrics(reference, baseline, support),
        "readout_only": _trajectory_metrics(reference, readout, support),
        "process_discrepancy": _trajectory_metrics(reference, process, support),
    }
    process_rmse = metrics["process_discrepancy"]["coordinate_rmse_m"]
    baseline_rmse = metrics["baseline"]["coordinate_rmse_m"]
    readout_rmse = metrics["readout_only"]["coordinate_rmse_m"]
    metrics["comparisons"] = {
        "process_minus_baseline_coordinate_rmse_m": process_rmse - baseline_rmse,
        "process_minus_readout_coordinate_rmse_m": process_rmse - readout_rmse,
        "process_reduction_vs_baseline_fraction": (
            1.0 - process_rmse / baseline_rmse if baseline_rmse > 0.0 else 0.0
        ),
        "process_reduction_vs_readout_fraction": (
            1.0 - process_rmse / readout_rmse if readout_rmse > 0.0 else 0.0
        ),
    }
    return metrics


def _validated_candidate_configuration(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = _json_data(value, name="candidate_configuration")
    _require(
        configuration.get("schema_version") == PROCESS_DISCREPANCY_SCHEMA_VERSION,
        "candidate configuration schema version changed",
    )
    _require(
        configuration.get("artifact_kind")
        == "DynamicsConsistentProcessCandidateV1",
        "candidate configuration artifact kind changed",
    )
    for name in ("force_basis_id", "process_id", "response_model_id"):
        _require(
            _is_sha256(configuration.get(name)),
            f"candidate configuration {name} must be a SHA-256 digest",
        )
    coefficient_count = configuration.get("force_coefficient_count")
    _require(
        isinstance(coefficient_count, int)
        and not isinstance(coefficient_count, bool)
        and coefficient_count >= 1,
        "candidate force_coefficient_count must be positive",
    )
    _require(
        configuration.get("contact_policy")
        in {"all_supported", "contact_only", "exclude_contact"},
        "candidate contact policy is unsupported",
    )
    for name in ("enforce_zero_net_force", "enforce_zero_net_torque"):
        _require(
            isinstance(configuration.get(name), bool),
            f"candidate {name} must be boolean",
        )
    for name in ("work_precision_per_watt2", "coefficient_precision_per_n2"):
        try:
            precision = float(configuration[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"candidate {name} must be numeric") from error
        _require(
            precision >= 0.0 and np.isfinite(precision),
            f"candidate {name} must be finite and nonnegative",
        )
    _require(
        bool(configuration.get("force_schedule_policy")),
        "candidate force_schedule_policy must be nonempty",
    )
    _require(
        isinstance(configuration.get("metadata"), dict),
        "candidate metadata must be a JSON object",
    )
    return configuration


def build_process_discrepancy_candidate_configuration(
    force_basis: DynamicsConsistentForceBasis,
    process: StableLatentForceProcess,
    *,
    response_model_id: str,
    work_precision_per_watt2: float,
    coefficient_precision_per_n2: float,
    force_schedule_policy: str = "posterior_mean",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the complete source-frozen configuration for one candidate."""

    _require(
        force_basis.coefficient_count == process.coefficient_count,
        "force basis and process dimensions differ",
    )
    _require(
        _is_sha256(response_model_id),
        "response_model_id must be a SHA-256 digest",
    )
    _require(
        work_precision_per_watt2 >= 0.0
        and np.isfinite(work_precision_per_watt2),
        "work precision must be finite and nonnegative",
    )
    _require(
        coefficient_precision_per_n2 >= 0.0
        and np.isfinite(coefficient_precision_per_n2),
        "coefficient precision must be finite and nonnegative",
    )
    _require(bool(force_schedule_policy), "force_schedule_policy must be nonempty")
    return _validated_candidate_configuration(
        {
            "schema_version": PROCESS_DISCREPANCY_SCHEMA_VERSION,
            "artifact_kind": "DynamicsConsistentProcessCandidateV1",
            "force_basis_id": force_basis.basis_id,
            "process_id": process.process_id,
            "response_model_id": response_model_id,
            "force_coefficient_count": force_basis.coefficient_count,
            "contact_policy": force_basis.contact_policy,
            "enforce_zero_net_force": force_basis.enforce_zero_net_force,
            "enforce_zero_net_torque": force_basis.enforce_zero_net_torque,
            "work_precision_per_watt2": float(work_precision_per_watt2),
            "coefficient_precision_per_n2": float(
                coefficient_precision_per_n2
            ),
            "force_schedule_policy": str(force_schedule_policy),
            "metadata": _json_data(
                {} if metadata is None else metadata,
                name="metadata",
            ),
        }
    )


def process_discrepancy_candidate_id(
    candidate_configuration: Mapping[str, Any],
) -> str:
    """Return the content address of a complete process candidate."""

    configuration = _validated_candidate_configuration(
        candidate_configuration
    )
    return hashlib.sha256(
        json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SourceFrozenProcessSelectionV1:
    """Content-addressed selection made without held-out target outcomes."""

    candidate_id: str
    candidate_configuration: Mapping[str, Any]
    selected: bool
    reason: str
    source_case_ids: tuple[str, ...]
    held_out_case_ids: tuple[str, ...]
    source_checksums: Mapping[str, str]
    source_metrics: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    target_outcomes_used_for_selection: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        configuration = _json_data(
            self.candidate_configuration,
            name="candidate_configuration",
        )
        _require(
            self.candidate_id == process_discrepancy_candidate_id(configuration),
            "candidate_id does not bind candidate_configuration",
        )
        _require(bool(self.reason), "reason must be nonempty")
        source_cases = tuple(str(value) for value in self.source_case_ids)
        held_out_cases = tuple(str(value) for value in self.held_out_case_ids)
        _require(
            source_cases and held_out_cases,
            "source and held-out case sets are required",
        )
        _require(
            len(set(source_cases)) == len(source_cases)
            and len(set(held_out_cases)) == len(held_out_cases),
            "case identifiers must be unique",
        )
        _require(
            set(source_cases).isdisjoint(held_out_cases),
            "source and held-out case sets must be disjoint",
        )
        _require(
            not self.target_outcomes_used_for_selection,
            "held-out target outcomes cannot be used for process selection",
        )
        checksums = dict(self.source_checksums)
        _require(
            set(checksums) == set(source_cases)
            and all(_is_sha256(value) for value in checksums.values()),
            "source_checksums must bind every source case to a SHA-256 digest",
        )
        object.__setattr__(self, "candidate_configuration", configuration)
        object.__setattr__(self, "source_case_ids", source_cases)
        object.__setattr__(self, "held_out_case_ids", held_out_cases)
        object.__setattr__(self, "source_checksums", dict(sorted(checksums.items())))
        source_metrics = _json_data(
            self.source_metrics,
            name="source_metrics",
        )
        thresholds = _json_data(self.thresholds, name="thresholds")
        try:
            source_case_count = int(source_metrics["source_case_count"])
            improvement_vs_baseline = float(
                source_metrics["mean_improvement_vs_baseline_fraction"]
            )
            improvement_vs_readout = float(
                source_metrics["mean_improvement_vs_readout_fraction"]
            )
            maximum_regression = float(
                source_metrics["maximum_case_regression_fraction"]
            )
            minimum_required = float(
                thresholds["minimum_mean_improvement_fraction"]
            )
            maximum_allowed = float(
                thresholds["maximum_case_regression_fraction"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "selection metrics or thresholds are incomplete"
            ) from error
        _require(
            source_case_count == len(source_cases),
            "source case count does not match source_case_ids",
        )
        _require(
            np.all(
                np.isfinite(
                    (
                        improvement_vs_baseline,
                        improvement_vs_readout,
                        maximum_regression,
                        minimum_required,
                        maximum_allowed,
                    )
                )
            ),
            "selection metrics and thresholds must be finite",
        )
        expected_selected = bool(
            min(improvement_vs_baseline, improvement_vs_readout)
            >= minimum_required
            and maximum_regression <= maximum_allowed
        )
        _require(
            self.selected == expected_selected,
            "selected flag is inconsistent with source metrics and thresholds",
        )
        expected_reason = (
            "source gate passed against baseline and readout-only controls"
            if expected_selected
            else "source gate rejected the process candidate"
        )
        _require(self.reason == expected_reason, "selection reason is inconsistent")
        object.__setattr__(self, "source_metrics", source_metrics)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(
            self,
            "metadata",
            _json_data(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROCESS_DISCREPANCY_SCHEMA_VERSION,
            "artifact_kind": "SourceFrozenProcessSelectionV1",
            "candidate_id": self.candidate_id,
            "candidate_configuration": self.candidate_configuration,
            "selected": self.selected,
            "reason": self.reason,
            "source_case_ids": list(self.source_case_ids),
            "held_out_case_ids": list(self.held_out_case_ids),
            "source_checksums": self.source_checksums,
            "source_metrics": self.source_metrics,
            "thresholds": self.thresholds,
            "target_outcomes_used_for_selection": False,
            "metadata": self.metadata,
        }

    @property
    def selection_id(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


def select_source_frozen_process_candidate(
    candidate_configuration: Mapping[str, Any],
    source_case_summaries: Mapping[str, Mapping[str, Any]],
    *,
    held_out_case_ids: tuple[str, ...],
    source_checksums: Mapping[str, str],
    minimum_mean_improvement_fraction: float = 0.0,
    maximum_case_regression_fraction: float = 0.0,
    target_outcomes_used_for_selection: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> SourceFrozenProcessSelectionV1:
    """Select a fully bound process candidate using source comparisons only."""

    configuration = _validated_candidate_configuration(
        candidate_configuration
    )
    candidate_id = process_discrepancy_candidate_id(configuration)
    _require(
        0.0 <= minimum_mean_improvement_fraction < 1.0
        and np.isfinite(minimum_mean_improvement_fraction),
        "minimum mean improvement must lie in [0, 1)",
    )
    _require(
        maximum_case_regression_fraction >= 0.0
        and np.isfinite(maximum_case_regression_fraction),
        "maximum case regression must be finite and nonnegative",
    )
    _require(source_case_summaries, "at least one source comparison is required")
    source_cases = tuple(sorted(source_case_summaries))
    baseline_values = []
    readout_values = []
    process_values = []
    maximum_regression = -np.inf
    for case_id in source_cases:
        summary = source_case_summaries[case_id]
        try:
            baseline = float(summary["baseline"]["coordinate_rmse_m"])
            readout = float(summary["readout_only"]["coordinate_rmse_m"])
            process = float(summary["process_discrepancy"]["coordinate_rmse_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid source comparison for {case_id}") from error
        _require(
            baseline > 0.0
            and readout > 0.0
            and process >= 0.0
            and np.all(np.isfinite((baseline, readout, process))),
            f"source comparison for {case_id} has invalid RMSE values",
        )
        baseline_values.append(baseline)
        readout_values.append(readout)
        process_values.append(process)
        maximum_regression = max(
            maximum_regression,
            process / baseline - 1.0,
            process / readout - 1.0,
        )
    baseline_mean = float(np.mean(baseline_values))
    readout_mean = float(np.mean(readout_values))
    process_mean = float(np.mean(process_values))
    improvement_vs_baseline = 1.0 - process_mean / baseline_mean
    improvement_vs_readout = 1.0 - process_mean / readout_mean
    minimum_improvement = min(improvement_vs_baseline, improvement_vs_readout)
    selected = bool(
        minimum_improvement >= minimum_mean_improvement_fraction
        and maximum_regression <= maximum_case_regression_fraction
    )
    reason = (
        "source gate passed against baseline and readout-only controls"
        if selected
        else "source gate rejected the process candidate"
    )
    source_metrics = {
        "source_case_count": len(source_cases),
        "mean_baseline_coordinate_rmse_m": baseline_mean,
        "mean_readout_coordinate_rmse_m": readout_mean,
        "mean_process_coordinate_rmse_m": process_mean,
        "mean_improvement_vs_baseline_fraction": improvement_vs_baseline,
        "mean_improvement_vs_readout_fraction": improvement_vs_readout,
        "maximum_case_regression_fraction": float(maximum_regression),
    }
    thresholds = {
        "minimum_mean_improvement_fraction": float(
            minimum_mean_improvement_fraction
        ),
        "maximum_case_regression_fraction": float(
            maximum_case_regression_fraction
        ),
    }
    return SourceFrozenProcessSelectionV1(
        candidate_id=candidate_id,
        candidate_configuration=configuration,
        selected=selected,
        reason=reason,
        source_case_ids=source_cases,
        held_out_case_ids=held_out_case_ids,
        source_checksums=source_checksums,
        source_metrics=source_metrics,
        thresholds=thresholds,
        target_outcomes_used_for_selection=target_outcomes_used_for_selection,
        metadata={} if metadata is None else metadata,
    )


def write_source_frozen_process_selection(
    path: str | Path,
    selection: SourceFrozenProcessSelectionV1,
) -> dict[str, str]:
    """Write an immutable JSON selection artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {**selection.to_dict(), "selection_id": selection.selection_id}
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "selection_id": selection.selection_id,
        "path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def load_source_frozen_process_selection(
    path: str | Path,
) -> SourceFrozenProcessSelectionV1:
    """Load and revalidate a source-frozen selection artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_kind") != "SourceFrozenProcessSelectionV1":
        raise ValueError("selection artifact kind changed")
    if int(payload.get("schema_version", -1)) != PROCESS_DISCREPANCY_SCHEMA_VERSION:
        raise ValueError("unsupported process-selection schema version")
    selection = SourceFrozenProcessSelectionV1(
        candidate_id=str(payload["candidate_id"]),
        candidate_configuration=payload["candidate_configuration"],
        selected=bool(payload["selected"]),
        reason=str(payload["reason"]),
        source_case_ids=tuple(payload["source_case_ids"]),
        held_out_case_ids=tuple(payload["held_out_case_ids"]),
        source_checksums=payload["source_checksums"],
        source_metrics=payload["source_metrics"],
        thresholds=payload["thresholds"],
        target_outcomes_used_for_selection=bool(
            payload["target_outcomes_used_for_selection"]
        ),
        metadata=payload["metadata"],
    )
    if selection.selection_id != payload.get("selection_id"):
        raise ValueError("process-selection artifact digest mismatch")
    return selection
