"""Mechanical Stage-2 gate for official-Warp equivariant-force records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phystwin_equivariant_force_source import (
    EquivariantForceSourceProtocol,
)
from .phystwin_equivariant_force_stage2 import (
    EQUIVARIANT_FORCE_STAGE2_CONTRACT,
    EquivariantForceStage2Protocol,
)
from .phystwin_residual_dynamics import _sha256


_METRICS = ("chamfer_distance_m", "track_error_m", "late_track_error_m")


def _positive_metric(
    record: Mapping[str, Any],
    section: str,
    metric: str,
) -> float:
    values = record.get(section)
    if not isinstance(values, Mapping):
        raise ValueError(f"official-Warp record omits {section}")
    result = float(values.get(metric, np.nan))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{section}.{metric} must be positive and finite")
    return result


def _mean_metric(
    records: Sequence[Mapping[str, Any]],
    section: str,
    metric: str,
) -> float:
    return float(
        np.mean([_positive_metric(record, section, metric) for record in records])
    )


def evaluate_equivariant_force_official_warp_gate(
    records: Sequence[Mapping[str, Any]],
    protocol: EquivariantForceSourceProtocol,
    execution_protocol: EquivariantForceStage2Protocol,
    *,
    force_target_competence_passed: bool,
) -> dict[str, Any]:
    """Apply the locked Stage-2 rules without fitting or opening a target."""

    if execution_protocol.seeds != protocol.training.seeds:
        raise ValueError("Stage-2 seeds differ from the source protocol")
    source_cases = tuple(str(case) for case in protocol.payload["source_cases"])
    if len(records) != len(source_cases):
        raise ValueError("official-Warp gate requires one record per source case")
    by_case = {}
    for raw in records:
        case = str(raw.get("case_id", ""))
        if not case or case in by_case:
            raise ValueError("official-Warp case IDs must be nonempty and unique")
        by_case[case] = raw
    if set(by_case) != set(source_cases):
        raise ValueError("official-Warp records differ from the locked source cases")

    case_results = []
    for case in source_cases:
        record = by_case[case]
        if record.get("target_artifacts_opened") is not False:
            raise ValueError(f"{case}: target boundary is not explicitly closed")
        source_checksums = record.get("source_checksums")
        if (
            not isinstance(source_checksums, Mapping)
            or source_checksums.get("stage2_source_manifest")
            != execution_protocol.source_manifest_sha256
            or source_checksums.get("official_simulator")
            != execution_protocol.official_simulator_sha256
            or source_checksums.get("stage2_implementation")
            != execution_protocol.implementation_sha256
        ):
            raise ValueError(f"{case}: Stage-2 source provenance changed")
        if (
            record.get("stage2_execution_contract")
            != EQUIVARIANT_FORCE_STAGE2_CONTRACT
        ):
            raise ValueError(f"{case}: Stage-2 execution contract changed")
        if record.get("seed_aggregation") != (
            "arithmetic_mean_force_field_per_frame_float64_then_float32"
        ):
            raise ValueError(f"{case}: seed aggregation changed")
        frame_contract = record.get("frame_contract")
        if not isinstance(frame_contract, Mapping) or any(
            frame_contract.get(key) != value
            for key, value in {
                "initial_state_frame": 0,
                "first_simulator_step_frame": 1,
                "fit_end_is_exclusive": True,
                "score_interval": "[fit_end_frame, train_end_frame)",
            }.items()
        ):
            raise ValueError(f"{case}: frame contract changed")
        parity = bool(record.get("zero_force_bitwise_parity", False))
        reference_supported = bool(
            record.get("readout_correction_reference_supported", False)
        )
        shrinkage = float(record.get("readout_correction_shrinkage", np.nan))
        if not np.isfinite(shrinkage):
            raise ValueError(f"{case}: correction shrinkage must be finite")
        ratios = {
            metric: (
                _positive_metric(record, "candidate", metric)
                / _positive_metric(record, "reference", metric)
            )
            for metric in _METRICS
        }
        case_results.append(
            {
                "case_id": case,
                "zero_force_bitwise_parity": parity,
                "readout_correction_reference_supported": reference_supported,
                "readout_correction_shrinkage": shrinkage,
                "ratios": ratios,
                "reference": {
                    metric: _positive_metric(record, "reference", metric)
                    for metric in _METRICS
                },
                "candidate": {
                    metric: _positive_metric(record, "candidate", metric)
                    for metric in _METRICS
                },
            }
        )

    aggregate_reference = {
        metric: _mean_metric(case_results, "reference", metric)
        for metric in _METRICS
    }
    aggregate_candidate = {
        metric: _mean_metric(case_results, "candidate", metric)
        for metric in _METRICS
    }
    aggregate_ratios = {
        metric: aggregate_candidate[metric] / aggregate_reference[metric]
        for metric in _METRICS
    }
    balanced_improvement = 1.0 - 0.5 * (
        aggregate_ratios["chamfer_distance_m"]
        + aggregate_ratios["track_error_m"]
    )

    fold_results = []
    for fold in protocol.payload["source_folds"]:
        selected = [by_case[str(case)] for case in fold["held_out_cases"]]
        ratios = {}
        for metric in ("chamfer_distance_m", "track_error_m"):
            ratios[metric] = (
                _mean_metric(selected, "candidate", metric)
                / _mean_metric(selected, "reference", metric)
            )
        fold_results.append(
            {
                "name": str(fold["name"]),
                "held_out_cases": list(fold["held_out_cases"]),
                "ratios": ratios,
                "both_win": all(value < 1.0 for value in ratios.values()),
            }
        )

    gate = protocol.payload["source_gate"]
    maximum_case_ratio = max(
        result["ratios"][metric]
        for result in case_results
        for metric in ("chamfer_distance_m", "track_error_m")
    )
    shrinkage_count = sum(
        result["readout_correction_reference_supported"]
        and result["readout_correction_shrinkage"]
        >= float(gate["minimum_readout_correction_shrinkage"])
        for result in case_results
    )
    checks = {
        "force_target_competence": bool(force_target_competence_passed),
        "zero_force_bitwise_parity": all(
            result["zero_force_bitwise_parity"] for result in case_results
        ),
        "minimum_balanced_improvement": (
            balanced_improvement
            >= float(gate["minimum_balanced_official_warp_improvement"])
        ),
        "aggregate_cd_and_track_improve": (
            aggregate_ratios["chamfer_distance_m"] < 1.0
            and aggregate_ratios["track_error_m"] < 1.0
        ),
        "minimum_both_win_folds": (
            sum(result["both_win"] for result in fold_results)
            >= int(gate["minimum_fold_aggregate_both_win_count"])
        ),
        "maximum_case_metric_ratio": (
            maximum_case_ratio <= float(gate["maximum_single_case_metric_ratio"])
        ),
        "minimum_shrinkage_case_count": (
            shrinkage_count >= int(gate["minimum_shrinkage_case_count"])
        ),
        "late_horizon_improves": (
            aggregate_ratios["late_track_error_m"] < 1.0
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "contract": protocol.payload["contract"],
        "stage2_execution_contract": EQUIVARIANT_FORCE_STAGE2_CONTRACT,
        "stage2_source_protocol_sha256": (
            execution_protocol.source_protocol_sha256
        ),
        "stage2_source_manifest_sha256": (
            execution_protocol.source_manifest_sha256
        ),
        "official_simulator_sha256": (
            execution_protocol.official_simulator_sha256
        ),
        "stage2_implementation_sha256": (
            execution_protocol.implementation_sha256
        ),
        "stage": "official_warp_source_gate",
        "source_gate_passed": passed,
        "independent_preregistered_evaluation_authorized": passed,
        "historical_target_access_authorized": False,
        "target_artifacts_opened": False,
        "checks": checks,
        "aggregate_reference": aggregate_reference,
        "aggregate_candidate": aggregate_candidate,
        "aggregate_ratios": aggregate_ratios,
        "balanced_improvement": balanced_improvement,
        "maximum_case_metric_ratio": maximum_case_ratio,
        "shrinkage_case_count": shrinkage_count,
        "folds": fold_results,
        "cases": case_results,
        "claim_boundary": (
            "A source pass authorizes only the next registered evaluation. "
            "It is not an independent state-of-the-art result."
        ),
    }


def write_equivariant_force_official_warp_gate(
    path: str | Path,
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Write one immutable machine-readable Stage-2 decision."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"path": str(output.resolve()), "sha256": _sha256(output)}
