"""Frozen aggregate gate for TAPNext++ depth-completion source transfer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_tapnextpp_competence import canonical_sha256, file_sha256
from .tapnextpp_transfer_staging import (
    TRANSFER_PROTOCOL_ID,
    validate_transfer_protocol,
)

CASE_RESULT_FILENAME = "tapnextpp_depth_completion_transfer_result.json"
AGGREGATE_RESULT_FILENAME = "tapnextpp_depth_completion_transfer_summary.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _mean_metric(
    records: list[dict[str, Any]],
    name: str,
) -> float | None:
    values = [
        float(record["metrics"][name])
        for record in records
        if record["metrics"].get(name) is not None
    ]
    return float(np.mean(values)) if values else None


def evaluate_transfer_panel(
    protocol_path: str | Path,
    source_manifest_path: str | Path,
    case_result_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Account for all fixed cases and apply the preregistered panel gates."""

    protocol_file = Path(protocol_path).resolve()
    source_file = Path(source_manifest_path).resolve()
    result_root = Path(case_result_root).resolve()
    output = Path(output_path).resolve()
    _require(not output.exists(), "aggregate transfer result already exists")
    protocol = _load_json(protocol_file)
    validate_transfer_protocol(protocol)
    source = _load_json(source_file)
    _require(
        source.get("artifact_kind")
        == "PhysTwinTAPNextPPDepthCompletionTransferSourceManifest"
        and source.get("protocol_id") == TRANSFER_PROTOCOL_ID,
        "source manifest kind changed",
    )
    _require(
        source.get("result_sha256") == canonical_sha256(source),
        "source manifest hash changed",
    )
    _require(
        source.get("protocol_sha256") == file_sha256(protocol_file),
        "source manifest binds another transfer protocol",
    )
    source_records = source.get("case_records")
    _require(
        isinstance(source_records, list)
        and [record.get("case") for record in source_records]
        == protocol["fixed_source_cases"],
        "source manifest case order changed",
    )

    evaluated: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    for source_record in source_records:
        case_name = source_record["case"]
        if source_record.get("status") != "prediction-ready":
            dispositions.append(
                {
                    "case": case_name,
                    "status": "technical-staging-failure",
                    "provider_gate_passed": False,
                }
            )
            continue
        result_path = result_root / case_name / CASE_RESULT_FILENAME
        if not result_path.is_file():
            dispositions.append(
                {
                    "case": case_name,
                    "status": "technical-prediction-or-evaluation-failure",
                    "provider_gate_passed": False,
                }
            )
            continue
        result = _load_json(result_path)
        _require(
            result.get("artifact_kind")
            == "PhysTwinTAPNextPPDepthCompletionTransferCaseResult"
            and result.get("case") == case_name,
            f"transfer result identity changed for {case_name}",
        )
        _require(
            result.get("result_sha256") == canonical_sha256(result),
            f"transfer result hash changed for {case_name}",
        )
        evaluated.append(result)
        dispositions.append(
            {
                "case": case_name,
                "status": "evaluated",
                "provider_gate_passed": bool(result["provider_gate_passed"]),
                "result_sha256": file_sha256(result_path),
            }
        )

    passing_count = sum(
        disposition["provider_gate_passed"] for disposition in dispositions
    )
    eligible_rows = sum(
        int(result["metrics"]["eligible_rows"]) for result in evaluated
    )
    supported_rows = sum(
        int(result["metrics"]["completed_supported_rows"])
        for result in evaluated
    )
    aggregate_support = supported_rows / max(eligible_rows, 1)
    balanced_rmse = _mean_metric(evaluated, "candidate_identity_rmse_m")
    balanced_gain = _mean_metric(
        evaluated,
        "relative_gain_over_persistence",
    )
    gate_config = protocol["aggregate_gates"]
    gates = {
        "passing_case_count": passing_count
        >= int(gate_config["minimum_passing_case_count"]),
        "aggregate_supported_fraction": aggregate_support
        >= float(gate_config["minimum_aggregate_supported_fraction"]),
        "case_balanced_relative_gain": balanced_gain is not None
        and balanced_gain
        >= float(
            gate_config[
                "minimum_case_balanced_relative_gain_over_persistence"
            ]
        ),
        "case_balanced_identity_rmse": balanced_rmse is not None
        and balanced_rmse
        <= float(gate_config["maximum_case_balanced_identity_rmse_m"]),
        "fixed_case_accounting": len(dispositions)
        == len(protocol["fixed_source_cases"]),
    }
    passed = all(gates.values())
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPDepthCompletionTransferSummary",
        "protocol_id": TRANSFER_PROTOCOL_ID,
        "protocol_sha256": file_sha256(protocol_file),
        "source_manifest_sha256": file_sha256(source_file),
        "case_count": len(dispositions),
        "evaluated_case_count": len(evaluated),
        "passing_case_count": passing_count,
        "technical_failure_count": len(dispositions) - len(evaluated),
        "metrics": {
            "eligible_rows": eligible_rows,
            "completed_supported_rows": supported_rows,
            "aggregate_supported_fraction": aggregate_support,
            "case_balanced_identity_rmse_m": balanced_rmse,
            "case_balanced_relative_gain_over_persistence": balanced_gain,
        },
        "gates": gates,
        "transfer_gate_passed": passed,
        "decision": (
            "authorize-separately-frozen-opened-source-assimilation-study"
            if passed
            else "stop-tapnextpp-depth-completion-route"
        ),
        "case_dispositions": dispositions,
        "information_boundary": {
            "source_prefix_competence_results_opened": True,
            "future_simulator_outcome_read": False,
            "held_v8_accessed": False,
            "failed_cases_replaced": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    summary["result_sha256"] = canonical_sha256(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "AGGREGATE_RESULT_FILENAME",
    "CASE_RESULT_FILENAME",
    "evaluate_transfer_panel",
]
