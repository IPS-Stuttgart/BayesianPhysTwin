"""Outcome-free closure for the Deform360 tactile-guard transfer.

This module may conclude that the registered advancement gates are impossible
from sealed predictions alone. It deliberately has no outcome-manifest or
target-trajectory input.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_tactile_guard_outcome_sealed import (
    CLAIM_LABEL,
    ORDINARY_STATUS,
    PROTOCOL_ID,
    TECHNICAL_FALLBACK_STATUS,
    canonical_sha256,
    canonical_text_sha256,
    file_sha256,
    load_json,
    load_outcome_sealed_protocol,
    validate_prediction_barrier,
    validate_prediction_seal,
    write_json,
)

PREOUTCOME_RESULT_ARTIFACT_KIND = (
    "Deform360TactileGuardPreOutcomeImpossibilityResultV1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _case_prediction_audit(
    *,
    record: Mapping[str, Any],
    protocol: Mapping[str, Any],
    prediction_root: Path,
) -> dict[str, Any]:
    case = str(record["case"])
    case_dir = prediction_root / case
    seal_path = case_dir / "prediction_seal.json"
    seal = load_json(seal_path)
    validate_prediction_seal(seal, protocol=protocol, prediction_dir=case_dir)

    _require(
        seal.get("status") == record.get("status")
        and file_sha256(seal_path) == record.get("file_sha256")
        and seal.get("result_sha256") == record.get("result_sha256"),
        f"prediction seal no longer matches barrier: {case}",
    )
    report_path = case_dir / "guarded_prediction_report.json"
    report = load_json(report_path)
    accepted = int(report.get("accepted_update_count", -1))
    _require(
        accepted >= 0 and accepted == int(seal.get("accepted_update_count", -1)),
        f"accepted-update count changed: {case}",
    )
    archive_path = case_dir / "guarded_prediction.npz"
    with np.load(archive_path, allow_pickle=False) as stored:
        guarded = np.asarray(stored["guarded_prediction_m"])
        baseline = np.asarray(stored["selected_baseline_m"])
        raw_candidate = np.asarray(stored["raw_candidate_m"])
    _require(
        guarded.shape == baseline.shape == raw_candidate.shape,
        f"prediction shapes differ: {case}",
    )
    guarded_differs = not np.array_equal(guarded, baseline)
    raw_differs = not np.array_equal(raw_candidate, baseline)
    if seal["status"] == TECHNICAL_FALLBACK_STATUS:
        _require(
            accepted == 0 and not guarded_differs,
            f"technical fallback is not exact: {case}",
        )
    else:
        _require(
            seal["status"] == ORDINARY_STATUS,
            f"unexpected prediction status: {case}",
        )
    return {
        "case": case,
        "status": seal["status"],
        "accepted_update_count": accepted,
        "raw_candidate_differs_from_registered_baseline": raw_differs,
        "guarded_prediction_differs_from_registered_baseline": guarded_differs,
        "prediction_seal_file_sha256": file_sha256(seal_path),
        "prediction_seal_result_sha256": seal["result_sha256"],
    }


def build_preoutcome_impossibility_result(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    barrier_path: str | Path,
    prediction_root: str | Path,
    runtime_revision: str,
) -> dict[str, Any]:
    """Prove a locked advancement gate unreachable without reading outcomes."""

    _require(
        len(runtime_revision) == 40
        and all(character in "0123456789abcdef" for character in runtime_revision),
        "runtime revision must be a lowercase 40-character Git revision",
    )
    protocol = load_outcome_sealed_protocol(protocol_path)
    barrier = validate_prediction_barrier(
        barrier_path,
        protocol_path=protocol_path,
        prediction_root=prediction_root,
    )
    root = Path(prediction_root).resolve()
    records = {
        str(record["case"]): record for record in barrier.get("records", [])
    }
    audits = [
        _case_prediction_audit(
            record=records[str(case["case"])],
            protocol=protocol,
            prediction_root=root,
        )
        for case in protocol["cohort"]["cases"]
    ]
    admitted_cases = [
        row for row in audits if int(row["accepted_update_count"]) > 0
    ]
    nontrivial_cases = [
        row
        for row in audits
        if bool(row["guarded_prediction_differs_from_registered_baseline"])
    ]
    minimum_wins = int(protocol["advancement_gates"]["minimum_joint_case_wins"])
    maximum_possible_wins = len(nontrivial_cases)
    _require(
        maximum_possible_wins < minimum_wins,
        "joint-win gate remains reachable; an outcome-free failure is not valid",
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREOUTCOME_RESULT_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_file_sha256"],
        "claim_label": CLAIM_LABEL,
        "runtime_revision": runtime_revision,
        "finalizer_source_sha256": canonical_text_sha256(Path(__file__)),
        "prediction_barrier": {
            "file_sha256": file_sha256(barrier_path),
            "result_sha256": barrier["result_sha256"],
            "barrier_passed": True,
        },
        "cohort_accounting": {
            "ordinary_successful_prediction_count": barrier[
                "ordinary_successful_prediction_count"
            ],
            "retained_technical_failure_count": barrier[
                "retained_technical_failure_count"
            ],
            "unsealable_case_count": barrier["unsealable_case_count"],
            "replacement_count": barrier["replacement_count"],
            "total_locked_case_count": barrier["total_locked_case_count"],
        },
        "guard_audit": {
            "accepted_update_count": sum(
                int(row["accepted_update_count"]) for row in audits
            ),
            "admitted_case_count": len(admitted_cases),
            "admitted_cases": [row["case"] for row in admitted_cases],
            "nontrivial_prediction_case_count": maximum_possible_wins,
            "nontrivial_prediction_cases": [row["case"] for row in nontrivial_cases],
            "exact_registered_baseline_case_count": len(audits)
            - maximum_possible_wins,
            "case_audits": audits,
        },
        "gate_impossibility_proof": {
            "gate": "minimum_joint_case_wins",
            "required_joint_case_wins": minimum_wins,
            "maximum_possible_joint_case_wins": maximum_possible_wins,
            "reason": (
                "A bit-exact baseline case can only tie, so strict joint wins are "
                "possible only for cases whose sealed guarded trajectory differs "
                "from the registered baseline."
            ),
        },
        "advancement_decision": {
            "status": "failed_before_outcome_opening",
            "reason": "minimum_joint_case_wins_is_mathematically_unreachable",
            "future_outcomes_opened": False,
            "target_metrics_scored": False,
            "state_of_the_art_claim_supported": False,
        },
        "information_boundary": {
            "future_identity_or_metric_read": False,
            "future_object_observation_read": False,
            "outcome_manifest_read": False,
            "target_trajectory_read": False,
            "held_v8_read": False,
        },
    }
    payload["artifact_sha256"] = canonical_sha256(
        payload, digest_key="artifact_sha256"
    )
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "pre-outcome result already exists")
    write_json(destination, payload)
    return payload


__all__ = [
    "PREOUTCOME_RESULT_ARTIFACT_KIND",
    "build_preoutcome_impossibility_result",
]
