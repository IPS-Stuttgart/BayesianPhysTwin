#!/usr/bin/env python3
"""Open V11 source identity targets only after every prediction is sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_action_supported_tapnextpp import (
    PROTOCOL_ID,
    PROVIDER_REPORT_FILENAME,
    validate_action_supported_provider_artifacts,
    validate_action_supported_query_artifacts,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_evaluation import (
    score_provider_case_arrays,
)
from bayesian_phystwin.observation_belief import file_sha256

CONFIG_RELATIVE_PATH = Path(
    "configs/sota/deform360_action_supported_tapnextpp_source_v11.json"
)
BARRIER_FILENAME = "prediction_completeness_barrier.json"
RESULT_FILENAME = "source_competence_result.json"
CHI_SQUARED_3D_90PCT = 6.251388631170325


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(payload: dict[str, Any], *, key: str) -> str:
    canonical = dict(payload)
    canonical.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    _require(isinstance(payload, dict), "source target payload is invalid")
    target = np.asarray(payload["object_points"], dtype=np.float64)
    visibility = np.asarray(payload["object_visibilities"], dtype=bool)
    validity = np.asarray(payload["object_motions_valid"], dtype=bool)
    _require(
        target.ndim == 3
        and target.shape[0] >= 58
        and target.shape[2] == 3
        and visibility.shape == validity.shape == target.shape[:2],
        "source target arrays changed shape",
    )
    return target, visibility, validity


def _target_path(processed_root: Path, case: str) -> Path:
    object_id, episode = case.rsplit("-ep", maxsplit=1)
    _require(
        len(episode) == 4 and episode.isdigit(),
        "case episode token is invalid",
    )
    return processed_root / object_id / f"episode_{episode}" / "final_data.pkl"


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    revision = _git_output(repo, "rev-parse", "HEAD")
    _require(not _git_output(repo, "status", "--porcelain"), "repository is dirty")
    protocol_path = repo / CONFIG_RELATIVE_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    cases = tuple(str(record["case"]) for record in protocol["cases"])
    _require(
        len(cases) == len(set(cases))
        == int(protocol["source_gate"]["locked_case_count"]),
        "source panel count changed",
    )
    run_root = args.run_root.resolve()
    output = args.output_dir.resolve()
    _require(not output.exists(), "source evaluation output already exists")
    output.mkdir(parents=True)

    prediction_rows: list[dict[str, Any]] = []
    provider_by_case: dict[str, dict[str, np.ndarray]] = {}
    for case in cases:
        case_root = run_root / case
        query_report, _ = validate_action_supported_query_artifacts(
            case_root / "query"
        )
        disposition_path = case_root / "disposition.json"
        disposition = json.loads(
            disposition_path.read_text(encoding="utf-8")
        )
        _require(
            disposition.get("case") == case
            and disposition.get("identity_target_read") is False
            and disposition.get("state_update_constructed") is False,
            f"source disposition is invalid: {case}",
        )
        row = {
            "case": case,
            "query_report_sha256": file_sha256(
                case_root / "query" / "action_supported_queries.json"
            ),
            "query_result_sha256": query_report["result_sha256"],
            "disposition_sha256": file_sha256(disposition_path),
            "status": disposition["status"],
        }
        if disposition["status"] == "provider_prediction_sealed":
            provider_report, provider_arrays = (
                validate_action_supported_provider_artifacts(
                    case_root / "provider"
                )
            )
            _require(
                provider_report.get("case") == case
                and provider_report.get("repository_revision") == revision,
                f"provider provenance differs: {case}",
            )
            row["provider_report_sha256"] = file_sha256(
                case_root / "provider" / PROVIDER_REPORT_FILENAME
            )
            row["provider_result_sha256"] = provider_report[
                "result_sha256"
            ]
            provider_by_case[case] = provider_arrays
        else:
            _require(
                disposition["status"] == "query_budget_abstention"
                and disposition.get("tracker_executed") is False,
                f"unsupported source disposition: {case}",
            )
        prediction_rows.append(row)

    barrier = {
        "schema_version": 1,
        "artifact_kind": (
            "Deform360ActionSupportedTAPNextPPPredictionBarrier"
        ),
        "protocol_id": PROTOCOL_ID,
        "repository_revision": revision,
        "protocol_sha256": file_sha256(protocol_path),
        "locked_case_count": len(cases),
        "provider_prediction_count": len(provider_by_case),
        "query_abstention_count": len(cases) - len(provider_by_case),
        "predictions": prediction_rows,
        "identity_target_read_before_barrier": False,
        "state_update_constructed": False,
        "held_v8_artifact_or_process_access": False,
    }
    barrier["barrier_sha256"] = _canonical_sha256(
        barrier,
        key="barrier_sha256",
    )
    (output / BARRIER_FILENAME).write_text(
        json.dumps(barrier, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    scores: list[dict[str, Any]] = []
    for case in cases:
        if case not in provider_by_case:
            scores.append(
                {
                    "case": case,
                    "status": "query_budget_abstention",
                    "provider": None,
                }
            )
            continue
        arrays = provider_by_case[case]
        target, visibility, validity = _load_target(
            _target_path(args.processed_root.resolve(), case)
        )
        score = score_provider_case_arrays(
            trajectory_world_m=arrays["trajectory_world_m"],
            accepted_support=arrays["accepted_support"],
            local_covariance_m2=arrays["local_covariance_m2"],
            shared_bias_standard_deviation_m=float(
                protocol["multiview"][
                    "shared_bias_standard_deviation_m"
                ]
            ),
            target_m=target,
            target_visibility=visibility,
            target_validity=validity,
            entity_ids=arrays["entity_ids"],
            birth_frames=arrays["birth_frames"],
            update_frames=arrays["update_frames"],
            expected_query_count=int(protocol["query"]["query_count"]),
        )
        scores.append(
            {
                "case": case,
                "status": "scored_prefix_provider",
                "provider": score,
            }
        )

    scored = [
        row["provider"]
        for row in scores
        if row["provider"] is not None
        and row["provider"]["provider_rmse_m"] is not None
    ]
    all_provider = [
        row["provider"]
        for row in scores
        if row["provider"] is not None
    ]
    scheduled_count = sum(
        int(row["scheduled_identity_count"]) for row in all_provider
    )
    supported_count = sum(
        int(row["supported_identity_count"]) for row in all_provider
    )
    pooled_support = (
        0.0 if scheduled_count == 0 else supported_count / scheduled_count
    )
    case_support_passes = sum(
        float(row["supported_fraction"])
        >= float(protocol["source_gate"]["minimum_case_supported_fraction"])
        for row in all_provider
    )
    provider_wins = sum(bool(row["provider_wins"]) for row in scored)
    rmse_values = np.asarray(
        [row["provider_rmse_m"] for row in scored],
        dtype=np.float64,
    )
    persistence_values = np.asarray(
        [row["persistence_rmse_m"] for row in scored],
        dtype=np.float64,
    )
    late_values = np.asarray(
        [row["late_provider_rmse_m"] for row in scored],
        dtype=np.float64,
    )
    relative_gain = (
        None
        if not len(scored) or float(np.mean(persistence_values)) <= 0.0
        else 1.0
        - float(np.mean(rmse_values)) / float(np.mean(persistence_values))
    )
    nees = np.asarray(
        [
            value
            for row in scored
            for value in row["mahalanobis_squared"]
        ],
        dtype=np.float64,
    )
    calibration = {
        "endpoint_nees_count": int(len(nees)),
        "mean_nees": None if not len(nees) else float(np.mean(nees)),
        "nominal_90pct_coordinate_coverage": (
            None
            if not len(nees)
            else float(np.mean(nees <= CHI_SQUARED_3D_90PCT))
        ),
        "chi_squared_3d_90pct_threshold": CHI_SQUARED_3D_90PCT,
        "raw_covariance_only_not_source_calibrated": True,
    }
    gate = protocol["source_gate"]
    gates = {
        "provider_prediction_count": (
            len(provider_by_case)
            >= int(gate["minimum_provider_prediction_count"])
        ),
        "pooled_supported_fraction": (
            pooled_support >= float(gate["minimum_pooled_supported_fraction"])
        ),
        "case_supported_fraction": (
            case_support_passes
            >= int(gate["minimum_case_support_pass_count"])
        ),
        "object_balanced_rmse": (
            len(scored) >= int(gate["minimum_scored_case_count"])
            and float(np.mean(rmse_values))
            <= float(gate["maximum_object_balanced_rmse_m"])
        ),
        "object_balanced_late_rmse": (
            len(scored) >= int(gate["minimum_scored_case_count"])
            and float(np.mean(late_values))
            <= float(gate["maximum_object_balanced_late_rmse_m"])
        ),
        "relative_gain_over_persistence": (
            relative_gain is not None
            and relative_gain
            >= float(gate["minimum_relative_gain_over_persistence"])
        ),
        "provider_case_wins": (
            provider_wins >= int(gate["minimum_provider_case_wins"])
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "artifact_kind": (
            "Deform360ActionSupportedTAPNextPPSourceCompetenceResult"
        ),
        "protocol_id": PROTOCOL_ID,
        "repository_revision": revision,
        "claim_boundary": (
            "opened-source prefix-only provider competence; no state update, "
            "future prediction, transfer, confirmation, or SOTA claim"
        ),
        "barrier_sha256": barrier["barrier_sha256"],
        "source_gate_passed": passed,
        "decision": (
            "authorize_separately_locked_guarded_state_update"
            if passed
            else "stop_action_supported_tapnextpp_route"
        ),
        "aggregate": {
            "locked_case_count": len(cases),
            "provider_prediction_count": len(provider_by_case),
            "scored_case_count": len(scored),
            "supported_identity_count": supported_count,
            "scheduled_identity_count": scheduled_count,
            "pooled_supported_fraction": pooled_support,
            "case_support_pass_count": case_support_passes,
            "provider_case_wins": provider_wins,
            "object_balanced_provider_rmse_m": (
                None if not len(scored) else float(np.mean(rmse_values))
            ),
            "object_balanced_persistence_rmse_m": (
                None
                if not len(scored)
                else float(np.mean(persistence_values))
            ),
            "object_balanced_late_provider_rmse_m": (
                None if not len(scored) else float(np.mean(late_values))
            ),
            "relative_gain_over_persistence": relative_gain,
            "calibration": calibration,
        },
        "gates": gates,
        "cases": scores,
        "information_boundary": {
            "prediction_sealed_before_identity_target_open": True,
            "maximum_scored_frame": 57,
            "state_update_constructed": False,
            "future_prediction_metric_read": False,
            "held_v8_artifact_or_process_access": False,
            "v1_sealed_target_access": False,
        },
    }
    result["result_sha256"] = _canonical_sha256(
        result,
        key="result_sha256",
    )
    (output / RESULT_FILENAME).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_gate_passed": passed,
                "provider_prediction_count": len(provider_by_case),
                "pooled_supported_fraction": pooled_support,
                "object_balanced_provider_rmse_m": (
                    result["aggregate"]["object_balanced_provider_rmse_m"]
                ),
                "relative_gain_over_persistence": relative_gain,
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
