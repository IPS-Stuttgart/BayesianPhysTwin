#!/usr/bin/env python3
"""Open V13 source prefix identities only after all predictions are sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    validate_adaptive_causal_response_query_artifacts,
)
from bayesian_phystwin.deform360_causal_response_tracker import (
    PROTOCOL_ID,
    PROVIDER_REPORT_FILENAME,
    validate_causal_response_tracker_artifacts,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_evaluation import (
    score_provider_case_arrays,
)
from bayesian_phystwin.observation_belief import file_sha256

CONFIG_RELATIVE_PATH = Path(
    "configs/sota/deform360_causal_response_tracker_v13.json"
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
        b"deform360-causal-response-tracker-v13-evaluation\0"
        + json.dumps(
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
    parser.add_argument("--adaptive-query-root", type=Path, required=True)
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
    return (
        processed_root
        / object_id
        / f"episode_{episode}"
        / "final_data.pkl"
    )


def _score(
    arrays: dict[str, np.ndarray],
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    *,
    prefix: str,
    shared_bias_standard_deviation_m: float,
) -> dict[str, Any]:
    name = f"{prefix}_" if prefix else ""
    return score_provider_case_arrays(
        trajectory_world_m=arrays[f"{name}trajectory_world_m"],
        accepted_support=arrays[f"{name}accepted_support"],
        local_covariance_m2=arrays[f"{name}local_covariance_m2"],
        shared_bias_standard_deviation_m=(
            shared_bias_standard_deviation_m
        ),
        target_m=target,
        target_visibility=visibility,
        target_validity=validity,
        entity_ids=arrays["entity_ids"],
        birth_frames=arrays["birth_frames"],
        update_frames=arrays["update_frames"],
        expected_query_count=16,
    )


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
    query_root = args.adaptive_query_root.resolve()
    output = args.output_dir.resolve()
    _require(not output.exists(), "source evaluation output already exists")
    output.mkdir(parents=True)

    prediction_rows: list[dict[str, Any]] = []
    provider_by_case: dict[str, dict[str, np.ndarray]] = {}
    for case in cases:
        query_report, _ = (
            validate_adaptive_causal_response_query_artifacts(
                query_root / case
            )
        )
        disposition_path = run_root / case / "disposition.json"
        disposition = json.loads(
            disposition_path.read_text(encoding="utf-8")
        )
        _require(
            disposition.get("case") == case
            and disposition.get("identity_target_read") is False
            and disposition.get(
                "state_or_readout_update_constructed"
            )
            is False
            and disposition.get("future_prediction_metric_read") is False
            and disposition.get(
                "held_v8_artifact_or_process_access"
            )
            is False,
            f"source disposition is invalid: {case}",
        )
        row = {
            "case": case,
            "adaptive_query_result_sha256": query_report[
                "result_sha256"
            ],
            "disposition_sha256": file_sha256(disposition_path),
            "status": disposition["status"],
        }
        if disposition["status"] == "tracker_prediction_sealed":
            provider_report, provider_arrays = (
                validate_causal_response_tracker_artifacts(
                    run_root / case / "provider"
                )
            )
            _require(
                provider_report.get("case") == case
                and provider_report.get("repository_revision") == revision,
                f"provider provenance differs: {case}",
            )
            row["provider_report_sha256"] = file_sha256(
                run_root
                / case
                / "provider"
                / PROVIDER_REPORT_FILENAME
            )
            row["provider_result_sha256"] = provider_report[
                "result_sha256"
            ]
            provider_by_case[case] = provider_arrays
        else:
            _require(
                disposition["status"] == "exact_query_abstention"
                and query_report["status"] == "abstained"
                and disposition.get("tracker_executed") is False,
                f"unsupported source disposition: {case}",
            )
        prediction_rows.append(row)

    gate = protocol["source_gate"]
    _require(
        len(provider_by_case)
        == int(gate["required_provider_prediction_count"]),
        "not every registered provider prediction is sealed",
    )
    barrier = {
        "schema_version": 1,
        "artifact_kind": (
            "Deform360CausalResponseTrackerV13PredictionBarrier"
        ),
        "protocol_id": PROTOCOL_ID,
        "repository_revision": revision,
        "protocol_sha256": file_sha256(protocol_path),
        "locked_case_count": len(cases),
        "provider_prediction_count": len(provider_by_case),
        "exact_query_abstention_count": (
            len(cases) - len(provider_by_case)
        ),
        "predictions": prediction_rows,
        "identity_target_read_before_barrier": False,
        "state_or_readout_update_constructed": False,
        "future_prediction_metric_read": False,
        "held_v8_artifact_or_process_access": False,
    }
    barrier["barrier_sha256"] = _canonical_sha256(
        barrier,
        key="barrier_sha256",
    )
    (output / BARRIER_FILENAME).write_text(
        json.dumps(barrier, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    scores: list[dict[str, Any]] = []
    shared_bias = float(
        protocol["multiview"]["shared_bias_standard_deviation_m"]
    )
    for case in cases:
        if case not in provider_by_case:
            scores.append(
                {
                    "case": case,
                    "status": "exact_query_abstention",
                    "provider": None,
                    "proposal": None,
                    "validation": None,
                }
            )
            continue
        arrays = provider_by_case[case]
        target, visibility, validity = _load_target(
            _target_path(args.processed_root.resolve(), case)
        )
        scores.append(
            {
                "case": case,
                "status": "scored_prefix_provider",
                "provider": _score(
                    arrays,
                    target,
                    visibility,
                    validity,
                    prefix="",
                    shared_bias_standard_deviation_m=shared_bias,
                ),
                "proposal": _score(
                    arrays,
                    target,
                    visibility,
                    validity,
                    prefix="proposal",
                    shared_bias_standard_deviation_m=shared_bias,
                ),
                "validation": _score(
                    arrays,
                    target,
                    visibility,
                    validity,
                    prefix="validation",
                    shared_bias_standard_deviation_m=shared_bias,
                ),
                "accepted_panel_disagreement_m": (
                    arrays["panel_disagreement_m"][
                        arrays["accepted_support"]
                    ].tolist()
                ),
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
        >= float(gate["minimum_case_supported_fraction"])
        for row in all_provider
    )
    provider_wins = sum(bool(row["provider_wins"]) for row in scored)
    rmse = np.asarray(
        [row["provider_rmse_m"] for row in scored],
        dtype=np.float64,
    )
    persistence = np.asarray(
        [row["persistence_rmse_m"] for row in scored],
        dtype=np.float64,
    )
    late = np.asarray(
        [row["late_provider_rmse_m"] for row in scored],
        dtype=np.float64,
    )
    relative_gain = (
        None
        if not len(scored) or float(np.mean(persistence)) <= 0.0
        else 1.0 - float(np.mean(rmse)) / float(np.mean(persistence))
    )
    disagreements = np.asarray(
        [
            value
            for row in scores
            for value in row.get("accepted_panel_disagreement_m", [])
        ],
        dtype=np.float64,
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
    gates = {
        "provider_prediction_count": (
            len(provider_by_case)
            == int(gate["required_provider_prediction_count"])
        ),
        "pooled_supported_fraction": (
            pooled_support >= float(gate["minimum_pooled_supported_fraction"])
        ),
        "case_supported_fraction": (
            case_support_passes
            >= int(gate["minimum_case_support_pass_count"])
        ),
        "scored_case_count": (
            len(scored) >= int(gate["minimum_scored_case_count"])
        ),
        "object_balanced_rmse": (
            len(scored) >= int(gate["minimum_scored_case_count"])
            and float(np.mean(rmse))
            <= float(gate["maximum_object_balanced_rmse_m"])
        ),
        "object_balanced_late_rmse": (
            len(scored) >= int(gate["minimum_scored_case_count"])
            and float(np.mean(late))
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
        "cross_panel_disagreement": (
            len(disagreements) > 0
            and float(np.mean(disagreements))
            <= float(gate["maximum_mean_cross_panel_disagreement_m"])
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "artifact_kind": (
            "Deform360CausalResponseTrackerV13SourceCompetenceResult"
        ),
        "protocol_id": PROTOCOL_ID,
        "repository_revision": revision,
        "claim_boundary": (
            "opened-source prefix-only provider competence; no state or "
            "readout update, future prediction, transfer, confirmation, "
            "or state-of-the-art claim"
        ),
        "barrier_sha256": barrier["barrier_sha256"],
        "source_gate_passed": passed,
        "decision": (
            "authorize_separately_locked_bias_aware_update"
            if passed
            else "stop_v13_tracker_provider_route"
        ),
        "aggregate": {
            "locked_case_count": len(cases),
            "provider_prediction_count": len(provider_by_case),
            "exact_query_abstention_count": (
                len(cases) - len(provider_by_case)
            ),
            "scored_case_count": len(scored),
            "supported_identity_count": supported_count,
            "scheduled_identity_count": scheduled_count,
            "pooled_supported_fraction": pooled_support,
            "case_support_pass_count": case_support_passes,
            "provider_case_wins": provider_wins,
            "object_balanced_provider_rmse_m": (
                None if not len(scored) else float(np.mean(rmse))
            ),
            "object_balanced_persistence_rmse_m": (
                None
                if not len(scored)
                else float(np.mean(persistence))
            ),
            "object_balanced_late_provider_rmse_m": (
                None if not len(scored) else float(np.mean(late))
            ),
            "relative_gain_over_persistence": relative_gain,
            "mean_accepted_cross_panel_disagreement_m": (
                None
                if not len(disagreements)
                else float(np.mean(disagreements))
            ),
            "calibration": calibration,
        },
        "gates": gates,
        "cases": scores,
        "information_boundary": {
            "prediction_sealed_before_identity_target_open": True,
            "maximum_scored_frame": 57,
            "tactile_event_read": False,
            "state_or_readout_update_constructed": False,
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
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_gate_passed": passed,
                "provider_prediction_count": len(provider_by_case),
                "pooled_supported_fraction": pooled_support,
                "object_balanced_provider_rmse_m": (
                    result["aggregate"][
                        "object_balanced_provider_rmse_m"
                    ]
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
