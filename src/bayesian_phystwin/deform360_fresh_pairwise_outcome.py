"""Barrier-gated outcome scoring for the fresh Deform360 pairwise study."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_fresh_pairwise_prediction import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    SELECTED_RAW_ARM,
)
from .deform360_fresh_pairwise_protocol import (
    EXPECTED_FRAME_COUNT,
    UPDATE_FRAMES,
    array_sha256,
    canonical_sha256,
    file_sha256,
    load_bound_cohort,
    load_fresh_pairwise_protocol,
    load_json,
    validate_belief_prediction_seal,
    validate_completeness_barrier,
    write_json,
)
from .deform360_fresh_source_lock import validate_fresh_source_admission
from .deform360_online_belief_evaluation import (
    score_deform360_hidden_trajectory,
)


OUTCOME_KIND = "Deform360FreshPairwiseOutcome"
SUMMARY_KIND = "Deform360FreshPairwiseOutcomeSummary"
ANALYSIS_ID = "deform360-fresh-pairwise-outcome-v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
ARMS = (PHYSICAL_ARM, PERSISTENCE_ARM, SELECTED_RAW_ARM, CANDIDATE_ARM)
COMPARATORS = (PHYSICAL_ARM, PERSISTENCE_ARM, SELECTED_RAW_ARM)
PRIMARY_METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)
ARCHIVE_ARRAY_KEYS = {
    PHYSICAL_ARM: "physical_prior_m",
    PERSISTENCE_ARM: "persistence_m",
    SELECTED_RAW_ARM: "selected_raw_backbone_m",
    CANDIDATE_ARM: "candidate_m",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_fresh_pairwise_outcome_analysis(
    analysis_path: str | Path,
    *,
    protocol_path: str | Path,
    cohort_path: str | Path,
    barrier_path: str | Path,
) -> dict[str, Any]:
    """Validate the outcome analysis lock fixed before future deserialization."""

    analysis = load_json(analysis_path)
    barrier = load_json(barrier_path)
    _require(
        analysis.get("schema_version") == 1
        and analysis.get("analysis_id") == ANALYSIS_ID,
        "fresh outcome analysis lock is incompatible",
    )
    _require(
        analysis.get("prediction_protocol")
        == {
            "protocol_id": "deform360-fresh-pairwise-belief-v1",
            "config_file_sha256": file_sha256(protocol_path),
        },
        "fresh outcome analysis uses another prediction protocol",
    )
    cohort = analysis.get("cohort", {})
    _require(
        cohort.get("lock_sha256")
        == "bafe26848ee83d8a4201e9d11d51af106370647f76ec702003e9ec51d3843729"
        and cohort.get("lock_file_sha256") == file_sha256(cohort_path)
        and cohort.get("case_count") == 12
        and cohort.get("physical_object_count") == 12,
        "fresh outcome analysis uses another cohort",
    )
    barrier_lock = analysis.get("completeness_barrier", {})
    _require(
        barrier_lock.get("artifact_kind")
        == "Deform360FreshPairwiseCompletenessBarrier"
        and barrier_lock.get("file_sha256") == file_sha256(barrier_path)
        and barrier_lock.get("result_sha256") == barrier.get("result_sha256")
        and barrier_lock.get("ordinary_prediction_count") == 12
        and barrier_lock.get("retained_technical_failure_count") == 0
        and barrier_lock.get("unsealable_case_count") == 0
        and barrier_lock.get("replacement_count") == 0,
        "fresh outcome analysis uses another completeness barrier",
    )
    scoring = analysis.get("scoring", {})
    _require(
        scoring.get("frame_count") == EXPECTED_FRAME_COUNT
        and scoring.get("update_frames") == list(UPDATE_FRAMES)
        and scoring.get("scored_frame_ranges_half_open")
        == [[20, 38], [39, 57], [58, 76]]
        and scoring.get("permanently_excluded_measurement_identity_count") == 16
        and scoring.get("primary_metrics") == list(PRIMARY_METRICS)
        and scoring.get("candidate_arm") == CANDIDATE_ARM
        and scoring.get("comparators") == list(COMPARATORS)
        and scoring.get("replicate_unit") == "physical object",
        "fresh outcome scoring contract changed",
    )
    _require(
        analysis.get("bootstrap")
        == {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "interval": 0.95,
            "resampling_unit": "physical object",
        }
        and analysis.get("transfer_gate")
        == {
            "per_comparator_metric": (
                "candidate object-balanced mean difference < 0 and "
                "object-cluster 95% upper bound < 0"
            ),
            "joint_rule": "both primary metrics pass against every comparator",
        },
        "fresh outcome analysis gate changed",
    )
    _require(
        analysis.get("calibration_claim", {}).get("evaluated") is False
        and analysis.get("official_sota_claim", {}).get("allowed") is False,
        "fresh outcome claim boundary changed",
    )
    return analysis


def _scored_frames(
    frame_count: int = EXPECTED_FRAME_COUNT,
    update_frames: Sequence[int] = UPDATE_FRAMES,
) -> tuple[int, ...]:
    updates = tuple(int(value) for value in update_frames)
    _require(
        frame_count == EXPECTED_FRAME_COUNT
        and updates == UPDATE_FRAMES,
        "fresh outcome scoring window changed",
    )
    scored: list[int] = []
    for index, update in enumerate(updates):
        stop = updates[index + 1] if index + 1 < len(updates) else frame_count
        scored.extend(range(update + 1, stop))
    return tuple(scored)


def score_fresh_pairwise_outcome_arrays(
    trajectories_m: Mapping[str, np.ndarray],
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    *,
    center_ids: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Score all frozen arms under identical hidden-identity conventions."""

    _require(set(trajectories_m) == set(ARMS), "fresh outcome arms changed")
    target = np.asarray(target_m)
    _require(
        target.ndim == 3
        and target.shape[0] == EXPECTED_FRAME_COUNT
        and target.shape[2] == 3,
        "fresh target must have shape (76, N, 3)",
    )
    scored_frames = _scored_frames()
    return {
        arm: score_deform360_hidden_trajectory(
            np.asarray(trajectories_m[arm]),
            target,
            visibility,
            validity,
            center_ids=np.asarray(center_ids, dtype=np.int64),
            scored_frames=scored_frames,
        )
        for arm in ARMS
    }


def _cluster_bootstrap(
    differences: Mapping[str, float],
    groups: Mapping[str, str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    object_ids = tuple(sorted(set(groups.values())))
    _require(len(object_ids) >= 2, "outcome bootstrap requires multiple objects")
    group_means = np.asarray(
        [
            np.mean(
                [
                    differences[case]
                    for case, object_id_for_case in groups.items()
                    if object_id_for_case == object_id
                ]
            )
            for object_id in object_ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    selected = rng.integers(
        0,
        len(group_means),
        size=(draws, len(group_means)),
    )
    bootstrap = np.mean(group_means[selected], axis=1)
    return {
        "episode_mean_difference_m": float(np.mean(list(differences.values()))),
        "object_balanced_mean_difference_m": float(np.mean(group_means)),
        "object_cluster_lower_95_m": float(np.quantile(bootstrap, 0.025)),
        "object_cluster_upper_95_m": float(np.quantile(bootstrap, 0.975)),
        "object_cluster_probability_improved": float(np.mean(bootstrap < 0.0)),
    }


def _relative_change(candidate: float, baseline: float) -> float | None:
    return None if baseline == 0.0 else candidate / baseline - 1.0


def summarize_fresh_pairwise_outcomes(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate one immutable fresh-object report per physical object."""

    _require(len(reports) == 12, "fresh outcome summary requires 12 reports")
    cases = [str(report["case"]) for report in reports]
    groups = {
        str(report["case"]): str(report["object_id"]) for report in reports
    }
    _require(
        len(set(cases)) == 12 and len(set(groups.values())) == 12,
        "fresh outcome reports are not 12 distinct physical objects",
    )
    aggregate = {
        arm: {
            metric: float(
                np.mean(
                    [
                        float(report["scores"][arm][metric])
                        for report in reports
                    ]
                )
            )
            for metric in PRIMARY_METRICS
        }
        for arm in ARMS
    }
    comparisons: dict[str, Any] = {}
    transfer_checks: dict[str, bool] = {}
    for comparator in COMPARATORS:
        metric_results: dict[str, Any] = {}
        metric_passes: list[bool] = []
        for metric in PRIMARY_METRICS:
            differences = {
                str(report["case"]): float(
                    report["scores"][CANDIDATE_ARM][metric]
                    - report["scores"][comparator][metric]
                )
                for report in reports
            }
            result = _cluster_bootstrap(differences, groups)
            result["relative_change"] = _relative_change(
                aggregate[CANDIDATE_ARM][metric],
                aggregate[comparator][metric],
            )
            result["wins"] = sum(value < 0.0 for value in differences.values())
            result["ties"] = sum(value == 0.0 for value in differences.values())
            result["regressions"] = sum(
                value > 0.0 for value in differences.values()
            )
            result["transfer_gate_passed"] = bool(
                result["object_balanced_mean_difference_m"] < 0.0
                and result["object_cluster_upper_95_m"] < 0.0
            )
            metric_passes.append(result["transfer_gate_passed"])
            metric_results[metric] = result
        comparison_passed = all(metric_passes)
        comparison_name = f"{CANDIDATE_ARM}_vs_{comparator}"
        transfer_checks[comparison_name] = comparison_passed
        comparisons[comparison_name] = {
            "metrics": metric_results,
            "joint_two_metric_wins": sum(
                all(
                    report["scores"][CANDIDATE_ARM][metric]
                    < report["scores"][comparator][metric]
                    for metric in PRIMARY_METRICS
                )
                for report in reports
            ),
            "joint_two_metric_regressions": sum(
                all(
                    report["scores"][CANDIDATE_ARM][metric]
                    > report["scores"][comparator][metric]
                    for metric in PRIMARY_METRICS
                )
                for report in reports
            ),
            "transfer_gate_passed": comparison_passed,
        }
    category_aggregate = {
        category: {
            arm: {
                metric: float(
                    np.mean(
                        [
                            report["scores"][arm][metric]
                            for report in reports
                            if report["category"] == category
                        ]
                    )
                )
                for metric in PRIMARY_METRICS
            }
            for arm in ARMS
        }
        for category in sorted({str(report["category"]) for report in reports})
    }
    return {
        "aggregate": aggregate,
        "category_aggregate": category_aggregate,
        "comparisons": comparisons,
        "transfer_gate": {
            "definition": (
                "candidate object-balanced mean difference and 95% "
                "object-bootstrap upper bound are both below zero for both "
                "primary metrics against each comparator"
            ),
            "checks": transfer_checks,
            "passed": all(transfer_checks.values()),
        },
    }


def _resolve_bound_file(
    local_path: Path,
    bound_record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    _require(local_path.is_file(), f"missing {label}: {local_path}")
    _require(
        file_sha256(local_path) == bound_record.get("file_sha256"),
        f"{label} checksum changed",
    )
    return local_path


def _load_prediction_case(
    prediction_dir: Path,
    *,
    protocol_config_sha256: str,
    cohort_lock_sha256: str,
    expected_case: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    seal_path = prediction_dir / "belief_prediction_seal.json"
    seal = load_json(seal_path)
    validate_belief_prediction_seal(
        seal,
        protocol_config_sha256=protocol_config_sha256,
        cohort_lock_sha256=cohort_lock_sha256,
    )
    _require(seal.get("case") == expected_case, "prediction case changed")
    report_path = _resolve_bound_file(
        prediction_dir / "belief_prediction_report.json",
        seal["prediction_report"],
        label="prediction report",
    )
    archive_path = _resolve_bound_file(
        prediction_dir / "belief_prediction.npz",
        seal["prediction_archive"],
        label="prediction archive",
    )
    report = load_json(report_path)
    _require(
        report.get("artifact_kind")
        == "Deform360FreshPairwiseBeliefPrediction"
        and report.get("case") == expected_case
        and report.get("result_sha256")
        == canonical_sha256(report, digest_key="result_sha256"),
        "prediction report is incompatible",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        required = set(ARCHIVE_ARRAY_KEYS.values()) | {"center_ids"}
        _require(required == set(stored.files), "prediction archive arrays changed")
        arrays = {
            name: np.asarray(stored[name]).copy() for name in sorted(required)
        }
    expected_array_hashes = report.get("prediction_archive", {}).get(
        "array_sha256"
    )
    _require(
        isinstance(expected_array_hashes, Mapping)
        and {
            name: array_sha256(value) for name, value in sorted(arrays.items())
        }
        == dict(expected_array_hashes),
        "prediction archive array checksum changed",
    )
    return seal, report, arrays


def evaluate_fresh_pairwise_outcomes(
    *,
    repository_root: str | Path,
    protocol_path: str | Path,
    cohort_path: str | Path,
    admission_root: str | Path,
    prediction_root: str | Path,
    processed_root: str | Path,
    barrier_path: str | Path,
    analysis_path: str | Path,
    output_dir: str | Path,
    operator_path: str | Path,
) -> dict[str, Any]:
    """Open the frozen future payloads once, after validating every seal."""

    repository = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    cohort_file = Path(cohort_path).resolve()
    admissions = Path(admission_root).resolve()
    predictions = Path(prediction_root).resolve()
    processed = Path(processed_root).resolve()
    barrier_file = Path(barrier_path).resolve()
    analysis_file = Path(analysis_path).resolve()
    output = Path(output_dir).resolve()
    operator = Path(operator_path).resolve()
    _require(not output.exists(), "fresh outcome output already exists")
    protocol = load_fresh_pairwise_protocol(
        protocol_file,
        repository_root=repository,
    )
    cohort = load_bound_cohort(cohort_file, protocol)
    barrier = validate_completeness_barrier(
        barrier_file,
        protocol_path=protocol_file,
        cohort_path=cohort_file,
        prediction_root=predictions,
    )
    analysis = load_fresh_pairwise_outcome_analysis(
        analysis_file,
        protocol_path=protocol_file,
        cohort_path=cohort_file,
        barrier_path=barrier_file,
    )

    # Validate and bind all predictions and future payload byte hashes before
    # deserializing the first future object trajectory.
    bound_cases: list[dict[str, Any]] = []
    for case_record in cohort["cases"]:
        case = str(case_record["case"])
        admission_path = admissions / f"{case}.admission.json"
        admission = load_json(admission_path)
        validate_fresh_source_admission(admission)
        _require(
            admission.get("accepted") is True
            and admission.get("admission_sha256")
            == case_record["admission_sha256"],
            f"source admission changed: {case}",
        )
        target_path = (
            processed
            / str(case_record["object_id"])
            / f"episode_{int(case_record['episode_id']):04d}"
            / "final_data.pkl"
        )
        _require(target_path.is_file(), f"missing future payload: {case}")
        target_sha256 = file_sha256(target_path)
        _require(
            target_sha256
            == admission["source_files"]["future_payload"]["sha256"],
            f"future payload checksum changed: {case}",
        )
        seal, report, arrays = _load_prediction_case(
            predictions / case,
            protocol_config_sha256=protocol["config_file_sha256"],
            cohort_lock_sha256=cohort["cohort_lock_sha256"],
            expected_case=case,
        )
        bound_cases.append(
            {
                "case_record": dict(case_record),
                "admission_path": admission_path,
                "admission": admission,
                "target_path": target_path,
                "target_sha256": target_sha256,
                "prediction_seal": seal,
                "prediction_report": report,
                "prediction_arrays": arrays,
            }
        )

    reports: list[dict[str, Any]] = []
    for bound in bound_cases:
        case_record = bound["case_record"]
        case = str(case_record["case"])
        with bound["target_path"].open("rb") as handle:
            target_payload = pickle.load(handle)
        _require(isinstance(target_payload, Mapping), f"invalid target payload: {case}")
        for name in (
            "object_points",
            "object_visibilities",
            "object_motions_valid",
        ):
            _require(name in target_payload, f"target lacks {name}: {case}")
        target = np.asarray(target_payload["object_points"])
        visibility = np.asarray(
            target_payload["object_visibilities"], dtype=bool
        )
        validity = np.asarray(target_payload["object_motions_valid"], dtype=bool)
        prediction_arrays = bound["prediction_arrays"]
        trajectories = {
            arm: np.asarray(prediction_arrays[archive_key])
            for arm, archive_key in ARCHIVE_ARRAY_KEYS.items()
        }
        first = trajectories[PHYSICAL_ARM]
        _require(
            all(value.shape == first.shape for value in trajectories.values())
            and first.shape == target.shape
            and target.shape[0] == EXPECTED_FRAME_COUNT
            and target.shape[2] == 3,
            f"target/prediction shape changed: {case}",
        )
        _require(
            np.array_equal(first[0].astype(np.float32), target[0].astype(np.float32)),
            f"target frame-zero identities changed: {case}",
        )
        scores = score_fresh_pairwise_outcome_arrays(
            trajectories,
            target,
            visibility,
            validity,
            center_ids=prediction_arrays["center_ids"],
        )
        prediction_report = bound["prediction_report"]
        reports.append(
            {
                "schema_version": 1,
                "artifact_kind": OUTCOME_KIND,
                "protocol_id": protocol["protocol_id"],
                "case": case,
                "object_id": str(case_record["object_id"]),
                "episode_id": int(case_record["episode_id"]),
                "category": str(case_record["category"]),
                "scores": scores,
                "source_only_decisions": prediction_report["method_report"][
                    "updates"
                ],
                "inputs": {
                    "source_admission": {
                        "path": str(bound["admission_path"]),
                        "file_sha256": file_sha256(bound["admission_path"]),
                        "admission_sha256": bound["admission"][
                            "admission_sha256"
                        ],
                    },
                    "future_payload": {
                        "path": str(bound["target_path"]),
                        "file_sha256": bound["target_sha256"],
                    },
                    "belief_prediction_seal": {
                        "result_sha256": bound["prediction_seal"][
                            "result_sha256"
                        ],
                    },
                },
                "information_boundary": {
                    "prediction_sealed_before_future_outcome": True,
                    "all_case_barrier_validated_before_future_deserialization": True,
                    "future_payload_opened_for_scoring": True,
                    "official_deform360_parity_claim": False,
                },
                "claim_boundary": (
                    "fresh-object transfer under explicit hidden-identity candidate "
                    "metric conventions; not official Deform360 3-D SOTA"
                ),
            }
        )
    aggregate = summarize_fresh_pairwise_outcomes(reports)
    repository_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    staging = output.with_name(f"{output.name}.staging-{os.getpid()}")
    _require(not staging.exists(), "fresh outcome staging path already exists")
    staging.mkdir(parents=True)
    try:
        artifacts = []
        for report in reports:
            report["result_sha256"] = canonical_sha256(
                report, digest_key="result_sha256"
            )
            report_path = write_json(staging / f"{report['case']}.json", report)
            artifacts.append(
                {
                    "case": report["case"],
                    "report_file": report_path.name,
                    "report_file_sha256": file_sha256(report_path),
                    "report_result_sha256": report["result_sha256"],
                }
            )
        summary: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": SUMMARY_KIND,
            "protocol_id": protocol["protocol_id"],
            "repository_revision": repository_revision,
            "protocol_config_sha256": protocol["config_file_sha256"],
            "cohort_lock_sha256": cohort["cohort_lock_sha256"],
            "barrier": {
                "path": str(barrier_file),
                "file_sha256": file_sha256(barrier_file),
                "result_sha256": barrier["result_sha256"],
            },
            "analysis_lock": {
                "path": str(analysis_file),
                "file_sha256": file_sha256(analysis_file),
                "analysis_id": analysis["analysis_id"],
            },
            "operator": {
                "path": str(operator),
                "file_sha256": file_sha256(operator),
                "evaluator_source_file_sha256": file_sha256(Path(__file__)),
            },
            "case_count": len(reports),
            "physical_object_count": len(
                {str(report["object_id"]) for report in reports}
            ),
            "scoring": {
                "update_frames": list(UPDATE_FRAMES),
                "scored_frames": list(_scored_frames()),
                "center_policy": (
                    "exclude the 16 source-selected measurement identities from "
                    "both primary metrics"
                ),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            **aggregate,
            "calibration_claim": {
                "evaluated": False,
                "reason": (
                    "the sealed candidate archive contains point predictions but "
                    "no frozen predictive covariance"
                ),
            },
            "official_sota_claim": {
                "allowed": False,
                "reason": (
                    "the cohort lock permits only fresh-object candidate metric "
                    "conventions until official Deform360 evaluator parity is "
                    "resolved"
                ),
            },
            "artifacts": artifacts,
            "information_boundary": {
                "all_predictions_hashed_before_future_outcome": True,
                "barrier_validated_before_first_future_deserialization": True,
                "future_payloads_opened_once_by_this_operator": True,
            },
            "claim_boundary": (
                "prospective fresh-object transfer under candidate hidden-identity "
                "metrics; neither calibrated uncertainty nor official Deform360 "
                "SOTA is claimed"
            ),
        }
        summary["result_sha256"] = canonical_sha256(
            summary, digest_key="result_sha256"
        )
        write_json(staging / "summary.json", summary)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


__all__ = [
    "ARMS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "COMPARATORS",
    "OUTCOME_KIND",
    "PRIMARY_METRICS",
    "SUMMARY_KIND",
    "evaluate_fresh_pairwise_outcomes",
    "load_fresh_pairwise_outcome_analysis",
    "score_fresh_pairwise_outcome_arrays",
    "summarize_fresh_pairwise_outcomes",
]
