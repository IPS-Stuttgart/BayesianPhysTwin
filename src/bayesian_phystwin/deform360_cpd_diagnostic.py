"""Independent non-rigid CPD controls for the open Deform360 panel.

The diagnostic deliberately reuses the already-audited open-27 loader,
measurement centres, update frames, risk decisions, continuation decisions,
hidden-identity support, and score implementation.  CPD receives only the
unordered point sets available at the current update and is refit from scratch
at every update.  It never carries a deformation state between observations.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from .cpd_registration import NonrigidCpdConfig, fit_nonrigid_cpd
from .deform360_online_belief_evaluation import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPECTED_SOURCE_EPISODES,
    PRIMARY_METRICS,
    UPDATE_FRAMES,
    _expected_episode_directories,
    _physical_object_cluster_bootstrap,
    _relative_change,
    _sha256,
    evaluate_deform360_online_belief_case,
    score_deform360_hidden_trajectory,
)


PROTOCOL_ID = "deform360-open27-independent-nonrigid-cpd-control-v2-development"
CPD_ARMS = (
    "independent_cpd_ungated",
    "independent_cpd_frozen_current",
    "independent_cpd_observed_backbone",
    "independent_cpd_risk_limited",
    "independent_cpd_matched_selector",
)
REFERENCE_ARMS = (
    "physical_prior",
    "persistence",
    "recursive_rbf_ungated",
    "recursive_rbf_risk_limited",
    "recursive_rbf_causal_continuation",
)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _symmetric_set_chamfer_m(first_m: np.ndarray, second_m: np.ndarray) -> float:
    """Return a small-set symmetric Euclidean Chamfer distance."""

    first = np.asarray(first_m, dtype=float)
    second = np.asarray(second_m, dtype=float)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[1:] != (3,)
        or second.shape[1:] != (3,)
        or len(first) == 0
        or len(second) == 0
    ):
        raise ValueError("Chamfer inputs must have nonempty shape (N, 3)")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Chamfer inputs must be finite")
    distances = np.linalg.norm(first[:, None] - second[None], axis=2)
    return 0.5 * (
        float(np.mean(np.min(distances, axis=1)))
        + float(np.mean(np.min(distances, axis=0)))
    )


def evaluate_deform360_cpd_case(
    episode_dir: str | Path,
    *,
    config: NonrigidCpdConfig | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate matched independent-CPD arms on one audited open episode."""

    baseline_report, baseline_arrays = evaluate_deform360_online_belief_case(
        episode_dir
    )
    cfg = config or NonrigidCpdConfig()
    prior = np.asarray(baseline_arrays["physical_prior_m"])
    persistence = np.asarray(baseline_arrays["persistence_m"])
    centers = np.asarray(baseline_arrays["center_ids"], dtype=np.int64)
    target_path = Path(str(baseline_report["inputs"]["target_data"]["path"])).resolve()
    if _sha256(target_path) != baseline_report["inputs"]["target_data"]["sha256"]:
        raise ValueError("audited target payload changed after baseline validation")
    target_data = _load_pickle(target_path)
    target = np.asarray(target_data["object_points"], dtype=float)
    visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
    validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
    if target.shape != prior.shape:
        raise ValueError("audited target no longer matches the physical prior")

    output_dtype = prior.dtype
    trajectories = {name: prior.copy() for name in CPD_ARMS}
    updates: list[dict[str, object]] = []
    baseline_updates = list(baseline_report["updates"])
    if [int(value["frame"]) for value in baseline_updates] != list(UPDATE_FRAMES):
        raise ValueError("baseline report does not use the fixed update frames")

    for update_index, (update, baseline_update) in enumerate(
        zip(UPDATE_FRAMES, baseline_updates, strict=True)
    ):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(target)
        )
        available = (
            visibility[update, centers]
            & validity[update, centers]
            & np.all(np.isfinite(target[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
        )
        available_ids = centers[available]
        if available_ids.tolist() != baseline_update["available_center_ids"]:
            raise ValueError("CPD observation support differs from audited baseline")
        can_fit = len(available_ids) >= 3
        accepted = bool(baseline_update["accepted"])
        continuation_selected = baseline_update["causal_continuation_selected"]
        if accepted and continuation_selected is None:
            raise ValueError("accepted baseline update lacks continuation decision")

        transform = None
        selected_transform = None
        physical_current_chamfer_m = None
        persistence_current_chamfer_m = None
        selected_backbone = None
        if can_fit:
            current_target = target[update, available_ids]
            transform = fit_nonrigid_cpd(
                prior[update, available_ids],
                current_target,
                config=cfg,
            )
            registered_current = transform.transform(prior[update]).astype(
                output_dtype,
                copy=False,
            )
            for frame in range(update + 1, stop):
                registered_future = transform.transform(prior[frame]).astype(
                    output_dtype,
                    copy=False,
                )
                trajectories["independent_cpd_ungated"][frame] = registered_future
                trajectories["independent_cpd_frozen_current"][frame] = (
                    registered_current
                )
                if accepted:
                    trajectories["independent_cpd_risk_limited"][frame] = (
                        registered_future
                    )
                    trajectories["independent_cpd_matched_selector"][frame] = (
                        registered_future
                        if bool(continuation_selected)
                        else registered_current
                    )

            physical_current_chamfer_m = _symmetric_set_chamfer_m(
                prior[update, available_ids],
                current_target,
            )
            persistence_current_chamfer_m = _symmetric_set_chamfer_m(
                persistence[update, available_ids],
                current_target,
            )
            selected_backbone = (
                "physical_prior"
                if physical_current_chamfer_m <= persistence_current_chamfer_m
                else "persistence"
            )
            selected_trajectory = (
                prior if selected_backbone == "physical_prior" else persistence
            )
            selected_transform = fit_nonrigid_cpd(
                selected_trajectory[update, available_ids],
                current_target,
                config=cfg,
            )
            for frame in range(update + 1, stop):
                trajectories["independent_cpd_observed_backbone"][frame] = (
                    selected_transform.transform(selected_trajectory[frame]).astype(
                        output_dtype,
                        copy=False,
                    )
                )

        for arm in (
            "independent_cpd_risk_limited",
            "independent_cpd_matched_selector",
        ):
            if not accepted and not np.array_equal(
                trajectories[arm][update + 1 : stop],
                prior[update + 1 : stop],
            ):
                raise AssertionError(f"{arm} violated exact prior fallback")
        if not can_fit:
            for arm in (
                "independent_cpd_ungated",
                "independent_cpd_frozen_current",
                "independent_cpd_observed_backbone",
            ):
                if not np.array_equal(
                    trajectories[arm][update + 1 : stop],
                    prior[update + 1 : stop],
                ):
                    raise AssertionError(
                        f"{arm} violated insufficient-support fallback"
                    )

        updates.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "available_center_count": int(len(available_ids)),
                "available_center_ids": available_ids.tolist(),
                "fit_performed": can_fit,
                "risk_gate_accepted": accepted,
                "causal_continuation_selected": continuation_selected,
                "current_observation_backbone_selection": {
                    "metric": "symmetric set Chamfer on current observed centres",
                    "physical_prior_m": physical_current_chamfer_m,
                    "persistence_m": persistence_current_chamfer_m,
                    "tie_break": "physical_prior",
                    "selected": selected_backbone,
                },
                "fit": (
                    None
                    if transform is None
                    else {
                        "iterations": transform.iterations,
                        "converged": transform.converged,
                        "source_rms_scale_m": transform.scale_m,
                        "final_variance_normalized2": (transform.variance_normalized2),
                        "effective_correspondence_count": (
                            transform.effective_correspondence_count
                        ),
                    }
                ),
                "selected_backbone_fit": (
                    None
                    if selected_transform is None
                    else {
                        "iterations": selected_transform.iterations,
                        "converged": selected_transform.converged,
                        "source_rms_scale_m": selected_transform.scale_m,
                        "final_variance_normalized2": (
                            selected_transform.variance_normalized2
                        ),
                        "effective_correspondence_count": (
                            selected_transform.effective_correspondence_count
                        ),
                    }
                ),
            }
        )

    scored_frames = tuple(int(value) for value in baseline_report["scored_frames"])
    scores = {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visibility,
            validity,
            center_ids=centers,
            scored_frames=scored_frames,
        )
        for arm, trajectory in trajectories.items()
    }
    for arm in REFERENCE_ARMS:
        scores[arm] = baseline_report["scores"][arm]

    report: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "case": baseline_report["case"],
        "object_id": baseline_report["object_id"],
        "episode_id": baseline_report["episode_id"],
        "center_ids": centers.tolist(),
        "update_frames": list(UPDATE_FRAMES),
        "scored_frames": list(scored_frames),
        "cpd_config": asdict(cfg),
        "updates": updates,
        "scores": scores,
        "inputs": baseline_report["inputs"],
        "information_boundary": {
            **baseline_report["information_boundary"],
            "registration_observation": (
                "unordered set of the same sparse fused material centres at the "
                "current update only"
            ),
            "fit_state_across_updates": "none; CPD is reinitialized each update",
            "scoring_centres_permanently_excluded": True,
        },
        "control_contract": {
            "algorithm": "non-rigid coherent point drift",
            "reference": "Myronenko and Song, IEEE TPAMI 2010",
            "coordinate_normalization": (
                "source centroid and source RMS radius; target uses the same transform"
            ),
            "query_rule": (
                "apply the current fitted CPD spatial map to each future physical-"
                "prior frame in the update interval"
            ),
            "frozen_current_control": (
                "repeat the CPD-registered physical state at the update over the "
                "entire next interval"
            ),
            "observed_backbone_control": (
                "select physical prior versus persistence independently at every "
                "update by current-centre symmetric set Chamfer, then fit CPD and "
                "query the selected continuation; no future observation is used"
            ),
            "ungated_support_rule": "fit iff at least three centres are available",
            "matched_risk_rule": "reuse the audited online-belief gate decision",
            "matched_selector_rule": (
                "reuse the audited causal continuation decision; otherwise repeat "
                "the CPD-registered current state"
            ),
        },
    }
    arrays = {
        "center_ids": centers,
        "physical_prior_m": prior,
        "persistence_m": persistence,
        "recursive_rbf_ungated_m": baseline_arrays["recursive_rbf_ungated_m"],
        "recursive_rbf_risk_limited_m": baseline_arrays["recursive_rbf_risk_limited_m"],
        "recursive_rbf_causal_continuation_m": baseline_arrays[
            "recursive_rbf_causal_continuation_m"
        ],
        **{f"{arm}_m": trajectory for arm, trajectory in trajectories.items()},
    }
    return report, arrays


def _comparison(
    reports: list[dict[str, object]],
    aggregate: Mapping[str, Mapping[str, float]],
    groups: Mapping[str, str],
    candidate: str,
    comparator: str,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for metric in PRIMARY_METRICS:
        differences = {
            str(report["case"]): float(
                report["scores"][candidate][metric]
                - report["scores"][comparator][metric]
            )
            for report in reports
        }
        result = _physical_object_cluster_bootstrap(differences, groups)
        result["relative_change"] = _relative_change(
            aggregate[candidate][metric],
            aggregate[comparator][metric],
        )
        result["episode_wins"] = int(
            np.sum(np.asarray(list(differences.values())) < 0.0)
        )
        metrics[metric] = result
    return {
        "metrics": metrics,
        "joint_two_metric_episode_wins": int(
            sum(
                all(
                    report["scores"][candidate][metric]
                    < report["scores"][comparator][metric]
                    for metric in PRIMARY_METRICS
                )
                for report in reports
            )
        ),
    }


def evaluate_deform360_cpd_cohort(
    root: str | Path,
    output: str | Path,
    *,
    config: NonrigidCpdConfig | None = None,
) -> dict[str, object]:
    """Persist the independent-CPD controls for exactly the open 27 episodes."""

    cohort_root = Path(root).resolve()
    output_dir = Path(output).resolve()
    expected = _expected_episode_directories()
    if len(expected) != 27:
        raise AssertionError("fixed source panel no longer contains 27 episodes")
    missing = [name for name in expected if not (cohort_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing fixed Deform360 episodes: {missing}")
    output_dir.mkdir(parents=True, exist_ok=False)

    reports: list[dict[str, object]] = []
    groups: dict[str, str] = {}
    artifacts: list[dict[str, object]] = []
    for case_name in expected:
        report, arrays = evaluate_deform360_cpd_case(
            cohort_root / case_name,
            config=config,
        )
        groups[case_name] = str(report["object_id"])
        report_path = output_dir / f"{case_name}.json"
        arrays_path = output_dir / f"{case_name}.npz"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(arrays_path, **arrays)
        artifacts.append(
            {
                "case": case_name,
                "report_sha256": _sha256(report_path),
                "arrays_sha256": _sha256(arrays_path),
            }
        )
        reports.append(report)

    arms = CPD_ARMS + REFERENCE_ARMS
    aggregate = {
        arm: {
            metric: float(
                np.mean([report["scores"][arm][metric] for report in reports])
            )
            for metric in PRIMARY_METRICS
        }
        for arm in arms
    }
    comparisons = {
        f"{candidate}_vs_{comparator}": _comparison(
            reports,
            aggregate,
            groups,
            candidate,
            comparator,
        )
        for candidate, comparator in (
            ("recursive_rbf_causal_continuation", "independent_cpd_ungated"),
            ("recursive_rbf_ungated", "independent_cpd_ungated"),
            ("recursive_rbf_causal_continuation", "independent_cpd_risk_limited"),
            ("recursive_rbf_causal_continuation", "independent_cpd_matched_selector"),
            ("recursive_rbf_risk_limited", "independent_cpd_risk_limited"),
            ("independent_cpd_ungated", "independent_cpd_frozen_current"),
            ("independent_cpd_observed_backbone", "independent_cpd_ungated"),
            ("independent_cpd_observed_backbone", "independent_cpd_frozen_current"),
            ("independent_cpd_ungated", "physical_prior"),
            ("independent_cpd_frozen_current", "physical_prior"),
            ("independent_cpd_observed_backbone", "physical_prior"),
            ("independent_cpd_observed_backbone", "persistence"),
            ("independent_cpd_risk_limited", "physical_prior"),
            ("independent_cpd_matched_selector", "physical_prior"),
        )
    }
    fit_count = int(
        sum(
            bool(update["fit_performed"])
            for report in reports
            for update in report["updates"]
        )
    )
    converged_count = int(
        sum(
            bool(update["fit"] is not None and update["fit"]["converged"])
            for report in reports
            for update in report["updates"]
        )
    )
    selected_backbone_fit_count = int(
        sum(
            update["selected_backbone_fit"] is not None
            for report in reports
            for update in report["updates"]
        )
    )
    selected_backbone_converged_count = int(
        sum(
            bool(
                update["selected_backbone_fit"] is not None
                and update["selected_backbone_fit"]["converged"]
            )
            for report in reports
            for update in report["updates"]
        )
    )
    selected_backbone_counts = {
        backbone: int(
            sum(
                update["current_observation_backbone_selection"]["selected"] == backbone
                for report in reports
                for update in report["updates"]
            )
        )
        for backbone in ("physical_prior", "persistence")
    }
    cfg = config or NonrigidCpdConfig()
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cohort_root": str(cohort_root),
        "episode_count": len(reports),
        "physical_object_count": len(set(groups.values())),
        "physical_objects": {
            key: list(value) for key, value in EXPECTED_SOURCE_EPISODES.items()
        },
        "cpd_config": asdict(cfg),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "fit_diagnostics": {
            "fit_count": fit_count,
            "converged_count": converged_count,
            "convergence_fraction": converged_count / fit_count,
            "selected_backbone_fit_count": selected_backbone_fit_count,
            "selected_backbone_converged_count": selected_backbone_converged_count,
            "selected_backbone_convergence_fraction": (
                selected_backbone_converged_count / selected_backbone_fit_count
            ),
            "selected_backbone_counts": selected_backbone_counts,
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "physical object",
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "development-only classical registration control on the already-open "
            "independent-source Deform360 panel; fused material-track measurements; "
            "not a held-target or official Table-4 result"
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "CPD_ARMS",
    "PROTOCOL_ID",
    "evaluate_deform360_cpd_case",
    "evaluate_deform360_cpd_cohort",
]
