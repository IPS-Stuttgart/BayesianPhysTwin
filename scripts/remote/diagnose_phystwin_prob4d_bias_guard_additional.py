#!/usr/bin/env python3
"""Run the unchanged Prob4D bias guard on the opened additional cloth source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.phystwin_official_evaluation import _nearest_distances
from bayesian_phystwin.phystwin_prob4d_action_guard import (
    Prob4DActionGuardConfig,
    build_guarded_action_conditioned_prob4d_candidate,
)
from bayesian_phystwin.phystwin_prob4d_bias_guard import (
    Prob4DBiasGuardConfig,
    build_guarded_prob4d_prefix_candidate,
)


BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260721


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--assimilation-root", type=Path, required=True)
    parser.add_argument("--selected-baseline-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-family",
        choices=("static", "action_conditioned"),
        default="static",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as error:
        if error.name != "numpy._core.numeric":
            raise
        import numpy.core as numpy_core
        import numpy.core.numeric as numpy_core_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
        with path.open("rb") as handle:
            return pickle.load(handle)


def _chamfer_m(
    trajectory: np.ndarray,
    final_data: Mapping[str, Any],
    *,
    start_frame: int,
    end_frame: int,
) -> float:
    object_points = np.asarray(final_data["object_points"], dtype=np.float64)
    visible = np.asarray(final_data["object_visibilities"], dtype=bool)
    surface_count = object_points.shape[1] + len(
        np.asarray(final_data["surface_points"])
    )
    values = []
    for frame in range(start_frame, end_frame):
        observed = object_points[frame, visible[frame]]
        distance, _ = _nearest_distances(
            np.asarray(trajectory[frame, :surface_count], dtype=np.float64),
            observed,
            p=1,
        )
        values.append(float(np.mean(distance)))
    return float(np.mean(values))


def _garment(case: str) -> str:
    for suffix in ("_fold", "_lift"):
        if case.endswith(suffix):
            return case[: -len(suffix)]
    return case


def _case_result(
    row: Mapping[str, Any],
    *,
    assimilation_root: Path,
    selected_baseline_root: Path,
    source_lock: Mapping[str, Any],
    candidate_family: str,
) -> dict[str, Any]:
    case = str(row["case"])
    case_dir = assimilation_root / case / "C_decoupled_robust"
    summary_path = case_dir / "summary.json"
    assimilation_path = case_dir / "assimilation.npz"
    if _sha256(summary_path) != row["C_summary_sha256"]:
        raise ValueError(f"arm-C summary checksum changed: {case}")
    if _sha256(assimilation_path) != row["C_assimilation_npz_sha256"]:
        raise ValueError(f"arm-C assimilation checksum changed: {case}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inputs = summary["inputs"]
    physical_path = Path(inputs["baseline"]["path"])
    final_data_path = Path(inputs["final_data"]["path"])
    split_path = Path(inputs["split"]["path"])
    for role, descriptor, path in (
        ("physical baseline", inputs["baseline"], physical_path),
        ("final data", inputs["final_data"], final_data_path),
        ("split", inputs["split"], split_path),
    ):
        if _sha256(path) != descriptor["sha256"]:
            raise ValueError(f"{role} checksum changed: {case}")
    selected_path = selected_baseline_root / case / "trajectory.pkl"
    selected = np.asarray(_load_pickle(selected_path), dtype=np.float32)
    physical = np.asarray(_load_pickle(physical_path), dtype=np.float32)
    final_data = _load_pickle(final_data_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    test_end = int(split["test"][1])
    if int(summary["train_end_frame"]) != train_end:
        raise ValueError(f"arm-C and split train boundaries differ: {case}")
    with np.load(assimilation_path, allow_pickle=False) as stored:
        positions = np.asarray(stored["position_flow_positions"])
        validity = np.asarray(stored["position_flow_valid"], dtype=bool)
        reliability = np.asarray(stored["position_flow_prior_reliability"])
        covariance = np.asarray(
            stored["position_flow_observation_covariance_m2"]
        )
        frame_indices = np.asarray(stored["frame_indices"], dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError(f"arm-C frame map is not identity: {case}")
    observed_count = positions.shape[1]
    physical_count = physical.shape[1]
    if observed_count > physical_count:
        raise ValueError(f"arm-C observations exceed physical state: {case}")

    padded_positions = np.full(
        (len(positions), physical_count, 3), np.nan, dtype=np.float32
    )
    padded_validity = np.zeros((len(positions), physical_count), dtype=bool)
    padded_reliability = np.full(
        (len(positions), physical_count), np.nan, dtype=np.float32
    )
    padded_covariance = np.full(
        (len(positions), physical_count, 3, 3), np.nan, dtype=np.float32
    )
    padded_positions[:, :observed_count] = positions
    padded_validity[:, :observed_count] = validity
    padded_reliability[:, :observed_count] = reliability
    padded_covariance[:, :observed_count] = covariance
    object_points = np.asarray(final_data["object_points"])
    surface_count = object_points.shape[1] + len(
        np.asarray(final_data["surface_points"])
    )
    arguments = (
        selected,
        physical[:train_end],
        padded_positions[:train_end],
        padded_validity[:train_end],
        padded_reliability[:train_end],
        padded_covariance[:train_end],
        object_points[:train_end],
        np.asarray(final_data["object_visibilities"], dtype=bool)[:train_end],
        np.asarray(final_data["object_motions_valid"], dtype=bool)[:train_end],
    )
    keywords = {
        "num_surface_points": surface_count,
        "source_lock": source_lock,
    }
    if candidate_family == "static":
        report, candidate, guarded = build_guarded_prob4d_prefix_candidate(
            *arguments,
            **keywords,
            config=Prob4DBiasGuardConfig(),
        )
        update = report["candidate"]["updates"][0]
    else:
        report, candidate, guarded = (
            build_guarded_action_conditioned_prob4d_candidate(
                *arguments,
                **keywords,
                config=Prob4DActionGuardConfig(),
            )
        )
        update = report["static_candidate_report"]["candidate"]["updates"][0]
    late_start = train_end + (2 * (test_end - train_end)) // 3
    scores = {
        arm: {
            "future_chamfer_m": _chamfer_m(
                trajectory,
                final_data,
                start_frame=train_end,
                end_frame=test_end,
            ),
            "late_chamfer_m": _chamfer_m(
                trajectory,
                final_data,
                start_frame=late_start,
                end_frame=test_end,
            ),
        }
        for arm, trajectory in (
            ("selected_baseline", selected),
            ("raw_candidate", candidate),
            ("guarded_candidate", guarded),
        )
    }
    difference = (
        scores["guarded_candidate"]["future_chamfer_m"]
        - scores["selected_baseline"]["future_chamfer_m"]
    )
    accepted = bool(report["candidate_accepted"])
    exact_fallback = accepted or np.array_equal(guarded, selected)
    if not exact_fallback:
        raise AssertionError(f"rejected case changed selected baseline: {case}")
    result = {
        "case": case,
        "garment": _garment(case),
        "candidate_available": bool(report["candidate_available"]),
        "candidate_accepted": accepted,
        "bit_exact_fallback": exact_fallback,
        "accepted_harmful": bool(accepted and difference > 0.0),
        "guarded_minus_selected_future_chamfer_m": difference,
        "scores": scores,
        "target_free_diagnostic": {
            "validation": report["validation"],
            "available_center_count": update["available_center_count"],
            "motion_center_count": update["motion_center_count"],
            "physical_response_rms_m": update["physical_response_rms_m"],
            "observed_motion_rms_m": update["observed_motion_rms_m"],
            "causal_physical_agreement_gain": update[
                "causal_physical_agreement_gain"
            ],
            "minimum_identifiable_fraction": update.get(
                "minimum_identifiable_fraction"
            ),
            "maximum_correction_m": update.get("maximum_correction_m"),
            "fallback_reason": update.get("fallback_reason"),
        },
        "inputs_sha256": {
            "arm_c_summary": _sha256(summary_path),
            "arm_c_assimilation": _sha256(assimilation_path),
            "selected_baseline": _sha256(selected_path),
            "physical_baseline": _sha256(physical_path),
            "final_data": _sha256(final_data_path),
            "split": _sha256(split_path),
        },
    }
    if candidate_family == "action_conditioned":
        result["action_conditioned_diagnostic"] = {
            "progress": report["progress"],
            "validation": report["validation"],
        }
    return result


def _cluster_interval(cases: list[dict[str, Any]]) -> dict[str, float]:
    by_garment: dict[str, list[float]] = {}
    for case in cases:
        by_garment.setdefault(case["garment"], []).append(
            case["guarded_minus_selected_future_chamfer_m"]
        )
    values = np.asarray(
        [np.mean(group) for group in by_garment.values()], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.mean(
        rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True),
        axis=1,
    )
    return {
        "garment_count": len(values),
        "mean_difference_m": float(np.mean(values)),
        "lower_95_m": float(np.quantile(draws, 0.025)),
        "upper_95_m": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    args = _parse_args()
    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    cases = [
        _case_result(
            row,
            assimilation_root=args.assimilation_root,
            selected_baseline_root=args.selected_baseline_root,
            source_lock=source_lock,
            candidate_family=args.candidate_family,
        )
        for row in source_summary["inputs"]
    ]
    arms = ("selected_baseline", "raw_candidate", "guarded_candidate")
    aggregate = {
        arm: {
            metric: float(np.mean([case["scores"][arm][metric] for case in cases]))
            for metric in ("future_chamfer_m", "late_chamfer_m")
        }
        for arm in arms
    }
    accepted = sum(case["candidate_accepted"] for case in cases)
    harmful = sum(case["accepted_harmful"] for case in cases)
    payload = {
        "artifact_kind": (
            "PhysTwinProb4DBiasGuardAdditionalSourceDiagnostic"
            if args.candidate_family == "static"
            else "PhysTwinProb4DActionGuardAdditionalSourceDiagnostic"
        ),
        "schema_version": 1,
        "case_count": len(cases),
        "garment_count": len({case["garment"] for case in cases}),
        "scores": aggregate,
        "guarded_cluster_bootstrap": _cluster_interval(cases),
        "candidate_available_count": sum(
            case["candidate_available"] for case in cases
        ),
        "candidate_accepted_count": accepted,
        "accepted_harmful_count": harmful,
        "all_rejections_bit_exact": all(
            case["candidate_accepted"] or case["bit_exact_fallback"]
            for case in cases
        ),
        "cases": cases,
        "inputs_sha256": {
            "source_summary": _sha256(args.source_summary),
            "source_lock": _sha256(args.source_lock),
        },
        "claim_boundary": (
            "unchanged-method diagnostic on an already-open, separate cloth "
            "source cohort; not confirmation, calibration, or SOTA evidence"
        ),
    }
    if args.candidate_family == "action_conditioned":
        payload["candidate_family"] = args.candidate_family
    payload["result_sha256"] = _canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "scores": aggregate,
                "cluster_bootstrap": payload["guarded_cluster_bootstrap"],
                "candidate_available_count": payload[
                    "candidate_available_count"
                ],
                "candidate_accepted_count": accepted,
                "accepted_harmful_count": harmful,
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
