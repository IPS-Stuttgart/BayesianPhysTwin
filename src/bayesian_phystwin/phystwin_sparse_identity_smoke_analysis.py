"""Locked analysis for the covariance-aware sparse-identity smoke."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PRIMARY_ARM = "causal_selected_dense_relative_cap_temporal"
METRICS = ("chamfer_distance_m", "track_error_m")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    result = {metric: float(payload[metric]) for metric in METRICS}
    if not all(math.isfinite(value) and value >= 0.0 for value in result.values()):
        raise ValueError("future metrics must be finite and nonnegative")
    return result


def _validate_runner_config(
    candidate: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    actual = candidate["config"]
    expected = protocol["method"]["runner_config"]
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"candidate runner setting {key!r} does not match protocol"
            )


def analyze_sparse_identity_smoke(
    candidate: dict[str, Any],
    comparator: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Read only the preregistered arm and diagnostics from one opened case."""

    cases = list(protocol["cases"])
    if len(cases) != 1:
        raise ValueError("the smoke protocol must contain exactly one case")
    case = cases[0]
    if set(candidate["case_results"]) != {case}:
        raise ValueError("candidate cases do not match the smoke protocol")
    if case not in comparator["case_results"]:
        raise ValueError("frozen comparator lacks the smoke case")
    if protocol["future_read"]["primary_arm"] != PRIMARY_ARM:
        raise ValueError("protocol primary arm is unexpected")
    if candidate["config"]["observation_source"] != (
        "final_data_plus_cotracker3_sparse_identity"
    ):
        raise ValueError("candidate observation source does not match protocol")
    if candidate["config"]["manual_prefix_override"]:
        raise ValueError("manual prefix override must be disabled")
    if comparator["config"]["observation_source"] != "final_data":
        raise ValueError("frozen comparator observation source is unexpected")
    if comparator["config"]["manual_prefix_override"]:
        raise ValueError("frozen comparator used manual prefix observations")
    _validate_runner_config(candidate, protocol)

    candidate_case = candidate["case_results"][case]
    comparator_case = comparator["case_results"][case]
    expected_baseline = protocol["inputs"]["physical_baseline"]
    if candidate_case["baseline_trajectory"]["sha256"] != expected_baseline["sha256"]:
        raise ValueError("candidate physical baseline hash does not match protocol")

    selector = candidate_case["causal_selection"]["selectors"][PRIMARY_ARM]
    comparator_selector = comparator_case["causal_selection"]["selectors"][PRIMARY_ARM]
    baseline = _metrics(candidate_case["baseline"])
    candidate_metrics = _metrics(selector["future_metrics"])
    comparator_metrics = _metrics(comparator_selector["future_metrics"])
    comparator_baseline = _metrics(comparator_case["baseline"])
    if baseline != comparator_baseline:
        raise ValueError("candidate and comparator physical baselines differ")

    cue_summary = candidate_case["cotracker_depth_lift"]
    diagnostics = {
        "identity_count": int(cue_summary["identity_count"]),
        "valid_fraction": float(cue_summary["valid_fraction"]),
        "two_view_fraction_of_valid": float(
            cue_summary["two_view_fraction_of_valid"]
        ),
        "mean_prior_reliability": float(cue_summary["mean_prior_reliability"]),
        "median_observation_std_m": (
            None
            if cue_summary["median_observation_std_m"] is None
            else float(cue_summary["median_observation_std_m"])
        ),
        "reliability_uses_phystwin_innovation": bool(
            cue_summary["reliability_uses_phystwin_innovation"]
        ),
        "innovation_likelihood_count": int(
            cue_summary["innovation_likelihood_count"]
        ),
    }
    if diagnostics["valid_fraction"] < 0.0 or diagnostics["valid_fraction"] > 1.0:
        raise ValueError("valid observation fraction lies outside [0, 1]")

    track_improvement = (
        comparator_metrics["track_error_m"] - candidate_metrics["track_error_m"]
    ) / comparator_metrics["track_error_m"]
    cd_regression = (
        candidate_metrics["chamfer_distance_m"]
        - comparator_metrics["chamfer_distance_m"]
    ) / comparator_metrics["chamfer_distance_m"]
    gates_config = protocol["smoke_gate"]
    gates = {
        "automatic_identity_support": (
            diagnostics["identity_count"]
            >= int(gates_config["minimum_identity_count"])
            and diagnostics["valid_fraction"]
            >= float(gates_config["minimum_valid_fraction"])
        ),
        "prior_reliability_is_residual_independent": (
            not diagnostics["reliability_uses_phystwin_innovation"]
        ),
        "innovation_enters_once": diagnostics["innovation_likelihood_count"] == 1,
        "prefix_selector_accepts_primary_arm": bool(selector["accepted"]),
        "track_improvement_over_frozen_dense": (
            track_improvement
            >= float(gates_config["minimum_track_improvement_fraction"])
        ),
        "cd_regression_within_tolerance": (
            cd_regression
            <= float(gates_config["maximum_cd_regression_fraction"])
        ),
        "both_metrics_no_worse_than_physical_baseline": all(
            candidate_metrics[metric] <= baseline[metric] for metric in METRICS
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "post-open one-case development smoke; not independent evidence",
        "case": case,
        "primary_arm": PRIMARY_ARM,
        "baseline": baseline,
        "frozen_released_dense": comparator_metrics,
        "candidate": candidate_metrics,
        "candidate_relative_to_frozen_released_dense": {
            "chamfer_distance_fraction": cd_regression,
            "track_error_fraction": -track_improvement,
        },
        "observation_diagnostics": diagnostics,
        "selected_prefix_settings": {
            "base_relative_candidate": selector["base_relative_candidate"],
            "selected_temporal_candidate": selector["selected_candidate"],
        },
        "gates": gates,
        "smoke_gate_passed": passed,
        "recommendation": (
            gates_config["pass_action"]
            if passed
            else gates_config["fail_action"]
        ),
    }


def analyze_sparse_identity_smoke_files(
    *,
    candidate_path: str | Path,
    comparator_path: str | Path,
    cue_path: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Validate locked input hashes and attach provenance to the compact result."""

    paths = {
        "candidate": Path(candidate_path),
        "comparator": Path(comparator_path),
        "cue": Path(cue_path),
        "protocol": Path(protocol_path),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    if hashes["comparator"] != protocol["inputs"]["frozen_comparator"]["sha256"]:
        raise ValueError("frozen comparator hash does not match protocol")
    if hashes["cue"] != protocol["inputs"]["cotracker3_cues"]["sha256"]:
        raise ValueError("CoTracker3 cue hash does not match protocol")
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    if candidate["inputs"]["family_selection"]["sha256"] != (
        protocol["inputs"]["family_selection"]["sha256"]
    ):
        raise ValueError("family-selection hash does not match protocol")
    comparator = json.loads(paths["comparator"].read_text(encoding="utf-8"))
    result = analyze_sparse_identity_smoke(candidate, comparator, protocol)
    result["inputs"] = {
        name: {"path": str(path), "sha256": hashes[name]}
        for name, path in paths.items()
    }
    return result
