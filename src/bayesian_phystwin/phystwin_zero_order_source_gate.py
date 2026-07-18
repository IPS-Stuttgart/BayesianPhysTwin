"""Outer source-transfer gate for nested zero-order topology/field search."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_piecewise_topology import load_piecewise_topology_artifact
from .phystwin_zero_order_topology import (
    METRICS,
    ZERO_ORDER_TOPOLOGY_CONTRACT,
    ZeroOrderTopologySearchConfig,
    generate_topology_field_candidates,
    select_topology_field_candidate,
)


ZERO_ORDER_SOURCE_GATE_CONTRACT = "phystwin-zero-order-source-gate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": _sha256(path)}


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def _metric_pair(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a metric object")
    result = {name: float(value[name]) for name in METRICS}
    if not all(np.isfinite(metric) and metric > 0.0 for metric in result.values()):
        raise ValueError(f"{label} metrics must be finite and positive")
    return result


def _search_config(raw: Mapping[str, object]) -> ZeroOrderTopologySearchConfig:
    return ZeroOrderTopologySearchConfig(
        region_count=int(raw["region_count"]),
        candidates_per_family=int(raw["candidates_per_family"]),
        seed=int(raw["seed"]),
        radius_bounds=tuple(float(value) for value in raw["radius_bounds"]),
        neighbour_bounds=tuple(float(value) for value in raw["neighbour_bounds"]),
        object_log_scale_bounds=tuple(
            float(value) for value in raw["object_log_scale_bounds"]
        ),
        region_log_scale_bounds=tuple(
            float(value) for value in raw["region_log_scale_bounds"]
        ),
        controller_log_scale_bounds=tuple(
            float(value) for value in raw["controller_log_scale_bounds"]
        ),
        minimum_fit_improvement=float(raw["minimum_nested_fit_improvement"]),
        maximum_fit_metric_ratio=float(raw["maximum_nested_fit_metric_ratio"]),
    )


def _validate_full_run(
    summary: Mapping[str, object],
    *,
    train_end: int,
    fit_end: int,
    expected_inputs: Mapping[str, object],
    topology_sha256: str,
) -> None:
    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("full run omits config")
    expected_config = {
        "variant": "hard",
        "train_end_frame": train_end,
        "fit_end_frame": fit_end,
        "epochs": 0,
        "spring_parameterization": "dense",
        "selection_metric": "official_3d",
        "deterministic_spring_forces": True,
        "optimize_collision": False,
        "dashpot_log_scale": 0.0,
        "drag_log_scale": 0.0,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise ValueError(f"full run changed {name}")
    inputs = summary.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("full run omits inputs")
    for name in ("final_data", "gt_track_3d"):
        observed = inputs.get(name)
        expected = expected_inputs.get(name)
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            raise ValueError(f"full run or prefix omits {name}")
        if observed.get("sha256") != expected.get("sha256"):
            raise ValueError(f"full run {name} differs from outer prefix")
    topology = inputs.get("spring_topology")
    if not isinstance(topology, Mapping) or topology.get("sha256") != topology_sha256:
        raise ValueError("full run used a different topology")
    parity = summary.get("selected_baseline_trajectory_parity")
    if not isinstance(parity, Mapping) or float(parity["vector_rmse_m"]) != 0.0:
        raise ValueError("zero-epoch full run differs from its topology baseline")


def run_zero_order_source_gate(
    source_root: str | Path,
    output_path: str | Path,
    source_protocol: str | Path,
) -> dict[str, object]:
    """Recompute nested selections and evaluate only the outer source suffix."""

    root = Path(source_root).resolve()
    output = Path(output_path).resolve()
    protocol_path = Path(source_protocol).resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported source protocol schema")
    if protocol.get("protocol_name") != "phystwin-zero-order-topology-field-source-v1":
        raise ValueError("source protocol is not the locked zero-order protocol")
    if _sha256(root / "locked_protocol.json") != _sha256(protocol_path):
        raise ValueError("source-root protocol differs from the requested lock")

    cases = protocol.get("transfer_cases")
    evidence = protocol.get("evidence_boundary")
    raw_search = protocol.get("search")
    acceptance = protocol.get("source_acceptance")
    if not isinstance(cases, list) or not cases:
        raise ValueError("source protocol contains no transfer cases")
    if not all(
        isinstance(value, Mapping) for value in (evidence, raw_search, acceptance)
    ):
        raise ValueError("source protocol omits gate configuration")
    config = _search_config(raw_search)
    candidates = generate_topology_field_candidates(config)
    expected_candidates = _jsonable([asdict(candidate) for candidate in candidates])
    if len(candidates) != int(raw_search["total_candidate_count_including_teacher"]):
        raise ValueError("locked candidate count disagrees with generation")
    outer_fraction = float(evidence["outer_fit_fraction_of_released_prefix"])
    nested_fraction = float(evidence["nested_selection_fraction_of_outer_fit"])

    case_results: dict[str, object] = {}
    code_commits: set[str] = set()
    official_commits: set[str] = set()
    for raw_case in cases:
        case = str(raw_case)
        case_root = root / "cases" / case
        outer_manifest_path = case_root / "outer_prefix" / "manifest.json"
        fit_manifest_path = case_root / "fit_prefix" / "manifest.json"
        outer_manifest = _load_json(outer_manifest_path)
        fit_manifest = _load_json(fit_manifest_path)
        for label, manifest in (("outer", outer_manifest), ("fit", fit_manifest)):
            if manifest.get("contract") != "phystwin-observation-prefix-plus-hold-v1":
                raise ValueError(f"{case}: unsupported {label} prefix contract")
        train_end = int(outer_manifest["prefix_end_frame"])
        fit_end = int(np.floor(outer_fraction * train_end))
        selection_start = int(np.floor(nested_fraction * fit_end))
        if int(fit_manifest["prefix_end_frame"]) != fit_end:
            raise ValueError(f"{case}: nested fit prefix boundary changed")
        outer_outputs = outer_manifest.get("outputs")
        fit_inputs = fit_manifest.get("inputs")
        fit_outputs = fit_manifest.get("outputs")
        if not all(
            isinstance(value, Mapping)
            for value in (outer_outputs, fit_inputs, fit_outputs)
        ):
            raise ValueError(f"{case}: prefix manifests omit identities")
        for name in ("final_data", "gt_track_3d", "released_trajectory"):
            if fit_inputs[name]["sha256"] != outer_outputs[name]["sha256"]:
                raise ValueError(f"{case}: fit prefix does not descend from outer {name}")

        search_root = case_root / "search"
        search_summary_path = search_root / "search_summary.json"
        search_summary = _load_json(search_summary_path)
        if search_summary.get("contract") != ZERO_ORDER_TOPOLOGY_CONTRACT:
            raise ValueError(f"{case}: wrong search contract")
        if search_summary.get("future_observations_used") is not False:
            raise ValueError(f"{case}: search is not future blind")
        plan_path = search_root / "search_plan.json"
        plan = _load_json(plan_path)
        if search_summary["search_plan"]["sha256"] != _sha256(plan_path):
            raise ValueError(f"{case}: search-plan hash changed")
        if plan.get("future_observations_used") is not False:
            raise ValueError(f"{case}: search plan is not future blind")
        if plan.get("config") != _jsonable(asdict(config)):
            raise ValueError(f"{case}: search config differs from lock")
        if plan.get("candidates") != expected_candidates:
            raise ValueError(f"{case}: candidate bank differs from lock")
        expected_interval = [selection_start, fit_end]
        if plan.get("selection_interval") != expected_interval:
            raise ValueError(f"{case}: nested selection interval changed")
        if plan["inputs"]["fit_final_data"]["sha256"] != fit_outputs["final_data"]["sha256"]:
            raise ValueError(f"{case}: search used a different fit artifact")

        candidate_results = search_summary.get("candidate_results")
        if not isinstance(candidate_results, Mapping):
            raise ValueError(f"{case}: search omits candidate results")
        selection_metrics = {
            candidate_id: _metric_pair(
                result["selection_metrics"], label=f"{case}.{candidate_id}"
            )
            for candidate_id, result in candidate_results.items()
            if isinstance(result, Mapping) and result.get("status") == "evaluated"
        }
        recomputed = select_topology_field_candidate(
            selection_metrics, candidates, config
        )
        observed_selection = search_summary.get("selection")
        if not isinstance(observed_selection, Mapping):
            raise ValueError(f"{case}: search omits selection")
        for name in ("selected_candidate_id", "candidate_accepted", "fallback"):
            if observed_selection.get(name) != recomputed[name]:
                raise ValueError(f"{case}: nested selection does not reproduce")
        selected_id = str(recomputed["selected_candidate_id"])
        selected_result = candidate_results[selected_id]
        selected_topology_identity = selected_result["topology"]["artifact"]
        selected_topology_path = Path(
            str(selected_topology_identity["path"])
        ).resolve()
        selected_topology_sha = _sha256(selected_topology_path)
        if selected_topology_identity["sha256"] != selected_topology_sha:
            raise ValueError(f"{case}: selected topology hash changed")
        identity_result = candidate_results["exact_teacher"]
        identity_topology_identity = identity_result["topology"]["artifact"]
        identity_topology_path = Path(
            str(identity_topology_identity["path"])
        ).resolve()
        identity_topology_sha = _sha256(identity_topology_path)
        if identity_topology_identity["sha256"] != identity_topology_sha:
            raise ValueError(f"{case}: identity topology hash changed")
        identity_topology = load_piecewise_topology_artifact(identity_topology_path)
        if (
            identity_topology.transfer.interpolated_edge_count != 0
            or identity_topology.transfer.removed_teacher_edge_count != 0
            or identity_topology.transfer.exact_edge_count
            != len(identity_topology.graph.springs)
        ):
            raise ValueError(f"{case}: exact teacher topology is not an identity")
        selected_topology = load_piecewise_topology_artifact(selected_topology_path)
        if selected_topology.diagnostics["object_component_count"] != 1 or (
            selected_topology.diagnostics["isolated_object_point_count"] != 0
        ):
            raise ValueError(f"{case}: selected topology is invalid")

        identity_summary_path = case_root / "identity" / "summary.json"
        selected_summary_path = case_root / "selected" / "summary.json"
        identity_summary = _load_json(identity_summary_path)
        selected_summary = _load_json(selected_summary_path)
        _validate_full_run(
            identity_summary,
            train_end=train_end,
            fit_end=fit_end,
            expected_inputs=outer_outputs,
            topology_sha256=identity_topology_sha,
        )
        _validate_full_run(
            selected_summary,
            train_end=train_end,
            fit_end=fit_end,
            expected_inputs=outer_outputs,
            topology_sha256=selected_topology_sha,
        )
        for summary in (identity_summary, selected_summary):
            code_commits.add(str(summary["code_commit"]))
            official_commits.add(str(summary["official_commit"]))
        identity_metrics = _metric_pair(
            identity_summary["official_evaluation"]["validation"],
            label=f"{case}.identity",
        )
        selected_metrics = _metric_pair(
            selected_summary["official_evaluation"]["validation"],
            label=f"{case}.selected",
        )
        ratios = {
            name: selected_metrics[name] / identity_metrics[name] for name in METRICS
        }
        case_results[case] = {
            "fit_interval": [0, fit_end],
            "nested_selection_interval": expected_interval,
            "outer_validation_interval": [fit_end, train_end],
            "selected_candidate_id": selected_id,
            "selected_family": selected_result["candidate"]["family"],
            "non_teacher_selected": selected_id != "exact_teacher",
            "identity_validation_metrics": identity_metrics,
            "selected_validation_metrics": selected_metrics,
            "selected_metric_ratios": ratios,
            "balanced_relative_improvement": float(
                1.0 - np.mean(list(ratios.values()))
            ),
            "both_metrics_improved": all(value < 1.0 for value in ratios.values()),
            "artifacts": {
                "outer_prefix_manifest": _identity(outer_manifest_path),
                "fit_prefix_manifest": _identity(fit_manifest_path),
                "search_plan": _identity(plan_path),
                "search_summary": _identity(search_summary_path),
                "identity_summary": _identity(identity_summary_path),
                "selected_summary": _identity(selected_summary_path),
                "identity_trajectory": _identity(
                    case_root / "identity" / "trajectory.pkl"
                ),
                "selected_trajectory": _identity(
                    case_root / "selected" / "trajectory.pkl"
                ),
            },
        }

    aggregate = {
        family: {
            metric: float(
                np.mean(
                    [
                        result[f"{family}_validation_metrics"][metric]
                        for result in case_results.values()
                    ]
                )
            )
            for metric in METRICS
        }
        for family in ("identity", "selected")
    }
    aggregate_ratios = {
        metric: aggregate["selected"][metric] / aggregate["identity"][metric]
        for metric in METRICS
    }
    aggregate_improvement = float(1.0 - np.mean(list(aggregate_ratios.values())))
    non_teacher_count = sum(
        int(result["non_teacher_selected"]) for result in case_results.values()
    )
    win_count = sum(
        int(result["both_metrics_improved"]) for result in case_results.values()
    )
    gate_passed = bool(
        non_teacher_count >= int(acceptance["minimum_non_teacher_selection_count"])
        and win_count >= int(acceptance["minimum_both_metric_win_count"])
        and aggregate_improvement
        >= float(acceptance["minimum_aggregate_balanced_improvement"])
        and all(value < 1.0 for value in aggregate_ratios.values())
    )
    result = {
        "schema_version": 1,
        "contract": ZERO_ORDER_SOURCE_GATE_CONTRACT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_protocol": _identity(protocol_path),
        "code_commits": sorted(code_commits),
        "official_commits": sorted(official_commits),
        "transfer_case_count": len(cases),
        "non_teacher_selection_count": non_teacher_count,
        "both_metric_win_count": win_count,
        "aggregate_validation_metrics": aggregate,
        "aggregate_metric_ratios": aggregate_ratios,
        "aggregate_balanced_relative_improvement": aggregate_improvement,
        "source_gate_passed": gate_passed,
        "selected_family": "per_object_zero_order" if gate_passed else "exact_teacher",
        "future_metrics_opened": False,
        "case_results": case_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(output)
    return result
