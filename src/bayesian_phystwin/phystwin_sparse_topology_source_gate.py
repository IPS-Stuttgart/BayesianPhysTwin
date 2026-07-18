"""Future-blind transfer gate for one fixed sparse PhysTwin topology."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_piecewise_topology import load_piecewise_topology_artifact


SPARSE_TOPOLOGY_SOURCE_GATE_CONTRACT = "phystwin-sparse-topology-source-gate-v1"


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


def _metric_pair(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a metric object")
    result = {
        name: float(value[name])
        for name in ("chamfer_distance_m", "track_error_m")
    }
    if not all(np.isfinite(metric) and metric > 0.0 for metric in result.values()):
        raise ValueError(f"{label} metrics must be finite and positive")
    return result


def _validate_run(
    summary: Mapping[str, object],
    *,
    train_end: int,
    fit_end: int,
    topology_sha256: str,
    expected_inputs: Mapping[str, object],
) -> None:
    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("run summary omits config")
    expected = {
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
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"run config {key} changed: {config.get(key)!r} != "
                f"{expected_value!r}"
            )
    inputs = summary.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("run summary omits inputs")
    for name in ("final_data", "gt_track_3d"):
        observed = inputs.get(name)
        expected = expected_inputs.get(name)
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            raise ValueError(f"run or prefix manifest omits {name}")
        if observed.get("sha256") != expected.get("sha256"):
            raise ValueError(f"run {name} differs from the prefix artifact")
    topology = inputs.get("spring_topology")
    if not isinstance(topology, Mapping):
        raise ValueError("run summary omits spring topology identity")
    if topology.get("sha256") != topology_sha256:
        raise ValueError("run used a different topology artifact")
    parity = summary.get("selected_baseline_trajectory_parity")
    if not isinstance(parity, Mapping) or float(parity["vector_rmse_m"]) != 0.0:
        raise ValueError("zero-epoch trajectory differs from its topology baseline")


def _validate_topology_sidecar(
    sidecar: Mapping[str, object],
    *,
    expected_radius: float,
    expected_neighbour: float,
    expected_normalization: str,
    expected_final_data_sha256: str,
) -> tuple[Path, str]:
    if sidecar.get("contract") != "phystwin-piecewise-topology-v1":
        raise ValueError("unsupported topology sidecar contract")
    if sidecar.get("future_observations_used") is not False:
        raise ValueError("topology sidecar is not future blind")
    artifact = sidecar.get("artifact")
    search = sidecar.get("search_coordinates")
    if not isinstance(artifact, Mapping) or not isinstance(search, Mapping):
        raise ValueError("topology sidecar omits artifact or search coordinates")
    inputs = sidecar.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(
        inputs.get("final_data"), Mapping
    ):
        raise ValueError("topology sidecar omits final-data identity")
    if inputs["final_data"].get("sha256") != expected_final_data_sha256:
        raise ValueError("topology used a different prefix artifact")
    radii = np.asarray(search["radius_multipliers"], dtype=float)
    neighbours = np.asarray(search["neighbour_multipliers"], dtype=float)
    if not np.all(radii == expected_radius) or not np.all(
        neighbours == expected_neighbour
    ):
        raise ValueError("topology proposal differs from the locked profile")
    if search.get("object_scale_normalization") != expected_normalization:
        raise ValueError("topology normalization differs from the lock")
    if float(search.get("requested_object_log_scale", 0.0)) != 0.0:
        raise ValueError("topology adds an unlocked object spring scale")
    if float(search.get("controller_log_scale", 0.0)) != 0.0:
        raise ValueError("topology adds an unlocked controller spring scale")
    path = Path(str(artifact["path"])).resolve()
    digest = _sha256(path)
    if artifact.get("sha256") != digest:
        raise ValueError("topology artifact hash differs from its sidecar")
    return path, digest


def run_sparse_topology_source_gate(
    source_root: str | Path,
    output_path: str | Path,
    source_protocol: str | Path,
) -> dict[str, object]:
    """Evaluate one fixed topology on sealed suffixes of transfer prefixes."""

    root = Path(source_root).resolve()
    output = Path(output_path).resolve()
    protocol_path = Path(source_protocol).resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported source protocol schema")
    if protocol.get("protocol_name") != "phystwin-global-sparse-topology-source-v1":
        raise ValueError("source protocol is not the locked sparse-topology protocol")
    locked = root / "locked_protocol.json"
    if _sha256(locked) != _sha256(protocol_path):
        raise ValueError("source-root protocol differs from the requested lock")

    cases = protocol.get("transfer_cases")
    candidate = protocol.get("candidate")
    evidence = protocol.get("evidence_boundary")
    acceptance = protocol.get("source_acceptance")
    if not isinstance(cases, list) or not cases:
        raise ValueError("source protocol contains no transfer cases")
    if not all(isinstance(value, Mapping) for value in (candidate, evidence, acceptance)):
        raise ValueError("source protocol omits gate configuration")
    radius = float(candidate["radius_multiplier"])
    neighbour = float(candidate["neighbour_multiplier"])
    normalization = str(candidate["object_scale_normalization"])
    fit_fraction = float(evidence["fit_fraction_of_released_prefix"])

    case_results: dict[str, object] = {}
    code_commits: set[str] = set()
    official_commits: set[str] = set()
    for raw_case in cases:
        case = str(raw_case)
        case_root = root / "cases" / case
        prefix_manifest = _load_json(case_root / "prefix" / "manifest.json")
        if prefix_manifest.get("contract") != "phystwin-observation-prefix-plus-hold-v1":
            raise ValueError(f"{case}: unsupported prefix contract")
        train_end = int(prefix_manifest["prefix_end_frame"])
        if int(prefix_manifest["hold_frame_index"]) != train_end:
            raise ValueError(f"{case}: hold sentinel moved")
        if int(prefix_manifest["output_frame_count"]) != train_end + 1:
            raise ValueError(f"{case}: prefix payload frame count changed")
        fit_end = int(np.floor(fit_fraction * train_end))
        prefix_outputs = prefix_manifest.get("outputs")
        if not isinstance(prefix_outputs, Mapping) or not isinstance(
            prefix_outputs.get("final_data"), Mapping
        ):
            raise ValueError(f"{case}: prefix manifest omits output identities")
        final_data_sha = str(prefix_outputs["final_data"]["sha256"])

        identity_sidecar_path = case_root / "identity" / "topology.json"
        candidate_sidecar_path = case_root / "candidate" / "topology.json"
        identity_sidecar = _load_json(identity_sidecar_path)
        candidate_sidecar = _load_json(candidate_sidecar_path)
        identity_path, identity_sha = _validate_topology_sidecar(
            identity_sidecar,
            expected_radius=1.0,
            expected_neighbour=1.0,
            expected_normalization=normalization,
            expected_final_data_sha256=final_data_sha,
        )
        candidate_path, candidate_sha = _validate_topology_sidecar(
            candidate_sidecar,
            expected_radius=radius,
            expected_neighbour=neighbour,
            expected_normalization=normalization,
            expected_final_data_sha256=final_data_sha,
        )
        identity_topology = load_piecewise_topology_artifact(identity_path)
        candidate_topology = load_piecewise_topology_artifact(candidate_path)
        if (
            identity_topology.transfer.interpolated_edge_count != 0
            or identity_topology.transfer.removed_teacher_edge_count != 0
            or identity_topology.transfer.exact_edge_count
            != len(identity_topology.graph.springs)
        ):
            raise ValueError(f"{case}: identity topology differs from the teacher")
        if not np.isclose(
            identity_topology.applied_object_log_scale, 0.0, atol=1e-12
        ) or not np.isclose(
            identity_topology.applied_controller_log_scale, 0.0, atol=1e-12
        ):
            raise ValueError(f"{case}: identity topology rescales teacher springs")
        if not np.array_equal(
            identity_topology.graph.vertices, candidate_topology.graph.vertices
        ):
            raise ValueError(f"{case}: topology candidates use different vertices")
        identity_total = float(
            np.sum(
                identity_topology.reference_spring_y[
                    : identity_topology.graph.num_object_springs
                ]
            )
        )
        candidate_total = float(
            np.sum(
                candidate_topology.reference_spring_y[
                    : candidate_topology.graph.num_object_springs
                ]
            )
        )
        if not np.isclose(candidate_total, identity_total, rtol=2e-6, atol=0.0):
            raise ValueError(f"{case}: total object stiffness is not preserved")
        diagnostics = candidate_topology.diagnostics
        if diagnostics["object_component_count"] != 1:
            raise ValueError(f"{case}: candidate object graph is disconnected")
        if diagnostics["isolated_object_point_count"] != 0:
            raise ValueError(f"{case}: candidate isolates object points")
        if candidate_topology.graph.num_object_springs >= (
            identity_topology.graph.num_object_springs
        ):
            raise ValueError(f"{case}: candidate is not sparser than identity")

        identity_summary_path = case_root / "identity" / "summary.json"
        candidate_summary_path = case_root / "candidate" / "summary.json"
        identity_summary = _load_json(identity_summary_path)
        candidate_summary = _load_json(candidate_summary_path)
        _validate_run(
            identity_summary,
            train_end=train_end,
            fit_end=fit_end,
            topology_sha256=identity_sha,
            expected_inputs=prefix_outputs,
        )
        _validate_run(
            candidate_summary,
            train_end=train_end,
            fit_end=fit_end,
            topology_sha256=candidate_sha,
            expected_inputs=prefix_outputs,
        )
        for summary in (identity_summary, candidate_summary):
            code_commits.add(str(summary["code_commit"]))
            official_commits.add(str(summary["official_commit"]))
        teacher_metrics = _metric_pair(
            identity_summary["official_evaluation"]["validation"],
            label=f"{case}.identity",
        )
        candidate_metrics = _metric_pair(
            candidate_summary["official_evaluation"]["validation"],
            label=f"{case}.candidate",
        )
        ratios = {
            metric: candidate_metrics[metric] / teacher_metrics[metric]
            for metric in teacher_metrics
        }
        improvement = float(1.0 - np.mean(list(ratios.values())))
        both_improved = all(value < 1.0 for value in ratios.values())
        case_results[case] = {
            "fit_interval": [0, fit_end],
            "sealed_validation_interval": [fit_end, train_end],
            "identity_validation_metrics": teacher_metrics,
            "candidate_validation_metrics": candidate_metrics,
            "candidate_metric_ratios": ratios,
            "balanced_relative_improvement": improvement,
            "both_metrics_improved": both_improved,
            "topology": {
                "identity_object_spring_count": int(
                    identity_topology.graph.num_object_springs
                ),
                "candidate_object_spring_count": int(
                    candidate_topology.graph.num_object_springs
                ),
                "candidate_applied_object_log_scale": float(
                    candidate_topology.applied_object_log_scale
                ),
                "candidate_diagnostics": diagnostics,
            },
            "artifacts": {
                "prefix_manifest": _identity(case_root / "prefix" / "manifest.json"),
                "identity_topology_sidecar": _identity(identity_sidecar_path),
                "candidate_topology_sidecar": _identity(candidate_sidecar_path),
                "identity_summary": _identity(identity_summary_path),
                "candidate_summary": _identity(candidate_summary_path),
                "identity_trajectory": _identity(
                    case_root / "identity" / "trajectory.pkl"
                ),
                "candidate_trajectory": _identity(
                    case_root / "candidate" / "trajectory.pkl"
                ),
            },
        }

    metrics = ("chamfer_distance_m", "track_error_m")
    aggregate = {
        family: {
            metric: float(
                np.mean(
                    [result[f"{family}_validation_metrics"][metric] for result in case_results.values()]
                )
            )
            for metric in metrics
        }
        for family in ("identity", "candidate")
    }
    aggregate_ratios = {
        metric: aggregate["candidate"][metric] / aggregate["identity"][metric]
        for metric in metrics
    }
    aggregate_improvement = float(1.0 - np.mean(list(aggregate_ratios.values())))
    win_count = sum(
        int(result["both_metrics_improved"]) for result in case_results.values()
    )
    gate_passed = bool(
        win_count >= int(acceptance["minimum_both_metric_win_count"])
        and aggregate_improvement
        >= float(acceptance["minimum_aggregate_balanced_improvement"])
        and all(value < 1.0 for value in aggregate_ratios.values())
    )
    result = {
        "schema_version": 1,
        "contract": SPARSE_TOPOLOGY_SOURCE_GATE_CONTRACT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_protocol": _identity(protocol_path),
        "code_commits": sorted(code_commits),
        "official_commits": sorted(official_commits),
        "transfer_case_count": len(cases),
        "both_metric_win_count": win_count,
        "aggregate_validation_metrics": aggregate,
        "aggregate_metric_ratios": aggregate_ratios,
        "aggregate_balanced_relative_improvement": aggregate_improvement,
        "source_gate_passed": gate_passed,
        "selected_family": "global_sparse_density_matched" if gate_passed else "exact_teacher",
        "future_metrics_opened": False,
        "case_results": case_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(output)
    return result
