#!/usr/bin/env python3
"""Run the locked leave-one-case-out native MatPhys source qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.matphys_native_source_v1 import (
    NativeMatPhysCaseEvidence,
    calibrated_covariance,
    gaussian_case_metrics,
    native_case_evidence,
    select_candidate_calibration,
    select_isotropic_calibration,
)
from bayesian_phystwin.matphys_warp_ensemble_v1 import file_sha256


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_id(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_content_id(record: dict[str, Any], *, id_field: str, name: str) -> None:
    identity = {key: value for key, value in record.items() if key != id_field}
    _require(
        record.get(id_field) == content_id(identity),
        f"{name} content identity changed",
    )


def _load_bound(record: object, *, name: str) -> Path:
    _require(isinstance(record, dict), f"{name} record is missing")
    path = Path(record.get("path", "")).resolve(strict=True)
    _require(
        path.is_file()
        and not path.is_symlink()
        and not any(parent.is_symlink() for parent in path.parents),
        f"{name} is not ordinary",
    )
    _require(file_sha256(path) == record.get("sha256"), f"{name} SHA-256 changed")
    _require(path.stat().st_size == record.get("byte_count"), f"{name} size changed")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-repository", type=Path, required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _case_evidence(
    root: Path, case_id: str
) -> tuple[NativeMatPhysCaseEvidence | None, dict[str, Any]]:
    case_manifest_path = root / "inputs" / case_id / "matphys_native_phystwin_case.json"
    prediction_manifest_path = (
        root / "predictions" / case_id / "matphys_fold_ensemble_prediction.json"
    )
    replay_manifest_path = (
        root / "replays" / case_id / "matphys_native_trajectory_ensemble.json"
    )
    _require(case_manifest_path.is_file(), f"{case_id}: case input is missing")
    _require(prediction_manifest_path.is_file(), f"{case_id}: prediction is missing")
    _require(replay_manifest_path.is_file(), f"{case_id}: replay result is missing")
    case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    _require(
        case_manifest.get("schema") == "bayesian-phystwin.matphys-native-phystwin-case"
        and case_manifest.get("schema_version") == 1,
        f"{case_id}: case manifest schema changed",
    )
    _validate_content_id(
        case_manifest, id_field="case_input_id", name=f"{case_id} case manifest"
    )
    _require(
        prediction_manifest.get("schema")
        == "bayesian-phystwin.matphys-fold-ensemble-prediction"
        and prediction_manifest.get("schema_version") == 1,
        f"{case_id}: prediction schema changed",
    )
    _validate_content_id(
        prediction_manifest,
        id_field="prediction_id",
        name=f"{case_id} prediction manifest",
    )
    _require(
        replay_manifest.get("schema")
        == "bayesian-phystwin.matphys-native-phystwin-trajectory-ensemble"
        and replay_manifest.get("schema_version") == 1,
        f"{case_id}: replay schema changed",
    )
    _validate_content_id(
        replay_manifest,
        id_field="result_id",
        name=f"{case_id} replay manifest",
    )
    _require(
        case_manifest.get("case_id") == case_id
        and prediction_manifest.get("case_id") == case_id
        and replay_manifest.get("case_id") == case_id,
        f"{case_id}: case identity changed",
    )
    _require(
        replay_manifest.get("case_input_id") == case_manifest.get("case_input_id")
        and replay_manifest.get("source_prediction_id")
        == prediction_manifest.get("prediction_id"),
        f"{case_id}: replay lineage failed",
    )
    runtime = replay_manifest.get("runtime")
    _require(
        isinstance(runtime, dict)
        and runtime.get("spring_force_accumulation") == "official-atomic-v1",
        f"{case_id}: spring-force accumulation changed",
    )
    replay_path = _load_bound(
        replay_manifest["output"], name=f"{case_id} replay archive"
    )
    parity = replay_manifest.get("parity")
    _require(isinstance(parity, dict), f"{case_id}: parity record is missing")
    _require(
        replay_manifest.get("passed") is parity.get("passed")
        and type(parity.get("passed")) is bool,
        f"{case_id}: replay and parity status disagree",
    )
    parity_rmse = float(parity.get("coordinate_rmse_m", float("nan")))
    parity_limit = float(parity.get("maximum_allowed_rmse_m", float("nan")))
    _require(
        np.isfinite(parity_rmse)
        and np.isfinite(parity_limit)
        and parity_limit > 0.0
        and parity["passed"] is (parity_rmse <= parity_limit),
        f"{case_id}: parity authorization was not re-derived",
    )
    prefix = tuple(int(value) for value in case_manifest["split"]["prefix"])
    future = tuple(int(value) for value in case_manifest["split"]["future"])
    provenance = {
        "status": (
            "ordinary_success"
            if replay_manifest["passed"]
            else "retained_native_parity_failure"
        ),
        "case_input_manifest_sha256": file_sha256(case_manifest_path),
        "case_input_id": case_manifest["case_input_id"],
        "prediction_manifest_sha256": file_sha256(prediction_manifest_path),
        "prediction_id": prediction_manifest["prediction_id"],
        "replay_manifest_sha256": file_sha256(replay_manifest_path),
        "replay_result_id": replay_manifest["result_id"],
        "replay_archive_sha256": replay_manifest["output"]["sha256"],
        "prefix": list(prefix),
        "future": list(future),
        "parity": parity,
    }
    if replay_manifest["passed"] is False:
        return None, provenance

    final_data_path = _load_bound(
        case_manifest["inputs"]["final_data"], name=f"{case_id} final data"
    )
    baseline_path = _load_bound(
        case_manifest["inputs"]["baseline_trajectory"],
        name=f"{case_id} baseline",
    )
    with final_data_path.open("rb") as stream:
        final_data = pickle.load(stream)
    with baseline_path.open("rb") as stream:
        baseline = np.asarray(pickle.load(stream), dtype=np.float64)
    with np.load(replay_path, allow_pickle=False) as archive:
        raw_covariance = np.asarray(
            archive["baseline_relative_total_covariance_m2"], dtype=np.float64
        )
    observed = np.asarray(final_data["object_points"], dtype=np.float64)
    visible = np.asarray(final_data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(final_data["object_motions_valid"], dtype=bool)
    original_count = observed.shape[1]
    evidence = native_case_evidence(
        case_id=case_id,
        observed_m=observed,
        baseline_mean_m=baseline[:, :original_count],
        valid_mask=visible & motion_valid,
        raw_covariance_m2=raw_covariance[:, :original_count],
        future_start=future[0],
        future_stop=future[1],
        identity_count=128,
    )
    return evidence, provenance


def main() -> None:
    args = _parse_args()
    execution = args.execution_repository.resolve(strict=True)
    _require(
        _git_revision(execution) == args.expected_execution_revision,
        "execution repository revision changed",
    )
    _require(
        not subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=execution,
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
        "execution repository is dirty",
    )
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(
        protocol.get("schema")
        == "bayesian-phystwin.matphys-native-phystwin-source-protocol"
        and protocol.get("schema_version") == 1
        and protocol.get("protocol_id") == "matphys-native-phystwin-source-v1",
        "source protocol changed",
    )
    root = args.evidence_root.resolve(strict=True)
    case_ids = tuple(str(value) for value in protocol["cases"])
    _require(len(case_ids) == 11 and len(set(case_ids)) == 11, "case roster changed")
    evidence_by_case: dict[str, NativeMatPhysCaseEvidence] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        evidence, provenance[case_id] = _case_evidence(root, case_id)
        if evidence is not None:
            evidence_by_case[case_id] = evidence

    scoring = protocol["scoring"]
    scale_grid = tuple(float(value) for value in scoring["scale_grid"])
    std_grid = tuple(float(value) for value in scoring["isotropic_std_m_grid"])
    case_results: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        if case_id not in evidence_by_case:
            case_results[case_id] = {
                "status": provenance[case_id]["status"],
                "provenance": provenance[case_id],
            }
    scorable_case_ids = tuple(
        case_id for case_id in case_ids if case_id in evidence_by_case
    )
    _require(
        len(scorable_case_ids) >= 2,
        "fewer than two native-parity cases are available for LOO scoring",
    )
    for held_out in scorable_case_ids:
        fit = tuple(
            evidence_by_case[case_id]
            for case_id in scorable_case_ids
            if case_id != held_out
        )
        _require(
            len(fit) >= 1,
            "at least two ordinary source cases are required for LOO scoring",
        )
        scale, candidate_std = select_candidate_calibration(
            fit,
            scale_grid=scale_grid,
            isotropic_std_grid_m=std_grid,
        )
        isotropic_std = select_isotropic_calibration(fit, isotropic_std_grid_m=std_grid)
        held = evidence_by_case[held_out]
        candidate_covariance = calibrated_covariance(
            held.raw_covariance_m2,
            scale=scale,
            isotropic_std_m=candidate_std,
        )
        isotropic_covariance = np.broadcast_to(
            np.eye(3) * isotropic_std * isotropic_std,
            (len(held.residual_m), 3, 3),
        )
        candidate = gaussian_case_metrics(held.residual_m, candidate_covariance)
        isotropic = gaussian_case_metrics(held.residual_m, isotropic_covariance)
        case_results[held_out] = {
            "status": "ordinary_success",
            "event_count": len(held.residual_m),
            "selected_candidate_scale": scale,
            "selected_candidate_isotropic_std_m": candidate_std,
            "selected_comparator_isotropic_std_m": isotropic_std,
            "candidate": candidate,
            "isotropic": isotropic,
            "candidate_nll_improvement_nats_per_event": (
                isotropic["nll_nats_per_event"] - candidate["nll_nats_per_event"]
            ),
            "candidate_to_isotropic_volume_ratio": (
                candidate["mean_ellipsoid_volume_m3"]
                / isotropic["mean_ellipsoid_volume_m3"]
            ),
            "provenance": provenance[held_out],
        }

    ordinary_results = [case_results[case_id] for case_id in scorable_case_ids]
    candidate_nll = float(
        np.mean(
            [value["candidate"]["nll_nats_per_event"] for value in ordinary_results]
        )
    )
    isotropic_nll = float(
        np.mean(
            [value["isotropic"]["nll_nats_per_event"] for value in ordinary_results]
        )
    )
    candidate_coverage = float(
        np.mean([value["candidate"]["coverage_90"] for value in ordinary_results])
    )
    candidate_volume = float(
        np.mean(
            [
                value["candidate"]["mean_ellipsoid_volume_m3"]
                for value in ordinary_results
            ]
        )
    )
    isotropic_volume = float(
        np.mean(
            [
                value["isotropic"]["mean_ellipsoid_volume_m3"]
                for value in ordinary_results
            ]
        )
    )
    case_wins = int(
        np.sum(
            [
                value["candidate_nll_improvement_nats_per_event"] > 0.0
                for value in ordinary_results
            ]
        )
    )
    retained_parity_failures = len(case_ids) - len(scorable_case_ids)
    gate = protocol["advancement_gate"]
    checks = {
        "case_nll_wins": case_wins >= int(gate["minimum_case_nll_wins"]),
        "equal_case_nll_improvement": (
            isotropic_nll - candidate_nll
            >= float(gate["minimum_equal_case_nll_improvement_nats_per_event"])
        ),
        "coverage_lower": candidate_coverage
        >= float(gate["minimum_candidate_coverage_90"]),
        "coverage_upper": candidate_coverage
        <= float(gate["maximum_candidate_coverage_90"]),
        "volume": candidate_volume / isotropic_volume
        <= float(gate["maximum_candidate_to_isotropic_mean_volume_ratio"]),
        "complete_case_accounting": len(case_results) == len(case_ids),
        "no_retained_native_parity_failures": retained_parity_failures
        <= int(gate["maximum_retained_native_parity_failures"]),
    }
    passed = all(checks.values())
    identity = {
        "schema": "bayesian-phystwin.matphys-native-phystwin-source-result",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "execution_revision": _git_revision(execution),
        "scorer_sha256": file_sha256(Path(__file__)),
        "source_module_sha256": file_sha256(
            execution / "src" / "bayesian_phystwin" / "matphys_native_source_v1.py"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_ids),
        "ordinary_scored_case_count": len(scorable_case_ids),
        "retained_native_parity_failure_count": retained_parity_failures,
        "case_nll_wins": case_wins,
        "equal_case_candidate_nll_nats_per_event": candidate_nll,
        "equal_case_isotropic_nll_nats_per_event": isotropic_nll,
        "equal_case_nll_improvement_nats_per_event": isotropic_nll - candidate_nll,
        "equal_case_candidate_coverage_90": candidate_coverage,
        "equal_case_candidate_mean_ellipsoid_volume_m3": candidate_volume,
        "equal_case_isotropic_mean_ellipsoid_volume_m3": isotropic_volume,
        "candidate_to_isotropic_mean_volume_ratio": candidate_volume / isotropic_volume,
        "gate_checks": checks,
        "passed": passed,
        "case_results": case_results,
        "information_boundary": {
            "source_only": True,
            "fresh_target_selected": False,
            "fresh_target_outcomes_opened": False,
            "held_v8_accessed": False,
            "dlo4_accessed": False,
            "dlo5_accessed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result = {**identity, "result_id": _canonical_id(identity)}
    _require(not args.output.exists(), "source result already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
