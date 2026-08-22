#!/usr/bin/env python3
"""Extract and aggregate the frozen MatPhys surface-UQ source comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import array_sha256
from bayesian_phystwin.matphys_surface_uq_v1 import (
    backproject_masked_depth,
    evaluate_guarded_leave_one_group_out,
    evaluate_leave_one_group_out,
    nearest_surface_events,
)

CASE_SCHEMA = "bayesian-phystwin.matphys-surface-uq-source-case-v1"
RESULT_SCHEMA = "bayesian-phystwin.matphys-surface-uq-source-result-v1"
EVENT_FILENAME = "matphys_surface_events.npz"
CASE_FILENAME = "matphys_surface_uq_case.json"
RESULT_FILENAME = "matphys_surface_uq_source_result.json"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} is invalid",
    )
    return source.resolve(strict=True)


def _json(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must contain a JSON object")
    return value


def _validated_content_id(value: Mapping[str, Any], *, name: str) -> str:
    identity = value.get("artifact_id")
    payload = {key: item for key, item in value.items() if key != "artifact_id"}
    _require(identity == content_id(payload), f"{name} content ID changed")
    return str(identity)


def _protocol(path: Path) -> dict[str, Any]:
    value = _json(path, name="MatPhys source protocol")
    identity = (
        value.get("schema"),
        value.get("schema_version"),
        value.get("protocol_name"),
    )
    _require(
        identity
        in {
            (
                "bayesian-phystwin.matphys-surface-uq-source-protocol-v1",
                1,
                "matphys-surface-uq-source-v1",
            ),
            (
                "bayesian-phystwin.matphys-surface-uq-source-protocol-v2",
                2,
                "matphys-surface-uq-source-v2",
            ),
        },
        "MatPhys source protocol identity changed",
    )
    if identity[2] == "matphys-surface-uq-source-v2":
        covariance = value.get("covariance")
        _require(
            value.get("predecessor")
            == {
                "protocol_sha256": (
                    "32d91f91b0897f6a0d02c79d389bdfe67dc955968e081f33d7c326406055422a"
                ),
                "replay_quality_result_sha256": (
                    "0cadb32674193e70521c8ff4b3fef174abe123a2003d4a769ee64566b086b7d0"
                ),
                "status": "terminal-source-replay-quality-failure",
            }
            and isinstance(covariance, Mapping)
            and cast(Mapping[str, Any], covariance)
            .get("selection_policy", {})
            .get("minimum_member_to_effective_replay_floor_ratio")
            == 2.0
            and cast(Mapping[str, Any], covariance)
            .get("selection_policy", {})
            .get("maximum_reference_to_replay_ratio")
            == 3.0,
            "guarded MatPhys source protocol changed",
        )
    return value


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)  # type: ignore[arg-type]
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _endpoint_clouds(
    manifest: Mapping[str, Any],
    *,
    endpoint_root: Path,
    maximum_points: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    records_value = manifest.get("endpoint_archives")
    _require(
        isinstance(records_value, list) and bool(records_value),
        "endpoint archive roster is empty",
    )
    records = cast(list[Mapping[str, Any]], records_value)
    clouds: list[list[np.ndarray]] = [[] for _ in range(18)]
    provenance: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_record in records:
        _require(isinstance(raw_record, Mapping), "endpoint archive record changed")
        record = cast(Mapping[str, Any], raw_record)
        camera_value = record.get("camera_id")
        relative_value = record.get("path")
        _require(
            isinstance(camera_value, str)
            and bool(camera_value)
            and camera_value not in seen
            and isinstance(relative_value, str)
            and bool(relative_value),
            "endpoint archive identity changed",
        )
        camera_id = cast(str, camera_value)
        relative = cast(str, relative_value)
        seen.add(camera_id)
        path = _ordinary_file(endpoint_root / relative, name="endpoint archive")
        _require(_file_sha256(path) == record.get("sha256"), "endpoint archive changed")
        with np.load(path) as archive:
            _require(
                set(archive.files)
                == {
                    "frame_indices",
                    "raw_frame_indices",
                    "depth_m",
                    "object_mask",
                    "intrinsics",
                    "camera_to_world",
                },
                "endpoint archive member roster changed",
            )
            frame_indices = np.asarray(archive["frame_indices"])
            raw_indices = np.asarray(archive["raw_frame_indices"])
            depth = np.asarray(archive["depth_m"])
            mask = np.asarray(archive["object_mask"])
            intrinsics = np.asarray(archive["intrinsics"])
            camera_to_world = np.asarray(archive["camera_to_world"])
        _require(
            np.array_equal(frame_indices, np.arange(58, 76))
            and raw_indices.shape == (18,)
            and depth.shape == mask.shape
            and depth.shape[0] == 18
            and intrinsics.shape == (3, 3)
            and camera_to_world.shape == (4, 4),
            "endpoint archive shape changed",
        )
        for local_frame in range(18):
            cloud = backproject_masked_depth(
                depth[local_frame],
                mask[local_frame],
                intrinsics,
                camera_to_world,
                maximum_points=maximum_points,
                subsample_key=(
                    f"{manifest['case_id']}/{camera_id}/{int(frame_indices[local_frame])}"
                ),
            )
            if len(cloud):
                clouds[local_frame].append(cloud)
        provenance.append(
            {
                "camera_id": camera_id,
                "path": relative,
                "sha256": record["sha256"],
            }
        )
    _require(len(seen) >= 3, "fewer than three endpoint cameras are available")
    return [
        np.concatenate(frame_clouds, axis=0)
        if frame_clouds
        else np.empty((0, 3), dtype=np.float64)
        for frame_clouds in clouds
    ], provenance


def extract_case(
    *,
    protocol_path: Path,
    endpoint_manifest_path: Path,
    deform_manifest_path: Path,
    warp_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = _protocol(protocol_path)
    endpoint = _json(endpoint_manifest_path, name="source endpoint manifest")
    _validated_content_id(endpoint, name="source endpoint")
    case_id = endpoint.get("case_id")
    source_panel = cast(Mapping[str, Any], protocol["source_panel"])
    _require(
        isinstance(case_id, str) and case_id in source_panel["case_ids"],
        "endpoint case is absent from the source denominator",
    )
    _require(not output_dir.exists(), "source case output already exists")
    output_dir.mkdir(parents=True)
    base = {
        "schema": CASE_SCHEMA,
        "schema_version": 1,
        "case_id": case_id,
        "protocol_sha256": _file_sha256(protocol_path),
        "endpoint_manifest_sha256": _file_sha256(endpoint_manifest_path),
        "information_boundary": {
            "opened_development_source_only": True,
            "target_or_confirmation_data_read": False,
            "held_v8_artifacts_accessed": False,
            "dlo4_or_dlo5_accessed": False,
            "deform_mean_changed": False,
            "replacement_allowed": False,
        },
    }
    if endpoint.get("status") != "success":
        result = {
            **base,
            "status": "retained-source-technical-failure",
            "endpoint_status": endpoint.get("status"),
        }
        result["artifact_id"] = content_id(result)
        write_atomic_json(result, output_dir / CASE_FILENAME, overwrite=False)
        return result

    seals_value = endpoint.get("prediction_seals")
    _require(isinstance(seals_value, Mapping), "endpoint prediction seals are missing")
    seals = cast(Mapping[str, Any], seals_value)
    _require(
        _file_sha256(deform_manifest_path)
        == seals.get("deform_prediction_manifest_sha256")
        and _file_sha256(warp_manifest_path)
        == seals.get("matphys_warp_manifest_sha256"),
        "source prediction seal changed",
    )
    deform = _json(deform_manifest_path, name="DEFORM prediction manifest")
    warp = _json(warp_manifest_path, name="MatPhys Warp manifest")
    uncertainty_policy = endpoint.get("uncertainty_policy", "matphys")
    _require(
        uncertainty_policy in {"matphys", "isotropic-fallback"},
        "source uncertainty policy changed",
    )
    if uncertainty_policy == "isotropic-fallback":
        _require(
            protocol.get("protocol_name") == "matphys-surface-uq-source-v2",
            "isotropic fallback is absent from the source protocol",
        )
    warp_boundary = warp.get("information_boundary")
    warp_parity = warp.get("parity")
    _require(
        deform.get("case") == case_id
        and deform.get("passed") is True
        and cast(Mapping[str, Any], deform["information_boundary"]).get("outcome_read")
        is False,
        "DEFORM prediction boundary changed",
    )
    _require(isinstance(warp_parity, Mapping), "MatPhys Warp parity is missing")
    typed_warp_parity = cast(Mapping[str, Any], warp_parity)
    reference_ratio = typed_warp_parity.get("reference_to_replay_ratio")
    member_ratio = typed_warp_parity.get("member_to_replay_ratio")
    _require(
        typed_warp_parity.get("maximum_reference_to_replay_ratio") == 3.0
        and typed_warp_parity.get("minimum_member_to_replay_ratio") == 2.0
        and type(reference_ratio) in {int, float}
        and type(member_ratio) in {int, float}
        and np.isfinite(reference_ratio)
        and np.isfinite(member_ratio),
        "MatPhys Warp replay-quality contract changed",
    )
    replay_quality_passed = bool(
        float(cast(float, reference_ratio)) <= 3.0
        and float(cast(float, member_ratio)) >= 2.0
    )
    _require(
        warp.get("case_id") == case_id
        and isinstance(warp_boundary, Mapping)
        and warp_boundary.get("target_future_outcomes_opened") is False
        and typed_warp_parity.get("passed") is warp.get("passed")
        and warp.get("passed") is replay_quality_passed
        and (
            (uncertainty_policy == "matphys" and warp.get("passed") is True)
            or (
                uncertainty_policy == "isotropic-fallback"
                and warp.get("passed") is False
            )
        ),
        "MatPhys Warp prediction boundary changed",
    )
    deform_archive_record = cast(
        Mapping[str, Any], deform["physical_prediction_archive"]
    )
    deform_archive_path = _ordinary_file(
        cast(str, deform_archive_record["path"]), name="DEFORM prediction archive"
    )
    _require(
        _file_sha256(deform_archive_path) == deform_archive_record["file_sha256"],
        "DEFORM prediction archive changed",
    )
    with np.load(deform_archive_path) as archive:
        mean = np.asarray(archive["prediction_m"])
    declared_arrays = cast(Mapping[str, Any], deform_archive_record["array_sha256"])
    _require(
        mean.dtype == np.float32
        and mean.flags.c_contiguous
        and mean.ndim == 3
        and mean.shape[0] == 76
        and mean.shape[2] == 3
        and array_sha256(mean) == declared_arrays["prediction_m"],
        "registered DEFORM mean changed",
    )

    warp_archive_path: Path | None = None
    warp_output: Mapping[str, Any] | None = None
    if uncertainty_policy == "matphys":
        warp_output = cast(Mapping[str, Any], warp["output"])
        warp_archive_path = _ordinary_file(
            cast(str, warp_output["path"]), name="MatPhys Warp archive"
        )
        _require(
            _file_sha256(warp_archive_path) == warp_output["sha256"],
            "MatPhys Warp archive changed",
        )
        with np.load(warp_archive_path) as archive:
            covariance = np.asarray(archive["member_total_covariance_m2"])
        _require(
            covariance.shape[0] == 76
            and covariance.shape[1] >= mean.shape[1]
            and covariance.shape[2:] == (3, 3)
            and np.all(np.isfinite(covariance)),
            "MatPhys covariance shape changed",
        )
    else:
        covariance = np.zeros((*mean.shape[:2], 3, 3), dtype=np.float64)

    outcome = cast(Mapping[str, Any], protocol["outcome"])
    clouds, endpoint_records = _endpoint_clouds(
        endpoint,
        endpoint_root=endpoint_manifest_path.parent,
        maximum_points=int(outcome["maximum_points_per_camera_frame"]),
    )
    events = nearest_surface_events(
        mean[58:76],
        covariance[58:76, : mean.shape[1]],
        clouds,
        maximum_distance_m=float(outcome["maximum_point_to_surface_distance_m"]),
    )
    accepted_fraction = events.accepted_event_count / events.attempted_event_count
    threshold = float(
        cast(Mapping[str, Any], protocol["source_gate"])[
            "minimum_accepted_event_fraction_per_case"
        ]
    )
    arrays = {
        "residual_m": events.residual_m,
        "covariance_m2": events.covariance_m2,
        "frame_index": events.frame_index + 58,
        "node_index": events.node_index,
        "nearest_distance_m": events.nearest_distance_m,
    }
    event_path = output_dir / EVENT_FILENAME
    _atomic_npz(event_path, arrays)
    result = {
        **base,
        "status": "scorable"
        if accepted_fraction >= threshold
        else "insufficient-support",
        "deform_mean": {
            "path": str(deform_archive_path),
            "file_sha256": deform_archive_record["file_sha256"],
            "prediction_m_sha256": declared_arrays["prediction_m"],
            "dtype": mean.dtype.str,
            "shape": list(mean.shape),
            "candidate_and_comparator_bytes_identical": True,
        },
        "uncertainty_policy": uncertainty_policy,
        "matphys_covariance": (
            {
                "path": str(warp_archive_path),
                "file_sha256": cast(Mapping[str, Any], warp_output)["sha256"],
                "array": "member_total_covariance_m2",
                "observed_node_count": int(mean.shape[1]),
            }
            if uncertainty_policy == "matphys"
            else {
                "status": "target-free-replay-quality-rejected",
                "used_for_scoring": False,
                "association_placeholder_covariance": "all-zero; never evaluated",
            }
        ),
        "endpoint_archives": endpoint_records,
        "attempted_event_count": events.attempted_event_count,
        "accepted_event_count": events.accepted_event_count,
        "accepted_event_fraction": accepted_fraction,
        "events": {
            "path": EVENT_FILENAME,
            "sha256": _file_sha256(event_path),
            "array_sha256": {
                key: array_sha256(value) for key, value in sorted(arrays.items())
            },
        },
    }
    result["artifact_id"] = content_id(result)
    write_atomic_json(result, output_dir / CASE_FILENAME, overwrite=False)
    return result


def retain_case(
    *,
    protocol_path: Path,
    case_id: str,
    status: str,
    evidence_manifest_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Retain a registered non-scorable source case without opening an outcome."""

    protocol = _protocol(protocol_path)
    source_panel = cast(Mapping[str, Any], protocol["source_panel"])
    _require(case_id in source_panel["case_ids"], "case is absent from source panel")
    _require(
        status in {"unavailable-physical-carrier", "retained-source-technical-failure"},
        "retained source status is invalid",
    )
    _require(bool(evidence_manifest_paths), "retained source evidence is empty")
    evidence = [
        {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for path in evidence_manifest_paths
    ]
    result = {
        "schema": CASE_SCHEMA,
        "schema_version": 1,
        "case_id": case_id,
        "protocol_sha256": _file_sha256(protocol_path),
        "status": status,
        "evidence_manifests": evidence,
        "information_boundary": {
            "opened_development_source_only": True,
            "source_scoring_outcome_read": False,
            "target_or_confirmation_data_read": False,
            "held_v8_artifacts_accessed": False,
            "dlo4_or_dlo5_accessed": False,
            "deform_mean_changed": False,
            "replacement_allowed": False,
        },
    }
    result["artifact_id"] = content_id(result)
    _require(not output_dir.exists(), "source case output already exists")
    output_dir.mkdir(parents=True)
    write_atomic_json(result, output_dir / CASE_FILENAME, overwrite=False)
    return result


def aggregate_source(
    *,
    protocol_path: Path,
    case_manifest_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    protocol = _protocol(protocol_path)
    source_panel = cast(Mapping[str, Any], protocol["source_panel"])
    expected = tuple(str(value) for value in source_panel["case_ids"])
    protocol_sha256 = _file_sha256(protocol_path)
    _require(
        len(case_manifest_paths) == len(expected), "source denominator is incomplete"
    )
    records: dict[str, dict[str, Any]] = {}
    manifest_digests: dict[str, str] = {}
    for path in case_manifest_paths:
        record = _json(path, name="source case manifest")
        _require(
            record.get("schema") == CASE_SCHEMA
            and record.get("schema_version") == 1
            and record.get("protocol_sha256") == protocol_sha256,
            "source case schema changed",
        )
        _validated_content_id(record, name="source case")
        case_value = record.get("case_id")
        _require(
            isinstance(case_value, str) and case_value not in records,
            "source case identity changed",
        )
        case_id = cast(str, case_value)
        records[case_id] = record
        manifest_digests[case_id] = _file_sha256(path)
    _require(set(records) == set(expected), "source case roster changed")
    scorable = [
        case_id for case_id in expected if records[case_id]["status"] == "scorable"
    ]
    minimum = int(
        cast(Mapping[str, Any], protocol["source_gate"])[
            "minimum_jointly_scored_case_count"
        ]
    )
    residual_groups: list[np.ndarray] = []
    covariance_groups: list[np.ndarray | None] = []
    for case_id in scorable:
        manifest_path = next(
            path
            for path in case_manifest_paths
            if _json(path, name="source case manifest")["case_id"] == case_id
        )
        record = records[case_id]
        event = cast(Mapping[str, Any], record["events"])
        event_path = _ordinary_file(
            manifest_path.parent / cast(str, event["path"]), name="source events"
        )
        _require(_file_sha256(event_path) == event["sha256"], "source events changed")
        with np.load(event_path) as archive:
            residual = np.asarray(archive["residual_m"])
            covariance = np.asarray(archive["covariance_m2"])
        declared = cast(Mapping[str, Any], event["array_sha256"])
        _require(
            array_sha256(residual) == declared["residual_m"]
            and array_sha256(covariance) == declared["covariance_m2"],
            "source event arrays changed",
        )
        residual_groups.append(residual)
        policy = records[case_id].get("uncertainty_policy", "matphys")
        _require(
            policy in {"matphys", "isotropic-fallback"},
            "source case uncertainty policy changed",
        )
        if policy == "isotropic-fallback":
            _require(
                np.count_nonzero(covariance) == 0,
                "fallback association covariance is not the frozen placeholder",
            )
        covariance_groups.append(covariance if policy == "matphys" else None)

    evaluation: dict[str, object] | None = None
    gates: dict[str, bool]
    if len(scorable) >= minimum:
        observation_floor = float(
            cast(Mapping[str, Any], protocol["covariance"])["observation_floor_m"]
        )
        if protocol.get("protocol_name") == "matphys-surface-uq-source-v2":
            evaluation = evaluate_guarded_leave_one_group_out(
                scorable,
                residual_groups,
                covariance_groups,
                observation_floor_m=observation_floor,
            )
        else:
            _require(
                all(value is not None for value in covariance_groups),
                "v1 source case unexpectedly selected a fallback",
            )
            evaluation = evaluate_leave_one_group_out(
                scorable,
                residual_groups,
                [value for value in covariance_groups if value is not None],
                observation_floor_m=observation_floor,
            )
        metrics = cast(Mapping[str, Any], evaluation["equal_case_metrics"])
        gate = cast(Mapping[str, Any], protocol["source_gate"])
        coverage_interval = cast(list[float], gate["candidate_coverage_90_interval"])
        gates = {
            "minimum_jointly_scored_case_count": True,
            "minimum_case_wins_against_isotropic_nll": int(
                metrics["candidate_nll_win_count"]
            )
            >= int(gate["minimum_case_wins_against_isotropic_nll"]),
            "minimum_equal_case_nll_improvement_nats": float(
                metrics["candidate_nll_improvement_nats"]
            )
            >= float(gate["minimum_equal_case_nll_improvement_nats"]),
            "candidate_coverage_90_interval": (
                float(coverage_interval[0])
                <= float(metrics["candidate_coverage_90"])
                <= float(coverage_interval[1])
            ),
            "minimum_ellipsoid_volume_reduction_vs_conformal": float(
                metrics["candidate_volume_reduction_vs_conformal"]
            )
            >= float(gate["minimum_ellipsoid_volume_reduction_vs_conformal"]),
        }
    else:
        gates = {"minimum_jointly_scored_case_count": False}
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_sha256": _file_sha256(protocol_path),
        "source_denominator_count": len(expected),
        "ordinary_scorable_count": len(scorable),
        "matphys_scorable_count": sum(
            records[case_id].get("uncertainty_policy", "matphys") == "matphys"
            for case_id in scorable
        ),
        "isotropic_fallback_scorable_count": sum(
            records[case_id].get("uncertainty_policy") == "isotropic-fallback"
            for case_id in scorable
        ),
        "retained_case_status": {
            case_id: records[case_id]["status"] for case_id in expected
        },
        "case_manifest_sha256": manifest_digests,
        "leave_one_case_out": evaluation,
        "gates": gates,
        "passed": len(gates) == 5 and all(gates.values()),
        "information_boundary": {
            "opened_development_source_only": True,
            "target_or_confirmation_data_read": False,
            "held_v8_artifacts_accessed": False,
            "dlo4_or_dlo5_accessed": False,
            "fresh_target_authorized": len(gates) == 5 and all(gates.values()),
            "frozen_deform_results_changed": False,
        },
    }
    result["result_id"] = content_id(result)
    _require(not output_dir.exists(), "source aggregate output already exists")
    output_dir.mkdir(parents=True)
    write_atomic_json(result, output_dir / RESULT_FILENAME, overwrite=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--protocol", type=Path, required=True)
    extract.add_argument("--endpoint-manifest", type=Path, required=True)
    extract.add_argument("--deform-prediction-manifest", type=Path, required=True)
    extract.add_argument("--matphys-warp-manifest", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    retain = subparsers.add_parser("retain")
    retain.add_argument("--protocol", type=Path, required=True)
    retain.add_argument("--case-id", required=True)
    retain.add_argument(
        "--status",
        choices=("unavailable-physical-carrier", "retained-source-technical-failure"),
        required=True,
    )
    retain.add_argument(
        "--evidence-manifest", type=Path, action="append", required=True
    )
    retain.add_argument("--output-dir", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--case-manifest", type=Path, action="append", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "extract":
        result = extract_case(
            protocol_path=_ordinary_file(args.protocol, name="MatPhys source protocol"),
            endpoint_manifest_path=_ordinary_file(
                args.endpoint_manifest, name="source endpoint manifest"
            ),
            deform_manifest_path=_ordinary_file(
                args.deform_prediction_manifest, name="DEFORM prediction manifest"
            ),
            warp_manifest_path=_ordinary_file(
                args.matphys_warp_manifest, name="MatPhys Warp manifest"
            ),
            output_dir=Path(args.output_dir).absolute(),
        )
    elif args.command == "retain":
        result = retain_case(
            protocol_path=_ordinary_file(args.protocol, name="MatPhys source protocol"),
            case_id=str(args.case_id),
            status=str(args.status),
            evidence_manifest_paths=[
                _ordinary_file(path, name="retained source evidence")
                for path in args.evidence_manifest
            ],
            output_dir=Path(args.output_dir).absolute(),
        )
    else:
        result = aggregate_source(
            protocol_path=_ordinary_file(args.protocol, name="MatPhys source protocol"),
            case_manifest_paths=[
                _ordinary_file(path, name="source case manifest")
                for path in args.case_manifest
            ],
            output_dir=Path(args.output_dir).absolute(),
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
