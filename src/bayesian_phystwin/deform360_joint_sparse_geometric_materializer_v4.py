"""Atomic Deform360 joint-sparse geometric v4 manifest materialization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
)
from .deform360_joint_sparse_geometric_batch_v4 import (
    _build_object_batch,
    _npz_payload,
    _record_for_file,
    _write_checksums,
)
from .deform360_joint_sparse_geometric_candidates_v4 import (
    _Candidate,
    _collect_stream_candidates,
)
from .deform360_joint_sparse_geometric_common_v4 import (
    EXPECTED_DEVELOPMENT_BOUNDARY,
    MATERIALIZER_CLAIM_BOUNDARY,
    MATERIALIZER_POLICY_SEMANTICS,
    MATERIALIZER_SCHEMA,
    MATERIALIZER_VERSION,
    PLAN_CASE_FIELDS,
    PLAN_EXCLUDED_FIELDS,
    PLAN_STREAM_FIELDS,
    V4_MANIFEST_SCHEMA,
    V4_MANIFEST_VERSION,
    _integer,
    _literal,
    _ordinary_directory,
    _ordinary_file,
    _require,
    _sha256_file,
    _verify_record,
    _write_json,
)
from .deform360_joint_sparse_geometric_source_v4 import _validate_sources
from .deform360_joint_sparse_observability_v4 import (
    DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
    DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
    DEFORM360_JOINT_SPARSE_SEMANTICS,
)


def materialize_manifest(
    *,
    metric_batch_root: str | Path,
    prediction_root: str | Path,
    production_result_path: str | Path,
    selection_path: str | Path,
    visual_provider_spec_path: str | Path,
    metric_policy_path: str | Path,
    camera_policy_path: str | Path,
    v4_policy_path: str | Path,
    materializer_policy_path: str | Path,
    implementation_revision: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Publish one complete v4 development manifest from retained results."""

    metric_batch = _ordinary_directory(Path(metric_batch_root), name="metric batch root")
    prediction = _ordinary_directory(Path(prediction_root), name="prediction root")
    production_path = _ordinary_file(Path(production_result_path), name="production result")
    selection_source = _ordinary_file(Path(selection_path), name="selection")
    provider_source = _ordinary_file(Path(visual_provider_spec_path), name="provider specification")
    metric_policy_source = _ordinary_file(Path(metric_policy_path), name="metric policy")
    camera_policy_source = _ordinary_file(Path(camera_policy_path), name="camera policy")
    v4_policy_source = _ordinary_file(Path(v4_policy_path), name="v4 policy")
    materializer_policy_source = _ordinary_file(Path(materializer_policy_path), name="materializer policy")
    revision = exact_revision(implementation_revision, name="implementation_revision")
    metric_result, plan, production, v4_policy, policy = _validate_sources(
        metric_batch_root=metric_batch,
        prediction_root=prediction,
        production_result_path=production_path,
        selection_path=selection_source,
        visual_provider_spec_path=provider_source,
        metric_policy_path=metric_policy_source,
        camera_policy_path=camera_policy_source,
        v4_policy_path=v4_policy_source,
        materializer_policy_path=materializer_policy_source,
    )
    target = Path(output_directory).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    _require(not os.path.lexists(target), "geometric v4 output already exists")
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    metric_root = _ordinary_directory(metric_batch / "metrics", name="metric root")
    try:
        shutil.copyfile(materializer_policy_source, temporary / "materializer-policy.json")
        shutil.copyfile(v4_policy_source, temporary / "v4-policy.json")
        raw_cases = plan.get("cases")
        _require(isinstance(raw_cases, list) and len(raw_cases) == 10, "metric plan cohort changed")
        excluded_rows = plan.get("excluded_streams")
        _require(isinstance(excluded_rows, list) and len(excluded_rows) == 11, "metric plan exclusions changed")
        excluded_by_object: Counter[str] = Counter()
        for index, raw in enumerate(excluded_rows):
            _require(isinstance(raw, Mapping), f"excluded stream {index} changed")
            row = cast(Mapping[str, Any], raw)
            require_exact_fields(row, expected=PLAN_EXCLUDED_FIELDS, name=f"excluded stream {index}")
            _require(row.get("reason") == "released-robot-geometry-outside-fixed-camera-prefix", "excluded stream reason changed")
            excluded_by_object[_literal(row.get("object_id"), name="excluded object_id")] += 1
        manifest_cases: list[dict[str, Any]] = []
        case_summaries: list[dict[str, Any]] = []
        plan_order: list[tuple[str, int]] = []
        for case_index, raw_case in enumerate(raw_cases):
            _require(isinstance(raw_case, Mapping), f"metric plan case {case_index} changed")
            case = cast(Mapping[str, Any], raw_case)
            require_exact_fields(case, expected=PLAN_CASE_FIELDS, name=f"metric plan case {case_index}")
            object_id = _literal(case.get("object_id"), name="object_id")
            episode_id = _integer(case.get("episode_id"), name="episode_id")
            stratum = _literal(case.get("stratum"), name="stratum")
            _require(stratum in {"sheet", "volumetric"}, "stratum changed")
            plan_order.append((object_id, episode_id))
            causal = case.get("causal_frame_range_half_open")
            _require(isinstance(causal, list) and len(causal) == 2, "causal range changed")
            causal_range = (
                _integer(causal[0], name="causal start"),
                _integer(causal[1], name="causal stop", minimum=1),
            )
            _require(causal_range[0] < causal_range[1], "causal range is empty")
            raw_streams = case.get("streams")
            _require(isinstance(raw_streams, list) and len(raw_streams) >= 2, "object lacks retained streams")
            candidates: list[_Candidate] = []
            dropped = 0
            sources: dict[str, str] = {
                "locks/selection.json": _sha256_file(selection_source),
                "locks/visual-provider-spec.json": _sha256_file(provider_source),
                "locks/metric-policy.json": _sha256_file(metric_policy_source),
                "locks/camera-policy.json": _sha256_file(camera_policy_source),
                "locks/v4-policy.json": _sha256_file(v4_policy_source),
                "locks/materializer-policy.json": _sha256_file(materializer_policy_source),
                "sources/metric-batch-result.json": _sha256_file(metric_batch / "metric-batch-result.json"),
                "sources/metric-prefix-plan.json": _sha256_file(metric_batch / "metric-prefix-plan.json"),
                "sources/visual-production-result.json": _sha256_file(production_path),
            }
            stream_metadata: list[dict[str, Any]] = []
            for stream_index, raw_stream in enumerate(raw_streams):
                _require(isinstance(raw_stream, Mapping), f"stream {stream_index} changed")
                stream = cast(Mapping[str, Any], raw_stream)
                require_exact_fields(stream, expected=PLAN_STREAM_FIELDS, name=f"stream {stream_index}")
                job_id = sha256_digest(stream.get("job_id"), name="job_id")
                camera_id = _literal(stream.get("camera_id"), name="camera_id")
                prediction_manifest, _ = _verify_record(
                    prediction, stream.get("prediction_manifest"), name="prediction manifest"
                )
                metric_prefix, _ = _verify_record(
                    metric_root, stream.get("metric_prefix"), name="metric prefix"
                )
                metric_calibration, _ = _verify_record(
                    metric_root, stream.get("metric_calibration"), name="metric calibration"
                )
                selected, stream_dropped, stream_sources, details = _collect_stream_candidates(
                    job_id=job_id,
                    camera_id=camera_id,
                    causal_range=causal_range,
                    prediction_manifest_path=prediction_manifest,
                    metric_prefix_path=metric_prefix,
                    metric_calibration_path=metric_calibration,
                    object_id=object_id,
                    episode_id=episode_id,
                    policy=policy,
                )
                candidates.extend(selected)
                dropped += stream_dropped
                overlap = set(sources) & set(stream_sources)
                _require(not overlap, f"source artifact paths repeat: {sorted(overlap)}")
                sources.update(stream_sources)
                stream_metadata.append(details)
            batch = _build_object_batch(
                candidates=candidates,
                selection_artifact_sha256=cast(str, policy["selection_artifact_sha256"]),
                visual_provider_lock_id=cast(str, policy["visual_provider_lock_id"]),
                implementation_revision=revision,
                object_id=object_id,
                episode_id=episode_id,
                stratum=stratum,
                excluded_factor_count=dropped,
                source_artifacts=sources,
                policy=policy,
                metadata={
                    "metric_batch_result_id": metric_result["result_id"],
                    "production_result_id": production["result_id"],
                    "prior_excluded_stream_count": excluded_by_object[object_id],
                    "retained_stream_count": len(raw_streams),
                    "dropped_by_camera_window_cap": dropped,
                    "stream_diagnostics": stream_metadata,
                },
            )
            case_root = temporary / "cases" / f"{case_index:02d}-{object_id}"
            case_root.mkdir(parents=True)
            descriptor_path = case_root / "descriptor.json"
            arrays_path = case_root / "arrays.npz"
            _write_json(descriptor_path, batch.identity_record())
            with arrays_path.open("xb") as stream:
                np.savez_compressed(stream, **_npz_payload(batch))
            manifest_cases.append(
                {
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stratum": stratum,
                    "input_id": batch.input_id,
                    "descriptor": _record_for_file(descriptor_path, root=temporary),
                    "arrays": _record_for_file(arrays_path, root=temporary),
                }
            )
            case_summaries.append(
                {
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stratum": stratum,
                    "input_id": batch.input_id,
                    "factor_count": len(batch.factor_ids),
                    "excluded_factor_count": batch.excluded_factor_count,
                    "distinct_camera_count": len(set(batch.camera_ids)),
                    "distinct_window_count": len(set(batch.window_ids)),
                    "distinct_spatial_cluster_count": len(set(batch.spatial_cluster_ids)),
                    "prior_excluded_stream_count": excluded_by_object[object_id],
                }
            )
        _require(plan_order == sorted(plan_order), "metric plan cases are not sorted")
        manifest_identity: dict[str, Any] = {
            "schema": V4_MANIFEST_SCHEMA,
            "schema_version": V4_MANIFEST_VERSION,
            "semantics": DEFORM360_JOINT_SPARSE_SEMANTICS,
            "protocol_id": DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
            "selection_artifact_sha256": policy["selection_artifact_sha256"],
            "visual_provider_lock_id": policy["visual_provider_lock_id"],
            "implementation_revision": revision,
            "policy_id": v4_policy.policy_id,
            "cases": manifest_cases,
            "information_boundary": EXPECTED_DEVELOPMENT_BOUNDARY,
            "claim_boundary": DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
        }
        manifest = {
            **manifest_identity,
            "manifest_id": content_id(manifest_identity),
        }
        _write_json(temporary / "manifest.json", manifest)
        materialization_identity = {
            "schema": MATERIALIZER_SCHEMA,
            "schema_version": MATERIALIZER_VERSION,
            "semantics": MATERIALIZER_POLICY_SEMANTICS,
            "implementation_revision": revision,
            "materializer_policy_id": policy["artifact_id"],
            "v4_policy_id": v4_policy.policy_id,
            "metric_batch_result_id": metric_result["result_id"],
            "production_result_id": production["result_id"],
            "manifest_id": manifest["manifest_id"],
            "case_count": len(manifest_cases),
            "cases": case_summaries,
            "information_boundary": EXPECTED_DEVELOPMENT_BOUNDARY,
            "claim_boundary": MATERIALIZER_CLAIM_BOUNDARY,
        }
        materialization = {
            **materialization_identity,
            "materialization_id": content_id(materialization_identity),
        }
        _write_json(temporary / "materialization-result.json", materialization)
        _write_checksums(temporary)
        _require(not os.path.lexists(target), "geometric v4 output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return cast(
        dict[str, Any],
        load_strict_json_object(target / "materialization-result.json", label="materialization result"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-batch-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-provider-spec", type=Path, required=True)
    parser.add_argument("--metric-policy", type=Path, required=True)
    parser.add_argument("--camera-policy", type=Path, required=True)
    parser.add_argument("--v4-policy", type=Path, required=True)
    parser.add_argument("--materializer-policy", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = materialize_manifest(
        metric_batch_root=arguments.metric_batch_root,
        prediction_root=arguments.prediction_root,
        production_result_path=arguments.production_result,
        selection_path=arguments.selection,
        visual_provider_spec_path=arguments.visual_provider_spec,
        metric_policy_path=arguments.metric_policy,
        camera_policy_path=arguments.camera_policy,
        v4_policy_path=arguments.v4_policy,
        materializer_policy_path=arguments.materializer_policy,
        implementation_revision=arguments.implementation_revision,
        output_directory=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
