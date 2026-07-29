from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import bayesian_phystwin.deform360_causal_response_direct_depth_prediction_v14 as runtime_module
from bayesian_phystwin.deform360_causal_response_adaptive_query import STRICT_ARM
from bayesian_phystwin.deform360_causal_response_direct_depth_admission_v14 import (
    ADMISSION_REPORT_FILENAME,
    load_v14_admission_prelock_protocol,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_physical import (
    PHYSICAL_ARCHIVE_FILENAME,
    PHYSICAL_MANIFEST_FILENAME,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_prediction_v14 import (
    build_v14_prediction_runtime,
    load_v14_prediction_runtime,
    write_v14_prediction_runtime,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_source_lock import (
    AdaptiveDirectDepthSourceCaseV14,
    AdaptiveDirectDepthSourceLockV14,
    validate_adaptive_direct_depth_source_lock_v14,
)

ROOT = Path(__file__).resolve().parents[1]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source_lock(
    path: Path, method_config_sha256: str
) -> AdaptiveDirectDepthSourceLockV14:
    cases = tuple(
        AdaptiveDirectDepthSourceCaseV14(
            case_id=f"fresh-runtime-{index:02d}-ep0000",
            case_hash=_digest(f"case-{index}"),
            object_hash=_digest(f"object-{index}"),
            metadata_sha256=_digest(f"metadata-{index}"),
            source_preflight_sha256=_digest(f"preflight-{index}"),
            carrier_artifact_sha256=_digest(f"carrier-{index}"),
            carrier_arm=STRICT_ARM,
            fold=index % 3,
        )
        for index in range(12)
    )
    provisional = AdaptiveDirectDepthSourceLockV14(
        repository_revision="a" * 40,
        method_config_sha256=method_config_sha256,
        exclusion_manifest_sha256=_digest("exclusion"),
        exclusion_manifest_file_sha256=_digest("exclusion-file"),
        synthetic_control_result_sha256=_digest("synthetic"),
        synthetic_control_file_sha256=_digest("synthetic-file"),
        excluded_object_hashes=(),
        cases=cases,
        selection_metadata_sha256=_digest("selection"),
        artifact_sha256="0" * 64,
    )
    descriptor = provisional.descriptor()
    descriptor.pop("artifact_sha256")
    artifact_sha256 = hashlib.sha256(
        b"deform360-causal-response-direct-depth-source-lock-v14\0"
        + json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    lock = replace(provisional, artifact_sha256=artifact_sha256)
    path.write_text(
        json.dumps(lock.descriptor(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_adaptive_direct_depth_source_lock_v14(path)


def test_prediction_runtime_binds_all_twelve_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_path = ROOT / "configs/sota/deform360_causal_response_direct_depth_v14.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    source_path = tmp_path / "source-lock.json"
    source_lock = _source_lock(source_path, method["config_sha256"])
    admission_prelock = (
        ROOT / "configs/sota/"
        "deform360_causal_response_direct_depth_v14_admission_prelock.json"
    )
    physical_prelock = (
        ROOT / "configs/sota/"
        "deform360_causal_response_direct_depth_v14_physical_prelock.json"
    )
    admission_config_sha256 = load_v14_admission_prelock_protocol(admission_prelock)[
        "config_sha256"
    ]
    admission_root = tmp_path / "admissions"
    physical_root = tmp_path / "physical"
    (admission_root / "rank-001").mkdir(parents=True)
    (physical_root / "rank-001").mkdir(parents=True)
    admission_reports: dict[Path, dict] = {}
    physical_manifests: dict[Path, dict] = {}
    for rank, case in enumerate(source_lock.cases, start=3):
        admission_dir = admission_root / f"rank-{rank:03d}"
        admission_dir.mkdir()
        (admission_dir / ADMISSION_REPORT_FILENAME).write_text(
            f"admission-{rank}\n",
            encoding="utf-8",
        )
        physical_dir = physical_root / f"rank-{rank:03d}"
        physical_dir.mkdir()
        (physical_dir / PHYSICAL_MANIFEST_FILENAME).write_text(
            f"physical-{rank}\n",
            encoding="utf-8",
        )
        (physical_dir / PHYSICAL_ARCHIVE_FILENAME).write_bytes(
            f"archive-{rank}".encode()
        )
        physical_artifact_sha256 = _digest(f"physical-artifact-{rank}")
        admission_reports[admission_dir.resolve()] = {
            "status": "admitted",
            "queue_rank": rank,
            "case_hash": case.case_hash,
            "object_hash": case.object_hash,
            "artifact_sha256": _digest(f"admission-artifact-{rank}"),
            "physical_artifact_sha256": physical_artifact_sha256,
            "admission_prelock_config_sha256": admission_config_sha256,
        }
        physical_manifests[physical_dir.resolve()] = {
            "queue_rank": rank,
            "case_hash": case.case_hash,
            "object_hash": case.object_hash,
            "artifact_sha256": physical_artifact_sha256,
        }

    monkeypatch.setattr(
        runtime_module,
        "validate_v14_admission_report",
        lambda directory: admission_reports[Path(directory).resolve()],
    )
    monkeypatch.setattr(
        runtime_module,
        "validate_v14_physical_artifacts",
        lambda directory, **_: (
            physical_manifests[Path(directory).resolve()],
            {},
        ),
    )
    implementation_paths = {}
    for name in (
        "prediction_module",
        "prediction_runner",
        "preflight_module",
        "runtime_builder",
    ):
        implementation_paths[name] = tmp_path / f"{name}.py"
        implementation_paths[name].write_text(f"# {name}\n", encoding="utf-8")

    payload = build_v14_prediction_runtime(
        repository_revision="b" * 40,
        method_protocol_path=method_path,
        source_lock_path=source_path,
        admission_prelock_path=admission_prelock,
        physical_prelock_path=physical_prelock,
        admission_root=admission_root,
        physical_root=physical_root,
        implementation_paths=implementation_paths,
    )
    repeated = build_v14_prediction_runtime(
        repository_revision="b" * 40,
        method_protocol_path=method_path,
        source_lock_path=source_path,
        admission_prelock_path=admission_prelock,
        physical_prelock_path=physical_prelock,
        admission_root=admission_root,
        physical_root=physical_root,
        implementation_paths=implementation_paths,
    )
    output = tmp_path / "prediction-runtime.json"
    write_v14_prediction_runtime(
        output,
        payload,
        method_protocol_path=method_path,
        source_lock_path=source_path,
        admission_prelock_path=admission_prelock,
        physical_prelock_path=physical_prelock,
    )
    loaded = load_v14_prediction_runtime(
        output,
        method_protocol_path=method_path,
        source_lock_path=source_path,
        admission_prelock_path=admission_prelock,
        physical_prelock_path=physical_prelock,
    )

    assert repeated == payload == loaded
    assert [record["queue_rank"] for record in loaded["cases"]] == list(range(3, 15))
    assert loaded["information_boundary"]["source_outcome_read"] is False

    admission_reports[next(iter(admission_reports))][
        "admission_prelock_config_sha256"
    ] = _digest("wrong-prelock")
    with pytest.raises(ValueError, match="prelock-mismatched"):
        build_v14_prediction_runtime(
            repository_revision="b" * 40,
            method_protocol_path=method_path,
            source_lock_path=source_path,
            admission_prelock_path=admission_prelock,
            physical_prelock_path=physical_prelock,
            admission_root=admission_root,
            physical_root=physical_root,
            implementation_paths=implementation_paths,
        )


def test_prediction_runtime_rejects_object_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_path = ROOT / "configs/sota/deform360_causal_response_direct_depth_v14.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    source_path = tmp_path / "source-lock.json"
    _source_lock(source_path, method["config_sha256"])
    admission_prelock = (
        ROOT / "configs/sota/"
        "deform360_causal_response_direct_depth_v14_admission_prelock.json"
    )
    physical_prelock = (
        ROOT / "configs/sota/"
        "deform360_causal_response_direct_depth_v14_physical_prelock.json"
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": runtime_module.RUNTIME_KIND,
        "contract": runtime_module.RUNTIME_CONTRACT,
        "protocol_id": runtime_module.RUNTIME_PROTOCOL_ID,
        "status": "locked_after_source_selection_before_prefix_scan",
        "parent_artifacts": {
            "method_protocol": {
                "semantic_sha256": method["config_sha256"],
                "file_sha256": runtime_module.file_sha256(method_path),
            },
            "source_lock": {
                "semantic_sha256": json.loads(source_path.read_text())[
                    "artifact_sha256"
                ],
                "file_sha256": runtime_module.file_sha256(source_path),
            },
            "admission_prelock": {
                "semantic_sha256": load_v14_admission_prelock_protocol(
                    admission_prelock
                )["config_sha256"],
                "file_sha256": runtime_module.file_sha256(admission_prelock),
            },
            "physical_prelock": {
                "semantic_sha256": runtime_module.load_v14_physical_prelock_protocol(
                    physical_prelock
                )["config_sha256"],
                "file_sha256": runtime_module.file_sha256(physical_prelock),
            },
        },
        "implementation": {
            "parent_commit": "b" * 40,
            "file_sha256": {
                name: _digest(name)
                for name in (
                    "prediction_module",
                    "prediction_runner",
                    "preflight_module",
                    "runtime_builder",
                )
            },
        },
        "numerical_contract": {
            "prefix_frame_count": runtime_module.PREFIX_FRAME_COUNT,
            "prediction_frame_count": runtime_module.PREDICTION_FRAME_COUNT,
            "depth_scale_to_m": 0.001,
            "tactile_aggregation": runtime_module.TACTILE_AGGREGATION,
            "tactile_values_are_calibrated_probabilities": False,
            "actuator_position_field": runtime_module.ACTUATOR_POSITION_FIELD,
        },
        "cases": [
            {
                "queue_rank": rank,
                "case_hash": case["case_hash"],
                "object_hash": (
                    _digest("substituted-object") if rank == 3 else case["object_hash"]
                ),
                **{
                    name: _digest(f"{name}-{rank}")
                    for name in (
                        "admission_artifact_sha256",
                        "admission_file_sha256",
                        "physical_artifact_sha256",
                        "physical_manifest_file_sha256",
                        "physical_archive_file_sha256",
                    )
                },
            }
            for rank, case in enumerate(
                json.loads(source_path.read_text())["cases"],
                start=3,
            )
        ],
        "information_boundary": {
            "maximum_object_observation_frame": (runtime_module.PREFIX_FRAME_COUNT - 1),
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "source_outcome_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    payload["config_sha256"] = runtime_module._canonical_sha256(payload)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="case ledger"):
        load_v14_prediction_runtime(
            runtime_path,
            method_protocol_path=method_path,
            source_lock_path=source_path,
            admission_prelock_path=admission_prelock,
            physical_prelock_path=physical_prelock,
        )
