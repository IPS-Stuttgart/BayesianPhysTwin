from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_observability_v4 import (
    DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
    DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
    DEFORM360_JOINT_SPARSE_SEMANTICS,
    Deform360JointSparseFactorBatchV4,
    Deform360JointSparseObservabilityPolicyV4,
    default_deform360_joint_sparse_information_boundary_v4,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/evaluate_deform360_joint_sparse_observability_v4.py"


def module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("joint_sparse_v4_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha(path),
        "byte_count": path.stat().st_size,
    }


def batch(object_id: str, stratum: str) -> Deform360JointSparseFactorBatchV4:
    state = np.zeros((6, 3, 3))
    state[0, 0, 0] = state[1, 1, 0] = 1
    state[2, 2, 1] = state[3, 0, 1] = 1
    state[4, 1, 2] = state[5, 2, 2] = 1
    return Deform360JointSparseFactorBatchV4(
        selection_artifact_sha256="1" * 64,
        visual_provider_lock_id="2" * 64,
        observation_artifact_id=hashlib.sha256(f"obs:{object_id}".encode()).hexdigest(),
        linearization_artifact_id=hashlib.sha256(f"lin:{object_id}".encode()).hexdigest(),
        implementation_revision="3" * 40,
        object_id=object_id,
        episode_id=0,
        stratum=stratum,
        factor_ids=tuple(
            hashlib.sha256(f"{object_id}:{index}".encode()).hexdigest()
            for index in range(6)
        ),
        camera_ids=("camera-a",) * 3 + ("camera-b",) * 3,
        window_ids=("window-0", "window-1") * 3,
        spatial_cluster_ids=tuple(f"cluster-{index}" for index in range(6)),
        correlation_group_ids=tuple(f"group-{index}" for index in range(6)),
        gauge_ids=("gauge-root",),
        gauge_prior_id="4" * 64,
        observation_covariance_m2=np.repeat(np.eye(3)[None], 6, axis=0),
        state_jacobian=state,
        local_gauge_jacobian=np.zeros((6, 3, 1)),
        gauge_indices=np.zeros(6, dtype=np.int64),
        parent_indices=np.array([-1], dtype=np.int64),
        transition_matrices=np.zeros((1, 1, 1)),
        innovation_scale_tril=np.ones((1, 1, 1)),
        query_jacobian=np.eye(3),
        prior_reliability=np.ones(6),
        association_probability=np.ones(6),
        composite_weight=np.ones(6),
        source_artifacts={"source.json": "5" * 64},
        metadata={"cohort": "development"},
    )


def write_bundle(
    root: Path, policy: Deform360JointSparseObservabilityPolicyV4
) -> tuple[Path, Path]:
    policy_path = root / "policy.json"
    policy_path.write_text(
        json.dumps(policy.to_record(), indent=2, sort_keys=True) + "\n"
    )
    cases = []
    for index, (object_id, stratum) in enumerate(
        (("a", "sheet"), ("b", "volumetric"))
    ):
        item = batch(object_id, stratum)
        descriptor = item.identity_record()
        descriptor["input_id"] = item.input_id
        descriptor_path = root / f"case-{index}.json"
        descriptor_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
        )
        arrays_path = root / f"case-{index}.npz"
        np.savez_compressed(arrays_path, **item.arrays())
        cases.append(
            {
                "object_id": object_id,
                "episode_id": 0,
                "stratum": stratum,
                "input_id": item.input_id,
                "descriptor": record(descriptor_path, root),
                "arrays": record(arrays_path, root),
            }
        )
    manifest = {
        "schema": "bayesian-phystwin.deform360-joint-sparse-observability-manifest",
        "schema_version": 4,
        "semantics": DEFORM360_JOINT_SPARSE_SEMANTICS,
        "protocol_id": DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
        "selection_artifact_sha256": "1" * 64,
        "visual_provider_lock_id": "2" * 64,
        "implementation_revision": "3" * 40,
        "policy_id": policy.policy_id,
        "cases": cases,
        "information_boundary": dict(
            default_deform360_joint_sparse_information_boundary_v4()
        ),
        "claim_boundary": DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
    }
    manifest["manifest_id"] = content_id(manifest)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return policy_path, manifest_path


def test_cli_publishes_supported_or_negative_development_result(
    tmp_path: Path,
) -> None:
    cli = module()
    policy = Deform360JointSparseObservabilityPolicyV4(
        minimum_distinct_spatial_clusters=4,
        minimum_supported_objects=2,
        minimum_supported_objects_per_stratum=1,
        maximum_single_camera_information_fraction=0.8,
        minimum_leave_one_camera_rank_fraction=1 / 3,
        minimum_leave_one_window_rank_fraction=1 / 3,
    )
    policy_path, manifest_path = write_bundle(tmp_path, policy)
    output = tmp_path / "output"
    assert (
        cli.main(
            [
                "--policy",
                str(policy_path),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "development-report.json").read_text())
    assert report["status"] == "development-design-supported"
    assert report["confirmation_access_authorized"] is False
    assert report["summary"]["supported_object_count"] == 2
    checksums = (output / "SHA256SUMS").read_text().splitlines()
    assert len(checksums) == 5

    negative_policy = Deform360JointSparseObservabilityPolicyV4(
        minimum_distinct_spatial_clusters=4,
        minimum_supported_objects=3,
        minimum_supported_objects_per_stratum=1,
        maximum_single_camera_information_fraction=0.8,
        minimum_leave_one_camera_rank_fraction=1 / 3,
        minimum_leave_one_window_rank_fraction=1 / 3,
    )
    negative_root = tmp_path / "negative"
    negative_root.mkdir()
    negative_policy_path, negative_manifest_path = write_bundle(
        negative_root, negative_policy
    )
    assert (
        cli.main(
            [
                "--policy",
                str(negative_policy_path),
                "--manifest",
                str(negative_manifest_path),
                "--output-dir",
                str(negative_root / "output"),
            ]
        )
        == 3
    )


def test_cli_fails_closed_on_array_and_descriptor_tampering(tmp_path: Path) -> None:
    cli = module()
    policy = Deform360JointSparseObservabilityPolicyV4(
        minimum_distinct_spatial_clusters=4,
        minimum_supported_objects=2,
        minimum_supported_objects_per_stratum=1,
        maximum_single_camera_information_fraction=0.8,
        minimum_leave_one_camera_rank_fraction=1 / 3,
        minimum_leave_one_window_rank_fraction=1 / 3,
    )
    policy_path, manifest_path = write_bundle(tmp_path, policy)
    arrays = tmp_path / "case-0.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256|byte count"):
        cli.evaluate_manifest(
            policy_path=policy_path,
            manifest_path=manifest_path,
            output=tmp_path / "bad-output",
        )

    clean = tmp_path / "clean"
    clean.mkdir()
    policy_path, manifest_path = write_bundle(clean, policy)
    descriptor = clean / "case-0.json"
    value = json.loads(descriptor.read_text())
    value["array_records"] = {}
    descriptor.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["descriptor"] = record(descriptor, clean)
    manifest.pop("manifest_id")
    manifest["manifest_id"] = content_id(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="array records"):
        cli.evaluate_manifest(
            policy_path=policy_path,
            manifest_path=manifest_path,
            output=clean / "bad-output",
        )
