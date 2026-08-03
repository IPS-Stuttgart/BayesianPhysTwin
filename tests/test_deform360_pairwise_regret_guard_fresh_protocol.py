from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    CATALOG_KIND,
    EXCLUSION_KIND,
    FROZEN_CAMERA_PANEL,
    HASH_NAMESPACE,
    FreshTechnicalLockConfig,
    _canonical_sha256,
    build_exclusion_union,
    build_fresh_download_manifest,
    build_fresh_source_plan,
    build_fresh_technical_lock,
    file_sha256,
    object_exclusion_hash,
    validate_exclusion_union,
    validate_fresh_download_manifest,
    validate_fresh_source_plan,
    validate_fresh_technical_lock,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _seal(payload: dict[str, object], key: str) -> dict[str, object]:
    result = dict(payload)
    result[key] = _canonical_sha256(result, digest_key=key)
    return result


def _exclusion(path: Path, object_ids: list[str], owner: str) -> str:
    payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": EXCLUSION_KIND,
            "hash_namespace": HASH_NAMESPACE,
            "owner": owner,
            "object_hashes": sorted(object_exclusion_hash(value) for value in object_ids),
            "source_artifact_sha256s": ["1" * 64],
            "information_boundary": {
                "target_artifact_read": False,
                "object_ids_emitted": False,
            },
        },
        "exclusion_sha256",
    )
    _write(path, payload)
    return str(payload["exclusion_sha256"])


def _catalog(path: Path) -> None:
    payload = _seal(
        {
            "schema_version": 1,
            "artifact_kind": CATALOG_KIND,
            "public_object_count": 3,
            "paper_reported_object_count": 3,
            "objects": [
                {"object_id": "001-rope", "oid": "1" * 40},
                {"object_id": "002-cloth", "oid": "2" * 40},
                {"object_id": "003-sponge", "oid": "3" * 40},
            ],
            "information_boundary": {
                "media_or_episode_payload_read": False,
                "public_directory_metadata_only": True,
                "target_metric_read": False,
            },
        },
        "catalog_sha256",
    )
    _write(path, payload)


def _source_bindings(tmp_path: Path) -> tuple[Path, Path]:
    protocol = tmp_path / "source_protocol.json"
    qualification = tmp_path / "source_qualification.json"
    _write(
        protocol,
        {
            "artifact_kind": "Deform360PairwiseRegretGuardSourceProtocol",
            "decision": {
                "source_gate_passed": True,
                "fresh_accuracy_evaluation_allowed": True,
            },
            "information_boundary": {
                "runtime_candidate_accepts_target": False,
                "runtime_candidate_accepts_outcome": False,
            },
        },
    )
    certificate = {
        "feature_center": [0.0],
        "feature_scale": [1.0],
        "standardized_feature_lower": [-1.0],
        "standardized_feature_upper": [1.0],
        "coefficients": [0.0, 0.0],
        "upper_residual_quantile": 0.0,
        "nominal_coverage": 0.5,
        "minimum_improvement": 0.0,
        "ridge_penalty": 1.0,
        "support_margin_std": 0.0,
        "source_group_count": 3,
        "finite_sample_rank": 2,
        "finite_sample_coverage": 0.5,
    }
    _write(
        qualification,
        {
            "artifact_kind": "Deform360PairwiseRegretGuardSourceQualification",
            "source_gate_passed": True,
            "fresh_accuracy_evaluation_allowed": True,
            "calibrated_safety_claim_allowed": False,
            "deployment_artifact": {"candidate_certificate": certificate},
            "information_boundary": {
                "runtime_candidate_accepts_target": False,
                "runtime_candidate_accepts_outcome": False,
                "fresh_outcomes_may_not_select_or_refit_this_lock": True,
            },
        },
    )
    return protocol, qualification


def _fixture(tmp_path: Path) -> tuple[list[Path], FreshTechnicalLockConfig, dict[str, Path]]:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    digests = sorted(
        [
            _exclusion(first, ["001-rope"], "first"),
            _exclusion(second, ["002-cloth"], "second"),
        ]
    )
    catalog = tmp_path / "catalog.json"
    metadata = tmp_path / "metadata.json"
    _catalog(catalog)
    _write(
        metadata,
        {
            "object": "003-sponge",
            "sequences": {
                "0": {
                    "action": "lift",
                    "bimanual": "no",
                    "nonprehensile": "no",
                },
                "1": {
                    "action": "squeeze",
                    "bimanual": "yes",
                    "nonprehensile": "no",
                },
                "2": {
                    "action": "drag",
                    "bimanual": "no",
                    "nonprehensile": "yes",
                },
                "3": {
                    "action": "bad metadata",
                    "bimanual": "yess",
                    "nonprehensile": "no",
                },
            },
        },
    )
    protocol, qualification = _source_bindings(tmp_path)
    config = FreshTechnicalLockConfig(
        expected_exclusion_manifest_sha256s=tuple(digests),
        expected_public_object_count=3,
        expected_excluded_public_object_count=2,
        expected_remaining_public_object_count=1,
        minimum_valid_episode_count=3,
        runtime_method_commit="a" * 40,
        source_protocol_file_sha256=file_sha256(protocol),
        source_qualification_file_sha256=file_sha256(qualification),
    )
    return [first, second], config, {
        "catalog": catalog,
        "metadata": metadata,
        "protocol": protocol,
        "qualification": qualification,
    }


def test_union_and_lock_retain_malformed_episode(tmp_path: Path) -> None:
    manifests, config, paths = _fixture(tmp_path)
    union = build_exclusion_union(manifests, config=config)
    validate_exclusion_union(union, config=config)
    union_path = tmp_path / "union.json"
    _write(union_path, union)
    lock = build_fresh_technical_lock(
        union_path,
        paths["catalog"],
        paths["metadata"],
        paths["protocol"],
        paths["qualification"],
        config=config,
    )
    validate_fresh_technical_lock(lock, config=config)
    selected = lock["selected_physical_object"]
    assert selected["object_id"] == "003-sponge"
    assert [row["episode_id"] for row in selected["valid_episodes"]] == [0, 1, 2]
    assert selected["rejected_episodes"] == [
        {
            "episode_id": 3,
            "action": "bad metadata",
            "bimanual": "yess",
            "nonprehensile": "no",
            "rejection_reasons": ["bimanual must be exactly 'yes' or 'no'"],
        }
    ]
    assert lock["claim_boundary"].startswith("This is a no-refit, single-object")


def test_union_requires_exact_frozen_manifest_inventory(tmp_path: Path) -> None:
    manifests, config, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest set differs"):
        build_exclusion_union(manifests[:1], config=config)


def test_union_rejects_target_boundary_change(tmp_path: Path) -> None:
    manifests, config, _ = _fixture(tmp_path)
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    payload["information_boundary"]["target_artifact_read"] = True
    payload["exclusion_sha256"] = _canonical_sha256(
        payload, digest_key="exclusion_sha256"
    )
    _write(manifests[0], payload)
    with pytest.raises(ValueError, match="crossed its target boundary"):
        build_exclusion_union(manifests, config=config)


def test_lock_rejects_changed_source_qualification(tmp_path: Path) -> None:
    manifests, config, paths = _fixture(tmp_path)
    union = build_exclusion_union(manifests, config=config)
    union_path = tmp_path / "union.json"
    _write(union_path, union)
    with paths["qualification"].open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="qualification file changed"):
        build_fresh_technical_lock(
            union_path,
            paths["catalog"],
            paths["metadata"],
            paths["protocol"],
            paths["qualification"],
            config=config,
        )


def test_lock_rejects_more_than_one_fresh_object(tmp_path: Path) -> None:
    manifests, config, paths = _fixture(tmp_path)
    union = build_exclusion_union(manifests, config=config)
    union["object_hashes"].pop()
    union["object_hash_count"] -= 1
    union["union_sha256"] = _canonical_sha256(union, digest_key="union_sha256")
    union_path = tmp_path / "union.json"
    _write(union_path, union)
    relaxed = FreshTechnicalLockConfig(
        expected_exclusion_manifest_sha256s=config.expected_exclusion_manifest_sha256s,
        expected_public_object_count=3,
        expected_excluded_public_object_count=1,
        expected_remaining_public_object_count=2,
        minimum_valid_episode_count=3,
        runtime_method_commit=config.runtime_method_commit,
        source_protocol_file_sha256=config.source_protocol_file_sha256,
        source_qualification_file_sha256=config.source_qualification_file_sha256,
    )
    with pytest.raises(ValueError, match="exactly one untouched object"):
        build_fresh_technical_lock(
            union_path,
            paths["catalog"],
            paths["metadata"],
            paths["protocol"],
            paths["qualification"],
            config=relaxed,
        )


def test_object_hash_uses_null_terminated_namespace() -> None:
    expected = hashlib.sha256(
        b"deform360-fresh-object-exclusion-v1\0" + b"003-sponge"
    ).hexdigest()
    assert object_exclusion_hash("003-sponge") == expected


def test_committed_fresh_artifacts_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    union = json.loads(
        (
            root
            / "configs/sota/deform360_pairwise_regret_guard_fresh_exclusion_union_v1.json"
        ).read_text(encoding="utf-8")
    )
    lock = json.loads(
        (
            root
            / "configs/sota/deform360_pairwise_regret_guard_fresh_technical_v1.json"
        ).read_text(encoding="utf-8")
    )
    plan = json.loads(
        (
            root
            / "configs/sota/deform360_pairwise_regret_guard_fresh_source_plan_v1.json"
        ).read_text(encoding="utf-8")
    )
    validate_exclusion_union(union)
    validate_fresh_technical_lock(lock)
    validate_fresh_source_plan(plan)
    assert union["object_hash_count"] == 191
    assert lock["selected_physical_object"]["valid_episode_count"] == 9
    assert lock["selected_physical_object"]["rejected_episode_count"] == 1
    assert plan["download"]["file_count"] == 828


def _real_source_tree(tmp_path: Path) -> Path:
    object_id = "197-hand-sanitizer"
    rows: list[dict[str, object]] = [
        {
            "type": "file",
            "oid": "1" * 40,
            "size": 1,
            "path": f"raw/{object_id}/metadata.json",
        }
    ]
    for index in range(3):
        rows.append(
            {
                "type": "file",
                "oid": f"{index + 2:x}" * 40,
                "size": 1,
                "path": f"raw/{object_id}/calibration_refined/calibration-{index}.json",
            }
        )
    for camera_index, camera in enumerate(FROZEN_CAMERA_PANEL):
        for episode_id in range(10):
            stem = f"{camera}_{episode_id:02d}"
            for extension in ("mp4", "txt"):
                rows.append(
                    {
                        "type": "file",
                        "oid": f"{(camera_index + episode_id) % 16:x}" * 40,
                        "size": 1,
                        "path": f"raw/{object_id}/{camera}/{stem}.{extension}",
                    }
                )
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_source_plan_uses_fixed_panel_and_skips_invalid_episode(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "configs/sota/deform360_pairwise_regret_guard_fresh_technical_v1.json"
    plan = build_fresh_source_plan(lock_path, _real_source_tree(tmp_path))
    validate_fresh_source_plan(plan)
    assert plan["camera_panel"] == list(FROZEN_CAMERA_PANEL)
    assert [row["episode_id"] for row in plan["episode_sources"]] == [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
    ]
    assert all(row["camera_count"] == 12 for row in plan["episode_sources"])


def test_download_manifest_verifies_locked_inventory(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "configs/sota/deform360_pairwise_regret_guard_fresh_technical_v1.json"
    plan = build_fresh_source_plan(lock_path, _real_source_tree(tmp_path))
    source_root = tmp_path / "download"
    metadata_path = source_root / "raw/197-hand-sanitizer/metadata.json"
    for row in plan["download"]["files"]:
        path = source_root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    plan["metadata_file_sha256"] = hashlib.sha256(b"x").hexdigest()
    plan["source_plan_sha256"] = _canonical_sha256(
        plan, digest_key="source_plan_sha256"
    )
    plan_path = tmp_path / "plan.json"
    _write(plan_path, plan)
    manifest = build_fresh_download_manifest(plan_path, source_root)
    validate_fresh_download_manifest(manifest)
    assert manifest["file_count"] == len(plan["download"]["files"])
    metadata_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source size changed|metadata checksum"):
        build_fresh_download_manifest(plan_path, source_root)
