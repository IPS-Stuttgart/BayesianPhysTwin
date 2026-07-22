from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_held_v8_outcome_artifacts as artifacts
import bayesian_phystwin.deform360_held_v8_query_artifacts as query_artifacts


CASE_NAME = "009-cloth-ep0001"


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _bound_file(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _write_lock(path: Path) -> None:
    value: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "TestHeldV8Lock",
        "protocol_id": artifacts.PROTOCOL_ID,
    }
    value["artifact_sha256"] = artifacts._artifact_sha256(value)
    _write_json(path, value)
    path.chmod(0o400)


def _reconstruction(
    identity_count: int = 24,
    *,
    explicit_identities: bool = False,
) -> dict[str, object]:
    index = np.arange(identity_count, dtype=np.float64)
    x0 = np.column_stack(
        (
            index * 0.002,
            (index % 5) * 0.003,
            0.20 + (index % 7) * 0.001,
        )
    )
    points = np.repeat(x0[None], artifacts.FRAME_COUNT, axis=0)
    time = np.arange(artifacts.FRAME_COUNT, dtype=np.float64)[:, None]
    points[:, :, 1] += time * 0.0001
    points[0] = x0
    visible = np.ones((artifacts.FRAME_COUNT, identity_count), dtype=bool)
    valid = visible.copy()
    result: dict[str, object] = {
        "object_points": points,
        "object_visibilities": visible,
        "object_motions_valid": valid,
        "object_colors": np.zeros(
            (artifacts.FRAME_COUNT, identity_count, 3), dtype=np.float32
        ),
        "provenance": {"backend": "unchanged-official-reconstruction-test"},
    }
    if explicit_identities:
        result["identity_ids"] = np.arange(100, 100 + identity_count, dtype=np.int32)
    return result


class _PermitConsumer:
    def __init__(self, expected_operation: str) -> None:
        self.expected_operation = expected_operation
        self.consumed = False

    def __call__(
        self,
        permit: object,
        *,
        case_name: str,
        operation: str,
    ) -> dict[str, object]:
        assert permit is self
        assert case_name == CASE_NAME
        assert operation == self.expected_operation
        if self.consumed:
            raise ValueError("permit already consumed")
        self.consumed = True
        return {
            "protocol_id": artifacts.PROTOCOL_ID,
            "case_name": case_name,
            "operation": operation,
            "single_use_consumed": True,
        }


def _write_bundle(
    root: Path,
    reconstruction: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, Path, Path, dict[str, object]]:
    lock = root / "lock.json"
    _write_lock(lock)
    target_archive = root / "official_target.npz"
    target_manifest = root / "official_target.json"
    query_archive = root / "official_x0.npz"
    query_manifest = root / "official_x0.json"
    source = _reconstruction() if reconstruction is None else reconstruction
    consumer = _PermitConsumer(artifacts.TARGET_RECONSTRUCTION_OPERATION)

    def load() -> dict[str, object]:
        assert consumer.consumed
        return source

    artifacts.write_official_target_and_frame_zero_query_artifacts(
        target_archive,
        target_manifest,
        query_archive,
        query_manifest,
        lock_path=lock,
        lock_sha256=str(_bound_file(lock)["sha256"]),
        case_name=CASE_NAME,
        role="calibration",
        target_reconstruction_permit=consumer,
        consume_target_reconstruction_permit=consumer,
        reconstruction_loader=load,
    )
    assert consumer.consumed
    return (
        lock,
        target_archive,
        target_manifest,
        query_archive,
        query_manifest,
        source,
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def _rewrite_target_and_manifest(
    target_archive: Path,
    target_manifest: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    target_archive.chmod(0o600)
    np.savez_compressed(target_archive, **arrays)
    target_archive.chmod(0o400)
    value = json.loads(target_manifest.read_text(encoding="utf-8"))
    value["archive"] = artifacts._bound_file(target_archive)
    value["array_records"] = artifacts._array_records(arrays)
    value["frame_zero_array_records"] = {
        "identity_ids": artifacts._array_record(arrays["identity_ids"]),
        "positions_m": artifacts._array_record(arrays["object_points"][0]),
    }
    value["artifact_sha256"] = artifacts._artifact_sha256(value)
    target_manifest.chmod(0o600)
    _write_json(target_manifest, value)
    target_manifest.chmod(0o400)


def test_target_and_x0_query_are_exactly_split_and_sealed(tmp_path: Path) -> None:
    lock, target_archive, target_manifest, query_archive, query_manifest, _ = (
        _write_bundle(tmp_path)
    )

    value = artifacts.validate_official_target_artifact(
        target_manifest,
        lock_path=lock,
        expected_case_name=CASE_NAME,
        expected_role="calibration",
    )
    target = _load_npz(target_archive)
    query = _load_npz(query_archive)

    assert set(target) == artifacts.TARGET_ARRAY_NAMES
    assert target["identity_ids"].dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(
        target["identity_ids"], np.arange(len(target["identity_ids"]))
    )
    assert target["object_points"].dtype == np.dtype(np.float32)
    assert target["object_points"].shape[0] == 76
    assert target["object_visibilities"].dtype == np.dtype(bool)
    assert target["object_motions_valid"].dtype == np.dtype(bool)
    assert set(query) == {"identity_ids", "positions_m"}
    assert artifacts._bit_equal(query["identity_ids"], target["identity_ids"])
    assert artifacts._bit_equal(query["positions_m"], target["object_points"][0])
    assert value["identity_rule"] == "implicit-point-axis-order-arange-v1"
    assert value["canonicalization"]["colour_read_or_stored"] is False
    assert value["information_boundary"]["identity_transport_performed"] is False
    assert value["information_boundary"]["source_to_target_assignment_performed"] is (
        False
    )
    for path in (target_archive, target_manifest, query_archive, query_manifest):
        assert path.stat().st_mode & 0o777 == 0o400


def test_failed_or_reused_reconstruction_permit_never_opens_target(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "lock.json"
    _write_lock(lock)
    opened = False

    def reject(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("permit rejected")

    def load() -> dict[str, object]:
        nonlocal opened
        opened = True
        return _reconstruction()

    with pytest.raises(ValueError, match="permit rejected"):
        artifacts.write_official_target_and_frame_zero_query_artifacts(
            tmp_path / "target.npz",
            tmp_path / "target.json",
            tmp_path / "x0.npz",
            tmp_path / "x0.json",
            lock_path=lock,
            lock_sha256=str(_bound_file(lock)["sha256"]),
            case_name=CASE_NAME,
            role="calibration",
            target_reconstruction_permit=object(),
            consume_target_reconstruction_permit=reject,
            reconstruction_loader=load,
        )
    assert opened is False
    assert {path.name for path in tmp_path.iterdir()} == {"lock.json"}

    consumer = _PermitConsumer(artifacts.TARGET_RECONSTRUCTION_OPERATION)
    consumer.consumed = True
    with pytest.raises(ValueError, match="already consumed"):
        artifacts.write_official_target_and_frame_zero_query_artifacts(
            tmp_path / "target.npz",
            tmp_path / "target.json",
            tmp_path / "x0.npz",
            tmp_path / "x0.json",
            lock_path=lock,
            lock_sha256=str(_bound_file(lock)["sha256"]),
            case_name=CASE_NAME,
            role="calibration",
            target_reconstruction_permit=consumer,
            consume_target_reconstruction_permit=consumer,
            reconstruction_loader=load,
        )
    assert opened is False


def test_future_mutation_cannot_change_sealed_x0_bytes(tmp_path: Path) -> None:
    reconstruction = _reconstruction(explicit_identities=True)
    lock, target_archive, target_manifest, query_archive, query_manifest, source = (
        _write_bundle(tmp_path, reconstruction)
    )
    query_archive_before = query_archive.read_bytes()
    query_manifest_before = query_manifest.read_bytes()

    source_points = np.asarray(source["object_points"])
    source_visible = np.asarray(source["object_visibilities"])
    source_points[1:] += 1000.0
    source_visible[1:] = False
    assert query_archive.read_bytes() == query_archive_before
    assert query_manifest.read_bytes() == query_manifest_before

    target = _load_npz(target_archive)
    target["object_points"][1, 0, 0] += np.float32(1.0)
    target_archive.chmod(0o600)
    np.savez_compressed(target_archive, **target)
    target_archive.chmod(0o400)
    with pytest.raises(ValueError, match="file binding changed"):
        artifacts.validate_official_target_artifact(target_manifest, lock_path=lock)
    assert query_archive.read_bytes() == query_archive_before
    assert query_manifest.read_bytes() == query_manifest_before

    _rewrite_target_and_manifest(target_archive, target_manifest, target)
    assert query_archive.read_bytes() == query_archive_before
    assert query_manifest.read_bytes() == query_manifest_before
    query_artifacts.validate_official_frame_zero_query_artifact(
        query_manifest, lock, expected_case_name=CASE_NAME
    )
    # The target was consistently rehashed for this synthetic mutation, so it
    # remains a valid artifact; crucially, its untouched x0 still matches.
    artifacts.validate_official_target_artifact(target_manifest, lock_path=lock)


def test_x0_archive_and_manifest_never_bind_future_mask_or_colour(
    tmp_path: Path,
) -> None:
    _, target_archive, _, query_archive, query_manifest, _ = _write_bundle(tmp_path)
    arrays = _load_npz(query_archive)
    manifest_bytes = query_manifest.read_bytes()

    assert set(arrays) == {"identity_ids", "positions_m"}
    assert str(target_archive).encode() not in manifest_bytes
    for forbidden in (
        b"object_points",
        b"object_visibilities",
        b"object_motions_valid",
        b"object_colors",
    ):
        assert forbidden not in manifest_bytes
    manifest = json.loads(manifest_bytes)
    boundary = manifest["information_boundary"]
    assert boundary["future_source_container_bound_or_hashed"] is False
    assert boundary["future_coordinates_present_or_read"] is False
    assert boundary["visibility_or_validity_present_or_read"] is False
    assert boundary["colour_present_or_read"] is False


@pytest.mark.parametrize("mutation", ["identity", "x0"])
def test_target_to_query_identity_or_x0_mismatch_fails(
    tmp_path: Path,
    mutation: str,
) -> None:
    reconstruction = _reconstruction(explicit_identities=mutation == "identity")
    lock, target_archive, target_manifest, _, _, _ = _write_bundle(
        tmp_path, reconstruction
    )
    target = _load_npz(target_archive)
    if mutation == "identity":
        target["identity_ids"] = target["identity_ids"] + np.int64(100)
    else:
        target["object_points"][0, 0, 0] = np.nextafter(
            target["object_points"][0, 0, 0], np.float32(1.0)
        )
    _rewrite_target_and_manifest(target_archive, target_manifest, target)

    expected = "identity bytes differ" if mutation == "identity" else "x0 bytes differ"
    with pytest.raises(ValueError, match=expected):
        artifacts.validate_official_target_artifact(target_manifest, lock_path=lock)


def test_target_modes_hashes_symlinks_and_exclusive_writes_fail_closed(
    tmp_path: Path,
) -> None:
    lock, target_archive, target_manifest, query_archive, query_manifest, _ = (
        _write_bundle(tmp_path)
    )

    target_archive.chmod(0o444)
    with pytest.raises(ValueError, match="mode 0400"):
        artifacts.validate_official_target_artifact(target_manifest, lock_path=lock)
    target_archive.chmod(0o400)

    linked_manifest = tmp_path / "linked-target.json"
    linked_manifest.symlink_to(target_manifest)
    with pytest.raises(ValueError, match="symlink"):
        artifacts.validate_official_target_artifact(linked_manifest, lock_path=lock)

    target_manifest.chmod(0o600)
    value = json.loads(target_manifest.read_text(encoding="utf-8"))
    value["case_name"] = "tampered-ep0000"
    _write_json(target_manifest, value)
    target_manifest.chmod(0o400)
    with pytest.raises(ValueError, match="content checksum"):
        artifacts.validate_official_target_artifact(target_manifest, lock_path=lock)

    opened = False

    def consume(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal opened
        opened = True
        return {}

    with pytest.raises(ValueError, match="already exists"):
        artifacts.write_official_target_and_frame_zero_query_artifacts(
            target_archive,
            target_manifest,
            query_archive,
            query_manifest,
            lock_path=lock,
            lock_sha256=str(_bound_file(lock)["sha256"]),
            case_name=CASE_NAME,
            role="calibration",
            target_reconstruction_permit=object(),
            consume_target_reconstruction_permit=consume,
            reconstruction_loader=_reconstruction,
        )
    assert opened is False


def _scoring_arrays(
    identity_count: int = 24,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    canonical = artifacts.canonicalize_official_reconstruction(
        _reconstruction(identity_count)
    )
    target = artifacts._target_arrays(canonical)
    queried = {
        "primary_prediction_m": target["object_points"].copy(),
        "selected_raw_backbone_m": target["object_points"].copy(),
        "identity_ids": target["identity_ids"].copy(),
        "positions_m": target["object_points"][0].copy(),
        "shared_support_mask": np.ones(identity_count, dtype=bool),
        "center_exclusion_mask": np.asarray(
            [*([True] * 16), *([False] * (identity_count - 16))], dtype=bool
        ),
        "frame_indices": np.arange(76, dtype=np.int64),
    }
    return queried, target


def test_scoring_adapter_accepts_m_less_or_greater_than_source_n() -> None:
    queried, target = _scoring_arrays(24)
    evidence = {"single_use_consumed": True}
    less = artifacts._assemble_direct_scoring_inputs(
        case_name=CASE_NAME,
        queried=queried,
        target=target,
        source_node_count=1447,
        permit_evidence=evidence,
    )
    greater = artifacts._assemble_direct_scoring_inputs(
        case_name=CASE_NAME,
        queried=queried,
        target=target,
        source_node_count=12,
        permit_evidence=evidence,
    )

    assert less.source_node_count == 1447
    assert greater.source_node_count == 12
    assert less.target_points_m.shape == greater.target_points_m.shape == (76, 24, 3)
    assert less.scoring_kwargs().keys() == greater.scoring_kwargs().keys()
    assert all(
        not value.flags.writeable
        for value in less.scoring_kwargs().values()
        if isinstance(value, np.ndarray)
    )


def test_scoring_adapter_consumes_future_permit_before_target_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried, target = _scoring_arrays(24)
    query_binding = {"path": "/sealed/x0.json", "sha256": "a" * 64, "size_bytes": 7}
    seal = {
        "official_query_manifest": query_binding,
        "official_query_manifest_artifact_sha256": "b" * 64,
    }
    target_manifest = {
        "official_query_manifest": query_binding,
        "official_query_manifest_artifact_sha256": "b" * 64,
    }
    consumed = False
    target_read = False

    monkeypatch.setattr(
        artifacts,
        "_load_validated_queried_prediction_arrays",
        lambda *_args, **_kwargs: (seal, queried),
    )

    def load_target(
        *_args: object, **_kwargs: object
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        nonlocal target_read
        assert consumed
        target_read = True
        return target_manifest, target

    monkeypatch.setattr(artifacts, "_load_validated_target_arrays", load_target)

    def reject(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("future score permit rejected")

    with pytest.raises(ValueError, match="permit rejected"):
        artifacts.load_direct_scoring_inputs_after_future_score_permit(
            case_name=CASE_NAME,
            queried_prediction_seal_path=tmp_path / "query-seal.json",
            target_manifest_path=tmp_path / "target.json",
            lock_path=tmp_path / "lock.json",
            future_score_permit=object(),
            consume_future_score_permit=reject,
        )
    assert target_read is False

    consumer = _PermitConsumer(artifacts.FUTURE_SCORE_OPERATION)

    def consume(
        permit: object,
        *,
        case_name: str,
        operation: str,
    ) -> dict[str, object]:
        nonlocal consumed
        evidence = consumer(permit, case_name=case_name, operation=operation)
        consumed = True
        return evidence

    values = artifacts.load_direct_scoring_inputs_after_future_score_permit(
        case_name=CASE_NAME,
        queried_prediction_seal_path=tmp_path / "query-seal.json",
        target_manifest_path=tmp_path / "target.json",
        lock_path=tmp_path / "lock.json",
        future_score_permit=consumer,
        consume_future_score_permit=consume,
        source_node_count=1447,
    )
    assert target_read is True
    assert values.object_id == "009-cloth"
    assert values.permit_evidence["single_use_consumed"] is True


def test_scoring_adapter_rejects_identity_or_x0_mismatch() -> None:
    queried, target = _scoring_arrays(24)
    identity_mismatch = {name: value.copy() for name, value in queried.items()}
    identity_mismatch["identity_ids"] += np.int64(1)
    with pytest.raises(ValueError, match="identity bytes differ"):
        artifacts._assemble_direct_scoring_inputs(
            case_name=CASE_NAME,
            queried=identity_mismatch,
            target=target,
            source_node_count=None,
            permit_evidence={},
        )

    x0_mismatch = {name: value.copy() for name, value in queried.items()}
    x0_mismatch["positions_m"][0, 0] = np.nextafter(
        x0_mismatch["positions_m"][0, 0], np.float32(1.0)
    )
    with pytest.raises(ValueError, match="frame-zero bytes differ"):
        artifacts._assemble_direct_scoring_inputs(
            case_name=CASE_NAME,
            queried=x0_mismatch,
            target=target,
            source_node_count=None,
            permit_evidence={},
        )


def test_module_is_v8_only_and_contains_no_assignment_or_scorer_import() -> None:
    source = Path(artifacts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("v7" in name for name in imported)
    assert not any("scoring" in name for name in imported)
    assert "scipy" not in imported
    assert "linear_sum_assignment" not in source
    assert artifacts.PROTOCOL_ID.endswith("v8")
