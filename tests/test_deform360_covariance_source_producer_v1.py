from __future__ import annotations

import copy
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_covariance_source_producer_v1 as producer
from bayesian_phystwin.deform360_covariance_source_inventory_v1 import (
    build_covariance_source_inventory_v1,
    publish_covariance_source_inventory_v1,
    validate_covariance_source_inventory_v1,
)
from bayesian_phystwin.deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseVisualWindowRowsV5,
)
from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
    SOURCE_ROSTER,
    validate_prediction_batch,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "protocols/locks/deform360_covariance_only_independent_validation_v1.json"
)
SELECTION = (
    ROOT / "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
BINDING = (
    ROOT
    / "protocols/locks/deform360_covariance_only_crossrepo_preregistration_binding_v1.json"
)
SOURCE_EXECUTION_LOCK = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
REVISION = "1" * 40


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _inventory_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "calibration-source"
    processed = tmp_path / "calibration-processed"
    forbidden = tmp_path / "confirmation"
    for root in (source, processed, forbidden):
        root.mkdir()
    for object_id, episode, _stratum in SOURCE_ROSTER:
        source_object = source / object_id
        source_object.mkdir()
        (source_object / "metadata.json").write_text(
            json.dumps({"episode_id": episode, "object_id": object_id}),
            encoding="utf-8",
        )
        processed_object = processed / "aligned" / object_id / "episode_0000"
        processed_object.mkdir(parents=True)
        np.save(processed_object / "prefix.npy", np.zeros((2, 3), dtype=np.float32))
    return source, processed, forbidden


def _build_inventory(tmp_path: Path) -> dict[str, Any]:
    source, processed, forbidden = _inventory_roots(tmp_path)
    return build_covariance_source_inventory_v1(
        protocol_path=PROTOCOL,
        selection_path=SELECTION,
        crossrepo_binding_path=BINDING,
        calibration_source_root=source,
        calibration_processed_root=processed,
        forbidden_confirmation_root=forbidden,
        implementation_revision=REVISION,
    )


def test_inventory_derives_exact_roster_and_reads_only_headers(tmp_path: Path) -> None:
    inventory = _build_inventory(tmp_path)
    assert validate_covariance_source_inventory_v1(inventory) == inventory
    assert [
        (row["object_id"], row["episode"], row["stratum"])
        for row in inventory["source_roster"]
    ] == list(SOURCE_ROSTER)
    assert inventory["missing_source_units"] == []
    assert inventory["information_boundary"] == {
        "array_values_read": False,
        "confirmation_root_entered": False,
        "file_payloads_scored": False,
        "source_roots_only": True,
        "source_suffix_used_for_prediction": False,
        "target_outcomes_opened": False,
    }
    npy_rows = [row for row in inventory["files"] if row["suffix"] == ".npy"]
    assert len(npy_rows) == 10
    assert all(row["array_header"]["shape"] == [2, 3] for row in npy_rows)


def test_inventory_is_deterministic_for_the_same_roots(tmp_path: Path) -> None:
    inventory = _build_inventory(tmp_path)
    repeat = build_covariance_source_inventory_v1(
        protocol_path=PROTOCOL,
        selection_path=SELECTION,
        crossrepo_binding_path=BINDING,
        calibration_source_root=tmp_path / "calibration-source",
        calibration_processed_root=tmp_path / "calibration-processed",
        forbidden_confirmation_root=tmp_path / "confirmation",
        implementation_revision=REVISION,
    )
    assert repeat == inventory


def test_inventory_rejects_changed_selection_and_forbidden_overlap(
    tmp_path: Path,
) -> None:
    source, processed, forbidden = _inventory_roots(tmp_path)
    changed = json.loads(SELECTION.read_text(encoding="utf-8"))
    changed["selection"]["calibration"] = list(
        reversed(changed["selection"]["calibration"])
    )
    changed_path = tmp_path / "changed-selection.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="selection lock bytes changed"):
        build_covariance_source_inventory_v1(
            protocol_path=PROTOCOL,
            selection_path=changed_path,
            crossrepo_binding_path=BINDING,
            calibration_source_root=source,
            calibration_processed_root=processed,
            forbidden_confirmation_root=forbidden,
            implementation_revision=REVISION,
        )
    with pytest.raises(ValueError, match="roots must differ"):
        build_covariance_source_inventory_v1(
            protocol_path=PROTOCOL,
            selection_path=SELECTION,
            crossrepo_binding_path=BINDING,
            calibration_source_root=source,
            calibration_processed_root=processed,
            forbidden_confirmation_root=processed,
            implementation_revision=REVISION,
        )


def _window(
    *,
    camera: str,
    points: np.ndarray,
    confidence: float = 1.0,
) -> Deform360JointSparseVisualWindowRowsV5:
    count = len(points)
    return Deform360JointSparseVisualWindowRowsV5(
        camera_id=camera,
        window_id=f"window:{camera}",
        frame_indices=np.zeros(count, dtype=np.int64),
        pixel_yx=np.zeros((count, 2), dtype=np.int64),
        point_world_m=np.asarray(points, dtype=np.float64),
        point_covariance_m2=np.repeat(
            (1e-6 * np.eye(3, dtype=np.float64))[None],
            count,
            axis=0,
        ),
        source_confidence=np.full(count, confidence, dtype=np.float64),
        mask_distance_pixels=np.full(count, 16.0, dtype=np.float64),
        overlap_disagreement_m=np.zeros(count, dtype=np.float64),
        contributor_count=np.ones(count, dtype=np.int64),
        source_artifact_ids={f"source/{camera}": _sha(camera)},
    )


def _physical(nodes: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(nodes, dtype=np.float64)[None], 58, axis=0)


def test_far_points_have_no_support_and_midpoint_residual_is_candidate_specific() -> (
    None
):
    nodes = np.asarray([[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]], dtype=np.float64)
    far = producer.estimate_covariance_source_residual_history_v1(
        visual_windows=(_window(camera="far", points=np.asarray([[17.0, 0.0, 0.0]])),),
        physical_prediction_m=_physical(nodes),
    )
    assert not np.any(far.valid)
    assert np.array_equal(far.residual_m, np.zeros_like(far.residual_m))

    midpoint = producer.estimate_covariance_source_residual_history_v1(
        visual_windows=(_window(camera="mid", points=np.asarray([[0.005, 0.0, 0.0]])),),
        physical_prediction_m=_physical(nodes),
    )
    assert midpoint.valid[0].tolist() == [True, True]
    assert midpoint.residual_m[0, 0, 0] == pytest.approx(0.005)
    assert midpoint.residual_m[0, 1, 0] == pytest.approx(-0.005)


def test_duplicate_camera_adds_no_precision_and_state_residual_is_not_reliability() -> (
    None
):
    nodes = np.asarray([[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]], dtype=np.float64)
    original = _window(camera="a", points=np.asarray([[0.002, 0.0, 0.0]]))
    duplicate = _window(camera="b", points=np.asarray([[0.002, 0.0, 0.0]]))
    once = producer.estimate_covariance_source_residual_history_v1(
        visual_windows=(original,),
        physical_prediction_m=_physical(nodes),
    )
    twice = producer.estimate_covariance_source_residual_history_v1(
        visual_windows=(original, duplicate),
        physical_prediction_m=_physical(nodes),
    )
    assert np.array_equal(once.valid, twice.valid)
    assert np.allclose(once.residual_m, twice.residual_m, atol=0.0, rtol=0.0)
    assert np.allclose(
        once.observation_covariance_m2,
        twice.observation_covariance_m2,
        atol=1e-15,
        rtol=0.0,
    )
    shifted = producer.estimate_covariance_source_residual_history_v1(
        visual_windows=(_window(camera="c", points=np.asarray([[0.004, 0.0, 0.0]])),),
        physical_prediction_m=_physical(nodes),
    )
    common = once.valid & shifted.valid
    assert np.any(common)
    assert np.allclose(
        once.prior_reliability[common],
        shifted.prior_reliability[common],
        atol=1e-15,
        rtol=0.0,
    )


def _runtime() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "implementation_revision": REVISION,
        "distribution": {"name": "bayesian-phystwin", "version": "test"},
        "environment": {
            "byteorder": "little",
            "machine": "test",
            "python_implementation": "CPython",
            "python_version": "3.12.0",
            "system": "Linux",
        },
        "numerical_runtime": {
            "float64_epsilon": 2.220446049250313e-16,
            "numpy_version": "2.0.0",
        },
    }
    return {**identity, "runtime_id": _sha("runtime")}


def _unit(tmp_path: Path, index: int) -> producer.CovarianceSourceUnitInputsV1:
    object_id, episode, stratum = SOURCE_ROSTER[index]
    physical = tmp_path / f"physical-{index}.npz"
    visual_a = tmp_path / f"visual-a-{index}.npz"
    visual_b = tmp_path / f"visual-b-{index}.npz"
    metric_a = tmp_path / f"metric-a-{index}.npz"
    metric_b = tmp_path / f"metric-b-{index}.npz"
    for path in (physical, visual_a, visual_b, metric_a, metric_b):
        path.write_bytes(path.name.encode("ascii"))
    return producer.CovarianceSourceUnitInputsV1(
        object_id=object_id,
        episode=episode,
        stratum=stratum,
        raw_prefix_range_half_open=(100, 158),
        physical_mode="warp_twin",
        physical_archive_path=physical,
        visual_inputs=(
            ("provider-a", visual_a, metric_a),
            ("provider-b", visual_b, metric_b),
        ),
        reserved_scoring_camera_ids=("score-a", "score-b"),
        source_artifacts={
            f"prefix/{object_id}.npz": {
                "path": f"source/{object_id}/prefix.npz",
                "sha256": _sha(object_id),
                "size_bytes": 1,
            }
        },
    )


def _arrays(*, exact_fallback: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    mean = np.zeros((18, 128, 3), dtype=np.float64)
    covariance = np.zeros((18, 128, 3, 3), dtype=np.float64)
    if not exact_fallback:
        covariance[:] = 1e-6 * np.eye(3, dtype=np.float64)
    arrays = {
        "mean_m": mean,
        "covariance_m2": covariance,
        "residual_history_m": np.zeros((58, 128, 3), dtype=np.float64),
        "residual_valid": np.zeros((58, 128), dtype=np.bool_),
        "observation_covariance_m2": np.zeros((58, 128, 3, 3), dtype=np.float64),
        "prior_reliability": np.zeros((58, 128), dtype=np.float64),
    }
    metadata = {
        "accepted": not exact_fallback,
        "decision": {
            "fallback_reasons": (
                ["insufficient-per-track-support"] if exact_fallback else []
            )
        },
        "diagnostic_code": (
            "insufficient-per-track-support" if exact_fallback else "accepted"
        ),
        "exact_fallback": exact_fallback,
        "provenance": {"provenance_id": _sha("provenance")},
    }
    return arrays, metadata


def test_artifacts_are_deterministic_and_complete_batch_is_gate_accepted(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    manifests = []
    for index in range(10):
        unit = _unit(tmp_path, index)
        arrays, metadata = _arrays(exact_fallback=index == 0)
        directory = (
            tmp_path
            / "panel"
            / "unit-artifacts"
            / f"{index:02d}-{unit.object_id}-ep{unit.episode:04d}"
        )
        manifest = producer._publish_unit_artifact(
            directory,
            unit=unit,
            arrays=arrays,
            metadata=metadata,
            runtime=runtime,
            source_inventory_id=_sha("inventory"),
        )
        assert producer._validate_unit_artifact(directory) == manifest
        manifests.append(manifest)
    batch, records = producer._publish_records_and_batch(
        tmp_path / "panel",
        unit_manifests=manifests,
        runtime=runtime,
    )
    assert len(records) == 100
    assert validate_prediction_batch(batch) == batch
    assert [
        (row["outer_fold_index"], row["source_unit_index"]) for row in batch["records"]
    ] == [(outer, unit) for outer in range(10) for unit in range(10)]
    first = (
        tmp_path
        / "panel"
        / "unit-artifacts"
        / (f"00-{SOURCE_ROSTER[0][0]}-ep{SOURCE_ROSTER[0][1]:04d}")
    )
    copy_root = tmp_path / "copy"
    copy_root.mkdir()
    unit = _unit(copy_root, 0)
    arrays, metadata = _arrays(exact_fallback=True)
    second = copy_root / "artifact"
    producer._publish_unit_artifact(
        second,
        unit=unit,
        arrays=arrays,
        metadata=metadata,
        runtime=runtime,
        source_inventory_id=_sha("inventory"),
    )
    assert hashlib.sha256(
        (first / "prediction-arrays.npz").read_bytes()
    ).hexdigest() == (
        hashlib.sha256((second / "prediction-arrays.npz").read_bytes()).hexdigest()
    )


def test_batch_rejects_reordering_and_technical_receipt_is_not_a_barrier(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    panel = tmp_path / "panel"
    (panel / "unit-artifacts").mkdir(parents=True)
    manifests = []
    for index in range(10):
        unit = _unit(tmp_path, index)
        arrays, metadata = _arrays(exact_fallback=False)
        directory = (
            panel
            / "unit-artifacts"
            / f"{index:02d}-{unit.object_id}-ep{unit.episode:04d}"
        )
        manifests.append(
            producer._publish_unit_artifact(
                directory,
                unit=unit,
                arrays=arrays,
                metadata=metadata,
                runtime=runtime,
                source_inventory_id=_sha("inventory"),
            )
        )
    batch, _records = producer._publish_records_and_batch(
        panel,
        unit_manifests=manifests,
        runtime=runtime,
    )
    changed = copy.deepcopy(batch)
    changed["records"][0], changed["records"][1] = (
        changed["records"][1],
        changed["records"][0],
    )
    changed.pop("batch_id")
    with pytest.raises(ValueError, match="source-unit order changed"):
        from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
            seal_prediction_batch,
        )

        seal_prediction_batch(changed)

    technical = producer.build_covariance_source_technical_receipt_v1(
        implementation_revision=REVISION,
        terminal_stage="source-panel-production",
        diagnostic_code="provider-materialization-failure",
    )
    assert technical["complete_barrier"] is False
    assert technical["prediction_record_count"] == 0
    assert technical["confirmation_prediction_authorized"] is False
    with pytest.raises(ValueError, match="bounded vocabulary"):
        producer.build_covariance_source_technical_receipt_v1(
            implementation_revision=REVISION,
            terminal_stage="source-panel-production",
            diagnostic_code="free-form-error",
        )


def test_unit_publication_is_no_clobber(tmp_path: Path) -> None:
    unit = _unit(tmp_path, 0)
    arrays, metadata = _arrays(exact_fallback=False)
    target = tmp_path / "artifact"
    producer._publish_unit_artifact(
        target,
        unit=unit,
        arrays=arrays,
        metadata=metadata,
        runtime=_runtime(),
        source_inventory_id=_sha("inventory"),
    )
    with pytest.raises(FileExistsError):
        producer._publish_unit_artifact(
            target,
            unit=unit,
            arrays=arrays,
            metadata=metadata,
            runtime=_runtime(),
            source_inventory_id=_sha("inventory"),
        )
    archive = target / "prediction-arrays.npz"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive bytes changed"):
        producer._validate_unit_artifact(target)
    shutil.rmtree(target)


def test_panel_rehash_rejects_a_record_that_differs_from_the_batch(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    panel = tmp_path / "panel"
    (panel / "unit-artifacts").mkdir(parents=True)
    manifests = []
    for index in range(10):
        unit = _unit(tmp_path, index)
        arrays, metadata = _arrays(exact_fallback=False)
        directory = (
            panel
            / "unit-artifacts"
            / f"{index:02d}-{unit.object_id}-ep{unit.episode:04d}"
        )
        manifests.append(
            producer._publish_unit_artifact(
                directory,
                unit=unit,
                arrays=arrays,
                metadata=metadata,
                runtime=runtime,
                source_inventory_id=_sha("inventory"),
            )
        )
    batch, record_digests = producer._publish_records_and_batch(
        panel,
        unit_manifests=manifests,
        runtime=runtime,
    )
    manifest_digests = {
        f"{index:02d}": producer._sha256_file(
            panel
            / "unit-artifacts"
            / f"{index:02d}-{SOURCE_ROSTER[index][0]}-ep{SOURCE_ROSTER[index][1]:04d}"
            / "prediction-manifest.json"
        )
        for index in range(10)
    }
    receipt_identity = {
        "schema": producer.PANEL_RECEIPT_SCHEMA,
        "schema_version": producer.PANEL_RECEIPT_VERSION,
        "status": "source-prediction-barrier-sealed",
        "software_protocol_id": producer.SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": producer.PAPER_PROTOCOL_ID,
        "crossrepo_binding_id": producer.CROSSREPO_BINDING_ID,
        "source_inventory_id": _sha("inventory"),
        "runtime_id": runtime["runtime_id"],
        "implementation_revision": REVISION,
        "upstream_execution_receipt_id": producer.UPSTREAM_EXECUTION_RECEIPT_ID,
        "prediction_batch_id": batch["batch_id"],
        "prediction_batch_file_sha256": producer._sha256_file(
            panel / "source-prediction-batch.json"
        ),
        "prediction_record_count": 100,
        "unit_artifact_count": 10,
        "candidate_unit_count": 10,
        "exact_fallback_unit_count": 0,
        "technical_failure_count": 0,
        "unit_manifest_file_sha256": manifest_digests,
        "record_file_sha256": record_digests,
        "information_boundary": dict(producer._INFORMATION_BOUNDARY),
        "source_suffix_scoring_authorized": False,
        "confirmation_prediction_authorized": False,
        "confirmation_outcome_opening_authorized": False,
        "claim_authorized": False,
    }
    receipt = {
        **receipt_identity,
        "receipt_id": producer.content_id(receipt_identity),
    }
    producer._write_json_once(panel / "source-panel-receipt.json", receipt)
    assert producer.validate_covariance_source_panel_v1(panel) == receipt

    record_path = panel / "records" / "00-00.json"
    changed = json.loads(record_path.read_text(encoding="utf-8"))
    changed["diagnostic_code"] = "changed"
    producer._write_json_once(panel / "records" / "replacement.json", changed)
    record_path.unlink()
    (panel / "records" / "replacement.json").rename(record_path)
    receipt["record_file_sha256"]["00-00.json"] = producer._sha256_file(record_path)
    receipt.pop("receipt_id")
    receipt["receipt_id"] = producer.content_id(receipt)
    (panel / "source-panel-receipt.json").unlink()
    producer._write_json_once(panel / "source-panel-receipt.json", receipt)
    with pytest.raises(ValueError, match="batch-bound unit record"):
        producer.validate_covariance_source_panel_v1(panel)


def test_inventory_records_npz_json_errors_and_publishes_once(tmp_path: Path) -> None:
    source, processed, forbidden = _inventory_roots(tmp_path)
    object_root = source / SOURCE_ROSTER[0][0]
    np.savez(object_root / "packed.npz", points=np.zeros((2, 3), dtype=np.float32))
    (object_root / "invalid.npy").write_bytes(b"not-an-npy")
    (object_root / "invalid.npz").write_bytes(b"not-an-npz")
    with zipfile.ZipFile(object_root / "invalid-member.npz", "w") as archive:
        archive.writestr("invalid.npy", b"not-an-npy")
    (object_root / "invalid.json").write_text("{", encoding="utf-8")
    inventory = build_covariance_source_inventory_v1(
        protocol_path=PROTOCOL,
        selection_path=SELECTION,
        crossrepo_binding_path=BINDING,
        calibration_source_root=source,
        calibration_processed_root=processed,
        forbidden_confirmation_root=forbidden,
        implementation_revision=REVISION,
    )
    npz = next(
        row for row in inventory["files"] if row["relative_path"].endswith("packed.npz")
    )
    invalid = next(
        row
        for row in inventory["files"]
        if row["relative_path"].endswith("invalid.json")
    )
    invalid_npy = next(
        row
        for row in inventory["files"]
        if row["relative_path"].endswith("invalid.npy")
    )
    invalid_npz = next(
        row
        for row in inventory["files"]
        if row["relative_path"].endswith("invalid.npz")
    )
    invalid_member = next(
        row
        for row in inventory["files"]
        if row["relative_path"].endswith("invalid-member.npz")
    )
    assert npz["npz_members"][0]["array_header"]["shape"] == [2, 3]
    assert invalid["json_error"] == "ValueError"
    assert invalid_npy["array_header_error"] == "ValueError"
    assert invalid_npz["npz_header_error"] == "BadZipFile"
    assert invalid_member["npz_members"][0]["array_header_error"] == "ValueError"

    output = tmp_path / "published" / "inventory.json"
    assert publish_covariance_source_inventory_v1(inventory, output) == output
    assert json.loads(output.read_text(encoding="utf-8")) == inventory
    with pytest.raises(FileExistsError):
        publish_covariance_source_inventory_v1(inventory, output)


def test_inventory_still_rejects_npz_member_directories(tmp_path: Path) -> None:
    source, processed, forbidden = _inventory_roots(tmp_path)
    object_root = source / SOURCE_ROSTER[0][0]
    with zipfile.ZipFile(object_root / "directory-member.npz", "w") as archive:
        archive.writestr("nested/", b"")
    with pytest.raises(ValueError, match="member directories are forbidden"):
        build_covariance_source_inventory_v1(
            protocol_path=PROTOCOL,
            selection_path=SELECTION,
            crossrepo_binding_path=BINDING,
            calibration_source_root=source,
            calibration_processed_root=processed,
            forbidden_confirmation_root=forbidden,
            implementation_revision=REVISION,
        )


def _source_plan_fixture(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    input_root = tmp_path / "source-inputs"
    run_root = input_root / "registered-run"
    evidence_root = tmp_path / "compact-evidence"
    run_root.mkdir(parents=True)
    evidence_root.mkdir()
    objects: list[dict[str, Any]] = []
    for index, (object_id, episode, stratum) in enumerate(SOURCE_ROSTER):
        case_id = f"{object_id}-ep{episode:04d}"
        physical = run_root / case_id / "prediction.npz"
        physical.parent.mkdir()
        physical.write_bytes(f"physical-{index}".encode("ascii"))
        physical_record = {
            "path": physical.relative_to(input_root).as_posix(),
            "physical_mode": "warp_twin",
            "sha256": producer._sha256_file(physical),
        }
        manifest = evidence_root / "physical-manifests" / case_id
        manifest.mkdir(parents=True)
        producer._write_json_once(
            manifest / "physical_prediction_manifest.json",
            {
                "artifact_kind": "Deform360BiasAwareProspectivePhysicalPrediction",
                "episode_id": episode,
                "information_boundary": {
                    "future_object_geometry_read": False,
                    "future_object_rgb_read": False,
                    "future_object_track_read": False,
                    "outcome_read": False,
                },
                "object_id": object_id,
                "passed": True,
                "physical_mode": "warp_twin",
                "physical_prediction_archive": {
                    "file_sha256": physical_record["sha256"]
                },
                "schema_version": 1,
                "stratum": stratum,
            },
        )
        windows = []
        for camera in ("provider-a", "provider-b"):
            visual = input_root / "visual" / case_id / camera / "prediction.npz"
            metric = input_root / "metric" / case_id / camera / "metric.npz"
            visual.parent.mkdir(parents=True)
            metric.parent.mkdir(parents=True)
            visual.write_bytes(f"visual-{index}-{camera}".encode("ascii"))
            metric.write_bytes(f"metric-{index}-{camera}".encode("ascii"))
            windows.append(
                {
                    "camera_id": camera,
                    "decoded_uniform": {
                        "path": visual.relative_to(input_root).as_posix(),
                        "sha256": producer._sha256_file(visual),
                    },
                    "metric_prefix": {
                        "path": metric.relative_to(input_root).as_posix(),
                        "sha256": producer._sha256_file(metric),
                    },
                }
            )
        objects.append(
            {
                "episode_id": episode,
                "object_id": object_id,
                "physical": physical_record,
                "raw_prefix_range_half_open": [100, 158],
                "reserved_endpoint_camera_ids": ["score-a", "score-b"],
                "stratum": stratum,
                "visual_windows": windows,
            }
        )
    return {"objects": objects}, input_root, evidence_root


def test_source_resolver_binds_disjoint_prefix_inputs(tmp_path: Path) -> None:
    plan, input_root, evidence_root = _source_plan_fixture(tmp_path)
    units = producer._resolve_source_inputs(
        source_plan=plan,
        input_root=input_root,
        upstream_run_root=input_root / "registered-run",
        upstream_evidence_root=evidence_root,
        common_artifacts={
            "contract.json": {
                "path": "contract.json",
                "sha256": _sha("contract"),
                "size_bytes": 1,
            }
        },
    )
    assert len(units) == 10
    assert [(unit.object_id, unit.episode, unit.stratum) for unit in units] == list(
        SOURCE_ROSTER
    )
    assert all(
        tuple(camera for camera, _visual, _metric in unit.visual_inputs)
        == ("provider-a", "provider-b")
        for unit in units
    )
    assert all(
        not set(unit.reserved_scoring_camera_ids).intersection(
            camera for camera, _visual, _metric in unit.visual_inputs
        )
        for unit in units
    )


def test_execution_receipt_is_content_and_boundary_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {
        "artifacts": {
            "prediction_batch": {"sha256": _sha("batch")},
            "prediction_receipt": {"sha256": _sha("receipt")},
            "source_plan": {"sha256": _sha("plan")},
        },
        "error": None,
        "exit_code": 0,
        "information_boundary": {
            "development_suffix_opened": False,
            "v5_confirmation_outcomes_used": False,
            "v5_confirmation_payloads_opened": False,
            "v6_target_outcomes_used": False,
            "v6_target_payloads_opened": False,
        },
        "physical_manifest_count": 10,
        "schema": "bayesian-phystwin.deform360-v6-source-prediction-execution-receipt",
        "schema_version": 1,
        "source_prediction_seal_count": 100,
        "source_revision": producer.UPSTREAM_REVISION,
        "status": "source-prediction-evidence-sealed",
        "terminal_stage": "completed",
    }
    receipt = {**identity, "receipt_id": producer.content_id(identity)}
    path = tmp_path / "execution-receipt.json"
    producer._write_json_once(path, receipt)
    monkeypatch.setattr(
        producer, "UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256", producer._sha256_file(path)
    )
    monkeypatch.setattr(
        producer, "UPSTREAM_EXECUTION_RECEIPT_ID", receipt["receipt_id"]
    )
    monkeypatch.setattr(
        producer, "UPSTREAM_PREDICTION_BATCH_FILE_SHA256", _sha("batch")
    )
    monkeypatch.setattr(
        producer, "UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256", _sha("receipt")
    )
    monkeypatch.setattr(producer, "UPSTREAM_SOURCE_PLAN_FILE_SHA256", _sha("plan"))
    assert producer._validate_execution_receipt(path) == receipt

    changed = copy.deepcopy(receipt)
    changed["information_boundary"]["development_suffix_opened"] = True
    changed.pop("receipt_id")
    changed["receipt_id"] = producer.content_id(changed)
    changed_path = tmp_path / "changed-execution-receipt.json"
    producer._write_json_once(changed_path, changed)
    monkeypatch.setattr(
        producer,
        "UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256",
        producer._sha256_file(changed_path),
    )
    monkeypatch.setattr(
        producer, "UPSTREAM_EXECUTION_RECEIPT_ID", changed["receipt_id"]
    )
    with pytest.raises(ValueError, match="target-closed source run"):
        producer._validate_execution_receipt(changed_path)


def test_unit_arrays_preserve_registered_mean_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _unit(tmp_path, 0)
    source_artifacts = dict(base.source_artifacts)
    for camera in ("provider-a", "provider-b"):
        source_artifacts[f"visual/{base.object_id}/{camera}/prefix.npz"] = {
            "path": f"visual/{camera}.npz",
            "sha256": _sha(camera),
            "size_bytes": 1,
        }
    unit = producer.CovarianceSourceUnitInputsV1(
        object_id=base.object_id,
        episode=base.episode,
        stratum=base.stratum,
        raw_prefix_range_half_open=base.raw_prefix_range_half_open,
        physical_mode=base.physical_mode,
        physical_archive_path=base.physical_archive_path,
        visual_inputs=base.visual_inputs,
        reserved_scoring_camera_ids=base.reserved_scoring_camera_ids,
        source_artifacts=source_artifacts,
    )
    physical = np.zeros((76, 128, 3), dtype=np.float64)
    monkeypatch.setattr(
        producer, "_load_physical_archive", lambda *_args, **_kwargs: (physical, None)
    )

    def prepare(**kwargs: Any) -> tuple[Deform360JointSparseVisualWindowRowsV5, None]:
        return (
            _window(
                camera=str(kwargs["camera_id"]),
                points=np.asarray([[0.001, 0.0, 0.0]], dtype=np.float64),
            ),
            None,
        )

    monkeypatch.setattr(
        producer, "prepare_deform360_disjoint_visual_window_v6_1", prepare
    )
    fit = producer.Deform360JointSparsePrefixFitV5(
        fit_object_ids=tuple(sorted(item[0] for item in SOURCE_ROSTER)),
        source_artifact_ids={"fit": _sha("fit")},
    )
    arrays, metadata = producer._unit_arrays(
        unit,
        source_inventory_id=_sha("inventory"),
        implementation_revision=REVISION,
        fit=fit,
    )
    assert arrays["mean_m"].shape == (18, 128, 3)
    assert arrays["covariance_m2"].shape == (18, 128, 3, 3)
    assert metadata["exact_fallback"] is True
    assert np.array_equal(arrays["covariance_m2"], np.zeros((18, 128, 3, 3)))


def test_complete_panel_publication_orchestrates_exact_100_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory_roots = tmp_path / "inventory-roots"
    inventory_roots.mkdir()
    inventory = _build_inventory(inventory_roots)
    inventory_path = tmp_path / "inventory.json"
    publish_covariance_source_inventory_v1(inventory, inventory_path)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    upstream_files = {
        name: upstream / f"{name}.json"
        for name in ("plan", "batch", "prediction-receipt", "execution-receipt")
    }
    for path in upstream_files.values():
        producer._write_json_once(path, {})
    monkeypatch.setattr(
        producer,
        "UPSTREAM_SOURCE_PLAN_FILE_SHA256",
        producer._sha256_file(upstream_files["plan"]),
    )
    monkeypatch.setattr(
        producer,
        "UPSTREAM_PREDICTION_BATCH_FILE_SHA256",
        producer._sha256_file(upstream_files["batch"]),
    )
    monkeypatch.setattr(
        producer,
        "UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256",
        producer._sha256_file(upstream_files["prediction-receipt"]),
    )
    monkeypatch.setattr(
        producer,
        "UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256",
        producer._sha256_file(upstream_files["execution-receipt"]),
    )
    monkeypatch.setattr(
        producer,
        "validate_deform360_joint_sparse_source_prediction_plan_v5",
        lambda *_args, **_kwargs: {
            "implementation_revision": producer.UPSTREAM_REVISION,
            "plan_id": producer.UPSTREAM_SOURCE_PLAN_ID,
        },
    )
    monkeypatch.setattr(
        producer,
        "validate_deform360_joint_sparse_source_prediction_batch_v5",
        lambda *_args, **_kwargs: {
            "prediction_batch_id": producer.UPSTREAM_PREDICTION_BATCH_ID
        },
    )
    monkeypatch.setattr(
        producer,
        "validate_deform360_joint_sparse_source_prediction_receipt_v5",
        lambda *_args, **_kwargs: {
            "receipt_id": producer.UPSTREAM_PREDICTION_RECEIPT_ID
        },
    )
    monkeypatch.setattr(producer, "_validate_execution_receipt", lambda _path: {})
    units = tuple(_unit(tmp_path, index) for index in range(10))
    monkeypatch.setattr(producer, "_resolve_source_inputs", lambda **_kwargs: units)

    def unit_arrays(
        unit: producer.CovarianceSourceUnitInputsV1, **_kwargs: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return _arrays(exact_fallback=unit.object_id == SOURCE_ROSTER[0][0])

    monkeypatch.setattr(producer, "_unit_arrays", unit_arrays)
    input_root = tmp_path / "input-root"
    run_root = input_root / "run-root"
    forbidden = tmp_path / "forbidden"
    output_parent = tmp_path / "results"
    run_root.mkdir(parents=True)
    forbidden.mkdir()
    output_parent.mkdir()
    output = output_parent / "panel"
    receipt = producer.publish_covariance_source_panel_v1(
        protocol_path=PROTOCOL,
        selection_path=SELECTION,
        crossrepo_binding_path=BINDING,
        source_execution_lock_path=SOURCE_EXECUTION_LOCK,
        source_inventory_path=inventory_path,
        upstream_source_plan_path=upstream_files["plan"],
        upstream_prediction_batch_path=upstream_files["batch"],
        upstream_prediction_receipt_path=upstream_files["prediction-receipt"],
        upstream_execution_receipt_path=upstream_files["execution-receipt"],
        upstream_run_root=run_root,
        input_root=input_root,
        forbidden_confirmation_root=forbidden,
        repository_root=ROOT,
        output_root=output,
        implementation_revision=REVISION,
    )
    assert receipt["prediction_record_count"] == 100
    assert receipt["candidate_unit_count"] == 9
    assert receipt["exact_fallback_unit_count"] == 1
    assert producer.validate_covariance_source_panel_v1(output) == receipt
    with pytest.raises(ValueError, match="refusing to overwrite"):
        producer.publish_covariance_source_panel_v1(
            protocol_path=PROTOCOL,
            selection_path=SELECTION,
            crossrepo_binding_path=BINDING,
            source_execution_lock_path=SOURCE_EXECUTION_LOCK,
            source_inventory_path=inventory_path,
            upstream_source_plan_path=upstream_files["plan"],
            upstream_prediction_batch_path=upstream_files["batch"],
            upstream_prediction_receipt_path=upstream_files["prediction-receipt"],
            upstream_execution_receipt_path=upstream_files["execution-receipt"],
            upstream_run_root=run_root,
            input_root=input_root,
            forbidden_confirmation_root=forbidden,
            repository_root=ROOT,
            output_root=output,
            implementation_revision=REVISION,
        )
