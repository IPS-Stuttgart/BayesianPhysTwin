import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_tapnextpp_competence import (
    canonical_sha256,
    file_sha256,
)


def _load_script(filename: str, name: str):
    path = Path(__file__).parents[1] / "scripts" / "remote" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _artifacts(tmp_path: Path, *, provider_gate_passed: bool):
    protocol = (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_tapnextpp_sparse_assimilation_source_v1.json"
    )
    frame_count = 8
    train_end = 5
    node_count = 6
    original_count = 3
    structure = np.zeros((node_count, 3), dtype=np.float64)
    structure[:, 0] = np.arange(node_count) * 0.01
    baseline = np.broadcast_to(
        structure,
        (frame_count, node_count, 3),
    ).copy()
    physical = tmp_path / "inference.pkl"
    with physical.open("wb") as stream:
        pickle.dump(baseline, stream)
    provider_points = baseline[1:4, [0]].copy()
    provider_points[:, 0, 1] += 0.005
    prediction_input = tmp_path / "prediction_input.npz"
    np.savez_compressed(
        prediction_input,
        prefix_object_points_m=baseline[:train_end, :original_count].astype(np.float32),
        prefix_object_visibilities=np.ones(
            (train_end, original_count),
            dtype=bool,
        ),
        prefix_motion_valid=np.ones(
            (train_end - 1, original_count),
            dtype=bool,
        ),
        structure_points_m=structure.astype(np.float32),
        original_point_count=np.asarray(original_count, np.int64),
        surface_point_count=np.asarray(2, np.int64),
        train_end_frame_exclusive=np.asarray(train_end, np.int64),
        future_end_frame_exclusive=np.asarray(frame_count, np.int64),
        object_radius=np.asarray(0.025),
        object_max_neighbours=np.asarray(4, np.int64),
        controller_radius=np.asarray(0.025),
        controller_max_neighbours=np.asarray(4, np.int64),
        provider_points_world_m=provider_points.astype(np.float32),
        provider_support=np.ones((3, 1), dtype=bool),
        provider_prior_reliability=np.ones((3, 1), dtype=np.float32),
        provider_covariance_m2=np.broadcast_to(
            np.eye(3) * 25e-6,
            (3, 1, 3, 3),
        ).astype(np.float32),
        provider_identity_ids=np.asarray([0], np.int64),
        provider_source_frame_start=np.asarray(1, np.int64),
        provider_source_frame_end_exclusive=np.asarray(4, np.int64),
        provider_gate_passed=np.asarray(provider_gate_passed, bool),
    )
    withheld = tmp_path / "withheld.npz"
    shifted = baseline[train_end:, :original_count].copy()
    shifted[:, :, 1] += 0.005
    manual_frame_zero = baseline[0, [0, 5]].copy()
    manual_future = np.broadcast_to(
        manual_frame_zero,
        (frame_count - train_end, 2, 3),
    ).copy()
    manual_future[:, :, 1] += 0.005
    np.savez_compressed(
        withheld,
        future_object_points_m=shifted.astype(np.float32),
        future_object_visibilities=np.ones(
            (frame_count - train_end, original_count),
            dtype=bool,
        ),
        manual_track_frame_zero_m=manual_frame_zero.astype(np.float32),
        future_manual_tracks_m=manual_future.astype(np.float32),
        provider_identity_ids=np.asarray([0], np.int64),
        train_end_frame_exclusive=np.asarray(train_end, np.int64),
        future_end_frame_exclusive=np.asarray(frame_count, np.int64),
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPSparseAssimilationSourceManifest",
        "protocol_id": "phystwin-tapnextpp-sparse-assimilation-source-v1",
        "protocol_sha256": file_sha256(protocol),
        "case_records": [
            {
                "case": "synthetic_case",
                "prediction_input": {
                    "path": str(prediction_input),
                    "sha256": file_sha256(prediction_input),
                },
                "withheld_outcome": {
                    "path": str(withheld),
                    "sha256": file_sha256(withheld),
                },
                "physical_trajectory": {
                    "path": str(physical),
                    "sha256": file_sha256(physical),
                },
            }
        ],
        "information_boundary": {"held_v8_accessed": False},
    }
    manifest["result_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return protocol, manifest_path, prediction_input, physical, withheld


def test_rejected_provider_is_exact_dense_fallback(tmp_path: Path) -> None:
    runner = _load_script(
        "run_phystwin_tapnextpp_sparse_assimilation.py",
        "tapnextpp_sparse_assimilation_runner_fallback_test",
    )
    protocol, manifest, prediction_input, physical, _ = _artifacts(
        tmp_path,
        provider_gate_passed=False,
    )
    output = tmp_path / "prediction"
    report = runner.predict_case(
        protocol,
        manifest,
        "synthetic_case",
        prediction_input,
        physical,
        output,
    )

    assert report["sparse_update"]["exact_dense_fallback"]
    with np.load(output / runner.PREDICTION_FILENAME, allow_pickle=False) as stored:
        np.testing.assert_array_equal(
            stored["tapnext_direct_future_m"],
            stored["dense_persistence_future_m"],
        )
        np.testing.assert_array_equal(
            stored["tapnext_graph_future_m"],
            stored["dense_persistence_future_m"],
        )


def test_prediction_seals_before_case_evaluation(tmp_path: Path) -> None:
    runner = _load_script(
        "run_phystwin_tapnextpp_sparse_assimilation.py",
        "tapnextpp_sparse_assimilation_runner_update_test",
    )
    evaluator = _load_script(
        "evaluate_phystwin_tapnextpp_sparse_assimilation.py",
        "tapnextpp_sparse_assimilation_evaluator_test",
    )
    protocol, manifest, prediction_input, physical, withheld = _artifacts(
        tmp_path,
        provider_gate_passed=True,
    )
    prediction = tmp_path / "prediction"
    report = runner.predict_case(
        protocol,
        manifest,
        "synthetic_case",
        prediction_input,
        physical,
        prediction,
    )
    assert report["information_boundary"]["future_real_outcome_read"] is False
    assert (prediction / runner.SEAL_FILENAME).is_file()

    output = tmp_path / evaluator.CASE_RESULT_FILENAME
    result = evaluator.evaluate_case(
        protocol,
        manifest,
        "synthetic_case",
        prediction,
        withheld,
        output,
    )
    assert result["information_boundary"]["prediction_sealed_before_future_open"]
    assert set(result["arms"]) == {
        "physical",
        "dense_persistence",
        "tapnext_direct",
        "tapnext_graph",
    }
    assert result["arms"]["tapnext_graph"]["observed_track_error_m"] is not None


def test_evidence_vectors_use_null_for_unsupported_rows() -> None:
    evaluator = _load_script(
        "evaluate_phystwin_tapnextpp_sparse_assimilation.py",
        "tapnextpp_sparse_assimilation_evaluator_null_test",
    )

    assert evaluator._json_vector(np.array([1.0, np.nan])) == [1.0, None]


def test_postopen_audit_detects_material_identity_mismatch(tmp_path: Path) -> None:
    runner = _load_script(
        "run_phystwin_tapnextpp_sparse_assimilation.py",
        "tapnextpp_sparse_assimilation_runner_association_audit_test",
    )
    audit = _load_script(
        "audit_phystwin_tapnextpp_material_association_postopen.py",
        "tapnextpp_sparse_assimilation_association_audit_test",
    )
    protocol, manifest, prediction_input, physical, withheld = _artifacts(
        tmp_path,
        provider_gate_passed=True,
    )
    case_root = tmp_path / "cases" / "synthetic_case"
    prediction = case_root / "prediction"
    runner.predict_case(
        protocol,
        manifest,
        "synthetic_case",
        prediction_input,
        physical,
        prediction,
    )

    outcome = case_root / "withheld_outcome" / audit.OUTCOME_FILENAME
    outcome.parent.mkdir(parents=True)
    with np.load(withheld, allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    with np.load(prediction / runner.PREDICTION_FILENAME, allow_pickle=False) as stored:
        frame_zero = np.asarray(stored["physical_frame_zero_m"])
    manual_frame_zero = np.asarray(values["manual_track_frame_zero_m"]).copy()
    manual_frame_zero[0] = frame_zero[-1]
    values["manual_track_frame_zero_m"] = manual_frame_zero
    np.savez_compressed(outcome, **values)

    result = audit._case_audit(case_root, "synthetic_case")

    assert result["scored_identity_count"] == 1
    assert result["exact_match_count"] == 0
    assert result["identities"][0]["benchmark_frame_zero_node"] == len(frame_zero) - 1
    assert not result["identities"][0]["exact_node_match"]


def test_runner_uses_fixed_material_displacement_mode(tmp_path: Path) -> None:
    runner = _load_script(
        "run_phystwin_tapnextpp_sparse_assimilation.py",
        "tapnextpp_material_transport_runner_test",
    )
    protocol, manifest_path, prediction_input, physical, _ = _artifacts(
        tmp_path,
        provider_gate_passed=True,
    )
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_payload["sparse_assimilation_mode"] = (
        "fixed_frame_zero_material_displacement"
    )
    material_protocol = tmp_path / "material_protocol.json"
    _write_json(material_protocol, protocol_payload)

    with np.load(prediction_input, allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    provider_points = np.asarray(values["provider_points_world_m"]).copy()
    provider_points[:, 0, 1] += np.array([0.0, 0.002, 0.004])
    values["provider_points_world_m"] = provider_points
    values["provider_material_node_indices"] = np.asarray([0], np.int64)
    np.savez_compressed(prediction_input, **values)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_sha256"] = file_sha256(material_protocol)
    manifest["case_records"][0]["prediction_input"]["sha256"] = file_sha256(
        prediction_input
    )
    manifest["result_sha256"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)

    report = runner.predict_case(
        material_protocol,
        manifest_path,
        "synthetic_case",
        prediction_input,
        physical,
        tmp_path / "material_prediction",
    )

    assert (
        report["method_config"]["sparse_assimilation_mode"]
        == "fixed_frame_zero_material_displacement"
    )
    assert report["sparse_update"]["accepted"]
    assert report["sparse_update"]["graph_update"]["observed_nodes"] == [0]
