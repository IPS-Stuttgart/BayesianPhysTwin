from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin.matphys_causal_bridge import (
    validate_causal_training_audit,
)
from bayesian_phystwin.matphys_reconstruction_control import (
    MATPHYS_RECONSTRUCTION_AUDIT_CONTRACT,
    MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
    MATPHYS_RECONSTRUCTION_OBJECTIVE_GUARD,
    MATPHYS_RECONSTRUCTION_RESULT_CONTRACT,
    MATPHYS_RECONSTRUCTION_SINGLE_CASE_LOADER_COMPATIBILITY,
    MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
    MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
    build_matphys_reconstruction_result,
    validate_matphys_reconstruction_audit,
    validate_matphys_reconstruction_protocol,
    write_matphys_reconstruction_audit,
)


def _identity(path: Path) -> dict[str, str]:
    import hashlib

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    case = tmp_path / "case_a"
    color = case / "color" / "0"
    color.mkdir(parents=True)
    split = case / "split.json"
    split.write_text(json.dumps({"train": [0, 3], "test": [3, 5], "frame_len": 5}))
    frames = {}
    for frame_id in (0, 2, 4):
        frame = color / f"{frame_id}.png"
        frame.write_bytes(f"frame-{frame_id}".encode())
        frames[frame_id] = frame
    proxy_root = tmp_path / "proxy"
    proxy_root.mkdir()
    mapping = proxy_root / "case_to_material.json"
    mapping.write_text(
        json.dumps(
            {
                "case_to_material": {"case_a": "cloth"},
                "class_to_id": {"cloth": 0},
            }
        )
    )
    node_sem = proxy_root / "node_sem.npz"
    train_ready = proxy_root / "train_ready.pt"
    node_sem.write_bytes(b"node-sem")
    train_ready.write_bytes(b"train-ready")
    proxy = proxy_root / "proxy_summary.json"
    proxy.write_text(
        json.dumps(
            {
                "contract": "causal-dino-graph-voronoi-parts-v1",
                "mapping": _identity(mapping),
                "cases": [
                    {
                        "name": "case_a",
                        "node_sem": _identity(node_sem),
                        "train_ready": _identity(train_ready),
                        "semantic_dimension": 8,
                    }
                ],
            }
        )
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    finiteness = tmp_path / "checkpoint_finiteness.json"
    finiteness.write_text(
        json.dumps(
            {
                "contract": "finite-model-and-optimizer-checkpoint-v1",
                "finite": True,
                "checkpoint": _identity(checkpoint),
            }
        )
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "case": {"case_id": "case_a"},
                "implementation": {
                    "matphys_revision": "a" * 40,
                    "epochs": 200,
                    "eval_every": 10,
                    "learning_rate": 0.0003,
                    "random_seed": 42,
                    "fit_all_frames": True,
                    "proxy_contract": "causal-dino-graph-voronoi-parts-v1",
                    "part_model_contract": "simple-videomae-dino-part-conditioning-v1",
                    "part_feature_scale": 1.0,
                    "warp_warning_compatibility": "warp-private-warn-signature-v1",
                    "single_case_loader_compatibility": (
                        "matphys-single-case-provisional-split-v1"
                    ),
                    "objective_guard": "exact-full-sequence-objective-v1",
                },
            }
        )
    )
    configuration = {
        "epochs": 200,
        "eval_every": 10,
        "learning_rate": 0.0003,
        "random_seed": 42,
        "fit_all_frames": True,
        "video_scope": MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
        "training_scope": MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
        "checkpoint_policy": MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
        "warp_warning_compatibility": "warp-private-warn-signature-v1",
        "single_case_loader_compatibility": (
            MATPHYS_RECONSTRUCTION_SINGLE_CASE_LOADER_COMPATIBILITY
        ),
        "objective_guard": MATPHYS_RECONSTRUCTION_OBJECTIVE_GUARD,
        "proxy_contract": "causal-dino-graph-voronoi-parts-v1",
        "part_model_contract": "simple-videomae-dino-part-conditioning-v1",
        "part_feature_scale": 1.0,
        "semantic_dimension": 8,
    }
    audit = tmp_path / "audit.json"
    write_matphys_reconstruction_audit(
        checkpoint,
        audit,
        protocol_path=protocol,
        source_repository="https://example.test/matphys",
        source_commit="a" * 40,
        data_root=tmp_path,
        case_name="case_a",
        split_path=split,
        accessed_frame_indices=[0, 2, 4],
        accessed_frame_paths=frames,
        objective_end_frame_exclusive=5,
        proxy_summary_path=proxy,
        training_configuration=configuration,
        runtime_access_log_paths=(finiteness,),
    )
    return audit, checkpoint, configuration


def _result_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    audit, checkpoint, _ = _fixture(tmp_path)
    audit_payload = json.loads(audit.read_text())
    split = Path(audit_payload["case"]["split"]["path"])
    final_data = tmp_path / "final_data.pkl"
    gt_track = tmp_path / "gt_track_3d.pkl"
    candidate_trajectory = tmp_path / "candidate.pkl"
    baseline_trajectory = tmp_path / "baseline.pkl"
    spring = tmp_path / "spring.npy"
    globals_path = tmp_path / "globals.json"
    for path, content in (
        (final_data, b"final-data"),
        (gt_track, b"gt-track"),
        (candidate_trajectory, b"candidate"),
        (baseline_trajectory, b"baseline"),
    ):
        path.write_bytes(content)
    spring_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(spring, spring_values, allow_pickle=False)
    globals_path.write_text(
        json.dumps(
            {
                "collide_elas": 0.1,
                "collide_fric": 0.2,
                "collide_object_elas": 0.3,
                "collide_object_fric": 0.4,
                "collision_dist": 0.01,
                "dashpot_damping": 10.0,
                "drag_damping": 2.0,
            }
        )
    )

    def metrics(path: Path, trajectory: Path, *, cd: float, track: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "inputs": {
                        "trajectory": _identity(trajectory),
                        "final_data": _identity(final_data),
                        "gt_track_3d": _identity(gt_track),
                        "split": _identity(split),
                    },
                    "split": json.loads(split.read_text()),
                    "evaluation": {
                        "train": {
                            "chamfer_distance_m": cd,
                            "track_error_m": track,
                        },
                        "test": {
                            "chamfer_distance_m": cd,
                            "track_error_m": track,
                        },
                    },
                }
            )
        )

    candidate_metrics = tmp_path / "candidate_metrics.json"
    baseline_metrics = tmp_path / "baseline_metrics.json"
    metrics(candidate_metrics, candidate_trajectory, cd=0.008, track=0.009)
    metrics(baseline_metrics, baseline_trajectory, cd=0.010, track=0.012)
    export_manifest = tmp_path / "export_manifest.json"
    export_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "contract": "matphys-all-frame-part-aware-reconstruction-export-v2",
                "claim_boundary": audit_payload["claim_boundary"],
                "future_observations_used": True,
                "predictive_use_authorized": False,
                "method": {
                    "part_model_contract": (
                        "simple-videomae-dino-part-conditioning-v1"
                    ),
                    "published_matphys_method": False,
                },
                "checkpoint": _identity(checkpoint),
                "training_audit": _identity(audit),
                "case": {
                    "name": "case_a",
                    "spring_field": {
                        **_identity(spring),
                        "count": 3,
                        "minimum": float(np.min(spring_values)),
                        "maximum": float(np.max(spring_values)),
                        "geometric_mean": float(
                            np.exp(np.mean(np.log(spring_values)))
                        ),
                    },
                    "global_parameters": _identity(globals_path),
                    "official_metrics": _identity(candidate_metrics),
                },
            }
        )
    )
    return audit, checkpoint, export_manifest, baseline_metrics


def test_reconstruction_audit_binds_deliberate_future_access(tmp_path: Path) -> None:
    audit, checkpoint, _ = _fixture(tmp_path)

    validated = validate_matphys_reconstruction_audit(audit, checkpoint)

    assert validated["contract"] == MATPHYS_RECONSTRUCTION_AUDIT_CONTRACT
    assert validated["future_observations_used"] is True
    assert validated["predictive_use_authorized"] is False
    assert validated["case"]["fitted_future_interval"] == [3, 5]


def test_reconstruction_result_recomputes_joint_terminal_decision(
    tmp_path: Path,
) -> None:
    audit, checkpoint, manifest, baseline = _result_fixture(tmp_path)

    result = build_matphys_reconstruction_result(
        audit, checkpoint, manifest, baseline
    )

    assert result["contract"] == MATPHYS_RECONSTRUCTION_RESULT_CONTRACT
    assert result["decision"] == {
        "capacity_pass": True,
        "backend_export_pass": True,
        "advance_to_source_only_causal_design": True,
        "authorizes_predictive_use_of_checkpoint": False,
    }
    assert result["terminal_test_metrics_mm"] == {
        "chamfer_distance": 8.0,
        "track_error": 9.0,
    }


def test_reconstruction_result_requires_both_metrics_to_improve(
    tmp_path: Path,
) -> None:
    audit, checkpoint, manifest, baseline = _result_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    metrics_path = Path(payload["case"]["official_metrics"]["path"])
    metrics = json.loads(metrics_path.read_text())
    metrics["evaluation"]["test"]["track_error_m"] = 0.013
    metrics_path.write_text(json.dumps(metrics))
    payload["case"]["official_metrics"] = _identity(metrics_path)
    manifest.write_text(json.dumps(payload))

    result = build_matphys_reconstruction_result(
        audit, checkpoint, manifest, baseline
    )

    assert result["decision"]["capacity_pass"] is False
    assert result["decision"]["advance_to_source_only_causal_design"] is False


def test_reconstruction_result_rejects_checkpoint_substitution(tmp_path: Path) -> None:
    audit, checkpoint, manifest, baseline = _result_fixture(tmp_path)
    replacement = tmp_path / "best_checkpoint.pth"
    replacement.write_bytes(checkpoint.read_bytes())

    with pytest.raises(ValueError, match="not bound by the reconstruction audit"):
        build_matphys_reconstruction_result(
            audit, replacement, manifest, baseline
        )


def test_reconstruction_result_rejects_metric_input_drift(tmp_path: Path) -> None:
    audit, checkpoint, manifest, baseline = _result_fixture(tmp_path)
    baseline_payload = json.loads(baseline.read_text())
    drifted = tmp_path / "drifted_final_data.pkl"
    drifted.write_bytes(b"drifted")
    baseline_payload["inputs"]["final_data"] = _identity(drifted)
    baseline.write_text(json.dumps(baseline_payload))

    with pytest.raises(ValueError, match="final_data identities differ"):
        build_matphys_reconstruction_result(
            audit, checkpoint, manifest, baseline
        )


def test_reconstruction_result_rejects_forged_spring_summary(tmp_path: Path) -> None:
    audit, checkpoint, manifest, baseline = _result_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["case"]["spring_field"]["minimum"] = 2.0
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="spring summary differs"):
        build_matphys_reconstruction_result(
            audit, checkpoint, manifest, baseline
        )


def test_reconstruction_checkpoint_cannot_pass_causal_validator(tmp_path: Path) -> None:
    audit, checkpoint, _ = _fixture(tmp_path)

    with pytest.raises(ValueError, match="does not forbid future observations"):
        validate_causal_training_audit(audit, checkpoint)


def test_reconstruction_audit_requires_future_rgb(tmp_path: Path) -> None:
    audit, checkpoint, configuration = _fixture(tmp_path)
    payload = json.loads(audit.read_text())
    protocol = payload["protocol"]["path"]
    proxy = payload["proxy"]["path"]
    split = payload["case"]["split"]["path"]
    frames = {
        int(record["frame_id"]): record["path"]
        for record in payload["case"]["accessed_frame_files"]
        if int(record["frame_id"]) < 3
    }

    with pytest.raises(ValueError, match="did not access future RGB"):
        write_matphys_reconstruction_audit(
            checkpoint,
            tmp_path / "bad.json",
            protocol_path=protocol,
            source_repository="https://example.test/matphys",
            source_commit="a" * 40,
            data_root=tmp_path,
            case_name="case_a",
            split_path=split,
            accessed_frame_indices=sorted(frames),
            accessed_frame_paths=frames,
            objective_end_frame_exclusive=5,
            proxy_summary_path=proxy,
            training_configuration=configuration,
        )


def test_reconstruction_audit_rejects_predictive_relabeling(tmp_path: Path) -> None:
    audit, checkpoint, _ = _fixture(tmp_path)
    payload = json.loads(audit.read_text())
    payload["predictive_use_authorized"] = True
    audit.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="predictive_use_authorized"):
        validate_matphys_reconstruction_audit(audit, checkpoint)


def test_reconstruction_audit_rejects_unwired_part_decoder(tmp_path: Path) -> None:
    audit, checkpoint, _ = _fixture(tmp_path)
    payload = json.loads(audit.read_text())
    payload["training_configuration"].pop("part_model_contract")
    audit.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="part_model_contract"):
        validate_matphys_reconstruction_audit(audit, checkpoint)


def test_reconstruction_protocol_rejects_unwired_part_decoder(tmp_path: Path) -> None:
    audit, _, configuration = _fixture(tmp_path)
    protocol = Path(json.loads(audit.read_text())["protocol"]["path"])
    payload = json.loads(protocol.read_text())
    payload["implementation"].pop("part_model_contract")
    protocol.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="training settings differ"):
        validate_matphys_reconstruction_protocol(
            protocol,
            case_name="case_a",
            source_commit="a" * 40,
            training_configuration=configuration,
        )


def test_reconstruction_audit_accepts_registered_compact_proxy(tmp_path: Path) -> None:
    audit, checkpoint, _ = _fixture(tmp_path)
    payload = json.loads(audit.read_text())
    proxy_path = Path(payload["proxy"]["path"])
    proxy = json.loads(proxy_path.read_text())
    compact_contract = "causal-dino-graph-parts-compact-unused-edge-semantics-v1"
    proxy["contract"] = compact_contract
    proxy_path.write_text(json.dumps(proxy))
    protocol_path = Path(payload["protocol"]["path"])
    protocol = json.loads(protocol_path.read_text())
    protocol["implementation"]["proxy_contract"] = compact_contract
    protocol_path.write_text(json.dumps(protocol))
    payload["proxy"] = _identity(proxy_path)
    payload["protocol"] = _identity(protocol_path)
    payload["training_configuration"]["proxy_contract"] = compact_contract
    audit.write_text(json.dumps(payload))

    validated = validate_matphys_reconstruction_audit(audit, checkpoint)

    assert validated["training_configuration"]["proxy_contract"] == compact_contract


def test_reconstruction_audit_rejects_extra_proxy_mapping_case(tmp_path: Path) -> None:
    audit, checkpoint, _ = _fixture(tmp_path)
    payload = json.loads(audit.read_text())
    proxy_path = Path(payload["proxy"]["path"])
    proxy = json.loads(proxy_path.read_text())
    mapping_path = Path(proxy["mapping"]["path"])
    mapping = json.loads(mapping_path.read_text())
    mapping["case_to_material"]["case_b"] = "cloth"
    mapping_path.write_text(json.dumps(mapping))
    proxy["mapping"] = _identity(mapping_path)
    proxy_path.write_text(json.dumps(proxy))
    payload["proxy"] = _identity(proxy_path)
    audit.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="exactly its case"):
        validate_matphys_reconstruction_audit(audit, checkpoint)


def test_reconstruction_runner_installs_part_adapter(monkeypatch) -> None:
    scripts = Path(__file__).parents[1] / "scripts" / "remote"
    path = scripts / "run_matphys_reconstruction_control.py"
    spec = importlib.util.spec_from_file_location(
        "run_matphys_reconstruction_control_test", path
    )
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(scripts))
    try:
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    received = {}

    def install(training, *, part_feature_dim, part_feature_scale):
        received.update(
            training=training,
            part_feature_dim=part_feature_dim,
            part_feature_scale=part_feature_scale,
        )

    monkeypatch.setattr(runner, "install_part_aware_simple_model", install)
    training = SimpleNamespace()
    dimension = runner._install_reconstruction_part_model(
        training,
        {"cases": [{"semantic_dimension": 1024}]},
        part_feature_scale=1.0,
    )

    assert dimension == 1024
    assert received == {
        "training": training,
        "part_feature_dim": 1024,
        "part_feature_scale": 1.0,
    }
    import ast

    for handler in (runner.train, runner.export):
        tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_install_reconstruction_part_model"
            for node in ast.walk(tree)
        )


def test_reconstruction_warp_warn_compatibility_preserves_once_signature(
    monkeypatch,
) -> None:
    scripts = Path(__file__).parents[1] / "scripts" / "remote"
    path = scripts / "run_matphys_reconstruction_control.py"
    spec = importlib.util.spec_from_file_location(
        "run_matphys_reconstruction_control_warn_test", path
    )
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(scripts))
    try:
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    messages = []
    fake_utils = SimpleNamespace()
    fake_warp = SimpleNamespace(_src=SimpleNamespace(utils=fake_utils))
    monkeypatch.setitem(sys.modules, "warp", fake_warp)
    monkeypatch.setitem(sys.modules, "warp._src", fake_warp._src)
    monkeypatch.setitem(sys.modules, "warp._src.utils", fake_utils)
    monkeypatch.setattr(
        "warnings.warn",
        lambda message, category=None, stacklevel=1: messages.append(
            (str(message), category, stacklevel)
        ),
    )

    runner._install_warp_warn_compatibility()
    fake_utils.warn("same", category=RuntimeWarning, stacklevel=3, once=True)
    fake_utils.warn("same", category=RuntimeWarning, stacklevel=3, once=True)
    fake_utils.warn("repeat", once=False)
    fake_utils.warn("repeat", once=False)

    assert messages == [
        ("same", RuntimeWarning, 4),
        ("repeat", None, 2),
        ("repeat", None, 2),
    ]


def test_reconstruction_single_case_loader_compatibility_avoids_empty_split(
    monkeypatch,
) -> None:
    scripts = Path(__file__).parents[1] / "scripts" / "remote"
    path = scripts / "run_matphys_reconstruction_control.py"
    spec = importlib.util.spec_from_file_location(
        "run_matphys_reconstruction_control_loader_test", path
    )
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(scripts))
    try:
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    received = {}
    full_dataset = ["only-case"]

    def create_train_test_dataloaders(*args, **kwargs):
        received.update(args=args, kwargs=kwargs)
        return full_dataset, "provisional-train", "provisional-test"

    training = SimpleNamespace(
        create_train_test_dataloaders=create_train_test_dataloaders
    )
    runner._install_single_case_loader_compatibility(training)

    result = training.create_train_test_dataloaders(
        cfg="cfg", batch_size=1, train_ratio=0.8
    )

    assert result == (full_dataset, "provisional-train", "provisional-test")
    assert received["kwargs"]["train_ratio"] == 1.0
    assert (
        training._single_case_loader_compatibility
        == MATPHYS_RECONSTRUCTION_SINGLE_CASE_LOADER_COMPATIBILITY
    )


def test_reconstruction_objective_guard_requires_exact_full_sequence(
    tmp_path: Path,
) -> None:
    scripts = Path(__file__).parents[1] / "scripts" / "remote"
    path = scripts / "run_matphys_reconstruction_control.py"
    spec = importlib.util.spec_from_file_location(
        "run_matphys_reconstruction_control_objective_test", path
    )
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(scripts))
    try:
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    case = tmp_path / "case_a"
    case.mkdir()
    (case / "split.json").write_text(
        json.dumps({"train": [0, 3], "test": [3, 5], "frame_len": 5})
    )
    training = SimpleNamespace(
        _resolve_train_frame=lambda args, case_name, train_frame: (
            5 if args.fit_all_frames else train_frame
        )
    )
    runner._OBJECTIVE_END_FRAMES.clear()
    runner._install_reconstruction_objective_guard(
        training, tmp_path, {"case_a": 5}
    )

    assert (
        training._resolve_train_frame(
            SimpleNamespace(fit_all_frames=True), "case_a", 3
        )
        == 5
    )
    assert runner._OBJECTIVE_END_FRAMES == {"case_a": 5}
    assert (
        training._reconstruction_objective_guard
        == MATPHYS_RECONSTRUCTION_OBJECTIVE_GUARD
    )
    with pytest.raises(RuntimeError, match="requires --fit_all_frames"):
        training._resolve_train_frame(
            SimpleNamespace(fit_all_frames=False), "case_a", 3
        )
