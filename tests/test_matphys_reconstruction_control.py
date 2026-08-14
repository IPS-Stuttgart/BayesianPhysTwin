from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesian_phystwin.matphys_causal_bridge import (
    validate_causal_training_audit,
)
from bayesian_phystwin.matphys_reconstruction_control import (
    MATPHYS_RECONSTRUCTION_AUDIT_CONTRACT,
    MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
    MATPHYS_RECONSTRUCTION_SINGLE_CASE_LOADER_COMPATIBILITY,
    MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
    MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
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
    )
    return audit, checkpoint, configuration


def test_reconstruction_audit_binds_deliberate_future_access(tmp_path: Path) -> None:
    audit, checkpoint, _ = _fixture(tmp_path)

    validated = validate_matphys_reconstruction_audit(audit, checkpoint)

    assert validated["contract"] == MATPHYS_RECONSTRUCTION_AUDIT_CONTRACT
    assert validated["future_observations_used"] is True
    assert validated["predictive_use_authorized"] is False
    assert validated["case"]["fitted_future_interval"] == [3, 5]


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
