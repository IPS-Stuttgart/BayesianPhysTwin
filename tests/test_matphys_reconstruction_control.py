from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.matphys_causal_bridge import (
    validate_causal_training_audit,
)
from bayesian_phystwin.matphys_reconstruction_control import (
    MATPHYS_RECONSTRUCTION_AUDIT_CONTRACT,
    MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
    MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
    MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
    validate_matphys_reconstruction_audit,
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
    node_sem = proxy_root / "node_sem.npz"
    train_ready = proxy_root / "train_ready.pt"
    node_sem.write_bytes(b"node-sem")
    train_ready.write_bytes(b"train-ready")
    proxy = proxy_root / "proxy_summary.json"
    proxy.write_text(
        json.dumps(
            {
                "contract": "causal-dino-graph-voronoi-parts-v1",
                "cases": [
                    {
                        "name": "case_a",
                        "node_sem": _identity(node_sem),
                        "train_ready": _identity(train_ready),
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
        "proxy_contract": "causal-dino-graph-voronoi-parts-v1",
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

    with pytest.raises(ValueError, match="unsupported or legacy MatPhys causal-audit"):
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
