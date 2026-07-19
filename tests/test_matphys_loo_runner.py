from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from bayesian_phystwin.matphys_causal_bridge import (
    matphys_fresh_fold_initialization,
    sha256_file,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts" / "remote" / "run_matphys_causal.py"
FOLD_RUNNER = REPOSITORY_ROOT / "scripts" / "remote" / "run_matphys_loo_folds.py"


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "source_training": {
                    "contract": "source-supervised-meta",
                    "epochs": 5,
                    "teacher_residual_log_scale": 0.5,
                    "learning_rate": 3.0e-5,
                    "gradient_clip": 5.0,
                    "teacher_proximity_weight": 1.0,
                    "random_seed": 42,
                    "initialization": matphys_fresh_fold_initialization(42),
                    "graph_parts": {
                        "part_count": 5,
                        "dino_model": "dinov2_vitl14_reg",
                        "dino_keyframes": 4,
                        "semantic_edge_weight": 4.0,
                        "part_feature_scale": 1.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    artifacts = {}
    for name in ("registration", "source_proxy", "target_proxy"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        artifacts[name] = _identity(path)
    workspace = tmp_path / "workspace.json"
    workspace.write_text(
        json.dumps(
            {
                "contract": "matphys-object-disjoint-loo-workspace-v1",
                "future_opened": False,
                "protocol": _identity(protocol),
                "folds": [
                    {
                        "fold_index": 0,
                        "held_out_object": "target_object",
                        "root": str(tmp_path / "fold_00"),
                        "source_cases": ["source_case"],
                        "target_cases": ["target_case"],
                        **artifacts,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return workspace, protocol


def _command(workspace: Path, tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        str(FOLD_RUNNER),
        str(workspace),
        "--runner",
        str(RUNNER),
        "--python",
        sys.executable,
        "--matphys-root",
        str(tmp_path / "matphys"),
        "--data-root",
        str(tmp_path / "data"),
        "--experiments-dir",
        str(tmp_path / "experiments"),
        "--experiments-optimization-dir",
        str(tmp_path / "optimization"),
        "--nproc-per-node",
        "1",
        "--dry-run",
    ]


def test_fresh_fold_dry_run_passes_seed_and_no_initialization_checkpoint(
    tmp_path: Path,
) -> None:
    workspace, _ = _workspace(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", ""))
    )

    completed = subprocess.run(
        _command(workspace, tmp_path),
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    training_command = completed.stdout.splitlines()[0]
    assert "--random-seed 42" in training_command
    assert "--videomae-model MCG-NJU/videomae-base" in training_command
    assert "--initialization-checkpoint" not in training_command


def test_fresh_fold_rejects_initialization_checkpoint(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"benchmark-trained")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", ""))
    )

    completed = subprocess.run(
        [
            *_command(workspace, tmp_path),
            "--initialization-checkpoint",
            str(checkpoint),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert "forbids a benchmark-trained initialization" in completed.stderr
