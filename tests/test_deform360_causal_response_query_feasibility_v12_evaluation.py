from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_causal_response_query import (
    CausalResponseQueryConfig,
    build_causal_response_query_schedule,
    write_causal_response_query_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs" / "sota" / "deform360_causal_response_query_feasibility_v12.json"
)
SCRIPT = (
    ROOT
    / "scripts"
    / "remote"
    / "evaluate_deform360_causal_response_query_v12_source.py"
)


def _admitted_schedule():
    node_count = 24
    camera_count = 12
    height = width = 96
    frame_zero = np.column_stack(
        (
            np.linspace(-0.18, 0.18, node_count),
            0.04 * np.sin(np.linspace(0.0, 2.0 * np.pi, node_count)),
            np.full(node_count, 2.0),
        )
    )
    graph_basis = np.zeros((node_count, 3, 16), dtype=np.float64)
    coordinate = np.linspace(-1.0, 1.0, node_count)
    for mode in range(16):
        graph_basis[:, mode % 3, mode] = coordinate ** (mode % 4 + 1)
    action_support = np.ones(node_count)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 60.0
    intrinsics[:, 1, 1] = 60.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    depth = np.full((camera_count, height, width), 2.0)
    masks = np.ones_like(depth, dtype=bool)
    return build_causal_response_query_schedule(
        frame_zero,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depth,
        masks,
        camera_ids=tuple(f"camera-{index:02d}" for index in range(camera_count)),
        proposal_camera_indices=np.arange(0, camera_count, 2),
        validation_camera_indices=np.arange(1, camera_count, 2),
        config=CausalResponseQueryConfig(),
    )


def _write_case_artifacts(root: Path, case_ids: list[str]) -> None:
    schedule = _admitted_schedule()
    root.mkdir(parents=True)
    physical_manifest = root / "physical.json"
    physical_archive = root / "physical.npz"
    physical_manifest.write_text("{}\n", encoding="utf-8")
    physical_archive.write_text("physical", encoding="utf-8")
    for case_id in case_ids:
        write_causal_response_query_artifacts(
            root / case_id,
            schedule,
            case_id=case_id,
            repository_revision="a" * 40,
            protocol_path=CONFIG,
            physical_manifest_path=physical_manifest,
            physical_archive_path=physical_archive,
            camera_certificate_sha256="b" * 64,
        )


def test_source_evaluator_passes_only_with_the_locked_complete_panel(
    tmp_path: Path,
) -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    case_ids = [row["case"] for row in protocol["cases"]]
    artifacts = tmp_path / "artifacts"
    _write_case_artifacts(artifacts, case_ids)
    output = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(ROOT),
            "--artifact-root",
            str(artifacts),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["status"] == "passed"
    assert result["admitted_case_count"] == 8
    assert result["technical_failure_count"] == 0
    assert result["gate"]["passed"] is True


def test_source_evaluator_counts_a_missing_case_as_a_technical_failure(
    tmp_path: Path,
) -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    case_ids = [row["case"] for row in protocol["cases"]]
    artifacts = tmp_path / "artifacts"
    _write_case_artifacts(artifacts, case_ids[:-1])
    output = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(ROOT),
            "--artifact-root",
            str(artifacts),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert result["technical_failure_count"] == 1
    assert result["gate"]["passed"] is False
