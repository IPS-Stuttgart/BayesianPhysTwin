from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    INFLATED_FALLBACK_ARM,
    STRICT_ARM,
    AdaptiveCausalResponseQueryConfig,
    build_adaptive_causal_response_query_schedule,
    write_adaptive_causal_response_query_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sota" / "deform360_causal_response_adaptive_query_v13.json"
SCRIPT = (
    ROOT
    / "scripts"
    / "remote"
    / "evaluate_deform360_causal_response_adaptive_query_v13_source.py"
)


def _schedule(*, fallback: bool = False):
    node_count = 24
    camera_count = 8
    height = width = 96
    coordinate = np.linspace(-1.0, 1.0, node_count)
    frame_zero = np.column_stack(
        (
            0.18 * coordinate,
            0.04 * np.sin(np.pi * coordinate),
            np.full(node_count, 2.0),
        )
    )
    graph_basis = np.zeros((node_count, 3, 16), dtype=np.float64)
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
    if fallback:
        masks[[2, 3, 6, 7]] = False
    schedule = build_adaptive_causal_response_query_schedule(
        frame_zero,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depth,
        masks,
        camera_ids=tuple(f"camera-{index:02d}" for index in range(camera_count)),
        config=AdaptiveCausalResponseQueryConfig(),
    )
    assert schedule.arm == (INFLATED_FALLBACK_ARM if fallback else STRICT_ARM)
    return schedule


def _write_case_artifacts(
    root: Path,
    case_ids: list[str],
    *,
    fallback: bool = False,
) -> None:
    schedule = _schedule(fallback=fallback)
    root.mkdir(parents=True)
    physical_manifest = root / "physical.json"
    physical_archive = root / "physical.npz"
    physical_manifest.write_text("{}\n", encoding="utf-8")
    physical_archive.write_text("physical", encoding="utf-8")
    for case_id in case_ids:
        write_adaptive_causal_response_query_artifacts(
            root / case_id,
            schedule,
            case_id=case_id,
            repository_revision="a" * 40,
            protocol_path=CONFIG,
            physical_manifest_path=physical_manifest,
            physical_archive_path=physical_archive,
            camera_certificate_sha256="b" * 64,
        )


def _evaluate(artifacts: Path, output: Path) -> dict:
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
    return json.loads(output.read_text(encoding="utf-8"))


def test_source_evaluator_requires_six_admissions_and_two_strict() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    case_ids = [row["case"] for row in protocol["cases"]]

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        artifacts = root / "artifacts"
        _write_case_artifacts(artifacts, case_ids)
        result = _evaluate(artifacts, root / "summary.json")

    assert result["status"] == "passed"
    assert result["admitted_case_count"] == 8
    assert result["strict_admitted_case_count"] == 8
    assert result["fallback_admitted_case_count"] == 0
    assert result["technical_failure_count"] == 0
    assert result["gate"]["passed"] is True


def test_source_evaluator_rejects_all_fallback_admissions(tmp_path: Path) -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    case_ids = [row["case"] for row in protocol["cases"]]
    artifacts = tmp_path / "artifacts"
    _write_case_artifacts(artifacts, case_ids, fallback=True)
    result = _evaluate(artifacts, tmp_path / "summary.json")

    assert result["admitted_case_count"] == 8
    assert result["strict_admitted_case_count"] == 0
    assert result["fallback_admitted_case_count"] == 8
    assert result["status"] == "failed"
    assert result["gate"]["passed"] is False


def test_source_evaluator_counts_missing_case_as_technical_failure(
    tmp_path: Path,
) -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    case_ids = [row["case"] for row in protocol["cases"]]
    artifacts = tmp_path / "artifacts"
    _write_case_artifacts(artifacts, case_ids[:-1])
    result = _evaluate(artifacts, tmp_path / "summary.json")

    assert result["status"] == "failed"
    assert result["technical_failure_count"] == 1
    assert result["gate"]["passed"] is False
