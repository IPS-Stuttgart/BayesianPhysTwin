"""Direct paired comparison of additional-cohort spatial controls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .phystwin_comparison import (
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)


def _load_run(run_dir: str | Path) -> tuple[dict[str, object], dict[str, Path]]:
    root = Path(run_dir)
    protocol_path = root / "locked_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"missing locked protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summaries = {
        path.parent.name: path
        for path in sorted((root / "cases").glob("*/summary.json"))
    }
    if not summaries:
        raise ValueError(f"run has no case summaries: {root}")
    return protocol, summaries


def _spatial_mode(
    protocol: dict[str, object],
    summaries: dict[str, Path],
) -> str:
    specification = protocol["specification"]
    mode = specification.get("spatial_mode")
    if mode is not None:
        return str(mode)
    first_path = next(iter(summaries.values()))
    first_summary = json.loads(first_path.read_text(encoding="utf-8"))
    mode = first_summary.get("config", {}).get("spatial_mode")
    if mode is not None:
        return str(mode)
    if specification.get("method") == "ungated capped persistent residual anchor":
        return "per_point"
    raise ValueError("run does not identify its spatial mode")


def compare_additional_anchor_controls(
    candidate_run_dir: str | Path,
    reference_run_dirs: Sequence[str | Path],
    *,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260710,
) -> dict[str, object]:
    """Compare one frozen anchor directly with fixed spatial controls."""

    if not reference_run_dirs:
        raise ValueError("at least one reference run is required")
    candidate_protocol, candidate_paths = _load_run(candidate_run_dir)
    candidate_mode = _spatial_mode(candidate_protocol, candidate_paths)
    clusters = {
        case: phystwin_physical_object_cluster(case) for case in candidate_paths
    }
    comparisons: dict[str, object] = {}
    for reference_run_dir in reference_run_dirs:
        reference_protocol, reference_paths = _load_run(reference_run_dir)
        if set(reference_paths) != set(candidate_paths):
            raise ValueError("candidate and reference runs must contain the same cases")
        reference_mode = _spatial_mode(reference_protocol, reference_paths)
        if reference_mode in comparisons:
            raise ValueError(f"duplicate reference spatial mode: {reference_mode}")
        paired: dict[
            str,
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
        ] = {}
        candidate_better = 0
        for case, candidate_path in candidate_paths.items():
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            reference = json.loads(reference_paths[case].read_text(encoding="utf-8"))
            candidate_future = candidate["future"]
            reference_future = reference["future"]
            candidate_baseline = np.asarray(
                candidate_future["baseline_chamfer_by_frame_m"], dtype=float
            )
            reference_baseline = np.asarray(
                reference_future["baseline_chamfer_by_frame_m"], dtype=float
            )
            if candidate_baseline.shape != reference_baseline.shape or not np.allclose(
                candidate_baseline,
                reference_baseline,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"released baseline differs for {case}")
            candidate_values = np.asarray(
                candidate_future["corrected_chamfer_by_frame_m"], dtype=float
            )
            reference_values = np.asarray(
                reference_future["corrected_chamfer_by_frame_m"], dtype=float
            )
            if candidate_values.shape != reference_values.shape:
                raise ValueError(f"future intervals differ for {case}")
            candidate_better += int(
                np.mean(candidate_values) < np.mean(reference_values)
            )
            paired[case] = (
                {"chamfer_distance_m": reference_values},
                {"chamfer_distance_m": candidate_values},
            )
        comparisons[reference_mode] = {
            "reference_run_dir": str(Path(reference_run_dir).resolve()),
            "reference_protocol_id": reference_protocol["protocol_id"],
            "candidate_better_case_count": candidate_better,
            "case_count": len(paired),
            "bootstrap": paired_block_bootstrap(
                paired,
                samples=bootstrap_samples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed,
                clusters=clusters,
            ),
        }
    return {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "effect_definition": (
            "100 * (candidate future CD / reference future CD - 1); "
            "negative favors candidate"
        ),
        "candidate": {
            "run_dir": str(Path(candidate_run_dir).resolve()),
            "spatial_mode": candidate_mode,
            "protocol_id": candidate_protocol["protocol_id"],
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "block_length": bootstrap_block_length,
            "seed": bootstrap_seed,
        },
        "comparisons": comparisons,
    }
