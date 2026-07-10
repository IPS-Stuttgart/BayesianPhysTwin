"""Locked full-cohort confirmation for matched residual baselines."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .phystwin_comparison import compare_phystwin_manifest
from .phystwin_confirmatory import (
    PhysTwinConfirmatoryProtocol,
    _canonical_json,
    _case_readout,
    _cohort_readout,
    _lock_protocol,
    _split_for_case,
)
from .phystwin_residual_baselines import (
    BASELINE_METHODS,
    fit_residual_dynamics_baselines,
)
from .phystwin_residual_dynamics import PhysTwinResidualDynamicsConfig


def _comparison_manifest(
    data_root: Path,
    output: Path,
    cases: tuple[str, ...],
    method: str,
) -> dict[str, object]:
    entries = []
    for case in cases:
        case_dir = data_root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        entries.append(
            {
                "name": case,
                "final_data": str((case_dir / "final_data.pkl").resolve()),
                "gt_track_3d": str((case_dir / "gt_track_3d.pkl").resolve()),
                "baseline_trajectory": str((case_dir / "inference.pkl").resolve()),
                "candidate_trajectory": str(
                    (output / "cases" / case / method / "trajectory.pkl").resolve()
                ),
                "start_frame": int(split["test"][0]),
                "end_frame": int(split["test"][1]),
            }
        )
    return {"schema_version": 1, "cases": entries}


def run_phystwin_baseline_confirmation(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    protocol: PhysTwinConfirmatoryProtocol | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run all matched comparators and bootstrap only the untouched cohort."""

    config = PhysTwinConfirmatoryProtocol() if protocol is None else protocol
    root = Path(data_root)
    source_manifest_path = root / "evaluation_subset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected = tuple(str(case) for case in source_manifest["selected_cases"])
    development = tuple(case for case in selected if case in config.development_cases)
    confirmation = tuple(case for case in selected if case not in config.development_cases)
    output = Path(output_dir)
    specification = {
        "methods": list(BASELINE_METHODS),
        "protocol": asdict(config),
        "data_manifest": {
            "path": str(source_manifest_path.resolve()),
            "selected_cases": list(selected),
        },
        "cohorts": {
            "development": list(development),
            "confirmation": list(confirmation),
        },
        "comparability": "same split, cap, latent ranks, validation gate, and actions as proposed method",
    }
    locked = _lock_protocol(output, specification)
    case_results: dict[str, dict[str, dict[str, object]]] = {
        method: {} for method in BASELINE_METHODS
    }
    for case in selected:
        case_dir = root / case
        fit_end, train_end, frame_count = _split_for_case(
            case_dir, config.fit_fraction
        )
        residual_config = PhysTwinResidualDynamicsConfig(
            fit_end_frame=fit_end,
            train_end_frame=train_end,
            rank_candidates=config.rank_candidates,
            persistence_candidates=config.persistence_candidates,
            ridge_candidates=config.ridge_candidates,
            projection_ridge=config.projection_ridge,
            interpolation_neighbors=config.interpolation_neighbors,
            maximum_state_multiplier=config.maximum_state_multiplier,
            maximum_residual_m=config.maximum_residual_m,
            minimum_validation_improvement=config.minimum_validation_improvement,
        )
        case_output = output / "cases" / case
        summary_path = case_output / "summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if _canonical_json(summary["config"]) != _canonical_json(
                asdict(residual_config)
            ):
                raise RuntimeError(f"cached case uses a different protocol: {case}")
        else:
            summary = fit_residual_dynamics_baselines(
                case_dir / "final_data.pkl",
                case_dir / "inference.pkl",
                case_dir / "gt_track_3d.pkl",
                case_output,
                config=residual_config,
            )
        for method in BASELINE_METHODS:
            case_results[method][case] = {
                "fit_end_frame": fit_end,
                "train_end_frame": train_end,
                "frame_count": frame_count,
                **_case_readout(summary["methods"][method]),
            }

    method_results: dict[str, object] = {}
    for method in BASELINE_METHODS:
        manifest = _comparison_manifest(root, output, confirmation, method)
        manifest_path = output / f"comparison_confirmation_{method}_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        comparison = compare_phystwin_manifest(
            manifest_path,
            output / f"comparison_confirmation_{method}.json",
            samples=config.bootstrap_samples,
            block_length=config.bootstrap_block_length,
            seed=config.bootstrap_seed,
        )
        method_results[method] = {
            "case_results": case_results[method],
            "confirmation": _cohort_readout(
                confirmation, case_results[method], comparison
            ),
        }

    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": method_results,
    }
    result_path = output / "baseline_confirmation_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
