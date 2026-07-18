"""Locked cohort evaluation of robust Bayesian PhysTwin residual anchoring."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .phystwin_bayesian_anchor import (
    BayesianResidualAnchorConfig,
    fit_bayesian_residual_anchor,
)
from .phystwin_comparison import compare_phystwin_manifest
from .phystwin_confirmation_lock import exclusively_owned_confirmation_output
from .phystwin_confirmatory import (
    DEVELOPMENT_CASES,
    _case_readout,
    _case_source_identity,
    _cohort_readout,
    _lock_protocol,
    _seal_case_cache,
    _split_for_case,
    _validate_cached_case,
)


@dataclass(frozen=True)
class BayesianAnchorConfirmationProtocol:
    """Frozen robust-filter and resampling choices for the full cohort."""

    fit_fraction: float = 0.75
    process_std_candidates_m: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.0025, 0.005)
    observation_std_candidates_m: tuple[float, ...] = (0.001, 0.0025, 0.005)
    initial_std_m: float = 0.01
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0
    interpolation_neighbors: int = 4
    maximum_residual_m: float = 0.01
    minimum_validation_improvement: float = 0.0
    bootstrap_samples: int = 10000
    bootstrap_block_length: int = 5
    bootstrap_seed: int = 20260710
    development_cases: tuple[str, ...] = DEVELOPMENT_CASES


def _comparison_manifest(
    data_root: Path,
    output: Path,
    cases: tuple[str, ...],
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
                    (output / "cases" / case / "trajectory.pkl").resolve()
                ),
                "start_frame": int(split["test"][0]),
                "end_frame": int(split["test"][1]),
            }
        )
    return {"schema_version": 1, "cases": entries}


def _fit_case(
    job: tuple[
        Path,
        Path,
        str,
        BayesianAnchorConfirmationProtocol,
        bool,
    ],
) -> tuple[str, dict[str, object]]:
    """Fit one independent case so full-cohort runs can use multiple processes."""

    root, output, case, config, force = job
    case_dir = root / case
    source_identity = _case_source_identity(case_dir)
    fit_end, train_end, frame_count = _split_for_case(
        case_dir, config.fit_fraction
    )
    anchor_config = BayesianResidualAnchorConfig(
        fit_end_frame=fit_end,
        train_end_frame=train_end,
        process_std_candidates_m=config.process_std_candidates_m,
        observation_std_candidates_m=config.observation_std_candidates_m,
        initial_std_m=config.initial_std_m,
        inlier_prior=config.inlier_prior,
        outlier_variance_multiplier=config.outlier_variance_multiplier,
        interpolation_neighbors=config.interpolation_neighbors,
        maximum_residual_m=config.maximum_residual_m,
        minimum_validation_improvement=config.minimum_validation_improvement,
    )
    case_output = output / "cases" / case
    summary_path = case_output / "summary.json"
    if summary_path.exists() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_cached_case(
            summary, asdict(anchor_config), case_dir, case_output, case
        )
    else:
        summary = fit_bayesian_residual_anchor(
            case_dir / "final_data.pkl",
            case_dir / "inference.pkl",
            case_dir / "gt_track_3d.pkl",
            case_output,
            config=anchor_config,
        )
        _seal_case_cache(
            summary,
            case_dir,
            case_output,
            expected_source_identity=source_identity,
        )
    return case, {
        "fit_end_frame": fit_end,
        "train_end_frame": train_end,
        "frame_count": frame_count,
        **_case_readout(summary),
        "posterior": summary["posterior"],
    }


@exclusively_owned_confirmation_output
def run_bayesian_anchor_confirmation(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    protocol: BayesianAnchorConfirmationProtocol | None = None,
    force: bool = False,
    workers: int = 1,
) -> dict[str, object]:
    """Fit each case causally and evaluate the nondevelopment cohort once."""

    if workers < 1:
        raise ValueError("workers must be positive")
    config = BayesianAnchorConfirmationProtocol() if protocol is None else protocol
    root = Path(data_root)
    source_manifest_path = root / "evaluation_subset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected = tuple(str(case) for case in source_manifest["selected_cases"])
    if len(selected) != len(set(selected)):
        raise ValueError("the data manifest contains duplicate cases")
    development = tuple(case for case in selected if case in config.development_cases)
    confirmation = tuple(case for case in selected if case not in config.development_cases)
    output = Path(output_dir)
    specification = {
        "method": "robust Bayesian random-walk endpoint anchoring",
        "protocol": asdict(config),
        "data_manifest": {
            "path": str(source_manifest_path.resolve()),
            "selected_cases": list(selected),
        },
        "cohorts": {
            "development": list(development),
            "confirmation": list(confirmation),
        },
        "status": "exploratory extension after the deterministic confirmation",
    }
    locked = _lock_protocol(output, specification)
    jobs = [(root, output, case, config, force) for case in selected]
    if workers == 1:
        fitted_cases = list(map(_fit_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fitted_cases = list(executor.map(_fit_case, jobs))
    case_results = dict(fitted_cases)

    manifest = _comparison_manifest(root, output, confirmation)
    manifest_path = output / "comparison_confirmation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison = compare_phystwin_manifest(
        manifest_path,
        output / "comparison_confirmation.json",
        samples=config.bootstrap_samples,
        block_length=config.bootstrap_block_length,
        seed=config.bootstrap_seed,
        cluster_by_phystwin_object=True,
    )
    accepted_uncertainty = np.array(
        [
            case_results[case]["posterior"][
                "median_final_future_predictive_std_m"
            ]
            for case in confirmation
            if case_results[case]["accepted_on_validation"]
        ],
        dtype=float,
    )
    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_results": case_results,
        "confirmation": _cohort_readout(
            confirmation, case_results, comparison
        ),
        "posterior_summary": {
            "accepted_case_count": len(accepted_uncertainty),
            "median_case_predictive_std_m": (
                float(np.median(accepted_uncertainty))
                if len(accepted_uncertainty)
                else None
            ),
            "geometric_mean_case_predictive_std_m": (
                float(math.exp(np.mean(np.log(accepted_uncertainty))))
                if len(accepted_uncertainty)
                else None
            ),
        },
    }
    result_path = output / "bayesian_anchor_confirmation_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
