"""Locked full-cohort confirmation of hierarchical mechanics plus discrepancy."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .phystwin_additional_bayesian_confirmation import (
    apply_fixed_bayesian_residual_anchor,
)
from .phystwin_additional_confirmation import _chamfer_by_frame
from .phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from .phystwin_confirmatory import (
    DEVELOPMENT_CASES,
    _canonical_json,
    _lock_protocol,
)
from .phystwin_controller_sensitivity import _compact_bootstrap
from .phystwin_headless_refit import (
    HeadlessPhysTwinRefitConfig,
    _git_commit,
    run_headless_phystwin_refit,
)
from .phystwin_joint_profile import combine_joint_profile_files
from .phystwin_residual_dynamics import (
    PhysTwinResidualDynamicsConfig,
    _load_pickle,
    _sha256,
    fit_action_conditioned_residual_dynamics,
)
from .phystwin_state_injection import _trajectory_error


MAIN_COMPONENT_PROTOCOL_ID = (
    "2a0507950ed40802756ad17e96de307a3640591b64ad585f6f5d9d235d84237d"
)
ADDITIONAL_COMPONENT_PROTOCOL_ID = (
    "0225d2bdac4a3dfebbef0dc57a14e36a3c5a8423ed9e3c96ab9d60ffc34e4770"
)
COMBINED_STAGES = (
    "profiles",
    "pool",
    "predictions",
    "discrepancy",
    "summary",
    "all",
)


@dataclass(frozen=True)
class CombinedFullCohortProtocol:
    """Frozen V2 hierarchy and V3 discrepancy settings."""

    main_fit_fraction: float = 0.75
    additional_profile_holdout_frames: int = 1
    profile_grid_count: int = 9
    profile_object_log_scale_half_width: float = 0.60
    profile_controller_log_scale_half_width: float = 1.50
    profile_object_prior_std: float = 0.15
    profile_controller_prior_std: float = 0.50
    object_deviation_stds: tuple[float, ...] = (0.03, 0.075, 0.15, 0.30, 0.60)
    object_deviation_prior_scale: float = 0.075
    profile_prediction_mass: float = 0.99
    residual_rank_candidates: tuple[int, ...] = (1, 2, 4, 8)
    residual_persistence_candidates: tuple[float, ...] = (
        0.0,
        0.5,
        0.8,
        0.95,
        1.0,
    )
    residual_ridge_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)
    residual_projection_ridge: float = 1e-6
    interpolation_neighbors: int = 4
    maximum_state_multiplier: float = 1.5
    maximum_residual_m: float = 0.01
    minimum_validation_improvement: float = 0.0
    deterministic_spring_forces: bool = True
    bootstrap_samples: int = 10000
    bootstrap_block_length: int = 5
    bootstrap_seed: int = 20260711


def balanced_profile_temperatures(
    fit_frame_counts: dict[str, int],
) -> dict[str, float]:
    """Normalize clustered likelihoods to the equal-case mean fit length."""

    if not fit_frame_counts or any(value < 1 for value in fit_frame_counts.values()):
        raise ValueError("fit frame counts must be positive")
    mean_count = float(np.mean(tuple(fit_frame_counts.values())))
    return {case: count / mean_count for case, count in fit_frame_counts.items()}


def combined_profile_fit_end(
    train_end_frame: int,
    *,
    cohort: str,
    main_fit_fraction: float,
    additional_holdout_frames: int,
) -> int:
    """Return the locked profile-likelihood endpoint for one cohort."""

    if train_end_frame < 5:
        raise ValueError("training interval is too short")
    if cohort == "main":
        fit_end = int(math.floor(main_fit_fraction * train_end_frame))
    elif cohort == "additional":
        fit_end = train_end_frame - additional_holdout_frames
    else:
        raise ValueError("cohort must be main or additional")
    if not 2 < fit_end < train_end_frame:
        raise ValueError("profile fit endpoint does not leave a valid split")
    return fit_end


def _manifest_name(cohort: str) -> str:
    if cohort == "main":
        return "evaluation_subset_manifest.json"
    if cohort == "additional":
        return "additional_evaluation_subset_manifest.json"
    raise ValueError("cohort must be main or additional")


def _cohort_cases(data_root: Path, cohort: str) -> tuple[str, ...]:
    manifest = json.loads(
        (data_root / _manifest_name(cohort)).read_text(encoding="utf-8")
    )
    available = tuple(str(case) for case in manifest["selected_cases"])
    if cohort == "main":
        return tuple(case for case in available if case not in DEVELOPMENT_CASES)
    return available


def _split(case_dir: Path) -> tuple[int, int]:
    split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
    train_start, train_end = (int(value) for value in split["train"])
    test_start, test_end = (int(value) for value in split["test"])
    if train_start != 0 or test_start != train_end:
        raise ValueError(f"unsupported split in {case_dir.name}")
    if test_end != int(split["frame_len"]):
        raise ValueError(f"test split does not end at frame_len in {case_dir.name}")
    return train_end, test_end


def _hard_cues(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path)


def _profile_config(
    protocol: CombinedFullCohortProtocol,
    *,
    fit_end_frame: int,
    train_end_frame: int,
) -> HeadlessPhysTwinRefitConfig:
    return HeadlessPhysTwinRefitConfig(
        variant="hard",
        train_end_frame=train_end_frame,
        fit_end_frame=fit_end_frame,
        epochs=0,
        optimize_collision=False,
        spring_parameterization="grouped",
        selection_metric="hard_valid_rmse",
        profile_grid_count=protocol.profile_grid_count,
        profile_object_log_scale_half_width=(
            protocol.profile_object_log_scale_half_width
        ),
        profile_controller_log_scale_half_width=(
            protocol.profile_controller_log_scale_half_width
        ),
        profile_object_prior_std=protocol.profile_object_prior_std,
        profile_controller_prior_std=protocol.profile_controller_prior_std,
        profile_prediction_mass=protocol.profile_prediction_mass,
        deterministic_spring_forces=protocol.deterministic_spring_forces,
    )


def _case_contracts(
    root: Path,
    cases: tuple[str, ...],
    cohort: str,
    protocol: CombinedFullCohortProtocol,
) -> dict[str, dict[str, int]]:
    result = {}
    for case in cases:
        train_end, frame_count = _split(root / case)
        fit_end = combined_profile_fit_end(
            train_end,
            cohort=cohort,
            main_fit_fraction=protocol.main_fit_fraction,
            additional_holdout_frames=protocol.additional_profile_holdout_frames,
        )
        result[case] = {
            "fit_end_frame": fit_end,
            "train_end_frame": train_end,
            "frame_count": frame_count,
            "fit_likelihood_frame_count": fit_end - 1,
        }
    return result


def _specification(
    official_repo: str | Path,
    root: Path,
    cohort: str,
    cases: tuple[str, ...],
    contracts: dict[str, dict[str, int]],
    protocol: CombinedFullCohortProtocol,
) -> dict[str, object]:
    manifest_path = root / _manifest_name(cohort)
    component_protocol = (
        MAIN_COMPONENT_PROTOCOL_ID
        if cohort == "main"
        else ADDITIONAL_COMPONENT_PROTOCOL_ID
    )
    return {
        "method": "hierarchical relative stiffness plus locked discrepancy",
        "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "official_repo": {
            "path": str(Path(official_repo).resolve()),
            "commit": _git_commit(official_repo),
        },
        "cohort": cohort,
        "cases": list(cases),
        "case_contracts": contracts,
        # Normalize tuples before the lock is written so a later JSON reload
        # compares equal to the in-memory specification.
        "protocol": json.loads(_canonical_json(asdict(protocol))),
        "hierarchy": {
            "scope": "one random-effects population across the full cohort",
            "relative_parameter": (
                "object-spring log scale relative to each released checkpoint"
            ),
            "trial_specific_parameter": "controller-spring log scale",
            "likelihood_balancing": "fit frame count divided by cohort mean",
            "matched_mechanical_update": (
                "released trajectory + deterministic hierarchical posterior "
                "trajectory - deterministic zero-scale trajectory"
            ),
            "matched_update_reason": (
                "remove fixed-order versus historical atomic replay as a control "
                "variate before adding discrepancy"
            ),
        },
        "discrepancy": {
            "main": "locked validation-gated action-conditioned residual",
            "additional": "locked label-free fixed Bayesian endpoint anchor",
            "component_protocol_id": component_protocol,
        },
        "data_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
        },
        "status": "post-hoc combined-method full-cohort confirmation",
    }


def _validate_execution_cases(
    requested: Iterable[str] | None,
    cases: tuple[str, ...],
) -> tuple[str, ...]:
    if requested is None:
        return cases
    selected = tuple(dict.fromkeys(str(case) for case in requested))
    missing = sorted(set(selected) - set(cases))
    if missing or not selected:
        raise ValueError("invalid execution cases: " + ", ".join(missing))
    return selected


def _run_profile(
    official_repo: str | Path,
    case_dir: Path,
    output: Path,
    config: HeadlessPhysTwinRefitConfig,
    *,
    profile_weights_path: Path | None,
) -> dict[str, object]:
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if _canonical_json(summary["config"]) != _canonical_json(asdict(config)):
            raise RuntimeError(f"cached profile uses a different config: {case_dir.name}")
        expected_weights = (
            None if profile_weights_path is None else _sha256(profile_weights_path)
        )
        cached_weights = summary["inputs"].get("profile_weights")
        cached_hash = None if cached_weights is None else cached_weights["sha256"]
        if cached_hash != expected_weights:
            raise RuntimeError(
                f"cached profile uses different hierarchy weights: {case_dir.name}"
            )
        return summary
    cues_path = output.parent / "hard_cues.npz"
    _hard_cues(cues_path)
    gt_path = case_dir / "gt_track_3d.pkl"
    return run_headless_phystwin_refit(
        official_repo=official_repo,
        final_data_path=case_dir / "final_data.pkl",
        optimal_params_path=case_dir / "optimal_params.pkl",
        checkpoint_path=case_dir / "checkpoint.pth",
        cues_path=cues_path,
        output_dir=output,
        released_trajectory_path=case_dir / "inference.pkl",
        gt_track_path=gt_path if gt_path.exists() else None,
        profile_weights_path=profile_weights_path,
        config=config,
    )


def run_combined_profile_stage(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str,
    protocol: CombinedFullCohortProtocol,
    execution_cases: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build raw deterministic 9x9 likelihood profiles for a case shard."""

    root = Path(data_root)
    cases = _cohort_cases(root, cohort)
    contracts = _case_contracts(root, cases, cohort, protocol)
    output = Path(output_dir)
    locked = _lock_protocol(
        output,
        _specification(official_repo, root, cohort, cases, contracts, protocol),
    )
    selected = _validate_execution_cases(execution_cases, cases)
    completed = []
    for case in selected:
        values = contracts[case]
        config = _profile_config(
            protocol,
            fit_end_frame=values["fit_end_frame"],
            train_end_frame=values["train_end_frame"],
        )
        _run_profile(
            official_repo,
            root / case,
            output / "cases" / case / "profile_raw",
            config,
            profile_weights_path=None,
        )
        completed.append(case)
    return {
        "stage": "profiles",
        "protocol_id": locked["protocol_id"],
        "completed_cases": completed,
    }


def run_combined_pool_stage(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str,
    protocol: CombinedFullCohortProtocol,
) -> dict[str, object]:
    """Fit the frozen random-effects hierarchy from all case profiles."""

    root = Path(data_root)
    cases = _cohort_cases(root, cohort)
    contracts = _case_contracts(root, cases, cohort, protocol)
    output = Path(output_dir)
    locked = _lock_protocol(
        output,
        _specification(official_repo, root, cohort, cases, contracts, protocol),
    )
    paths = {
        case: output / "cases" / case / "profile_raw" / "parameter_profile.npz"
        for case in cases
    }
    missing = [case for case, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing raw profiles: " + ", ".join(missing))
    temperatures = balanced_profile_temperatures(
        {
            case: contracts[case]["fit_likelihood_frame_count"]
            for case in cases
        }
    )
    summary = combine_joint_profile_files(
        paths,
        output / "hierarchical_pool",
        object_prior_std=protocol.profile_object_prior_std,
        controller_prior_std=protocol.profile_controller_prior_std,
        likelihood_temperatures=temperatures,
        object_deviation_stds=protocol.object_deviation_stds,
        object_deviation_prior_scale=protocol.object_deviation_prior_scale,
    )
    return {
        "stage": "pool",
        "protocol_id": locked["protocol_id"],
        "pooling": summary,
    }


def matched_hierarchical_trajectory(
    released: np.ndarray,
    zero_replay: np.ndarray,
    posterior: np.ndarray,
) -> np.ndarray:
    """Transport a paired deterministic parameter delta onto released PhysTwin."""

    released_values = np.asarray(released, dtype=float)
    zero_values = np.asarray(zero_replay, dtype=float)
    posterior_values = np.asarray(posterior, dtype=float)
    if not (
        released_values.shape == zero_values.shape == posterior_values.shape
        and released_values.ndim == 3
        and released_values.shape[2] == 3
    ):
        raise ValueError("released, zero, and posterior trajectories must match")
    if not (
        np.all(np.isfinite(released_values))
        and np.all(np.isfinite(zero_values))
        and np.all(np.isfinite(posterior_values))
    ):
        raise ValueError("matched hierarchy trajectories must be finite")
    return released_values + posterior_values - zero_values


def _export_posterior_mean(profile_dir: Path, released_path: Path) -> tuple[Path, Path]:
    source = profile_dir / "parameter_profile.npz"
    posterior_path = profile_dir / "posterior_mean_trajectory.pkl"
    matched_path = profile_dir / "matched_hierarchical_trajectory.pkl"
    with np.load(source) as archive:
        posterior = np.asarray(archive["posterior_mean_trajectory"], dtype=np.float32)
    zero_replay = np.asarray(
        _load_pickle(profile_dir / "baseline_trajectory.pkl"), dtype=np.float32
    )
    released = np.asarray(_load_pickle(released_path), dtype=np.float32)
    matched = matched_hierarchical_trajectory(
        released,
        zero_replay,
        posterior,
    ).astype(np.float32)
    with posterior_path.open("wb") as handle:
        pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with matched_path.open("wb") as handle:
        pickle.dump(matched, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return posterior_path, matched_path


def run_combined_prediction_stage(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str,
    protocol: CombinedFullCohortProtocol,
    execution_cases: Iterable[str] | None = None,
) -> dict[str, object]:
    """Generate posterior-mean trajectories using frozen hierarchical weights."""

    root = Path(data_root)
    cases = _cohort_cases(root, cohort)
    contracts = _case_contracts(root, cases, cohort, protocol)
    output = Path(output_dir)
    locked = _lock_protocol(
        output,
        _specification(official_repo, root, cohort, cases, contracts, protocol),
    )
    selected = _validate_execution_cases(execution_cases, cases)
    completed = []
    for case in selected:
        weights = output / "hierarchical_pool" / f"{case}.npz"
        if not weights.is_file():
            raise FileNotFoundError(f"missing hierarchical weights for {case}")
        values = contracts[case]
        config = _profile_config(
            protocol,
            fit_end_frame=values["fit_end_frame"],
            train_end_frame=values["train_end_frame"],
        )
        profile_dir = output / "cases" / case / "profile_hierarchical"
        _run_profile(
            official_repo,
            root / case,
            profile_dir,
            config,
            profile_weights_path=weights,
        )
        _export_posterior_mean(profile_dir, root / case / "inference.pkl")
        completed.append(case)
    return {
        "stage": "predictions",
        "protocol_id": locked["protocol_id"],
        "completed_cases": completed,
    }


def _residual_config(
    protocol: CombinedFullCohortProtocol,
    *,
    fit_end_frame: int,
    train_end_frame: int,
) -> PhysTwinResidualDynamicsConfig:
    return PhysTwinResidualDynamicsConfig(
        fit_end_frame=fit_end_frame,
        train_end_frame=train_end_frame,
        rank_candidates=protocol.residual_rank_candidates,
        persistence_candidates=protocol.residual_persistence_candidates,
        ridge_candidates=protocol.residual_ridge_candidates,
        projection_ridge=protocol.residual_projection_ridge,
        interpolation_neighbors=protocol.interpolation_neighbors,
        maximum_state_multiplier=protocol.maximum_state_multiplier,
        maximum_residual_m=protocol.maximum_residual_m,
        minimum_validation_improvement=protocol.minimum_validation_improvement,
    )


def run_combined_discrepancy_stage(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str,
    protocol: CombinedFullCohortProtocol,
    execution_cases: Iterable[str] | None = None,
) -> dict[str, object]:
    """Apply the locked residual or fixed anchor to hierarchical trajectories."""

    root = Path(data_root)
    cases = _cohort_cases(root, cohort)
    contracts = _case_contracts(root, cases, cohort, protocol)
    output = Path(output_dir)
    locked = _lock_protocol(
        output,
        _specification(official_repo, root, cohort, cases, contracts, protocol),
    )
    selected = _validate_execution_cases(execution_cases, cases)
    completed = []
    for case in selected:
        case_dir = root / case
        baseline = (
            output
            / "cases"
            / case
            / "profile_hierarchical"
            / "matched_hierarchical_trajectory.pkl"
        )
        if not baseline.is_file():
            raise FileNotFoundError(f"missing hierarchical trajectory for {case}")
        case_output = output / "cases" / case / "combined"
        summary_path = case_output / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["inputs"]["baseline_trajectory"]["sha256"] != _sha256(
                baseline
            ):
                raise RuntimeError(f"cached discrepancy uses another baseline: {case}")
        elif cohort == "main":
            summary = fit_action_conditioned_residual_dynamics(
                case_dir / "final_data.pkl",
                baseline,
                case_dir / "gt_track_3d.pkl",
                case_output,
                config=_residual_config(
                    protocol,
                    fit_end_frame=contracts[case]["fit_end_frame"],
                    train_end_frame=contracts[case]["train_end_frame"],
                ),
            )
        else:
            summary = apply_fixed_bayesian_residual_anchor(
                case_dir / "final_data.pkl",
                baseline,
                case_output,
                train_end_frame=contracts[case]["train_end_frame"],
                maximum_residual_m=protocol.maximum_residual_m,
                interpolation_neighbors=protocol.interpolation_neighbors,
            )
        completed.append(
            {
                "case": case,
                "trajectory": summary["outputs"]["trajectory"],
            }
        )
    return {
        "stage": "discrepancy",
        "protocol_id": locked["protocol_id"],
        "completed_cases": completed,
    }


def _validate_component_run(component_run: Path, cohort: str) -> str:
    locked = json.loads(
        (component_run / "locked_protocol.json").read_text(encoding="utf-8")
    )
    expected = (
        MAIN_COMPONENT_PROTOCOL_ID
        if cohort == "main"
        else ADDITIONAL_COMPONENT_PROTOCOL_ID
    )
    if locked["protocol_id"] != expected:
        raise RuntimeError("component-only run has an unexpected locked protocol")
    return expected


def _case_metrics(
    case_dir: Path,
    trajectory_path: Path,
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, np.ndarray]:
    data = _load_pickle(case_dir / "final_data.pkl")
    trajectory = np.asarray(_load_pickle(trajectory_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    num_surface_points = len(observed[0]) + len(np.asarray(data["surface_points"]))
    track_path = case_dir / "gt_track_3d.pkl"
    if track_path.exists():
        return official_metrics_by_frame(
            trajectory,
            observed,
            visible,
            np.asarray(_load_pickle(track_path), dtype=float),
            num_surface_points=num_surface_points,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    return {
        "chamfer_distance_m": _chamfer_by_frame(
            trajectory,
            observed,
            visible,
            num_surface_points=num_surface_points,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    }


def _percent_change(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, float]:
    return {
        metric: 100.0
        * (float(np.mean(candidate[metric])) / float(np.mean(values)) - 1.0)
        for metric, values in baseline.items()
    }


def run_combined_summary_stage(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    component_run: str | Path,
    *,
    cohort: str,
    protocol: CombinedFullCohortProtocol,
) -> dict[str, object]:
    """Evaluate all components and the combined method under paired bootstraps."""

    root = Path(data_root)
    cases = _cohort_cases(root, cohort)
    contracts = _case_contracts(root, cases, cohort, protocol)
    output = Path(output_dir)
    component = Path(component_run)
    component_protocol_id = _validate_component_run(component, cohort)
    locked = _lock_protocol(
        output,
        _specification(official_repo, root, cohort, cases, contracts, protocol),
    )
    clusters = {case: phystwin_physical_object_cluster(case) for case in cases}
    case_results: dict[str, object] = {}
    comparison_names = {
        "zero_replay_vs_released": ("released", "zero_replay"),
        "hierarchical_raw_vs_released": ("released", "hierarchical_raw"),
        "hierarchical_raw_vs_zero_replay": ("zero_replay", "hierarchical_raw"),
        "hierarchical_vs_released": ("released", "hierarchical"),
        "component_vs_released": ("released", "component"),
        "combined_vs_released": ("released", "combined"),
        "combined_vs_component": ("component", "combined"),
        "combined_vs_hierarchical": ("hierarchical", "combined"),
    }
    paired = {name: {} for name in comparison_names}
    for case in cases:
        case_dir = root / case
        values = contracts[case]
        profile_dir = output / "cases" / case / "profile_hierarchical"
        paths = {
            "released": case_dir / "inference.pkl",
            "zero_replay": profile_dir / "baseline_trajectory.pkl",
            "hierarchical_raw": profile_dir / "posterior_mean_trajectory.pkl",
            "hierarchical": profile_dir / "matched_hierarchical_trajectory.pkl",
            "component": component / "cases" / case / "trajectory.pkl",
            "combined": output / "cases" / case / "combined" / "trajectory.pkl",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{case} is missing trajectories: {', '.join(missing)}")
        metrics = {
            name: _case_metrics(
                case_dir,
                path,
                start_frame=values["train_end_frame"],
                end_frame=values["frame_count"],
            )
            for name, path in paths.items()
        }
        for comparison, (baseline_name, candidate_name) in comparison_names.items():
            paired[comparison][case] = (
                metrics[baseline_name],
                metrics[candidate_name],
            )
        method_readout = {
            name: {
                "mean_m": {
                    metric: float(np.mean(metric_values))
                    for metric, metric_values in method_metrics.items()
                },
                "percent_change_vs_released": (
                    {metric: 0.0 for metric in metrics["released"]}
                    if name == "released"
                    else _percent_change(metrics["released"], method_metrics)
                ),
            }
            for name, method_metrics in metrics.items()
        }
        zero_trajectory = np.asarray(_load_pickle(paths["zero_replay"]), dtype=float)
        released_trajectory = np.asarray(_load_pickle(paths["released"]), dtype=float)
        future_slice = slice(values["train_end_frame"], values["frame_count"])
        profile_summary = json.loads(
            (profile_dir / "summary.json").read_text(encoding="utf-8")
        )
        discrepancy_summary = json.loads(
            (output / "cases" / case / "combined" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        case_results[case] = {
            "physical_object": clusters[case],
            "fit_end_frame": values["fit_end_frame"],
            "train_end_frame": values["train_end_frame"],
            "methods": method_readout,
            "zero_replay_future_trajectory_error": _trajectory_error(
                released_trajectory[future_slice], zero_trajectory[future_slice]
            ),
            "hierarchical_profile": {
                "prediction_particle_count": profile_summary["parameter_profile"][
                    "prediction_particle_count"
                ],
                "prediction_retained_mass": profile_summary["parameter_profile"][
                    "prediction_retained_mass"
                ],
                "object_log_scale_mean": profile_summary["parameter_profile"][
                    "prediction_object_log_scale_mean"
                ],
                "controller_log_scale_mean": profile_summary[
                    "parameter_profile"
                ]["prediction_controller_log_scale_mean"],
            },
            "discrepancy": (
                {
                    "accepted_on_validation": discrepancy_summary["selection"][
                        "accepted"
                    ],
                    "validation_relative_improvement": discrepancy_summary[
                        "selection"
                    ]["relative_improvement"],
                    "selected_candidate": discrepancy_summary["selection"][
                        "selected_candidate"
                    ],
                    "correction": discrepancy_summary["correction"],
                }
                if cohort == "main"
                else {
                    "selection": "none",
                    "correction": discrepancy_summary["correction"],
                    "posterior": discrepancy_summary["posterior"],
                }
            ),
        }

    comparisons = {
        name: _compact_bootstrap(
            paired_block_bootstrap(
                values,
                samples=protocol.bootstrap_samples,
                block_length=protocol.bootstrap_block_length,
                seed=protocol.bootstrap_seed,
                clusters=clusters,
            )
        )
        for name, values in paired.items()
    }
    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "component_protocol_id": component_protocol_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort,
        "case_count": len(cases),
        "physical_object_count": len(set(clusters.values())),
        "case_results": case_results,
        "comparisons": comparisons,
        "pooling": json.loads(
            (output / "hierarchical_pool" / "summary.json").read_text(
                encoding="utf-8"
            )
        ),
        "interpretation_boundary": {
            "primary_incremental_test": "combined_vs_component",
            "replay_control": (
                "zero-scale fixed-order replay is reported separately from released"
            ),
            "matched_hierarchy": (
                "primary hierarchy is released plus paired posterior-minus-zero "
                "deterministic delta; raw posterior is diagnostic"
            ),
            "main_selection": (
                "locked validation CD and manual-track selection; future held out"
                if cohort == "main"
                else "not applicable"
            ),
            "additional_selection": (
                "fixed hierarchy and Bayesian anchor; no labels or model selection"
                if cohort == "additional"
                else "not applicable"
            ),
            "rendering_metrics": "not recomputed",
        },
    }
    result_path = output / "combined_full_cohort_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(result_path.resolve())
    return result


def run_combined_confirmation_stage(
    official_repo: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str,
    stage: str,
    component_run: str | Path | None = None,
    protocol: CombinedFullCohortProtocol | None = None,
    execution_cases: Iterable[str] | None = None,
) -> dict[str, object]:
    """Dispatch one reproducible stage of the combined confirmation."""

    if stage not in COMBINED_STAGES:
        raise ValueError("unsupported combined confirmation stage")
    config = CombinedFullCohortProtocol() if protocol is None else protocol
    results = []
    stages = (
        ("profiles", "pool", "predictions", "discrepancy", "summary")
        if stage == "all"
        else (stage,)
    )
    for current in stages:
        if current == "profiles":
            result = run_combined_profile_stage(
                official_repo,
                data_root,
                output_dir,
                cohort=cohort,
                protocol=config,
                execution_cases=execution_cases,
            )
        elif current == "pool":
            result = run_combined_pool_stage(
                official_repo,
                data_root,
                output_dir,
                cohort=cohort,
                protocol=config,
            )
        elif current == "predictions":
            result = run_combined_prediction_stage(
                official_repo,
                data_root,
                output_dir,
                cohort=cohort,
                protocol=config,
                execution_cases=execution_cases,
            )
        elif current == "discrepancy":
            result = run_combined_discrepancy_stage(
                official_repo,
                data_root,
                output_dir,
                cohort=cohort,
                protocol=config,
                execution_cases=execution_cases,
            )
        else:
            if component_run is None:
                raise ValueError("summary stage requires component_run")
            result = run_combined_summary_stage(
                official_repo,
                data_root,
                output_dir,
                component_run,
                cohort=cohort,
                protocol=config,
            )
        results.append(result)
    return {
        "schema_version": 1,
        "cohort": cohort,
        "requested_stage": stage,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
