"""Causal Bayesian overlays for external PhysTwin-compatible trajectories."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import pickle
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_baseline_confirmation import PhysTwinConfirmatoryProtocol
from .phystwin_bayesian_anchor import (
    BayesianResidualAnchorConfig,
    fit_bayesian_residual_anchor,
)
from .phystwin_bayesian_confirmation import BayesianAnchorConfirmationProtocol
from .phystwin_confirmation_lock import exclusively_owned_confirmation_output
from .phystwin_confirmatory import _lock_protocol, _split_for_case
from .phystwin_confirmatory import DEVELOPMENT_CASES
from .phystwin_residual_baselines import fit_residual_dynamics_baselines
from .phystwin_residual_dynamics import PhysTwinResidualDynamicsConfig
from .phystwin_sota_comparison import (
    PHYSTWIN_TABLE1_CASES,
    aggregate_phystwin_sota_comparison,
)


EXTERNAL_BACKBONE_SCHEMA_VERSION = 1
EXTERNAL_VERTEX_CONTRACT = "phystwin-observed-prefix-then-surface-v1"
EXTERNAL_COORDINATE_FRAME = "phystwin-world-metres-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _resolve_manifest_path(manifest_path: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validate_file_identity(
    manifest_path: Path,
    identity: object,
    *,
    label: str,
) -> Path:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} must be a file identity")
    path = _resolve_manifest_path(manifest_path, identity.get("path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(identity.get("sha256", ""))
    if not expected_hash or _sha256_file(path) != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def _validate_component_manifests(
    manifest_path: Path,
    backbone: dict[str, object],
    case_entries: dict[str, dict[str, object]],
) -> None:
    """Revalidate merged per-case provenance instead of trusting copied fields."""

    raw_components = backbone.get("component_manifests")
    if raw_components is None:
        return
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("component_manifests must be a nonempty list")

    shared_fields = (
        "name",
        "source_repository",
        "source_commit",
        "future_observations_used",
        "coordinate_frame",
        "vertex_contract",
        "proxy_contract",
        "claim_boundary",
    )
    component_case_names: set[str] = set()
    for index, raw_component in enumerate(raw_components):
        label = f"component_manifests[{index}]"
        if not isinstance(raw_component, dict):
            raise ValueError(f"{label} must be an object")
        component_path = _validate_file_identity(
            manifest_path,
            raw_component,
            label=label,
        )
        component = json.loads(component_path.read_text(encoding="utf-8"))
        if component.get("schema_version") != EXTERNAL_BACKBONE_SCHEMA_VERSION:
            raise ValueError(f"{label} has an unsupported schema")
        component_backbone = component.get("backbone")
        component_cases = component.get("cases")
        if not isinstance(component_backbone, dict) or not isinstance(
            component_cases, list
        ):
            raise ValueError(f"{label} is malformed")
        for field in shared_fields:
            if component_backbone.get(field) != backbone.get(field):
                raise ValueError(f"{label} disagrees on backbone field {field}")

        declared_names = raw_component.get("cases")
        actual_names = [
            str(entry.get("name", ""))
            for entry in component_cases
            if isinstance(entry, dict)
        ]
        if (
            not isinstance(declared_names, list)
            or len(actual_names) != len(component_cases)
            or actual_names != [str(name) for name in declared_names]
        ):
            raise ValueError(f"{label} case declaration mismatch")
        if any(not name or name in component_case_names for name in actual_names):
            raise ValueError("component case names must be nonempty and unique")
        component_case_names.update(actual_names)

        for identity_name in ("checkpoint", "causal_training_audit"):
            declared_identity = raw_component.get(identity_name)
            if declared_identity != component_backbone.get(identity_name):
                raise ValueError(f"{label} {identity_name} identity mismatch")
            _validate_file_identity(
                component_path,
                declared_identity,
                label=f"{label}.{identity_name}",
            )

        for component_entry in component_cases:
            assert isinstance(component_entry, dict)
            name = str(component_entry["name"])
            merged_entry = case_entries.get(name)
            if merged_entry is None:
                raise ValueError(f"{label} contains undeclared case {name}")
            component_trajectory = _resolve_manifest_path(
                component_path, component_entry.get("trajectory")
            )
            merged_trajectory = _resolve_manifest_path(
                manifest_path, merged_entry.get("trajectory")
            )
            if component_trajectory != merged_trajectory:
                raise ValueError(f"{label} trajectory path changed for {name}")
            for field in (
                "sha256",
                "evidence_end_frame_exclusive",
                "initial_alignment_tolerance_m",
            ):
                if component_entry.get(field) != merged_entry.get(field):
                    raise ValueError(f"{label} case field {field} changed for {name}")

    if component_case_names != set(case_entries):
        raise ValueError("component manifests do not cover every merged case")


def validate_external_backbone_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    *,
    require_full_cohort: bool = True,
) -> dict[str, object]:
    """Validate provenance, causality, geometry, and bytes before fitting.

    The Bayesian overlay assumes that the first observed-point count vertices
    retain PhysTwin's material identity and ordering. Methods with a different
    topology need an explicit association adapter before entering this path.
    """

    root = Path(data_root).resolve()
    source = Path(manifest_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXTERNAL_BACKBONE_SCHEMA_VERSION:
        raise ValueError("unsupported external-backbone manifest schema")
    backbone = payload.get("backbone")
    if not isinstance(backbone, dict):
        raise ValueError("manifest backbone must be an object")
    for field in ("name", "source_repository", "source_commit"):
        if not str(backbone.get(field, "")).strip():
            raise ValueError(f"manifest backbone is missing {field}")
    if backbone.get("future_observations_used") is not False:
        raise ValueError("external backbone must explicitly forbid future observations")
    if backbone.get("coordinate_frame") != EXTERNAL_COORDINATE_FRAME:
        raise ValueError("external backbone uses an unsupported coordinate frame")
    if backbone.get("vertex_contract") != EXTERNAL_VERTEX_CONTRACT:
        raise ValueError("external backbone uses an unsupported vertex contract")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest cases must be a nonempty list")
    case_entries: dict[str, dict[str, object]] = {}
    for raw_entry in raw_cases:
        if not isinstance(raw_entry, dict):
            raise ValueError("every manifest case must be an object")
        name = str(raw_entry.get("name", ""))
        if not name or name in case_entries:
            raise ValueError("manifest case names must be nonempty and unique")
        case_entries[name] = raw_entry

    _validate_component_manifests(source, backbone, case_entries)

    expected_cases = PHYSTWIN_TABLE1_CASES if require_full_cohort else tuple(case_entries)
    if tuple(case_entries) != expected_cases:
        raise ValueError(
            "external trajectories do not match the required ordered cohort; "
            f"expected={list(expected_cases)}, actual={list(case_entries)}"
        )

    validated_cases: list[dict[str, object]] = []
    for name, entry in case_entries.items():
        case_root = root / name
        split = json.loads((case_root / "split.json").read_text(encoding="utf-8"))
        train_end = int(split["train"][1])
        test_end = int(split["test"][1])
        evidence_end = int(entry.get("evidence_end_frame_exclusive", -1))
        if not 1 <= evidence_end <= train_end:
            raise ValueError(
                f"{name}: evidence end must be inside the released training interval"
            )
        trajectory_path = _resolve_manifest_path(source, entry.get("trajectory"))
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)
        expected_hash = str(entry.get("sha256", ""))
        actual_hash = _sha256_file(trajectory_path)
        if expected_hash != actual_hash:
            raise ValueError(f"{name}: trajectory SHA-256 mismatch")

        trajectory = np.asarray(_load_pickle(trajectory_path), dtype=float)
        if trajectory.ndim != 3 or trajectory.shape[2] != 3:
            raise ValueError(f"{name}: trajectory must have shape (T, N, 3)")
        if trajectory.shape[0] < test_end:
            raise ValueError(f"{name}: trajectory does not cover the held-out interval")
        if not np.isfinite(trajectory).all():
            raise ValueError(f"{name}: trajectory contains non-finite values")

        final_data = _load_pickle(case_root / "final_data.pkl")
        observed = np.asarray(final_data["object_points"], dtype=float)
        if trajectory.shape[1] < observed.shape[1]:
            raise ValueError(f"{name}: trajectory omits observed PhysTwin vertices")
        initial_error = trajectory[0, : observed.shape[1]] - observed[0]
        initial_norm = np.linalg.norm(initial_error, axis=1)
        tolerance = float(entry.get("initial_alignment_tolerance_m", 1e-6))
        if tolerance <= 0.0:
            raise ValueError(f"{name}: initial alignment tolerance must be positive")
        maximum_initial_error = float(np.max(initial_norm, initial=0.0))
        if maximum_initial_error > tolerance:
            raise ValueError(
                f"{name}: material-vertex alignment exceeds {tolerance:g} m"
            )
        validated_cases.append(
            {
                "name": name,
                "trajectory": str(trajectory_path),
                "sha256": actual_hash,
                "evidence_end_frame_exclusive": evidence_end,
                "train_end_frame_exclusive": train_end,
                "test_end_frame_exclusive": test_end,
                "frame_count": int(trajectory.shape[0]),
                "vertex_count": int(trajectory.shape[1]),
                "maximum_initial_alignment_error_m": maximum_initial_error,
                "initial_alignment_tolerance_m": tolerance,
            }
        )

    return {
        "schema_version": EXTERNAL_BACKBONE_SCHEMA_VERSION,
        "manifest": {"path": str(source), "sha256": _sha256_file(source)},
        "backbone": backbone,
        "cases": validated_cases,
    }


def _load_cached_summary(
    path: Path,
    *,
    expected_config: dict[str, object],
    expected_baseline_sha256: str,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    recorded_config = json.dumps(
        summary.get("config"), sort_keys=True, separators=(",", ":")
    )
    normalized_expected = json.dumps(
        expected_config, sort_keys=True, separators=(",", ":")
    )
    if recorded_config != normalized_expected:
        raise RuntimeError(f"cached overlay uses a different protocol: {path}")
    recorded = summary.get("inputs", {}).get("baseline_trajectory", {})
    if recorded.get("sha256") != expected_baseline_sha256:
        raise RuntimeError(f"cached overlay uses a different backbone: {path}")
    return summary


def _stage_trajectory(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != expected_hash:
            raise RuntimeError(f"staged trajectory differs from its manifest: {destination}")
        return
    shutil.copy2(source, destination)
    if _sha256_file(destination) != expected_hash:
        raise RuntimeError(f"trajectory changed while being staged: {source}")


def _fit_external_case(
    job: tuple[
        Path,
        Path,
        dict[str, object],
        BayesianAnchorConfirmationProtocol,
        PhysTwinConfirmatoryProtocol,
        bool,
    ],
) -> tuple[str, dict[str, object]]:
    root, output, entry, bayesian_protocol, residual_protocol, force = job
    case = str(entry["name"])
    case_root = root / case
    case_output = output / "cases" / case
    staged_baseline = case_output / "backbone" / "trajectory.pkl"
    baseline_hash = str(entry["sha256"])
    _stage_trajectory(Path(str(entry["trajectory"])), staged_baseline, baseline_hash)
    fit_end, train_end, frame_count = _split_for_case(
        case_root, bayesian_protocol.fit_fraction
    )
    if residual_protocol.fit_fraction != bayesian_protocol.fit_fraction:
        raise ValueError("Bayesian and residual overlays must use the same split")

    bayesian_config = BayesianResidualAnchorConfig(
        fit_end_frame=fit_end,
        train_end_frame=train_end,
        process_std_candidates_m=bayesian_protocol.process_std_candidates_m,
        observation_std_candidates_m=bayesian_protocol.observation_std_candidates_m,
        initial_std_m=bayesian_protocol.initial_std_m,
        inlier_prior=bayesian_protocol.inlier_prior,
        outlier_variance_multiplier=bayesian_protocol.outlier_variance_multiplier,
        interpolation_neighbors=bayesian_protocol.interpolation_neighbors,
        maximum_residual_m=bayesian_protocol.maximum_residual_m,
        minimum_validation_improvement=(
            bayesian_protocol.minimum_validation_improvement
        ),
    )
    bayesian_output = case_output / "bayesian_anchor"
    bayesian_summary = None if force else _load_cached_summary(
        bayesian_output / "summary.json",
        expected_config=asdict(bayesian_config),
        expected_baseline_sha256=baseline_hash,
    )
    if bayesian_summary is None:
        bayesian_summary = fit_bayesian_residual_anchor(
            case_root / "final_data.pkl",
            staged_baseline,
            case_root / "gt_track_3d.pkl",
            bayesian_output,
            config=bayesian_config,
        )

    residual_config = PhysTwinResidualDynamicsConfig(
        fit_end_frame=fit_end,
        train_end_frame=train_end,
        rank_candidates=residual_protocol.rank_candidates,
        persistence_candidates=residual_protocol.persistence_candidates,
        ridge_candidates=residual_protocol.ridge_candidates,
        projection_ridge=residual_protocol.projection_ridge,
        interpolation_neighbors=residual_protocol.interpolation_neighbors,
        maximum_state_multiplier=residual_protocol.maximum_state_multiplier,
        maximum_residual_m=residual_protocol.maximum_residual_m,
        minimum_validation_improvement=(
            residual_protocol.minimum_validation_improvement
        ),
    )
    residual_output = case_output / "residual_baselines"
    residual_summary = None if force else _load_cached_summary(
        residual_output / "summary.json",
        expected_config=asdict(residual_config),
        expected_baseline_sha256=baseline_hash,
    )
    if residual_summary is None:
        residual_summary = fit_residual_dynamics_baselines(
            case_root / "final_data.pkl",
            staged_baseline,
            case_root / "gt_track_3d.pkl",
            residual_output,
            config=residual_config,
        )

    bayesian_selection = bayesian_summary["selection"]
    last_selection = residual_summary["methods"]["last_residual"]["selection"]
    choices = [(1.0, 0, "backbone", staged_baseline)]
    if bayesian_selection["accepted"]:
        choices.append(
            (
                float(bayesian_selection["selected_candidate"]["selection_score"]),
                1,
                "bayesian_anchor",
                Path(str(bayesian_summary["outputs"]["trajectory"])),
            )
        )
    if last_selection["accepted"]:
        choices.append(
            (
                float(last_selection["selected_candidate"]["selection_score"]),
                2,
                "last_residual",
                Path(
                    str(
                        residual_summary["methods"]["last_residual"]["outputs"][
                            "trajectory"
                        ]
                    )
                ),
            )
        )
    selected_score, _, selected_method, selected_source = min(choices)
    selected_path = case_output / "validation_selected" / "trajectory.pkl"
    selected_hash = _sha256_file(selected_source)
    if selected_path.exists() and force:
        selected_path.unlink()
    _stage_trajectory(selected_source, selected_path, selected_hash)

    case_summary = {
        "name": case,
        "fit_end_frame_exclusive": fit_end,
        "train_end_frame_exclusive": train_end,
        "frame_count": frame_count,
        "backbone_sha256": baseline_hash,
        "selector": {
            "selected_method": selected_method,
            "selected_validation_score_relative_to_backbone": selected_score,
            "candidates": {
                "backbone": 1.0,
                "bayesian_anchor": float(
                    bayesian_selection["selected_candidate"]["selection_score"]
                ),
                "last_residual": float(
                    last_selection["selected_candidate"]["selection_score"]
                ),
            },
        },
        "test": {
            "bayesian_anchor": bayesian_summary["test"],
            "last_residual": residual_summary["methods"]["last_residual"]["test"],
        },
        "outputs": {
            "backbone": str(staged_baseline.resolve()),
            "bayesian_anchor": str(
                Path(str(bayesian_summary["outputs"]["trajectory"])).resolve()
            ),
            "last_residual": str(
                Path(
                    str(
                        residual_summary["methods"]["last_residual"]["outputs"][
                            "trajectory"
                        ]
                    )
                ).resolve()
            ),
            "validation_selected": str(selected_path.resolve()),
        },
    }
    case_summary_path = case_output / "external_overlay_summary.json"
    case_summary_path.write_text(
        json.dumps(case_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return case, case_summary


def _development_comparison(
    case_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    case_names = tuple(case_results)
    expected_order = tuple(case for case in DEVELOPMENT_CASES if case in case_names)
    if case_names != expected_order:
        raise ValueError(
            "development smoke cases must be an ordered subset of "
            f"{list(DEVELOPMENT_CASES)}"
        )

    method_names = (
        "external_backbone",
        "external_plus_bayesian",
        "external_plus_last_residual",
        "external_validation_selected",
    )
    per_method: dict[str, dict[str, object]] = {}
    for method in method_names:
        per_case: dict[str, dict[str, float]] = {}
        for case, result in case_results.items():
            tests = result["test"]
            if method == "external_backbone":
                metrics = tests["bayesian_anchor"]["baseline_official_evaluation"]
            elif method == "external_plus_bayesian":
                metrics = tests["bayesian_anchor"]["corrected_official_evaluation"]
            elif method == "external_plus_last_residual":
                metrics = tests["last_residual"]["corrected_official_evaluation"]
            else:
                selected = result["selector"]["selected_method"]
                if selected == "backbone":
                    metrics = tests["bayesian_anchor"][
                        "baseline_official_evaluation"
                    ]
                elif selected == "bayesian_anchor":
                    metrics = tests["bayesian_anchor"][
                        "corrected_official_evaluation"
                    ]
                elif selected == "last_residual":
                    metrics = tests["last_residual"][
                        "corrected_official_evaluation"
                    ]
                else:
                    raise ValueError(f"unsupported selected method: {selected}")
            per_case[case] = {
                "chamfer_distance_m": float(metrics["chamfer_distance_m"]),
                "track_error_m": float(metrics["track_error_m"]),
            }
        per_method[method] = {
            "equal_case_mean": {
                metric: float(
                    np.mean([values[metric] for values in per_case.values()])
                )
                for metric in ("chamfer_distance_m", "track_error_m")
            },
            "per_case": per_case,
        }
    return {
        "status": "development-only integration smoke; not cohort evidence",
        "case_count": len(case_names),
        "cases": list(case_names),
        "methods": per_method,
    }


@exclusively_owned_confirmation_output
def run_external_backbone_overlay(
    data_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    force: bool = False,
    workers: int = 1,
    development_smoke: bool = False,
) -> dict[str, object]:
    """Run frozen corrections over a full or explicit development backbone."""

    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    validated = validate_external_backbone_manifest(
        root,
        manifest_path,
        require_full_cohort=not development_smoke,
    )
    if development_smoke:
        names = tuple(str(entry["name"]) for entry in validated["cases"])
        expected_order = tuple(case for case in DEVELOPMENT_CASES if case in names)
        if names != expected_order:
            raise ValueError(
                "development smoke manifest must be an ordered subset of "
                f"{list(DEVELOPMENT_CASES)}"
            )
    bayesian_protocol = BayesianAnchorConfirmationProtocol()
    residual_protocol = PhysTwinConfirmatoryProtocol()
    specification = {
        "method": "external-backbone causal Bayesian overlay",
        "backbone_manifest": validated["manifest"],
        "backbone": validated["backbone"],
        "bayesian_protocol": asdict(bayesian_protocol),
        "residual_protocol": asdict(residual_protocol),
        "selector": (
            "minimum validation score among exact backbone, accepted Bayesian "
            "anchor, and accepted last residual; ties prefer the backbone"
        ),
        "status": (
            "development-only integration smoke; not cohort evidence"
            if development_smoke
            else "exploratory on the previously examined PhysTwin cohort"
        ),
    }
    locked = _lock_protocol(output, specification)
    jobs = [
        (
            root,
            output,
            entry,
            bayesian_protocol,
            residual_protocol,
            force,
        )
        for entry in validated["cases"]
    ]
    if workers == 1:
        fitted = list(map(_fit_external_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fitted = list(executor.map(_fit_external_case, jobs))
    case_results = dict(fitted)

    if development_smoke:
        comparison = _development_comparison(case_results)
    else:
        trajectory_root = output / "cases" / "{case}"
        comparison = aggregate_phystwin_sota_comparison(
            root,
            {
                "external_backbone": str(
                    trajectory_root / "backbone" / "trajectory.pkl"
                ),
                "external_plus_bayesian": str(
                    trajectory_root / "bayesian_anchor" / "trajectory.pkl"
                ),
                "external_plus_last_residual": str(
                    trajectory_root
                    / "residual_baselines"
                    / "last_residual"
                    / "trajectory.pkl"
                ),
                "external_validation_selected": str(
                    trajectory_root / "validation_selected" / "trajectory.pkl"
                ),
            },
            output / "sota_comparison.json",
        )
    selection_counts: dict[str, int] = {}
    for result in case_results.values():
        method = str(result["selector"]["selected_method"])
        selection_counts[method] = selection_counts.get(method, 0) + 1
    summary = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": validated["backbone"],
        "manifest": validated["manifest"],
        "case_results": case_results,
        "selection_counts": selection_counts,
        "comparison": comparison,
    }
    summary_name = (
        "external_backbone_development_smoke_summary.json"
        if development_smoke
        else "external_backbone_overlay_summary.json"
    )
    summary_path = output / summary_name
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary
