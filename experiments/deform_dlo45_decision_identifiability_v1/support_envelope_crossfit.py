"""Cross-fitted trajectory-conformal regret envelopes for DEFORM DLO4/DLO5.

Version 1 used eight untouched source-test trajectories per DLO, which makes a
finite 90% split-conformal radius impossible. This follow-up creates two
outcome-independent complementary source routes. Each route trains and tunes
only on 28 trajectories and calibrates on the other 28. A held trajectory is
assigned to exactly one route by a metadata-only hash before its payload is
read. Thus each deployed route remains an ordinary split-conformal procedure;
windows are never promoted to independent calibration units.

The evaluation is retrospective because the official DLO4/DLO5 evaluation
outcomes were opened by earlier studies. No held outcome is used for the
cross-fit split, route, hyperparameter selection, conformal radius, regret
budget grid, or retries.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.conformal_regret_envelope_v1 import (
    support_robust_decision,
    trajectory_conformal_regret_envelope,
)

from ._common import (
    ATOL,
    DLOS,
    Model,
    Protocol,
    canonical_sha256,
    load_protocol,
    read_json,
    sha256_file,
    trajectory_paths,
    write_json,
)
from ._evaluation import calibration_score, evaluate_paths, hyperparameter_grid
from ._model import build_pool, fit_model
from .support_envelope import (
    WindowMeasurement,
    _as_json,
    _measure_paths,
    _summarize_policy,
    _trajectory_tensors,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CONTRACT: Final = "deform-dlo45-crossfit-conformal-regret-envelope-v2"
REQUEST_CONTRACT: Final = "deform-dlo45-crossfit-conformal-regret-envelope-request-v2"
PARENT_CONTRACT: Final = "deform-dlo45-decision-identifiability-v1"
ROUTE_COUNT: Final = 2
SOURCE_RESULT_NAME: Final = "source_crossfit_envelope.json"
SOURCE_SEAL_NAME: Final = "source_seal.json"
MODEL_DIRECTORY_NAME: Final = "route_models"


@dataclass(frozen=True)
class CrossfitProtocol:
    dataset_repository: str
    dataset_commit: str
    route_count: int
    training_count_per_route: int
    calibration_count_per_route: int
    nested_fit_count: int
    nested_tune_count: int
    source_split_domain: str
    nested_split_domain: str
    target_route_domain: str
    candidate_action_mask: tuple[bool, ...]
    miscoverage_levels: tuple[float, ...]
    primary_miscoverage: float
    regret_budget_grid: tuple[float, ...]
    primary_regret_budget: float
    bootstrap_replicates: int
    bootstrap_seed: int
    claim_boundary: str


def _tuple_of(value: object, cast: type) -> tuple:
    if not isinstance(value, list):
        raise ValueError("protocol arrays must be JSON arrays")
    return tuple(cast(item) for item in value)


def load_crossfit_protocol(path: Path) -> CrossfitProtocol:
    value = read_json(path)
    crossfit = value.get("crossfit")
    calibration = value.get("calibration")
    decision = value.get("decision")
    bootstrap = value.get("bootstrap")
    if (
        value.get("contract") != CONTRACT
        or value.get("schema_version") != 2
        or value.get("parent_contract") != PARENT_CONTRACT
        or not isinstance(crossfit, dict)
        or not isinstance(calibration, dict)
        or not isinstance(decision, dict)
        or not isinstance(bootstrap, dict)
    ):
        raise ValueError("invalid crossfit conformal-regret protocol")
    result = CrossfitProtocol(
        dataset_repository=str(value["dataset_repository"]),
        dataset_commit=str(value["dataset_commit"]),
        route_count=int(crossfit["route_count"]),
        training_count_per_route=int(crossfit["training_count_per_route"]),
        calibration_count_per_route=int(crossfit["calibration_count_per_route"]),
        nested_fit_count=int(crossfit["nested_fit_count"]),
        nested_tune_count=int(crossfit["nested_tune_count"]),
        source_split_domain=str(crossfit["source_split_domain"]),
        nested_split_domain=str(crossfit["nested_split_domain"]),
        target_route_domain=str(crossfit["target_route_domain"]),
        candidate_action_mask=_tuple_of(calibration["candidate_action_mask"], bool),
        miscoverage_levels=_tuple_of(calibration["miscoverage_levels"], float),
        primary_miscoverage=float(calibration["primary_miscoverage"]),
        regret_budget_grid=_tuple_of(decision["regret_budget_grid"], float),
        primary_regret_budget=float(decision["primary_regret_budget"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
        claim_boundary=str(value["claim_boundary"]),
    )
    if (
        result.route_count != ROUTE_COUNT
        or result.training_count_per_route != 28
        or result.calibration_count_per_route != 28
        or result.nested_fit_count != 21
        or result.nested_tune_count != 7
        or result.nested_fit_count + result.nested_tune_count
        != result.training_count_per_route
        or result.candidate_action_mask != (False, True, True)
        or not result.miscoverage_levels
        or result.primary_miscoverage not in result.miscoverage_levels
        or any(not 0.0 < alpha < 1.0 for alpha in result.miscoverage_levels)
        or not result.regret_budget_grid
        or result.primary_regret_budget not in result.regret_budget_grid
        or any(budget < 0.0 for budget in result.regret_budget_grid)
        or result.bootstrap_replicates < 1
        or not all(
            (
                result.source_split_domain,
                result.nested_split_domain,
                result.target_route_domain,
            )
        )
    ):
        raise ValueError("invalid frozen crossfit settings")
    return result


def validate_request(
    path: Path, protocol: CrossfitProtocol, protocol_sha256: str
) -> dict[str, object]:
    value = read_json(path)
    if (
        value.get("contract") != REQUEST_CONTRACT
        or value.get("schema_version") != 2
        or value.get("status") != "authorized"
        or value.get("protocol_sha256") != protocol_sha256
        or tuple(value.get("dlos", ())) != DLOS
        or value.get("source_only_calibration") is not True
        or value.get("target_route_uses_metadata_only") is not True
        or value.get("target_tuning") is not False
        or value.get("target_retries") is not False
        or value.get("report_complete_frontier") is not True
        or not isinstance(value.get("run_key"), str)
        or not str(value["run_key"]).strip()
    ):
        raise ValueError("invalid crossfit conformal-regret request")
    return value


def _hash_key(domain: str, dlo: str, name: str, suffix: str = "") -> bytes:
    payload = f"{domain}\0{dlo}\0{name}\0{suffix}".encode()
    return hashlib.sha256(payload).digest()


def complementary_source_routes(
    names: Sequence[str], dlo: str, protocol: CrossfitProtocol
) -> tuple[dict[str, tuple[str, ...]], ...]:
    unique = tuple(names)
    if len(unique) != 56 or len(set(unique)) != 56:
        raise ValueError(f"{dlo}: expected 56 unique source trajectories")
    ordered = tuple(
        sorted(
            unique,
            key=lambda name: (
                _hash_key(protocol.source_split_domain, dlo, name),
                name,
            ),
        )
    )
    first = ordered[: protocol.training_count_per_route]
    second = ordered[protocol.training_count_per_route :]
    if len(first) != 28 or len(second) != 28:
        raise RuntimeError("crossfit source halves have incorrect sizes")
    return (
        {"training": first, "calibration": second},
        {"training": second, "calibration": first},
    )


def nested_training_split(
    names: Sequence[str], dlo: str, route: int, protocol: CrossfitProtocol
) -> dict[str, tuple[str, ...]]:
    unique = tuple(names)
    if len(unique) != protocol.training_count_per_route or len(set(unique)) != len(
        unique
    ):
        raise ValueError("nested model-selection roster has incorrect size")
    ordered = tuple(
        sorted(
            unique,
            key=lambda name: (
                _hash_key(
                    protocol.nested_split_domain,
                    dlo,
                    name,
                    str(route),
                ),
                name,
            ),
        )
    )
    return {
        "fit": ordered[: protocol.nested_fit_count],
        "tune": ordered[protocol.nested_fit_count :],
    }


def target_route(name: str, dlo: str, protocol: CrossfitProtocol) -> int:
    digest = _hash_key(protocol.target_route_domain, dlo, name)
    return int.from_bytes(digest[:8], "big") % protocol.route_count


def _paths_for_names(paths: Sequence[Path], names: Sequence[str]) -> tuple[Path, ...]:
    by_name = {path.name: path for path in paths}
    result = tuple(by_name[name] for name in names)
    if len(result) != len(names):
        raise RuntimeError("path roster lost a source trajectory")
    return result


def _select_route_model(
    train_paths: tuple[Path, ...],
    dlo: str,
    route: int,
    route_training_names: tuple[str, ...],
    parent_protocol: Protocol,
    protocol: CrossfitProtocol,
) -> tuple[Model, dict[str, object]]:
    nested = nested_training_split(
        route_training_names,
        dlo,
        route,
        protocol,
    )
    fit_features, fit_residuals, _ = build_pool(
        train_paths,
        nested["fit"],
        parent_protocol,
    )
    tune_paths = _paths_for_names(train_paths, nested["tune"])
    candidates: list[tuple[tuple[float, ...], dict[str, object]]] = []
    for settings in hyperparameter_grid(parent_protocol):
        model = fit_model(
            fit_features,
            fit_residuals,
            cluster_count=int(settings["cluster_count"]),
            neighbors=int(settings["neighbors"]),
            temperature_scale=float(settings["temperature_scale"]),
            regret_tolerance=float(settings["regret_tolerance"]),
            protocol=parent_protocol,
        )
        tune_result = evaluate_paths(tune_paths, model, parent_protocol)
        candidates.append(
            (
                calibration_score(tune_result),
                {
                    "settings": dict(settings),
                    "tune_result": tune_result,
                },
            )
        )
    candidates.sort(
        key=lambda item: (
            item[0],
            tuple(sorted(item[1]["settings"].items())),
        )
    )
    selected = candidates[0][1]
    settings = selected["settings"]
    assert isinstance(settings, dict)
    features, residuals, _ = build_pool(
        train_paths,
        route_training_names,
        parent_protocol,
    )
    model = fit_model(
        features,
        residuals,
        cluster_count=int(settings["cluster_count"]),
        neighbors=int(settings["neighbors"]),
        temperature_scale=float(settings["temperature_scale"]),
        regret_tolerance=float(settings["regret_tolerance"]),
        protocol=parent_protocol,
    )
    return model, {
        "route": route,
        "training_names": list(route_training_names),
        "nested_split": {key: list(value) for key, value in nested.items()},
        "selected_settings": settings,
        "tune_result": selected["tune_result"],
        "candidate_count": len(candidates),
    }


def _save_model(path: Path, model: Model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=model.features,
        residuals=model.residuals,
        class_labels=model.class_labels,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        action_scales=model.action_scales,
        scalars=np.asarray(
            (
                model.loss_floor,
                float(model.neighbors),
                model.temperature_scale,
                model.regret_tolerance,
            ),
            dtype=np.float64,
        ),
    )


def _load_model(path: Path) -> Model:
    with np.load(path, allow_pickle=False) as archive:
        scalars = np.asarray(archive["scalars"], dtype=np.float64)
        return Model(
            features=np.asarray(archive["features"], dtype=np.float64),
            residuals=np.asarray(archive["residuals"], dtype=np.float64),
            class_labels=np.asarray(archive["class_labels"], dtype=np.int64),
            feature_mean=np.asarray(archive["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(archive["feature_scale"], dtype=np.float64),
            action_scales=np.asarray(archive["action_scales"], dtype=np.float64),
            loss_floor=float(scalars[0]),
            neighbors=int(round(float(scalars[1]))),
            temperature_scale=float(scalars[2]),
            regret_tolerance=float(scalars[3]),
        )


def _route_model_path(root: Path, dlo: str, route: int) -> Path:
    return root / MODEL_DIRECTORY_NAME / f"{dlo.lower()}-route-{route}.npz"


def _calibrate_route(
    calibration_paths: tuple[Path, ...],
    model: Model,
    parent_protocol: Protocol,
    dlo: str,
    protocol: CrossfitProtocol,
) -> tuple[dict[str, object], list[WindowMeasurement]]:
    records = _measure_paths(calibration_paths, model, parent_protocol, dlo)
    names, realized, registered, _ = _trajectory_tensors(
        records,
        len(model.action_scales),
    )
    levels: dict[str, object] = {}
    for alpha in protocol.miscoverage_levels:
        envelope = trajectory_conformal_regret_envelope(
            realized,
            registered,
            miscoverage=alpha,
            candidate_action_mask=np.asarray(protocol.candidate_action_mask),
        )
        levels[f"{alpha:.6f}"] = envelope.summary() | {
            "trajectory_nonconformity_scores": (
                envelope.trajectory_nonconformity_scores.tolist()
            )
        }
    return {
        "calibration_trajectory_names": list(names),
        "calibration_trajectory_count": len(names),
        "decision_count_per_trajectory": int(realized.shape[1]),
        "envelopes": levels,
    }, records


def run_source(
    *,
    parent_protocol_path: Path,
    crossfit_protocol_path: Path,
    request_path: Path,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, object]:
    parent_protocol = load_protocol(parent_protocol_path)
    protocol = load_crossfit_protocol(crossfit_protocol_path)
    request = validate_request(
        request_path,
        protocol,
        sha256_file(crossfit_protocol_path),
    )
    source_dlos: dict[str, object] = {}
    model_hashes: dict[str, str] = {}
    for dlo in DLOS:
        paths = trajectory_paths(dataset_root, dlo, "train")
        routes = complementary_source_routes(
            [path.name for path in paths],
            dlo,
            protocol,
        )
        route_records: dict[str, object] = {}
        for route, split in enumerate(routes):
            model, selection = _select_route_model(
                paths,
                dlo,
                route,
                split["training"],
                parent_protocol,
                protocol,
            )
            calibration_paths = _paths_for_names(paths, split["calibration"])
            calibration, _ = _calibrate_route(
                calibration_paths,
                model,
                parent_protocol,
                dlo,
                protocol,
            )
            model_path = _route_model_path(output_root, dlo, route)
            _save_model(model_path, model)
            relative_model_path = str(model_path.relative_to(output_root))
            model_hashes[relative_model_path] = sha256_file(model_path)
            route_records[str(route)] = (
                selection
                | calibration
                | {
                    "calibration_names": list(split["calibration"]),
                    "model_path": relative_model_path,
                    "model_sha256": model_hashes[relative_model_path],
                    "calibration_data_used_for_model": False,
                }
            )
        source_dlos[dlo] = {
            "routes": route_records,
            "source_trajectory_count": len(paths),
        }

    source = {
        "contract": CONTRACT,
        "schema_version": 2,
        "stage": "source-calibration",
        "parent_contract": PARENT_CONTRACT,
        "parent_protocol_sha256": sha256_file(parent_protocol_path),
        "crossfit_protocol_sha256": sha256_file(crossfit_protocol_path),
        "request_sha256": sha256_file(request_path),
        "run_key": request["run_key"],
        "target_data_read": False,
        "target_outcomes_used": False,
        "dlos": source_dlos,
        "model_hashes": model_hashes,
        "claim_boundary": protocol.claim_boundary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = output_root / SOURCE_RESULT_NAME
    write_json(source_path, source)
    seal = {
        "contract": CONTRACT,
        "schema_version": 2,
        "stage": "source-seal",
        "source_result_sha256": sha256_file(source_path),
        "canonical_source_sha256": canonical_sha256(source),
        "crossfit_protocol_sha256": sha256_file(crossfit_protocol_path),
        "request_sha256": sha256_file(request_path),
        "model_hashes": model_hashes,
        "target_data_read": False,
        "target_outcomes_used": False,
    }
    write_json(output_root / SOURCE_SEAL_NAME, seal)
    return source


def _source_route_record(
    source: Mapping[str, object], dlo: str, route: int
) -> Mapping[str, object]:
    dlos = source.get("dlos")
    if not isinstance(dlos, dict):
        raise ValueError("missing source DLO records")
    dlo_record = dlos.get(dlo)
    if not isinstance(dlo_record, dict):
        raise ValueError(f"missing source record for {dlo}")
    routes = dlo_record.get("routes")
    if not isinstance(routes, dict) or not isinstance(routes.get(str(route)), dict):
        raise ValueError(f"missing source route {dlo}/{route}")
    return routes[str(route)]  # type: ignore[return-value]


def _source_radius(
    source: Mapping[str, object], dlo: str, route: int, alpha: float
) -> float:
    record = _source_route_record(source, dlo, route)
    envelopes = record.get("envelopes")
    if not isinstance(envelopes, dict):
        raise ValueError("missing source route envelopes")
    level = envelopes.get(f"{alpha:.6f}")
    if not isinstance(level, dict):
        raise ValueError("missing source route miscoverage level")
    return float(level["radius"])


def _trajectory_score(
    records: Sequence[WindowMeasurement],
    candidate_action_mask: tuple[bool, ...],
) -> float:
    if not records:
        raise ValueError("trajectory has no decisions")
    realized = np.stack([record.realized_regret for record in records])
    registered = np.stack([record.registered_regret for record in records])
    mask = np.asarray(candidate_action_mask, dtype=np.bool_)
    return float(np.max(realized[:, mask] - registered[:, mask]))


def _route_target_measurements(
    paths: tuple[Path, ...],
    models: Mapping[int, Model],
    parent_protocol: Protocol,
    dlo: str,
    protocol: CrossfitProtocol,
) -> tuple[list[WindowMeasurement], list[int], dict[str, int]]:
    all_records: list[WindowMeasurement] = []
    route_by_record: list[int] = []
    route_by_trajectory: dict[str, int] = {}
    for path in paths:
        route = target_route(path.name, dlo, protocol)
        route_by_trajectory[path.name] = route
        records = _measure_paths((path,), models[route], parent_protocol, dlo)
        all_records.extend(records)
        route_by_record.extend([route] * len(records))
    return all_records, route_by_record, route_by_trajectory


def _coverage_by_route(
    records: Sequence[WindowMeasurement],
    route_by_trajectory: Mapping[str, int],
    source: Mapping[str, object],
    dlo: str,
    alpha: float,
    protocol: CrossfitProtocol,
) -> dict[str, object]:
    grouped: dict[str, list[WindowMeasurement]] = defaultdict(list)
    for record in records:
        grouped[record.trajectory].append(record)
    per_trajectory: list[dict[str, object]] = []
    for name in sorted(grouped):
        route = route_by_trajectory[name]
        radius = _source_radius(source, dlo, route, alpha)
        score = _trajectory_score(grouped[name], protocol.candidate_action_mask)
        per_trajectory.append(
            {
                "trajectory": name,
                "route": route,
                "score": score,
                "radius": radius,
                "covered": bool(score <= radius + ATOL),
            }
        )
    route_summaries: dict[str, object] = {}
    for route in range(protocol.route_count):
        subset = [item for item in per_trajectory if item["route"] == route]
        covered_count = sum(bool(item["covered"]) for item in subset)
        route_summaries[str(route)] = {
            "trajectory_count": len(subset),
            "covered_trajectory_count": covered_count,
            "empirical_trajectory_coverage": covered_count / max(len(subset), 1),
        }
    covered_count = sum(bool(item["covered"]) for item in per_trajectory)
    return {
        "trajectory_count": len(per_trajectory),
        "covered_trajectory_count": covered_count,
        "empirical_trajectory_coverage": (covered_count / max(len(per_trajectory), 1)),
        "routes": route_summaries,
        "per_trajectory": per_trajectory,
    }


def _policy_actions(
    records: Sequence[WindowMeasurement],
    route_by_record: Sequence[int],
    source: Mapping[str, object],
    dlo: str,
    alpha: float,
    budget: float,
) -> list[int]:
    if len(records) != len(route_by_record):
        raise ValueError("route roster does not match target decisions")
    actions: list[int] = []
    for record, route in zip(records, route_by_record, strict=True):
        radius = _source_radius(source, dlo, route, alpha)
        decision = support_robust_decision(
            record.registered_regret,
            conformal_radius=radius,
            regret_tolerance=budget,
            fallback_action_index=0,
        )
        actions.append(decision.selected_action_index)
    return actions


def _verify_source_seal(
    source_root: Path,
    crossfit_protocol_path: Path,
    request_path: Path,
) -> tuple[dict[str, object], dict[tuple[str, int], Model]]:
    source_path = source_root / SOURCE_RESULT_NAME
    seal_path = source_root / SOURCE_SEAL_NAME
    source = read_json(source_path)
    seal = read_json(seal_path)
    if (
        source.get("contract") != CONTRACT
        or source.get("stage") != "source-calibration"
        or seal.get("contract") != CONTRACT
        or seal.get("stage") != "source-seal"
        or seal.get("source_result_sha256") != sha256_file(source_path)
        or seal.get("crossfit_protocol_sha256") != sha256_file(crossfit_protocol_path)
        or seal.get("request_sha256") != sha256_file(request_path)
    ):
        raise ValueError("crossfit source seal mismatch")
    hashes = seal.get("model_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("source seal has no model hashes")
    models: dict[tuple[str, int], Model] = {}
    for dlo in DLOS:
        for route in range(ROUTE_COUNT):
            path = _route_model_path(source_root, dlo, route)
            relative = str(path.relative_to(source_root))
            if hashes.get(relative) != sha256_file(path):
                raise ValueError(f"route model hash mismatch: {dlo}/{route}")
            models[(dlo, route)] = _load_model(path)
    return source, models


def run_target(
    *,
    parent_protocol_path: Path,
    crossfit_protocol_path: Path,
    request_path: Path,
    dataset_root: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, object]:
    parent_protocol = load_protocol(parent_protocol_path)
    protocol = load_crossfit_protocol(crossfit_protocol_path)
    request = validate_request(
        request_path,
        protocol,
        sha256_file(crossfit_protocol_path),
    )
    source, models = _verify_source_seal(
        source_root,
        crossfit_protocol_path,
        request_path,
    )

    result_dlos: dict[str, object] = {}
    all_records: list[WindowMeasurement] = []
    combined_actions: dict[str, list[int]] = defaultdict(list)
    per_decision: list[dict[str, object]] = []
    for dlo_index, dlo in enumerate(DLOS):
        paths = trajectory_paths(dataset_root, dlo, "eval")
        route_models = {
            route: models[(dlo, route)] for route in range(protocol.route_count)
        }
        records, routes, route_by_trajectory = _route_target_measurements(
            paths,
            route_models,
            parent_protocol,
            dlo,
            protocol,
        )
        all_records.extend(records)
        route_counts = {
            str(route): sum(value == route for value in route_by_trajectory.values())
            for route in range(protocol.route_count)
        }
        if min(route_counts.values()) < 1:
            raise RuntimeError(f"target routing left an empty route for {dlo}")
        policies: dict[str, object] = {}
        coverage: dict[str, object] = {}
        for alpha_index, alpha in enumerate(protocol.miscoverage_levels):
            alpha_key = f"{alpha:.6f}"
            coverage[alpha_key] = _coverage_by_route(
                records,
                route_by_trajectory,
                source,
                dlo,
                alpha,
                protocol,
            )
            budget_records: dict[str, object] = {}
            policies[alpha_key] = budget_records
            for budget_index, budget in enumerate(protocol.regret_budget_grid):
                actions = _policy_actions(
                    records,
                    routes,
                    source,
                    dlo,
                    alpha,
                    budget,
                )
                summary = _summarize_policy(
                    records,
                    actions,
                    regret_budget=budget,
                    bootstrap_replicates=protocol.bootstrap_replicates,
                    bootstrap_seed=(
                        protocol.bootstrap_seed
                        + 10000 * dlo_index
                        + 100 * alpha_index
                        + budget_index
                    ),
                )
                budget_key = f"{budget:.6f}"
                budget_records[budget_key] = _as_json(summary)
                combined_actions[f"{alpha_key}/{budget_key}"].extend(actions)
        base_actions = [record.base_certificate_action for record in records]
        base_summary = _summarize_policy(
            records,
            base_actions,
            regret_budget=0.05,
            bootstrap_replicates=protocol.bootstrap_replicates,
            bootstrap_seed=protocol.bootstrap_seed + 50000 + dlo_index,
        )
        result_dlos[dlo] = {
            "trajectory_count": len(paths),
            "decision_count": len(records),
            "target_route_counts": route_counts,
            "target_route_by_trajectory": route_by_trajectory,
            "base_finite_support_certificate": _as_json(base_summary),
            "coverage": coverage,
            "policies": policies,
        }
        for record, route in zip(records, routes, strict=True):
            per_decision.append(
                {
                    "stable_id": record.stable_id,
                    "dlo": record.dlo,
                    "trajectory": record.trajectory,
                    "current_frame": record.current_frame,
                    "route": route,
                    "registered_regret": record.registered_regret.tolist(),
                    "realized_regret": record.realized_regret.tolist(),
                    "physical_mse": record.physical_mse.tolist(),
                    "base_certificate_action": record.base_certificate_action,
                }
            )

    combined: dict[str, object] = {}
    for alpha_index, alpha in enumerate(protocol.miscoverage_levels):
        alpha_key = f"{alpha:.6f}"
        budgets: dict[str, object] = {}
        combined[alpha_key] = budgets
        for budget_index, budget in enumerate(protocol.regret_budget_grid):
            budget_key = f"{budget:.6f}"
            summary = _summarize_policy(
                all_records,
                combined_actions[f"{alpha_key}/{budget_key}"],
                regret_budget=budget,
                bootstrap_replicates=protocol.bootstrap_replicates,
                bootstrap_seed=(
                    protocol.bootstrap_seed + 90000 + 100 * alpha_index + budget_index
                ),
            )
            budgets[budget_key] = _as_json(summary)

    primary_alpha_key = f"{protocol.primary_miscoverage:.6f}"
    primary_budget_key = f"{protocol.primary_regret_budget:.6f}"
    result = {
        "contract": CONTRACT,
        "schema_version": 2,
        "stage": "held-target-audit",
        "run_key": request["run_key"],
        "parent_protocol_sha256": sha256_file(parent_protocol_path),
        "crossfit_protocol_sha256": sha256_file(crossfit_protocol_path),
        "request_sha256": sha256_file(request_path),
        "source_result_sha256": sha256_file(source_root / SOURCE_RESULT_NAME),
        "source_seal_sha256": sha256_file(source_root / SOURCE_SEAL_NAME),
        "target_tuning": False,
        "target_retries": False,
        "target_route_uses_metadata_only": True,
        "dlos": result_dlos,
        "combined_frontier": combined,
        "primary": combined[primary_alpha_key][primary_budget_key],
        "per_decision": per_decision,
        "claim_boundary": protocol.claim_boundary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "target_result.json", result)
    compact = {
        "contract": CONTRACT,
        "schema_version": 2,
        "status": "complete",
        "run_key": request["run_key"],
        "primary_miscoverage": protocol.primary_miscoverage,
        "primary_nominal_trajectory_coverage": (1.0 - protocol.primary_miscoverage),
        "primary_regret_budget": protocol.primary_regret_budget,
        "primary": result["primary"],
        "combined_frontier": combined,
        "coverage": {
            dlo: result_dlos[dlo]["coverage"][primary_alpha_key] for dlo in DLOS
        },
        "target_route_counts": {
            dlo: result_dlos[dlo]["target_route_counts"] for dlo in DLOS
        },
        "claim_boundary": protocol.claim_boundary,
    }
    write_json(output_root / "compact_result.json", compact)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("source", "target"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--parent-protocol", type=Path, required=True)
        sub.add_argument("--crossfit-protocol", type=Path, required=True)
        sub.add_argument("--request", type=Path, required=True)
        sub.add_argument("--dataset-root", type=Path, required=True)
        sub.add_argument("--output-root", type=Path, required=True)
        if command == "target":
            sub.add_argument("--source-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source":
        run_source(
            parent_protocol_path=args.parent_protocol,
            crossfit_protocol_path=args.crossfit_protocol,
            request_path=args.request,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
    else:
        run_target(
            parent_protocol_path=args.parent_protocol,
            crossfit_protocol_path=args.crossfit_protocol,
            request_path=args.request,
            dataset_root=args.dataset_root,
            source_root=args.source_root,
            output_root=args.output_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
