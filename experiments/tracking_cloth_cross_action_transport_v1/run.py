#!/usr/bin/env python3
"""Source-frozen cross-action transport on Tracking Cloth trajectory geometry."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from bayesian_phystwin_experiments.interventional_cause_adequacy_v1 import (
    CauseFamilyAdequacyStatus,
    InterventionalCauseFamilyAdequacyV1,
)
from bayesian_phystwin_experiments.interventional_transport_quotient_v1 import (
    InterventionalTransportQuotientV1,
)
from bayesian_phystwin_experiments.target_directed_intervention_design_v1 import (
    InterventionDesignStatus,
    TargetDirectedInterventionDesignV1,
)
from experiments.tracking_cloth_action_feasibility_v1._data import (
    source_trajectory,
)
from experiments.tracking_cloth_self_collision_selective_twin_v1.data import (
    audit_dataset,
)

SCHEMA = "bayesian-phystwin.tracking-cloth-cross-action-transport"
RESULT_SCHEMA = f"{SCHEMA}.result"
SEAL_SCHEMA = f"{SCHEMA}.prediction-seal"
SHA256_LENGTH = 64


@dataclass(frozen=True)
class CaseKey:
    material: str
    interaction: str
    repetition: int


@dataclass(frozen=True)
class LoadedFeature:
    key: CaseKey
    feature: np.ndarray
    path: str | None
    file_sha256: str | None
    sample_count: int
    pair_count: int
    cutoff: int
    initial_diameter_m: float


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest_record(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _finite_float(value: object, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _read_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("schema_version") != 1:
        raise ValueError("unsupported protocol schema")
    materials = tuple(str(item) for item in value["materials"])
    interactions = tuple(str(item) for item in value["interactions"])
    if len(materials) != 4 or len(set(materials)) != len(materials):
        raise ValueError("protocol must freeze four distinct materials")
    if len(interactions) != 3 or len(set(interactions)) != len(interactions):
        raise ValueError("protocol must freeze three distinct interactions")
    if int(value["model_repetition"]) != 1:
        raise ValueError("model repetition must remain 1")
    if int(value["retrospective_target_repetition"]) != 2:
        raise ValueError("retrospective target repetition must remain 2")
    if int(value["reserved_confirmation_repetition"]) != 3:
        raise ValueError("reserved confirmation repetition must remain 3")
    if tuple(int(item) for item in value["source_repetitions"]) != (1, 2):
        raise ValueError("the parser authorization must remain repetitions 1 and 2")
    if not 1 <= int(value["pca_components"]) <= 10:
        raise ValueError("pca_components must be in [1, 10]")
    if _finite_float(value["affine_constraint_weight"], "affine_constraint_weight") <= 0:
        raise ValueError("affine_constraint_weight must be positive")
    if value["retrospective_status"].get("globally_fresh_target") is not False:
        raise ValueError("the repetition-2 study must be labelled retrospective")
    if value["retrospective_status"].get("rep3_confirmation_authorized") is not False:
        raise ValueError("the protocol may not authorize repetition 3")
    return value


def build_pairwise_shape_trajectory(
    cloth: np.ndarray,
    *,
    cutoff: int,
    initial_diameter_m: float,
) -> np.ndarray:
    """Return scale-normalized pairwise-distance changes after the prefix."""

    points = np.asarray(cloth, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (20, 3):
        raise ValueError("cloth must have shape (time, 20, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("cloth must be finite")
    if not 0 <= int(cutoff) < points.shape[0] - 1:
        raise ValueError("cutoff must leave at least one future sample")
    diameter = _finite_float(initial_diameter_m, "initial_diameter_m")
    if diameter <= 0:
        raise ValueError("initial_diameter_m must be positive")

    first, second = np.triu_indices(points.shape[1], k=1)
    distances = np.linalg.norm(points[:, first] - points[:, second], axis=2)
    feature = (distances[int(cutoff) + 1 :] - distances[0]) / diameter
    if feature.shape[1] != 190 or feature.shape[0] < 2:
        raise ValueError("unexpected pairwise trajectory shape")
    if not np.all(np.isfinite(feature)):
        raise ValueError("pairwise trajectory must be finite")
    return np.ascontiguousarray(feature, dtype=np.float64)


def canonical_pca(
    rows: np.ndarray,
    *,
    n_components: int,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit deterministic source-only PCA with canonical component signs."""

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("rows must be a two-dimensional sample matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rows must be finite")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * float(singular_values[0]),
    )
    rank = int(np.sum(singular_values > tolerance))
    if int(n_components) > rank:
        raise ValueError(
            f"requested {n_components} PCA components but source rank is {rank}"
        )
    components = np.asarray(components[: int(n_components)], dtype=np.float64)
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    total_energy = float(np.sum(singular_values**2))
    retained = float(np.sum(singular_values[: int(n_components)] ** 2))
    explained = retained / total_energy if total_energy > 0 else 1.0
    return (
        np.ascontiguousarray(mean),
        np.ascontiguousarray(components),
        np.ascontiguousarray(singular_values),
        explained,
    )


def constrained_affine_coefficients(
    prototypes: np.ndarray,
    query: np.ndarray,
    *,
    constraint_weight: float,
) -> np.ndarray:
    """Least-squares coefficients with a numerically explicit sum-to-one row."""

    bank = np.asarray(prototypes, dtype=np.float64)
    target = np.asarray(query, dtype=np.float64)
    if bank.ndim != 2 or target.shape != (bank.shape[1],):
        raise ValueError("prototype/query dimensions do not agree")
    weight = _finite_float(constraint_weight, "constraint_weight")
    if weight <= 0:
        raise ValueError("constraint_weight must be positive")
    design = np.vstack((bank.T, weight * np.ones((1, bank.shape[0]))))
    response = np.concatenate((target, np.asarray([weight])))
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return np.ascontiguousarray(coefficients)


def _case_value(case: object, *names: str) -> object | None:
    if isinstance(case, Mapping):
        for name in names:
            if name in case:
                return case[name]
        return None
    for name in names:
        if hasattr(case, name):
            return getattr(case, name)
    return None


def _case_key(case: object) -> CaseKey | None:
    material = _case_value(case, "material")
    interaction = _case_value(case, "interaction", "action")
    repetition = _case_value(case, "repetition", "repeat", "rep")
    if material is None or interaction is None or repetition is None:
        return None
    try:
        return CaseKey(str(material), str(interaction), int(repetition))
    except (TypeError, ValueError):
        return None


def _walk_cases(value: object, seen: set[int]) -> Iterable[object]:
    if value is None or isinstance(value, (str, bytes, Path, np.ndarray)):
        return
    identifier = id(value)
    if identifier in seen:
        return
    seen.add(identifier)
    if _case_key(value) is not None:
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_cases(item, seen)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _walk_cases(item, seen)
        return
    if hasattr(value, "cases"):
        yield from _walk_cases(getattr(value, "cases"), seen)
    elif is_dataclass(value):
        yield from _walk_cases(asdict(value), seen)


def _call_audit(dataset_root: Path, protocol: Mapping[str, Any]) -> object:
    signature = inspect.signature(audit_dataset)
    kwargs: dict[str, object] = {}
    positional: list[object] = []
    for index, parameter in enumerate(signature.parameters.values()):
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        name = parameter.name
        if name in {"root", "dataset_root", "path"}:
            value: object = dataset_root
        elif name == "protocol":
            value = protocol
        elif name in {"archive_sha256", "expected_archive_sha256"}:
            value = protocol["dataset_archive_sha256"]
        elif name in {"extracted_sha256", "expected_extracted_sha256"}:
            value = protocol["dataset_extracted_sha256"]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        elif index == 0:
            value = dataset_root
        else:
            raise TypeError(f"unsupported required audit_dataset argument: {name}")
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            kwargs[name] = value
    return audit_dataset(*positional, **kwargs)


def _discover_case_map(
    dataset_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[CaseKey, object], object]:
    audit = _call_audit(dataset_root, protocol)
    case_map: dict[CaseKey, object] = {}
    for case in _walk_cases(audit, set()):
        key = _case_key(case)
        if key is not None:
            if key in case_map:
                raise ValueError(f"duplicate dataset case: {key}")
            case_map[key] = case
    required = {
        CaseKey(material, interaction, repetition)
        for material in protocol["materials"]
        for interaction in protocol["interactions"]
        for repetition in (1, 2)
    }
    missing = sorted(required - case_map.keys(), key=lambda item: tuple(asdict(item).values()))
    if missing:
        raise ValueError(
            "dataset audit did not expose required cases: "
            + ", ".join(str(item) for item in missing[:5])
        )
    return case_map, audit


def _case_path(case: object) -> Path | None:
    value = _case_value(case, "path", "csv_path", "file", "filename")
    if value is None:
        return None
    path = Path(str(value))
    return path if path.exists() and path.is_file() else None


def _load_feature(
    case: object,
    key: CaseKey,
    protocol: Mapping[str, Any],
) -> LoadedFeature:
    view = source_trajectory(case, protocol)
    feature = build_pairwise_shape_trajectory(
        np.asarray(view.cloth),
        cutoff=int(view.cutoff),
        initial_diameter_m=float(view.initial_diameter_m),
    )
    path = _case_path(case)
    return LoadedFeature(
        key=key,
        feature=feature,
        path=str(path) if path is not None else None,
        file_sha256=_file_digest(path) if path is not None else None,
        sample_count=int(feature.shape[0]),
        pair_count=int(feature.shape[1]),
        cutoff=int(view.cutoff),
        initial_diameter_m=float(view.initial_diameter_m),
    )


def _encode(feature: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    flat = np.asarray(feature, dtype=np.float64).reshape(-1)
    if flat.shape != mean.shape:
        raise ValueError("feature does not match the frozen PCA input shape")
    return np.ascontiguousarray(components @ (flat - mean))


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64
    )
    return float(np.sqrt(np.mean(difference**2)))


def _contrast_fit(
    source_bank: np.ndarray,
    query: np.ndarray,
    *,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    bank = np.asarray(source_bank, dtype=np.float64)
    target = np.asarray(query, dtype=np.float64)
    mean = bank.mean(axis=0)
    design = (bank - mean).T
    coefficients = np.linalg.pinv(design, rcond=relative_tolerance) @ (target - mean)
    residual = target - mean - design @ coefficients
    return coefficients, mean, design, float(np.linalg.norm(residual))


def _select_diagnostic(
    source_features: np.ndarray,
    source_queries: np.ndarray,
    interactions: Sequence[str],
    *,
    relative_tolerance: float,
) -> tuple[int, list[dict[str, Any]], np.ndarray]:
    material_count, interaction_count = source_queries.shape[:2]
    selection_rows: list[dict[str, Any]] = []
    per_action_scores = np.zeros(interaction_count, dtype=np.float64)
    for diagnostic_index, diagnostic in enumerate(interactions):
        fold_errors: list[float] = []
        fold_query_errors: list[float] = []
        for held_material in range(material_count):
            train = [index for index in range(material_count) if index != held_material]
            coefficients, _, _, source_residual = _contrast_fit(
                source_queries[train, diagnostic_index],
                source_queries[held_material, diagnostic_index],
                relative_tolerance=relative_tolerance,
            )
            fold_query_errors.append(source_residual)
            for target_index in range(interaction_count):
                if target_index == diagnostic_index:
                    continue
                target_bank = source_features[train, target_index]
                target_mean = target_bank.mean(axis=0)
                target_centered = target_bank - target_mean
                prediction = target_mean + np.tensordot(
                    coefficients,
                    target_centered,
                    axes=(0, 0),
                )
                fold_errors.append(
                    _rmse(prediction, source_features[held_material, target_index])
                )
        score = float(np.mean(fold_errors))
        per_action_scores[diagnostic_index] = score
        selection_rows.append(
            {
                "diagnostic_interaction": diagnostic,
                "leave_one_material_out_target_geometry_rmse": score,
                "leave_one_material_out_source_query_residual_mean": float(
                    np.mean(fold_query_errors)
                ),
                "leave_one_material_out_source_query_residual_max": float(
                    np.max(fold_query_errors)
                ),
                "fold_target_count": len(fold_errors),
            }
        )
    selected = min(
        range(interaction_count),
        key=lambda index: (per_action_scores[index], interactions[index]),
    )
    return selected, selection_rows, per_action_scores


def _source_adequacy_radius(
    source_queries: np.ndarray,
    diagnostic_index: int,
    *,
    relative_tolerance: float,
) -> tuple[float, np.ndarray]:
    material_count = source_queries.shape[0]
    scores = np.zeros(material_count, dtype=np.float64)
    for held_material in range(material_count):
        train = [index for index in range(material_count) if index != held_material]
        _, _, _, scores[held_material] = _contrast_fit(
            source_queries[train, diagnostic_index],
            source_queries[held_material, diagnostic_index],
            relative_tolerance=relative_tolerance,
        )
    return float(np.max(scores)), scores


def _certificate_for_material(
    *,
    material: str,
    diagnostic_query: np.ndarray,
    source_queries: np.ndarray,
    source_features: np.ndarray,
    diagnostic_index: int,
    target_indices: Sequence[int],
    interactions: Sequence[str],
    noise_radius: float,
    protocol_id: str,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[
    InterventionalCauseFamilyAdequacyV1,
    InterventionalTransportQuotientV1,
    dict[str, TargetDirectedInterventionDesignV1],
    dict[str, Any],
]:
    diagnostic_bank = source_queries[:, diagnostic_index]
    diagnostic_mean = diagnostic_bank.mean(axis=0)
    material_signatures = (diagnostic_bank - diagnostic_mean).T
    gauge_signatures = np.zeros((material_signatures.shape[0], 3), dtype=np.float64)
    residual = diagnostic_query - diagnostic_mean
    cause_signatures = {
        "material_contrast": material_signatures,
        "rigid_pairwise_gauge": gauge_signatures,
    }
    adequacy = InterventionalCauseFamilyAdequacyV1(
        residual_id=_array_digest(residual),
        intervention_roster_id=protocol_id,
        whitening_id=_digest_record(
            {
                "semantics": "source-only-pca-of-scale-normalized-pairwise-distance-change",
                "protocol_id": protocol_id,
            }
        ),
        cause_signature_ids={
            name: _array_digest(value) for name, value in cause_signatures.items()
        },
        cause_signatures=cause_signatures,
        whitened_residual=residual,
        noise_radius=noise_radius,
        relative_rank_tolerance=relative_tolerance,
        absolute_rank_tolerance=absolute_tolerance,
        metadata={
            "dataset": "Tracking Cloth Deformation v1",
            "material": material,
            "diagnostic_interaction": interactions[diagnostic_index],
            "target_outcomes_used": False,
            "retrospective_target_repetition_opened": False,
        },
    )
    target_maps: dict[str, np.ndarray] = {}
    for target_index in target_indices:
        target_bank = source_queries[:, target_index]
        target_centered = (target_bank - target_bank.mean(axis=0)).T
        target_maps[interactions[target_index]] = np.hstack(
            (target_centered, np.zeros((target_centered.shape[0], 3)))
        )
    quotient = InterventionalTransportQuotientV1(
        adequacy_certificate=adequacy,
        target_intervention_roster_id=protocol_id,
        target_transport_ids={
            target: _array_digest(value) for target, value in target_maps.items()
        },
        target_maps=target_maps,
        relative_rank_tolerance=relative_tolerance,
        absolute_rank_tolerance=absolute_tolerance,
        metadata={
            "prediction_space": "source-only trajectory-geometry PCA",
            "target_outcomes_used": False,
        },
    )

    source_design = np.hstack((material_signatures, gauge_signatures))
    candidate_designs: dict[str, np.ndarray] = {}
    for candidate_index, candidate in enumerate(interactions):
        if candidate_index == diagnostic_index:
            continue
        bank = source_queries[:, candidate_index]
        centered = (bank - bank.mean(axis=0)).T
        candidate_designs[candidate] = np.hstack(
            (centered, np.zeros((centered.shape[0], 3)))
        )
    designs: dict[str, TargetDirectedInterventionDesignV1] = {}
    decisions: dict[str, Any] = {}
    for target_index in target_indices:
        target = interactions[target_index]
        record = quotient.record_for(target)
        design: TargetDirectedInterventionDesignV1 | None = None
        if adequacy.status in {
            CauseFamilyAdequacyStatus.ADEQUATE_UNIQUE,
            CauseFamilyAdequacyStatus.ADEQUATE_SET_VALUED,
        } and not record.full_transport_permitted:
            design = TargetDirectedInterventionDesignV1(
                source_design_id=_array_digest(source_design),
                target_query_id=_array_digest(target_maps[target]),
                candidate_roster_id=protocol_id,
                source_design=source_design,
                target_map=target_maps[target],
                candidate_intervention_ids={
                    candidate: _array_digest(value)
                    for candidate, value in candidate_designs.items()
                },
                candidate_designs=candidate_designs,
                intervention_costs={candidate: 1.0 for candidate in candidate_designs},
                relative_rank_tolerance=relative_tolerance,
                absolute_rank_tolerance=absolute_tolerance,
                metadata={"target_outcomes_used": False},
            )
            designs[target] = design

        if adequacy.status is CauseFamilyAdequacyStatus.NO_DETECTABLE_ERROR:
            disposition = "no_detectable_error"
            reason = "diagnostic-response-within-source-noise-radius"
        elif adequacy.status is CauseFamilyAdequacyStatus.UNMODELED_CAUSE:
            disposition = "none_of_the_above"
            reason = "diagnostic-response-outside-source-material-span"
        elif record.full_transport_permitted:
            disposition = (
                "explain_and_transport"
                if adequacy.unique_coefficients
                else "transport_without_cause"
            )
            reason = "target-invariant-over-complete-coefficient-ambiguity"
        elif design is not None and design.status in {
            InterventionDesignStatus.TARGET_IDENTIFIED,
            InterventionDesignStatus.ALREADY_IDENTIFIABLE,
        }:
            disposition = "probe_then_reassess"
            reason = "source-only-candidate-probe-identifies-target"
        elif design is not None and design.status is InterventionDesignStatus.PARTIAL_IMPROVEMENT:
            disposition = "partial_only_fallback"
            reason = "only-a-target-subspace-is-identifiable"
        else:
            disposition = "abstain"
            reason = "registered-source-probe-roster-cannot-identify-target"
        decisions[target] = {
            "disposition": disposition,
            "reason": reason,
            "adequacy_status": adequacy.status.value,
            "transport_status": record.status.value,
            "transport_permitted": bool(record.full_transport_permitted),
            "registered_explanation_unique": bool(adequacy.unique_coefficients),
            "coefficient_ambiguity_dimension": int(adequacy.nullity),
            "target_identifiable_dimension": int(record.identifiable_dimension),
            "target_ambiguity_dimension": int(record.ambiguity_dimension),
            "selected_interventions": (
                list(design.selected_interventions) if design is not None else []
            ),
            "selected_intervention_cost": (
                design.selected_total_cost if design is not None else None
            ),
        }
    return adequacy, quotient, designs, decisions


def _npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_predictions(
    *,
    dataset_root: Path,
    protocol_path: Path,
    work_dir: Path,
) -> dict[str, Any]:
    protocol = _read_protocol(protocol_path)
    protocol_id = _digest_record(protocol)
    materials = tuple(protocol["materials"])
    interactions = tuple(protocol["interactions"])
    model_rep = int(protocol["model_repetition"])
    target_rep = int(protocol["retrospective_target_repetition"])
    case_map, _ = _discover_case_map(dataset_root, protocol)

    source_loaded: list[LoadedFeature] = []
    for material in materials:
        for interaction in interactions:
            key = CaseKey(material, interaction, model_rep)
            source_loaded.append(
                _load_feature(case_map[key], key, protocol)
            )
    shapes = {item.feature.shape for item in source_loaded}
    if len(shapes) != 1:
        raise ValueError(f"source features do not share one shape: {sorted(shapes)}")
    feature_shape = next(iter(shapes))
    source_features = np.stack(
        [item.feature for item in source_loaded], axis=0
    ).reshape(len(materials), len(interactions), *feature_shape)
    flat_source = source_features.reshape(
        len(materials) * len(interactions), -1
    )
    mean, components, singular_values, explained = canonical_pca(
        flat_source,
        n_components=int(protocol["pca_components"]),
        relative_tolerance=float(protocol["rank_relative_tolerance"]),
        absolute_tolerance=float(protocol["rank_absolute_tolerance"]),
    )
    source_queries = np.stack(
        [_encode(item.feature, mean, components) for item in source_loaded],
        axis=0,
    ).reshape(len(materials), len(interactions), components.shape[0])
    diagnostic_index, selection_rows, selection_scores = _select_diagnostic(
        source_features,
        source_queries,
        interactions,
        relative_tolerance=float(protocol["rank_relative_tolerance"]),
    )
    target_indices = tuple(
        index for index in range(len(interactions)) if index != diagnostic_index
    )
    radius, source_adequacy_scores = _source_adequacy_radius(
        source_queries,
        diagnostic_index,
        relative_tolerance=float(protocol["rank_relative_tolerance"]),
    )

    diagnostic_loaded: list[LoadedFeature] = []
    for material in materials:
        key = CaseKey(material, interactions[diagnostic_index], target_rep)
        diagnostic_loaded.append(
            _load_feature(case_map[key], key, protocol)
        )
    if {item.feature.shape for item in diagnostic_loaded} != {feature_shape}:
        raise ValueError("repetition-2 diagnostic feature shape differs from source")
    diagnostic_features = np.stack(
        [item.feature for item in diagnostic_loaded], axis=0
    )
    diagnostic_queries = np.stack(
        [_encode(item.feature, mean, components) for item in diagnostic_loaded],
        axis=0,
    )

    target_count = len(target_indices)
    coefficients = np.zeros((len(materials), len(materials) + 3))
    candidate_query = np.zeros((len(materials), target_count, components.shape[0]))
    guarded_query = np.zeros_like(candidate_query)
    candidate_full = np.zeros((len(materials), target_count, *feature_shape))
    guarded_full = np.zeros_like(candidate_full)
    action_mean = np.zeros_like(candidate_full)
    same_material = np.zeros_like(candidate_full)
    diagnostic_copy = np.repeat(
        diagnostic_features[:, None, :, :], target_count, axis=1
    )
    wrong_action = np.zeros_like(candidate_full)
    nearest_indices = np.zeros(len(materials), dtype=np.int64)
    decision_records: dict[str, Any] = {}

    diagnostic_bank_full = source_features[:, diagnostic_index]
    for material_index, material in enumerate(materials):
        distances = np.asarray(
            [
                _rmse(diagnostic_features[material_index], prototype)
                for prototype in diagnostic_bank_full
            ]
        )
        nearest_indices[material_index] = int(np.argmin(distances))
        adequacy, quotient, _, decisions = _certificate_for_material(
            material=material,
            diagnostic_query=diagnostic_queries[material_index],
            source_queries=source_queries,
            source_features=source_features,
            diagnostic_index=diagnostic_index,
            target_indices=target_indices,
            interactions=interactions,
            noise_radius=radius,
            protocol_id=protocol_id,
            relative_tolerance=float(protocol["rank_relative_tolerance"]),
            absolute_tolerance=float(protocol["rank_absolute_tolerance"]),
        )
        coefficients[material_index] = adequacy.minimum_norm_coefficients
        material_coefficients = adequacy.minimum_norm_coefficients[: len(materials)]
        per_material: dict[str, Any] = {
            "diagnostic_feature_sha256": _array_digest(
                diagnostic_features[material_index]
            ),
            "diagnostic_query_sha256": _array_digest(
                diagnostic_queries[material_index]
            ),
            "nearest_source_material": materials[nearest_indices[material_index]],
            "nearest_source_material_index": int(nearest_indices[material_index]),
            "nearest_source_rmse": float(distances[nearest_indices[material_index]]),
            "source_material_distances": {
                name: float(value) for name, value in zip(materials, distances)
            },
            "adequacy_certificate_id": adequacy.artifact_id,
            "adequacy_status": adequacy.status.value,
            "orthogonal_residual_norm": float(adequacy.orthogonal_residual_norm),
            "noise_radius": float(adequacy.noise_radius),
            "coefficient_nullity": int(adequacy.nullity),
            "material_contrast_coefficients": [
                float(value) for value in material_coefficients
            ],
            "targets": {},
        }
        for target_position, target_index in enumerate(target_indices):
            target = interactions[target_index]
            target_bank_query = source_queries[:, target_index]
            target_query_mean = target_bank_query.mean(axis=0)
            target_bank_full = source_features[:, target_index]
            target_full_mean = target_bank_full.mean(axis=0)
            target_query_centered = target_bank_query - target_query_mean
            target_full_centered = target_bank_full - target_full_mean
            candidate_query_value = target_query_mean + np.tensordot(
                material_coefficients,
                target_query_centered,
                axes=(0, 0),
            )
            candidate_full_value = target_full_mean + np.tensordot(
                material_coefficients,
                target_full_centered,
                axes=(0, 0),
            )
            candidate_query[material_index, target_position] = candidate_query_value
            candidate_full[material_index, target_position] = candidate_full_value
            action_mean[material_index, target_position] = target_full_mean
            same_material[material_index, target_position] = target_bank_full[
                material_index
            ]
            other_target_position = 1 - target_position
            other_target_index = target_indices[other_target_position]
            other_bank = source_features[:, other_target_index]
            other_mean = other_bank.mean(axis=0)
            wrong_action[material_index, target_position] = other_mean + np.tensordot(
                material_coefficients,
                other_bank - other_mean,
                axes=(0, 0),
            )
            decision = decisions[target]
            record = quotient.record_for(target)
            if decision["transport_permitted"]:
                guarded_query[material_index, target_position] = (
                    target_query_mean + record.identifiable_effect
                )
                guarded_full[material_index, target_position] = candidate_full_value
            else:
                guarded_query[material_index, target_position] = target_query_mean
                guarded_full[material_index, target_position] = target_full_mean
            per_material["targets"][target] = {
                **decision,
                "candidate_query_sha256": _array_digest(candidate_query_value),
                "candidate_full_sha256": _array_digest(candidate_full_value),
                "guarded_query_sha256": _array_digest(
                    guarded_query[material_index, target_position]
                ),
                "guarded_full_sha256": _array_digest(
                    guarded_full[material_index, target_position]
                ),
            }
        decision_records[material] = per_material

    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = work_dir / "model.npz"
    predictions_path = work_dir / "predictions.npz"
    _npz_atomic(
        model_path,
        source_features=source_features,
        source_queries=source_queries,
        pca_mean=mean,
        pca_components=components,
        singular_values=singular_values,
        selection_scores=selection_scores,
        source_adequacy_scores=source_adequacy_scores,
        selected_diagnostic_index=np.asarray(diagnostic_index, dtype=np.int64),
        target_indices=np.asarray(target_indices, dtype=np.int64),
    )
    _npz_atomic(
        predictions_path,
        diagnostic_features=diagnostic_features,
        diagnostic_queries=diagnostic_queries,
        coefficients=coefficients,
        candidate_query=candidate_query,
        guarded_query=guarded_query,
        candidate_full=candidate_full,
        guarded_full=guarded_full,
        action_mean=action_mean,
        same_material=same_material,
        diagnostic_copy=diagnostic_copy,
        wrong_action=wrong_action,
        nearest_indices=nearest_indices,
    )
    seal: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol_id,
        "protocol_sha256": _file_digest(protocol_path),
        "code_revision": os.environ.get("GITHUB_SHA"),
        "model_file": model_path.name,
        "model_sha256": _file_digest(model_path),
        "predictions_file": predictions_path.name,
        "predictions_sha256": _file_digest(predictions_path),
        "materials": list(materials),
        "interactions": list(interactions),
        "selected_diagnostic_interaction": interactions[diagnostic_index],
        "held_target_interactions": [interactions[index] for index in target_indices],
        "feature_shape": list(feature_shape),
        "pca_components": int(components.shape[0]),
        "pca_explained_source_energy_fraction": explained,
        "source_selection": selection_rows,
        "source_adequacy_radius": radius,
        "source_adequacy_scores": {
            material: float(value)
            for material, value in zip(materials, source_adequacy_scores)
        },
        "source_files": [
            {
                "material": item.key.material,
                "interaction": item.key.interaction,
                "repetition": item.key.repetition,
                "path": item.path,
                "sha256": item.file_sha256,
            }
            for item in source_loaded
        ],
        "diagnostic_files": [
            {
                "material": item.key.material,
                "interaction": item.key.interaction,
                "repetition": item.key.repetition,
                "path": item.path,
                "sha256": item.file_sha256,
            }
            for item in diagnostic_loaded
        ],
        "numeric_access": {
            "repetition_1_cases": len(source_loaded),
            "repetition_2_diagnostic_cases": len(diagnostic_loaded),
            "repetition_2_held_target_cases": 0,
            "repetition_3_cases": 0,
        },
        "target_outcomes_used_to_select_or_fit": False,
        "retrospective_target_repetition_previously_available_to_project": True,
        "decisions": decision_records,
        "claim_boundary": protocol["claim_boundary"],
    }
    seal["seal_id"] = _digest_record(seal)
    _atomic_write_json(work_dir / "seal.json", seal)
    return seal


def _load_verified_seal(
    work_dir: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    seal_path = work_dir / "seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema") != SEAL_SCHEMA or seal.get("schema_version") != 1:
        raise ValueError("unsupported prediction seal")
    expected_id = seal.pop("seal_id")
    actual_id = _digest_record(seal)
    seal["seal_id"] = expected_id
    if expected_id != actual_id:
        raise ValueError("prediction seal identity mismatch")
    protocol = _read_protocol(protocol_path)
    if seal["protocol_id"] != _digest_record(protocol):
        raise ValueError("prediction seal is not bound to this protocol")
    if seal["protocol_sha256"] != _file_digest(protocol_path):
        raise ValueError("protocol bytes changed after prediction sealing")
    if seal["numeric_access"] != {
        "repetition_1_cases": 12,
        "repetition_2_diagnostic_cases": 4,
        "repetition_2_held_target_cases": 0,
        "repetition_3_cases": 0,
    }:
        raise ValueError("prediction seal violates the frozen numeric-access boundary")
    model_path = work_dir / seal["model_file"]
    predictions_path = work_dir / seal["predictions_file"]
    if _file_digest(model_path) != seal["model_sha256"]:
        raise ValueError("sealed model hash mismatch")
    if _file_digest(predictions_path) != seal["predictions_sha256"]:
        raise ValueError("sealed prediction hash mismatch")
    with np.load(model_path, allow_pickle=False) as archive:
        model = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(predictions_path, allow_pickle=False) as archive:
        predictions = {name: np.asarray(archive[name]) for name in archive.files}
    return seal, model, predictions


def _exact_pairing_scores(
    *,
    truth: np.ndarray,
    source_features: np.ndarray,
    coefficients: np.ndarray,
    target_indices: Sequence[int],
) -> tuple[list[dict[str, Any]], float, int]:
    material_count = source_features.shape[0]
    permutations = list(itertools.permutations(range(material_count)))
    rows: list[dict[str, Any]] = []
    for permutation in permutations:
        case_errors: list[float] = []
        for material_index in range(material_count):
            beta = coefficients[material_index, :material_count]
            for target_position, target_index in enumerate(target_indices):
                bank = source_features[:, target_index]
                mean = bank.mean(axis=0)
                centered = bank - mean
                prediction = mean + np.tensordot(
                    beta,
                    centered[np.asarray(permutation)],
                    axes=(0, 0),
                )
                case_errors.append(
                    _rmse(prediction, truth[material_index, target_position])
                )
        rows.append(
            {
                "permutation": list(permutation),
                "identity": permutation == tuple(range(material_count)),
                "mean_rmse": float(np.mean(case_errors)),
            }
        )
    identity_score = next(row["mean_rmse"] for row in rows if row["identity"])
    rank = 1 + sum(
        row["mean_rmse"] < identity_score - 1e-15
        for row in rows
        if not row["identity"]
    )
    p_value = sum(
        row["mean_rmse"] <= identity_score + 1e-15 for row in rows
    ) / len(rows)
    rows.sort(key=lambda row: (row["mean_rmse"], row["permutation"]))
    return rows, float(p_value), int(rank)


def _report(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate_metrics"]
    checks = result["checks"]
    lines = [
        "# Tracking Cloth cross-action transport source-target result",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        f"Selected diagnostic interaction: `{result['selected_diagnostic_interaction']}`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    ordered = [
        "guarded_transport_rmse",
        "candidate_transport_rmse",
        "action_mean_rmse",
        "same_material_persistence_rmse",
        "diagnostic_copy_rmse",
        "wrong_action_rmse",
        "relative_gain_vs_action_mean",
        "persistence_gap_recovery",
        "diagnostic_adequacy_fraction",
        "material_match_accuracy",
        "exact_material_pairing_p_value",
    ]
    for key in ordered:
        value = aggregate.get(key)
        rendered = "n/a" if value is None else f"{float(value):.8f}"
        lines.append(f"| `{key}` | {rendered} |")
    lines += ["", "## Frozen checks", ""]
    for key, value in checks.items():
        lines.append(f"- `{key}`: **{'pass' if value else 'fail'}**")
    lines += [
        "",
        "## Information boundary",
        "",
        "Repetition 1 supplied all model and diagnostic-selection information. "
        "Only the selected repetition-2 diagnostic interaction was parsed before "
        "the prediction seal. The other eight repetition-2 interaction trajectories "
        "were opened only after the seal was verified. Repetition 3 was not parsed.",
        "",
        "This result is retrospective rather than independently blind because "
        "repetition 2 was available to earlier project analyses. A positive source "
        "gate would justify only a separately reviewed repetition-3 confirmation "
        "protocol; it does not itself authorize repetition-3 access.",
        "",
        "## Claim boundary",
        "",
        str(result["claim_boundary"]),
        "",
        f"Result ID: `{result['result_id']}`.",
        "",
    ]
    return "\n".join(lines)


def score_predictions(
    *,
    dataset_root: Path,
    protocol_path: Path,
    work_dir: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    protocol = _read_protocol(protocol_path)
    seal, model, predictions = _load_verified_seal(work_dir, protocol_path)
    materials = tuple(protocol["materials"])
    interactions = tuple(protocol["interactions"])
    diagnostic_index = int(model["selected_diagnostic_index"])
    target_indices = tuple(int(item) for item in model["target_indices"])
    target_rep = int(protocol["retrospective_target_repetition"])
    if seal["selected_diagnostic_interaction"] != interactions[diagnostic_index]:
        raise ValueError("selected diagnostic identity mismatch")
    case_map, _ = _discover_case_map(dataset_root, protocol)

    truth_loaded: list[LoadedFeature] = []
    for material in materials:
        for target_index in target_indices:
            key = CaseKey(material, interactions[target_index], target_rep)
            truth_loaded.append(_load_feature(case_map[key], key, protocol))
    feature_shape = tuple(int(item) for item in seal["feature_shape"])
    if {item.feature.shape for item in truth_loaded} != {feature_shape}:
        raise ValueError("held target feature shape differs from the sealed model")
    truth = np.stack([item.feature for item in truth_loaded], axis=0).reshape(
        len(materials), len(target_indices), *feature_shape
    )

    prediction_names = (
        "guarded_full",
        "candidate_full",
        "action_mean",
        "same_material",
        "diagnostic_copy",
        "wrong_action",
    )
    for name in prediction_names:
        if predictions[name].shape != truth.shape:
            raise ValueError(f"sealed prediction shape mismatch for {name}")

    per_case: list[dict[str, Any]] = []
    metric_vectors: dict[str, list[float]] = {name: [] for name in prediction_names}
    for material_index, material in enumerate(materials):
        for target_position, target_index in enumerate(target_indices):
            row: dict[str, Any] = {
                "material": material,
                "diagnostic_interaction": interactions[diagnostic_index],
                "target_interaction": interactions[target_index],
                "target_repetition": target_rep,
                "target_file": truth_loaded[
                    material_index * len(target_indices) + target_position
                ].path,
                "target_file_sha256": truth_loaded[
                    material_index * len(target_indices) + target_position
                ].file_sha256,
            }
            for name in prediction_names:
                value = _rmse(
                    predictions[name][material_index, target_position],
                    truth[material_index, target_position],
                )
                row[f"{name}_rmse"] = value
                metric_vectors[name].append(value)
            decision = seal["decisions"][material]["targets"][
                interactions[target_index]
            ]
            row.update(
                {
                    "disposition": decision["disposition"],
                    "adequacy_status": decision["adequacy_status"],
                    "transport_status": decision["transport_status"],
                    "transport_permitted": decision["transport_permitted"],
                }
            )
            per_case.append(row)

    means = {
        name: float(np.mean(values)) for name, values in metric_vectors.items()
    }
    pairing_rows, pairing_p, pairing_rank = _exact_pairing_scores(
        truth=truth,
        source_features=model["source_features"],
        coefficients=predictions["coefficients"],
        target_indices=target_indices,
    )
    action_mean_rmse = means["action_mean"]
    guarded_rmse = means["guarded_full"]
    persistence_rmse = means["same_material"]
    denominator = action_mean_rmse - persistence_rmse
    persistence_gap_recovery = (
        (action_mean_rmse - guarded_rmse) / denominator
        if denominator > 1e-15
        else None
    )
    relative_gain = (
        (action_mean_rmse - guarded_rmse) / action_mean_rmse
        if action_mean_rmse > 0
        else 0.0
    )
    adequacy_statuses = [
        seal["decisions"][material]["adequacy_status"] for material in materials
    ]
    adequate_fraction = sum(
        status in {"adequate_unique", "adequate_set_valued"}
        for status in adequacy_statuses
    ) / len(adequacy_statuses)
    nearest = predictions["nearest_indices"].astype(int)
    material_match_accuracy = float(
        np.mean(nearest == np.arange(len(materials), dtype=int))
    )
    aggregate = {
        "held_target_case_count": len(per_case),
        "guarded_transport_rmse": guarded_rmse,
        "candidate_transport_rmse": means["candidate_full"],
        "action_mean_rmse": action_mean_rmse,
        "same_material_persistence_rmse": persistence_rmse,
        "diagnostic_copy_rmse": means["diagnostic_copy"],
        "wrong_action_rmse": means["wrong_action"],
        "relative_gain_vs_action_mean": relative_gain,
        "persistence_gap_recovery": persistence_gap_recovery,
        "diagnostic_adequacy_fraction": adequate_fraction,
        "material_match_accuracy": material_match_accuracy,
        "exact_material_pairing_p_value": pairing_p,
        "exact_material_pairing_rank": pairing_rank,
        "transport_permitted_case_count": sum(
            bool(row["transport_permitted"]) for row in per_case
        ),
    }
    checks = {
        "eight_held_action_cases_scored": len(per_case) == 8,
        "rep3_numeric_access_remained_zero": True,
        "adequacy_fraction_meets_frozen_floor": (
            adequate_fraction
            >= float(protocol["minimum_adequate_material_fraction"])
        ),
        "diagnostic_material_match_meets_frozen_floor": (
            material_match_accuracy
            >= float(protocol["minimum_material_match_accuracy"])
        ),
        "guarded_transport_beats_action_mean": (
            guarded_rmse < action_mean_rmse
            if protocol["require_gain_over_action_mean"]
            else True
        ),
        "guarded_transport_beats_wrong_action": (
            guarded_rmse < means["wrong_action"]
            if protocol["require_gain_over_wrong_action"]
            else True
        ),
        "guarded_transport_beats_diagnostic_copy": (
            guarded_rmse < means["diagnostic_copy"]
            if protocol["require_gain_over_diagnostic_copy"]
            else True
        ),
        "persistence_gap_recovery_meets_frozen_floor": (
            persistence_gap_recovery is not None
            and persistence_gap_recovery
            >= float(protocol["minimum_persistence_gap_recovery"])
        ),
        "identity_pairing_passes_exact_permutation_control": (
            pairing_p <= float(protocol["maximum_exact_pairing_p_value"])
        ),
    }
    decision = (
        "cross-action-transport-source-gate-pass"
        if all(checks.values())
        else "cross-action-transport-source-gate-negative"
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "decision": decision,
        "protocol_id": seal["protocol_id"],
        "prediction_seal_id": seal["seal_id"],
        "selected_diagnostic_interaction": interactions[diagnostic_index],
        "held_target_interactions": [interactions[index] for index in target_indices],
        "source_selection": seal["source_selection"],
        "aggregate_metrics": aggregate,
        "checks": checks,
        "per_case": per_case,
        "material_pairing_permutations": pairing_rows,
        "numeric_access": {
            "repetition_1_cases": 12,
            "repetition_2_diagnostic_cases_before_seal": 4,
            "repetition_2_held_target_cases_after_seal": 8,
            "repetition_3_cases": 0,
        },
        "retrospective": True,
        "independently_blind_confirmation": False,
        "rep3_confirmation_authorized": False,
        "rep3_next_step": (
            "separate-protocol-review-candidate"
            if decision == "cross-action-transport-source-gate-pass"
            else "not-authorized"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = _digest_record(result)
    _atomic_write_json(output, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("seal", "score"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.json"),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.stage == "seal":
        seal = seal_predictions(
            dataset_root=args.dataset_root,
            protocol_path=args.protocol,
            work_dir=args.work_dir,
        )
        print(
            json.dumps(
                {
                    "seal_id": seal["seal_id"],
                    "selected_diagnostic_interaction": seal[
                        "selected_diagnostic_interaction"
                    ],
                    "held_target_cases_read": seal["numeric_access"][
                        "repetition_2_held_target_cases"
                    ],
                    "rep3_cases_read": seal["numeric_access"][
                        "repetition_3_cases"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output is None or args.report is None:
        parser.error("--output and --report are required for --stage score")
    result = score_predictions(
        dataset_root=args.dataset_root,
        protocol_path=args.protocol,
        work_dir=args.work_dir,
        output=args.output,
        report_path=args.report,
    )
    print(
        json.dumps(
            {"decision": result["decision"], "result_id": result["result_id"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
