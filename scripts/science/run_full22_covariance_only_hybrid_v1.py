#!/usr/bin/env python3
"""Cross-fit a covariance-only hybrid on sealed released-PhysTwin evidence.

The predictive mean is the exact ``last_residual`` array. Only covariance from
one frozen Bayesian endpoint model is transplanted. Donor identity and one
positive scale per forecast horizon are selected inside leave-one-object-out
folds. The released 22-object cohort is already open, so the result is
retrospective development evidence rather than a confirmatory claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from bayesian_phystwin.covariance_only_hybrid import (
    compose_covariance_only_hybrid,
)
from bayesian_phystwin.covariance_only_hybrid_analysis import (
    DONORS,
    HORIZONS,
    bootstrap_family,
    crossfit_select,
    effect_matrices,
    metric_for_fold,
    score_scale_grid,
    score_zero_covariance,
)

PROTOCOL_SCHEMA: Final = "bayesian-phystwin-full22-covariance-only-hybrid-v1"
RESULT_SCHEMA: Final = "bayesian-phystwin-full22-covariance-only-hybrid-result-v1"
SCHEMA_VERSION: Final = 1
REFERENCE: Final = "last_residual"
EXPECTED_CASE_COUNT: Final = 22
LOWER_HEX: Final = frozenset("0123456789abcdef")

# Backward-compatible private alias used by focused regression tests.
_effect_matrices = effect_matrices


@dataclass(frozen=True, slots=True)
class PredictionCaseRecord:
    case_id: str
    path: str
    sha256: str


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _atomic_output_directory(target: Path, *, force: bool) -> Iterator[Path]:
    target = target.resolve()
    if target.exists():
        if not force:
            raise FileExistsError(f"output already exists: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        yield temporary
        if target.exists():
            raise FileExistsError(f"output appeared during publication: {target}")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(f"{name} must be a single canonical line")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = " positive" if positive else ""
        raise ValueError(f"{name} must be a finite{qualifier} number")
    return result


def _literal_sha(value: object, *, name: str, length: int) -> str:
    text = _text(value, name=name)
    if len(text) != length or any(character not in LOWER_HEX for character in text):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return text


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}") from error
    return _mapping(payload, name=str(path))


def load_protocol(path: Path) -> Mapping[str, object]:
    """Load and validate the frozen covariance-only development protocol."""

    protocol = _load_json(path)
    declared = _literal_sha(protocol.get("protocol_id"), name="protocol_id", length=64)
    identity = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if declared != _canonical_sha256(identity):
        raise ValueError("protocol_id does not match the canonical protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unexpected covariance-only protocol contract")
    if protocol.get("status") != "retrospective-cross-fitted-development-only":
        raise ValueError("protocol must remain retrospective and non-claim-bearing")
    hypothesis = _mapping(protocol.get("hypothesis"), name="hypothesis")
    if hypothesis.get("reference_mean_candidate") != REFERENCE:
        raise ValueError("reference mean candidate changed")
    if tuple(_sequence(hypothesis.get("covariance_donors"), name="donors")) != DONORS:
        raise ValueError("covariance donor roster or order changed")
    if hypothesis.get("point_prediction_change_allowed") is not False:
        raise ValueError("point-prediction changes must remain prohibited")
    calibration = _mapping(protocol.get("calibration"), name="calibration")
    if (
        tuple(_sequence(calibration.get("horizon_bins"), name="horizon_bins"))
        != HORIZONS
    ):
        raise ValueError("horizon bins changed")
    scales = tuple(
        _finite(value, name=f"covariance_scales[{index}]", positive=True)
        for index, value in enumerate(
            _sequence(calibration.get("covariance_scales"), name="covariance_scales")
        )
    )
    if not scales or tuple(sorted(set(scales))) != scales or 1.0 not in scales:
        raise ValueError("covariance_scales must be sorted, unique, and contain one")
    if calibration.get("isotropic_variance_m2") != 0.0:
        raise ValueError("v1 must not tune an isotropic variance")
    inference = _mapping(protocol.get("inference"), name="inference")
    _integer(
        inference.get("bootstrap_replicates"),
        name="bootstrap_replicates",
        minimum=1000,
    )
    _integer(inference.get("bootstrap_seed"), name="bootstrap_seed")
    confidence = _finite(inference.get("confidence"), name="confidence")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie strictly inside (0.5, 1)")
    boundary = _mapping(
        protocol.get("information_boundary"), name="information_boundary"
    )
    for field in (
        "claim_authorized",
        "selection_authorized",
        "promotion_authorized",
        "fresh_target_outcomes_used",
    ):
        if boundary.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    return protocol


def _discover_source_root(path: Path) -> Path:
    marker = Path("predictions") / REFERENCE / "prediction_manifest.json"
    if (path / marker).is_file():
        return path.resolve()
    matches = sorted(path.rglob(str(marker)))
    if len(matches) != 1:
        raise ValueError(
            f"source artifact must contain exactly one {marker}; found {len(matches)}"
        )
    return matches[0].parents[2].resolve()


def _prediction_records(
    source_root: Path,
    candidate_id: str,
    *,
    expected_protocol_id: str,
) -> tuple[Mapping[str, object], dict[str, PredictionCaseRecord]]:
    manifest = _load_json(
        source_root / "predictions" / candidate_id / "prediction_manifest.json"
    )
    if manifest.get("contract") != (
        "bayesian-phystwin-full22-discrepancy-prediction-manifest"
    ):
        raise ValueError(f"unexpected prediction manifest for {candidate_id}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("candidate_id") != candidate_id
        or manifest.get("protocol_id") != expected_protocol_id
    ):
        raise ValueError(f"prediction lineage changed for {candidate_id}")
    records: dict[str, PredictionCaseRecord] = {}
    for index, raw in enumerate(
        _sequence(manifest.get("case_records"), name=f"{candidate_id}.case_records")
    ):
        row = _mapping(raw, name=f"{candidate_id}.case_records[{index}]")
        case_id = _text(row.get("case_id"), name="case_id")
        if case_id in records or row.get("prediction_success") is not True:
            raise ValueError(f"invalid prediction record for {candidate_id}/{case_id}")
        records[case_id] = PredictionCaseRecord(
            case_id=case_id,
            path=_text(row.get("path"), name="path"),
            sha256=_literal_sha(row.get("sha256"), name="sha256", length=64),
        )
    if (
        len(records) != EXPECTED_CASE_COUNT
        or manifest.get("case_count") != EXPECTED_CASE_COUNT
    ):
        raise ValueError(f"expected {EXPECTED_CASE_COUNT} cases for {candidate_id}")
    return manifest, records


def _load_case_npz(path: Path, expected_sha256: str) -> Mapping[str, np.ndarray]:
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"prediction case digest changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.asarray(archive[name]) for name in archive.files}
    expected = {
        "validation_mean_m",
        "validation_covariance_m2",
        "future_mean_m",
        "future_covariance_m2",
        "prediction_success",
        "fit_end",
        "train_end",
        "frame_count",
    }
    if set(result) != expected or not bool(result["prediction_success"].item()):
        raise ValueError(f"prediction case contract changed: {path}")
    return result


def _exact_reference_mean(value: object, *, case_id: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"reference future mean is not a NumPy array for {case_id}")
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"reference future mean is not float64 for {case_id}")
    if not value.flags.c_contiguous:
        raise ValueError(f"reference future mean is not C-contiguous for {case_id}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"reference future mean is not finite for {case_id}")
    return value


def _horizon_groups(frame_count: int) -> dict[str, np.ndarray]:
    if frame_count < len(HORIZONS):
        raise ValueError("future interval is too short for three horizon bins")
    return {
        label: chunk
        for label, chunk in zip(
            HORIZONS,
            np.array_split(np.arange(frame_count, dtype=np.int64), len(HORIZONS)),
            strict=True,
        )
    }


def _action_family(case_id: str) -> str:
    normalized = case_id.lower().replace("-", "_")
    for family in (
        "rope_double_hand",
        "double_stretch",
        "double_lift",
        "single_clift",
        "single_lift",
        "single_push",
        "weird_package",
    ):
        if family in normalized:
            return family
    return "unclassified"


def _regime(family: str) -> str:
    if family in {
        "rope_double_hand",
        "double_stretch",
        "double_lift",
        "single_clift",
        "single_lift",
    }:
        return "lifting_or_stretching"
    if family == "single_push":
        return "pushing"
    return "other"


def _strata(case_ids: Sequence[str], matrix: np.ndarray) -> list[dict[str, object]]:
    effects = np.mean(matrix, axis=1)
    families = [_action_family(case_id) for case_id in case_ids]
    rows: list[dict[str, object]] = []
    for level, labels in (
        ("action_family", families),
        ("registered_regime", [_regime(family) for family in families]),
    ):
        for label in sorted(set(labels)):
            indices = [index for index, value in enumerate(labels) if value == label]
            rows.append(
                {
                    "level": level,
                    "label": label,
                    "case_count": len(indices),
                    "mean_nll_difference": float(np.mean(effects[indices])),
                    "hybrid_better_case_count": int(np.sum(effects[indices] < 0.0)),
                    "hybrid_worse_case_count": int(np.sum(effects[indices] > 0.0)),
                    "confirmatory": False,
                }
            )
    return rows


def _comparison_markdown(report: Mapping[str, object]) -> str:
    rows = [
        _mapping(row, name="comparison")
        for row in _sequence(report["comparisons"], name="comparisons")
        if _mapping(row, name="comparison")["aggregation"] == "overall"
    ]
    lines = [
        "# Full-22 covariance-only hybrid result",
        "",
        "All NLL effects are hybrid minus zero-covariance `last_residual`; lower is better.",
        "The predictive mean is the exact same array object in every case, so track and",
        "Chamfer effects are exactly zero by construction.",
        "",
        "| Arm | Mean NLL effect | Simultaneous 95% CI | Decision |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{}` | {:.3f} | [{:.3f}, {:.3f}] | {} |".format(
                row["arm"],
                float(row["mean_nll_difference"]),
                float(row["simultaneous_interval_lower"]),
                float(row["simultaneous_interval_upper"]),
                row["familywise_decision"],
            )
        )
    summary = _mapping(report["summary"], name="summary")
    full_fit = _mapping(report["full_source_fit"], name="full_source_fit")
    coverage = _mapping(report["coverage_and_width"], name="coverage_and_width")
    lines.extend(
        [
            "",
            "## Frozen full-source fit for a separate fresh study",
            "",
            f"- Donor: `{full_fit['selected_donor']}`.",
            f"- Early/middle/late scales: `{full_fit['selected_scales']}`.",
            f"- Cross-fitted conclusion: `{summary['primary_conclusion']}`.",
            f"- Exact-mean cases: `{summary['exact_mean_identity_case_count']}`.",
            "- Marginal coverage: reference {:.3f}, hybrid {:.3f}.".format(
                float(coverage["reference_marginal_coverage"]),
                float(coverage["hybrid_marginal_coverage"]),
            ),
            "",
            "## Boundary",
            "",
            str(report["scientific_boundary"]),
            "",
            "`claim_authorized=false`, `selection_authorized=false`, and",
            "`promotion_authorized=false`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(
    source_root: Path,
    data_root: Path,
    protocol: Mapping[str, object],
    *,
    source_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Run the exact-mean, leave-one-object-out covariance experiment."""

    source = _mapping(protocol["source"], name="source")
    source_protocol_id = _literal_sha(
        source.get("source_tournament_protocol_id"),
        name="source_tournament_protocol_id",
        length=64,
    )
    discovered = _discover_source_root(source_root)
    manifests: dict[str, Mapping[str, object]] = {}
    records: dict[str, dict[str, PredictionCaseRecord]] = {}
    for candidate_id in (REFERENCE, *DONORS):
        manifests[candidate_id], records[candidate_id] = _prediction_records(
            discovered,
            candidate_id,
            expected_protocol_id=source_protocol_id,
        )
    case_ids = tuple(sorted(records[REFERENCE]))
    if any(tuple(sorted(records[donor])) != case_ids for donor in DONORS):
        raise ValueError("prediction candidate case rosters differ")

    calibration = _mapping(protocol["calibration"], name="calibration")
    scales = tuple(float(value) for value in calibration["covariance_scales"])
    scoring = _mapping(protocol["scoring"], name="scoring")
    observation_std_m = _finite(
        scoring["observation_std_m"],
        name="observation_std_m",
        positive=True,
    )
    eigenvalue_floor_m2 = _finite(
        scoring["eigenvalue_floor_m2"],
        name="eigenvalue_floor_m2",
        positive=True,
    )
    coverage_z = _finite(
        scoring["marginal_coverage_z"],
        name="marginal_coverage_z",
        positive=True,
    )
    shape = (len(case_ids), len(HORIZONS))
    reference_nll = np.empty(shape, dtype=np.float64)
    reference_coverage = np.empty(shape, dtype=np.float64)
    reference_width = np.empty(shape, dtype=np.float64)
    grid_shape = (len(case_ids), len(DONORS), len(HORIZONS), len(scales))
    nll_grid = np.empty(grid_shape, dtype=np.float64)
    coverage_grid = np.empty(grid_shape, dtype=np.float64)
    width_grid = np.empty(grid_shape, dtype=np.float64)

    from bayesian_phystwin.phystwin_confirmatory import _split_for_case
    from bayesian_phystwin.phystwin_residual_dynamics import (
        _load_pickle,
        _target_validity,
    )

    for case_index, case_id in enumerate(case_ids):
        reference_record = records[REFERENCE][case_id]
        reference_prediction = _load_case_npz(
            discovered / "predictions" / REFERENCE / reference_record.path,
            reference_record.sha256,
        )
        fit_end, train_end, frame_count = _split_for_case(
            data_root / case_id,
            float(_mapping(protocol["cohort"], name="cohort")["fit_fraction"]),
        )
        if (
            int(reference_prediction["fit_end"]) != fit_end
            or int(reference_prediction["train_end"]) != train_end
            or int(reference_prediction["frame_count"]) != frame_count
        ):
            raise ValueError(f"reference split changed for {case_id}")
        data = _load_pickle(data_root / case_id / "final_data.pkl")
        baseline = np.asarray(
            _load_pickle(data_root / case_id / "inference.pkl"),
            dtype=np.float64,
        )[:frame_count]
        observed = np.asarray(data["object_points"], dtype=np.float64)[:frame_count]
        valid = _target_validity(
            np.asarray(data["object_visibilities"], dtype=bool),
            np.asarray(data["object_motions_valid"], dtype=bool),
        )[:frame_count]
        residual_future = (observed - baseline[:, : observed.shape[1]])[train_end:]
        valid_future = valid[train_end:]
        reference_mean = _exact_reference_mean(
            reference_prediction["future_mean_m"],
            case_id=case_id,
        )
        if reference_mean.shape != residual_future.shape:
            raise ValueError(f"reference future shape changed for {case_id}")
        error = residual_future - reference_mean
        groups = _horizon_groups(len(error))
        for horizon_index, horizon in enumerate(HORIZONS):
            values = score_zero_covariance(
                error[groups[horizon]],
                valid_future[groups[horizon]],
                observation_std_m=observation_std_m,
                eigenvalue_floor_m2=eigenvalue_floor_m2,
                marginal_coverage_z=coverage_z,
            )
            reference_nll[case_index, horizon_index] = values[0]
            reference_coverage[case_index, horizon_index] = values[1]
            reference_width[case_index, horizon_index] = values[2]
        for donor_index, donor in enumerate(DONORS):
            donor_record = records[donor][case_id]
            donor_prediction = _load_case_npz(
                discovered / "predictions" / donor / donor_record.path,
                donor_record.sha256,
            )
            if (
                int(donor_prediction["fit_end"]) != fit_end
                or int(donor_prediction["train_end"]) != train_end
                or int(donor_prediction["frame_count"]) != frame_count
            ):
                raise ValueError(f"donor split changed for {donor}/{case_id}")
            covariance = np.asarray(
                donor_prediction["future_covariance_m2"],
                dtype=np.float64,
            )
            identity_probe = compose_covariance_only_hybrid(
                reference_mean,
                covariance,
                reference_predictor_id=REFERENCE,
                covariance_donor_id=donor,
                covariance_scale=1.0,
                metadata={"case_id": case_id, "role": "identity-probe"},
            )
            if identity_probe.mean_m is not reference_mean:
                raise AssertionError("hybrid mean identity was not preserved")
            for horizon_index, horizon in enumerate(HORIZONS):
                indices = groups[horizon]
                nll, coverage, width = score_scale_grid(
                    error[indices],
                    covariance[indices],
                    valid_future[indices],
                    scales=scales,
                    observation_std_m=observation_std_m,
                    eigenvalue_floor_m2=eigenvalue_floor_m2,
                    marginal_coverage_z=coverage_z,
                )
                nll_grid[case_index, donor_index, horizon_index] = nll
                coverage_grid[case_index, donor_index, horizon_index] = coverage
                width_grid[case_index, donor_index, horizon_index] = width

    folds, full_fit = crossfit_select(case_ids, nll_grid, scales)
    effects = effect_matrices(reference_nll, nll_grid, scales, folds)
    selected_coverage = metric_for_fold(coverage_grid, folds, scales)
    selected_width = metric_for_fold(width_grid, folds, scales)
    hybrid_record_ids: dict[str, str] = {}
    for case_id, fold in zip(case_ids, folds, strict=True):
        reference_record = records[REFERENCE][case_id]
        donor_record = records[fold.selected_donor][case_id]
        reference_prediction = _load_case_npz(
            discovered / "predictions" / REFERENCE / reference_record.path,
            reference_record.sha256,
        )
        donor_prediction = _load_case_npz(
            discovered / "predictions" / fold.selected_donor / donor_record.path,
            donor_record.sha256,
        )
        reference_mean = _exact_reference_mean(
            reference_prediction["future_mean_m"],
            case_id=case_id,
        )
        groups = _horizon_groups(len(reference_mean))
        schedule = np.empty(reference_mean.shape[:-1], dtype=np.float64)
        for horizon, scale in zip(HORIZONS, fold.selected_scales, strict=True):
            schedule[groups[horizon], :] = scale
        prediction = compose_covariance_only_hybrid(
            reference_mean,
            donor_prediction["future_covariance_m2"],
            reference_predictor_id=REFERENCE,
            covariance_donor_id=fold.selected_donor,
            covariance_scale=schedule,
            metadata={
                "case_id": case_id,
                "selection": "leave-one-object-out-source-only",
                "horizon_scales": dict(
                    zip(HORIZONS, fold.selected_scales, strict=True)
                ),
            },
        )
        if prediction.mean_m is not reference_mean:
            raise AssertionError("final covariance-only hybrid changed the mean object")
        hybrid_record_ids[case_id] = str(prediction.record.artifact_id)

    inference = _mapping(protocol["inference"], name="inference")
    primary_arm = "crossfit_selected_scaled_covariance"
    primary_rows = bootstrap_family(
        {primary_arm: effects[primary_arm]},
        arm_order=(primary_arm,),
        replicates=int(inference["bootstrap_replicates"]),
        seed=int(inference["bootstrap_seed"]),
        confidence=float(inference["confidence"]),
    )
    diagnostic_order = tuple(arm for arm in effects if arm != primary_arm)
    diagnostic_rows = bootstrap_family(
        effects,
        arm_order=diagnostic_order,
        replicates=int(inference["bootstrap_replicates"]),
        seed=int(inference["bootstrap_seed"]) + 1000,
        confidence=float(inference["confidence"]),
    )
    primary_overall = next(
        row for row in primary_rows if row["aggregation"] == "overall"
    )
    decision = primary_overall["familywise_decision"]
    conclusion = {
        "hybrid_better": "crossfitted-covariance-only-nll-gain",
        "hybrid_worse": "crossfitted-covariance-only-nll-regression",
        "inconclusive": "crossfitted-covariance-only-result-inconclusive",
    }[str(decision)]
    fold_counts = Counter(fold.selected_donor for fold in folds)
    report: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "analysis_status": "retrospective-cross-fitted-development-only",
        "protocol_id": protocol["protocol_id"],
        "source": dict(source_metadata),
        "source_root_identity": {
            "prediction_manifest_ids": {
                candidate_id: manifests[candidate_id]["prediction_artifact_sha256"]
                for candidate_id in (REFERENCE, *DONORS)
            },
            "case_ids": list(case_ids),
        },
        "hypothesis": {
            "reference_mean_candidate": REFERENCE,
            "covariance_donors": list(DONORS),
            "mean_object_identity_preserved_for_every_case": True,
            "point_prediction_changed": False,
            "track_error_difference_m": 0.0,
            "chamfer_distance_difference_m": 0.0,
        },
        "calibration": {
            "outer_folds": len(case_ids),
            "training_units_per_fold": len(case_ids) - 1,
            "horizon_bins": list(HORIZONS),
            "covariance_scales": list(scales),
            "selected_donor_fold_counts": dict(sorted(fold_counts.items())),
            "folds": [
                {
                    "held_case_id": fold.held_case_id,
                    "selected_donor": fold.selected_donor,
                    "selected_scales": list(fold.selected_scales),
                    "donor_scales": {
                        donor: list(fold.donor_scales[donor]) for donor in DONORS
                    },
                    "donor_training_scores": dict(fold.donor_training_scores),
                    "hybrid_record_id": hybrid_record_ids[fold.held_case_id],
                }
                for fold in folds
            ],
        },
        "full_source_fit": full_fit,
        "comparisons": [*primary_rows, *diagnostic_rows],
        "coverage_and_width": {
            "nominal_marginal_coverage": scoring["nominal_marginal_coverage"],
            "reference_marginal_coverage": float(np.mean(reference_coverage)),
            "hybrid_marginal_coverage": float(np.mean(selected_coverage)),
            "reference_mean_full_width_m": float(np.mean(reference_width)),
            "hybrid_mean_full_width_m": float(np.mean(selected_width)),
            "hybrid_to_reference_width_ratio": float(
                np.mean(selected_width) / np.mean(reference_width)
            ),
        },
        "strata": _strata(case_ids, effects[primary_arm]),
        "summary": {
            "primary_conclusion": conclusion,
            "primary_arm": primary_arm,
            "selected_donor_fold_counts": dict(sorted(fold_counts.items())),
            "full_source_selected_donor": full_fit["selected_donor"],
            "full_source_selected_scales": full_fit["selected_scales"],
            "exact_mean_identity_case_count": len(hybrid_record_ids),
            "claim_authorized": False,
            "selection_authorized": False,
            "promotion_authorized": False,
        },
        "scientific_boundary": (
            "The released full-22 cohort was already open before this hypothesis. "
            "Leave-one-object-out selection prevents each scored object from tuning "
            "its own donor or scale, but this remains retrospective development "
            "evidence and cannot establish fresh-object transfer or authorize a claim."
        ),
        "claim_authorized": False,
        "selection_authorized": False,
        "promotion_authorized": False,
    }
    report["report_id"] = _canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analyzer-revision", required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-run-attempt", type=int, required=True)
    parser.add_argument("--source-artifact-id", type=int, required=True)
    parser.add_argument("--source-artifact-name", required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    source = _mapping(protocol["source"], name="source")
    expected = {
        "run_id": args.source_run_id,
        "run_attempt": args.source_run_attempt,
        "artifact_id": args.source_artifact_id,
        "artifact_name": args.source_artifact_name,
        "artifact_digest": args.source_artifact_digest,
    }
    for field, value in expected.items():
        if source.get(field) != value:
            raise ValueError(f"source metadata differs for {field}")
    source_metadata = {
        **expected,
        "source_head_sha": source["source_head_sha"],
        "source_tournament_protocol_id": source["source_tournament_protocol_id"],
        "analyzer_revision": _literal_sha(
            args.analyzer_revision,
            name="analyzer_revision",
            length=40,
        ),
    }
    report = run_analysis(
        args.source_root,
        args.data_root,
        protocol,
        source_metadata=source_metadata,
    )
    with _atomic_output_directory(args.output_dir, force=args.force) as temporary:
        _write_json(temporary / "result.json", report)
        summary = {
            "schema": RESULT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": report["status"],
            "analysis_status": report["analysis_status"],
            "protocol_id": report["protocol_id"],
            "report_id": report["report_id"],
            "summary": report["summary"],
            "coverage_and_width": report["coverage_and_width"],
            "full_source_fit": report["full_source_fit"],
            "source": report["source"],
            "claim_authorized": False,
            "selection_authorized": False,
            "promotion_authorized": False,
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "result.md").write_text(
            _comparison_markdown(report),
            encoding="utf-8",
        )
        shutil.copy2(args.protocol, temporary / "protocol.json")
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{_file_sha256(path)}  {path.name}\n" for path in files),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "primary_conclusion": _mapping(
                    report["summary"],
                    name="summary",
                )["primary_conclusion"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
