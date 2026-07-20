"""Frozen query-field selection on the already-open Deform360 source panel.

This is a development-only interpolation study.  It consumes exactly the
audited open 27-case online-belief run and its bound independent-source
outcomes.  Assimilation identities and field anchors are permanently removed
from scoring.  No held Deform360 protocol or artifact is accepted here.  The
source ``target_data.pkl`` files are trusted local research artifacts and are
deserialized only after their prediction/outcome bindings and checksums pass;
this loader must not be used on untrusted pickle payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_frozen_query_field import (
    FrameZeroQuerySet,
    FrozenFieldConfig,
    FrozenFieldGeometry,
    FrozenNodalDisplacementField,
    query_frozen_nodal_field,
)
from .deform360_online_belief_evaluation import (
    CENTER_COUNT,
    EXPECTED_SOURCE_EPISODES,
    PROTOCOL_ID as SOURCE_PROTOCOL_ID,
    _post_update_scored_frames,
    _sha256,
    _symmetric_euclidean_chamfer_m,
    evaluate_deform360_online_belief_case,
)
from .phystwin_online_belief import deterministic_farthest_point_ids


PROTOCOL_ID = "deform360-open27-frozen-query-field-v1-development"
ARTIFACT_KIND = "Deform360Open27QueryFieldDevelopmentDecision"
PRIMARY_ARM = "recursive_rbf_risk_limited"
COMPARATOR_ARM = "physical_prior"
ANCHOR_COUNTS = (64, 128, 256)
GAUSSIAN_NEIGHBOR_COUNTS = (4, 8, 12)
GAUSSIAN_LENGTH_SCALE_FRACTIONS = (0.05, 0.10, 0.20)
SUPPORT_RADIUS_FRACTION = 0.50
OBJECT_SCALE_QUANTILES = (0.05, 0.95)
SELECTION_TOLERANCE_M = 1e-12
SELECTION_METRIC = "equal_arm_field_native_identity_rmse_m"
TARGET_RANKING_METRIC = "primary_target_identity_rmse_m"
SOURCE_PREDICTION_ARCHIVE_BASENAMES = ("prediction.npz", "sealed_prediction.npz")

_AGGREGATE_METRICS = (
    "primary_target_identity_rmse_m",
    "primary_target_symmetric_chamfer_m",
    "comparator_target_identity_rmse_m",
    "comparator_target_symmetric_chamfer_m",
    "primary_field_native_identity_rmse_m",
    "primary_field_native_symmetric_chamfer_m",
    "comparator_field_native_identity_rmse_m",
    "comparator_field_native_symmetric_chamfer_m",
    "equal_arm_field_native_identity_rmse_m",
    "geometry_supported_query_fraction",
    "geometry_kth_within_support_fraction",
    "geometry_nearest_distance_mean_m",
    "geometry_nearest_distance_p95_m",
    "geometry_nearest_distance_maximum_m",
    "geometry_kth_distance_mean_m",
    "geometry_kth_distance_p95_m",
    "geometry_kth_distance_maximum_m",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expected_case_names() -> tuple[str, ...]:
    return tuple(
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episodes in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episodes
    )


def _case_identity(case_name: str) -> tuple[str, int]:
    for object_id, episodes in EXPECTED_SOURCE_EPISODES.items():
        prefix = f"{object_id}-ep"
        if case_name.startswith(prefix):
            episode = int(case_name[len(prefix) :])
            if episode in episodes and case_name == f"{object_id}-ep{episode:04d}":
                return object_id, episode
    raise ValueError(f"case is outside the fixed open27 whitelist: {case_name}")


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _load_pickle_mapping(path: Path) -> Mapping[str, Any]:
    """Load one already-authenticated, trusted source-panel pickle payload."""

    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"target payload is not a mapping: {path}")
    return value


def _object_scale_m(frame_zero_points_m: np.ndarray) -> float:
    points = np.asarray(frame_zero_points_m, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 0
        and np.all(np.isfinite(points)),
        "frame-zero object points must have finite nonempty shape (N, 3)",
    )
    bounds = np.quantile(
        points,
        np.asarray(OBJECT_SCALE_QUANTILES, dtype=np.float64),
        axis=0,
        method="linear",
    )
    scale = float(np.linalg.norm(bounds[1] - bounds[0]))
    _require(np.isfinite(scale) and scale > 0.0, "robust object scale is not positive")
    return scale


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    operator_id: str
    neighbor_count: int
    length_scale_fraction: float

    def descriptor(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "operator_id": self.operator_id,
            "neighbor_count": self.neighbor_count,
            "length_scale_fraction": self.length_scale_fraction,
            "support_radius_fraction": SUPPORT_RADIUS_FRACTION,
        }

    def config(self, object_scale_m: float) -> FrozenFieldConfig:
        support = SUPPORT_RADIUS_FRACTION * object_scale_m
        if self.operator_id == "nearest-v1":
            return FrozenFieldConfig(
                operator_id="nearest-v1",
                maximum_support_distance_m=support,
                unsupported_query_policy="emit-prediction-and-mask-v1",
            )
        return FrozenFieldConfig(
            operator_id="gaussian-knn-normalized-v1",
            maximum_support_distance_m=support,
            unsupported_query_policy="emit-prediction-and-mask-v1",
            gaussian_neighbor_count=self.neighbor_count,
            gaussian_length_scale_m=self.length_scale_fraction * object_scale_m,
        )


def _candidate_grid() -> tuple[_Candidate, ...]:
    candidates = [
        _Candidate(
            candidate_id="nearest-v1",
            operator_id="nearest-v1",
            neighbor_count=1,
            length_scale_fraction=0.0,
        )
    ]
    for count in GAUSSIAN_NEIGHBOR_COUNTS:
        for fraction in GAUSSIAN_LENGTH_SCALE_FRACTIONS:
            fraction_code = f"{int(round(100 * fraction)):02d}"
            candidates.append(
                _Candidate(
                    candidate_id=(
                        "gaussian-knn-normalized-v1-"
                        f"k{count:02d}-length{fraction_code}pct"
                    ),
                    operator_id="gaussian-knn-normalized-v1",
                    neighbor_count=count,
                    length_scale_fraction=fraction,
                )
            )
    return tuple(candidates)


def _trajectory_metrics(
    predicted_m: np.ndarray,
    reference_m: np.ndarray,
    availability: np.ndarray,
    *,
    scored_frames: Sequence[int],
) -> dict[str, object]:
    predicted = np.asarray(predicted_m, dtype=np.float64)
    reference = np.asarray(reference_m, dtype=np.float64)
    available = np.asarray(availability, dtype=bool)
    _require(
        predicted.shape == reference.shape
        and predicted.ndim == 3
        and predicted.shape[2] == 3,
        "predicted and reference trajectories must share shape (T, M, 3)",
    )
    _require(
        available.shape == predicted.shape[:2],
        "trajectory availability must have shape (T, M)",
    )
    identity_by_frame: list[float] = []
    chamfer_by_frame: list[float] = []
    count_by_frame: list[int] = []
    for frame_value in scored_frames:
        frame = int(frame_value)
        _require(0 <= frame < len(predicted), "scored frame exceeds trajectory")
        mask = (
            available[frame]
            & np.all(np.isfinite(predicted[frame]), axis=1)
            & np.all(np.isfinite(reference[frame]), axis=1)
        )
        _require(np.any(mask), f"no supported query identity at frame {frame}")
        residual = predicted[frame, mask] - reference[frame, mask]
        identity_by_frame.append(float(np.sqrt(np.mean(np.square(residual)))))
        chamfer_by_frame.append(
            _symmetric_euclidean_chamfer_m(
                predicted[frame, mask], reference[frame, mask]
            )
        )
        count_by_frame.append(int(np.sum(mask)))
    _require(bool(identity_by_frame), "no fixed post-update frame was scored")
    return {
        "identity_rmse_m": float(np.mean(identity_by_frame)),
        "symmetric_chamfer_m": float(np.mean(chamfer_by_frame)),
        "scored_identity_count_per_frame": {
            "minimum": int(np.min(count_by_frame)),
            "mean": float(np.mean(count_by_frame)),
            "maximum": int(np.max(count_by_frame)),
        },
    }


def _geometry_diagnostics(
    nearest_distance_m: np.ndarray,
    kth_distance_m: np.ndarray,
    supported_mask: np.ndarray,
    *,
    support_radius_m: float,
) -> dict[str, object]:
    nearest = np.asarray(nearest_distance_m, dtype=np.float64)
    kth = np.asarray(kth_distance_m, dtype=np.float64)
    supported = np.asarray(supported_mask, dtype=bool)
    _require(
        nearest.ndim == kth.ndim == supported.ndim == 1
        and len(nearest) == len(kth) == len(supported)
        and len(nearest) > 0,
        "query geometry diagnostics differ in size",
    )
    _require(
        np.array_equal(supported, nearest <= support_radius_m),
        "frozen field support mask differs from the fixed radius rule",
    )
    return {
        "query_count": len(nearest),
        "supported_query_count": int(np.sum(supported)),
        "supported_query_fraction": float(np.mean(supported)),
        "kth_within_support_count": int(np.sum(kth <= support_radius_m)),
        "kth_within_support_fraction": float(np.mean(kth <= support_radius_m)),
        "nearest_distance_m": {
            "mean": float(np.mean(nearest)),
            "p95": float(np.quantile(nearest, 0.95, method="linear")),
            "maximum": float(np.max(nearest)),
        },
        "kth_distance_m": {
            "mean": float(np.mean(kth)),
            "p95": float(np.quantile(kth, 0.95, method="linear")),
            "maximum": float(np.max(kth)),
        },
    }


def _flatten_result(result: Mapping[str, Any]) -> dict[str, float]:
    target = result["target_scores"]
    fidelity = result["field_native_fidelity"]
    geometry = result["geometry"]
    flattened = {
        "primary_target_identity_rmse_m": float(target["primary"]["identity_rmse_m"]),
        "primary_target_symmetric_chamfer_m": float(
            target["primary"]["symmetric_chamfer_m"]
        ),
        "comparator_target_identity_rmse_m": float(
            target["comparator"]["identity_rmse_m"]
        ),
        "comparator_target_symmetric_chamfer_m": float(
            target["comparator"]["symmetric_chamfer_m"]
        ),
        "primary_field_native_identity_rmse_m": float(
            fidelity["primary"]["identity_rmse_m"]
        ),
        "primary_field_native_symmetric_chamfer_m": float(
            fidelity["primary"]["symmetric_chamfer_m"]
        ),
        "comparator_field_native_identity_rmse_m": float(
            fidelity["comparator"]["identity_rmse_m"]
        ),
        "comparator_field_native_symmetric_chamfer_m": float(
            fidelity["comparator"]["symmetric_chamfer_m"]
        ),
        "equal_arm_field_native_identity_rmse_m": float(
            fidelity["equal_arm_identity_rmse_m"]
        ),
        "geometry_supported_query_fraction": float(
            geometry["supported_query_fraction"]
        ),
        "geometry_kth_within_support_fraction": float(
            geometry["kth_within_support_fraction"]
        ),
        "geometry_nearest_distance_mean_m": float(
            geometry["nearest_distance_m"]["mean"]
        ),
        "geometry_nearest_distance_p95_m": float(geometry["nearest_distance_m"]["p95"]),
        "geometry_nearest_distance_maximum_m": float(
            geometry["nearest_distance_m"]["maximum"]
        ),
        "geometry_kth_distance_mean_m": float(geometry["kth_distance_m"]["mean"]),
        "geometry_kth_distance_p95_m": float(geometry["kth_distance_m"]["p95"]),
        "geometry_kth_distance_maximum_m": float(geometry["kth_distance_m"]["maximum"]),
    }
    _require(
        set(flattened) == set(_AGGREGATE_METRICS),
        "query-field aggregate metric contract changed",
    )
    return flattened


def evaluate_query_field_case_arrays(
    primary_native_m: np.ndarray,
    comparator_native_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
    *,
    anchor_count: int,
    candidate: _Candidate,
    scored_frames: Sequence[int],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate one predeclared field candidate on permanently hidden identities."""

    primary = np.asarray(primary_native_m)
    comparator = np.asarray(comparator_native_m)
    target = np.asarray(target_m)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    _require(
        primary.dtype == comparator.dtype == target.dtype == np.dtype(np.float32),
        "native and target trajectories must have dtype float32",
    )
    _require(
        primary.shape == comparator.shape == target.shape
        and primary.ndim == 3
        and primary.shape[2] == 3,
        "native and target trajectories must share shape (T, N, 3)",
    )
    _require(
        visible.shape == valid.shape == target.shape[:2],
        "future visibility and validity must have shape (T, N)",
    )
    _require(
        centers.shape == (CENTER_COUNT,)
        and len(np.unique(centers)) == CENTER_COUNT
        and np.all((0 <= centers) & (centers < target.shape[1])),
        f"exactly {CENTER_COUNT} unique assimilation center IDs are required",
    )
    _require(
        np.array_equal(primary[0], comparator[0])
        and np.array_equal(primary[0], target[0]),
        "both native arms and target must share frame-zero identities",
    )
    _require(
        np.all(np.isfinite(primary))
        and np.all(np.isfinite(comparator))
        and np.all(np.isfinite(target[0])),
        "field inputs and target frame zero must be finite",
    )
    _require(anchor_count in ANCHOR_COUNTS, "anchor count is outside the frozen grid")

    all_ids = np.arange(target.shape[1], dtype=np.int64)
    remaining = all_ids[~np.isin(all_ids, centers, assume_unique=True)]
    _require(
        len(remaining) > anchor_count,
        "anchor selection leaves no permanently held-out query identity",
    )
    selected = deterministic_farthest_point_ids(comparator[0], remaining, anchor_count)
    anchor_ids = np.sort(selected, kind="mergesort")
    query_ids = remaining[~np.isin(remaining, anchor_ids, assume_unique=True)]
    _require(
        len(query_ids) > 0
        and not np.any(np.isin(query_ids, centers, assume_unique=True))
        and not np.any(np.isin(query_ids, anchor_ids, assume_unique=True)),
        "query identities overlap centers or field anchors",
    )

    object_scale = _object_scale_m(comparator[0])
    config = candidate.config(object_scale)
    geometry = FrozenFieldGeometry(
        anchor_ids=anchor_ids,
        anchor_positions_m=comparator[0, anchor_ids],
        assimilation_anchor_ids=np.empty(0, dtype=np.int64),
    )
    field = FrozenNodalDisplacementField(
        geometry=geometry,
        primary_nodal_trajectory_m=primary[:, anchor_ids],
        comparator_nodal_trajectory_m=comparator[:, anchor_ids],
        config=config,
    )
    queries = FrameZeroQuerySet(
        identity_ids=query_ids,
        positions_m=target[0, query_ids],
    )
    queried = query_frozen_nodal_field(field, queries)
    _require(
        not np.any(queried.exact_anchor_mask),
        "permanently held-out query unexpectedly equals a field anchor",
    )
    _require(
        np.any(queried.supported_identity_mask),
        "fixed support radius covers no permanently held-out identity",
    )

    support = queried.supported_identity_mask
    fidelity_available = np.broadcast_to(
        support[None], (len(target), len(query_ids))
    ).copy()
    target_available = (
        visible[:, query_ids]
        & valid[:, query_ids]
        & np.all(np.isfinite(target[:, query_ids]), axis=2)
        & fidelity_available
    )
    primary_target = _trajectory_metrics(
        queried.primary_prediction_m,
        target[:, query_ids],
        target_available,
        scored_frames=scored_frames,
    )
    comparator_target = _trajectory_metrics(
        queried.comparator_prediction_m,
        target[:, query_ids],
        target_available,
        scored_frames=scored_frames,
    )
    primary_fidelity = _trajectory_metrics(
        queried.primary_prediction_m,
        primary[:, query_ids],
        fidelity_available,
        scored_frames=scored_frames,
    )
    comparator_fidelity = _trajectory_metrics(
        queried.comparator_prediction_m,
        comparator[:, query_ids],
        fidelity_available,
        scored_frames=scored_frames,
    )
    geometry_report = _geometry_diagnostics(
        queried.nearest_anchor_distance_m,
        queried.kth_anchor_distance_m,
        support,
        support_radius_m=config.maximum_support_distance_m,
    )
    result: dict[str, object] = {
        "candidate": candidate.descriptor(),
        "resolved_config_m": {
            "maximum_support_distance_m": config.maximum_support_distance_m,
            "unsupported_query_policy": config.unsupported_query_policy,
            "gaussian_length_scale_m": config.gaussian_length_scale_m,
        },
        "target_scores": {
            "primary": primary_target,
            "comparator": comparator_target,
            "shared_mask": (
                "fixed support AND future visibility AND future motion-validity"
            ),
        },
        "field_native_fidelity": {
            "primary": primary_fidelity,
            "comparator": comparator_fidelity,
            "equal_arm_identity_rmse_m": 0.5
            * (
                float(primary_fidelity["identity_rmse_m"])
                + float(comparator_fidelity["identity_rmse_m"])
            ),
            "shared_mask": "fixed geometric support; no future target value or mask",
        },
        "geometry": geometry_report,
    }
    arrays = {
        "anchor_ids": anchor_ids,
        "query_ids": query_ids,
        "supported_query_mask": support,
    }
    return result, arrays


def _validate_exact_run_inventory(run_dir: Path) -> tuple[str, ...]:
    expected = tuple(sorted(_expected_case_names()))
    observed_reports = tuple(
        sorted(
            path.stem for path in run_dir.glob("*.json") if path.name != "summary.json"
        )
    )
    observed_archives = tuple(sorted(path.stem for path in run_dir.glob("*.npz")))
    _require(
        observed_reports == expected and observed_archives == expected,
        "audited run is not exactly the fixed open27 report/archive panel",
    )
    return expected


def _validate_summary(
    summary_path: Path,
    source_root: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    _require(summary_path.is_file(), f"missing audited summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(isinstance(summary, dict), "audited summary must be a JSON object")
    expected = _validate_exact_run_inventory(run_dir)
    _require(
        summary.get("protocol_id") == SOURCE_PROTOCOL_ID,
        "audited summary uses a different source protocol",
    )
    _require(
        int(summary.get("episode_count", -1)) == len(expected) == 27,
        "audited summary does not contain exactly 27 episodes",
    )
    _require(
        Path(str(summary.get("cohort_root", ""))).resolve() == source_root,
        "audited summary is bound to a different source cohort",
    )
    expected_objects = {
        key: list(value) for key, value in EXPECTED_SOURCE_EPISODES.items()
    }
    _require(
        summary.get("physical_objects") == expected_objects,
        "audited summary physical-object whitelist changed",
    )
    records = summary.get("artifacts")
    _require(isinstance(records, list), "audited summary lacks artifact bindings")
    by_case: dict[str, Mapping[str, Any]] = {}
    for value in records:
        _require(isinstance(value, Mapping), "audited artifact record is malformed")
        case = str(value.get("case", ""))
        _require(case not in by_case, f"duplicate audited artifact record: {case}")
        by_case[case] = value
    _require(tuple(sorted(by_case)) == expected, "audited artifact whitelist changed")
    return summary, by_case


def _validate_bound_input(
    audited_report: Mapping[str, Any],
    validated_report: Mapping[str, Any],
    *,
    role: str,
    expected_path: Path,
) -> dict[str, str]:
    audited = audited_report.get("inputs", {}).get(role, {})
    validated = validated_report.get("inputs", {}).get(role, {})
    _require(
        isinstance(audited, Mapping) and isinstance(validated, Mapping),
        f"{role} binding is malformed",
    )
    path = Path(str(audited.get("path", ""))).resolve()
    _require(path == expected_path.resolve(), f"{role} path left its source case")
    expected_hash = str(audited.get("sha256", ""))
    _require(
        bool(expected_hash)
        and expected_hash == str(validated.get("sha256", ""))
        and Path(str(validated.get("path", ""))).resolve() == path
        and _sha256(path) == expected_hash,
        f"{role} binding or checksum changed",
    )
    return {"path": str(path), "sha256": expected_hash}


def _load_audited_case(
    source_root: Path,
    run_dir: Path,
    case_name: str,
    artifact_record: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    object_id, episode_id = _case_identity(case_name)
    episode_dir = source_root / case_name
    _require(episode_dir.is_dir(), f"missing fixed source episode: {case_name}")
    report_path = run_dir / f"{case_name}.json"
    archive_path = run_dir / f"{case_name}.npz"
    _require(report_path.is_file() and archive_path.is_file(), f"missing {case_name}")
    _require(
        _sha256(report_path) == artifact_record.get("report_sha256")
        and _sha256(archive_path) == artifact_record.get("arrays_sha256"),
        f"audited report/archive checksum changed: {case_name}",
    )
    audited_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        audited_report.get("protocol_id") == SOURCE_PROTOCOL_ID
        and audited_report.get("case") == case_name
        and audited_report.get("object_id") == object_id
        and int(audited_report.get("episode_id", -1)) == episode_id,
        f"audited report identity changed: {case_name}",
    )

    validated_report, validated_arrays = evaluate_deform360_online_belief_case(
        episode_dir
    )
    archive_record = audited_report.get("inputs", {}).get("prediction_archive", {})
    _require(
        isinstance(archive_record, Mapping),
        "prediction_archive binding is malformed",
    )
    declared_archive = Path(str(archive_record.get("path", "")))
    _require(
        declared_archive.name in SOURCE_PREDICTION_ARCHIVE_BASENAMES
        and declared_archive.parent.resolve() == episode_dir
        and declared_archive.is_file()
        and not declared_archive.is_symlink(),
        "prediction_archive is not a direct known source-episode file",
    )
    bound_inputs = {
        role: _validate_bound_input(
            audited_report,
            validated_report,
            role=role,
            expected_path=expected_path,
        )
        for role, expected_path in (
            ("prediction_seal", episode_dir / "prediction_seal.json"),
            ("prediction_archive", declared_archive),
            ("target_data", episode_dir / "target_data.pkl"),
            ("outcome", episode_dir / "outcome.json"),
        )
    }
    scored_frames = tuple(int(value) for value in audited_report["scored_frames"])
    _require(
        scored_frames == _post_update_scored_frames(76)
        and scored_frames
        == tuple(int(value) for value in validated_report["scored_frames"]),
        f"fixed post-update frames changed: {case_name}",
    )

    required = {
        "center_ids",
        "physical_prior_m",
        "recursive_rbf_risk_limited_m",
    }
    with np.load(archive_path, allow_pickle=False) as stored:
        _require(
            required.issubset(stored.files), f"audited NPZ lacks arrays: {case_name}"
        )
        arrays = {key: np.asarray(stored[key]).copy() for key in required}
    for key in required:
        _require(
            np.array_equal(arrays[key], validated_arrays[key]),
            f"audited NPZ {key} differs from the bound source outcome: {case_name}",
        )
    centers = arrays["center_ids"]
    _require(
        centers.dtype == np.dtype(np.int64)
        and centers.shape == (CENTER_COUNT,)
        and centers.tolist() == audited_report["center_ids"],
        f"audited assimilation centers changed: {case_name}",
    )
    target_path = episode_dir / "target_data.pkl"
    target_payload = _load_pickle_mapping(target_path)
    target = np.asarray(target_payload["object_points"])
    visibility = np.asarray(target_payload["object_visibilities"])
    validity = np.asarray(target_payload["object_motions_valid"])
    prior = arrays["physical_prior_m"]
    primary = arrays["recursive_rbf_risk_limited_m"]
    _require(
        prior.dtype == primary.dtype == target.dtype == np.dtype(np.float32),
        f"audited trajectory dtype changed: {case_name}",
    )
    _require(
        prior.shape == primary.shape == target.shape
        and prior.shape[0] == 76
        and visibility.shape == validity.shape == target.shape[:2],
        f"audited trajectory shape changed: {case_name}",
    )
    _require(
        visibility.dtype == validity.dtype == np.dtype(bool),
        f"audited target masks changed dtype: {case_name}",
    )
    _require(
        np.array_equal(prior[0], target[0]) and np.array_equal(primary[0], target[0]),
        f"audited frame-zero identity binding changed: {case_name}",
    )
    metadata: dict[str, object] = {
        "case": case_name,
        "object_id": object_id,
        "episode_id": episode_id,
        "scored_frames": list(scored_frames),
        "input_hashes": {
            "audited_report_sha256": _sha256(report_path),
            "audited_npz_sha256": _sha256(archive_path),
            **{
                f"source_{role}_sha256": value["sha256"]
                for role, value in bound_inputs.items()
            },
            "primary_native_array_sha256": _sha256_array(primary),
            "comparator_native_array_sha256": _sha256_array(prior),
            "center_ids_array_sha256": _sha256_array(centers),
        },
    }
    loaded = {
        "primary": primary,
        "comparator": prior,
        "target": target,
        "visibility": visibility.astype(bool, copy=False),
        "validity": validity.astype(bool, copy=False),
        "centers": centers,
    }
    return metadata, loaded


def _aggregate_records(
    records: Sequence[tuple[str, str, Mapping[str, float]]],
) -> dict[str, object]:
    _require(bool(records), "cannot aggregate an empty query-field panel")
    cases = [case for case, _, _ in records]
    _require(len(cases) == len(set(cases)), "duplicate case in query-field aggregate")
    object_ids = tuple(sorted({object_id for _, object_id, _ in records}))
    by_object = {
        object_id: {
            metric: float(
                np.mean(
                    [
                        values[metric]
                        for _, group, values in records
                        if group == object_id
                    ]
                )
            )
            for metric in _AGGREGATE_METRICS
        }
        for object_id in object_ids
    }
    equal_case = {
        metric: float(np.mean([values[metric] for _, _, values in records]))
        for metric in _AGGREGATE_METRICS
    }
    equal_object = {
        metric: float(np.mean([by_object[value][metric] for value in object_ids]))
        for metric in _AGGREGATE_METRICS
    }
    return {
        "case_count": len(records),
        "object_count": len(object_ids),
        "equal_case_mean": equal_case,
        "by_object_equal_case_mean": by_object,
        "equal_object_mean": equal_object,
    }


def _tie_key(candidate: _Candidate) -> tuple[int, float, str]:
    return (
        candidate.neighbor_count,
        candidate.length_scale_fraction,
        candidate.candidate_id,
    )


def _rank_with_tolerance(
    rows: Sequence[dict[str, object]],
    *,
    value_key: str,
    candidates: Mapping[str, _Candidate],
) -> list[dict[str, object]]:
    remaining = list(rows)
    ranked: list[dict[str, object]] = []
    while remaining:
        minimum = min(float(row[value_key]) for row in remaining)
        tied = [
            row
            for row in remaining
            if float(row[value_key]) <= minimum + SELECTION_TOLERANCE_M
        ]
        tied.sort(key=lambda row: _tie_key(candidates[str(row["candidate_id"])]))
        for row in tied:
            ranked.append({"rank": len(ranked) + 1, **row})
        tied_ids = {id(row) for row in tied}
        remaining = [row for row in remaining if id(row) not in tied_ids]
    return ranked


def build_query_field_development_decision(
    source_root: str | Path,
    audited_run_dir: str | Path,
) -> dict[str, object]:
    """Build the deterministic open27 field-selection decision in memory."""

    source = Path(source_root).resolve()
    run_dir = Path(audited_run_dir).resolve()
    _require(source.is_dir(), f"source root is not a directory: {source}")
    _require(run_dir.is_dir(), f"audited run is not a directory: {run_dir}")
    expected = tuple(sorted(_expected_case_names()))
    observed_source_dirs = tuple(
        sorted(path.name for path in source.iterdir() if path.is_dir())
    )
    _require(
        observed_source_dirs == expected,
        "source root is not exactly the fixed open27 episode whitelist",
    )
    summary_path = run_dir / "summary.json"
    _, artifact_records = _validate_summary(summary_path, source, run_dir)

    candidates = _candidate_grid()
    candidate_by_id = {value.candidate_id: value for value in candidates}
    _require(
        len(candidate_by_id) == len(candidates) == 10,
        "frozen query-field candidate grid changed",
    )
    flat_records: dict[str, dict[int, list[tuple[str, str, dict[str, float]]]]] = {
        candidate.candidate_id: {count: [] for count in ANCHOR_COUNTS}
        for candidate in candidates
    }
    cases: dict[str, object] = {}
    input_hashes: dict[str, object] = {}
    for case_name in expected:
        metadata, loaded = _load_audited_case(
            source,
            run_dir,
            case_name,
            artifact_records[case_name],
        )
        object_scale = _object_scale_m(loaded["comparator"][0])
        case_record: dict[str, object] = {
            "object_id": metadata["object_id"],
            "episode_id": metadata["episode_id"],
            "object_scale_m": object_scale,
            "anchor_counts": {},
        }
        input_hashes[case_name] = metadata["input_hashes"]
        for anchor_count in ANCHOR_COUNTS:
            anchor_record: dict[str, object] = {"candidates": {}}
            common_arrays: dict[str, np.ndarray] | None = None
            for candidate in candidates:
                result, selected_arrays = evaluate_query_field_case_arrays(
                    loaded["primary"],
                    loaded["comparator"],
                    loaded["target"],
                    loaded["visibility"],
                    loaded["validity"],
                    loaded["centers"],
                    anchor_count=anchor_count,
                    candidate=candidate,
                    scored_frames=metadata["scored_frames"],
                )
                if common_arrays is None:
                    common_arrays = selected_arrays
                    anchor_record.update(
                        {
                            "anchor_count": anchor_count,
                            "query_count": len(selected_arrays["query_ids"]),
                            "anchor_ids": selected_arrays["anchor_ids"].tolist(),
                            "anchor_ids_sha256": _sha256_array(
                                selected_arrays["anchor_ids"]
                            ),
                            "query_identity_ids_sha256": _sha256_array(
                                selected_arrays["query_ids"]
                            ),
                            "assimilation_center_ids_sha256": _sha256_array(
                                loaded["centers"]
                            ),
                        }
                    )
                else:
                    for key in ("anchor_ids", "query_ids"):
                        _require(
                            np.array_equal(selected_arrays[key], common_arrays[key]),
                            "candidate changed the frozen anchor/query identity split",
                        )
                anchor_record["candidates"][candidate.candidate_id] = result
                flat_records[candidate.candidate_id][anchor_count].append(
                    (
                        case_name,
                        str(metadata["object_id"]),
                        _flatten_result(result),
                    )
                )
            case_record["anchor_counts"][str(anchor_count)] = anchor_record
        cases[case_name] = case_record

    aggregates: dict[str, object] = {}
    selection_rows: list[dict[str, object]] = []
    for candidate in candidates:
        by_anchor = {
            str(anchor_count): _aggregate_records(
                flat_records[candidate.candidate_id][anchor_count]
            )
            for anchor_count in ANCHOR_COUNTS
        }
        across_anchor = {
            metric: float(
                np.mean(
                    [
                        by_anchor[str(anchor_count)]["equal_object_mean"][metric]
                        for anchor_count in ANCHOR_COUNTS
                    ]
                )
            )
            for metric in _AGGREGATE_METRICS
        }
        aggregates[candidate.candidate_id] = {
            "candidate": candidate.descriptor(),
            "by_anchor_count": by_anchor,
            "equal_anchor_count_mean_of_equal_object_means": across_anchor,
        }
        selection_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "selection_objective_m": across_anchor[SELECTION_METRIC],
            }
        )
    selection_ranking = _rank_with_tolerance(
        selection_rows,
        value_key="selection_objective_m",
        candidates=candidate_by_id,
    )
    selected_id = str(selection_ranking[0]["candidate_id"])
    selected = candidate_by_id[selected_id]

    descriptive_rankings: dict[str, object] = {}
    for anchor_count in ANCHOR_COUNTS:
        rows = []
        for candidate in candidates:
            aggregate = aggregates[candidate.candidate_id]["by_anchor_count"][
                str(anchor_count)
            ]["equal_object_mean"]
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "primary_target_identity_rmse_m": aggregate[
                        "primary_target_identity_rmse_m"
                    ],
                    "primary_target_symmetric_chamfer_m": aggregate[
                        "primary_target_symmetric_chamfer_m"
                    ],
                    "comparator_target_identity_rmse_m": aggregate[
                        "comparator_target_identity_rmse_m"
                    ],
                    "comparator_target_symmetric_chamfer_m": aggregate[
                        "comparator_target_symmetric_chamfer_m"
                    ],
                }
            )
        descriptive_rankings[str(anchor_count)] = {
            "ranking_metric": TARGET_RANKING_METRIC,
            "status": (
                "descriptive open-development target ranking; excluded from the "
                "fidelity-based selection artifact"
            ),
            "ranking": _rank_with_tolerance(
                rows,
                value_key=TARGET_RANKING_METRIC,
                candidates=candidate_by_id,
            ),
        }

    decision: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "inputs": {
            "source_root": str(source),
            "audited_run_dir": str(run_dir),
            "audited_summary": {
                "path": str(summary_path),
                "sha256": _sha256(summary_path),
            },
            "case_artifact_hashes": input_hashes,
        },
        "fixed_protocol": {
            "physical_objects": {
                key: list(value) for key, value in EXPECTED_SOURCE_EPISODES.items()
            },
            "case_count": len(expected),
            "assimilation_center_count": CENTER_COUNT,
            "assimilation_centers": "audited 16 IDs; excluded before anchor FPS",
            "anchor_counts": list(ANCHOR_COUNTS),
            "anchor_selection": (
                "deterministic frame-zero FPS from all non-center identities; "
                "selected IDs sorted before field construction"
            ),
            "query_identities": (
                "all permanent non-center, non-anchor identities for each anchor tier"
            ),
            "primary_native_arm": PRIMARY_ARM,
            "comparator_native_arm": COMPARATOR_ARM,
            "scored_frames": list(_post_update_scored_frames(76)),
            "object_scale": {
                "rule": "Euclidean diagonal of coordinate-wise frame-zero quantile bbox",
                "lower_quantile": OBJECT_SCALE_QUANTILES[0],
                "upper_quantile": OBJECT_SCALE_QUANTILES[1],
                "quantile_method": "linear",
                "geometry": "all native frame-zero material identities",
            },
            "support_radius_fraction": SUPPORT_RADIUS_FRACTION,
            "unsupported_query_policy": "emit-prediction-and-mask-v1",
            "target_score_mask": (
                "fixed geometry support AND future visibility AND future validity; "
                "identical for primary and comparator"
            ),
            "fidelity_mask": (
                "fixed geometry support and finite native trajectories only; no "
                "future target coordinate, visibility, or validity"
            ),
            "metric_aggregation": (
                "coordinate RMSE and symmetric Euclidean Chamfer are computed per "
                "fixed frame then averaged equally over frames"
            ),
            "panel_aggregation": {
                "equal_case": "mean over 27 episodes",
                "equal_object": (
                    "mean within each physical object, then equal mean over 5 objects"
                ),
            },
        },
        "candidate_grid": [candidate.descriptor() for candidate in candidates],
        "case_results": cases,
        "aggregates": aggregates,
        "selection": {
            "status": "locked using only non-held open-development evidence",
            "metric": SELECTION_METRIC,
            "rule": (
                "minimize the equal-object mean of the equal-arm primary/comparator "
                "field-vs-native identity RMSE, averaged equally across the 64, "
                "128, and 256 anchor conditions"
            ),
            "tie_tolerance_m": SELECTION_TOLERANCE_M,
            "tie_break": (
                "within tolerance choose fewer neighbors, then smaller length-scale "
                "fraction, then lexicographic candidate ID; nearest uses k=1 and f=0"
            ),
            "ranking": selection_ranking,
            "selected_candidate_id": selected_id,
            "selected_config": selected.descriptor(),
            "selected_objective_m": float(
                selection_ranking[0]["selection_objective_m"]
            ),
            "future_target_scores_used_for_selection": False,
            "future_target_masks_used_for_selection": False,
        },
        "descriptive_target_score_rankings_by_anchor_count": descriptive_rankings,
        "claim_boundary": (
            "development-only interpolation selection on the already-open audited "
            "independent-source Deform360-27 panel; not a held-target result, not "
            "the native Deform360 evaluator, and not a state-of-the-art claim"
        ),
    }
    json.dumps(decision, sort_keys=True, allow_nan=False)
    return decision


def write_query_field_development_decision(
    source_root: str | Path,
    audited_run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Build and exclusively write the deterministic development decision."""

    decision = build_query_field_development_decision(source_root, audited_run_dir)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            decision,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return decision


__all__ = [
    "ANCHOR_COUNTS",
    "ARTIFACT_KIND",
    "PROTOCOL_ID",
    "build_query_field_development_decision",
    "evaluate_query_field_case_arrays",
    "write_query_field_development_decision",
]
