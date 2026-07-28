"""Fail-closed Deform360 evaluator-parity contracts and sensitivity metrics.

The Deform360 paper publishes aggregate benchmark values, but its public
repository does not currently include the evaluator, exact splits, or complete
aggregation contract needed to reproduce those values.  This module keeps
candidate conventions useful for diagnosis without allowing them to be
misreported as official parity.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


ARTIFACT_KIND = "Deform360Official3DParityContract"
AUDIT_KIND = "Deform360Official3DParityPublicAudit"
SCHEMA_VERSION = 1
BENCHMARK_SETTINGS = ("per_episode", "multi_episode", "multi_object")
FIELD_STATUSES = ("authoritative", "candidate", "missing")
EXACT_NUMERIC_SOURCE_KINDS = frozenset(
    {"released_evaluator", "author_confirmed_contract"}
)

REQUIRED_3D_FIELDS = (
    "benchmark_setting",
    "training_case_manifest",
    "evaluation_case_manifest",
    "future_frame_manifest",
    "particle_identity_alignment",
    "prediction_preprocessing",
    "ground_truth_preprocessing",
    "validity_visibility_mask",
    "coordinate_frame",
    "length_unit",
    "chamfer_direction",
    "chamfer_point_distance",
    "chamfer_reduction",
    "track_correspondence",
    "track_point_distance",
    "track_reduction",
    "frame_aggregation",
    "episode_aggregation",
    "object_aggregation",
    "missing_case_policy",
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

PUBLIC_SOURCE_BINDINGS: dict[str, dict[str, Any]] = {
    "deform360_arxiv_v1": {
        "kind": "official_paper",
        "authority": "deform360_authoritative",
        "url": "https://arxiv.org/abs/2607.05390",
        "revision": "arxiv:2607.05390v1",
        "content_sha256": (
            "66d2bfecd6ec9b829cd810913238821adc143c4831393704ed4bcc4ccc09e05c"
        ),
        "bound_files": {
            "main.tex": (
                "b2239e4e83bf62b0960119b92c9a370f2769c2026ed5ebae75de145ceefdfe95"
            ),
            "appendix.tex": (
                "b8ccbb6c8b588d85dcd1b05908780473b712e9fc4018a237beeaac41a6e5e0b0"
            ),
        },
    },
    "deform360_public_repo": {
        "kind": "official_repository_without_evaluator",
        "authority": "deform360_authoritative",
        "url": "https://github.com/lhy0807/deform360",
        "revision": "d8522a4403b766aeb387510c04e89032a56fdf35",
        "content_sha256": None,
        "bound_files": {
            "README.md": (
                "52f2ebed1800eb8c1e6dde05fefaca15ebba4456f0756b1ca05cfc4380fc8f7a"
            ),
            "deform360/processing/control_points_stage.py": (
                "9ff82c86c22e38c56dd2ce5d872850afb6ffeb502da7338baf0b55108afb7373"
            ),
            "deform360/processing/pcd_stage.py": (
                "87553e1ea3dac5a90e46114c76aaf65901b43a064025626ae6871523065c864d"
            ),
            "deform360/processing/tracking_stage.py": (
                "04533cd9cd900ae2f5bd139568ed1a2442661f14ceda009dd7bb85e4fbd83ec2"
            ),
        },
    },
    "pgrd_candidate_metric": {
        "kind": "candidate_implementation",
        "authority": "external_method_only",
        "url": "https://github.com/shivanshpatel35/pgrd",
        "revision": "e294d96723054f77a1cfdd3c2c052de7b7cd9ce3",
        "content_sha256": None,
        "bound_files": {
            "experiments/train/metric.py": (
                "39745215e9eeebf735b9ce23b1e8f0052ae19d1a81019cf60a25d52eb7cbf991"
            ),
            "experiments/train/metric_eval.py": (
                "240168a5ab80c872fdf424b7efc5d53893ae442f15496a2dcf3d0681186d1716"
            ),
            "experiments/train/eval.py": (
                "80a95f1b477bc3852f08d5bd33cc13f33d5152f2798f60c572716d441587c606"
            ),
        },
    },
}

PUBLISHED_3D_REFERENCE_SCORES: dict[str, dict[str, dict[str, float]]] = {
    "per_episode": {
        "PGND": {"future_cd": 0.073, "future_track_error": 0.073},
        "ParticleFormer": {"future_cd": 0.044, "future_track_error": 0.041},
        "PhysTwin": {"future_cd": 0.014, "future_track_error": 0.025},
    },
    "multi_episode": {
        "PGND": {"future_cd": 0.130, "future_track_error": 0.144},
        "ParticleFormer": {"future_cd": 0.051, "future_track_error": 0.079},
    },
    "multi_object": {
        "PGND": {"future_cd": 0.429, "future_track_error": 0.320},
        "ParticleFormer": {"future_cd": 0.038, "future_track_error": 0.048},
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = deepcopy(dict(payload))
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field(
    status: str,
    value: Any,
    *,
    source_id: str | None,
    locator: str | None,
    note: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "value": value,
        "source_id": source_id,
        "locator": locator,
        "note": note,
    }


def _missing(note: str) -> dict[str, Any]:
    return _field("missing", None, source_id=None, locator=None, note=note)


def _validate_source(source_id: str, source: Mapping[str, Any]) -> None:
    _require(bool(source_id), "source id is empty")
    _require(isinstance(source.get("kind"), str), f"{source_id} kind is missing")
    _require(
        isinstance(source.get("authority"), str),
        f"{source_id} authority is missing",
    )
    _require(isinstance(source.get("url"), str), f"{source_id} URL is missing")
    revision = source.get("revision")
    content_sha256 = source.get("content_sha256")
    _require(
        (isinstance(revision, str) and bool(revision))
        or (isinstance(content_sha256, str) and bool(_HEX64.fullmatch(content_sha256))),
        f"{source_id} has no immutable revision or content digest",
    )
    if isinstance(revision, str) and len(revision) == 40:
        _require(
            bool(_HEX40.fullmatch(revision)),
            f"{source_id} Git revision is malformed",
        )
    if content_sha256 is not None:
        _require(
            isinstance(content_sha256, str) and bool(_HEX64.fullmatch(content_sha256)),
            f"{source_id} content digest is malformed",
        )
    bound_files = source.get("bound_files")
    _require(isinstance(bound_files, Mapping), f"{source_id} bound files are missing")
    for name, digest in bound_files.items():
        _require(isinstance(name, str) and bool(name), "bound source filename is empty")
        _require(
            isinstance(digest, str) and bool(_HEX64.fullmatch(digest)),
            f"{source_id} bound file digest is malformed",
        )


def _allowed_authoritative_source_kinds(field_name: str) -> frozenset[str]:
    if field_name == "benchmark_setting":
        return EXACT_NUMERIC_SOURCE_KINDS | frozenset({"official_paper"})
    return EXACT_NUMERIC_SOURCE_KINDS


def seal_parity_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical JSON-compatible copy with a tamper-evident digest."""

    sealed = json.loads(json.dumps(contract, allow_nan=False))
    sealed["contract_sha256"] = _canonical_sha256(sealed, digest_key="contract_sha256")
    return sealed


def audit_parity_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one contract and report whether an official claim is allowed."""

    _require(contract.get("schema_version") == SCHEMA_VERSION, "wrong schema version")
    _require(contract.get("artifact_kind") == ARTIFACT_KIND, "wrong artifact kind")
    setting = contract.get("benchmark_setting")
    _require(setting in BENCHMARK_SETTINGS, "unknown benchmark setting")
    _require(
        contract.get("contract_sha256")
        == _canonical_sha256(contract, digest_key="contract_sha256"),
        "contract checksum changed",
    )

    sources = contract.get("sources")
    fields = contract.get("fields")
    _require(isinstance(sources, Mapping) and bool(sources), "sources are missing")
    _require(isinstance(fields, Mapping), "fields are missing")
    _require(
        set(fields) == set(REQUIRED_3D_FIELDS),
        "required parity fields changed",
    )
    for source_id, source in sources.items():
        _require(isinstance(source, Mapping), f"{source_id} source is malformed")
        _validate_source(str(source_id), source)

    by_status: dict[str, list[str]] = {status: [] for status in FIELD_STATUSES}
    for field_name in REQUIRED_3D_FIELDS:
        evidence = fields[field_name]
        _require(isinstance(evidence, Mapping), f"{field_name} evidence is malformed")
        _require(
            set(evidence) == {"status", "value", "source_id", "locator", "note"},
            f"{field_name} evidence schema changed",
        )
        status = evidence.get("status")
        _require(status in FIELD_STATUSES, f"{field_name} status is invalid")
        _require(
            isinstance(evidence.get("note"), str) and bool(evidence["note"]),
            f"{field_name} note is missing",
        )
        if status == "missing":
            _require(
                evidence.get("value") is None, f"{field_name} missing value exists"
            )
            _require(
                evidence.get("source_id") is None and evidence.get("locator") is None,
                f"{field_name} missing evidence claims a source",
            )
        else:
            source_id = evidence.get("source_id")
            _require(
                isinstance(source_id, str) and source_id in sources,
                f"{field_name} source is not bound",
            )
            _require(
                evidence.get("value") is not None, f"{field_name} value is missing"
            )
            _require(
                isinstance(evidence.get("locator"), str) and bool(evidence["locator"]),
                f"{field_name} locator is missing",
            )
            source_kind = str(sources[source_id]["kind"])
            if status == "authoritative":
                _require(
                    sources[source_id]["authority"] == "deform360_authoritative"
                    and source_kind in _allowed_authoritative_source_kinds(field_name),
                    f"{field_name} is not backed by an authoritative evaluator contract",
                )
        by_status[str(status)].append(field_name)

    parity_ready = not by_status["candidate"] and not by_status["missing"]
    request = [
        {
            "field": field_name,
            "current_status": "candidate",
            "request": (
                "Confirm this exact convention in a released evaluator or a "
                "content-hashed author contract."
            ),
        }
        for field_name in by_status["candidate"]
    ]
    request.extend(
        {
            "field": field_name,
            "current_status": "missing",
            "request": "Release or explicitly specify this field.",
        }
        for field_name in by_status["missing"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "Deform360Official3DParityContractAudit",
        "benchmark_setting": setting,
        "contract_sha256": contract["contract_sha256"],
        "parity_ready": parity_ready,
        "official_claim_allowed": parity_ready,
        "allowed_claim_label": (
            "official_deform360_3d_parity"
            if parity_ready
            else "candidate_convention_sensitivity_only"
        ),
        "field_counts": {status: len(names) for status, names in by_status.items()},
        "authoritative_fields": by_status["authoritative"],
        "candidate_fields": by_status["candidate"],
        "missing_fields": by_status["missing"],
        "information_request": request,
        "claim_boundary": (
            "Published aggregate values may be quoted as paper results, but local "
            "scores cannot be called official or directly SOTA-comparable until "
            "every required field is authoritative."
        ),
    }


def require_official_parity(contract: Mapping[str, Any]) -> None:
    """Raise unless the contract permits an official Deform360 3D claim."""

    audit = audit_parity_contract(contract)
    if not audit["parity_ready"]:
        unresolved = list(audit["candidate_fields"]) + list(audit["missing_fields"])
        raise ValueError(
            "official Deform360 parity is unresolved: " + ", ".join(unresolved)
        )


def _candidate_public_fields(setting: str) -> dict[str, dict[str, Any]]:
    _require(setting in BENCHMARK_SETTINGS, "unknown benchmark setting")
    fields = {
        name: _missing("Not specified exactly in the public release.")
        for name in REQUIRED_3D_FIELDS
    }
    fields["benchmark_setting"] = _field(
        "authoritative",
        setting,
        source_id="deform360_arxiv_v1",
        locator="main.tex lines 399-499",
        note="The paper defines per-episode, multi-episode, and multi-object settings.",
    )
    candidate_values = {
        "particle_identity_alignment": (
            "Prediction and downsampled ground truth use matching row indices."
        ),
        "prediction_preprocessing": (
            "Use the evaluator-provided predicted xyz array without an additional "
            "metric-space alignment."
        ),
        "ground_truth_preprocessing": (
            "Use xyz_gt_downsampled for track error and full xyz_gt for Chamfer "
            "unless camera-drop evaluation is enabled."
        ),
        "chamfer_direction": "prediction_to_ground_truth",
        "chamfer_point_distance": "euclidean",
        "chamfer_reduction": "mean_nearest_neighbor",
        "track_correspondence": "shared_particle_index",
        "track_point_distance": "squared_coordinate_residual",
        "track_reduction": "mean_over_points_and_xyz_coordinates",
        "frame_aggregation": "arithmetic_mean_over_evaluation_steps",
        "episode_aggregation": "episode_mean_then_arithmetic_mean_over_steps",
    }
    candidate_locators = {
        "particle_identity_alignment": "metric_eval.py lines 301-305",
        "prediction_preprocessing": "metric_eval.py lines 301-308",
        "ground_truth_preprocessing": "metric_eval.py lines 301-308",
        "chamfer_direction": "metric.py lines 59-64",
        "chamfer_point_distance": "metric.py lines 59-64",
        "chamfer_reduction": "metric.py lines 59-64",
        "track_correspondence": "metric.py lines 53-56",
        "track_point_distance": "metric.py lines 53-56",
        "track_reduction": "metric.py lines 53-56",
        "frame_aggregation": "eval.py lines 1043-1071",
        "episode_aggregation": "eval.py lines 1020-1071",
    }
    for field_name, value in candidate_values.items():
        fields[field_name] = _field(
            "candidate",
            value,
            source_id="pgrd_candidate_metric",
            locator=candidate_locators[field_name],
            note=(
                "This is an exact convention in the PGRD implementation, not an "
                "authoritative Deform360 evaluator contract."
            ),
        )
    official_pipeline_candidates = {
        "particle_identity_alignment": (
            "The released pcd stage seeds one fixed point set and advects it through "
            "the episode, preserving row identity.",
            "deform360/processing/pcd_stage.py lines 325-420",
        ),
        "ground_truth_preprocessing": (
            "The released PhysTwin bundle stacks active pcd_clean point sets inside "
            "the detected contact window into final_data.object_points.",
            "deform360/processing/control_points_stage.py lines 319-389",
        ),
        "future_frame_manifest": (
            "The released PhysTwin bundle writes train=[0,floor(0.8 F)] and "
            "test=[floor(0.8 F),F] for its detected contact window.",
            "deform360/processing/control_points_stage.py lines 391-401",
        ),
        "validity_visibility_mask": (
            "The released PhysTwin bundle writes all-true object_visibilities and "
            "object_motions_valid arrays.",
            "deform360/processing/control_points_stage.py lines 373-388",
        ),
        "coordinate_frame": (
            "Released depth lifting, tracking, point-cloud advection, and controller "
            "points use the calibrated episode world frame.",
            "deform360/processing/tracking_stage.py lines 163-204",
        ),
        "length_unit": (
            "Released depth, world points, robot translations, and controller "
            "geometry are expressed in metres.",
            "README.md lines 440-479",
        ),
    }
    for field_name, (value, locator) in official_pipeline_candidates.items():
        fields[field_name] = _field(
            "candidate",
            value,
            source_id="deform360_public_repo",
            locator=locator,
            note=(
                "This is authoritative for the released annotation/PhysTwin-bundle "
                "pipeline, but the repository explicitly omits the benchmark "
                "evaluator and does not bind this convention to the paper table."
            ),
        )
    fields["training_case_manifest"] = _missing(
        "The paper names a training subset but does not release exact case membership."
    )
    fields["evaluation_case_manifest"] = _missing(
        "The exact ordered object/episode evaluation cohort is not public."
    )
    fields["object_aggregation"] = _missing(
        "The object weighting used in the reported aggregate is not public."
    )
    fields["missing_case_policy"] = _missing(
        "The treatment of failed, missing, or unequal-length episodes is not public."
    )
    return fields


def build_public_parity_contract(setting: str) -> dict[str, Any]:
    """Build the strongest contract supported by currently public evidence."""

    contract = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "benchmark_setting": setting,
        "sources": deepcopy(PUBLIC_SOURCE_BINDINGS),
        "fields": _candidate_public_fields(setting),
    }
    return seal_parity_contract(contract)


def _as_points(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    _require(
        array.ndim == 2 and array.shape[1] == 3 and len(array) > 0,
        f"{label} must have shape (N,3)",
    )
    _require(np.all(np.isfinite(array)), f"{label} contains non-finite values")
    return array


def _nearest_squared(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    result = np.empty(len(query), dtype=float)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        delta = query[start:stop, None, :] - reference[None, :, :]
        result[start:stop] = np.min(np.sum(delta * delta, axis=2), axis=1)
    return result


def candidate_chamfer_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Return common Chamfer variants without designating one as official."""

    prediction = _as_points(predicted, label="predicted points")
    ground_truth = _as_points(target, label="target points")
    pred_to_target_sq = _nearest_squared(prediction, ground_truth)
    target_to_pred_sq = _nearest_squared(ground_truth, prediction)
    pred_to_target = np.sqrt(pred_to_target_sq)
    target_to_pred = np.sqrt(target_to_pred_sq)
    return {
        "pred_to_target_mean_euclidean_m": float(np.mean(pred_to_target)),
        "target_to_pred_mean_euclidean_m": float(np.mean(target_to_pred)),
        "symmetric_mean_euclidean_m": float(
            0.5 * (np.mean(pred_to_target) + np.mean(target_to_pred))
        ),
        "pred_to_target_mean_squared_m2": float(np.mean(pred_to_target_sq)),
        "target_to_pred_mean_squared_m2": float(np.mean(target_to_pred_sq)),
        "symmetric_mean_squared_m2": float(
            0.5 * (np.mean(pred_to_target_sq) + np.mean(target_to_pred_sq))
        ),
    }


def candidate_track_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return plausible identity-track reductions with explicit units."""

    prediction = np.asarray(predicted, dtype=float)
    ground_truth = np.asarray(target, dtype=float)
    _require(
        prediction.ndim == 2 and prediction.shape[1] == 3,
        "predicted tracks must have shape (N,3)",
    )
    _require(
        ground_truth.shape == prediction.shape,
        "target tracks must share predicted track identities",
    )
    if valid_mask is None:
        mask = np.ones(len(prediction), dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        _require(mask.shape == (len(prediction),), "valid mask must have shape (N,)")
    _require(np.any(mask), "valid mask selects no tracks")
    prediction = prediction[mask]
    ground_truth = ground_truth[mask]
    _require(
        np.all(np.isfinite(prediction)) and np.all(np.isfinite(ground_truth)),
        "selected tracks contain non-finite values",
    )
    squared = (prediction - ground_truth) ** 2
    point_squared = np.sum(squared, axis=1)
    return {
        "coordinate_mse_m2": float(np.mean(squared)),
        "coordinate_rmse_m": float(np.sqrt(np.mean(squared))),
        "mean_point_euclidean_m": float(np.mean(np.sqrt(point_squared))),
        "point_rmse_m": float(np.sqrt(np.mean(point_squared))),
    }


def aggregate_metric_sensitivity(
    case_frames: Mapping[str, Sequence[float] | np.ndarray],
    case_to_object: Mapping[str, str],
) -> dict[str, float]:
    """Expose frame-, episode-, and object-balanced aggregation differences."""

    _require(bool(case_frames), "case metrics are empty")
    _require(set(case_frames) == set(case_to_object), "case/object mapping changed")
    normalized: dict[str, np.ndarray] = {}
    for case, values in case_frames.items():
        array = np.asarray(values, dtype=float)
        _require(
            array.ndim == 1 and len(array) > 0 and np.all(np.isfinite(array)),
            f"{case} frame metrics are invalid",
        )
        normalized[str(case)] = array
    episode_means = {
        case: float(np.mean(values)) for case, values in normalized.items()
    }
    object_means = []
    for object_id in sorted(set(case_to_object.values())):
        cases = [
            case
            for case, assigned_object in case_to_object.items()
            if assigned_object == object_id
        ]
        _require(bool(cases), f"{object_id} has no episodes")
        object_means.append(float(np.mean([episode_means[case] for case in cases])))
    return {
        "frame_pooled_mean": float(
            np.mean(np.concatenate([normalized[case] for case in sorted(normalized)]))
        ),
        "episode_balanced_mean": float(np.mean(list(episode_means.values()))),
        "object_balanced_mean": float(np.mean(object_means)),
    }


def build_public_parity_audit() -> dict[str, Any]:
    """Return a sealed public-evidence audit and deterministic ambiguity example."""

    contracts = {
        setting: build_public_parity_contract(setting) for setting in BENCHMARK_SETTINGS
    }
    audits = {
        setting: audit_parity_contract(contract)
        for setting, contract in contracts.items()
    }
    prediction = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": AUDIT_KIND,
        "public_sources": deepcopy(PUBLIC_SOURCE_BINDINGS),
        "published_reference_scores": deepcopy(PUBLISHED_3D_REFERENCE_SCORES),
        "contracts": contracts,
        "audits": audits,
        "metric_ambiguity_example": {
            "prediction_m": prediction.tolist(),
            "target_m": target.tolist(),
            "chamfer": candidate_chamfer_metrics(prediction, target),
            "track": candidate_track_metrics(prediction, target),
        },
        "aggregation_ambiguity_example": aggregate_metric_sensitivity(
            {
                "object-a-episode-1": np.zeros(10),
                "object-a-episode-2": np.zeros(1),
                "object-b-episode-1": np.asarray([12.0]),
            },
            {
                "object-a-episode-1": "object-a",
                "object-a-episode-2": "object-a",
                "object-b-episode-1": "object-b",
            },
        ),
        "conclusion": (
            "No public contract is parity-ready. The published table values are "
            "reference results; local values remain candidate-convention "
            "sensitivity results until the missing evaluator contract is supplied."
        ),
    }
    report["report_sha256"] = _canonical_sha256(report, digest_key="report_sha256")
    return report


def write_parity_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
