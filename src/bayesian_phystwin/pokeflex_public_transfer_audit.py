"""Retrospective public-action transfer audit for the frozen PokeFlex update."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .pokeflex_action_robust_all18 import (
    SOURCE_FIELD,
    validate_all18_calibration,
)
from .pokeflex_action_robust_final_freshness import (
    validate_final_freshness_audit,
)
from .pokeflex_instance_freshness import public_take_ids

PROTOCOL_KIND = "PokeFlexActionRobustPublicTransferAuditProtocol"
PROTOCOL_ID = "pokeflex-action-robust-public78-retrospective-v6"
PROTOCOL_SHA256 = (
    "f108baede896f32ee7150efc7dd2fe54fb51bfe374cc5e4e97f4969dca381eec"
)
RESULT_KIND = "PokeFlexActionRobustPublicTransferAuditResult"
SMOKE_KIND = "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke"
BASE_EFFECTIVE_SCALE = 0.125
GLOBAL_MULTIPLIER = 1.0
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_720
AUDIT_RUNNER_FILE_SHA256 = (
    "75ce66d0d12620ff1d0eaf98787af6502cc1dfd0af4bb4b5dd7ed28653a823e5"
)
SOURCE_PROJECTION_RUNNER_FILE_SHA256 = (
    "08157ac8d232d0118ef8d29b6661c1855339c8ec76b2b37cf050e0438812abc5"
)
LEGACY_RUNNER_FILE_SHA256 = (
    "79ba8946653a55a70dc0b990e874754397e18948b9b7ba541158c6641cfc4b43"
)
REGISTRATION_PROTOCOL_FILE_SHA256 = (
    "397dbe38abc91b901cc08849f16ef477af83ec3ef7c8b86054dc3bef9433002c"
)
UPSTREAM_COMMIT = "aaa8726072834a95bbe97e1a113588968c36e185"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _inventory_sha256(values: Sequence[str] | set[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("ascii")
    ).hexdigest()


def _object_name(take_id: str) -> str:
    object_name, separator, number = take_id.rpartition("_T")
    _require(bool(separator) and number.isdigit(), "invalid PokeFlex take id")
    return object_name


def protocol_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical protocol digest."""

    return _canonical_sha256(payload, "protocol_sha256")


def result_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical result digest."""

    return _canonical_sha256(payload, "result_sha256")


def file_sha256(path: str | Path) -> str:
    """Hash one file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_transfer_partitions(
    calibration: Mapping[str, Any],
    freshness: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return the source, prospective, and retrospective public partitions."""

    calibration_validation = validate_all18_calibration(calibration)
    freshness_validation = validate_final_freshness_audit(freshness)
    public = set(public_take_ids())
    source = {
        str(take_id)
        for row in calibration["objects"].values()
        for take_id in row["source_take_ids"]
    }
    prospective = set(freshness_validation["target_take_ids"])
    retrospective = public - source - prospective
    _require(len(public) == 116, "public PokeFlex inventory changed")
    _require(len(source) == 36, "source-action inventory changed")
    _require(len(prospective) == 2, "prospective inventory changed")
    _require(len(retrospective) == 78, "retrospective inventory changed")
    _require(not (source & prospective), "source and prospective actions overlap")
    _require(
        set(calibration_validation["multipliers"])
        == {_object_name(take_id) for take_id in public},
        "all-object multiplier map is incomplete",
    )
    prior = set(freshness["prior_exposure_audit"]["take_ids"])
    _require(
        prior == source | retrospective,
        "retrospective cohort is not the registered previously exposed complement",
    )
    return {
        "public": tuple(sorted(public)),
        "source": tuple(sorted(source)),
        "prospective": tuple(sorted(prospective)),
        "retrospective": tuple(sorted(retrospective)),
    }


def _validate_archive_inventory(
    archive_inventory: Mapping[str, Any],
    expected_take_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    _require(
        int(archive_inventory.get("zip_count", -1)) == len(expected_take_ids),
        "retrospective ZIP inventory count changed",
    )
    takes = archive_inventory.get("takes")
    _require(isinstance(takes, Mapping), "retrospective ZIP inventory is missing")
    _require(set(takes) == set(expected_take_ids), "retrospective ZIP inventory changed")
    selected: dict[str, dict[str, Any]] = {}
    for take_id in expected_take_ids:
        row = takes.get(take_id)
        _require(isinstance(row, Mapping), "retrospective ZIP row is missing")
        digest = str(row.get("sha256", ""))
        size = int(row.get("bytes", -1))
        relative = str(row.get("relative_path", ""))
        _require(
            len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
            "retrospective ZIP digest is invalid",
        )
        _require(size > 0, "retrospective ZIP size is invalid")
        _require(Path(relative).name == f"{take_id}.zip", "ZIP path changed take id")
        selected[take_id] = {
            "relative_path": relative,
            "sha256": digest,
            "bytes": size,
        }
    return selected


def build_public_transfer_protocol(
    calibration: Mapping[str, Any],
    freshness: Mapping[str, Any],
    archive_inventory: Mapping[str, Any],
    *,
    archive_inventory_file_sha256: str,
    locked_at_utc: str,
) -> dict[str, Any]:
    """Build the fixed 78-action retrospective audit protocol."""

    partitions = public_transfer_partitions(calibration, freshness)
    retrospective = partitions["retrospective"]
    archives = _validate_archive_inventory(archive_inventory, retrospective)
    _require(
        len(archive_inventory_file_sha256) == 64,
        "archive inventory file digest is invalid",
    )
    multipliers = {
        object_name: float(row["multiplier"])
        for object_name, row in calibration["objects"].items()
    }
    effective_scales = {
        object_name: BASE_EFFECTIVE_SCALE * multiplier
        for object_name, multiplier in multipliers.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PROTOCOL_KIND,
        "protocol_id": PROTOCOL_ID,
        "locked_at_utc": locked_at_utc,
        "status": "retrospective audit locked before the new 78-action rerun",
        "claim_boundary": (
            "The 78 outcomes were exposed before this method-specific rerun. The "
            "audit measures broad public-action transfer but is not a prospective "
            "confirmation, an official eighteen-take reproduction, or a direct "
            "state-of-the-art claim. The two v5 prospective actions are always "
            "reported separately before any combined 80-action summary."
        ),
        "cohort": {
            "public_take_count": len(partitions["public"]),
            "source_take_count": len(partitions["source"]),
            "prospective_take_count": len(partitions["prospective"]),
            "retrospective_take_count": len(retrospective),
            "source_take_ids": list(partitions["source"]),
            "prospective_take_ids": list(partitions["prospective"]),
            "retrospective_take_ids": list(retrospective),
            "retrospective_inventory_sha256": _inventory_sha256(retrospective),
            "selection_rule": (
                "all 116 public poking takes minus the 36 all18 source actions and "
                "the two prospectively sealed v5 actions"
            ),
            "replacement_allowed": False,
        },
        "source_calibration": {
            "path": "configs/sota/pokeflex_action_robust_scale_all18_v4.json",
            "calibration_sha256": calibration["calibration_sha256"],
            "calibration_file_sha256": (
                "00cdf5732f5dbf7eb0f899ebbb536260d9e66c0a151b41e4a784cbe4aaf110"
            ),
            "field": SOURCE_FIELD,
            "base_effective_scale": BASE_EFFECTIVE_SCALE,
            "global_multiplier": GLOBAL_MULTIPLIER,
            "multipliers": dict(sorted(multipliers.items())),
            "effective_scales": dict(sorted(effective_scales.items())),
        },
        "prospective_reference": {
            "protocol_sha256": (
                "6b47542ba5ede7a1f04e35f4d453eeb55b8394a0b935018dc6dcf9a9c0174724"
            ),
            "summary_sha256": (
                "da35b35fa2b6bbf13f079be5fa6b5705320ac351dbb8cfab06d7a27ec9687112"
            ),
            "strict_advancement_over_global_passed": False,
            "retuning_from_these_outcomes": False,
        },
        "archive_inventory": {
            "source_host": "gpuserver6000",
            "source_root": archive_inventory.get("root"),
            "inventory_file_sha256": archive_inventory_file_sha256,
            "selected_total_bytes": sum(row["bytes"] for row in archives.values()),
            "takes": dict(sorted(archives.items())),
        },
        "implementation": {
            "source_projection_runner": (
                "scripts/remote/stage_pokeflex_public_transfer_archive.py"
            ),
            "source_projection_runner_file_sha256": (
                SOURCE_PROJECTION_RUNNER_FILE_SHA256
            ),
            "audit_runner": (
                "scripts/remote/run_pokeflex_public_transfer_audit_take.py"
            ),
            "audit_runner_file_sha256": AUDIT_RUNNER_FILE_SHA256,
            "legacy_runner": (
                "scripts/remote/run_pokeflex_checkpoint_registration_smoke.py"
            ),
            "legacy_runner_file_sha256": LEGACY_RUNNER_FILE_SHA256,
            "registration_protocol": (
                "configs/sota/pokeflex_bayesian_registration_v1.json"
            ),
            "registration_protocol_file_sha256": (
                REGISTRATION_PROTOCOL_FILE_SHA256
            ),
            "upstream_commit": UPSTREAM_COMMIT,
            "future_observation_used": False,
            "per_take_scales": (
                "released checkpoint, global 0.125, and the frozen object-specific "
                "effective scale; duplicate scales are evaluated once"
            ),
        },
        "evaluation": {
            "primary_metric": "CD_UL1_mm",
            "surface_sample_count": 10_000,
            "surface_sample_seed": BOOTSTRAP_SEED,
            "summaries": [
                "retrospective 78 actions",
                "prospective two actions",
                "combined 80 non-source actions",
            ],
            "aggregation": [
                "object-balanced",
                "action-balanced",
                "frame-balanced",
            ],
            "uncertainty": (
                "paired object-cluster bootstrap with 20000 replicates and fixed "
                "seed 20260720"
            ),
            "references": ["released checkpoint", "global effective scale 0.125"],
        },
        "interpretation_gate": {
            "role": "justify a genuinely fresh baseline-relative guarded evaluation",
            "required_for_each_reference": [
                "positive object-balanced relative improvement",
                "97.5% object-cluster bootstrap upper bound on candidate-minus-reference below zero",
                "at least 12 of 18 object means improve",
                "worst object relative regression no larger than 1%",
            ],
            "does_not_override_v5_prospective_failure": True,
        },
        "forbidden": [
            "retuning any object multiplier from these 78 outcomes",
            "mixing incompatible historical smoke artifacts into the audit",
            "combining the prospective two and retrospective 78 without reporting both first",
            "claiming an official PokeFlex eighteen-take comparison without author mapping",
            "claiming prospective confirmation from the 78 previously exposed actions",
            "using frame f or later Kinect observations to predict frame f",
            "routing server-to-server payloads through the jump server",
            "touching any held-v8 artifact or process",
        ],
    }
    payload["protocol_sha256"] = protocol_sha256(payload)
    validate_public_transfer_protocol(payload, bind_registered_digest=False)
    return payload


def validate_public_transfer_protocol(
    payload: Mapping[str, Any],
    *,
    bind_registered_digest: bool = True,
) -> dict[str, Any]:
    """Validate the immutable cohort, method, and evidence boundary."""

    _require(payload.get("schema_version") == 1, "protocol schema changed")
    _require(payload.get("artifact_kind") == PROTOCOL_KIND, "protocol kind changed")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol id changed")
    _require(
        payload.get("protocol_sha256") == protocol_sha256(payload),
        "protocol checksum mismatch",
    )
    if bind_registered_digest:
        _require(payload.get("protocol_sha256") == PROTOCOL_SHA256, "protocol changed")
    cohort = payload.get("cohort")
    _require(isinstance(cohort, Mapping), "cohort is missing")
    retrospective = tuple(str(x) for x in cohort.get("retrospective_take_ids", ()))
    source = tuple(str(x) for x in cohort.get("source_take_ids", ()))
    prospective = tuple(str(x) for x in cohort.get("prospective_take_ids", ()))
    _require(len(retrospective) == len(set(retrospective)) == 78, "retrospective cohort changed")
    _require(len(source) == len(set(source)) == 36, "source cohort changed")
    _require(len(prospective) == len(set(prospective)) == 2, "prospective cohort changed")
    _require(
        set(retrospective) | set(source) | set(prospective) == set(public_take_ids()),
        "public partition changed",
    )
    _require(
        not (set(retrospective) & set(source))
        and not (set(retrospective) & set(prospective))
        and not (set(source) & set(prospective)),
        "public partitions overlap",
    )
    _require(
        cohort.get("retrospective_inventory_sha256")
        == _inventory_sha256(retrospective),
        "retrospective inventory digest changed",
    )
    source_calibration = payload.get("source_calibration")
    _require(isinstance(source_calibration, Mapping), "source calibration is missing")
    _require(source_calibration.get("field") == SOURCE_FIELD, "correction field changed")
    _require(
        float(source_calibration.get("base_effective_scale", -1.0))
        == BASE_EFFECTIVE_SCALE,
        "base scale changed",
    )
    multipliers = source_calibration.get("multipliers")
    scales = source_calibration.get("effective_scales")
    _require(isinstance(multipliers, Mapping), "multiplier map is missing")
    _require(isinstance(scales, Mapping), "effective scale map is missing")
    _require(len(multipliers) == len(scales) == 18, "all-object scale map changed")
    for object_name, multiplier in multipliers.items():
        _require(
            float(scales[object_name]) == BASE_EFFECTIVE_SCALE * float(multiplier),
            "effective scale is inconsistent",
        )
    implementation = payload.get("implementation")
    _require(isinstance(implementation, Mapping), "implementation binding is missing")
    _require(
        implementation.get("source_projection_runner_file_sha256")
        == SOURCE_PROJECTION_RUNNER_FILE_SHA256,
        "source projection runner changed",
    )
    _require(
        implementation.get("audit_runner_file_sha256") == AUDIT_RUNNER_FILE_SHA256,
        "audit runner changed",
    )
    _require(
        implementation.get("legacy_runner_file_sha256")
        == LEGACY_RUNNER_FILE_SHA256,
        "legacy runner changed",
    )
    _require(
        implementation.get("registration_protocol_file_sha256")
        == REGISTRATION_PROTOCOL_FILE_SHA256,
        "registration protocol changed",
    )
    _require(
        implementation.get("upstream_commit") == UPSTREAM_COMMIT,
        "upstream commit changed",
    )
    archives = payload.get("archive_inventory")
    _require(isinstance(archives, Mapping), "archive inventory is missing")
    archive_rows = archives.get("takes")
    _require(isinstance(archive_rows, Mapping), "archive rows are missing")
    _require(set(archive_rows) == set(retrospective), "archive cohort changed")
    for take_id, row in archive_rows.items():
        _require(isinstance(row, Mapping), "archive row is invalid")
        _require(Path(str(row.get("relative_path", ""))).name == f"{take_id}.zip", "archive path changed")
        digest = str(row.get("sha256", ""))
        _require(len(digest) == 64, "archive digest is invalid")
        _require(int(row.get("bytes", -1)) > 0, "archive size is invalid")
    return {
        "passed": True,
        "protocol_sha256": payload["protocol_sha256"],
        "retrospective_take_ids": retrospective,
        "source_take_ids": source,
        "prospective_take_ids": prospective,
        "multipliers": {str(k): float(v) for k, v in multipliers.items()},
        "effective_scales": {str(k): float(v) for k, v in scales.items()},
    }


def _scale_key(scale: float) -> str:
    return f"checkpoint_{SOURCE_FIELD}_residual_scale_{scale:g}"


def take_row_from_smoke(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    artifact_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Extract one fixed-scale retrospective row from a new smoke artifact."""

    validation = validate_public_transfer_protocol(protocol)
    _require(payload.get("artifact_kind") == SMOKE_KIND, "smoke kind changed")
    _require(payload.get("future_observation_used") is False, "future observation leaked")
    _require(
        payload.get("public_transfer_protocol_sha256") == protocol["protocol_sha256"],
        "smoke protocol changed",
    )
    _require(
        payload.get("legacy_runner_file_sha256") == LEGACY_RUNNER_FILE_SHA256,
        "smoke runner changed",
    )
    take = payload.get("take")
    _require(isinstance(take, Mapping), "smoke take metadata is missing")
    take_id = str(take.get("id", ""))
    _require(
        take_id in validation["retrospective_take_ids"],
        "smoke take is outside the retrospective cohort",
    )
    _require(
        payload.get("upstream", {}).get("git_commit") == UPSTREAM_COMMIT,
        "smoke upstream changed",
    )
    _require(SOURCE_FIELD in payload.get("correction_fields", ()), "smoke field changed")
    object_name = _object_name(take_id)
    candidate_scale = validation["effective_scales"][object_name]
    aggregates = payload.get("aggregates")
    _require(isinstance(aggregates, Mapping), "smoke aggregates are missing")
    required = {
        "checkpoint": "released_checkpoint",
        "global": _scale_key(BASE_EFFECTIVE_SCALE),
        "candidate": _scale_key(candidate_scale),
    }
    values: dict[str, float] = {}
    for label, key in required.items():
        row = aggregates.get(key)
        _require(isinstance(row, Mapping), f"{label} aggregate is missing")
        value = float(row.get("mean_CD_UL1_mm", np.nan))
        _require(np.isfinite(value) and value > 0.0, f"{label} score is invalid")
        values[label] = value
    targets = payload.get("targets")
    _require(isinstance(targets, list) and targets, "smoke target rows are missing")
    frames = []
    for target in targets:
        _require(isinstance(target, Mapping), "smoke target row is invalid")
        frame = {
            "target_frame": int(target["target_frame"]),
            "checkpoint_CD_UL1_mm": float(target["released_checkpoint_CD_UL1_mm"]),
            "global_CD_UL1_mm": float(target[required["global"]]),
            "candidate_CD_UL1_mm": float(target[required["candidate"]]),
        }
        _require(
            all(np.isfinite(value) and value > 0.0 for key, value in frame.items() if key != "target_frame"),
            "smoke frame score is invalid",
        )
        frames.append(frame)
    for label, key in (
        ("checkpoint", "checkpoint_CD_UL1_mm"),
        ("global", "global_CD_UL1_mm"),
        ("candidate", "candidate_CD_UL1_mm"),
    ):
        _require(
            np.isclose(values[label], np.mean([row[key] for row in frames]), atol=1e-12),
            f"{label} aggregate does not match frame rows",
        )
    row = {
        "take_id": take_id,
        "object_name": object_name,
        "candidate_multiplier": validation["multipliers"][object_name],
        "candidate_effective_scale": candidate_scale,
        "scored_frame_count": len(frames),
        "checkpoint_CD_UL1_mm": values["checkpoint"],
        "global_CD_UL1_mm": values["global"],
        "candidate_CD_UL1_mm": values["candidate"],
        "frames": frames,
    }
    if artifact_file_sha256 is not None:
        _require(len(artifact_file_sha256) == 64, "artifact digest is invalid")
        row["artifact_file_sha256"] = artifact_file_sha256
    return row


def _prospective_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(payload.get("target_meshes_opened_after_complete_barrier") is True, "prospective barrier evidence is missing")
    rows = []
    for raw in payload.get("objects", ()):
        frames = [
            {
                "target_frame": int(frame["target_frame"]),
                "checkpoint_CD_UL1_mm": float(frame["baseline_CD_UL1_mm"]),
                "global_CD_UL1_mm": float(frame["global_candidate_CD_UL1_mm"]),
                "candidate_CD_UL1_mm": float(frame["candidate_CD_UL1_mm"]),
            }
            for frame in raw["frames"]
        ]
        rows.append(
            {
                "take_id": str(raw["take_id"]),
                "object_name": str(raw["object_name"]),
                "candidate_effective_scale": None,
                "scored_frame_count": len(frames),
                "checkpoint_CD_UL1_mm": float(raw["baseline_mean_CD_UL1_mm"]),
                "global_CD_UL1_mm": float(raw["global_candidate_mean_CD_UL1_mm"]),
                "candidate_CD_UL1_mm": float(raw["candidate_mean_CD_UL1_mm"]),
                "frames": frames,
            }
        )
    _require(len(rows) == 2, "prospective row count changed")
    return rows


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_key: str,
) -> dict[str, Any]:
    by_object: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["object_name"])].append(row)
    object_names = sorted(by_object)
    candidate_object = np.asarray(
        [np.mean([float(row["candidate_CD_UL1_mm"]) for row in by_object[name]]) for name in object_names],
        dtype=np.float64,
    )
    reference_object = np.asarray(
        [np.mean([float(row[reference_key]) for row in by_object[name]]) for name in object_names],
        dtype=np.float64,
    )
    differences = candidate_object - reference_object
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(object_names),
        size=(BOOTSTRAP_REPLICATES, len(object_names)),
    )
    boot = np.mean(differences[indices], axis=1)
    take_differences = np.asarray(
        [float(row["candidate_CD_UL1_mm"]) - float(row[reference_key]) for row in rows],
        dtype=np.float64,
    )
    tolerance = 1e-12
    relative_by_object = (reference_object - candidate_object) / reference_object
    return {
        "reference": reference_key.removesuffix("_CD_UL1_mm"),
        "object_count": len(object_names),
        "take_count": len(rows),
        "object_balanced_reference_CD_UL1_mm": float(np.mean(reference_object)),
        "object_balanced_candidate_CD_UL1_mm": float(np.mean(candidate_object)),
        "object_balanced_relative_improvement": float(
            (np.mean(reference_object) - np.mean(candidate_object))
            / np.mean(reference_object)
        ),
        "object_win_count": int(np.sum(differences < -tolerance)),
        "object_tie_count": int(np.sum(np.abs(differences) <= tolerance)),
        "object_loss_count": int(np.sum(differences > tolerance)),
        "take_win_count": int(np.sum(take_differences < -tolerance)),
        "take_tie_count": int(np.sum(np.abs(take_differences) <= tolerance)),
        "take_loss_count": int(np.sum(take_differences > tolerance)),
        "minimum_object_relative_improvement": float(np.min(relative_by_object)),
        "bootstrap_lower_candidate_minus_reference_CD_UL1_mm": float(
            np.quantile(boot, 0.025)
        ),
        "bootstrap_upper_candidate_minus_reference_CD_UL1_mm": float(
            np.quantile(boot, 0.975)
        ),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute action-, object-, and frame-balanced paired summaries."""

    _require(bool(rows), "cannot summarize an empty cohort")
    by_object: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["object_name"])].append(row)
    method_keys = ("checkpoint_CD_UL1_mm", "global_CD_UL1_mm", "candidate_CD_UL1_mm")
    object_balanced = {
        key: float(
            np.mean(
                [
                    np.mean([float(row[key]) for row in object_rows])
                    for object_rows in by_object.values()
                ]
            )
        )
        for key in method_keys
    }
    action_balanced = {
        key: float(np.mean([float(row[key]) for row in rows])) for key in method_keys
    }
    all_frames = [frame for row in rows for frame in row["frames"]]
    frame_balanced = {
        key: float(np.mean([float(frame[key]) for frame in all_frames]))
        for key in method_keys
    }
    return {
        "object_count": len(by_object),
        "take_count": len(rows),
        "frame_count": len(all_frames),
        "object_balanced": object_balanced,
        "action_balanced": action_balanced,
        "frame_balanced": frame_balanced,
        "candidate_vs_checkpoint": _comparison(
            rows,
            reference_key="checkpoint_CD_UL1_mm",
        ),
        "candidate_vs_global": _comparison(
            rows,
            reference_key="global_CD_UL1_mm",
        ),
    }


def _interpretation_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for name in ("candidate_vs_checkpoint", "candidate_vs_global"):
        comparison = summary[name]
        checks = {
            "positive_object_balanced_improvement": (
                float(comparison["object_balanced_relative_improvement"]) > 0.0
            ),
            "cluster_bootstrap_upper_below_zero": (
                float(comparison["bootstrap_upper_candidate_minus_reference_CD_UL1_mm"])
                < 0.0
            ),
            "at_least_twelve_object_wins": int(comparison["object_win_count"]) >= 12,
            "worst_object_regression_within_one_percent": (
                float(comparison["minimum_object_relative_improvement"]) >= -0.01
            ),
        }
        rows[name] = {"checks": checks, "passed": all(checks.values())}
    return {
        "comparisons": rows,
        "passed": all(row["passed"] for row in rows.values()),
        "authorizes_retuning": False,
        "authorizes_sota_claim": False,
        "role": "evidence for or against a genuinely fresh guarded-update evaluation",
    }


def build_public_transfer_result(
    protocol: Mapping[str, Any],
    smoke_artifacts: Mapping[str, Mapping[str, Any]],
    *,
    smoke_artifact_file_sha256s: Mapping[str, str],
    prospective_target_result: Mapping[str, Any],
    prospective_target_result_file_sha256: str,
) -> dict[str, Any]:
    """Aggregate the retrospective 78 while preserving the prospective boundary."""

    validation = validate_public_transfer_protocol(protocol)
    expected = set(validation["retrospective_take_ids"])
    _require(set(smoke_artifacts) == expected, "smoke artifact inventory changed")
    _require(set(smoke_artifact_file_sha256s) == expected, "smoke digest inventory changed")
    retrospective_rows = [
        take_row_from_smoke(
            smoke_artifacts[take_id],
            protocol,
            artifact_file_sha256=smoke_artifact_file_sha256s[take_id],
        )
        for take_id in validation["retrospective_take_ids"]
    ]
    prospective_rows = _prospective_rows(prospective_target_result)
    _require(
        {row["take_id"] for row in prospective_rows}
        == set(validation["prospective_take_ids"]),
        "prospective take inventory changed",
    )
    combined_rows = retrospective_rows + prospective_rows
    retrospective_summary = summarize_rows(retrospective_rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "protocol_sha256": protocol["protocol_sha256"],
        "evidence_order": [
            "retrospective 78-action audit",
            "previously sealed prospective two-action result",
            "combined 80 non-source actions",
        ],
        "retrospective": {
            "classification": "previously exposed outcomes",
            "rows": retrospective_rows,
            "summary": retrospective_summary,
            "interpretation_gate": _interpretation_gate(retrospective_summary),
        },
        "prospective": {
            "classification": "sealed before target access under v5",
            "target_result_file_sha256": prospective_target_result_file_sha256,
            "rows": prospective_rows,
            "summary": summarize_rows(prospective_rows),
            "strict_advancement_over_global_passed": False,
        },
        "combined_non_source": {
            "classification": "descriptive combination; mixed evidence status",
            "rows": combined_rows,
            "summary": summarize_rows(combined_rows),
        },
        "decision": {
            "retrospective_interpretation_gate_passed": _interpretation_gate(
                retrospective_summary
            )["passed"],
            "prospective_v5_strict_advancement_passed": False,
            "current_method_establishes_strict_superiority_over_global": False,
            "retuning_from_public_outcomes_authorized": False,
            "independent_fresh_evaluation_required_for_advancement": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["result_sha256"] = result_sha256(payload)
    return payload


__all__ = [
    "BASE_EFFECTIVE_SCALE",
    "AUDIT_RUNNER_FILE_SHA256",
    "LEGACY_RUNNER_FILE_SHA256",
    "PROTOCOL_ID",
    "PROTOCOL_KIND",
    "PROTOCOL_SHA256",
    "RESULT_KIND",
    "SOURCE_PROJECTION_RUNNER_FILE_SHA256",
    "build_public_transfer_protocol",
    "build_public_transfer_result",
    "file_sha256",
    "protocol_sha256",
    "public_transfer_partitions",
    "result_sha256",
    "summarize_rows",
    "take_row_from_smoke",
    "validate_public_transfer_protocol",
]
