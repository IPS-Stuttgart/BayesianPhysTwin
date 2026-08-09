"""Cross-action scale calibration for the five unavailable PokeFlex targets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .pokeflex_action_robust_all18 import SOURCE_FIELD

PROTOCOL_KIND = "PokeFlexMissingFiveScaleSourceProtocol"
PROTOCOL_ID = "pokeflex-missing5-scale-source-v5"
RESULT_KIND = "PokeFlexMissingFiveScaleSourceResult"
SMOKE_KIND = "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke"
BASE_EFFECTIVE_SCALE = 0.125
GLOBAL_MULTIPLIER = 1.0
CANDIDATE_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
UPSTREAM_COMMIT = "aaa8726072834a95bbe97e1a113588968c36e185"

OFFICIAL_TARGET_TAKES = {
    "3dPrintedCylinder": "3dPrintedCylinder_T7",
    "3dPrintedHeart": "3dPrintedHeart_T14",
    "3dPrintedPizza": "3dPrintedPizza_T13",
    "Pillow": "Pillow_T8",
    "Sponge": "Sponge_T10",
}

SOURCE_TAKES = {
    "3dPrintedCylinder": tuple(f"3dPrintedCylinder_T{index}" for index in range(1, 7)),
    "3dPrintedHeart": tuple(f"3dPrintedHeart_T{index}" for index in range(1, 7)),
    "3dPrintedPizza": tuple(f"3dPrintedPizza_T{index}" for index in range(1, 7)),
    "Pillow": tuple(f"Pillow_T{index}" for index in range(1, 8)),
    "Sponge": tuple(f"Sponge_T{index}" for index in range(1, 6)),
}


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


def file_sha256(path: str | Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical protocol digest."""

    return _canonical_sha256(payload, "protocol_sha256")


def result_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical source-result digest."""

    return _canonical_sha256(payload, "result_sha256")


def object_name(take_id: str) -> str:
    """Return the physical-object identifier encoded in a take id."""

    name, separator, number = str(take_id).rpartition("_T")
    _require(bool(separator) and number.isdigit(), "invalid PokeFlex take id")
    return name


def source_take_ids() -> tuple[str, ...]:
    """Return the complete source inventory in deterministic order."""

    return tuple(sorted(take for takes in SOURCE_TAKES.values() for take in takes))


def _valid_digest(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _valid_git_revision(value: object) -> bool:
    text = str(value)
    return len(text) in (40, 64) and all(char in "0123456789abcdef" for char in text)


def build_source_protocol(
    archive_inventory: Mapping[str, Mapping[str, Any]],
    *,
    locked_at_utc: str,
    implementation_revision: str,
    source_projection_runner_file_sha256: str,
    source_runner_file_sha256: str,
    legacy_runner_file_sha256: str,
    registration_protocol_file_sha256: str,
) -> dict[str, Any]:
    """Build the source-only all-action scale protocol."""

    expected = source_take_ids()
    _require(
        set(archive_inventory) == set(expected), "source archive inventory changed"
    )
    archives: dict[str, dict[str, Any]] = {}
    for take_id in expected:
        row = archive_inventory[take_id]
        relative_path = str(row.get("relative_path", ""))
        digest = str(row.get("sha256", ""))
        byte_count = int(row.get("bytes", -1))
        _require(
            Path(relative_path).name == f"{take_id}.zip", "source archive path changed"
        )
        _require(_valid_digest(digest), "source archive digest is invalid")
        _require(byte_count > 0, "source archive byte count is invalid")
        archives[take_id] = {
            "relative_path": relative_path,
            "sha256": digest,
            "bytes": byte_count,
        }

    implementation_hashes = {
        "source_projection_runner_file_sha256": source_projection_runner_file_sha256,
        "source_runner_file_sha256": source_runner_file_sha256,
        "legacy_runner_file_sha256": legacy_runner_file_sha256,
        "registration_protocol_file_sha256": registration_protocol_file_sha256,
    }
    _require(
        all(_valid_digest(value) for value in implementation_hashes.values()),
        "implementation file digest is invalid",
    )
    _require(
        _valid_git_revision(implementation_revision),
        "implementation revision is invalid",
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PROTOCOL_KIND,
        "protocol_id": PROTOCOL_ID,
        "locked_at_utc": locked_at_utc,
        "status": "source-only scale bank locked before the six-scale rerun",
        "claim_boundary": (
            "All source outcomes were previously open. The protocol uses only public "
            "takes of the five objects whose exact official target takes remain "
            "unavailable. It cannot establish an official-target or state-of-the-art "
            "result; it may only freeze a candidate before those five targets arrive."
        ),
        "source_cohort": {
            "objects": list(SOURCE_TAKES),
            "object_count": len(SOURCE_TAKES),
            "take_count": len(expected),
            "takes_by_object": {
                name: list(takes) for name, takes in SOURCE_TAKES.items()
            },
            "official_target_takes": dict(OFFICIAL_TARGET_TAKES),
            "official_target_outcomes_used": False,
            "selection_rule": (
                "all public poking takes released for each object before its unavailable "
                "official validation take"
            ),
            "replacement_allowed": False,
        },
        "archive_inventory": {
            "source_host": "gpuserver6000",
            "source_root": "/mnt/lexar4tb/pokeflex/poking",
            "selected_total_bytes": sum(row["bytes"] for row in archives.values()),
            "takes": archives,
        },
        "method": {
            "field": SOURCE_FIELD,
            "base_effective_scale": BASE_EFFECTIVE_SCALE,
            "global_multiplier": GLOBAL_MULTIPLIER,
            "candidate_multipliers": list(CANDIDATE_MULTIPLIERS),
            "candidate_effective_scales": [
                BASE_EFFECTIVE_SCALE * value for value in CANDIDATE_MULTIPLIERS
            ],
            "selection": (
                "maximize the worst source-action gain over global, then mean gain; "
                "promote only when leave-one-action-out selections never regress their "
                "held action and win strictly on at least half of held actions"
            ),
            "fallback": "global multiplier 1.0 exactly",
            "missing_T_WE_action": (
                "discard every affected correction and use the released checkpoint"
            ),
        },
        "implementation": {
            "revision": implementation_revision,
            "source_projection_runner": (
                "scripts/remote/stage_pokeflex_missing5_scale_source_archive.py"
            ),
            "source_runner": (
                "scripts/remote/run_pokeflex_missing5_scale_source_take.py"
            ),
            "legacy_runner": (
                "scripts/remote/run_pokeflex_checkpoint_registration_smoke.py"
            ),
            "registration_protocol": (
                "configs/sota/pokeflex_bayesian_registration_v1.json"
            ),
            "upstream_commit": UPSTREAM_COMMIT,
            **implementation_hashes,
        },
        "information_boundary": {
            "target_frame": "f",
            "observation_frames": "f-5 through f-1",
            "robot_history": "through f-1",
            "source_target_meshes_used_only_for_scale_calibration": True,
            "unavailable_official_target_members_read": False,
            "held_v8_accessed": False,
        },
        "source_gate": {
            "complete_take_count": len(expected),
            "minimum_adjusted_object_count": 2,
            "maximum_full_source_action_regressions": 0,
            "maximum_loo_held_action_regressions": 0,
            "minimum_strict_loo_win_fraction": 0.5,
            "synthetic_positive_controls_required": 12,
            "synthetic_placebo_admissions_allowed": 0,
        },
        "forbidden": [
            "using any of the five unavailable official target archives or outcomes",
            "using any official-target score to choose a multiplier",
            "weakening leave-one-action-out safety after source scores are computed",
            "changing the frozen V4 official-evaluation protocol",
            "touching any held-v8 runtime, target, query, score, barrier, or outcome artifact",
        ],
    }
    payload["protocol_sha256"] = protocol_sha256(payload)
    validate_source_protocol(payload)
    return payload


def validate_source_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a missing-five source protocol and return its fixed inventory."""

    _require(payload.get("schema_version") == 1, "source protocol schema changed")
    _require(
        payload.get("artifact_kind") == PROTOCOL_KIND, "source protocol kind changed"
    )
    _require(payload.get("protocol_id") == PROTOCOL_ID, "source protocol id changed")
    _require(
        payload.get("protocol_sha256") == protocol_sha256(payload),
        "source protocol digest changed",
    )
    cohort = payload.get("source_cohort")
    _require(isinstance(cohort, Mapping), "source cohort is missing")
    assert isinstance(cohort, Mapping)
    _require(
        cohort.get("takes_by_object")
        == {name: list(takes) for name, takes in SOURCE_TAKES.items()},
        "source take partition changed",
    )
    _require(
        cohort.get("official_target_takes") == OFFICIAL_TARGET_TAKES,
        "official target partition changed",
    )
    _require(
        cohort.get("official_target_outcomes_used") is False,
        "target outcome boundary changed",
    )
    _require(
        not (set(source_take_ids()) & set(OFFICIAL_TARGET_TAKES.values())),
        "source and official target takes overlap",
    )
    method = payload.get("method")
    _require(isinstance(method, Mapping), "source method is missing")
    assert isinstance(method, Mapping)
    _require(method.get("field") == SOURCE_FIELD, "source correction field changed")
    _require(
        tuple(float(value) for value in method.get("candidate_multipliers", ()))
        == CANDIDATE_MULTIPLIERS,
        "source multiplier bank changed",
    )
    _require(
        float(method.get("global_multiplier", -1.0)) == GLOBAL_MULTIPLIER,
        "global multiplier changed",
    )
    _require(
        tuple(float(value) for value in method.get("candidate_effective_scales", ()))
        == tuple(BASE_EFFECTIVE_SCALE * value for value in CANDIDATE_MULTIPLIERS),
        "source effective-scale bank changed",
    )
    implementation = payload.get("implementation")
    _require(isinstance(implementation, Mapping), "implementation binding is missing")
    assert isinstance(implementation, Mapping)
    _require(
        implementation.get("upstream_commit") == UPSTREAM_COMMIT,
        "upstream commit changed",
    )
    _require(
        _valid_git_revision(implementation.get("revision")),
        "implementation revision is invalid",
    )
    for field in (
        "source_projection_runner_file_sha256",
        "source_runner_file_sha256",
        "legacy_runner_file_sha256",
        "registration_protocol_file_sha256",
    ):
        _require(_valid_digest(implementation.get(field)), f"{field} is invalid")
    boundary = payload.get("information_boundary")
    _require(isinstance(boundary, Mapping), "information boundary is missing")
    assert isinstance(boundary, Mapping)
    _require(
        boundary.get("unavailable_official_target_members_read") is False,
        "official target boundary changed",
    )
    _require(boundary.get("held_v8_accessed") is False, "held-v8 boundary changed")
    source_gate = payload.get("source_gate")
    _require(isinstance(source_gate, Mapping), "source gate is missing")
    assert isinstance(source_gate, Mapping)
    _require(
        int(source_gate.get("complete_take_count", -1)) == len(source_take_ids()),
        "source gate take count changed",
    )
    _require(
        int(source_gate.get("minimum_adjusted_object_count", -1)) == 2,
        "source gate adjusted-object threshold changed",
    )
    _require(
        float(source_gate.get("minimum_strict_loo_win_fraction", -1.0)) == 0.5,
        "source gate LOO threshold changed",
    )
    archive = payload.get("archive_inventory")
    _require(isinstance(archive, Mapping), "source archive inventory is missing")
    assert isinstance(archive, Mapping)
    takes = archive.get("takes")
    _require(isinstance(takes, Mapping), "source archive rows are missing")
    assert isinstance(takes, Mapping)
    _require(set(takes) == set(source_take_ids()), "source archive inventory changed")
    for take_id, row in takes.items():
        _require(isinstance(row, Mapping), "source archive row is invalid")
        assert isinstance(row, Mapping)
        _require(_valid_digest(row.get("sha256")), "source archive digest is invalid")
        _require(int(row.get("bytes", -1)) > 0, "source archive size is invalid")
        _require(
            Path(str(row.get("relative_path", ""))).name == f"{take_id}.zip",
            "source archive path changed",
        )
        expected_path = Path(object_name(take_id)) / f"{take_id}.zip"
        _require(
            Path(str(row.get("relative_path", ""))) == expected_path,
            "source archive relative path changed",
        )
    return {
        "passed": True,
        "protocol_sha256": str(payload["protocol_sha256"]),
        "source_take_ids": source_take_ids(),
        "archive_inventory": takes,
    }


def _score_key(multiplier: float) -> str:
    scale = BASE_EFFECTIVE_SCALE * multiplier
    return f"checkpoint_{SOURCE_FIELD}_residual_scale_{scale:g}"


def take_row_from_smoke(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract one validated all-scale source row from a smoke artifact."""

    validation = validate_source_protocol(protocol)
    _require(payload.get("artifact_kind") == SMOKE_KIND, "source smoke kind changed")
    _require(
        payload.get("future_observation_used") is False, "future observation was used"
    )
    _require(
        payload.get("missing5_source_protocol_sha256") == validation["protocol_sha256"],
        "source protocol binding changed",
    )
    _require(
        tuple(payload.get("correction_fields", ())) == (SOURCE_FIELD,),
        "source field changed",
    )
    implementation = protocol["implementation"]
    _require(
        payload.get("source_runner_file_sha256")
        == implementation["source_runner_file_sha256"],
        "source runner binding changed",
    )
    _require(
        payload.get("legacy_runner_file_sha256")
        == implementation["legacy_runner_file_sha256"],
        "legacy runner binding changed",
    )
    take = payload.get("take")
    _require(isinstance(take, Mapping), "source take metadata is missing")
    assert isinstance(take, Mapping)
    take_id = str(take.get("id", ""))
    _require(
        take_id in validation["source_take_ids"], "source take is outside protocol"
    )
    expected_archive = validation["archive_inventory"][take_id]
    _require(
        payload.get("source_archive_sha256") == expected_archive["sha256"],
        "source archive binding changed",
    )
    _require(
        _valid_digest(payload.get("projection_manifest_sha256")),
        "projection manifest binding is invalid",
    )
    _require(
        payload.get("official_target_outcome_used") is False,
        "official target outcome boundary changed",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 boundary changed")
    upstream = payload.get("upstream")
    _require(isinstance(upstream, Mapping), "source upstream metadata is missing")
    assert isinstance(upstream, Mapping)
    _require(
        upstream.get("git_commit") == UPSTREAM_COMMIT, "source upstream commit changed"
    )
    aggregates = payload.get("aggregates")
    _require(isinstance(aggregates, Mapping), "source aggregates are missing")
    assert isinstance(aggregates, Mapping)
    scores: dict[str, float] = {}
    for multiplier in CANDIDATE_MULTIPLIERS:
        row = aggregates.get(_score_key(multiplier))
        _require(isinstance(row, Mapping), "source multiplier score is missing")
        assert isinstance(row, Mapping)
        score = float(row.get("mean_CD_UL1_mm", math.nan))
        _require(
            math.isfinite(score) and score >= 0.0, "source multiplier score is invalid"
        )
        scores[f"{multiplier:g}"] = score
    updates = payload.get("updates")
    _require(
        isinstance(updates, Sequence) and not isinstance(updates, (str, bytes)),
        "source update rows are missing",
    )
    assert isinstance(updates, Sequence) and not isinstance(updates, (str, bytes))
    supported = sum(
        bool(row.get("accepted")) and bool(row.get("action_supported"))
        for row in updates
        if isinstance(row, Mapping)
    )
    return {
        "take_id": take_id,
        "object_name": object_name(take_id),
        "supported_frame_count": int(supported),
        "scores_CD_UL1_mm": scores,
    }


def _row_gain(row: Mapping[str, Any], multiplier: float) -> float:
    scores = row.get("scores_CD_UL1_mm")
    _require(isinstance(scores, Mapping), "source scale scores are missing")
    assert isinstance(scores, Mapping)
    baseline = float(scores.get(f"{GLOBAL_MULTIPLIER:g}", math.nan))
    candidate = float(scores.get(f"{multiplier:g}", math.nan))
    _require(
        math.isfinite(baseline) and baseline > 0.0 and math.isfinite(candidate),
        "source scale scores are invalid",
    )
    return (baseline - candidate) / baseline


def select_maximin_multiplier(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the source multiplier with the strongest action-level lower envelope."""

    source_rows = tuple(rows)
    _require(len(source_rows) >= 2, "at least two source actions are required")
    take_ids = tuple(str(row.get("take_id", "")) for row in source_rows)
    _require(len(set(take_ids)) == len(take_ids), "source actions repeat")
    objects = {object_name(take_id) for take_id in take_ids}
    _require(len(objects) == 1, "source actions change physical object")

    candidates: list[dict[str, Any]] = []
    for multiplier in CANDIDATE_MULTIPLIERS:
        gains = tuple(_row_gain(row, multiplier) for row in source_rows)
        minimum = float(min(gains))
        mean = float(np.mean(gains))
        candidates.append(
            {
                "multiplier": multiplier,
                "gains": gains,
                "minimum_gain": minimum,
                "mean_gain": mean,
            }
        )
    eligible = [row for row in candidates if row["minimum_gain"] >= -1e-12]
    _require(bool(eligible), "global source multiplier is unexpectedly ineligible")
    selected = max(
        eligible,
        key=lambda row: (
            row["minimum_gain"],
            row["mean_gain"],
            -abs(math.log(row["multiplier"])),
            -row["multiplier"],
        ),
    )
    return {
        "object_name": next(iter(objects)),
        "source_take_ids": list(take_ids),
        "multiplier": float(selected["multiplier"]),
        "effective_scale": BASE_EFFECTIVE_SCALE * float(selected["multiplier"]),
        "source_relative_improvements": list(selected["gains"]),
        "minimum_source_relative_improvement": float(selected["minimum_gain"]),
        "mean_source_relative_improvement": float(selected["mean_gain"]),
    }


def select_cross_validated_multiplier(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Promote a maximin scale only when leave-one-action-out transfer is safe."""

    source_rows = tuple(rows)
    full = select_maximin_multiplier(source_rows)
    loo: list[dict[str, Any]] = []
    for index, held in enumerate(source_rows):
        training = source_rows[:index] + source_rows[index + 1 :]
        selected = select_maximin_multiplier(training)
        gain = _row_gain(held, float(selected["multiplier"]))
        loo.append(
            {
                "held_take_id": str(held["take_id"]),
                "selected_multiplier": float(selected["multiplier"]),
                "held_relative_improvement": float(gain),
            }
        )
    minimum_loo = float(min(row["held_relative_improvement"] for row in loo))
    strict_loo_wins = sum(row["held_relative_improvement"] > 1e-12 for row in loo)
    required_wins = math.ceil(len(loo) / 2)
    promoted = bool(
        float(full["multiplier"]) != GLOBAL_MULTIPLIER
        and minimum_loo >= -1e-12
        and strict_loo_wins >= required_wins
    )
    multiplier = float(full["multiplier"]) if promoted else GLOBAL_MULTIPLIER
    gains = tuple(_row_gain(row, multiplier) for row in source_rows)
    deployed_loo: list[dict[str, Any]] = [
        {
            "held_take_id": row["held_take_id"],
            "selected_multiplier": (
                row["selected_multiplier"] if promoted else GLOBAL_MULTIPLIER
            ),
            "held_relative_improvement": (
                row["held_relative_improvement"] if promoted else 0.0
            ),
        }
        for row in loo
    ]
    return {
        **full,
        "unpromoted_full_multiplier": float(full["multiplier"]),
        "multiplier": multiplier,
        "effective_scale": BASE_EFFECTIVE_SCALE * multiplier,
        "selection_reason": (
            "cross-action-maximin-promoted"
            if promoted
            else "global-cross-validation-fallback"
        ),
        "promoted": promoted,
        "source_relative_improvements": list(gains),
        "minimum_source_relative_improvement": float(min(gains)),
        "mean_source_relative_improvement": float(np.mean(gains)),
        "loo": loo,
        "deployed_loo": deployed_loo,
        "minimum_loo_relative_improvement": minimum_loo,
        "strict_loo_win_count": strict_loo_wins,
        "required_strict_loo_win_count": required_wins,
        "loo_regression_count": sum(
            row["held_relative_improvement"] < -1e-12 for row in loo
        ),
        "deployed_loo_regression_count": sum(
            row["held_relative_improvement"] < -1e-12 for row in deployed_loo
        ),
    }


def synthetic_control_summary() -> dict[str, Any]:
    """Exercise the production selector on positive and placebo controls."""

    positive_passes = 0
    placebo_admissions = 0
    for control_index in range(12):
        positive = []
        placebo = []
        for take_index in range(6):
            baseline = 10.0 + 0.1 * take_index
            positive_scores = {
                "0.5": baseline * 0.995,
                "1": baseline,
                "1.5": baseline * 0.985,
                "2": baseline * (0.97 + 0.0001 * control_index),
                "3": baseline * 1.01,
                "4": baseline * 1.03,
            }
            placebo_scores = {
                "0.5": baseline * (1.01 if take_index % 2 else 0.99),
                "1": baseline,
                "1.5": baseline * (0.99 if take_index % 2 else 1.01),
                "2": baseline * 1.01,
                "3": baseline * 1.02,
                "4": baseline * 1.03,
            }
            positive.append(
                {
                    "take_id": f"Control{control_index}_T{take_index + 1}",
                    "scores_CD_UL1_mm": positive_scores,
                }
            )
            placebo.append(
                {
                    "take_id": f"Placebo{control_index}_T{take_index + 1}",
                    "scores_CD_UL1_mm": placebo_scores,
                }
            )
        positive_passes += int(select_cross_validated_multiplier(positive)["promoted"])
        placebo_admissions += int(
            select_cross_validated_multiplier(placebo)["promoted"]
        )
    return {
        "positive_control_count": 12,
        "positive_detection_count": positive_passes,
        "placebo_control_count": 12,
        "placebo_admission_count": placebo_admissions,
        "passed": positive_passes == 12 and placebo_admissions == 0,
    }


def build_source_result(
    artifacts: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    artifact_file_sha256s: Mapping[str, str],
    implementation_revision: str,
) -> dict[str, Any]:
    """Build the complete source calibration and its cross-action gate."""

    validation = validate_source_protocol(protocol)
    implementation = protocol["implementation"]
    _require(
        implementation_revision == implementation["revision"],
        "source result implementation revision changed",
    )
    rows = [take_row_from_smoke(artifact, protocol) for artifact in artifacts]
    _require(
        {row["take_id"] for row in rows} == set(validation["source_take_ids"]),
        "source result cohort is incomplete",
    )
    _require(
        set(artifact_file_sha256s) == set(validation["source_take_ids"]),
        "source artifact digest inventory changed",
    )
    _require(
        all(_valid_digest(value) for value in artifact_file_sha256s.values()),
        "source artifact digest is invalid",
    )
    grouped = {
        name: sorted(
            (row for row in rows if row["object_name"] == name),
            key=lambda row: row["take_id"],
        )
        for name in SOURCE_TAKES
    }
    objects = {
        name: select_cross_validated_multiplier(object_rows)
        for name, object_rows in grouped.items()
    }
    controls = synthetic_control_summary()
    selected_gains = [
        gain for row in objects.values() for gain in row["source_relative_improvements"]
    ]
    candidate_loo_regressions = sum(
        row["loo_regression_count"] for row in objects.values()
    )
    deployed_loo_regressions = sum(
        row["deployed_loo_regression_count"] for row in objects.values()
    )
    adjusted = sum(row["promoted"] for row in objects.values())
    source_gate = {
        "complete_take_count": len(rows),
        "adjusted_object_count": adjusted,
        "source_action_regression_count": sum(gain < -1e-12 for gain in selected_gains),
        "candidate_loo_held_action_regression_count": candidate_loo_regressions,
        "deployed_loo_held_action_regression_count": deployed_loo_regressions,
        "minimum_source_action_relative_improvement": float(min(selected_gains)),
        "mean_source_action_relative_improvement": float(np.mean(selected_gains)),
        "controls_passed": bool(controls["passed"]),
    }
    source_gate["passed"] = bool(
        source_gate["complete_take_count"] == len(source_take_ids())
        and source_gate["adjusted_object_count"]
        >= int(protocol["source_gate"]["minimum_adjusted_object_count"])
        and source_gate["source_action_regression_count"]
        <= int(protocol["source_gate"]["maximum_full_source_action_regressions"])
        and source_gate["deployed_loo_held_action_regression_count"]
        <= int(protocol["source_gate"]["maximum_loo_held_action_regressions"])
        and source_gate["mean_source_action_relative_improvement"] > 0.0
        and source_gate["controls_passed"]
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "protocol_sha256": validation["protocol_sha256"],
        "implementation_revision": implementation_revision,
        "claim_boundary": (
            "Source-only cross-action calibration on previously open public takes. "
            "A pass may freeze scales for the five unavailable official targets but "
            "is not target evidence or a state-of-the-art result."
        ),
        "objects": objects,
        "source_gate": source_gate,
        "synthetic_controls": controls,
        "rows": sorted(rows, key=lambda row: row["take_id"]),
        "artifact_file_sha256s": dict(sorted(artifact_file_sha256s.items())),
        "official_target_outcomes_used": False,
        "held_v8_accessed": False,
    }
    payload["result_sha256"] = result_sha256(payload)
    return payload


__all__ = [
    "BASE_EFFECTIVE_SCALE",
    "CANDIDATE_MULTIPLIERS",
    "GLOBAL_MULTIPLIER",
    "OFFICIAL_TARGET_TAKES",
    "PROTOCOL_ID",
    "PROTOCOL_KIND",
    "RESULT_KIND",
    "SOURCE_TAKES",
    "build_source_protocol",
    "build_source_result",
    "file_sha256",
    "object_name",
    "protocol_sha256",
    "result_sha256",
    "select_cross_validated_multiplier",
    "select_maximin_multiplier",
    "source_take_ids",
    "synthetic_control_summary",
    "take_row_from_smoke",
    "validate_source_protocol",
]
