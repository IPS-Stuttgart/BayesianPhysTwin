"""Conservative source selection for causal PokeFlex state corrections.

The selector operates only on already generated causal candidate banks.  It
prefers the smallest correction that clears a whole-object transfer gate and
keeps the released checkpoint exactly on unsupported frames.  Hidden target
errors are used only for source-family selection and never as online features.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "pokeflex-conservative-shrinkage-source-v1"
PROTOCOL_KIND = "PokeFlexConservativeShrinkageSourceProtocol"
PARENT_PROTOCOL_SHA256 = (
    "c68a33d82ee4c7474a09d30806df14cd3f8d3437acb2f4f1ad947cc83e09be33"
)
BASELINE_ARM = "released_checkpoint"
_ARM_PATTERN = re.compile(
    r"^checkpoint_action_local_state_relative_"
    r"(?P<radius>[0-9]+(?:\.[0-9]+)?)_residual_scale_"
    r"(?P<scale>[0-9]+(?:\.[0-9]+)?)$"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _take_identity(take_id: str) -> tuple[str, str]:
    object_name, separator, take_number = take_id.rpartition("_T")
    _require(
        bool(separator) and bool(object_name) and take_number.isdigit(),
        f"invalid PokeFlex take id: {take_id}",
    )
    return object_name, f"T{take_number}"


def _canonical_protocol_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_pokeflex_conservative_shrinkage_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Load and fail-closed validate the source protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "unsupported protocol schema")
    _require(payload.get("artifact_kind") == PROTOCOL_KIND, "protocol kind changed")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol id changed")
    observed = _canonical_protocol_sha256(payload)
    _require(payload.get("protocol_sha256") == observed, "protocol checksum mismatch")
    parent = payload.get("parent_protocol")
    _require(isinstance(parent, Mapping), "parent protocol is missing")
    _require(
        parent.get("protocol_sha256") == PARENT_PROTOCOL_SHA256,
        "parent protocol checksum changed",
    )
    boundary = payload.get("evidence_boundary")
    _require(isinstance(boundary, Mapping), "evidence boundary is missing")
    _require(boundary.get("target_objects_remain_sealed") is True, "target opened")
    _require(boundary.get("replacement_allowed") is False, "replacement changed")
    take_ids = tuple(map(str, boundary.get("opened_source_take_ids", ())))
    _require(len(take_ids) >= 9, "source protocol has too few takes")
    _require(len(set(take_ids)) == len(take_ids), "source take inventory repeats")
    selection = payload.get("selection")
    _require(isinstance(selection, Mapping), "selection rule is missing")
    _require(
        selection.get("tie_break")
        == "smallest positive scale, then smallest support radius, then arm name",
        "selection tie-break changed",
    )
    return {
        "path": str(source),
        "payload": payload,
        "protocol_sha256": observed,
        "opened_source_take_ids": take_ids,
    }


@dataclass(frozen=True, slots=True)
class ConservativeShrinkageConfig:
    """Whole-object source gate and deterministic shrinkage preference."""

    minimum_object_balanced_improvement: float = 0.01
    maximum_object_regression: float = 0.0
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        _require(
            np.isfinite(self.minimum_object_balanced_improvement)
            and self.minimum_object_balanced_improvement >= 0.0,
            "minimum improvement must be finite and nonnegative",
        )
        _require(
            np.isfinite(self.maximum_object_regression)
            and self.maximum_object_regression >= 0.0,
            "maximum regression must be finite and nonnegative",
        )
        _require(
            np.isfinite(self.numerical_tolerance) and self.numerical_tolerance >= 0.0,
            "numerical tolerance must be finite and nonnegative",
        )


@dataclass(frozen=True, slots=True)
class _Arm:
    name: str
    radius_fraction: float
    scale: float


@dataclass(frozen=True, slots=True)
class _Take:
    take_id: str
    object_name: str
    baseline_mean_mm: float
    arm_mean_mm: Mapping[str, float]
    fallback_frame_count: int
    fallback_mismatch_count: Mapping[str, int]


def _candidate_arms(target: Mapping[str, Any]) -> tuple[_Arm, ...]:
    arms = []
    for name in sorted(target):
        match = _ARM_PATTERN.match(str(name))
        if match is None:
            continue
        scale = float(match.group("scale"))
        if scale <= 0.0:
            continue
        arms.append(
            _Arm(
                name=str(name),
                radius_fraction=float(match.group("radius")),
                scale=scale,
            )
        )
    _require(bool(arms), "candidate bank has no positive shrinkage arm")
    return tuple(arms)


def _extract_take(payload: Mapping[str, Any], expected_arms: tuple[_Arm, ...]) -> _Take:
    _require(
        payload.get("artifact_kind")
        == "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "unexpected source artifact kind",
    )
    _require(payload.get("future_observation_used") is False, "future input was used")
    take_id = str(payload.get("take", {}).get("id", ""))
    object_name, _ = _take_identity(take_id)
    targets = payload.get("targets")
    updates = payload.get("updates")
    _require(isinstance(targets, list) and targets, "source targets are missing")
    _require(isinstance(updates, list), "source updates are missing")
    updates_by_frame = {int(row["target_frame"]): row for row in updates}
    _require(len(updates_by_frame) == len(updates), "source update frames repeat")

    observed_arms = _candidate_arms(targets[0])
    _require(observed_arms == expected_arms, "candidate arm inventory changed")
    baseline_values: list[float] = []
    arm_values: dict[str, list[float]] = {arm.name: [] for arm in expected_arms}
    fallback_count = 0
    fallback_mismatch = {arm.name: 0 for arm in expected_arms}
    for target in targets:
        _require(
            _candidate_arms(target) == expected_arms, "frame candidate bank changed"
        )
        frame = int(target["target_frame"])
        baseline = float(target["released_checkpoint_CD_UL1_mm"])
        _require(np.isfinite(baseline) and baseline > 0.0, "baseline error is invalid")
        baseline_values.append(baseline)
        update = updates_by_frame.get(frame)
        _require(update is not None, "target has no causal source update record")
        supported = bool(update.get("accepted")) and bool(
            update.get("action_supported")
        )
        if not supported:
            fallback_count += 1
        for arm in expected_arms:
            value = float(target[arm.name])
            _require(np.isfinite(value) and value > 0.0, "candidate error is invalid")
            arm_values[arm.name].append(value)
            if not supported and value != baseline:
                fallback_mismatch[arm.name] += 1
    return _Take(
        take_id=take_id,
        object_name=object_name,
        baseline_mean_mm=float(np.mean(baseline_values)),
        arm_mean_mm={
            name: float(np.mean(values)) for name, values in arm_values.items()
        },
        fallback_frame_count=fallback_count,
        fallback_mismatch_count=fallback_mismatch,
    )


def _object_means(
    takes: Sequence[_Take],
    arm_name: str,
) -> tuple[dict[str, float], dict[str, float]]:
    baseline_by_object: dict[str, list[float]] = defaultdict(list)
    candidate_by_object: dict[str, list[float]] = defaultdict(list)
    for take in takes:
        baseline_by_object[take.object_name].append(take.baseline_mean_mm)
        candidate_by_object[take.object_name].append(take.arm_mean_mm[arm_name])
    baseline = {
        name: float(np.mean(values)) for name, values in baseline_by_object.items()
    }
    candidate = {
        name: float(np.mean(values)) for name, values in candidate_by_object.items()
    }
    return baseline, candidate


def _arm_statistics(takes: Sequence[_Take], arm: _Arm) -> dict[str, Any]:
    baseline, candidate = _object_means(takes, arm.name)
    improvements = {
        name: 1.0 - candidate[name] / baseline[name] for name in sorted(baseline)
    }
    return {
        "arm": arm.name,
        "scale": arm.scale,
        "radius_fraction": arm.radius_fraction,
        "object_balanced_relative_improvement": float(
            np.mean(tuple(improvements.values()))
        ),
        "object_win_count": int(sum(value > 0.0 for value in improvements.values())),
        "minimum_object_relative_improvement": float(min(improvements.values())),
        "objects": {
            name: {
                "baseline_mean_CD_UL1_mm": baseline[name],
                "candidate_mean_CD_UL1_mm": candidate[name],
                "relative_improvement": improvements[name],
            }
            for name in sorted(baseline)
        },
    }


def _select_arm(
    takes: Sequence[_Take],
    arms: tuple[_Arm, ...],
    config: ConservativeShrinkageConfig,
) -> tuple[_Arm | None, list[dict[str, Any]]]:
    statistics = [_arm_statistics(takes, arm) for arm in arms]
    eligible = []
    for arm, result in zip(arms, statistics, strict=True):
        improvement = float(result["object_balanced_relative_improvement"])
        worst = float(result["minimum_object_relative_improvement"])
        if (
            improvement + config.numerical_tolerance
            >= config.minimum_object_balanced_improvement
            and worst + config.maximum_object_regression + config.numerical_tolerance
            >= 0.0
        ):
            eligible.append(arm)
    if not eligible:
        return None, statistics
    selected = min(
        eligible,
        key=lambda arm: (arm.scale, arm.radius_fraction, arm.name),
    )
    return selected, statistics


def evaluate_pokeflex_conservative_shrinkage_source(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_take_ids: Sequence[str] | None = None,
    config: ConservativeShrinkageConfig | None = None,
) -> dict[str, Any]:
    """Cross-fit and freeze the smallest whole-object-safe correction arm."""

    _require(bool(payloads), "at least one source artifact is required")
    cfg = config or ConservativeShrinkageConfig()
    first_targets = payloads[0].get("targets")
    _require(isinstance(first_targets, list) and first_targets, "targets are missing")
    arms = _candidate_arms(first_targets[0])
    takes = tuple(_extract_take(payload, arms) for payload in payloads)
    take_ids = tuple(take.take_id for take in takes)
    _require(len(set(take_ids)) == len(take_ids), "source take is duplicated")
    if expected_take_ids is not None:
        expected = tuple(map(str, expected_take_ids))
        _require(set(take_ids) == set(expected), "source take inventory changed")
        _require(len(take_ids) == len(expected), "source take count changed")
    objects = sorted({take.object_name for take in takes})
    _require(len(objects) >= 3, "source gate needs at least three objects")

    selected, arm_statistics = _select_arm(takes, arms, cfg)
    _require(selected is not None, "no arm passes the whole-object source gate")
    selected_result = next(row for row in arm_statistics if row["arm"] == selected.name)

    folds = []
    held_improvements = []
    for held_object in objects:
        training = tuple(take for take in takes if take.object_name != held_object)
        held = tuple(take for take in takes if take.object_name == held_object)
        fold_arm, _ = _select_arm(training, arms, cfg)
        _require(fold_arm is not None, f"fold {held_object} has no eligible arm")
        held_result = _arm_statistics(held, fold_arm)
        held_improvement = float(held_result["object_balanced_relative_improvement"])
        held_improvements.append(held_improvement)
        folds.append(
            {
                "held_object": held_object,
                "selected_arm": fold_arm.name,
                "held_relative_improvement": held_improvement,
            }
        )

    fallback_frames = sum(take.fallback_frame_count for take in takes)
    fallback_mismatches = sum(
        take.fallback_mismatch_count[selected.name] for take in takes
    )
    stable_selection = all(row["selected_arm"] == selected.name for row in folds)
    held_nonregression = all(
        value + cfg.maximum_object_regression + cfg.numerical_tolerance >= 0.0
        for value in held_improvements
    )
    gate_passed = (
        stable_selection
        and held_nonregression
        and fallback_mismatches == 0
        and float(selected_result["object_balanced_relative_improvement"])
        + cfg.numerical_tolerance
        >= cfg.minimum_object_balanced_improvement
    )
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexConservativeShrinkageSourceResult",
        "claim_status": "source_development_only",
        "source_gate_passed": gate_passed,
        "config": {
            "minimum_object_balanced_improvement": (
                cfg.minimum_object_balanced_improvement
            ),
            "maximum_object_regression": cfg.maximum_object_regression,
            "tie_break": (
                "smallest positive scale, then smallest support radius, then arm name"
            ),
        },
        "take_count": len(takes),
        "object_count": len(objects),
        "selected_arm": selected.name,
        "selected_result": selected_result,
        "cross_fitted": {
            "folds": folds,
            "stable_selection": stable_selection,
            "held_object_win_count": int(
                sum(value > 0.0 for value in held_improvements)
            ),
            "held_object_balanced_relative_improvement": float(
                np.mean(held_improvements)
            ),
            "held_nonregression": held_nonregression,
        },
        "fallback": {
            "unsupported_frame_count": fallback_frames,
            "selected_arm_metric_mismatch_count": fallback_mismatches,
            "exact_metric_fallback_passed": fallback_mismatches == 0,
        },
        "candidate_bank": arm_statistics,
        "target_objects_opened": False,
    }


__all__ = [
    "BASELINE_ARM",
    "ConservativeShrinkageConfig",
    "load_pokeflex_conservative_shrinkage_protocol",
    "evaluate_pokeflex_conservative_shrinkage_source",
]
