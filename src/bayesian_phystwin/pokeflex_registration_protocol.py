"""Validation for the prospective PokeFlex Bayesian registration protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


POKEFLEX_OBJECTS = frozenset(
    {
        "3dPrintedBunny",
        "3dPrintedCylinder",
        "3dPrintedHeart",
        "3dPrintedPizza",
        "3dPrintedPyramid",
        "Beanbag",
        "FoamCylinder",
        "FoamDice",
        "FoamHalfSphere",
        "MemoryFoam",
        "Pillow",
        "PlushDice",
        "PlushMoon",
        "PlushOctopus",
        "PlushTurtle",
        "PlushVolleyball",
        "Sponge",
        "ToiletPaperRoll",
    }
)
POKEFLEX_REGISTRATION_PROTOCOL_ID = "pokeflex-bayesian-registration-v1"
POKEFLEX_REGISTRATION_PROTOCOL_SHA256 = (
    "c68a33d82ee4c7474a09d30806df14cd3f8d3437acb2f4f1ad947cc83e09be33"
)
POKEFLEX_ACTION_GUARD_DEVELOPMENT_LOCK_ID = "pokeflex-action-guard-development-v1"
POKEFLEX_ACTION_GUARD_DEVELOPMENT_LOCK_SHA256 = (
    "4796ca2cf1f45d9e6cd810de13650126ef3f9dba48087f3d839754d2c37630c6"
)


def _canonical_protocol_bytes(payload: Mapping[str, Any]) -> bytes:
    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    canonical.pop("lock_sha256", None)
    return json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def pokeflex_registration_protocol_sha256(payload: Mapping[str, Any]) -> str:
    """Return the lock hash, excluding the embedded hash field itself."""

    return hashlib.sha256(_canonical_protocol_bytes(payload)).hexdigest()


def _objects(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"PokeFlex cohort {key} must be a nonempty list")
    result = tuple(str(item) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"PokeFlex cohort {key} contains duplicates")
    return result


def validate_pokeflex_registration_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the scientific and causal boundaries of the v1 lock."""

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported PokeFlex registration protocol schema")
    if payload.get("artifact_kind") != "PokeFlexBayesianRegistrationProtocol":
        raise ValueError("unexpected PokeFlex registration artifact kind")
    if payload.get("protocol_id") != POKEFLEX_REGISTRATION_PROTOCOL_ID:
        raise ValueError("PokeFlex registration protocol id changed")

    observed_hash = pokeflex_registration_protocol_sha256(payload)
    if payload.get("protocol_sha256") != observed_hash:
        raise ValueError("PokeFlex registration protocol checksum mismatch")
    if (
        POKEFLEX_REGISTRATION_PROTOCOL_SHA256 != "TO_BE_FILLED"
        and observed_hash != POKEFLEX_REGISTRATION_PROTOCOL_SHA256
    ):
        raise ValueError("PokeFlex registration protocol differs from canonical lock")

    upstream = payload.get("upstream")
    if not isinstance(upstream, Mapping):
        raise ValueError("PokeFlex protocol omits upstream provenance")
    if upstream.get("code_commit") != "aaa8726072834a95bbe97e1a113588968c36e185":
        raise ValueError("PokeFlex upstream code commit changed")
    reference = upstream.get("paper_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("PokeFlex protocol omits the paper reference")
    if float(reference.get("cd_ul1_mm", float("nan"))) != 6.498:
        raise ValueError("PokeFlex published CD_UL1 reference changed")
    if int(reference.get("history_frames", -1)) != 5:
        raise ValueError("PokeFlex history length changed")

    cohort = payload.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("PokeFlex protocol omits its cohort")
    development = _objects(cohort, "development_objects")
    calibration = _objects(cohort, "calibration_objects")
    target = _objects(cohort, "target_objects")
    excluded_raw = cohort.get("excluded_objects")
    if not isinstance(excluded_raw, Mapping):
        raise ValueError("PokeFlex protocol omits excluded-object provenance")
    excluded = tuple(str(item) for item in excluded_raw)
    partitions = (set(development), set(calibration), set(target), set(excluded))
    for index, first in enumerate(partitions):
        for second in partitions[index + 1 :]:
            if first & second:
                raise ValueError("PokeFlex object partitions overlap")
    if set().union(*partitions) != set(POKEFLEX_OBJECTS):
        raise ValueError("PokeFlex object partitions do not cover the release")
    if excluded != ("3dPrintedBunny",):
        raise ValueError("previously inspected PokeFlex object is not isolated")
    if cohort.get("held_out_take") != "T2":
        raise ValueError("PokeFlex held-out take changed")
    if cohort.get("development_smoke_take") != "FoamDice_T3":
        raise ValueError("PokeFlex development smoke take changed")

    inputs = payload.get("causal_input_contract")
    if not isinstance(inputs, Mapping):
        raise ValueError("PokeFlex protocol omits causal inputs")
    required_inputs = {
        "allowed_observation_frames": "f-5 through f-1 only",
        "forbidden_observation_frames": "f and all later frames",
        "history_frame_count": 5,
        "target_take_deformed_mesh_allowed_before_final_scoring": False,
        "synthetic_point_clouds_from_target_mesh_allowed": False,
        "future_mesh_cropping_or_registration_allowed": False,
    }
    for key, expected in required_inputs.items():
        if inputs.get(key) != expected:
            raise ValueError(f"PokeFlex causal input contract changed: {key}")

    methods = payload.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("PokeFlex protocol omits methods")
    constraints = methods.get("candidate_constraints")
    if not isinstance(constraints, Mapping):
        raise ValueError("PokeFlex protocol omits candidate constraints")
    for key in (
        "observation_reliability_is_residual_independent",
        "state_innovation_is_processed_once",
        "camera_correlation_is_not_treated_as_independent_evidence",
        "common_mode_camera_bias_is_explicit",
        "assignment_mixture_spread_enters_covariance",
    ):
        if constraints.get(key) is not True:
            raise ValueError(f"PokeFlex reliability constraint changed: {key}")
    if "byte-for-byte" not in str(constraints.get("exact_fallback", "")):
        raise ValueError("PokeFlex exact fallback guarantee changed")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("PokeFlex protocol omits evaluation")
    if evaluation.get("primary_metric") != "CD_UL1_mm":
        raise ValueError("PokeFlex primary metric changed")
    sampling = evaluation.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("surface_points") != 10000:
        raise ValueError("PokeFlex surface sampling changed")

    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("PokeFlex protocol omits gates")
    sota = gates.get("direct_published_sota")
    if not isinstance(sota, Mapping) or sota.get("mean_target_CD_UL1_mm_below") != 6.498:
        raise ValueError("PokeFlex direct SOTA gate changed")

    return {
        "passed": True,
        "protocol_sha256": observed_hash,
        "development_objects": development,
        "calibration_objects": calibration,
        "target_objects": target,
        "excluded_objects": excluded,
    }


def load_pokeflex_registration_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate the canonical prospective protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = validate_pokeflex_registration_protocol(payload)
    result["path"] = str(source)
    result["payload"] = payload
    return result


def validate_pokeflex_action_guard_development_lock(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the prospective take boundary and frozen action policy."""

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported PokeFlex action-guard lock schema")
    if payload.get("artifact_kind") != "PokeFlexActionGuardDevelopmentLock":
        raise ValueError("unexpected PokeFlex action-guard artifact kind")
    if payload.get("lock_id") != POKEFLEX_ACTION_GUARD_DEVELOPMENT_LOCK_ID:
        raise ValueError("PokeFlex action-guard lock id changed")
    if payload.get("parent_protocol_sha256") != POKEFLEX_REGISTRATION_PROTOCOL_SHA256:
        raise ValueError("PokeFlex action-guard parent protocol changed")

    observed_hash = hashlib.sha256(_canonical_protocol_bytes(payload)).hexdigest()
    if payload.get("lock_sha256") != observed_hash:
        raise ValueError("PokeFlex action-guard checksum mismatch")
    if observed_hash != POKEFLEX_ACTION_GUARD_DEVELOPMENT_LOCK_SHA256:
        raise ValueError("PokeFlex action-guard differs from canonical lock")

    boundary = payload.get("evidence_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("PokeFlex action-guard omits evidence boundary")
    if boundary.get("prospective_development_validation_takes") != [
        "T1",
        "T4",
        "T5",
        "T6",
    ]:
        raise ValueError("PokeFlex prospective development takes changed")
    if boundary.get("reserved_development_takes") != ["T7", "T8"]:
        raise ValueError("PokeFlex reserved development takes changed")
    if boundary.get("intentionally_unopened_take") != "T2":
        raise ValueError("PokeFlex held-out take changed")
    if boundary.get("calibration_and_target_objects_remain_sealed") is not True:
        raise ValueError("PokeFlex calibration or target boundary changed")

    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("PokeFlex action-guard candidate is missing")
    required_candidate = {
        "field": "action_local_state",
        "tool_history_frames": 4,
        "contact_candidate_count": 32,
        "influence_radius_m": 0.06,
        "minimum_action_force_n": 3.0,
        "strong_update_force_n": 15.0,
        "weak_scale": 0.125,
        "strong_scale": 0.5,
        "fallback": "released checkpoint vertices byte-for-byte",
    }
    for key, expected in required_candidate.items():
        if candidate.get(key) != expected:
            raise ValueError(f"PokeFlex action-guard candidate changed: {key}")

    return {
        "passed": True,
        "lock_sha256": observed_hash,
        "prospective_development_validation_takes": tuple(
            boundary["prospective_development_validation_takes"]
        ),
    }


def load_pokeflex_action_guard_development_lock(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the canonical action-guard development lock."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = validate_pokeflex_action_guard_development_lock(payload)
    result["path"] = str(source)
    result["payload"] = payload
    return result
