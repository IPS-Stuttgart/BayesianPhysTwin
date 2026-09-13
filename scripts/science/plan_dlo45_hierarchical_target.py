#!/usr/bin/env python3
"""Plan DLO4/DLO5 target carriers without loading target numerical values.

The planner consumes the authorized header-only parent-artifact census and the
already-frozen DLO2/DLO3 source plan. It reuses the exact semantic vocabulary,
requires the same residual-construction mode used on both source DLOs, and
scores only names, roles, shapes, dtypes, and source-shape compatibility.
Ambiguous mappings fail closed and are published unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

TARGET_DLOS = ("DLO4", "DLO5")
MIN_SCORE_MARGIN = 5


class TargetPlanError(RuntimeError):
    """Raised when target planning violates authorization or source semantics."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def target_identity(text: str) -> str | None:
    lower = text.lower()
    identities = []
    for dlo in TARGET_DLOS:
        number = dlo[-1]
        if any(token in lower for token in (dlo.lower(), f"dlo_{number}", f"dlo-{number}")):
            identities.append(dlo)
    return identities[0] if len(set(identities)) == 1 else None


def normalize_roles(value: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(str(role) for role in value)))


def descriptor(
    *,
    record: Mapping[str, Any],
    group_id: str,
    locator_type: str,
    key: str,
    shape: Iterable[int],
    dtype: str,
    roles: Iterable[str],
    member: str | None = None,
    nested_member: str | None = None,
) -> dict[str, Any]:
    combined = " ".join(
        value for value in (record["relative_path"], member, nested_member, key) if value
    )
    identity = target_identity(combined)
    return {
        "group_id": group_id,
        "container_path": record["path"],
        "container_relative_path": record["relative_path"],
        "container_sha256": record["sha256"],
        "locator_type": locator_type,
        "member": member,
        "nested_member": nested_member,
        "key": key,
        "shape": [int(value) for value in shape],
        "dtype": str(dtype),
        "elements": int(__import__("math").prod(shape)),
        "roles": list(normalize_roles(roles)),
        "target_identity": identity,
        "identifier": "::".join(
            value
            for value in (record["path"], member, nested_member, key)
            if value is not None
        ),
    }


def descriptors_from_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    schema = record["schema"]
    kind = schema.get("kind")
    values: list[dict[str, Any]] = []
    if kind == "npy":
        values.append(
            descriptor(
                record=record,
                group_id=f"direct-npy::{Path(record['path']).parent.as_posix()}",
                locator_type="direct-npy",
                key=Path(record["path"]).stem,
                shape=schema["shape"],
                dtype=schema["dtype"],
                roles=record.get("roles", []),
            )
        )
    elif kind == "npz":
        group_id = f"direct-npz::{record['path']}"
        for key, array in schema.get("arrays", {}).items():
            values.append(
                descriptor(
                    record=record,
                    group_id=group_id,
                    locator_type="direct-npz",
                    key=key,
                    shape=array["shape"],
                    dtype=array["dtype"],
                    roles=array.get("roles", []),
                    member=array.get("member"),
                )
            )
    elif kind == "zip":
        for array in schema.get("numpy_headers", []):
            if array.get("kind") == "nested-npz":
                group_id = f"nested-npz::{record['path']}::{array['member']}"
                for key, nested in array.get("arrays", {}).items():
                    values.append(
                        descriptor(
                            record=record,
                            group_id=group_id,
                            locator_type="zip-nested-npz",
                            key=key,
                            shape=nested["shape"],
                            dtype=nested["dtype"],
                            roles=nested.get("roles", []),
                            member=array["member"],
                            nested_member=nested.get("member"),
                        )
                    )
            elif "shape" in array and "dtype" in array:
                parent = str(Path(array["member"]).parent)
                group_id = f"zip-npy-group::{record['path']}::{parent}"
                values.append(
                    descriptor(
                        record=record,
                        group_id=group_id,
                        locator_type="zip-npy",
                        key=Path(array["member"]).stem,
                        shape=array["shape"],
                        dtype=array["dtype"],
                        roles=array.get("roles", []),
                        member=array["member"],
                    )
                )
    return values


def shape_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a = tuple(left["shape"])
    b = tuple(right["shape"])
    return a == b or (len(a) == len(b) + 1 and a[1:] == b) or (len(b) == len(a) + 1 and b[1:] == a)


def source_shape_score(target: Mapping[str, Any], source: Mapping[str, Any] | None) -> int:
    if source is None:
        return 0
    a = tuple(target["shape"])
    b = tuple(source["shape"])
    score = 0
    if len(a) == len(b):
        score += 3
    if a[-1:] == b[-1:]:
        score += 6
    if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]:
        score += 4
    if str(target["dtype"])[0] == str(source["dtype"])[0]:
        score += 1
    return score


def candidates_with_role(values: Iterable[Mapping[str, Any]], role: str) -> list[dict[str, Any]]:
    return [dict(value) for value in values if role in value["roles"]]


def build_options(
    values: list[dict[str, Any]],
    *,
    dlo: str,
    required_mode: str,
    source_reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    eligible = [
        value for value in values
        if value["target_identity"] in {None, dlo}
    ]
    residuals = candidates_with_role(eligible, "residual")
    observations = candidates_with_role(eligible, "observation")
    physicals = [
        value for value in eligible
        if "physical" in value["roles"] and "residual" not in value["roles"]
    ]
    states = candidates_with_role(eligible, "state")
    actions = candidates_with_role(eligible, "action")
    contacts = candidates_with_role(eligible, "contact")
    trajectory_ids = candidates_with_role(eligible, "trajectory_id")
    times = candidates_with_role(eligible, "time")
    options: list[dict[str, Any]] = []

    def best(role_values: list[dict[str, Any]], reference_role: str, anchor: Mapping[str, Any]) -> dict[str, Any] | None:
        compatible = [value for value in role_values if shape_compatible(anchor, value)]
        if not compatible:
            return None
        compatible.sort(
            key=lambda value: (
                -source_shape_score(value, source_reference.get(reference_role)),
                value["identifier"],
            )
        )
        return compatible[0]

    if required_mode == "explicit-residual":
        for residual in residuals:
            physical = best(physicals, "physical", residual)
            state = best(states, "state", residual)
            feature = physical or state
            if feature is None:
                continue
            action = best(actions, "action", residual)
            contact = best(contacts, "contact", residual)
            trajectory = best(trajectory_ids, "trajectory_id", residual)
            time = best(times, "time", residual)
            score = 25
            score += source_shape_score(residual, source_reference.get("residual"))
            score += source_shape_score(feature, source_reference.get("physical"))
            score += 7 if trajectory is not None else 0
            score += 4 if action is not None else 0
            score += 4 if contact is not None else 0
            options.append(
                {
                    "dlo": dlo,
                    "mode": required_mode,
                    "group_id": residual["group_id"],
                    "score": score,
                    "residual": residual,
                    "observation": None,
                    "physical": feature,
                    "state": state,
                    "action": action,
                    "contact": contact,
                    "trajectory_id": trajectory,
                    "time": time,
                    "residual_definition": "stored residual array",
                }
            )
    elif required_mode == "observation-minus-physical":
        for observation in observations:
            for physical in physicals:
                if not shape_compatible(observation, physical):
                    continue
                state = best(states, "state", observation) or physical
                action = best(actions, "action", observation)
                contact = best(contacts, "contact", observation)
                trajectory = best(trajectory_ids, "trajectory_id", observation)
                time = best(times, "time", observation)
                score = 28
                score += source_shape_score(observation, source_reference.get("observation"))
                score += source_shape_score(physical, source_reference.get("physical"))
                score += 7 if trajectory is not None else 0
                score += 4 if action is not None else 0
                score += 4 if contact is not None else 0
                options.append(
                    {
                        "dlo": dlo,
                        "mode": required_mode,
                        "group_id": observation["group_id"],
                        "score": score,
                        "residual": None,
                        "observation": observation,
                        "physical": physical,
                        "state": state,
                        "action": action,
                        "contact": contact,
                        "trajectory_id": trajectory,
                        "time": time,
                        "residual_definition": "observation - physical prediction",
                    }
                )
    else:
        raise TargetPlanError(f"unregistered source residual mode {required_mode!r}")

    # Arrays in one logical artifact group are preferred. Cross-group mappings
    # are not allowed because they could silently combine different reruns.
    options = [
        option
        for option in options
        if all(
            value is None or value["group_id"] == option["group_id"]
            for value in (
                option["residual"],
                option["observation"],
                option["physical"],
                option["state"],
                option["action"],
                option["contact"],
                option["trajectory_id"],
                option["time"],
            )
        )
    ]
    options.sort(key=lambda value: (-value["score"], value["group_id"], json.dumps(value, sort_keys=True)))
    return options


def verify_inputs(
    census: Mapping[str, Any],
    authorization: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    claimed = authorization.get("authorization_id")
    unsigned = dict(authorization)
    unsigned.pop("authorization_id", None)
    if canonical_hash(unsigned) != claimed:
        raise TargetPlanError("authorization ID mismatch")
    if census.get("authorization_id") != claimed:
        raise TargetPlanError("census/authorization mismatch")
    if census.get("source_model_id") != authorization.get("source_model_id"):
        raise TargetPlanError("census/source-model mismatch")
    boundary = census.get("information_boundary", {})
    if boundary.get("target_numeric_array_values_loaded") is not False:
        raise TargetPlanError("target values were loaded before planning")
    if boundary.get("target_performance_metric_computed") is not False:
        raise TargetPlanError("target performance was computed before planning")
    if authorization.get("selection_frozen_before_target") is not True:
        raise TargetPlanError("source selection was not frozen")
    if source_plan.get("ready_for_panel_build") is not True:
        raise TargetPlanError("source plan was not structurally complete")
    modes = {value["mode"] for value in source_plan["selected_carriers"].values()}
    if len(modes) != 1:
        raise TargetPlanError(f"source DLOs used different residual modes: {modes}")
    mode = next(iter(modes))
    references: dict[str, Any] = {}
    for role in (
        "residual", "observation", "physical", "state", "action", "contact", "trajectory_id", "time"
    ):
        role_values = [
            value.get(role)
            for value in source_plan["selected_carriers"].values()
            if value.get(role) is not None
        ]
        if role_values:
            references[role] = role_values[0]
    return mode, references


def plan(
    census: Mapping[str, Any],
    authorization: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> dict[str, Any]:
    required_mode, source_reference = verify_inputs(census, authorization, source_plan)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in census.get("ranked_files", []):
        for value in descriptors_from_record(record):
            groups[value["group_id"]].append(value)

    ranked: dict[str, list[dict[str, Any]]] = {dlo: [] for dlo in TARGET_DLOS}
    for group_values in groups.values():
        identities = {
            value["target_identity"] for value in group_values if value["target_identity"] is not None
        }
        for dlo in TARGET_DLOS:
            if identities and dlo not in identities:
                continue
            ranked[dlo].extend(
                build_options(
                    group_values,
                    dlo=dlo,
                    required_mode=required_mode,
                    source_reference=source_reference,
                )
            )

    decisions = {}
    selected = {}
    for dlo in TARGET_DLOS:
        options = ranked[dlo]
        options.sort(key=lambda value: (-value["score"], value["group_id"], json.dumps(value, sort_keys=True)))
        top = options[0] if options else None
        second = options[1] if len(options) > 1 else None
        margin = None if top is None else top["score"] - (second["score"] if second else 0)
        complete = bool(
            top
            and top["physical"] is not None
            and top["trajectory_id"] is not None
            and (
                top["residual"] is not None
                or top["observation"] is not None
            )
        )
        unambiguous = bool(
            top and complete and (second is None or margin is not None and margin >= MIN_SCORE_MARGIN)
        )
        decisions[dlo] = {
            "candidate_count": len(options),
            "top_score": None if top is None else top["score"],
            "runner_up_score": None if second is None else second["score"],
            "score_margin": margin,
            "structurally_complete": complete,
            "unambiguous": unambiguous,
            "reason": (
                "selected"
                if unambiguous
                else "no candidate"
                if top is None
                else "incomplete role set"
                if not complete
                else "top alternatives are not separated by frozen margin"
            ),
        }
        if unambiguous:
            selected[dlo] = top
        ranked[dlo] = options[:100]

    result = {
        "schema": "bayesian-phystwin.hierarchical-missing-physics-dlo45-target-plan",
        "schema_version": 1,
        "authorization_id": authorization["authorization_id"],
        "source_model_id": authorization["source_model_id"],
        "source_plan_id": source_plan["plan_id"],
        "target_census_id": census["census_id"],
        "required_residual_mode_from_source": required_mode,
        "source_shape_reference": source_reference,
        "minimum_score_margin": MIN_SCORE_MARGIN,
        "decisions": decisions,
        "selected_carriers": selected,
        "ranked_options": ranked,
        "ready_for_prediction_seal": len(selected) == len(TARGET_DLOS),
        "information_boundary": {
            "target_numerical_values_loaded": False,
            "target_scores_parsed": False,
            "target_performance_metric_computed": False,
            "source_semantic_vocabulary_changed": False,
            "source_residual_mode_changed": False,
            "source_group_selection_changed": False,
            "source_coefficients_changed": False,
            "ambiguous_target_carrier_auto_selected": False,
        },
    }
    result["plan_id"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = plan(
        json.loads(arguments.census.read_text(encoding="utf-8")),
        json.loads(arguments.authorization.read_text(encoding="utf-8")),
        json.loads(arguments.source_plan.read_text(encoding="utf-8")),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "plan_id": result["plan_id"],
                "ready_for_prediction_seal": result["ready_for_prediction_seal"],
                "required_residual_mode_from_source": result[
                    "required_residual_mode_from_source"
                ],
                "decisions": result["decisions"],
                "selected_carriers": result["selected_carriers"],
                "information_boundary": result["information_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
