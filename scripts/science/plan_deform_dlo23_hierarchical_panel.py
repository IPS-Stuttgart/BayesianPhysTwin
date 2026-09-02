#!/usr/bin/env python3
"""Plan a real DLO2/DLO3 residual-panel adapter from a source-only census.

The planner never opens DLO4/DLO5. It verifies every retained source artifact by
SHA-256, inspects only safe NumPy carriers, classifies array semantics using a
frozen vocabulary, and emits ranked adapter alternatives. It does not silently
choose a carrier unless one candidate per source DLO is structurally complete
and separated from the runner-up by a frozen score margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SOURCE_DLOS = ("DLO2", "DLO3")
FORBIDDEN_TOKENS = ("dlo4", "dlo_4", "dlo-4", "dlo5", "dlo_5", "dlo-5", "dlo45", "33361441865")
MIN_ARRAY_ELEMENTS = 64
MIN_SCORE_MARGIN = 5
ROLE_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "residual": ("residual", "discrep", "correction", "delta"),
    "observation": ("ground_truth", "groundtruth", "truth", "observ", "measur", "reference", "real", "target"),
    "physical": ("physical", "baseline", "simulat", "rollout", "prediction_phys", "pred_phys"),
    "state": ("state", "position", "node", "point", "coord", "configuration", "trajectory"),
    "action": ("action", "command", "control", "input", "actuat", "endpoint", "end_effector", "robot"),
    "contact": ("contact", "force", "wrench", "torque", "grip"),
    "trajectory_id": ("trajectory_id", "trajectory_name", "case_id", "case_name", "sequence_id", "episode_id"),
    "time": ("time", "timestamp", "step"),
    "coefficient": ("coefficient", "coeff", "weight", "parameter"),
}


class PlanError(RuntimeError):
    """Raised when a source adapter cannot be planned without guessing."""


@dataclass(frozen=True)
class ArrayDescriptor:
    file: Path
    key: str
    shape: tuple[int, ...]
    dtype: str
    elements: int
    roles: tuple[str, ...]

    @property
    def identifier(self) -> str:
        return f"{self.file.as_posix()}::{self.key}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def forbidden(path: Path) -> bool:
    lower = path.as_posix().lower()
    return any(token in lower for token in FORBIDDEN_TOKENS)


def dlo_identity(path: Path, metadata_text: str = "") -> str | None:
    lower = (path.as_posix() + " " + metadata_text).lower()
    identities = []
    for dlo in SOURCE_DLOS:
        number = dlo[-1]
        patterns = (dlo.lower(), f"dlo_{number}", f"dlo-{number}")
        if any(pattern in lower for pattern in patterns):
            identities.append(dlo)
    return identities[0] if len(set(identities)) == 1 else None


def classify(name: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())
    roles = []
    for role, patterns in ROLE_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            roles.append(role)
    return tuple(roles)


def numeric_descriptors(path: Path) -> list[ArrayDescriptor]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype.kind not in "biufc" or array.size < MIN_ARRAY_ELEMENTS:
            return []
        return [
            ArrayDescriptor(
                file=path,
                key=path.stem,
                shape=tuple(int(value) for value in array.shape),
                dtype=str(array.dtype),
                elements=int(array.size),
                roles=classify(path.stem),
            )
        ]
    if path.suffix.lower() != ".npz":
        return []
    descriptors = []
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            array = archive[key]
            if array.dtype.kind not in "biufc" or array.size < MIN_ARRAY_ELEMENTS:
                continue
            descriptors.append(
                ArrayDescriptor(
                    file=path,
                    key=key,
                    shape=tuple(int(value) for value in array.shape),
                    dtype=str(array.dtype),
                    elements=int(array.size),
                    roles=classify(key),
                )
            )
    return descriptors


def compatible_shape(left: ArrayDescriptor, right: ArrayDescriptor) -> bool:
    if left.shape == right.shape:
        return True
    # Allow a leading singleton or trajectory dimension when the remaining
    # sample/geometry dimensions are identical. The executor must preserve the
    # leading trajectory identity rather than flattening it into i.i.d. rows.
    if len(left.shape) == len(right.shape) + 1 and left.shape[1:] == right.shape:
        return True
    if len(right.shape) == len(left.shape) + 1 and right.shape[1:] == left.shape:
        return True
    return False


def role_candidates(descriptors: Iterable[ArrayDescriptor], role: str) -> list[ArrayDescriptor]:
    return [descriptor for descriptor in descriptors if role in descriptor.roles]


def array_record(value: ArrayDescriptor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "file": value.file.as_posix(),
        "key": value.key,
        "shape": list(value.shape),
        "dtype": value.dtype,
        "elements": value.elements,
        "roles": list(value.roles),
        "identifier": value.identifier,
    }


def carrier_options(path: Path, descriptors: list[ArrayDescriptor]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    residuals = role_candidates(descriptors, "residual")
    observations = role_candidates(descriptors, "observation")
    physicals = [
        item for item in descriptors
        if "physical" in item.roles and "residual" not in item.roles
    ]
    states = role_candidates(descriptors, "state")
    actions = role_candidates(descriptors, "action")
    contacts = role_candidates(descriptors, "contact")
    trajectory_ids = role_candidates(descriptors, "trajectory_id")
    times = role_candidates(descriptors, "time")

    for residual in residuals:
        state_matches = [state for state in states if compatible_shape(residual, state)]
        physical_matches = [value for value in physicals if compatible_shape(residual, value)]
        feature = physical_matches[0] if physical_matches else (state_matches[0] if state_matches else None)
        score = 24 + (8 if feature is not None else 0)
        score += 4 if trajectory_ids else 0
        score += 3 if actions else 0
        score += 3 if contacts else 0
        options.append(
            {
                "mode": "explicit-residual",
                "score": score,
                "residual": array_record(residual),
                "observation": None,
                "physical": array_record(feature),
                "state": array_record(state_matches[0] if state_matches else None),
                "action": array_record(actions[0] if actions else None),
                "contact": array_record(contacts[0] if contacts else None),
                "trajectory_id": array_record(trajectory_ids[0] if trajectory_ids else None),
                "time": array_record(times[0] if times else None),
                "residual_definition": "stored residual array",
            }
        )

    for observation in observations:
        for physical in physicals:
            if not compatible_shape(observation, physical):
                continue
            state_matches = [state for state in states if compatible_shape(observation, state)]
            score = 28
            score += 4 if trajectory_ids else 0
            score += 3 if actions else 0
            score += 3 if contacts else 0
            options.append(
                {
                    "mode": "observation-minus-physical",
                    "score": score,
                    "residual": None,
                    "observation": array_record(observation),
                    "physical": array_record(physical),
                    "state": array_record(state_matches[0] if state_matches else physical),
                    "action": array_record(actions[0] if actions else None),
                    "contact": array_record(contacts[0] if contacts else None),
                    "trajectory_id": array_record(trajectory_ids[0] if trajectory_ids else None),
                    "time": array_record(times[0] if times else None),
                    "residual_definition": "observation - physical prediction",
                }
            )

    for option in options:
        option["carrier_file"] = path.as_posix()
    options.sort(key=lambda value: (-value["score"], value["mode"], json.dumps(value, sort_keys=True)))
    return options


def plan(census: Mapping[str, Any]) -> dict[str, Any]:
    boundary = census.get("information_boundary", {})
    for key in (
        "dlo4_directory_children_enumerated",
        "dlo4_payload_read",
        "dlo5_directory_children_enumerated",
        "dlo5_payload_read",
        "protected_parent_run_33361441865_artifact_read",
        "target_scores_read",
        "target_dependent_model_selection",
    ):
        if boundary.get(key) is not False:
            raise PlanError(f"input census does not prove source-only custody: {key}")

    records = census.get("ranked_candidates", [])
    by_dlo: dict[str, list[dict[str, Any]]] = {dlo: [] for dlo in SOURCE_DLOS}
    coefficient_candidates = []
    backend_candidates = []
    rejected = []
    for record in records:
        path = Path(record["path"])
        if forbidden(path):
            raise PlanError(f"forbidden target path escaped census: {path}")
        if not path.is_file() or path.is_symlink():
            rejected.append({"path": path.as_posix(), "reason": "missing-or-symlink"})
            continue
        if sha256(path) != record["sha256"]:
            raise PlanError(f"source artifact changed after census: {path}")
        metadata_text = json.dumps(record.get("schema", {}), sort_keys=True)
        identity = dlo_identity(path, metadata_text)
        lower = (path.as_posix() + " " + metadata_text).lower()
        if "coeff" in lower or "coefficient" in lower:
            coefficient_candidates.append({"path": path.as_posix(), "sha256": record["sha256"], "dlo": identity})
        if "pyelastica" in lower or "alternate" in lower or "backend" in lower:
            backend_candidates.append({"path": path.as_posix(), "sha256": record["sha256"], "dlo": identity})
        if identity is None or path.suffix.lower() not in {".npz", ".npy"}:
            continue
        try:
            descriptors = numeric_descriptors(path)
        except Exception as error:
            rejected.append({"path": path.as_posix(), "reason": f"numeric-inspection:{error!r}"})
            continue
        for option in carrier_options(path, descriptors):
            option["dlo"] = identity
            option["carrier_sha256"] = record["sha256"]
            by_dlo[identity].append(option)

    selected: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for dlo, options in by_dlo.items():
        options.sort(key=lambda value: (-value["score"], value["carrier_file"], value["mode"]))
        top = options[0] if options else None
        runner_up = options[1] if len(options) > 1 else None
        margin = None if top is None else top["score"] - (runner_up["score"] if runner_up else 0)
        structurally_complete = bool(top and top["physical"] is not None)
        trajectory_identifiable = bool(
            top and (
                top["trajectory_id"] is not None
                or len(top["physical"]["shape"]) >= 3
                or re.search(r"(?:trajectory|case|sequence|run|rollout)[-_]?\d+", top["carrier_file"], re.IGNORECASE)
            )
        )
        unambiguous = bool(
            top
            and structurally_complete
            and trajectory_identifiable
            and (runner_up is None or margin is not None and margin >= MIN_SCORE_MARGIN)
        )
        decisions[dlo] = {
            "candidate_count": len(options),
            "top_score": None if top is None else top["score"],
            "runner_up_score": None if runner_up is None else runner_up["score"],
            "score_margin": margin,
            "structurally_complete": structurally_complete,
            "trajectory_identifiable": trajectory_identifiable,
            "unambiguous": unambiguous,
            "reason": (
                "selected"
                if unambiguous
                else "no candidate"
                if top is None
                else "missing physical/state feature carrier"
                if not structurally_complete
                else "trajectory identity not recoverable"
                if not trajectory_identifiable
                else "top alternatives are not separated by frozen margin"
            ),
        }
        if unambiguous:
            selected[dlo] = top
        by_dlo[dlo] = options[:100]

    result = {
        "schema": "bayesian-phystwin.deform-dlo23-hierarchical-panel-plan",
        "schema_version": 1,
        "source_census_id": census["census_id"],
        "source_dlos": list(SOURCE_DLOS),
        "frozen_semantic_vocabulary": {key: list(value) for key, value in ROLE_PATTERNS.items()},
        "minimum_score_margin": MIN_SCORE_MARGIN,
        "decisions": decisions,
        "selected_carriers": selected,
        "ranked_options": by_dlo,
        "coefficient_candidates": coefficient_candidates,
        "alternate_backend_candidates": backend_candidates,
        "rejected_candidates": rejected,
        "ready_for_panel_build": bool(len(selected) == len(SOURCE_DLOS)),
        "information_boundary": {
            "dlo2_dlo3_source_arrays_inspected": True,
            "dlo4_payload_read": False,
            "dlo5_payload_read": False,
            "protected_parent_target_result_read": False,
            "target_outcome_used_for_semantic_mapping": False,
            "ambiguous_carrier_auto_selected": False,
        },
    }
    result["plan_id"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    census = json.loads(arguments.census.read_text(encoding="utf-8"))
    result = plan(census)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "plan_id": result["plan_id"],
                "ready_for_panel_build": result["ready_for_panel_build"],
                "decisions": result["decisions"],
                "selected_carriers": result["selected_carriers"],
                "coefficient_candidate_count": len(result["coefficient_candidates"]),
                "alternate_backend_candidate_count": len(result["alternate_backend_candidates"]),
                "information_boundary": result["information_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
