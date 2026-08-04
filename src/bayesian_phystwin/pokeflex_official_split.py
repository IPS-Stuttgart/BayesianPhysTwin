"""Outcome-blind provenance audit for the PokeFlex validation split.

The paper evaluator names recordings in an internal namespace, while the public
archive uses renamed objects and, for five recordings, does not contain the
requested take number.  This module resolves exact public filenames only.  It
never guesses that a lower-numbered public take is an internal validation take.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/pokeflex-dataset/reconstruction"
UPSTREAM_MAIN_COMMIT = "aaa8726072834a95bbe97e1a113588968c36e185"
UPSTREAM_MAIN_EVALUATOR_SHA256 = (
    "ea1854ba5224b8aec2e8ba6b80fb762eba7314b925e87ca7775d810003615b60"
)
UPSTREAM_PUBLIC_ALIGNMENT_COMMIT = "fa484b0fa94f59f51e8c5f2293a6b1bc378b7375"
UPSTREAM_PUBLIC_ALIGNMENT_EVALUATOR_SHA256 = (
    "44df8ddb288a30882bdc4bd1e809ee4d0a805e96e7fb80bbabf2665b87371267"
)
UPSTREAM_PUBLIC_ALIGNMENT_PREPROCESSING_SHA256 = (
    "fd0eb5b9d7129f575734f906af4481596779f893fd5af76b5e0d230b68201b8a"
)

# Exact order from upstream main:test/evaluate.py.
OFFICIAL_LEGACY_TAKE_IDS = (
    "MemoryFoam_T2",
    "Volleyball_T4",
    "HalfSphere_T3",
    "Bunny_T1",
    "Pyramid_T6",
    "Dice_T3",
    "Moon_T1",
    "Octopus_T6",
    "PlushDice_T8",
    "Turtle_T3",
    "Fjadrar_T8",
    "Cylinder_T7",
    "Beanbag_T6",
    "Heart_T14",
    "Foam_T1",
    "ToiletPaper_T1",
    "Sponge_T10",
    "Pizza_T13",
)

# Object-identity projection into the public release namespace.  Take numbers
# remain unchanged; changing them would be an unsupported recording mapping.
PUBLIC_OBJECT_BY_LEGACY_OBJECT = {
    "MemoryFoam": "MemoryFoam",
    "Volleyball": "PlushVolleyball",
    "HalfSphere": "FoamHalfSphere",
    "Bunny": "3dPrintedBunny",
    "Pyramid": "3dPrintedPyramid",
    "Dice": "FoamDice",
    "Moon": "PlushMoon",
    "Octopus": "PlushOctopus",
    "PlushDice": "PlushDice",
    "Turtle": "PlushTurtle",
    "Fjadrar": "Pillow",
    "Cylinder": "3dPrintedCylinder",
    "Beanbag": "Beanbag",
    "Heart": "3dPrintedHeart",
    "Foam": "FoamCylinder",
    "ToiletPaper": "ToiletPaperRoll",
    "Sponge": "Sponge",
    "Pizza": "3dPrintedPizza",
}

PUBLIC_ALIGNMENT_EXAMPLE_TAKE_IDS = ("FoamDice_T3",)
_TAKE_PATTERN = re.compile(r"^(?P<object>.+)_T(?P<take>[1-9][0-9]*)$")


def _split_take_id(take_id: str) -> tuple[str, int]:
    match = _TAKE_PATTERN.fullmatch(take_id)
    if match is None:
        raise ValueError(f"invalid PokeFlex take ID: {take_id!r}")
    return match.group("object"), int(match.group("take"))


def public_take_id(legacy_take_id: str) -> str:
    """Project an evaluator ID into the public object namespace.

    This function deliberately preserves the take number.  It is an object-name
    projection, not an internal-to-public recording mapping.
    """

    legacy_object, take_number = _split_take_id(legacy_take_id)
    try:
        public_object = PUBLIC_OBJECT_BY_LEGACY_OBJECT[legacy_object]
    except KeyError as exc:
        raise ValueError(f"unknown legacy PokeFlex object: {legacy_object!r}") from exc
    return f"{public_object}_T{take_number}"


OFFICIAL_PUBLIC_CANDIDATE_TAKE_IDS = tuple(
    public_take_id(take_id) for take_id in OFFICIAL_LEGACY_TAKE_IDS
)


def canonical_audit_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an audit while excluding its self-referential digest field."""

    canonical = dict(payload)
    canonical.pop("audit_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_public_take_ids(
    public_root: str | Path,
    *,
    require_nonempty_archives: bool = True,
) -> tuple[str, ...]:
    """Discover syntactically valid take IDs from a public archive tree."""

    root = Path(public_root)
    if not root.is_dir():
        raise FileNotFoundError(f"PokeFlex public root not found: {root}")

    take_ids: set[str] = set()
    for archive in root.rglob("*.zip"):
        if require_nonempty_archives and archive.stat().st_size == 0:
            continue
        _split_take_id(archive.stem)
        take_ids.add(archive.stem)
    return tuple(sorted(take_ids))


def audit_official_split(public_take_ids: Iterable[str]) -> dict[str, Any]:
    """Build a deterministic, outcome-blind exact-filename audit."""

    inventory_list = list(public_take_ids)
    if len(inventory_list) != len(set(inventory_list)):
        raise ValueError("public PokeFlex inventory contains duplicate take IDs")
    inventory = set(inventory_list)

    max_take_by_object: dict[str, int] = {}
    for take_id in inventory:
        object_name, take_number = _split_take_id(take_id)
        max_take_by_object[object_name] = max(
            take_number,
            max_take_by_object.get(object_name, 0),
        )

    cases: list[dict[str, Any]] = []
    exact_matches: list[str] = []
    unresolved: list[str] = []
    for legacy_take_id, candidate_take_id in zip(
        OFFICIAL_LEGACY_TAKE_IDS,
        OFFICIAL_PUBLIC_CANDIDATE_TAKE_IDS,
        strict=True,
    ):
        public_object, requested_take = _split_take_id(candidate_take_id)
        present = candidate_take_id in inventory
        cases.append(
            {
                "legacy_evaluator_take_id": legacy_take_id,
                "public_identity_projection": candidate_take_id,
                "exact_public_filename_present": present,
                "requested_take_number": requested_take,
                "maximum_available_take_number": max_take_by_object.get(public_object),
            }
        )
        (exact_matches if present else unresolved).append(candidate_take_id)

    materializable = not unresolved
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexOfficialSplitSourceAudit",
        "evidence_boundary": (
            "Archive names and upstream source only; no target mesh, prediction, "
            "or metric outcome is read."
        ),
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "main_commit": UPSTREAM_MAIN_COMMIT,
            "main_evaluator_sha256": UPSTREAM_MAIN_EVALUATOR_SHA256,
            "public_alignment_branch_commit": UPSTREAM_PUBLIC_ALIGNMENT_COMMIT,
            "public_alignment_evaluator_sha256": (
                UPSTREAM_PUBLIC_ALIGNMENT_EVALUATOR_SHA256
            ),
            "public_alignment_preprocessing_sha256": (
                UPSTREAM_PUBLIC_ALIGNMENT_PREPROCESSING_SHA256
            ),
            "public_alignment_evaluation_take_ids": list(
                PUBLIC_ALIGNMENT_EXAMPLE_TAKE_IDS
            ),
            "public_alignment_scope": (
                "The official testing-scripts branch replaces the internal 18-take "
                "evaluation list with FoamDice_T3; it does not publish an "
                "internal-to-public mapping for the remaining recordings."
            ),
        },
        "public_inventory": {
            "take_count": len(inventory),
            "maximum_take_number_by_object": dict(sorted(max_take_by_object.items())),
        },
        "official_split": {
            "case_count": len(cases),
            "cases": cases,
            "exact_public_match_count": len(exact_matches),
            "exact_public_match_take_ids": exact_matches,
            "unresolved_count": len(unresolved),
            "unresolved_public_identity_projections": unresolved,
            "exact_public_split_materializable": materializable,
        },
        "author_data_request": {
            "purpose": (
                "Reproduce the published 18-object validation aggregate without "
                "guessing recording identities."
            ),
            "preferred_response": (
                "A mapping for all 18 legacy evaluator IDs to public archive IDs, "
                "with SHA-256 for every archive."
            ),
            "minimum_unresolved_legacy_take_ids": [
                case["legacy_evaluator_take_id"]
                for case in cases
                if not case["exact_public_filename_present"]
            ],
            "minimum_unresolved_public_identity_projections": unresolved,
            "alternative_response": (
                "The processed validation set used for the paper, accompanied by a "
                "file manifest and SHA-256 digests."
            ),
            "not_acceptable": (
                "A guessed lower-numbered public take, outcome-selected replacement, "
                "or unverified filename alias."
            ),
        },
        "decision": (
            "full_official_split_available"
            if materializable
            else "author_mapping_or_processed_validation_set_required"
        ),
    }
    payload["audit_sha256"] = canonical_audit_sha256(payload)
    return payload


def write_official_split_audit(
    public_root: str | Path,
    output_path: str | Path,
    *,
    require_nonempty_archives: bool = True,
) -> dict[str, Any]:
    """Discover, audit, and persist a canonical source artifact."""

    take_ids = discover_public_take_ids(
        public_root,
        require_nonempty_archives=require_nonempty_archives,
    )
    payload = audit_official_split(take_ids)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "OFFICIAL_LEGACY_TAKE_IDS",
    "OFFICIAL_PUBLIC_CANDIDATE_TAKE_IDS",
    "PUBLIC_ALIGNMENT_EXAMPLE_TAKE_IDS",
    "UPSTREAM_MAIN_COMMIT",
    "UPSTREAM_PUBLIC_ALIGNMENT_COMMIT",
    "audit_official_split",
    "canonical_audit_sha256",
    "discover_public_take_ids",
    "public_take_id",
    "write_official_split_audit",
]
