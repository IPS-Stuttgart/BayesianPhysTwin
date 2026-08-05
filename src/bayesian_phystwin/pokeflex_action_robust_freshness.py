"""Fresh-take custody for the action-robust PokeFlex scale evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .pokeflex_instance_freshness import public_take_ids

FRESHNESS_AUDIT_KIND = "PokeFlexActionRobustFreshTakeExclusionAudit"
FRESHNESS_AUDIT_ID = "pokeflex-action-robust-fresh6-exclusion-audit-v3"
FRESHNESS_AUDIT_SHA256 = (
    "8f6fb4c9c7faec43c0ff98acc1b480073fe1f2ba3c6f534036ceb1b86059e67b"
)
FRESHNESS_AUDIT_FILE_SHA256 = (
    "cffcf0cf2a0d54a2e9fbba232ca88ebeb6ea114b3fc4364d83f6548a6d7bfce5"
)
PREVIOUS_FRESHNESS_AUDIT_SHA256 = (
    "b9afe9cb4fe3f1e6b07a919ecd7fa204093308993b2b3cbf33887862d58c348e"
)
PREVIOUS_FRESHNESS_AUDIT_FILE_SHA256 = (
    "ce505be2565e8e720a6f6224909db93be5d56a3c64d3de8d7897ba2e58dca671"
)
PUBLIC_INVENTORY_SHA256 = (
    "90121a8d060f50288bb52872556e38ce808de4d2a642d1691ff1cff20b5b1e96"
)
GIT_REF_SNAPSHOT_SHA256 = (
    "fa29eae2a27b59cc04a19dbc1ad76d22d622fd30d479130e5f5dada215b86657"
)
GIT_REF_COUNT = 379
SERVER_SCAN_CUTOFF_UTC = "2026-08-04T18:19:58Z"
SELECTION_SALT = "pokeflex-action-robust-shrinkage-fresh6-selection-v3"
PRIOR_EXCLUSION_UNION_SHA256 = (
    "314e352e83a0c140f62f1ef3450fde53758f302d8407f05136fb06747f01e5d5"
)
ELIGIBLE_INVENTORY_SHA256 = (
    "d8a632f9d4a276a913312efbf18eb425f40c35e2345a47f0f5f09e5cf0c15181"
)
SELECTED_INVENTORY_SHA256 = (
    "a33d9e86154e5a6a2429e837172a5ff6fb272f6f2912389f52d066e5b595d28e"
)
SELECTED_ZIP_SHA256 = {
    "3dPrintedCylinder_T6": (
        "af6ae950bccb7c7ef7dc5aa3b69e7b4460d48dc20bb18e9e1ad504b07584f00c"
    ),
    "3dPrintedPizza_T1": (
        "35ff95bd1fcc851359abfa662dc69c7554146badd22c1846554b2aed388d2ae1"
    ),
    "Beanbag_T7": (
        "29d8d2c2300521313b25a4faa2eec56cbe4c9a48fe875afde203afd47cda08be"
    ),
    "FoamCylinder_T4": (
        "23e7bd666e63c2a68d4e1033d28e0b8c01a41f7b863b320f7467bd3419f4d02e"
    ),
    "Pillow_T5": (
        "3ffeb1c4a9536e32a35b3ffb4054e50502dfb57fac524a9133a9f03ada133b79"
    ),
    "PlushDice_T7": (
        "bc137a01b50994f43a72b60c20241b16070d061b89bc9482c13e5b7c1956cc31"
    ),
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _inventory_sha256(values: set[str] | tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("ascii")
    ).hexdigest()


def _object_name(take_id: str) -> str:
    object_name, separator, take_number = take_id.rpartition("_T")
    _require(bool(separator) and take_number.isdigit(), "invalid take id")
    return object_name


def freshness_audit_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("audit_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _select_per_object(eligible: set[str]) -> tuple[str, ...]:
    by_object: dict[str, list[str]] = {}
    for take_id in eligible:
        by_object.setdefault(_object_name(take_id), []).append(take_id)
    return tuple(
        sorted(
            min(
                takes,
                key=lambda take_id: hashlib.sha256(
                    SELECTION_SALT.encode("ascii") + b"\0" + take_id.encode("ascii")
                ).hexdigest(),
            )
            for takes in by_object.values()
        )
    )


def build_action_robust_freshness_audit(
    previous_audit: Mapping[str, Any],
    *,
    locked_at_utc: str,
) -> dict[str, Any]:
    """Exclude both prior campaigns and select one take per remaining object."""

    _require(
        previous_audit.get("audit_sha256") == PREVIOUS_FRESHNESS_AUDIT_SHA256,
        "previous freshness audit changed",
    )
    public = set(public_take_ids())
    _require(len(public) == 116, "public take count changed")
    _require(
        _inventory_sha256(public) == PUBLIC_INVENTORY_SHA256,
        "public inventory changed",
    )
    previous_prior = set(previous_audit["prior_exposure_audit"]["take_ids"])
    second_target = set(previous_audit["selection"]["take_ids"])
    prior = previous_prior | second_target
    eligible = public - prior
    _require(len(prior) == 108, "expanded exclusion count changed")
    _require(len(eligible) == 8, "third eligible count changed")
    selected = _select_per_object(eligible)
    _require(len(selected) == 6, "third selected cohort changed")
    _require(set(selected) == set(SELECTED_ZIP_SHA256), "selected ZIP map changed")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": FRESHNESS_AUDIT_KIND,
        "audit_id": FRESHNESS_AUDIT_ID,
        "locked_at_utc": locked_at_utc,
        "outcomes_opened_during_audit": False,
        "previous_audit_sha256": PREVIOUS_FRESHNESS_AUDIT_SHA256,
        "previous_audit_file_sha256": PREVIOUS_FRESHNESS_AUDIT_FILE_SHA256,
        "public_archive": {
            "root": "/mnt/lexar4tb/pokeflex/poking",
            "take_count": len(public),
            "sorted_newline_inventory_sha256": _inventory_sha256(public),
        },
        "prior_exposure_audit": {
            "excluded_take_count": len(prior),
            "sorted_newline_union_sha256": _inventory_sha256(prior),
            "take_ids": sorted(prior),
            "scope": (
                "the v2 exclusion union plus every take opened by the second "
                "fresh12 campaign"
            ),
        },
        "post_v2_exact_exposure_scan": {
            "cutoff_utc": SERVER_SCAN_CUTOFF_UTC,
            "candidate_take_count": len(eligible),
            "candidate_inventory_sha256": _inventory_sha256(eligible),
            "git_ref_snapshot_sha256": GIT_REF_SNAPSHOT_SHA256,
            "git_ref_count": GIT_REF_COUNT,
            "git_unregistered_exact_matches": [],
            "gpuserver6000_recent_unregistered_exact_matches": [],
            "gpuserver4090_recent_unregistered_exact_matches": [],
            "authorized_inventory_artifact": (
                "configs/sota/pokeflex_instance_fresh12_exclusion_audit_v2.json"
            ),
            "held_v8_runtime_paths_pruned": True,
            "note": (
                "The authorized v2 inventory enumerates all eight candidates; "
                "the exact scan found no other occurrence. Held-v8 is a disjoint "
                "Deform360 protocol and remained outside the ownership boundary."
            ),
        },
        "eligibility": {
            "eligible_take_count": len(eligible),
            "eligible_object_count": len({_object_name(value) for value in eligible}),
            "sorted_newline_inventory_sha256": _inventory_sha256(eligible),
            "take_ids": sorted(eligible),
        },
        "selection": {
            "rule": (
                "for each eligible object, choose the take with minimum lowercase "
                "SHA-256 of salt, NUL, and take ID"
            ),
            "salt": SELECTION_SALT,
            "selected_take_count": len(selected),
            "sorted_newline_inventory_sha256": _inventory_sha256(selected),
            "take_ids": list(selected),
            "zip_sha256": dict(SELECTED_ZIP_SHA256),
        },
        "claim_boundary": (
            "These six exact takes are a third interaction panel of already "
            "studied physical objects. They test repeated-action transfer only, "
            "not unseen-object generalization or the published validation split."
        ),
    }
    payload["audit_sha256"] = freshness_audit_sha256(payload)
    validate_action_robust_freshness_audit(
        payload,
        bind_registered_digest=False,
    )
    return payload


def validate_action_robust_freshness_audit(
    payload: Mapping[str, Any],
    *,
    bind_registered_digest: bool = True,
) -> dict[str, Any]:
    _require(payload.get("schema_version") == 1, "freshness schema changed")
    _require(
        payload.get("artifact_kind") == FRESHNESS_AUDIT_KIND,
        "freshness kind changed",
    )
    _require(payload.get("audit_id") == FRESHNESS_AUDIT_ID, "freshness id changed")
    _require(
        payload.get("audit_sha256") == freshness_audit_sha256(payload),
        "freshness checksum mismatch",
    )
    if bind_registered_digest:
        _require(bool(FRESHNESS_AUDIT_SHA256), "registered freshness audit is unset")
        _require(
            payload.get("audit_sha256") == FRESHNESS_AUDIT_SHA256,
            "registered freshness audit changed",
        )
    _require(
        payload.get("outcomes_opened_during_audit") is False,
        "freshness audit opened outcomes",
    )
    _require(
        payload.get("previous_audit_sha256") == PREVIOUS_FRESHNESS_AUDIT_SHA256,
        "previous freshness digest changed",
    )
    _require(
        payload.get("previous_audit_file_sha256")
        == PREVIOUS_FRESHNESS_AUDIT_FILE_SHA256,
        "previous freshness bytes changed",
    )
    public = payload.get("public_archive")
    _require(isinstance(public, Mapping), "public inventory is missing")
    _require(int(public.get("take_count", -1)) == 116, "public take count changed")
    _require(
        public.get("sorted_newline_inventory_sha256") == PUBLIC_INVENTORY_SHA256,
        "public inventory digest changed",
    )
    prior = payload.get("prior_exposure_audit")
    _require(isinstance(prior, Mapping), "prior exposure audit is missing")
    prior_ids = tuple(str(value) for value in prior.get("take_ids", ()))
    _require(len(prior_ids) == len(set(prior_ids)) == 108, "prior inventory changed")
    _require(
        _inventory_sha256(set(prior_ids)) == PRIOR_EXCLUSION_UNION_SHA256,
        "prior exclusion inventory changed",
    )
    scan = payload.get("post_v2_exact_exposure_scan")
    _require(isinstance(scan, Mapping), "exact exposure scan is missing")
    _require(
        scan.get("git_ref_snapshot_sha256") == GIT_REF_SNAPSHOT_SHA256,
        "Git ref snapshot changed",
    )
    _require(int(scan.get("git_ref_count", -1)) == GIT_REF_COUNT, "Git ref count changed")
    for key in (
        "git_unregistered_exact_matches",
        "gpuserver6000_recent_unregistered_exact_matches",
        "gpuserver4090_recent_unregistered_exact_matches",
    ):
        _require(scan.get(key) == [], "candidate take was previously exposed")
    eligibility = payload.get("eligibility")
    _require(isinstance(eligibility, Mapping), "eligibility is missing")
    eligible = tuple(str(value) for value in eligibility.get("take_ids", ()))
    _require(len(eligible) == len(set(eligible)) == 8, "eligible inventory changed")
    _require(set(eligible).isdisjoint(prior_ids), "eligible take was exposed")
    _require(
        _inventory_sha256(set(eligible)) == ELIGIBLE_INVENTORY_SHA256,
        "eligible inventory digest changed",
    )
    selection = payload.get("selection")
    _require(isinstance(selection, Mapping), "selection is missing")
    selected = tuple(str(value) for value in selection.get("take_ids", ()))
    _require(selected == _select_per_object(set(eligible)), "selection changed")
    _require(
        _inventory_sha256(set(selected)) == SELECTED_INVENTORY_SHA256,
        "selected inventory changed",
    )
    _require(
        dict(selection.get("zip_sha256", {})) == SELECTED_ZIP_SHA256,
        "selected ZIP bytes changed",
    )
    return {
        "passed": True,
        "audit_sha256": payload["audit_sha256"],
        "target_take_ids": selected,
    }


__all__ = [
    "ELIGIBLE_INVENTORY_SHA256",
    "FRESHNESS_AUDIT_FILE_SHA256",
    "FRESHNESS_AUDIT_ID",
    "FRESHNESS_AUDIT_KIND",
    "FRESHNESS_AUDIT_SHA256",
    "PRIOR_EXCLUSION_UNION_SHA256",
    "SELECTED_INVENTORY_SHA256",
    "SELECTED_ZIP_SHA256",
    "SELECTION_SALT",
    "build_action_robust_freshness_audit",
    "freshness_audit_sha256",
    "validate_action_robust_freshness_audit",
]
