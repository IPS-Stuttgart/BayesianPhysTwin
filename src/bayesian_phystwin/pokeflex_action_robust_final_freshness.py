"""Final public-take custody for the PokeFlex action-robust evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .pokeflex_instance_freshness import public_take_ids

FRESHNESS_AUDIT_KIND = "PokeFlexActionRobustFinalFreshTakeExclusionAudit"
FRESHNESS_AUDIT_ID = "pokeflex-action-robust-fresh2-exclusion-audit-v5"
FRESHNESS_AUDIT_SHA256 = (
    "2d7fa2d2b1147abe23e333cf80152ba43cc405b086ad8aebb9775fadf397675b"
)
FRESHNESS_AUDIT_FILE_SHA256 = (
    "32e3b95c4449bf33e06aeefff6f25581375b8ca6ee0cf1276c41efaba0fce98b"
)
PREVIOUS_FRESHNESS_AUDIT_SHA256 = (
    "8f6fb4c9c7faec43c0ff98acc1b480073fe1f2ba3c6f534036ceb1b86059e67b"
)
PREVIOUS_FRESHNESS_AUDIT_FILE_SHA256 = (
    "cffcf0cf2a0d54a2e9fbba232ca88ebeb6ea114b3fc4364d83f6548a6d7bfce5"
)
PUBLIC_INVENTORY_SHA256 = (
    "90121a8d060f50288bb52872556e38ce808de4d2a642d1691ff1cff20b5b1e96"
)
PREVIOUS_PRIOR_EXCLUSION_UNION_SHA256 = (
    "314e352e83a0c140f62f1ef3450fde53758f302d8407f05136fb06747f01e5d5"
)
PREVIOUS_SELECTED_INVENTORY_SHA256 = (
    "a33d9e86154e5a6a2429e837172a5ff6fb272f6f2912389f52d066e5b595d28e"
)
PRIOR_EXCLUSION_UNION_SHA256 = (
    "e5f15062fde9f250e40244578da82d39164abda6444e86ab47584d4696cf7f25"
)
ELIGIBLE_INVENTORY_SHA256 = (
    "c06b21fa11e8c5507246364a6da9c80097fa85c7dc6f2fafce1e0e8b1842e6e1"
)
SELECTED_INVENTORY_SHA256 = ELIGIBLE_INVENTORY_SHA256
SERVER_SCAN_CUTOFF_UTC = "2026-08-05T00:00:00Z"
SELECTED_ZIP_SHA256 = {
    "Pillow_T4": (
        "1d5cf1c344f0515faf1f3e1dd33d3bb995c9502689bfbab15e6b5c3142fe7049"
    ),
    "PlushDice_T3": (
        "3dfb9b174a5c306026d0e532ad4c1c9c6ebb4be876998b3b1e3a42c504bfc8a0"
    ),
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _inventory_sha256(values: set[str] | tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("ascii")
    ).hexdigest()


def freshness_audit_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the canonical audit without its self-referential digest."""

    canonical = dict(payload)
    canonical.pop("audit_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_final_freshness_audit(
    previous_audit: Mapping[str, Any],
    *,
    locked_at_utc: str,
) -> dict[str, Any]:
    """Bind the exhaustive two-take complement left by the v3 campaign."""

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
    previous_selected = set(previous_audit["selection"]["take_ids"])
    _require(len(previous_prior) == 108, "previous exclusion count changed")
    _require(
        _inventory_sha256(previous_prior) == PREVIOUS_PRIOR_EXCLUSION_UNION_SHA256,
        "previous exclusion inventory changed",
    )
    _require(len(previous_selected) == 6, "previous selected count changed")
    _require(
        _inventory_sha256(previous_selected) == PREVIOUS_SELECTED_INVENTORY_SHA256,
        "previous selected inventory changed",
    )
    prior = previous_prior | previous_selected
    eligible = public - prior
    selected = tuple(sorted(eligible))
    _require(len(prior) == 114, "final exclusion count changed")
    _require(len(eligible) == 2, "final eligible count changed")
    _require(set(selected) == set(SELECTED_ZIP_SHA256), "final complement changed")

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
                "the v3 exclusion union plus all six prospectively scored v3 "
                "third-panel takes"
            ),
        },
        "post_v3_exact_exposure_scan": {
            "cutoff_utc": SERVER_SCAN_CUTOFF_UTC,
            "candidate_take_count": len(eligible),
            "candidate_inventory_sha256": _inventory_sha256(eligible),
            "git_unregistered_exact_outcome_matches": [],
            "gpuserver6000_unregistered_outcome_matches": [],
            "gpuserver4090_unregistered_outcome_matches": [],
            "raw_zip_presence_is_not_outcome_exposure": True,
            "held_v8_runtime_paths_pruned": True,
            "note": (
                "Only the registered raw ZIPs were present. No prediction, score, "
                "or target-result artifact for either take was found before lock."
            ),
        },
        "eligibility": {
            "rule": "exhaustive public inventory complement after the v3 campaign",
            "eligible_take_count": len(eligible),
            "eligible_object_count": 2,
            "sorted_newline_inventory_sha256": _inventory_sha256(eligible),
            "take_ids": list(selected),
        },
        "selection": {
            "rule": "select the complete two-take eligible complement without ranking",
            "selected_take_count": len(selected),
            "sorted_newline_inventory_sha256": _inventory_sha256(selected),
            "take_ids": list(selected),
            "zip_sha256": dict(SELECTED_ZIP_SHA256),
        },
        "claim_boundary": (
            "These are the final two previously unscored public poking takes. "
            "Their physical objects were studied earlier, so they test new-action "
            "transfer, not unseen-object generalization or the published validation split."
        ),
    }
    payload["audit_sha256"] = freshness_audit_sha256(payload)
    validate_final_freshness_audit(payload, bind_registered_digest=False)
    return payload


def validate_final_freshness_audit(
    payload: Mapping[str, Any],
    *,
    bind_registered_digest: bool = True,
) -> dict[str, Any]:
    """Validate that the registered cohort is exactly the final public complement."""

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
    _require(len(prior_ids) == len(set(prior_ids)) == 114, "prior inventory changed")
    _require(
        _inventory_sha256(set(prior_ids)) == PRIOR_EXCLUSION_UNION_SHA256,
        "prior exclusion inventory changed",
    )
    scan = payload.get("post_v3_exact_exposure_scan")
    _require(isinstance(scan, Mapping), "exact exposure scan is missing")
    for key in (
        "git_unregistered_exact_outcome_matches",
        "gpuserver6000_unregistered_outcome_matches",
        "gpuserver4090_unregistered_outcome_matches",
    ):
        _require(scan.get(key) == [], "candidate take outcome was previously exposed")
    _require(
        scan.get("raw_zip_presence_is_not_outcome_exposure") is True,
        "raw archive custody statement changed",
    )
    eligibility = payload.get("eligibility")
    _require(isinstance(eligibility, Mapping), "eligibility is missing")
    eligible = tuple(str(value) for value in eligibility.get("take_ids", ()))
    _require(len(eligible) == len(set(eligible)) == 2, "eligible inventory changed")
    _require(set(eligible).isdisjoint(prior_ids), "eligible take was exposed")
    _require(
        _inventory_sha256(set(eligible)) == ELIGIBLE_INVENTORY_SHA256,
        "eligible inventory digest changed",
    )
    selection = payload.get("selection")
    _require(isinstance(selection, Mapping), "selection is missing")
    selected = tuple(str(value) for value in selection.get("take_ids", ()))
    _require(selected == tuple(sorted(eligible)), "final complement selection changed")
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
    "build_final_freshness_audit",
    "freshness_audit_sha256",
    "validate_final_freshness_audit",
]
