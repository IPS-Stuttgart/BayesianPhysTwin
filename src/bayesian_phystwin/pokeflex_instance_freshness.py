"""Fresh-take custody for the second PokeFlex transfer evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

FRESHNESS_AUDIT_KIND = "PokeFlexInstanceFreshTakeExclusionAudit"
FRESHNESS_AUDIT_ID = "pokeflex-instance-fresh12-exclusion-audit-v2"
FRESHNESS_AUDIT_SHA256 = (
    "b9afe9cb4fe3f1e6b07a919ecd7fa204093308993b2b3cbf33887862d58c348e"
)
FRESHNESS_AUDIT_FILE_SHA256 = (
    "ce505be2565e8e720a6f6224909db93be5d56a3c64d3de8d7897ba2e58dca671"
)
PREVIOUS_FRESHNESS_AUDIT_SHA256 = (
    "fa2062f97e3ae496705717b7e2851b64b748a0417c270fea5649b9aa8cc96bbc"
)
PUBLIC_INVENTORY_SHA256 = (
    "90121a8d060f50288bb52872556e38ce808de4d2a642d1691ff1cff20b5b1e96"
)
PREVIOUS_ELIGIBLE_INVENTORY_SHA256 = (
    "f4b605ac1fdf36f341789698d47c8657f32861426766a1e882996abe713e4923"
)
GIT_REF_SNAPSHOT_SHA256 = (
    "5696fce3b2c6ccbbbe1892192485bbb8e4dcd6dc26a107595a482ad633d51e06"
)
GIT_REF_COUNT = 375
SELECTION_SALT = "pokeflex-instance-shrinkage-fresh12-selection-v2"
PRIOR_EXCLUSION_UNION_SHA256 = (
    "6cb1c809f70411fd0c35f30ae5bf891c8d1f87646e9cbf2ad27b5c428c3ac615"
)
ELIGIBLE_INVENTORY_SHA256 = (
    "89b7b30a6325dc1789984c1101dd98b9e8ef2b7349946d7bab9a913d0f7baa9f"
)
SELECTED_INVENTORY_SHA256 = (
    "e5020f7dc9bcc2a47a35bdc6b6c0b24f55f20ee43b5ebd10d7af08983a25fa43"
)

PUBLIC_OBJECT_TAKE_COUNTS = {
    "3dPrintedBunny": 7,
    "3dPrintedCylinder": 6,
    "3dPrintedHeart": 6,
    "3dPrintedPizza": 6,
    "3dPrintedPyramid": 6,
    "Beanbag": 7,
    "FoamCylinder": 7,
    "FoamDice": 8,
    "FoamHalfSphere": 6,
    "MemoryFoam": 6,
    "Pillow": 7,
    "PlushDice": 8,
    "PlushMoon": 6,
    "PlushOctopus": 7,
    "PlushTurtle": 6,
    "PlushVolleyball": 6,
    "Sponge": 5,
    "ToiletPaperRoll": 6,
}

SELECTED_ZIP_SHA256 = {
    "3dPrintedCylinder_T1": (
        "ffcd4b36ac501b2e2c96f08e84d4ae0469a4a0718724a2cfae7c1dc954808f1d"
    ),
    "3dPrintedPizza_T4": (
        "73038bcfe296840d53008003bde9bf9874d6f38689ddaaceb05d01c1736ef554"
    ),
    "3dPrintedPyramid_T5": (
        "ed55ed9cebeb89c01c5ccf6223a94fb14003c4da712e6a29b6523f880c37a911"
    ),
    "Beanbag_T1": ("01d55155dc1356181f9ae3654d8f37edf95c201881e3be4d58fc70c4b48841d8"),
    "FoamCylinder_T6": (
        "2e2e5948052d9e1f4dfd669c433109d2e75d9482ab170bf8b3f62399f25ab3a4"
    ),
    "FoamHalfSphere_T6": (
        "55df53071b45f513d98b5a91aa398b0c45728d9cb42259d579a4e43ccd4864a5"
    ),
    "Pillow_T3": ("eeee6794bbf3631b9eb05fbbe2158653dab0ec0f8dfb50996c775a7441a72f40"),
    "PlushDice_T6": (
        "469eb374d4447fdc96009c5410f5f3222eb85dd4bd83cb22c1a26573ad06b2af"
    ),
    "PlushMoon_T4": (
        "a2b54585572ee26f0a33507ae8363cdf426fc2c0ef5206cce763708f14e32643"
    ),
    "PlushTurtle_T6": (
        "40b322971a20fce9713c55e5f565103ef9e1ee253bfb3923ddb13e9398d0e5ce"
    ),
    "PlushVolleyball_T6": (
        "eb29a341810b514baecffa25e727adaf7bf18857a94f25fbeb69ac2cf9a37385"
    ),
    "Sponge_T5": ("02d1d3ce94c2dceac3f339fbae7e63326fddad7553a6b81767a5c1d7c291ddd8"),
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _inventory_sha256(values: set[str] | tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("ascii")
    ).hexdigest()


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


def public_take_ids() -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{object_name}_T{index}"
            for object_name, count in PUBLIC_OBJECT_TAKE_COUNTS.items()
            for index in range(1, count + 1)
        )
    )


def _object_name(take_id: str) -> str:
    object_name, separator, take_number = take_id.rpartition("_T")
    _require(bool(separator) and take_number.isdigit(), "invalid take id")
    return object_name


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


def build_instance_freshness_audit(
    previous_audit: Mapping[str, Any],
    *,
    locked_at_utc: str,
) -> dict[str, Any]:
    """Build the second-cohort lock from the first frozen freshness audit."""

    _require(
        previous_audit.get("audit_sha256") == PREVIOUS_FRESHNESS_AUDIT_SHA256,
        "previous freshness audit changed",
    )
    public = set(public_take_ids())
    _require(len(public) == 116, "public take count changed")
    _require(
        _inventory_sha256(public) == PUBLIC_INVENTORY_SHA256, "public inventory changed"
    )

    prior = set(previous_audit["prior_exposure_audit"]["take_ids"])
    first_target = set(previous_audit["selection"]["take_ids"])
    first_eligible = public - prior
    _require(
        _inventory_sha256(first_eligible) == PREVIOUS_ELIGIBLE_INVENTORY_SHA256,
        "previous eligible inventory changed",
    )
    expanded_prior = prior | first_target
    eligible = public - expanded_prior
    _require(len(expanded_prior) == 96, "expanded exclusion count changed")
    _require(len(eligible) == 20, "second eligible count changed")
    selected = _select_per_object(eligible)
    _require(len(selected) == 12, "second selected cohort changed")
    _require(set(selected) == set(SELECTED_ZIP_SHA256), "selected ZIP map changed")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": FRESHNESS_AUDIT_KIND,
        "audit_id": FRESHNESS_AUDIT_ID,
        "locked_at_utc": locked_at_utc,
        "outcomes_opened_during_audit": False,
        "previous_audit_sha256": PREVIOUS_FRESHNESS_AUDIT_SHA256,
        "public_archive": {
            "root": "/mnt/lexar4tb/pokeflex/poking",
            "take_count": len(public),
            "sorted_newline_inventory_sha256": _inventory_sha256(public),
        },
        "prior_exposure_audit": {
            "excluded_take_count": len(expanded_prior),
            "sorted_newline_union_sha256": _inventory_sha256(expanded_prior),
            "take_ids": sorted(expanded_prior),
            "scope": (
                "the original v1 exposure union plus every take opened by the "
                "first fresh12 campaign"
            ),
        },
        "post_v1_exact_exposure_scan": {
            "cutoff_utc": previous_audit["locked_at_utc"],
            "candidate_take_count": len(eligible),
            "candidate_inventory_sha256": _inventory_sha256(eligible),
            "git_ref_snapshot_sha256": GIT_REF_SNAPSHOT_SHA256,
            "git_ref_count": GIT_REF_COUNT,
            "git_exact_matches": [],
            "gpuserver6000_recent_exact_matches": [],
            "gpuserver4090_recent_exact_matches": [],
            "held_v8_runtime_paths_pruned": True,
            "note": (
                "held-v8 is a disjoint Deform360 protocol and remained outside "
                "the scan ownership boundary"
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
            "The selected take IDs were not present in the original exposure "
            "union, the first fresh12 target, current tracked Git ref tips, or "
            "recent server-side text provenance. They are new interactions of "
            "previously studied physical objects, not unseen objects."
        ),
    }
    payload["audit_sha256"] = freshness_audit_sha256(payload)
    validate_instance_freshness_audit(payload)
    return payload


def validate_instance_freshness_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == 1, "freshness schema changed")
    _require(
        payload.get("artifact_kind") == FRESHNESS_AUDIT_KIND, "freshness kind changed"
    )
    _require(payload.get("audit_id") == FRESHNESS_AUDIT_ID, "freshness id changed")
    _require(
        payload.get("audit_sha256") == freshness_audit_sha256(payload),
        "freshness checksum mismatch",
    )
    _require(
        payload.get("outcomes_opened_during_audit") is False, "audit opened outcomes"
    )
    public = payload.get("public_archive")
    _require(isinstance(public, Mapping), "public inventory is missing")
    _require(
        public.get("sorted_newline_inventory_sha256") == PUBLIC_INVENTORY_SHA256,
        "public digest changed",
    )
    _require(int(public.get("take_count", -1)) == 116, "public take count changed")
    prior = payload.get("prior_exposure_audit")
    _require(isinstance(prior, Mapping), "prior exposure audit is missing")
    prior_take_ids = tuple(str(value) for value in prior.get("take_ids", ()))
    _require(len(prior_take_ids) == 96, "prior exclusion count changed")
    _require(len(set(prior_take_ids)) == 96, "prior exclusion contains duplicates")
    _require(
        prior.get("sorted_newline_union_sha256") == PRIOR_EXCLUSION_UNION_SHA256,
        "prior exclusion digest changed",
    )
    _require(
        _inventory_sha256(set(prior_take_ids)) == PRIOR_EXCLUSION_UNION_SHA256,
        "prior exclusion inventory changed",
    )
    exposure = payload.get("post_v1_exact_exposure_scan")
    _require(isinstance(exposure, Mapping), "exposure scan is missing")
    for key in (
        "git_exact_matches",
        "gpuserver6000_recent_exact_matches",
        "gpuserver4090_recent_exact_matches",
    ):
        _require(exposure.get(key) == [], "second-cohort take was previously exposed")
    selection = payload.get("selection")
    _require(isinstance(selection, Mapping), "fresh selection is missing")
    eligibility = payload.get("eligibility")
    _require(isinstance(eligibility, Mapping), "fresh eligibility is missing")
    eligible = tuple(str(value) for value in eligibility.get("take_ids", ()))
    _require(len(eligible) == 20, "eligible take count changed")
    _require(len(set(eligible)) == 20, "eligible inventory contains duplicates")
    _require(set(eligible).isdisjoint(prior_take_ids), "eligible take was exposed")
    _require(
        eligibility.get("sorted_newline_inventory_sha256") == ELIGIBLE_INVENTORY_SHA256,
        "eligible inventory digest changed",
    )
    _require(
        _inventory_sha256(set(eligible)) == ELIGIBLE_INVENTORY_SHA256,
        "eligible inventory changed",
    )
    selected = tuple(selection.get("take_ids", ()))
    _require(
        selected == _select_per_object(set(eligible)),
        "selection changed",
    )
    _require(
        selection.get("sorted_newline_inventory_sha256") == SELECTED_INVENTORY_SHA256,
        "selected inventory digest changed",
    )
    _require(
        _inventory_sha256(set(selected)) == SELECTED_INVENTORY_SHA256,
        "selected inventory changed",
    )
    _require(
        dict(selection.get("zip_sha256", {})) == SELECTED_ZIP_SHA256,
        "ZIP bytes changed",
    )
    return {
        "passed": True,
        "audit_sha256": payload["audit_sha256"],
        "target_take_ids": selected,
    }


__all__ = [
    "FRESHNESS_AUDIT_ID",
    "FRESHNESS_AUDIT_FILE_SHA256",
    "FRESHNESS_AUDIT_KIND",
    "FRESHNESS_AUDIT_SHA256",
    "ELIGIBLE_INVENTORY_SHA256",
    "PRIOR_EXCLUSION_UNION_SHA256",
    "SELECTED_ZIP_SHA256",
    "SELECTED_INVENTORY_SHA256",
    "SELECTION_SALT",
    "build_instance_freshness_audit",
    "freshness_audit_sha256",
    "public_take_ids",
    "validate_instance_freshness_audit",
]
