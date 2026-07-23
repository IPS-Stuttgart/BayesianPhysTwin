"""Metadata-only two-commit lock for adaptive-covariance confirmation.

The implementation and every target-free runner are committed first as H1.
Only then may :func:`build_confirmation_cohort_lock` use H1 to determine the
cohort.  The resulting JSON is committed separately as H2.  Building and
validating the lock use only constants copied from the hash-bound name
taxonomy, integer episode IDs, and cryptographic identities; they never read
the dataset, network, predictions, measurements, targets, or metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ARTIFACT_KIND = "Deform360AdaptiveCovarianceConfirmationCohortLockV1"
PROTOCOL_ID = "deform360-adaptive-covariance-confirmation-v1"

DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
RAW_TREE_ID = "a4a36e0669bcc86ab79e7ffb35aada7f2334c570"
OBJECT_INVENTORY_COUNT = 190
OBJECT_INVENTORY_SHA256 = (
    "cf82fec6c715c3dafa2f3cadbeb6402f68443305fc40ff418075fe2cc22febb2"
)

TAXONOMY_SOURCE_COMMIT = "d72ca1dee49841b7d3b020da4380d0ef0a3f7d7c"
TAXONOMY_SOURCE_PATH = (
    "configs/sota/deform360_bias_aware_guarded_belief_prospective_v1.json"
)
TAXONOMY_SOURCE_SHA256 = (
    "9e933a93e28869cc67101300cd0990feb148841355f6a2416dc8cb92f595fa01"
)

EXCLUSION_UNION_SHA256 = (
    "1c937d23a37d9a330e157c0ad40a92131775c378a856364d5fa489a2b52c0bd1"
)
EXPECTED_STRATA = ("filament", "sheet", "volumetric")
OBJECT_QUOTAS: Mapping[str, int] = {
    "filament": 1,
    "sheet": 8,
    "volumetric": 8,
}
EPISODE_CANDIDATE_IDS = tuple(range(10))
EPISODES_PER_OBJECT = 2

# This is an exact copy of config.candidate_pools in the taxonomy artifact
# identified above.  It is intentionally a name-only manual taxonomy.  In
# particular, this protocol does not infer a new, broader material class from
# the remaining 62 public object names.
NAME_ONLY_TAXONOMY: Mapping[str, tuple[str, ...]] = {
    "filament": (
        "072-cotton-clohesline",
        "075-leather",
        "076-rubber-bands",
        "078-fishing-line",
        "080-wool",
        "088-snake",
        "123-pipe-cleaner",
        "143-silicone-wristband",
        "160-hose",
        "161-tube",
        "174-chain",
        "181-belt",
    ),
    "sheet": (
        "010-orange-cloth",
        "011-green-cloth",
        "012-hat-cloth",
        "013-glove-cloth",
        "014-glove-vinyl-cloth",
        "015-airbag-cloth",
        "017-chessboard-cloth",
        "018-trashbag-cloth",
        "019-trashbag-plastic-cloth",
        "020-cutting-mat-cloth",
        "023-cleaning-cloth",
        "024-glass-cleaner-cloth",
        "025-bag-small-cloth",
        "026-sock-cloth",
        "027-umbrella-bag-cloth",
        "028-ziplog-cloth",
        "029-foam-cloth",
        "030-bandage-cloth",
        "030-foam-flat-cloth",
        "031-cotton-cloth",
        "032-teabag-cloth",
        "034-plastic-bag-cloth",
        "036-napkin-cloth",
        "037-mop-cloth",
        "038-black-bag-cloth",
        "038-mat-cloth",
        "042-necktie-cloth",
        "055-lettuce-cloth",
        "060-bread-cloth",
        "065-pita-bread-cloth",
        "066-glove-half-black-cloth",
        "082-curtain-cloth",
        "084-apron-cloth",
        "087-plastic-bag-blue-cloth",
        "091-net-cloth",
        "098-beach-ball-cloth",
        "103-ice-pack-cloth",
        "105-clay-cloth",
        "107-mitt-cloth",
        "108-drying-mat-cloth",
        "109-pouch-cloth",
        "110-shower-cap-cloth",
        "111-headband-cloth",
        "112-wristband-cloth",
        "114-finger-wrap-cloth",
        "115-cotton-gauze-cloth",
        "116-hydrogel-patch-cloth",
        "118-envelope-cloth",
        "119-seal-cloth",
        "122-sheets-cloth",
        "124-tulle-fabric-cloth",
        "136-foam-letters-cloth",
        "137-kitchen-napkin-cloth",
        "140-rubber-glove-cloth",
        "142-shoe-sole-cloth",
        "144-jar-opener-cloth",
        "148-crepe-paper-cloth",
        "149-sticker-paper-cloth",
        "150-shredded-packing-paper-cloth",
        "151-parchment-paper-cloth",
        "156-mesh-produce-bag-cloth",
        "157-sack-cloth",
        "158-jewelry-pouch-cloth",
        "165-glove-yellow-cloth",
        "166-glove-green-cloth",
        "167-glove-gray-cloth",
        "169-pencilcase-cloth",
        "172-napkin-case-cloth",
        "173-poster-paper-cloth",
        "175-plastic-bag-cloth",
        "176-candy-packet-cloth",
        "178-bottle-accessory-cloth",
        "179-towel-black-cloth",
        "182-plastic-sheets-cloth",
        "183-shower-cap-transparent-cloth",
        "184-foam-roller-thick-cloth",
        "198-kneepad-cloth",
    ),
    "volumetric": (
        "045-cat",
        "047-rectangle-sponge",
        "048-butter-sponge",
        "050-boxing",
        "053-squeezer",
        "056-makeup-sponge",
        "057-kitchen-sponge",
        "058-roll-napkin",
        "062-banana",
        "063-flower",
        "093-squeezable-fruit",
        "095-watermelon",
        "097-pillow",
        "099-teeth",
        "100-puppet",
        "102-stress-ball",
        "120-bread-plush",
        "121-croissant-plush",
        "126-jellyfish",
        "138-sponge-stamps",
        "139-rubber-ball",
        "152-slime",
        "153-cake",
        "155-crystal-slime",
        "163-bear",
        "164-sheep",
        "168-cat-big",
        "185-cheese",
        "186-monster",
        "187-white-bear",
        "188-foam-roll-small",
        "189-bear-big",
        "190-monkey",
        "191-sloth-green",
        "192-fish",
        "193-frog",
        "194-fish-orange",
        "195-hello-kitty-brown",
        "196-hello-kitty-white",
    ),
}

# Every episode of these physical identities is excluded.  The list is the
# exact sorted union of open27, formal-held, and the two bias-prospective
# identity sets.  It is not a case-level exclusion.
EXCLUDED_OBJECT_IDS = (
    "002-rope-silk",
    "011-green-cloth",
    "015-airbag-cloth",
    "066-glove-half-black-cloth",
    "072-cotton-clohesline",
    "075-leather",
    "076-rubber-bands",
    "078-fishing-line",
    "080-wool",
    "081-stripe-rope",
    "083-blanket-cloth",
    "085-scarf-cloth",
    "088-snake",
    "091-net-cloth",
    "092-squirrel",
    "100-puppet",
    "112-wristband-cloth",
    "120-bread-plush",
    "121-croissant-plush",
    "123-pipe-cleaner",
    "139-rubber-ball",
    "143-silicone-wristband",
    "160-hose",
    "161-tube",
    "163-bear",
    "164-sheep",
    "165-glove-yellow-cloth",
    "168-cat-big",
    "170-spider",
    "174-chain",
    "175-plastic-bag-cloth",
)

_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")
_FULL_SHA256 = re.compile(r"[0-9a-f]{64}")
_OBJECT_RANK_DOMAIN = b"deform360-confirmation-object-rank-v1"
_EPISODE_RANK_DOMAIN = b"deform360-confirmation-episode-rank-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"cohort lock has duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_full_lower_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and _FULL_SHA1.fullmatch(value) is not None
        and value != "0" * 40
    )


def _is_full_lower_sha256(value: object) -> bool:
    return isinstance(value, str) and _FULL_SHA256.fullmatch(value) is not None


def _newline_inventory_sha256(values: Sequence[str]) -> str:
    payload = b"".join((value + "\n").encode("utf-8") for value in values)
    return hashlib.sha256(payload).hexdigest()


def selection_seed_sha256(implementation_commit_h1: str) -> str:
    """Return ``SHA256(protocol_id || NUL || H1 || NUL || dataset_revision)``."""

    _require(
        _is_full_lower_sha1(implementation_commit_h1),
        "H1 must be a full non-null lowercase 40-hex commit SHA",
    )
    payload = (
        PROTOCOL_ID.encode("ascii")
        + b"\0"
        + implementation_commit_h1.encode("ascii")
        + b"\0"
        + DATASET_REVISION.encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def framed_sha256(*frames: bytes) -> str:
    """Hash byte frames as repeated ``uint64_be(length) || payload``."""

    digest = hashlib.sha256()
    for frame in frames:
        _require(isinstance(frame, bytes), "hash frames must be bytes")
        digest.update(len(frame).to_bytes(8, byteorder="big", signed=False))
        digest.update(frame)
    return digest.hexdigest()


def object_rank_sha256(seed_sha256: str, stratum: str, object_id: str) -> str:
    """Rank a taxonomy identity without reading any object data."""

    _require(_is_full_lower_sha256(seed_sha256), "selection seed is not SHA-256")
    _require(stratum in NAME_ONLY_TAXONOMY, f"unknown stratum: {stratum}")
    _require(
        object_id in NAME_ONLY_TAXONOMY[stratum],
        f"object is outside the exact {stratum} name-only taxonomy",
    )
    return framed_sha256(
        _OBJECT_RANK_DOMAIN,
        bytes.fromhex(seed_sha256),
        stratum.encode("utf-8"),
        object_id.encode("utf-8"),
    )


def episode_rank_sha256(
    seed_sha256: str,
    stratum: str,
    object_id: str,
    episode_id: int,
) -> str:
    """Rank one integer episode ID without reading episode metadata or media."""

    _require(
        type(episode_id) is int and episode_id in EPISODE_CANDIDATE_IDS,
        "episode ID must be an integer in [0, 9]",
    )
    _require(stratum in NAME_ONLY_TAXONOMY, f"unknown stratum: {stratum}")
    _require(
        object_id in NAME_ONLY_TAXONOMY[stratum],
        f"object is outside the exact {stratum} name-only taxonomy",
    )
    _require(_is_full_lower_sha256(seed_sha256), "selection seed is not SHA-256")
    return framed_sha256(
        _EPISODE_RANK_DOMAIN,
        bytes.fromhex(seed_sha256),
        stratum.encode("utf-8"),
        object_id.encode("utf-8"),
        str(episode_id).encode("ascii"),
    )


def eligible_objects(stratum: str) -> tuple[str, ...]:
    """Return the exact taxonomy pool after identity-level exclusions."""

    _require(stratum in NAME_ONLY_TAXONOMY, f"unknown stratum: {stratum}")
    excluded = frozenset(EXCLUDED_OBJECT_IDS)
    return tuple(
        object_id
        for object_id in NAME_ONLY_TAXONOMY[stratum]
        if object_id not in excluded
    )


def _selected_object_records(
    seed_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for stratum in EXPECTED_STRATA:
        ranked_objects = sorted(
            eligible_objects(stratum),
            key=lambda object_id: (
                object_rank_sha256(seed_sha256, stratum, object_id),
                object_id,
            ),
        )
        selected = ranked_objects[: OBJECT_QUOTAS[stratum]]
        stratum_records = []
        for selection_rank, object_id in enumerate(selected):
            ranked_episodes = sorted(
                EPISODE_CANDIDATE_IDS,
                key=lambda episode_id: (
                    episode_rank_sha256(
                        seed_sha256,
                        stratum,
                        object_id,
                        episode_id,
                    ),
                    episode_id,
                ),
            )
            episode_records = [
                {
                    "episode_id": episode_id,
                    "episode_rank_sha256": episode_rank_sha256(
                        seed_sha256,
                        stratum,
                        object_id,
                        episode_id,
                    ),
                    "case_id": f"{object_id}-ep{episode_id:04d}",
                }
                for episode_id in ranked_episodes[:EPISODES_PER_OBJECT]
            ]
            stratum_records.append(
                {
                    "selection_rank": selection_rank,
                    "object_id": object_id,
                    "object_rank_sha256": object_rank_sha256(
                        seed_sha256,
                        stratum,
                        object_id,
                    ),
                    "episodes": episode_records,
                }
            )
        records[stratum] = stratum_records
    return records


def cohort_lock_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a lock while excluding its declared self-digest."""

    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _build_lock_without_digest(implementation_commit_h1: str) -> dict[str, Any]:
    seed_sha256 = selection_seed_sha256(implementation_commit_h1)
    selected = _selected_object_records(seed_sha256)
    selected_case_ids = [
        episode["case_id"]
        for stratum in EXPECTED_STRATA
        for record in selected[stratum]
        for episode in record["episodes"]
    ]
    taxonomy_copy = {
        stratum: list(NAME_ONLY_TAXONOMY[stratum]) for stratum in EXPECTED_STRATA
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "metadata-only-cohort-lock-for-h2",
        "claim_boundary": (
            "Prospective cross-object confirmation of the frozen adaptive-"
            "covariance routing policy against its declared causal camera-budget "
            "comparators. This is not official Deform360 open-loop parity and "
            "does not by itself permit a state-of-the-art claim."
        ),
        "two_commit_freeze": {
            "implementation_commit_h1": implementation_commit_h1,
            "h1_role": (
                "complete target-free implementation, tests, method parameters, "
                "runner, and validator; fixed before cohort identities are known"
            ),
            "h2_role": (
                "this deterministic metadata-only cohort lock and no method or "
                "selection-code change"
            ),
            "selection_seed_sha256": seed_sha256,
            "selection_seed_preimage": (
                "UTF8(protocol_id) || NUL || ASCII(H1) || NUL || "
                "ASCII(dataset_revision)"
            ),
        },
        "dataset_binding": {
            "repository": DATASET_REPOSITORY,
            "repo_type": "dataset",
            "revision": DATASET_REVISION,
            "raw_folder_tree_id": RAW_TREE_ID,
            "public_object_directory_count": OBJECT_INVENTORY_COUNT,
            "public_object_inventory_sha256": OBJECT_INVENTORY_SHA256,
            "public_object_inventory_algorithm": (
                "list nonrecursive raw/ RepoFolder names at the pinned revision; "
                "remove literal raw/ prefix; Python-sort ASCII object IDs; "
                "SHA256 UTF-8(object_id + LF) for every ID including final LF"
            ),
        },
        "name_only_taxonomy_binding": {
            "source_commit": TAXONOMY_SOURCE_COMMIT,
            "source_path": TAXONOMY_SOURCE_PATH,
            "source_file_sha256": TAXONOMY_SOURCE_SHA256,
            "copied_field": "config.candidate_pools",
            "candidate_pools": taxonomy_copy,
            "candidate_pool_counts": {
                stratum: len(NAME_ONLY_TAXONOMY[stratum]) for stratum in EXPECTED_STRATA
            },
            "semantics": (
                "Exact pre-existing manual name-only deformation strata; not "
                "labels inferred from media and not an exhaustive taxonomy of "
                "all public objects."
            ),
            "broader_class_or_reclassification_added": False,
        },
        "fresh_identity_exclusions": {
            "scope": (
                "physical object identity; every episode of every listed object "
                "is excluded"
            ),
            "object_ids": list(EXCLUDED_OBJECT_IDS),
            "object_count": len(EXCLUDED_OBJECT_IDS),
            "object_ids_sha256": EXCLUSION_UNION_SHA256,
            "digest_algorithm": (
                "Python-sort object IDs; SHA256 UTF-8(object_id + LF) for every "
                "ID including final LF"
            ),
            "freshness_boundary": (
                "Fresh means absent from this exact hash-bound union of open27, "
                "formal-held, and bias-prospective identities; it does not assert "
                "that no unlisted external process has ever accessed the object."
            ),
        },
        "selection": {
            "object_quotas": dict(OBJECT_QUOTAS),
            "episodes_per_object": EPISODES_PER_OBJECT,
            "episode_candidate_ids": list(EPISODE_CANDIDATE_IDS),
            "object_rank": (
                "SHA256 of uint64-big-endian-length-framed domain, raw 32-byte "
                "seed, UTF-8 stratum, and UTF-8 object_id; sort by digest then ID"
            ),
            "episode_rank": (
                "SHA256 of uint64-big-endian-length-framed domain, raw 32-byte "
                "seed, UTF-8 stratum, UTF-8 object_id, and ASCII episode_id; "
                "sort by digest then integer ID"
            ),
            "selection_inputs_only": [
                "protocol ID",
                "implementation commit H1",
                "dataset revision",
                "hash-bound name-only taxonomy object IDs",
                "hash-bound exclusion object IDs",
                "integer episode IDs 0 through 9",
            ],
            "network_or_dataset_read_by_lock_generator": False,
            "prediction_measurement_target_or_metric_read_by_lock_generator": False,
        },
        "taxonomy_imbalance_disclosure": {
            "eligible_object_counts": {
                stratum: len(eligible_objects(stratum)) for stratum in EXPECTED_STRATA
            },
            "selected_object_counts": dict(OBJECT_QUOTAS),
            "filament_statement": (
                "After exact identity exclusions, 181-belt is the sole eligible "
                "filament identity and is included deterministically; filament "
                "evidence is therefore one object with two nested episodes."
            ),
            "statistical_boundary": (
                "Episodes are not independent physical-object replicates. "
                "Filament cannot support a stratum-level population claim, and "
                "the overall cohort is intentionally unbalanced 1/8/8 rather "
                "than repaired by inventing or reclassifying objects."
            ),
        },
        "camera_budget_semantics": {
            "unit": (
                "selected RGB camera streams admitted to causal per-update "
                "tracking and triangulation"
            ),
            "nested_policy": (
                "A target-free all-calibrated-camera frame-zero planner fixes a "
                "four-camera subset nested inside an eight-camera subset; future "
                "object frames are used only from the currently attempted subset."
            ),
            "routes": [
                {
                    "route": "accept_four",
                    "dynamic_tracked_camera_streams_attempted": 4,
                },
                {
                    "route": "accept_eight",
                    "dynamic_tracked_camera_streams_attempted": 8,
                },
                {
                    "route": "physical_fallback",
                    "dynamic_tracked_camera_streams_attempted": 8,
                    "future_visual_update_applied": False,
                },
            ],
            "physical_fallback": (
                "A rejected eight-view observation returns the bit-exact physical "
                "prior and does not update either RBF state."
            ),
            "accounting_boundary": (
                "The budget is not the total number of calibrated scene cameras, "
                "not a hardware-acquisition count, and not an official Deform360 "
                "open-loop sensor budget. Any offline run that precomputes both "
                "budgets must report that fact and cannot claim realized compute "
                "or acquisition savings from routed-view accounting alone."
            ),
        },
        "cohort": selected,
        "case_count": len(selected_case_ids),
        "selected_case_ids": selected_case_ids,
        "information_boundary": {
            "scope": (
                "this deterministic lock-generator invocation; global freshness "
                "is limited to the separately declared exclusion-union boundary"
            ),
            "object_names_and_integer_episode_ids_only": True,
            "selected_object_media_read": False,
            "selected_object_action_metadata_read": False,
            "prediction_or_measurement_read": False,
            "target_array_or_future_geometry_read": False,
            "metric_or_outcome_read": False,
            "network_accessed": False,
        },
    }


def build_confirmation_cohort_lock(
    implementation_commit_h1: str,
) -> dict[str, Any]:
    """Build the deterministic cohort lock to be committed as H2."""

    payload = _build_lock_without_digest(implementation_commit_h1)
    payload["artifact_sha256"] = cohort_lock_sha256(payload)
    validate_confirmation_cohort_lock(
        payload,
        expected_implementation_commit_h1=implementation_commit_h1,
    )
    return payload


def validate_confirmation_cohort_lock(
    payload: Mapping[str, Any],
    *,
    expected_implementation_commit_h1: str | None = None,
) -> dict[str, Any]:
    """Strictly validate every binding, rank, episode, and disclosure."""

    _require(isinstance(payload, Mapping), "cohort lock must be an object")
    freeze = payload.get("two_commit_freeze")
    _require(
        isinstance(freeze, Mapping),
        "cohort lock two_commit_freeze must be an object",
    )
    h1 = freeze.get("implementation_commit_h1")
    _require(
        _is_full_lower_sha1(h1),
        "lock H1 must be a full non-null lowercase 40-hex commit SHA",
    )
    if expected_implementation_commit_h1 is not None:
        _require(
            _is_full_lower_sha1(expected_implementation_commit_h1),
            "expected H1 must be a full non-null lowercase 40-hex commit SHA",
        )
        _require(h1 == expected_implementation_commit_h1, "lock H1 changed")

    declared_digest = payload.get("artifact_sha256")
    _require(_is_full_lower_sha256(declared_digest), "lock digest is not SHA-256")
    observed_digest = cohort_lock_sha256(payload)
    _require(declared_digest == observed_digest, "cohort lock checksum mismatch")

    _require(
        _newline_inventory_sha256(EXCLUDED_OBJECT_IDS) == EXCLUSION_UNION_SHA256,
        "compiled exclusion union identity is invalid",
    )
    _require(
        tuple(sorted(EXCLUDED_OBJECT_IDS)) == EXCLUDED_OBJECT_IDS,
        "compiled exclusion union is not sorted",
    )
    all_taxonomy_objects = [
        object_id
        for stratum in EXPECTED_STRATA
        for object_id in NAME_ONLY_TAXONOMY[stratum]
    ]
    _require(
        len(all_taxonomy_objects) == len(set(all_taxonomy_objects)),
        "compiled name-only taxonomy contains duplicate identities",
    )
    _require(
        eligible_objects("filament") == ("181-belt",),
        "filament eligibility no longer has the audited sole identity",
    )
    for stratum in EXPECTED_STRATA:
        _require(
            len(eligible_objects(stratum)) >= OBJECT_QUOTAS[stratum],
            f"{stratum} quota exceeds eligible identity count",
        )

    expected = _build_lock_without_digest(h1)
    expected["artifact_sha256"] = cohort_lock_sha256(expected)
    _require(
        dict(payload) == expected,
        "cohort lock differs from deterministic H1-derived lock",
    )
    case_ids = payload["selected_case_ids"]
    _require(len(case_ids) == len(set(case_ids)) == 34, "cohort must have 34 cases")
    selected_objects = [
        record["object_id"]
        for stratum in EXPECTED_STRATA
        for record in payload["cohort"][stratum]
    ]
    _require(
        len(selected_objects) == len(set(selected_objects)) == 17,
        "cohort must have 17 distinct physical identities",
    )
    _require(
        not set(selected_objects).intersection(EXCLUDED_OBJECT_IDS),
        "cohort contains an excluded physical identity",
    )
    return {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "implementation_commit_h1": h1,
        "artifact_sha256": observed_digest,
        "object_count": len(selected_objects),
        "case_count": len(case_ids),
        "selected_objects": selected_objects,
        "selected_case_ids": list(case_ids),
    }


def write_confirmation_cohort_lock(
    path: str | Path,
    implementation_commit_h1: str,
) -> dict[str, Any]:
    """Atomically create an absent JSON lock without overwriting any path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not destination.exists() and not destination.is_symlink(),
        f"cohort lock output already exists: {destination}",
    )
    payload = build_confirmation_cohort_lock(implementation_commit_h1)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError(
                f"cohort lock output already exists: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_confirmation_cohort_lock(
    path: str | Path,
    *,
    expected_implementation_commit_h1: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate a previously generated lock."""

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cohort lock is invalid JSON") from error
    _require(isinstance(payload, dict), "cohort lock must be a JSON object")
    validate_confirmation_cohort_lock(
        payload,
        expected_implementation_commit_h1=expected_implementation_commit_h1,
    )
    return payload


__all__ = [
    "ARTIFACT_KIND",
    "DATASET_REVISION",
    "EXCLUDED_OBJECT_IDS",
    "EXCLUSION_UNION_SHA256",
    "NAME_ONLY_TAXONOMY",
    "OBJECT_INVENTORY_SHA256",
    "OBJECT_QUOTAS",
    "PROTOCOL_ID",
    "RAW_TREE_ID",
    "TAXONOMY_SOURCE_COMMIT",
    "TAXONOMY_SOURCE_PATH",
    "TAXONOMY_SOURCE_SHA256",
    "build_confirmation_cohort_lock",
    "cohort_lock_sha256",
    "eligible_objects",
    "episode_rank_sha256",
    "framed_sha256",
    "load_confirmation_cohort_lock",
    "object_rank_sha256",
    "selection_seed_sha256",
    "validate_confirmation_cohort_lock",
    "write_confirmation_cohort_lock",
]
