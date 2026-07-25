"""Prospective Deform360 protocol for the frozen bias-aware v4 candidate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PROTOCOL_ID = "deform360-bias-aware-guarded-belief-prospective-v1"
DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
SELECTION_SEED = PROTOCOL_ID
EXPECTED_STRATA = ("filament", "sheet", "volumetric")
EXPECTED_UPDATE_FRAMES = (19, 38, 57)
EXPECTED_FRAME_COUNT = 76
EXPECTED_CENTER_COUNT = 16
EXPECTED_CAMERA_COUNT = 8

SOURCE_COMMIT = "06f75a4406289384228a988df959e3c2af44510e"
SOURCE_SUMMARY_SHA256 = (
    "dbad5fd3b4d572d515d38b9bb31df84a2f036c223aaed3aa0810c25fbec3e015"
)
SOURCE_LOCK_SHA256 = (
    "5f5672d35aa41e276f1dd5ace54b6694b0139ff2a562e3c3a24558fa555c9dd6"
)
SOURCE_LOCK_FINITE_COVERAGE = 0.8
SOURCE_LOCK_GROUP_COUNT = 4
SOURCE_MINIMUM_IMPROVEMENT_M = 0.000005

OPEN_OR_RESERVED_OBJECTS = frozenset(
    {
        "001-rope",
        "002-rope-silk",
        "003-cable",
        "004-rubber-band",
        "005-thread",
        "008-pink-cloth",
        "009-yellow-cloth",
        "016-shirt-cloth",
        "021-bag-cloth",
        "022-handkerchief",
        "033-mask-cloth",
        "035-wipe-cloth",
        "040-paper-cloth",
        "041-wrap-paper-cloth",
        "043-dog",
        "044-doll",
        "046-sponge",
        "049-ball",
        "052-rubber-duck",
        "067-paracord",
        "068-nylon-rope",
        "069-jump-rope",
        "071-climbing-rope",
        "073-shoelace",
        "074-string",
        "077-hemp-rope",
        "079-chain-metal",
        "081-stripe-rope",
        "083-blanket-cloth",
        "085-scarf-cloth",
        "086-cotton-scarf-cloth",
        "090-sloth",
        "092-squirrel",
        "096-octopus",
        "117-bubble-wrap-cloth",
        "125-rabbit",
        "145-rubber-toy",
        "146-frog",
        "170-spider",
        "171-penguin",
    }
)

CANDIDATE_POOLS: Mapping[str, tuple[str, ...]] = {
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

EXPECTED_CALIBRATION_COHORT: Mapping[str, Mapping[str, tuple[int, ...]]] = {
    "filament": {
        "160-hose": (1,),
        "174-chain": (1,),
        "076-rubber-bands": (0,),
    },
    "sheet": {
        "175-plastic-bag-cloth": (3,),
        "011-green-cloth": (0,),
        "015-airbag-cloth": (6,),
    },
    "volumetric": {
        "163-bear": (1,),
        "100-puppet": (9,),
        "168-cat-big": (0,),
    },
}

EXPECTED_TARGET_COHORT: Mapping[str, Mapping[str, tuple[int, ...]]] = {
    "filament": {
        "075-leather": (3, 1),
        "123-pipe-cleaner": (4, 7),
        "080-wool": (4, 2),
        "143-silicone-wristband": (9, 7),
    },
    "sheet": {
        "165-glove-yellow-cloth": (1, 9),
        "066-glove-half-black-cloth": (0, 1),
        "112-wristband-cloth": (0, 6),
        "091-net-cloth": (8, 4),
    },
    "volumetric": {
        "139-rubber-ball": (3, 0),
        "121-croissant-plush": (7, 1),
        "120-bread-plush": (6, 8),
        "164-sheep": (5, 1),
    },
}

_CANONICAL_CONFIG_SHA256 = (
    "b6b19be5eaadf830a77f36cccddd38f5b7a35527ca21f7743d2ef147fceabbce"
)


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


def protocol_config_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the complete protocol while excluding its declared digest."""

    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def metadata_ranked_objects(stratum: str) -> tuple[str, ...]:
    """Rank one declared name-only pool using the fixed protocol seed."""

    _require(stratum in CANDIDATE_POOLS, f"unknown stratum: {stratum}")
    return tuple(
        sorted(
            CANDIDATE_POOLS[stratum],
            key=lambda object_id: hashlib.sha256(
                f"{SELECTION_SEED}:object:{stratum}:{object_id}".encode()
            ).hexdigest(),
        )
    )


def metadata_ranked_episode_ids(
    object_id: str, role: str, count: int
) -> tuple[int, ...]:
    """Choose episodes from IDs 0--9 without reading object media."""

    _require(role in {"calibration", "target"}, "unknown cohort role")
    _require(1 <= count <= 10, "episode count must lie in [1, 10]")
    ranked = sorted(
        range(10),
        key=lambda episode: hashlib.sha256(
            (
                f"{SELECTION_SEED}:episode:{role}:{object_id}:{episode}"
            ).encode()
        ).hexdigest(),
    )
    return tuple(ranked[:count])


def _cohort_records(
    cohort: Mapping[str, Mapping[str, tuple[int, ...]]],
) -> dict[str, list[dict[str, object]]]:
    return {
        stratum: [
            {"object_id": object_id, "episode_ids": list(episode_ids)}
            for object_id, episode_ids in cohort[stratum].items()
        ]
        for stratum in EXPECTED_STRATA
    }


def build_bias_aware_prospective_protocol() -> dict[str, Any]:
    """Build the one canonical protocol payload before declaring its digest."""

    config: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "locked-before-selected-object-download-or-media-access",
        "locked_at": "2026-07-20",
        "claim_boundary": (
            "Prospective cross-object accuracy and non-regression test of a "
            "source-frozen bias-aware online belief update. This is not official "
            "Deform360 Table-4 parity and does not permit a direct state-of-the-"
            "art claim."
        ),
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "public_object_directory_count_at_lock": 190,
            "selection_evidence": (
                "Pinned-revision top-level object directory names only; no "
                "selected-object image, video, geometry, tactile stream, action "
                "metadata, or metric was accessed."
            ),
            "prelock_server_path_audit": [
                {
                    "host": "gpuserver6000",
                    "roots": [
                        "/mnt/corsair/florianpfaff",
                        "/home/florianpfaff",
                    ],
                    "selected_object_matching_path_count": 0,
                    "audit_scope": "path names only",
                },
                {
                    "host": "gpuserver4090",
                    "roots": [
                        "/mnt/corsair/florianpfaff",
                        "/home/florianpfaff",
                    ],
                    "selected_object_matching_path_count": 0,
                    "audit_scope": "path names only",
                },
            ],
        },
        "open_or_reserved_objects": sorted(OPEN_OR_RESERVED_OBJECTS),
        "candidate_pool_definition": (
            "Manual name-only deformation strata frozen before media access; "
            "all object choices within each declared pool use SHA-256 rank."
        ),
        "candidate_pools": {
            stratum: list(CANDIDATE_POOLS[stratum]) for stratum in EXPECTED_STRATA
        },
        "selection": {
            "seed": SELECTION_SEED,
            "object_rank": "SHA256(seed:object:stratum:object_id)",
            "calibration_objects_per_stratum": 3,
            "target_objects_per_stratum": 4,
            "calibration_object_slice": "rank[0:3]",
            "target_object_slice": "rank[3:7]",
            "episode_rank": "SHA256(seed:episode:role:object_id:episode_id)",
            "calibration_episodes_per_object": 1,
            "target_episodes_per_object": 2,
            "episode_candidate_ids": list(range(10)),
        },
        "calibration_cohort": _cohort_records(EXPECTED_CALIBRATION_COHORT),
        "target_cohort": _cohort_records(EXPECTED_TARGET_COHORT),
        "observation": {
            "frame_count": EXPECTED_FRAME_COUNT,
            "update_frames": list(EXPECTED_UPDATE_FRAMES),
            "post_update_scored_frames": "20..37, 39..56, and 58..75",
            "center_count": EXPECTED_CENTER_COUNT,
            "camera_count": EXPECTED_CAMERA_COUNT,
            "causal_prefix": "Update u receives exactly RGB frames [0,u].",
            "target_future_rgb_allowed_before_prediction_seal": False,
            "target_future_geometry_allowed_before_prediction_seal": False,
            "target_metric_allowed_before_prediction_seal": False,
            "alltracker_and_triangulation_contract": (
                "reuse the hash-bound raw-camera source pipeline from the source "
                "v4 result without target-conditioned changes"
            ),
        },
        "method": {
            "source_commit": SOURCE_COMMIT,
            "source_summary_sha256": SOURCE_SUMMARY_SHA256,
            "source_lock_sha256": SOURCE_LOCK_SHA256,
            "candidate": (
                "bias-aware response-constrained persistent correction v4"
            ),
            "baseline": "selected raw backbone",
            "fallback": "bit-exact selected raw baseline",
            "minimum_physical_agreement_gain": 0.40,
            "minimum_improvement_m": SOURCE_MINIMUM_IMPROVEMENT_M,
            "target_outcome_input": False,
            "future_observation_input": False,
            "state_innovation_likelihood_count": 1,
            "prior_reliability_uses_state_innovation": False,
            "implementation_sha256": {
                "bias_aware_belief.py": (
                    "fc6fd6b9b8bb8fbb84b515532553f6de3a9d8f43e7e45f0d3b7c4b9627e981cd"
                ),
                "deform360_bias_aware_belief_development.py": (
                    "35acbca215e0d0e98189f1e2ca81dff573817242fec807a5b22485ab2ba0a7b3"
                ),
                "deform360_raw_pairwise_correspondence_diagnostic.py": (
                    "8a5df910d7354d418ae745925784d8f4a90c92ff8a6a38c32a5ebdb93718cc17"
                ),
                "deform360_raw_camera_observation.py": (
                    "2c24e587e9acd1dda589363240a81b268863fbe808ce05c58f8a5b70f12f76c3"
                ),
            },
        },
        "calibration_gate": {
            "role": (
                "prospective selector calibration only; no method-family, "
                "feature, threshold, rank, covariance, or observation change"
            ),
            "permitted_change": "refit only the direct source-group regret bound",
            "inherited_source_group_count": SOURCE_LOCK_GROUP_COUNT,
            "inherited_finite_sample_coverage": SOURCE_LOCK_FINITE_COVERAGE,
            "minimum_new_eligible_object_groups": 5,
            "minimum_combined_eligible_object_groups": 9,
            "required_finite_sample_coverage": 0.90,
            "required_upper_regret_m": -SOURCE_MINIMUM_IMPROVEMENT_M,
            "within_object_score": (
                "maximum worst-primary interval regret over every eligible "
                "episode/update for that object"
            ),
            "co_primary_object_balanced_mean_regret_must_be_negative": True,
            "accepted_harmful_object_count_allowed": 0,
            "minimum_evaluable_objects": 7,
            "minimum_evaluable_objects_per_stratum": 2,
            "quality_failure_replacement_allowed": False,
            "target_access_if_gate_fails": "forbidden",
            "failed_gate_action": (
                "publish calibration failure and keep every target future sealed"
            ),
        },
        "target_evaluation": {
            "unit_of_replication": "physical object",
            "episodes_nested_within_object": 2,
            "co_primary_metrics": [
                "post_update_hidden_identity_rmse_m",
                "post_update_hidden_symmetric_chamfer_m",
            ],
            "comparison": "frozen v4 selector minus exact selected raw baseline",
            "minimum_evaluable_objects": 9,
            "minimum_evaluable_objects_per_stratum": 3,
            "replacement_allowed": False,
            "success_gates": {
                "both_object_balanced_mean_differences_negative": True,
                "both_object_cluster_upper_95_bounds_negative": True,
                "no_stratum_mean_regression": True,
                "accepted_harmful_object_rate_at_most": 0.10,
                "every_rejection_bit_exact_fallback": True,
                "all_quality_failures_reported": True,
            },
            "direct_official_sota_claim_allowed": False,
            "permitted_positive_claim": (
                "The source-frozen guarded online update prospectively improves "
                "its strong selected raw/physical backbone across fresh public "
                "deformable objects under the declared causal-prefix protocol."
            ),
        },
        "information_order": [
            "commit method, source evidence, prospective protocol, and runner",
            "download only the 21 locked objects at the pinned revision",
            "stage calibration prefixes without exposing their futures",
            "seal every calibration baseline and v4 prediction or quality failure",
            "open and score calibration futures",
            "freeze or reject the combined direct group-regret bound",
            "if and only if calibration passes, stage and seal all target predictions",
            "verify the complete target prediction cohort seal",
            "open target futures and score every eligible object without replacement",
        ],
    }
    payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "config": config}
    payload["config_sha256"] = protocol_config_sha256(payload)
    return payload


def _normalized_expected_cohort(
    role: str,
) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
    return (
        EXPECTED_CALIBRATION_COHORT
        if role == "calibration"
        else EXPECTED_TARGET_COHORT
    )


def _validate_cohort(
    raw: object, role: str
) -> dict[str, dict[str, tuple[int, ...]]]:
    _require(isinstance(raw, Mapping), f"{role} cohort must be an object")
    _require(tuple(raw) == EXPECTED_STRATA, f"{role} strata changed")
    per_stratum = 3 if role == "calibration" else 4
    episode_count = 1 if role == "calibration" else 2
    expected = _normalized_expected_cohort(role)
    normalized: dict[str, dict[str, tuple[int, ...]]] = {}
    selected: list[str] = []
    for stratum in EXPECTED_STRATA:
        records = raw[stratum]
        _require(isinstance(records, Sequence), f"{role} {stratum} is not a list")
        _require(len(records) == per_stratum, f"{role} {stratum} count changed")
        object_map: dict[str, tuple[int, ...]] = {}
        ranked = metadata_ranked_objects(stratum)
        expected_objects = (
            ranked[:3] if role == "calibration" else ranked[3:7]
        )
        for record in records:
            _require(isinstance(record, Mapping), "cohort record is not an object")
            object_id = str(record.get("object_id", ""))
            episodes = tuple(int(value) for value in record.get("episode_ids", ()))
            _require(object_id not in OPEN_OR_RESERVED_OBJECTS, "cohort reuses data")
            _require(object_id in CANDIDATE_POOLS[stratum], "object left its pool")
            _require(object_id not in object_map, "object repeated within stratum")
            _require(
                episodes
                == metadata_ranked_episode_ids(object_id, role, episode_count),
                f"{role} episode rank changed: {object_id}",
            )
            object_map[object_id] = episodes
            selected.append(object_id)
        _require(tuple(object_map) == expected_objects, f"{role} object rank changed")
        normalized[stratum] = object_map
    _require(len(selected) == len(set(selected)), f"{role} object repeated")
    _require(normalized == expected, f"{role} cohort changed")
    return normalized


def _validate_method(config: Mapping[str, Any]) -> None:
    method = config.get("method")
    _require(isinstance(method, Mapping), "method binding is missing")
    _require(method.get("source_commit") == SOURCE_COMMIT, "source commit changed")
    _require(
        method.get("source_summary_sha256") == SOURCE_SUMMARY_SHA256,
        "source summary changed",
    )
    _require(
        method.get("source_lock_sha256") == SOURCE_LOCK_SHA256,
        "source lock changed",
    )
    _require(
        method.get("candidate")
        == "bias-aware response-constrained persistent correction v4",
        "candidate changed",
    )
    _require(
        method.get("baseline") == "selected raw backbone",
        "baseline changed",
    )
    _require(
        method.get("fallback") == "bit-exact selected raw baseline",
        "fallback changed",
    )
    _require(
        float(method.get("minimum_physical_agreement_gain", -1.0)) == 0.40,
        "physical agreement threshold changed",
    )
    _require(
        float(method.get("minimum_improvement_m", -1.0))
        == SOURCE_MINIMUM_IMPROVEMENT_M,
        "minimum improvement changed",
    )
    _require(method.get("target_outcome_input") is False, "target entered method")
    _require(
        method.get("future_observation_input") is False,
        "future observation entered method",
    )


def _validate_calibration_and_target(config: Mapping[str, Any]) -> None:
    calibration = config.get("calibration_gate")
    _require(isinstance(calibration, Mapping), "calibration gate is missing")
    _require(
        calibration.get("permitted_change")
        == "refit only the direct source-group regret bound",
        "calibration may alter the method",
    )
    _require(
        int(calibration.get("inherited_source_group_count", -1))
        == SOURCE_LOCK_GROUP_COUNT,
        "source group count changed",
    )
    _require(
        float(calibration.get("inherited_finite_sample_coverage", -1.0))
        == SOURCE_LOCK_FINITE_COVERAGE,
        "source coverage changed",
    )
    _require(
        int(calibration.get("minimum_new_eligible_object_groups", -1)) == 5,
        "minimum calibration support changed",
    )
    _require(
        float(calibration.get("required_finite_sample_coverage", -1.0)) == 0.90,
        "calibration coverage changed",
    )
    _require(
        calibration.get("target_access_if_gate_fails") == "forbidden",
        "failed calibration no longer blocks target access",
    )

    target = config.get("target_evaluation")
    _require(isinstance(target, Mapping), "target evaluation is missing")
    _require(
        target.get("co_primary_metrics")
        == [
            "post_update_hidden_identity_rmse_m",
            "post_update_hidden_symmetric_chamfer_m",
        ],
        "target metrics changed",
    )
    _require(
        target.get("unit_of_replication") == "physical object",
        "replication unit changed",
    )
    _require(
        int(target.get("minimum_evaluable_objects", -1)) == 9,
        "minimum evaluable object count changed",
    )
    _require(
        int(target.get("minimum_evaluable_objects_per_stratum", -1)) == 3,
        "minimum stratum count changed",
    )
    _require(target.get("replacement_allowed") is False, "replacement was enabled")
    _require(
        target.get("direct_official_sota_claim_allowed") is False,
        "protocol now permits an invalid direct SOTA claim",
    )


def load_bias_aware_prospective_protocol(path: str | Path) -> dict[str, Any]:
    """Load and fully validate the immutable prospective protocol."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "protocol must contain an object")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema changed")
    observed_hash = protocol_config_sha256(payload)
    _require(payload.get("config_sha256") == observed_hash, "config hash changed")
    _require(observed_hash == _CANONICAL_CONFIG_SHA256, "noncanonical config")
    config = payload.get("config")
    _require(isinstance(config, Mapping), "protocol config is missing")
    _require(config.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    dataset = config.get("dataset")
    _require(isinstance(dataset, Mapping), "dataset binding is missing")
    _require(dataset.get("repository") == DATASET_REPOSITORY, "dataset changed")
    _require(dataset.get("revision") == DATASET_REVISION, "revision changed")
    pools = config.get("candidate_pools")
    _require(isinstance(pools, Mapping), "candidate pools are missing")
    _require(
        {key: tuple(value) for key, value in pools.items()} == CANDIDATE_POOLS,
        "candidate pools changed",
    )
    calibration = _validate_cohort(config.get("calibration_cohort"), "calibration")
    target = _validate_cohort(config.get("target_cohort"), "target")
    calibration_objects = {
        object_id for records in calibration.values() for object_id in records
    }
    target_objects = {
        object_id for records in target.values() for object_id in records
    }
    _require(not calibration_objects & target_objects, "cohort roles overlap")
    _validate_method(config)
    _validate_calibration_and_target(config)
    return {
        "payload": dict(payload),
        "config": dict(config),
        "config_sha256": observed_hash,
        "calibration_cohort": calibration,
        "target_cohort": target,
    }


__all__ = [
    "CANDIDATE_POOLS",
    "DATASET_REPOSITORY",
    "DATASET_REVISION",
    "EXPECTED_CALIBRATION_COHORT",
    "EXPECTED_CAMERA_COUNT",
    "EXPECTED_CENTER_COUNT",
    "EXPECTED_FRAME_COUNT",
    "EXPECTED_STRATA",
    "EXPECTED_TARGET_COHORT",
    "EXPECTED_UPDATE_FRAMES",
    "OPEN_OR_RESERVED_OBJECTS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SELECTION_SEED",
    "build_bias_aware_prospective_protocol",
    "load_bias_aware_prospective_protocol",
    "metadata_ranked_episode_ids",
    "metadata_ranked_objects",
    "protocol_config_sha256",
]
