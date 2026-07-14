"""Outcome-free lock for the public Deform360 replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPLICATION_SCHEMA_VERSION = 1
REPLICATION_PROTOCOL_ID = "causal4d-deform360-shared-physics-replication-v1"
PINNED_PILOT_TAG = "deform360-001-rope-public-v1"
PINNED_DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
PINNED_OFFICIAL_PHYSTWIN_COMMIT = "2b6630528141b9cba5a7677c8b88b2129b4a8390"
CANONICAL_REPLICATION_CONFIG_SHA256 = (
    "f0aab308345807b2183f653306a062d4ad0295584b6b283deb99d29b3c247934"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def replication_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _selected_objects(config: Mapping[str, Any]) -> dict[str, list[str]]:
    seed = str(config["selection_seed"])
    selected: dict[str, list[str]] = {}
    for stratum, specification in config["strata"].items():
        ranked = sorted(
            specification["candidate_pool"],
            key=lambda object_id: _sha256_text(f"{seed}:{stratum}:{object_id}"),
        )
        selected[stratum] = ranked[: int(specification["selected_count"])]
    return selected


def _split_for_object(
    config: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    seed = str(config["selection_seed"])
    object_id = str(record["object_id"])
    episodes = {int(key): value for key, value in record["episodes"].items()}
    required_parity = "no" if int(record["selection_rank"]) == 0 else "yes"
    eligible = [
        index
        for index, metadata in episodes.items()
        if metadata["bimanual"] == required_parity
        and metadata.get("nonprehensile", "no") == "no"
    ]
    target = min(
        eligible,
        key=lambda index: _sha256_text(f"{seed}:target:{object_id}:{index}"),
    )
    remaining = [index for index in episodes if index != target]
    calibration: list[int] = []
    for parity in ("no", "yes"):
        pool = [
            index
            for index in remaining
            if episodes[index]["bimanual"] == parity
            and episodes[index].get("nonprehensile", "no") == "no"
        ]
        selected = min(
            pool,
            key=lambda index: _sha256_text(
                f"{seed}:calibration:{object_id}:{parity}:{index}"
            ),
        )
        calibration.append(selected)
        remaining.remove(selected)
    selected = min(
        remaining,
        key=lambda index: _sha256_text(f"{seed}:calibration:{object_id}:any:{index}"),
    )
    calibration.append(selected)
    remaining.remove(selected)
    return {
        "source_episode_ids": sorted(remaining),
        "calibration_episode_ids": sorted(calibration),
        "target_episode_id": target,
        "target_action": episodes[target]["action"],
        "target_bimanual": required_parity == "yes",
    }


def validate_deform360_replication_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every metadata-only cohort and split decision."""

    _require(
        payload.get("schema_version") == REPLICATION_SCHEMA_VERSION,
        "unsupported replication schema",
    )
    observed_sha = replication_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed_sha,
        "replication config checksum mismatch",
    )
    _require(
        observed_sha == CANONICAL_REPLICATION_CONFIG_SHA256,
        "replication config differs from canonical lock",
    )
    config = payload["config"]
    _require(config["protocol_id"] == REPLICATION_PROTOCOL_ID, "protocol id changed")
    _require(config["pilot_release_tag"] == PINNED_PILOT_TAG, "pilot tag changed")
    _require(
        config["dataset_revision"] == PINNED_DATASET_REVISION,
        "dataset revision changed",
    )
    _require(
        config["official_phystwin_commit"] == PINNED_OFFICIAL_PHYSTWIN_COMMIT,
        "official PhysTwin revision changed",
    )

    expected_selected = _selected_objects(config)
    actual_selected: dict[str, list[str]] = {}
    target_parities = []
    for record in config["cohort"]:
        stratum = record["stratum"]
        actual_selected.setdefault(stratum, []).append(record["object_id"])
        expected_hash = _sha256_text(
            f"{config['selection_seed']}:{stratum}:{record['object_id']}"
        )
        _require(record["selection_hash"] == expected_hash, "object hash changed")
        expected_split = _split_for_object(config, record)
        for key, expected in expected_split.items():
            _require(record[key] == expected, f"{record['object_id']} {key} changed")
        groups = (
            record["source_episode_ids"],
            record["calibration_episode_ids"],
            [record["target_episode_id"]],
        )
        flattened = [index for group in groups for index in group]
        _require(
            sorted(flattened) == list(range(10)) and len(set(flattened)) == 10,
            f"{record['object_id']} split does not partition ten episodes",
        )
        target_parities.append(bool(record["target_bimanual"]))

    for stratum, expected in expected_selected.items():
        _require(
            actual_selected.get(stratum, []) == expected,
            f"{stratum} cohort differs from hash ranking",
        )
    _require(len(config["cohort"]) == 6, "replication must contain six objects")
    _require(
        target_parities.count(False) == target_parities.count(True) == 3,
        "target actions must balance unimanual and bimanual cases",
    )
    methods = config["method_arms"]
    _require(len(methods) == len(set(methods)) == 7, "method arms changed")
    _require(
        methods[0] == "constant_persistence" and methods[-1] == "full_tactile_oracle",
        "baseline or oracle ordering changed",
    )
    warp_gate = config["gates"]["official_warp_feasibility"]
    _require(
        warp_gate["allowed_source_episode_ids"] == [0, 3, 4, 5, 8],
        "Warp gate source set changed",
    )
    _require(
        6 in warp_gate["forbidden_episode_ids"],
        "exhausted 001-rope target is not forbidden",
    )
    boundary = config["information_boundary"]
    _require(
        boundary["selected_object_media_accessed_before_lock"] is False,
        "preregistration claims selected media were already accessed",
    )
    _require(
        boundary["target_future_geometry_allowed_before_prediction_seal"] is False
        and boundary["target_future_tactile_allowed_before_prediction_seal"] is False
        and boundary["full_tactile_oracle_post_seal_only"] is True,
        "target information boundary changed",
    )
    return {
        "passed": True,
        "protocol_id": config["protocol_id"],
        "config_sha256": observed_sha,
        "object_count": len(config["cohort"]),
        "target_unimanual_count": target_parities.count(False),
        "target_bimanual_count": target_parities.count(True),
        "selected_objects": [record["object_id"] for record in config["cohort"]],
    }


def load_deform360_replication_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_deform360_replication_protocol(payload)
    return payload
