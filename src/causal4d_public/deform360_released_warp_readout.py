"""Locked source protocol for dense readout of official-Warp rope forecasts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

RELEASED_WARP_READOUT_SCHEMA_VERSION = 1
RELEASED_WARP_READOUT_PROTOCOL_ID = "deform360-released-warp-readout-source-v1"
PINNED_AUTHOR_RELEASE_REVISION = "93280cbb466de6b9e59927c58a99fd3b9e91900e"
PINNED_OFFICIAL_PHYSTWIN_COMMIT = "2b6630528141b9cba5a7677c8b88b2129b4a8390"
CANONICAL_RELEASED_WARP_READOUT_CONFIG_SHA256 = (
    "b7d06929079fd97ad8e2b6dbe149946851321bb6dc23a92c44cc9b2a0409e0c0"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def released_warp_readout_config_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical checksum of a protocol payload."""

    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _episode_records(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = config.get("episodes")
    _require(isinstance(records, list), "episode records are missing")
    _require(
        all(isinstance(record, Mapping) for record in records),
        "episode records must contain objects",
    )
    return records


def validate_released_warp_readout_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the outcome-free source lock and its information boundary."""

    _require(
        payload.get("schema_version") == RELEASED_WARP_READOUT_SCHEMA_VERSION,
        "unsupported released-Warp readout schema",
    )
    observed_sha = released_warp_readout_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed_sha,
        "released-Warp readout checksum mismatch",
    )
    _require(
        observed_sha == CANONICAL_RELEASED_WARP_READOUT_CONFIG_SHA256,
        "released-Warp readout config differs from the canonical lock",
    )
    config = payload.get("config")
    _require(isinstance(config, Mapping), "released-Warp readout config is missing")
    _require(
        config.get("protocol_id") == RELEASED_WARP_READOUT_PROTOCOL_ID,
        "released-Warp readout protocol id changed",
    )
    _require(config.get("object_id") == "001-rope", "development object changed")
    _require(
        config.get("development_source_episode_ids") == [0, 3, 4, 5, 8],
        "development source episodes changed",
    )
    _require(
        config.get("forbidden_episode_ids") == [1, 2, 6, 7, 9],
        "forbidden episodes changed",
    )

    release = config.get("author_release")
    _require(isinstance(release, Mapping), "author release is missing")
    _require(
        release.get("revision") == PINNED_AUTHOR_RELEASE_REVISION,
        "author release revision changed",
    )
    _require(
        release.get("fps") == 30 and release.get("coordinate_unit") == "m",
        "author release units or rate changed",
    )
    manifest = release.get("pcd_frame_manifest")
    _require(isinstance(manifest, Mapping), "released pcd manifest is missing")
    _require(
        manifest.get("file_count") == 75
        and _valid_sha256(manifest.get("sha256")),
        "released pcd manifest changed",
    )

    source_gate = config.get("prior_source_gate")
    _require(isinstance(source_gate, Mapping), "prior source gate is missing")
    _require(
        source_gate.get("official_phystwin_commit")
        == PINNED_OFFICIAL_PHYSTWIN_COMMIT,
        "official PhysTwin revision changed",
    )
    _require(
        source_gate.get("pooled_candidate_index") == 115,
        "pooled candidate changed",
    )
    expected_candidates = {"0": 115, "3": 119, "4": 115, "5": 195, "8": 197}
    _require(
        source_gate.get("leave_one_source_out_candidate_by_episode")
        == expected_candidates,
        "leave-one-source candidates changed",
    )
    for field in (
        "feasibility_json_sha256",
        "feasibility_result_sha256",
        "feasibility_npz_sha256",
        "pooling_control_json_sha256",
        "pooling_control_result_sha256",
    ):
        _require(_valid_sha256(source_gate.get(field)), f"{field} is invalid")

    records = _episode_records(config)
    episode_ids = [int(record["episode_id"]) for record in records]
    _require(
        episode_ids == config["development_source_episode_ids"],
        "episode record order changed",
    )
    pcd_file_count = 0
    for record in records:
        episode_id = int(record["episode_id"])
        _require(
            int(record["loo_candidate_index"])
            == expected_candidates[str(episode_id)],
            f"episode {episode_id} candidate changed",
        )
        for field in (
            "metadata_sha256",
            "split_sha256",
            "robot_sha256",
            "source_json_sha256",
            "source_npz_sha256",
            "source_result_sha256",
            "pcd_files_sha256",
        ):
            _require(
                _valid_sha256(record.get(field)),
                f"episode {episode_id} {field} is invalid",
            )
        train = list(map(int, record["author_train_window"]))
        test = list(map(int, record["author_test_window"]))
        origin = int(record["matched_origin_frame"])
        previous = int(record["previous_state_frame"])
        evaluation = list(map(int, record["evaluation_frames"]))
        _require(
            len(train) == len(test) == 2
            and train[1] == test[0]
            and train[0] < train[1] < test[1],
            f"episode {episode_id} split is invalid",
        )
        _require(
            previous == origin - 2 and origin < evaluation[0],
            f"episode {episode_id} matched origin is invalid",
        )
        _require(
            all(test[0] <= frame < test[1] for frame in evaluation)
            and all(
                later - earlier == 2
                for earlier, later in pairwise(evaluation)
            ),
            f"episode {episode_id} evaluation frames are invalid",
        )
        pcd_file_count += 1 + len(evaluation)
    _require(
        pcd_file_count == manifest["file_count"],
        "episode pcd count differs from the locked manifest",
    )

    readout = config.get("dense_readout")
    _require(isinstance(readout, Mapping), "dense readout is missing")
    _require(
        readout.get("future_particle_use_for_association") is False
        and readout.get("expected_direct_particle_support_fraction") == 1.0,
        "dense readout boundary changed",
    )
    boundary = config.get("information_boundary")
    _require(isinstance(boundary, Mapping), "information boundary is missing")
    _require(
        boundary.get("source_only") is True
        and boundary.get("future_contact_active_used") is False
        and boundary.get("future_object_particles_used_by_simulator") is False
        and boundary.get("future_object_particles_used_by_readout_association")
        is False
        and boundary.get(
            "future_object_particles_used_only_after_prediction_seal_for_scoring"
        )
        is True
        and boundary.get("candidate_reselection_from_dense_scores") is False
        and boundary.get("forbidden_episode_access") is False
        and boundary.get("held_v8_access") is False,
        "information boundary changed",
    )
    gate = config.get("transfer_gate")
    _require(isinstance(gate, Mapping), "transfer gate is missing")
    _require(
        gate.get("evaluated_arm") == "loo_finite_velocity_rotated_offset"
        and gate.get("minimum_mean_chamfer_improvement_vs_matched_persistence")
        == 0.05
        and gate.get("minimum_episode_chamfer_wins") == 3
        and gate.get("maximum_mean_identity_error_ratio_vs_matched_persistence")
        == 1.0,
        "transfer gate changed",
    )
    return {
        "passed": True,
        "protocol_id": RELEASED_WARP_READOUT_PROTOCOL_ID,
        "config_sha256": observed_sha,
        "episode_ids": episode_ids,
        "pcd_file_count": pcd_file_count,
    }


def load_released_warp_readout_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate the canonical source protocol."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "protocol must contain an object")
    validate_released_warp_readout_protocol(payload)
    return payload


def released_pcd_manifest(
    released_object_root: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every released particle frame authorized by the protocol."""

    root = Path(released_object_root)
    records: list[dict[str, str]] = []
    for episode in _episode_records(payload["config"]):
        episode_id = int(episode["episode_id"])
        frames = [
            int(episode["matched_origin_frame"]),
            *map(int, episode["evaluation_frames"]),
        ]
        for frame in frames:
            relative_path = (
                Path(f"episode_{episode_id}")
                / "pcd_clean"
                / f"{frame:06d}.npz"
            )
            path = root / relative_path
            _require(path.is_file(), f"released pcd frame is missing: {path}")
            records.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "sha256": _sha256_file(path),
                }
            )
    records.sort(key=lambda record: record["relative_path"])
    return {
        "file_count": len(records),
        "sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
        "records": records,
    }


def validate_released_warp_readout_inputs(
    payload: Mapping[str, Any],
    *,
    released_object_root: str | Path,
    source_observation_root: str | Path,
    prior_gate_artifact_root: str | Path,
) -> dict[str, Any]:
    """Validate all source inputs before a prediction process starts."""

    static = validate_released_warp_readout_protocol(payload)
    config = payload["config"]
    released_root = Path(released_object_root)
    source_root = Path(source_observation_root)
    gate_root = Path(prior_gate_artifact_root)
    source_files = 0
    for record in _episode_records(config):
        episode_id = int(record["episode_id"])
        episode_root = released_root / f"episode_{episode_id}"
        expected_files = {
            episode_root / "metadata.json": record["metadata_sha256"],
            episode_root / "split.json": record["split_sha256"],
            episode_root / "robot" / "robot.npy": record["robot_sha256"],
            source_root
            / f"deform360_001_rope_source{episode_id}_observation_v5.json": record[
                "source_json_sha256"
            ],
            source_root
            / f"deform360_001_rope_source{episode_id}_observation_v5.npz": record[
                "source_npz_sha256"
            ],
        }
        for path, expected_sha in expected_files.items():
            _require(path.is_file(), f"locked source input is missing: {path}")
            _require(
                _sha256_file(path) == expected_sha,
                f"locked source input checksum differs: {path}",
            )
            source_files += 1
        split = json.loads((episode_root / "split.json").read_text(encoding="utf-8"))
        _require(
            split.get("train") == record["author_train_window"]
            and split.get("test") == record["author_test_window"],
            f"episode {episode_id} released split changed",
        )
        source_json = json.loads(
            (
                source_root
                / f"deform360_001_rope_source{episode_id}_observation_v5.json"
            ).read_text(encoding="utf-8")
        )
        _require(
            source_json.get("result_sha256") == record["source_result_sha256"],
            f"episode {episode_id} source result checksum changed",
        )
        with np.load(
            source_root
            / f"deform360_001_rope_source{episode_id}_observation_v5.npz",
            allow_pickle=False,
        ) as archive:
            frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        required_frames = {
            int(record["previous_state_frame"]),
            int(record["matched_origin_frame"]),
            *map(int, record["evaluation_frames"]),
        }
        _require(
            required_frames.issubset(set(map(int, frames))),
            f"episode {episode_id} source observation misses required frames",
        )

    prior = config["prior_source_gate"]
    gate_files = {
        gate_root / "deform360_001_rope_official_warp_feasibility_v1.json": prior[
            "feasibility_json_sha256"
        ],
        gate_root / "deform360_001_rope_official_warp_feasibility_v1.npz": prior[
            "feasibility_npz_sha256"
        ],
        gate_root / "deform360_001_rope_official_warp_pooling_control_v1.json": prior[
            "pooling_control_json_sha256"
        ],
    }
    for path, expected_sha in gate_files.items():
        _require(path.is_file(), f"prior gate artifact is missing: {path}")
        _require(
            _sha256_file(path) == expected_sha,
            f"prior gate artifact checksum differs: {path}",
        )

    pcd = released_pcd_manifest(released_root, payload)
    expected_pcd = config["author_release"]["pcd_frame_manifest"]
    _require(
        pcd["file_count"] == expected_pcd["file_count"]
        and pcd["sha256"] == expected_pcd["sha256"],
        "released pcd manifest differs from the lock",
    )
    return {
        **static,
        "source_file_count": source_files,
        "prior_gate_file_count": len(gate_files),
        "pcd_manifest_sha256": pcd["sha256"],
    }


__all__ = [
    "CANONICAL_RELEASED_WARP_READOUT_CONFIG_SHA256",
    "load_released_warp_readout_protocol",
    "released_pcd_manifest",
    "released_warp_readout_config_sha256",
    "validate_released_warp_readout_inputs",
    "validate_released_warp_readout_protocol",
]
