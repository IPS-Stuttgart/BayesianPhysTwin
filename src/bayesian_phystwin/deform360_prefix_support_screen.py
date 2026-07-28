"""Causal endpoint association-support screening for sentinel queries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform360_dynamic_query import projection_matrices
from .observation_belief import array_sha256
from .tapnextpp_birth_association import (
    SET_VALUED_COVARIANCE_ASSOCIATION,
    BirthAssociationConfig,
    propose_birth_query_pixels,
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PrefixAssociationSupportConfig:
    """Frozen causal support choices for the V7 source arm."""

    birth_frame: int = 51
    update_frame: int = 57
    minimum_camera_support: int = 3
    search_radius_px: int = 12
    depth_scale_m: float = 0.03
    minimum_candidate_count: int = 1

    def __post_init__(self) -> None:
        _require(
            0 <= self.birth_frame < self.update_frame,
            "support-screen birth must precede update",
        )
        _require(
            self.minimum_camera_support >= 3,
            "support-screen evidence requires at least three cameras",
        )
        _require(self.search_radius_px >= 1, "search radius must be positive")
        _require(
            np.isfinite(self.depth_scale_m) and self.depth_scale_m > 0.0,
            "depth scale must be positive",
        )
        _require(
            self.minimum_candidate_count >= 1,
            "minimum candidate count must be positive",
        )

    def association_config(self) -> BirthAssociationConfig:
        return BirthAssociationConfig(
            search_radius_px=self.search_radius_px,
            depth_scale_m=self.depth_scale_m,
            minimum_candidate_count=self.minimum_candidate_count,
            association_mode=SET_VALUED_COVARIANCE_ASSOCIATION,
        )


@dataclass(frozen=True)
class PrefixAssociationSupportScreen:
    """Immutable eligible-node set from two permitted prefix endpoints."""

    entity_ids: np.ndarray
    birth_support_count: np.ndarray
    update_support_count: np.ndarray
    eligible: np.ndarray
    config: PrefixAssociationSupportConfig
    birth_valid_sha256: str
    update_valid_sha256: str
    physical_endpoint_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        entities = _readonly(self.entity_ids, dtype=np.int64)
        birth = _readonly(self.birth_support_count, dtype=np.int64)
        update = _readonly(self.update_support_count, dtype=np.int64)
        eligible = _readonly(self.eligible, dtype=bool)
        count = len(entities)
        _require(
            entities.shape == birth.shape == update.shape == eligible.shape
            == (count,),
            "prefix-support arrays changed shape",
        )
        _require(
            count > 0
            and len(np.unique(entities)) == count
            and np.all(entities >= 0),
            "prefix-support entity IDs are invalid",
        )
        _require(
            np.all(birth >= 0) and np.all(update >= 0),
            "prefix-support counts are negative",
        )
        expected = (
            birth >= self.config.minimum_camera_support
        ) & (update >= self.config.minimum_camera_support)
        _require(
            np.array_equal(eligible, expected),
            "prefix-support eligibility differs from the frozen rule",
        )
        for name, digest in (
            ("birth_valid_sha256", self.birth_valid_sha256),
            ("update_valid_sha256", self.update_valid_sha256),
            ("physical_endpoint_sha256", self.physical_endpoint_sha256),
            ("artifact_sha256", self.artifact_sha256),
        ):
            _require(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"{name} is not a SHA-256 digest",
            )
        object.__setattr__(self, "entity_ids", entities)
        object.__setattr__(self, "birth_support_count", birth)
        object.__setattr__(self, "update_support_count", update)
        object.__setattr__(self, "eligible", eligible)

    @property
    def eligible_entity_ids(self) -> np.ndarray:
        values = np.ascontiguousarray(self.entity_ids[self.eligible])
        values.setflags(write=False)
        return values

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360PrefixAssociationSupportScreen",
            "config": asdict(self.config),
            "entity_ids": self.entity_ids.tolist(),
            "birth_support_count": self.birth_support_count.tolist(),
            "update_support_count": self.update_support_count.tolist(),
            "eligible": self.eligible.tolist(),
            "eligible_entity_ids": self.eligible_entity_ids.tolist(),
            "birth_valid_sha256": self.birth_valid_sha256,
            "update_valid_sha256": self.update_valid_sha256,
            "physical_endpoint_sha256": self.physical_endpoint_sha256,
            "information_boundary": {
                "observed_frames_read": [
                    self.config.birth_frame,
                    self.config.update_frame,
                ],
                "association_probability_used_for_selection": False,
                "association_entropy_used_for_selection": False,
                "state_innovation_used_for_reliability": False,
                "future_frame_after_update_read": False,
                "future_identity_read": False,
                "target_metric_read": False,
            },
        }


def build_prefix_association_support_screen(
    physical_positions_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    *,
    candidate_entity_ids: np.ndarray | None = None,
    config: PrefixAssociationSupportConfig | None = None,
) -> PrefixAssociationSupportScreen:
    """Screen identities by endpoint association existence, not confidence."""

    cfg = config or PrefixAssociationSupportConfig()
    physical = np.asarray(physical_positions_m, dtype=np.float64)
    depths = np.asarray(depths_m)
    masks = np.asarray(object_masks, dtype=bool)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[2] == 3
        and len(physical) > cfg.update_frame
        and np.all(
            np.isfinite(
                physical[[cfg.birth_frame, cfg.update_frame]]
            )
        ),
        "physical endpoint positions are invalid",
    )
    camera_count = len(matrices)
    _require(
        matrices.shape == (camera_count, 3, 3)
        and poses.shape == (camera_count, 4, 4),
        "support-screen camera calibration is invalid",
    )
    _require(
        depths.ndim == 4
        and depths.shape[0] == camera_count
        and depths.shape[1] > cfg.update_frame
        and masks.shape == depths.shape,
        "support-screen depth or mask volume is invalid",
    )
    _require(
        camera_count >= cfg.minimum_camera_support,
        "support-screen camera count is insufficient",
    )
    entities = (
        np.arange(physical.shape[1], dtype=np.int64)
        if candidate_entity_ids is None
        else np.asarray(candidate_entity_ids, dtype=np.int64)
    )
    _require(
        entities.ndim == 1
        and len(entities) > 0
        and len(np.unique(entities)) == len(entities)
        and np.all((entities >= 0) & (entities < physical.shape[1])),
        "support-screen candidate IDs are invalid",
    )
    entities = np.sort(entities, kind="mergesort")
    projections = projection_matrices(matrices, poses)
    association_cfg = cfg.association_config()
    endpoint_valid: list[np.ndarray] = []
    for frame in (cfg.birth_frame, cfg.update_frame):
        proposal = propose_birth_query_pixels(
            physical[frame, entities],
            projections,
            poses,
            depths[:, frame],
            masks[:, frame],
            config=association_cfg,
        )
        endpoint_valid.append(
            np.asarray(proposal["valid"], dtype=bool)
        )
    birth_valid, update_valid = endpoint_valid
    birth_support = np.sum(birth_valid, axis=0)
    update_support = np.sum(update_valid, axis=0)
    eligible = (
        birth_support >= cfg.minimum_camera_support
    ) & (update_support >= cfg.minimum_camera_support)
    physical_endpoints = physical[
        [cfg.birth_frame, cfg.update_frame]
    ][:, entities]
    descriptor: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360PrefixAssociationSupportScreen",
        "config": asdict(cfg),
        "entity_ids": entities.tolist(),
        "birth_support_count": birth_support.tolist(),
        "update_support_count": update_support.tolist(),
        "eligible": eligible.tolist(),
        "eligible_entity_ids": entities[eligible].tolist(),
        "birth_valid_sha256": array_sha256(birth_valid),
        "update_valid_sha256": array_sha256(update_valid),
        "physical_endpoint_sha256": array_sha256(physical_endpoints),
        "information_boundary": {
            "observed_frames_read": [cfg.birth_frame, cfg.update_frame],
            "association_probability_used_for_selection": False,
            "association_entropy_used_for_selection": False,
            "state_innovation_used_for_reliability": False,
            "future_frame_after_update_read": False,
            "future_identity_read": False,
            "target_metric_read": False,
        },
    }
    artifact_sha256 = _canonical_sha256(
        descriptor,
        digest_key="artifact_sha256",
    )
    result = PrefixAssociationSupportScreen(
        entity_ids=entities,
        birth_support_count=birth_support,
        update_support_count=update_support,
        eligible=eligible,
        config=cfg,
        birth_valid_sha256=descriptor["birth_valid_sha256"],
        update_valid_sha256=descriptor["update_valid_sha256"],
        physical_endpoint_sha256=descriptor["physical_endpoint_sha256"],
        artifact_sha256=artifact_sha256,
    )
    _require(
        _canonical_sha256(
            result.descriptor(),
            digest_key="artifact_sha256",
        )
        == result.artifact_sha256,
        "prefix-support descriptor changed after construction",
    )
    return result


__all__ = [
    "PrefixAssociationSupportConfig",
    "PrefixAssociationSupportScreen",
    "build_prefix_association_support_screen",
]
