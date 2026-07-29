"""Adaptive complete-camera carrier for causal-response state updates.

The module is deliberately limited to frame-zero carrier construction. It uses
only complete causal camera streams, registered geometry, physical action
support, and frame-zero RGB-D masks. It never consumes tracker output,
innovations, identity targets, future observations, or evaluation metrics.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_query import (
    CausalResponseQueryConfig,
    CausalResponseQuerySchedule,
    build_causal_response_query_schedule,
)
from .deform360_dynamic_query import projection_matrices
from .observation_belief import array_sha256, file_sha256
from .tapnextpp_birth_association import (
    SET_VALUED_COVARIANCE_ASSOCIATION,
    BirthAssociationConfig,
    propose_birth_query_pixels,
)

CONTRACT = "deform360-causal-response-adaptive-query-v13"
ARTIFACT_KIND = "Deform360CausalResponseAdaptiveQueryV13"
ARCHIVE_FILENAME = "causal_response_adaptive_query_v13.npz"
REPORT_FILENAME = "causal_response_adaptive_query_v13.json"
STRICT_ARM = "strict_3plus3"
INFLATED_FALLBACK_ARM = "inflated_2plus2"
ABSTAINED_ARM = "abstained_insufficient_2plus2"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(payload: Mapping[str, Any], *, key: str) -> str:
    canonical = dict(payload)
    canonical.pop(key, None)
    return hashlib.sha256(
        b"deform360-causal-response-adaptive-query-v13\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AdaptiveCausalResponseQueryConfig:
    """Frozen adaptive-panel and support-ladder settings."""

    prefix_frame_count: int = 58
    query_count: int = 16
    graph_basis_rank: int = 8
    selected_camera_count: int = 8
    panel_camera_count: int = 4
    strict_minimum_support_per_panel: int = 3
    fallback_minimum_support_per_panel: int = 2
    fallback_covariance_inflation: float = 4.0
    shared_bias_std_m: float = 0.005
    action_support_threshold: float = 0.10
    association_support_probability: float = 0.50
    association_search_radius_px: int = 12
    association_depth_scale_m: float = 0.03
    association_minimum_candidate_count: int = 1
    visibility_weight: float = 0.50
    mode_information_weight: float = 1.00
    spatial_diversity_weight: float = 1.00

    def __post_init__(self) -> None:
        _require(self.prefix_frame_count >= 2, "prefix is too short")
        _require(self.query_count >= 1, "query count must be positive")
        _require(self.graph_basis_rank >= 1, "graph rank must be positive")
        _require(
            self.selected_camera_count == 2 * self.panel_camera_count
            and self.panel_camera_count >= 4,
            "adaptive panels must be equal disjoint panels of at least four cameras",
        )
        _require(
            2
            <= self.fallback_minimum_support_per_panel
            < self.strict_minimum_support_per_panel
            <= self.panel_camera_count,
            "support ladder is invalid",
        )
        _require(
            np.isfinite(self.fallback_covariance_inflation)
            and self.fallback_covariance_inflation >= 4.0,
            "fallback covariance inflation must be at least fourfold",
        )
        _require(
            np.isfinite(self.shared_bias_std_m) and self.shared_bias_std_m > 0.0,
            "shared bias standard deviation must be positive",
        )
        _require(
            np.isfinite(self.action_support_threshold)
            and 0.0 < self.action_support_threshold <= 1.0,
            "action support threshold must lie in (0, 1]",
        )
        _require(
            np.isfinite(self.association_support_probability)
            and 0.0 < self.association_support_probability <= 1.0,
            "association support threshold must lie in (0, 1]",
        )
        _require(
            self.association_search_radius_px >= 1
            and self.association_minimum_candidate_count >= 1
            and np.isfinite(self.association_depth_scale_m)
            and self.association_depth_scale_m > 0.0,
            "association settings are invalid",
        )
        weights = (
            self.visibility_weight,
            self.mode_information_weight,
            self.spatial_diversity_weight,
        )
        _require(
            all(np.isfinite(value) and value >= 0.0 for value in weights)
            and any(value > 0.0 for value in weights),
            "query score weights are invalid",
        )


@dataclass(frozen=True)
class AdaptiveCameraPanels:
    """Deterministic disjoint camera panels and their carrier counts."""

    proposal_indices: np.ndarray
    validation_indices: np.ndarray
    strict_eligible_count: int
    fallback_eligible_count: int
    supported_incidence_count: int
    association_probability_mass: float

    def __post_init__(self) -> None:
        proposal = np.ascontiguousarray(
            np.asarray(self.proposal_indices, dtype=np.int64)
        )
        validation = np.ascontiguousarray(
            np.asarray(self.validation_indices, dtype=np.int64)
        )
        _require(
            proposal.ndim == validation.ndim == 1
            and len(proposal) == len(validation)
            and len(proposal) >= 4
            and len(np.unique(proposal)) == len(proposal)
            and len(np.unique(validation)) == len(validation)
            and not np.intersect1d(proposal, validation).size,
            "adaptive camera panels are invalid",
        )
        _require(
            self.strict_eligible_count >= 0
            and self.fallback_eligible_count >= self.strict_eligible_count
            and self.supported_incidence_count >= 0
            and np.isfinite(self.association_probability_mass)
            and self.association_probability_mass >= 0.0,
            "adaptive panel score is invalid",
        )
        proposal.setflags(write=False)
        validation.setflags(write=False)
        object.__setattr__(self, "proposal_indices", proposal)
        object.__setattr__(self, "validation_indices", validation)

    @property
    def selected_indices(self) -> np.ndarray:
        selected = np.concatenate((self.proposal_indices, self.validation_indices))
        selected.setflags(write=False)
        return selected

    @property
    def score(self) -> tuple[int, int, int, float]:
        return (
            self.strict_eligible_count,
            self.fallback_eligible_count,
            self.supported_incidence_count,
            self.association_probability_mass,
        )


@dataclass(frozen=True)
class AdaptiveCausalResponseQuerySchedule:
    """Outer V13 provenance around one immutable query schedule."""

    config: AdaptiveCausalResponseQueryConfig
    available_camera_ids: tuple[str, ...]
    panels: AdaptiveCameraPanels
    arm: str
    covariance_inflation: float
    query_schedule: CausalResponseQuerySchedule
    input_array_sha256: Mapping[str, str]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(
            len(self.available_camera_ids) >= self.config.selected_camera_count
            and len(set(self.available_camera_ids)) == len(self.available_camera_ids),
            "available camera identifiers are invalid",
        )
        _require(
            np.all(self.panels.selected_indices >= 0)
            and np.all(self.panels.selected_indices < len(self.available_camera_ids))
            and len(self.panels.proposal_indices) == self.config.panel_camera_count
            and len(self.panels.validation_indices) == self.config.panel_camera_count,
            "adaptive panels reference unavailable cameras",
        )
        _require(
            self.arm in {STRICT_ARM, INFLATED_FALLBACK_ARM, ABSTAINED_ARM},
            "adaptive support arm is invalid",
        )
        expected_support = (
            self.config.strict_minimum_support_per_panel
            if self.arm == STRICT_ARM
            else self.config.fallback_minimum_support_per_panel
        )
        expected_inflation = (
            1.0 if self.arm == STRICT_ARM else self.config.fallback_covariance_inflation
        )
        _require(
            self.query_schedule.config.minimum_camera_support_per_panel
            == expected_support,
            "query schedule support differs from its adaptive arm",
        )
        _require(
            np.isclose(self.covariance_inflation, expected_inflation),
            "adaptive covariance inflation changed",
        )
        selected_names = tuple(
            self.available_camera_ids[index] for index in self.panels.selected_indices
        )
        _require(
            self.query_schedule.camera_ids == selected_names,
            "query schedule cameras differ from the adaptive panel",
        )
        if self.arm == STRICT_ARM:
            _require(
                self.panels.strict_eligible_count >= self.config.query_count,
                "strict arm lacks the registered query budget",
            )
        elif self.arm == INFLATED_FALLBACK_ARM:
            _require(
                self.panels.strict_eligible_count
                < self.config.query_count
                <= self.panels.fallback_eligible_count,
                "inflated fallback did not follow the registered ladder",
            )
        else:
            _require(
                self.panels.fallback_eligible_count < self.config.query_count
                and not self.query_schedule.admitted,
                "abstention has a complete fallback carrier",
            )
        for digest in (*self.input_array_sha256.values(), self.artifact_sha256):
            _require(_valid_digest(str(digest)), "adaptive query digest is invalid")
        object.__setattr__(
            self,
            "input_array_sha256",
            dict(sorted(self.input_array_sha256.items())),
        )

    @property
    def admitted(self) -> bool:
        return self.query_schedule.admitted

    @property
    def selected_camera_ids(self) -> tuple[str, ...]:
        return tuple(
            self.available_camera_ids[index] for index in self.panels.selected_indices
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            **self.query_schedule.arrays(),
            "selected_complete_camera_indices": self.panels.selected_indices,
            "proposal_complete_camera_indices": self.panels.proposal_indices,
            "validation_complete_camera_indices": self.panels.validation_indices,
        }

    def descriptor(self) -> dict[str, Any]:
        arrays = self.arrays()
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalResponseAdaptiveQueryScheduleV13",
            "contract": CONTRACT,
            "config": asdict(self.config),
            "arm": self.arm,
            "admitted": self.admitted,
            "covariance_inflation": self.covariance_inflation,
            "shared_bias_variance_m2": self.config.shared_bias_std_m**2,
            "available_camera_ids": list(self.available_camera_ids),
            "selected_camera_ids": list(self.selected_camera_ids),
            "proposal_complete_camera_indices": self.panels.proposal_indices.tolist(),
            "validation_complete_camera_indices": (
                self.panels.validation_indices.tolist()
            ),
            "panel_score": {
                "strict_eligible_count": self.panels.strict_eligible_count,
                "fallback_eligible_count": self.panels.fallback_eligible_count,
                "supported_incidence_count": self.panels.supported_incidence_count,
                "association_probability_mass": (
                    self.panels.association_probability_mass
                ),
            },
            "query_schedule": self.query_schedule.descriptor(),
            "input_array_sha256": dict(self.input_array_sha256),
            "output_array_sha256": {
                name: array_sha256(values) for name, values in sorted(arrays.items())
            },
            "information_boundary": {
                "object_observation_frames_used_for_selection": [0],
                "physical_frames_used_for_selection": [0],
                "complete_causal_camera_certificate_used": True,
                "known_action_support_used": True,
                "tracker_output_read": False,
                "state_innovation_read": False,
                "identity_target_read": False,
                "future_object_observation_read": False,
                "future_metric_read": False,
                "state_update_constructed": False,
                "proposal_and_validation_panels_disjoint": True,
                "two_view_fallback_requires_covariance_inflation": True,
                "shared_bias_nuisance_declared": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def select_adaptive_camera_panels(
    association_valid: np.ndarray,
    association_probability: np.ndarray,
    action_support: np.ndarray,
    *,
    config: AdaptiveCausalResponseQueryConfig | None = None,
) -> AdaptiveCameraPanels:
    """Select a deterministic target-free 4+4 panel from complete cameras."""

    cfg = config or AdaptiveCausalResponseQueryConfig()
    valid = np.asarray(association_valid, dtype=bool)
    probability = np.asarray(association_probability, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
    _require(
        valid.ndim == 2
        and probability.shape == valid.shape
        and support.shape == (valid.shape[1],)
        and valid.shape[0] >= cfg.selected_camera_count,
        "adaptive panel inputs changed shape",
    )
    _require(
        np.all(np.isfinite(probability))
        and np.all((probability >= 0.0) & (probability <= 1.0))
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "adaptive panel probabilities are invalid",
    )
    supported = valid & (probability >= cfg.association_support_probability)
    action_eligible = support >= cfg.action_support_threshold
    camera_ids = tuple(range(valid.shape[0]))
    best: AdaptiveCameraPanels | None = None
    best_key: (
        tuple[tuple[int, int, int, float], tuple[int, ...], tuple[int, ...]] | None
    ) = None
    for selected in itertools.combinations(camera_ids, cfg.selected_camera_count):
        selected_set = set(selected)
        for proposal in itertools.combinations(selected, cfg.panel_camera_count):
            validation = tuple(sorted(selected_set.difference(proposal)))
            if proposal > validation:
                continue
            proposal_count = np.sum(supported[np.asarray(proposal)], axis=0)
            validation_count = np.sum(supported[np.asarray(validation)], axis=0)
            strict = (
                action_eligible
                & (proposal_count >= cfg.strict_minimum_support_per_panel)
                & (validation_count >= cfg.strict_minimum_support_per_panel)
            )
            fallback = (
                action_eligible
                & (proposal_count >= cfg.fallback_minimum_support_per_panel)
                & (validation_count >= cfg.fallback_minimum_support_per_panel)
            )
            incidence = int(
                np.sum((proposal_count + validation_count)[action_eligible])
            )
            mass = float(
                np.sum(
                    probability[np.asarray(selected)][:, action_eligible],
                    dtype=np.float64,
                )
            )
            candidate = AdaptiveCameraPanels(
                proposal_indices=np.asarray(proposal, dtype=np.int64),
                validation_indices=np.asarray(validation, dtype=np.int64),
                strict_eligible_count=int(np.sum(strict)),
                fallback_eligible_count=int(np.sum(fallback)),
                supported_incidence_count=incidence,
                association_probability_mass=mass,
            )
            key = (candidate.score, proposal, validation)
            if best is None:
                best = candidate
                best_key = key
                continue
            assert best_key is not None
            if candidate.score > best_key[0] or (
                candidate.score == best_key[0]
                and (proposal, validation) < (best_key[1], best_key[2])
            ):
                best = candidate
                best_key = key
    if best is None:
        raise AssertionError("adaptive panel selector produced no panel")
    return best


def build_adaptive_causal_response_query_schedule(
    physical_frame_zero_m: np.ndarray,
    graph_basis: np.ndarray,
    action_support: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    frame_zero_depths_m: np.ndarray,
    frame_zero_object_masks: np.ndarray,
    *,
    camera_ids: tuple[str, ...],
    config: AdaptiveCausalResponseQueryConfig | None = None,
) -> AdaptiveCausalResponseQuerySchedule:
    """Build the strict-first adaptive V13 carrier without outcome access."""

    cfg = config or AdaptiveCausalResponseQueryConfig()
    frame_zero = np.asarray(physical_frame_zero_m, dtype=np.float64)
    basis = np.asarray(graph_basis, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    depths = np.asarray(frame_zero_depths_m, dtype=np.float64)
    masks = np.asarray(frame_zero_object_masks, dtype=bool)
    camera_count = len(camera_ids)
    _require(
        frame_zero.ndim == 2
        and frame_zero.shape[1] == 3
        and basis.shape[0] == len(frame_zero)
        and support.shape == (len(frame_zero),),
        "adaptive physical inputs changed shape",
    )
    _require(
        matrices.shape == (camera_count, 3, 3)
        and poses.shape == (camera_count, 4, 4)
        and depths.ndim == 3
        and depths.shape[0] == camera_count
        and masks.shape == depths.shape
        and camera_count >= cfg.selected_camera_count,
        "adaptive camera inputs changed shape",
    )
    association = propose_birth_query_pixels(
        frame_zero,
        projection_matrices(matrices, poses),
        poses,
        depths,
        masks,
        config=BirthAssociationConfig(
            search_radius_px=cfg.association_search_radius_px,
            depth_scale_m=cfg.association_depth_scale_m,
            minimum_candidate_count=cfg.association_minimum_candidate_count,
            association_mode=SET_VALUED_COVARIANCE_ASSOCIATION,
        ),
    )
    panels = select_adaptive_camera_panels(
        association["valid"],
        association["association_probability"],
        support,
        config=cfg,
    )
    if panels.strict_eligible_count >= cfg.query_count:
        arm = STRICT_ARM
        minimum_support = cfg.strict_minimum_support_per_panel
        inflation = 1.0
    elif panels.fallback_eligible_count >= cfg.query_count:
        arm = INFLATED_FALLBACK_ARM
        minimum_support = cfg.fallback_minimum_support_per_panel
        inflation = cfg.fallback_covariance_inflation
    else:
        arm = ABSTAINED_ARM
        minimum_support = cfg.fallback_minimum_support_per_panel
        inflation = cfg.fallback_covariance_inflation
    selected = panels.selected_indices
    local_proposal = np.arange(cfg.panel_camera_count, dtype=np.int64)
    local_validation = np.arange(
        cfg.panel_camera_count,
        cfg.selected_camera_count,
        dtype=np.int64,
    )
    query_schedule = build_causal_response_query_schedule(
        frame_zero,
        basis,
        support,
        matrices[selected],
        poses[selected],
        depths[selected],
        masks[selected],
        camera_ids=tuple(camera_ids[index] for index in selected),
        proposal_camera_indices=local_proposal,
        validation_camera_indices=local_validation,
        config=CausalResponseQueryConfig(
            prefix_frame_count=cfg.prefix_frame_count,
            query_count=cfg.query_count,
            graph_basis_rank=cfg.graph_basis_rank,
            action_support_threshold=cfg.action_support_threshold,
            minimum_camera_support_per_panel=minimum_support,
            association_support_probability=cfg.association_support_probability,
            association_search_radius_px=cfg.association_search_radius_px,
            association_depth_scale_m=cfg.association_depth_scale_m,
            association_minimum_candidate_count=(
                cfg.association_minimum_candidate_count
            ),
            visibility_weight=cfg.visibility_weight,
            mode_information_weight=cfg.mode_information_weight,
            spatial_diversity_weight=cfg.spatial_diversity_weight,
        ),
    )
    input_hashes = {
        "physical_frame_zero_m": array_sha256(frame_zero),
        "graph_basis": array_sha256(basis),
        "action_support": array_sha256(support),
        "intrinsics": array_sha256(matrices),
        "camera_to_world": array_sha256(poses),
        "frame_zero_depths_m": array_sha256(depths),
        "frame_zero_object_masks": array_sha256(masks),
    }
    provisional = AdaptiveCausalResponseQuerySchedule(
        config=cfg,
        available_camera_ids=tuple(map(str, camera_ids)),
        panels=panels,
        arm=arm,
        covariance_inflation=inflation,
        query_schedule=query_schedule,
        input_array_sha256=input_hashes,
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor(), key="artifact_sha256")
    result = AdaptiveCausalResponseQuerySchedule(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )
    _require(
        _canonical_sha256(result.descriptor(), key="artifact_sha256")
        == result.artifact_sha256,
        "adaptive query descriptor changed after construction",
    )
    return result


def write_adaptive_causal_response_query_artifacts(
    output_dir: str | Path,
    schedule: AdaptiveCausalResponseQuerySchedule,
    *,
    case_id: str,
    repository_revision: str,
    protocol_path: str | Path,
    physical_manifest_path: str | Path,
    physical_archive_path: str | Path,
    camera_certificate_sha256: str,
) -> dict[str, Any]:
    """Seal one target-free V13 carrier disposition."""

    _require(bool(case_id), "case ID is empty")
    _require(
        len(repository_revision) == 40
        and all(character in "0123456789abcdef" for character in repository_revision),
        "repository revision is invalid",
    )
    _require(_valid_digest(camera_certificate_sha256), "camera digest is invalid")
    output = Path(output_dir).resolve()
    _require(not output.exists(), "adaptive query output directory already exists")
    output.mkdir(parents=True)
    arrays = schedule.arrays()
    archive_path = output / ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        **{
            name: np.ascontiguousarray(np.asarray(values))
            for name, values in arrays.items()
        },
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "contract": CONTRACT,
        "case": case_id,
        "status": "admitted" if schedule.admitted else "abstained",
        "repository_revision": repository_revision,
        "schedule": schedule.descriptor(),
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "physical_manifest": file_sha256(physical_manifest_path),
            "physical_archive": file_sha256(physical_archive_path),
            "camera_certificate": camera_certificate_sha256,
        },
        "archive": {
            "filename": ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(values) for name, values in sorted(arrays.items())
            },
        },
        "information_boundary": {
            **schedule.descriptor()["information_boundary"],
            "tactile_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(report, key="result_sha256")
    (output / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_adaptive_causal_response_query_artifacts(output)
    return report


def validate_adaptive_causal_response_query_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate one immutable V13 carrier disposition."""

    output = Path(output_dir).resolve()
    report = json.loads((output / REPORT_FILENAME).read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == ARTIFACT_KIND
        and report.get("contract") == CONTRACT
        and report.get("status") in {"admitted", "abstained"}
        and report.get("result_sha256")
        == _canonical_sha256(report, key="result_sha256"),
        "adaptive query report is invalid",
    )
    schedule = report["schedule"]
    _require(
        schedule.get("artifact_sha256")
        == _canonical_sha256(schedule, key="artifact_sha256"),
        "adaptive query schedule checksum changed",
    )
    boundary = report["information_boundary"]
    required_false = (
        "tracker_output_read",
        "state_innovation_read",
        "identity_target_read",
        "future_object_observation_read",
        "future_metric_read",
        "state_update_constructed",
        "tactile_read",
        "held_v8_artifact_or_process_access",
    )
    _require(
        boundary.get("object_observation_frames_used_for_selection") == [0]
        and boundary.get("physical_frames_used_for_selection") == [0]
        and all(boundary.get(name) is False for name in required_false),
        "adaptive query crossed its information boundary",
    )
    archive_path = output / ARCHIVE_FILENAME
    _require(
        report["archive"]["filename"] == ARCHIVE_FILENAME
        and report["archive"]["file_sha256"] == file_sha256(archive_path),
        "adaptive query archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    observed = {name: array_sha256(values) for name, values in sorted(arrays.items())}
    _require(
        observed
        == report["archive"]["array_sha256"]
        == schedule["output_array_sha256"],
        "adaptive query array checksum changed",
    )
    _require(
        report["status"] == ("admitted" if schedule["admitted"] else "abstained"),
        "adaptive query status differs from its schedule",
    )
    return report, arrays


__all__ = [
    "ABSTAINED_ARM",
    "ARCHIVE_FILENAME",
    "ARTIFACT_KIND",
    "CONTRACT",
    "INFLATED_FALLBACK_ARM",
    "REPORT_FILENAME",
    "STRICT_ARM",
    "AdaptiveCameraPanels",
    "AdaptiveCausalResponseQueryConfig",
    "AdaptiveCausalResponseQuerySchedule",
    "build_adaptive_causal_response_query_schedule",
    "select_adaptive_camera_panels",
    "validate_adaptive_causal_response_query_artifacts",
    "write_adaptive_causal_response_query_artifacts",
]
