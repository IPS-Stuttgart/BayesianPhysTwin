from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_preflight import (
    deform360_object_hash,
)
from bayesian_phystwin.deform360_event_conditioned_window_v15 import (
    EventConditionedWindowConfig,
    EventPanelEvidence,
    select_event_conditioned_window,
)
from bayesian_phystwin.deform360_event_shape_signature_v15 import (
    EventShapeSignatureConfig,
)
from bayesian_phystwin.deform360_object_exclusion import (
    file_sha256,
    load_object_exclusion_manifest,
)


def _config() -> EventConditionedWindowConfig:
    return EventConditionedWindowConfig(
        lag_frames=6,
        first_candidate_frame=8,
        forecast_horizon_frames=10,
    )


def _panel(
    signature_m: np.ndarray,
    *,
    camera_support: int = 3,
    gripper_clear: np.ndarray | None = None,
) -> EventPanelEvidence:
    frame_count, component_count = signature_m.shape
    return EventPanelEvidence(
        component_signature_m=signature_m,
        variance_m2=np.full_like(signature_m, 1e-8),
        available=np.ones_like(signature_m, dtype=bool),
        camera_support=np.full_like(
            signature_m,
            camera_support,
            dtype=np.int64,
        ),
        gripper_clear=(
            np.ones_like(signature_m, dtype=bool)
            if gripper_clear is None
            else gripper_clear
        ),
        component_ids=np.arange(component_count, dtype=np.int64),
    )


def _scene(
    *,
    event_frame: int | None = 10,
) -> tuple[EventPanelEvidence, EventPanelEvidence, np.ndarray, np.ndarray]:
    frame_count = 40
    base = np.asarray([0.08, 0.10, 0.12, 0.14], dtype=np.float64)
    proposal_signature = np.repeat(base[None], frame_count, axis=0)
    validation_signature = proposal_signature.copy()
    if event_frame is not None:
        change = np.asarray([0.0030, -0.0025, 0.0020, -0.0035])
        proposal_signature[event_frame:] += change
        validation_signature[event_frame:] += 0.95 * change
    tactile = np.zeros(frame_count, dtype=np.float64)
    tactile[6:] = 0.9
    actuator = np.zeros((frame_count, 1, 3), dtype=np.float64)
    actuator[:, 0, 0] = 0.00025 * np.arange(frame_count)
    return (
        _panel(proposal_signature),
        _panel(validation_signature),
        tactile,
        actuator,
    )


def _select(
    scene: tuple[EventPanelEvidence, EventPanelEvidence, np.ndarray, np.ndarray],
):
    proposal, validation, tactile, actuator = scene
    return select_event_conditioned_window(
        "fresh-object-token",
        proposal,
        validation,
        tactile,
        actuator,
        config=_config(),
    )


def test_selects_earliest_causal_event_and_reserves_fixed_future() -> None:
    result = _select(_scene())

    assert result.admitted
    assert result.selected_attempt is not None
    assert result.selected_attempt.branch_frame == 10
    assert result.selected_attempt.birth_frame == 4
    assert result.forecast_frame_range_half_open == (11, 21)
    assert [attempt.branch_frame for attempt in result.attempts] == [8, 9, 10]
    assert result.descriptor()["information_boundary"][
        "physical_prediction_used_for_population_selection"
    ] is False


def test_future_values_after_selected_branch_cannot_change_the_artifact() -> None:
    original = _scene()
    proposal, validation, tactile, actuator = original
    modified_proposal = proposal.component_signature_m.copy()
    modified_validation = validation.component_signature_m.copy()
    modified_tactile = tactile.copy()
    modified_actuator = actuator.copy()
    modified_proposal[11:] = np.nan
    modified_validation[11:] = 1e6
    modified_tactile[11:] = np.nan
    modified_actuator[11:] = np.nan
    modified = (
        _panel(modified_proposal),
        _panel(modified_validation),
        modified_tactile,
        modified_actuator,
    )

    first = _select(original)
    second = _select(modified)

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.descriptor() == second.descriptor()


def test_no_event_is_an_abstention_and_reserved_tail_is_not_read() -> None:
    original = _scene(event_frame=None)
    proposal, validation, tactile, actuator = original
    modified_proposal = proposal.component_signature_m.copy()
    modified_validation = validation.component_signature_m.copy()
    modified_tactile = tactile.copy()
    modified_actuator = actuator.copy()
    modified_proposal[30:] = np.nan
    modified_validation[30:] = 1e6
    modified_tactile[30:] = np.nan
    modified_actuator[30:] = np.nan
    modified = (
        _panel(modified_proposal),
        _panel(modified_validation),
        modified_tactile,
        modified_actuator,
    )

    first = _select(original)
    second = _select(modified)

    assert not first.admitted
    assert first.forecast_frame_range_half_open is None
    assert first.maximum_observation_frame == 29
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.descriptor()["information_boundary"][
        "no_event_is_an_explicit_abstention"
    ] is True


def test_rigid_motion_does_not_trigger_a_pairwise_shape_event() -> None:
    result = _select(_scene(event_frame=None))

    assert not result.admitted
    assert {
        attempt.reason for attempt in result.attempts
    } == {"insufficient-gripper-excluded-nonrigid-response"}


def test_cross_panel_direction_disagreement_rejects_the_event() -> None:
    proposal, validation, tactile, actuator = _scene()
    disagreeing = validation.component_signature_m.copy()
    base = disagreeing[9].copy()
    disagreeing[10:] = 2.0 * base - disagreeing[10:]

    result = _select(
        (
            proposal,
            _panel(disagreeing),
            tactile,
            actuator,
        )
    )

    assert not result.admitted
    assert any(
        attempt.reason == "cross-panel-direction-disagreement"
        for attempt in result.attempts
    )


def test_gripper_overlapping_components_cannot_form_an_event() -> None:
    proposal, validation, tactile, actuator = _scene()
    blocked = np.ones_like(proposal.gripper_clear)
    blocked[:, :2] = False

    result = _select(
        (
            _panel(proposal.component_signature_m, gripper_clear=blocked),
            _panel(validation.component_signature_m, gripper_clear=blocked),
            tactile,
            actuator,
        )
    )

    assert not result.admitted
    assert result.attempts[0].reason == "insufficient-shape-component-support"


def test_duplicated_camera_count_does_not_increase_event_confidence() -> None:
    proposal, validation, tactile, actuator = _scene()
    ordinary = _select((proposal, validation, tactile, actuator))
    duplicated = _select(
        (
            _panel(proposal.component_signature_m, camera_support=100),
            _panel(validation.component_signature_m, camera_support=100),
            tactile,
            actuator,
        )
    )

    assert ordinary.admitted and duplicated.admitted
    assert ordinary.selected_attempt is not None
    assert duplicated.selected_attempt is not None
    assert (
        ordinary.selected_attempt.proposal_signal_to_noise
        == duplicated.selected_attempt.proposal_signal_to_noise
    )
    assert (
        ordinary.selected_attempt.validation_signal_to_noise
        == duplicated.selected_attempt.validation_signal_to_noise
    )
    assert (
        ordinary.selected_attempt.cross_panel_reduced_nis
        == duplicated.selected_attempt.cross_panel_reduced_nis
    )


def test_tactile_is_causal_support_but_not_metric_evidence() -> None:
    proposal, validation, _, actuator = _scene()
    without_contact = np.zeros(proposal.frame_count, dtype=np.float64)

    result = _select((proposal, validation, without_contact, actuator))

    assert not result.admitted
    assert any(
        attempt.reason == "insufficient-tactile-contact-support"
        for attempt in result.attempts
    )


def test_panel_component_contract_must_match() -> None:
    proposal, validation, tactile, actuator = _scene()
    mismatched = EventPanelEvidence(
        component_signature_m=validation.component_signature_m,
        variance_m2=validation.variance_m2,
        available=validation.available,
        camera_support=validation.camera_support,
        gripper_clear=validation.gripper_clear,
        component_ids=np.asarray([0, 1, 2, 9]),
    )

    with pytest.raises(ValueError, match="component contract"):
        select_event_conditioned_window(
            "fresh-object-token",
            proposal,
            mismatched,
            tactile,
            actuator,
            config=_config(),
        )


def test_prelock_binds_closed_arms_and_exact_selector_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    prelock = json.loads(
        (
            root
            / "configs"
            / "sota"
            / "deform360_event_conditioned_belief_v15_prelock.json"
        ).read_text(encoding="utf-8")
    )
    selector = dict(prelock["event_selector"])
    assert selector.pop("contract") == "deform360-event-conditioned-window-v15"
    assert selector == asdict(EventConditionedWindowConfig())
    provider = dict(prelock["shape_signature_provider"])
    assert provider.pop("contract") == "deform360-event-shape-signature-v15"
    for implementation_note in (
        "panel_aggregation",
        "temporal_processing",
        "tracker_or_material_identity_used",
        "physical_prediction_used",
    ):
        provider.pop(implementation_note)
    assert provider == json.loads(
        json.dumps(asdict(EventShapeSignatureConfig()))
    )
    assert prelock["status"] == (
        "implementation_prelock_before_any_v15_source_selection"
    )
    assert not any(
        prelock["scope"][key]
        for key in (
            "v12_rerun_authorized",
            "v13_rerun_authorized",
            "v14_rerun_authorized",
            "source_or_target_outcome_opened",
        )
    )
    exclusion = prelock["held_v8_exclusion_provenance"]
    exclusion_path = root / exclusion["path"]
    assert hashlib.sha256(exclusion_path.read_bytes()).hexdigest() == exclusion[
        "file_sha256"
    ]
    exclusion_payload = json.loads(exclusion_path.read_text(encoding="utf-8"))
    assert exclusion_payload["exclusion_sha256"] == exclusion["canonical_sha256"]
    fresh = prelock["fresh_object_exclusion"]
    fresh_path = root / fresh["path"]
    fresh_payload = load_object_exclusion_manifest(fresh_path)
    assert file_sha256(fresh_path) == fresh["file_sha256"]
    assert fresh_payload["exclusion_sha256"] == fresh["canonical_sha256"]
    assert (
        len(fresh_payload["object_hashes"])
        == fresh["unique_object_hash_count"]
        == 191
    )
    prior = load_object_exclusion_manifest(
        root / "configs" / "sota" / "deform360_fresh_object_exclusion_v14.json"
    )
    reserved = load_object_exclusion_manifest(
        root
        / "configs"
        / "sota"
        / "deform360_v14_reserved_queue_exclusion_v1.json"
    )
    assert len(prior["object_hashes"]) == 138
    assert len(reserved["object_hashes"]) == 53
    assert not set(prior["object_hashes"]).intersection(reserved["object_hashes"])
    assert set(fresh_payload["object_hashes"]) == set(
        prior["object_hashes"]
    ).union(reserved["object_hashes"])

    feasibility_path = root / prelock["deform360_source_feasibility"]["path"]
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    canonical = dict(feasibility)
    digest = canonical.pop("artifact_sha256")
    assert digest == hashlib.sha256(
        b"deform360-event-conditioned-source-feasibility-v15\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    catalog = json.loads(
        (
            root
            / "results"
            / "sota"
            / "deform360_fresh_source_lock_v1"
            / "public_object_catalog_2026-07-26.json"
        ).read_text(encoding="utf-8")
    )
    public_hashes = {
        deform360_object_hash(str(row["object_id"])) for row in catalog["objects"]
    }
    remaining = public_hashes.difference(fresh_payload["object_hashes"])
    queue = json.loads(
        (
            root
            / "configs"
            / "sota"
            / "deform360_causal_response_direct_depth_v14_staging_queue.json"
        ).read_text(encoding="utf-8")
    )
    rejected = {
        row["object_hash"]
        for row in queue["metadata_dispositions"]["rejected_hash_only"]
    }
    assert len(public_hashes) == 190
    assert len(remaining) == 1
    assert remaining == rejected
    assert feasibility["counts"] == {
        "excluded_public_object_count": 189,
        "exclusion_union_object_count": 191,
        "public_catalog_object_count": 190,
        "remaining_metadata_admissible_object_count": 0,
        "remaining_public_object_count": 1,
        "required_fresh_source_object_count": 12,
    }
