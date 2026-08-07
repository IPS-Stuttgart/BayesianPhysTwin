from __future__ import annotations

import copy
from pathlib import Path

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_calibration_bundle import Deform360CohortUnitV1
from bayesian_phystwin.deform360_calibration_execution import (
    Deform360Stage0SelectionV1,
)
from bayesian_phystwin.deform360_calibration_observability_binding import (
    DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY,
    DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY,
    Deform360ConfirmationOpeningAuthorizationV1,
)
from bayesian_phystwin.deform360_prob4d_target_outcome_authorization import (
    BAYESIAN_PHYSTWIN_REPOSITORY,
    BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY,
    BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY,
    BPT_STAGE0_SOURCE_KEY,
    BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY,
    BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY,
    CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY,
    DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SCHEMA,
    PREDICTION_PROVIDER_REVISION_METADATA_KEY,
    PROB4D_COHORT_BINDING_CLAIM_BOUNDARY,
    PROB4D_COHORT_BINDING_SCHEMA,
    PROB4D_COHORT_BINDING_SOURCE_KEY,
    PROB4D_PROMOTION_LOCK_CLAIM_BOUNDARY,
    PROB4D_PROMOTION_LOCK_SCHEMA,
    PROB4D_PROMOTION_LOCK_SOURCE_KEY,
    PROB4D_REPOSITORY,
    PROB4D_TARGET_ADMISSION_CLAIM_BOUNDARY,
    PROB4D_TARGET_ADMISSION_SCHEMA,
    PROB4D_TARGET_ADMISSION_SOURCE_KEY,
    Deform360Prob4DTargetOutcomeAuthorizationV1,
    build_deform360_prob4d_target_outcome_authorization,
    load_deform360_prob4d_target_outcome_authorization,
    save_deform360_prob4d_target_outcome_authorization,
    verify_deform360_prob4d_target_outcome_authorization,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
)

BPT_REVISION = "a" * 40
PROB4D_REVISION = "b" * 40
PREDICTION_PROVIDER_REVISION = "c" * 40
MOTIONCRAFTER_REVISION = "d" * 40
DATASET_REVISION = "e" * 40
PROCESSING_REVISION = "f" * 40
MODEL_SET_ID = "1" * 64
RUN_SPEC_ID = "2" * 64


def _digest(index: int) -> str:
    return f"{index:064x}"


def _unit(object_id: str, *, stratum: str, index: int) -> Deform360CohortUnitV1:
    return Deform360CohortUnitV1(
        object_id=object_id,
        episode_id=0,
        stratum=stratum,  # type: ignore[arg-type]
        metadata_path=f"raw/{object_id}/metadata.json",
        metadata_sha256=_digest(100 + index),
    )


def _stage0() -> Deform360Stage0SelectionV1:
    calibration = tuple(
        [
            _unit(f"cal-sheet-{index}", stratum="sheet", index=index)
            for index in range(5)
        ]
        + [
            _unit(
                f"cal-volumetric-{index}",
                stratum="volumetric",
                index=5 + index,
            )
            for index in range(5)
        ]
    )
    confirmation = tuple(
        [
            _unit(
                f"target-sheet-{index}",
                stratum="sheet",
                index=10 + index,
            )
            for index in range(6)
        ]
        + [
            _unit(
                f"target-volumetric-{index}",
                stratum="volumetric",
                index=16 + index,
            )
            for index in range(6)
        ]
    )
    return Deform360Stage0SelectionV1(
        source_sha256=_digest(200),
        selection_artifact_sha256=_digest(201),
        selection_sha256=_digest(202),
        content_selection_sha256=_digest(203),
        protocol_sha256=_digest(204),
        dataset_revision=DATASET_REVISION,
        processing_revision=PROCESSING_REVISION,
        implementation_revision=BPT_REVISION,
        calibration_units=calibration,
        confirmation_units=confirmation,
    )


def _visual_provider() -> Deform360VisualProviderLockV1:
    return Deform360VisualProviderLockV1(
        provider_revision=PREDICTION_PROVIDER_REVISION,
        provider_manifest_id=_digest(210),
        provider_attestation_sha256=_digest(211),
        motioncrafter_revision=MOTIONCRAFTER_REVISION,
        model_set_id=MODEL_SET_ID,
        root_seed=20260805,
        seed_policy="per-object-derived-seed-v1",
        window_size=25,
        overlap=8,
        height=320,
        width=640,
        storage_dtype="float32",
        initial_metric_frame_prior_id=_digest(212),
        additional_metric_anchor_policy="none",
        max_gauge_rank=64,
        minimum_retained_gauge_trace=0.999,
    )


def _confirmation_authorization(
    stage0: Deform360Stage0SelectionV1,
    provider: Deform360VisualProviderLockV1,
) -> Deform360ConfirmationOpeningAuthorizationV1:
    return Deform360ConfirmationOpeningAuthorizationV1(
        execution_seal_id=_digest(220),
        calibration_bundle_id=_digest(221),
        confirmation_opening_token=_digest(222),
        stage0_selection_artifact_sha256=(stage0.selection_artifact_sha256),
        visual_provider_lock_id=provider.artifact_id,
        evidence_use_ledger_id=_digest(223),
        calibration_source_run_record_sha256=_digest(224),
        calibration_source_run_record_file_sha256=_digest(225),
        calibration_source_revision=BPT_REVISION,
        calibration_observability_report_id=_digest(226),
        calibration_observability_report_file_sha256=_digest(227),
        calibration_observability_physical_query_id=_digest(228),
        calibration_observability_implementation_revision=BPT_REVISION,
        source_artifacts={
            DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY: _digest(225),
            DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY: _digest(227),
        },
    )


def _unit_record(unit: Deform360CohortUnitV1) -> dict[str, object]:
    return {
        "object_id": unit.object_id,
        "stratum": unit.stratum,
        "episode_id": unit.episode_id,
        "metadata_path": unit.metadata_path,
        "metadata_sha256": unit.metadata_sha256,
    }


def _cohort_binding(stage0: Deform360Stage0SelectionV1) -> dict[str, object]:
    calibration = sorted(
        (_unit_record(unit) for unit in stage0.calibration_units),
        key=lambda item: str(item["object_id"]),
    )
    target = sorted(
        (_unit_record(unit) for unit in stage0.confirmation_units),
        key=lambda item: str(item["object_id"]),
    )
    descriptor: dict[str, object] = {
        "schema_name": PROB4D_COHORT_BINDING_SCHEMA,
        "schema_version": 1,
        "source_repository": BAYESIAN_PHYSTWIN_REPOSITORY,
        "source_revision": BPT_REVISION,
        "source_path": (
            "protocols/locks/"
            "deform360_official_hub_visuotactile_v1_selection.json"
        ),
        "selection_schema": (
            "bayesian-phystwin/deform360-official-hub-selection-v1"
        ),
        "selection_schema_version": 1,
        "selection_artifact_sha256": stage0.selection_artifact_sha256,
        "content_selection_sha256": stage0.content_selection_sha256,
        "selection_sha256": stage0.selection_sha256,
        "selection_implementation_revision": stage0.implementation_revision,
        "protocol_id": stage0.protocol_id,
        "protocol_sha256": stage0.protocol_sha256,
        "dataset_repository": "brownu/deform360",
        "dataset_requested_revision": "main",
        "dataset_resolved_revision": stage0.dataset_revision,
        "processing_repository": "lhy0807/deform360",
        "processing_revision": stage0.processing_revision,
        "statistical_unit": "physical-object",
        "calibration_units": calibration,
        "target_units": target,
        "calibration_group_ids": sorted(
            str(item["object_id"]) for item in calibration
        ),
        "target_group_ids": sorted(str(item["object_id"]) for item in target),
        "information_boundary": {
            "object_directory_names_opened": True,
            "object_metadata_json_opened": True,
            "camera_media_opened": False,
            "tactile_arrays_opened": False,
            "robot_arrays_opened": False,
            "geometry_annotations_opened": False,
            "target_outcomes_opened": False,
        },
        "replacement_allowed_after_payload_access": False,
        "claim_boundary": PROB4D_COHORT_BINDING_CLAIM_BOUNDARY,
    }
    return {**descriptor, "cohort_binding_id": content_id(descriptor)}


def _promotion_arms() -> list[dict[str, object]]:
    roles = (
        "cross_window_identity_marginalized",
        "framewise_explicit_joint_gauge",
        "persistent_explicit_joint_gauge",
        "physical_fallback",
        "rowwise_gauge_marginalized",
        "sensor_assisted",
        "visual_baseline",
    )
    return [
        {
            "arm_id": role,
            "role": role,
            "query_method_id": f"query-{role}",
            "provider_method_id": (
                None if role == "physical_fallback" else f"provider-{role}"
            ),
            "sensor_assisted": role == "sensor_assisted",
            "metadata": {},
        }
        for role in roles
    ]


def _promotion_lock(
    stage0: Deform360Stage0SelectionV1,
    provider: Deform360VisualProviderLockV1,
    confirmation: Deform360ConfirmationOpeningAuthorizationV1,
    cohort: dict[str, object],
) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "schema_name": PROB4D_PROMOTION_LOCK_SCHEMA,
        "schema_version": 1,
        "experiment_id": "deform360-prob4d-target-outcome-test-v1",
        "source_repository": PROB4D_REPOSITORY,
        "source_revision": PROB4D_REVISION,
        "bayesian_phystwin_repository": BAYESIAN_PHYSTWIN_REPOSITORY,
        "bayesian_phystwin_revision": BPT_REVISION,
        "motioncrafter_revision": provider.motioncrafter_revision,
        "model_set_id": provider.model_set_id,
        "prediction_run_spec_id": RUN_SPEC_ID,
        "provider_evaluation_manifest_sha256": _digest(230),
        "frozen_artifact_ids": {
            "provider_configuration": provider.artifact_id,
            "gauge_calibration": _digest(231),
            "point_calibration": _digest(232),
            "source_reliability_calibration": _digest(233),
            "material_identity_calibration": _digest(234),
            "selection_lock": _digest(235),
            "bayesian_guard_configuration": _digest(236),
            "cohort_binding": cohort["cohort_binding_id"],
        },
        "development_group_ids": ["development-a"],
        "calibration_group_ids": sorted(
            unit.object_id for unit in stage0.calibration_units
        ),
        "target_group_ids": sorted(
            unit.object_id for unit in stage0.confirmation_units
        ),
        "arms": _promotion_arms(),
        "provider_reference_arm_id": "visual_baseline",
        "primary_query_arm_id": "cross_window_identity_marginalized",
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 17,
        "minimum_target_group_count": 12,
        "query_superiority_margin_mm": 0.25,
        "harmful_update_margin_mm": 0.0,
        "maximum_harmful_accepted_updates": 0,
        "maximum_worst_group_regression_mm": 0.0,
        "maximum_technical_failures": 0,
        "minimum_mean_accepted_coverage": 0.9,
        "metadata": {
            BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY: (
                confirmation.authorization_id
            ),
            BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY: provider.artifact_id,
        },
        "claim_boundary": PROB4D_PROMOTION_LOCK_CLAIM_BOUNDARY,
    }
    return {**descriptor, "promotion_lock_id": content_id(descriptor)}


def _target_admission(
    stage0: Deform360Stage0SelectionV1,
    provider: Deform360VisualProviderLockV1,
    confirmation: Deform360ConfirmationOpeningAuthorizationV1,
    cohort: dict[str, object],
    lock: dict[str, object],
) -> dict[str, object]:
    entries = []
    for index, unit in enumerate(
        sorted(stage0.confirmation_units, key=lambda item: item.object_id)
    ):
        entries.append(
            {
                "group_id": unit.object_id,
                "episode_id": unit.episode_id,
                "stratum": unit.stratum,
                "sequence_id": f"{unit.object_id}-episode-{unit.episode_id}",
                "manifest_sha256": _digest(300 + index),
                "manifest_artifact_id": _digest(320 + index),
                "provider_run_id": _digest(340 + index),
                "causal_frame_stop": 12,
                "admitted_payloads": [
                    {
                        "payload_id": _digest(360 + index),
                        "window_id": f"window-{index}",
                        "output_frame_ids": [0, 1],
                        "source_frame_start": 0,
                        "source_frame_stop_exclusive": 8,
                        "dependence_group_ids": ["shared-model"],
                    }
                ],
            }
        )
    descriptor: dict[str, object] = {
        "schema_name": PROB4D_TARGET_ADMISSION_SCHEMA,
        "schema_version": 1,
        "promotion_lock_id": lock["promotion_lock_id"],
        "cohort_binding_id": cohort["cohort_binding_id"],
        "source_repository": lock["source_repository"],
        "source_revision": lock["source_revision"],
        "prediction_run_spec_id": lock["prediction_run_spec_id"],
        "provider_family": "motioncrafter",
        "provider_repository": provider.motioncrafter_repository,
        "provider_revision": provider.motioncrafter_revision,
        "model_set_id": provider.model_set_id,
        "loader_id": _digest(380),
        "coordinate_semantics": "sequence-local-sim3",
        "point_semantics": "dense-point-map",
        "flow_semantics": "absent",
        "ray_semantics": "absent",
        "source_dependency_semantics": (
            "per-output-exclusive-source-frame-interval-v1"
        ),
        "target_outcomes_used": False,
        "entries": entries,
        "metadata": {
            BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY: (
                confirmation.authorization_id
            ),
            BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY: provider.artifact_id,
            PREDICTION_PROVIDER_REVISION_METADATA_KEY: (
                provider.provider_revision
            ),
            CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY: True,
        },
        "claim_boundary": PROB4D_TARGET_ADMISSION_CLAIM_BOUNDARY,
    }
    return {
        **descriptor,
        "target_provider_admission_id": content_id(descriptor),
    }


def _source_artifacts() -> dict[str, str]:
    return {
        BPT_STAGE0_SOURCE_KEY: _digest(400),
        BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY: _digest(401),
        BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY: _digest(402),
        PROB4D_COHORT_BINDING_SOURCE_KEY: _digest(403),
        PROB4D_PROMOTION_LOCK_SOURCE_KEY: _digest(404),
        PROB4D_TARGET_ADMISSION_SOURCE_KEY: _digest(405),
    }


def _bundle() -> dict[str, object]:
    stage0 = _stage0()
    provider = _visual_provider()
    confirmation = _confirmation_authorization(stage0, provider)
    cohort = _cohort_binding(stage0)
    lock = _promotion_lock(stage0, provider, confirmation, cohort)
    admission = _target_admission(
        stage0,
        provider,
        confirmation,
        cohort,
        lock,
    )
    return {
        "stage0_selection": stage0,
        "visual_provider_lock": provider,
        "confirmation_opening_authorization": confirmation,
        "prob4d_cohort_binding": cohort,
        "prob4d_promotion_lock": lock,
        "target_provider_admission": admission,
        "source_artifacts": _source_artifacts(),
        "confirmation_provider_inputs_opened": True,
        "metadata": {"statistical_unit": "physical_object"},
    }


def _rehash(value: dict[str, object], id_field: str) -> dict[str, object]:
    result = copy.deepcopy(value)
    result.pop(id_field)
    return {**result, id_field: content_id(result)}


def test_builds_post_provider_pre_outcome_authorization() -> None:
    arguments = _bundle()
    authorization = build_deform360_prob4d_target_outcome_authorization(
        **arguments
    )

    assert authorization.status == "authorized-before-target-outcome-access"
    assert len(authorization.confirmation_group_ids) == 12
    assert authorization.confirmation_provider_inputs_opened is True
    assert authorization.target_outcomes_opened is False
    assert authorization.target_outcomes_used is False
    assert authorization.to_record()["schema"] == (
        DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SCHEMA
    )
    metadata = authorization.query_result_metadata()
    assert metadata["target_provider_admission_id"] == (
        authorization.target_provider_admission_id
    )
    assert metadata["deform360_prob4d_target_outcome_authorization_id"] == (
        authorization.authorization_id
    )
    verify_deform360_prob4d_target_outcome_authorization(
        authorization,
        **arguments,
    )


def test_authorization_round_trip_and_nonoverwrite(tmp_path: Path) -> None:
    authorization = build_deform360_prob4d_target_outcome_authorization(
        **_bundle()
    )
    path = tmp_path / "authorization.json"
    save_deform360_prob4d_target_outcome_authorization(authorization, path)
    assert load_deform360_prob4d_target_outcome_authorization(path) == authorization
    with pytest.raises(FileExistsError):
        save_deform360_prob4d_target_outcome_authorization(authorization, path)


def test_rejects_stage0_target_substitution_even_when_rehashed() -> None:
    arguments = _bundle()
    cohort = copy.deepcopy(arguments["prob4d_cohort_binding"])
    cohort["target_units"][0]["episode_id"] = 1
    arguments["prob4d_cohort_binding"] = _rehash(
        cohort,
        "cohort_binding_id",
    )

    with pytest.raises(ValueError, match="target units differ from Stage 0"):
        build_deform360_prob4d_target_outcome_authorization(**arguments)


def test_rejects_provider_configuration_drift_even_when_rehashed() -> None:
    arguments = _bundle()
    lock = copy.deepcopy(arguments["prob4d_promotion_lock"])
    lock["frozen_artifact_ids"]["provider_configuration"] = _digest(999)
    arguments["prob4d_promotion_lock"] = _rehash(lock, "promotion_lock_id")

    with pytest.raises(ValueError, match="another provider configuration"):
        build_deform360_prob4d_target_outcome_authorization(**arguments)


def test_rejects_target_outcome_use_even_when_rehashed() -> None:
    arguments = _bundle()
    admission = copy.deepcopy(arguments["target_provider_admission"])
    admission["target_outcomes_used"] = True
    arguments["target_provider_admission"] = _rehash(
        admission,
        "target_provider_admission_id",
    )

    with pytest.raises(ValueError, match="used target outcomes"):
        build_deform360_prob4d_target_outcome_authorization(**arguments)


def test_rejects_unbound_prediction_provider_revision() -> None:
    arguments = _bundle()
    admission = copy.deepcopy(arguments["target_provider_admission"])
    admission["metadata"][PREDICTION_PROVIDER_REVISION_METADATA_KEY] = "9" * 40
    arguments["target_provider_admission"] = _rehash(
        admission,
        "target_provider_admission_id",
    )

    with pytest.raises(ValueError, match="prediction-provider revision"):
        build_deform360_prob4d_target_outcome_authorization(**arguments)


def test_rejects_payload_lineage_crossing_the_frozen_cutoff() -> None:
    arguments = _bundle()
    admission = copy.deepcopy(arguments["target_provider_admission"])
    admission["entries"][0]["admitted_payloads"][0][
        "source_frame_stop_exclusive"
    ] = 13
    arguments["target_provider_admission"] = _rehash(
        admission,
        "target_provider_admission_id",
    )

    with pytest.raises(ValueError, match="crosses its causal cutoff"):
        build_deform360_prob4d_target_outcome_authorization(**arguments)


def test_requires_explicit_provider_input_opening_acknowledgement() -> None:
    arguments = _bundle()
    arguments["confirmation_provider_inputs_opened"] = False

    with pytest.raises(ValueError, match="provider inputs must be opened"):
        build_deform360_prob4d_target_outcome_authorization(**arguments)


def test_rejects_tampered_authorization_identity(tmp_path: Path) -> None:
    authorization = build_deform360_prob4d_target_outcome_authorization(
        **_bundle()
    )
    record = authorization.to_record()
    record["authorization_id"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(
        __import__("json").dumps(record, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="authorization_id does not match"):
        load_deform360_prob4d_target_outcome_authorization(path)


def test_loader_rejects_unknown_fields() -> None:
    authorization = build_deform360_prob4d_target_outcome_authorization(
        **_bundle()
    )
    record = authorization.to_record()
    record["unexpected"] = True

    with pytest.raises(ValueError, match="fields changed"):
        Deform360Prob4DTargetOutcomeAuthorizationV1.from_mapping(record)
