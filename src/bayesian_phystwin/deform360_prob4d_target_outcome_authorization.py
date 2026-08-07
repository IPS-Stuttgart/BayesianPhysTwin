"""Bind Prob4D target admission before Deform360 target-outcome access.

Stage 1 authorizes one opening of predictor-side confirmation inputs after the
calibration source and observability evidence are sealed.  Prob4D prediction
manifests can only be admitted after those visual inputs have been processed.
This module therefore defines a second, later information boundary: exact Prob4D
cohort, promotion-lock, and target-admission artifacts must agree with the frozen
BayesianPhysTwin cohort and Stage-1 authorization before target outcomes are
opened or scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_calibration_execution import Deform360Stage0SelectionV1
from .deform360_calibration_observability_binding import (
    Deform360ConfirmationOpeningAuthorizationV1,
)
from .deform360_visual_provider_lock import Deform360VisualProviderLockV1

PROB4D_REPOSITORY = "IPS-Stuttgart/Prob4D"
BAYESIAN_PHYSTWIN_REPOSITORY = "IPS-Stuttgart/BayesianPhysTwin"
DEFORM360_SELECTION_PATH = (
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)

PROB4D_COHORT_BINDING_SCHEMA = "prob4d.deform360-official-hub-cohort-binding"
PROB4D_COHORT_BINDING_VERSION = 1
PROB4D_PROMOTION_LOCK_SCHEMA = "prob4d.heldout-provider-promotion-lock"
PROB4D_PROMOTION_LOCK_VERSION = 1
PROB4D_TARGET_ADMISSION_SCHEMA = "prob4d.heldout-target-provider-admission"
PROB4D_TARGET_ADMISSION_VERSION = 1

PROB4D_COHORT_BINDING_CLAIM_BOUNDARY = (
    "This target-free artifact binds Prob4D promotion to BayesianPhysTwin's exact "
    "official-Hub Stage-0 Deform360 calibration and confirmation selection. It "
    "authenticates cohort custody and the names/metadata-only information boundary; "
    "it is not provider-competence, physical-benefit, uncertainty-calibration, safety, "
    "Causal4D, or state-of-the-art evidence."
)
PROB4D_PROMOTION_LOCK_CLAIM_BOUNDARY = (
    "This lock authenticates a target-free held-out Prob4D-to-BayesianPhysTwin "
    "promotion protocol. It freezes complete object/session splits, comparison "
    "arms, source/model/calibration identities, bootstrap settings, and decision "
    "margins before target outcomes are opened. It is not empirical evidence."
)
PROB4D_TARGET_ADMISSION_CLAIM_BOUNDARY = (
    "This artifact admits exact provider-manifest metadata for every frozen target "
    "object before target outcomes are evaluated. It binds source/model/loader "
    "semantics, causal cutoffs, manifest bytes, and admitted payload identities. "
    "It does not open prediction payloads or target outcomes and does not establish "
    "provider competence, calibration, BayesianPhysTwin benefit, Causal4D benefit, "
    "deployment safety, or state of the art."
)

BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY = (
    "bayesian_phystwin_confirmation_opening_authorization_id"
)
BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY = "bayesian_phystwin_visual_provider_lock_id"
PREDICTION_PROVIDER_REVISION_METADATA_KEY = "prediction_provider_revision"
CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY = "confirmation_provider_inputs_only"

DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SCHEMA = (
    "bayesian-phystwin.deform360-prob4d-target-outcome-authorization"
)
DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_VERSION = 1
DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SEMANTICS = (
    "confirmation-provider-inputs-opened-prob4d-admitted-target-outcomes-closed-v1"
)
DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_STATUS = (
    "authorized-before-target-outcome-access"
)
DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_CLAIM_BOUNDARY = (
    "Post-provider, pre-outcome information-order and artifact-binding evidence "
    "only. A valid authorization proves that the exact Prob4D cohort, promotion "
    "lock, and admitted provider manifests agree with the frozen BayesianPhysTwin "
    "confirmation cohort while target outcomes remain closed. It does not establish "
    "provider competence, physical-query accuracy, tactile benefit, uncertainty "
    "calibration, Causal4D benefit, deployment safety, or state of the art."
)

BPT_STAGE0_SOURCE_KEY = "sources/bayesian-phystwin/stage0-selection.json"
BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY = (
    "sources/bayesian-phystwin/visual-provider-lock.json"
)
BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY = (
    "sources/bayesian-phystwin/confirmation-opening-authorization.json"
)
PROB4D_COHORT_BINDING_SOURCE_KEY = "sources/prob4d/cohort-binding.json"
PROB4D_PROMOTION_LOCK_SOURCE_KEY = "sources/prob4d/promotion-lock.json"
PROB4D_TARGET_ADMISSION_SOURCE_KEY = "sources/prob4d/target-provider-admission.json"
_REQUIRED_SOURCE_KEYS = frozenset(
    {
        BPT_STAGE0_SOURCE_KEY,
        BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY,
        BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY,
        PROB4D_COHORT_BINDING_SOURCE_KEY,
        PROB4D_PROMOTION_LOCK_SOURCE_KEY,
        PROB4D_TARGET_ADMISSION_SOURCE_KEY,
    }
)

_COHORT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_repository",
        "source_revision",
        "source_path",
        "selection_schema",
        "selection_schema_version",
        "selection_artifact_sha256",
        "content_selection_sha256",
        "selection_sha256",
        "selection_implementation_revision",
        "protocol_id",
        "protocol_sha256",
        "dataset_repository",
        "dataset_requested_revision",
        "dataset_resolved_revision",
        "processing_repository",
        "processing_revision",
        "statistical_unit",
        "calibration_units",
        "target_units",
        "calibration_group_ids",
        "target_group_ids",
        "information_boundary",
        "replacement_allowed_after_payload_access",
        "claim_boundary",
        "cohort_binding_id",
    }
)
_COHORT_UNIT_FIELDS = frozenset(
    {"object_id", "stratum", "episode_id", "metadata_path", "metadata_sha256"}
)
_PROMOTION_LOCK_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "experiment_id",
        "source_repository",
        "source_revision",
        "bayesian_phystwin_repository",
        "bayesian_phystwin_revision",
        "motioncrafter_revision",
        "model_set_id",
        "prediction_run_spec_id",
        "provider_evaluation_manifest_sha256",
        "frozen_artifact_ids",
        "development_group_ids",
        "calibration_group_ids",
        "target_group_ids",
        "arms",
        "provider_reference_arm_id",
        "primary_query_arm_id",
        "bootstrap_resamples",
        "bootstrap_seed",
        "minimum_target_group_count",
        "query_superiority_margin_mm",
        "harmful_update_margin_mm",
        "maximum_harmful_accepted_updates",
        "maximum_worst_group_regression_mm",
        "maximum_technical_failures",
        "minimum_mean_accepted_coverage",
        "metadata",
        "claim_boundary",
        "promotion_lock_id",
    }
)
_PROMOTION_ARM_FIELDS = frozenset(
    {
        "arm_id",
        "role",
        "query_method_id",
        "provider_method_id",
        "sensor_assisted",
        "metadata",
    }
)
_REQUIRED_PROMOTION_ROLES = frozenset(
    {
        "physical_fallback",
        "visual_baseline",
        "rowwise_gauge_marginalized",
        "framewise_explicit_joint_gauge",
        "persistent_explicit_joint_gauge",
        "cross_window_identity_marginalized",
        "sensor_assisted",
    }
)
_ADMISSION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "promotion_lock_id",
        "cohort_binding_id",
        "source_repository",
        "source_revision",
        "prediction_run_spec_id",
        "provider_family",
        "provider_repository",
        "provider_revision",
        "model_set_id",
        "loader_id",
        "coordinate_semantics",
        "point_semantics",
        "flow_semantics",
        "ray_semantics",
        "source_dependency_semantics",
        "target_outcomes_used",
        "entries",
        "metadata",
        "claim_boundary",
        "target_provider_admission_id",
    }
)
_ADMISSION_ENTRY_FIELDS = frozenset(
    {
        "group_id",
        "episode_id",
        "stratum",
        "sequence_id",
        "manifest_sha256",
        "manifest_artifact_id",
        "provider_run_id",
        "causal_frame_stop",
        "admitted_payloads",
    }
)
_ADMISSION_PAYLOAD_FIELDS = frozenset(
    {
        "payload_id",
        "window_id",
        "output_frame_ids",
        "source_frame_start",
        "source_frame_stop_exclusive",
        "dependence_group_ids",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "authorization_id",
        "status",
        "confirmation_opening_authorization_id",
        "confirmation_opening_token",
        "stage0_selection_artifact_sha256",
        "visual_provider_lock_id",
        "prob4d_cohort_binding_id",
        "prob4d_promotion_lock_id",
        "target_provider_admission_id",
        "prob4d_source_revision",
        "bayesian_phystwin_revision",
        "prediction_provider_revision",
        "motioncrafter_revision",
        "model_set_id",
        "prediction_run_spec_id",
        "confirmation_group_ids",
        "target_manifest_sha256_by_group",
        "target_manifest_artifact_id_by_group",
        "target_provider_run_id_by_group",
        "target_payload_ids_by_group",
        "source_artifacts",
        "confirmation_provider_inputs_opened",
        "target_outcomes_opened",
        "target_outcomes_used",
        "metadata",
        "claim_boundary",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_array(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return value


def _canonical_json_strings(value: object, *, name: str) -> tuple[str, ...]:
    raw = _json_array(value, name=name)
    items = tuple(nonempty_string(item, name=f"{name}[{index}]") for index, item in enumerate(raw))
    if items != tuple(sorted(items)) or len(set(items)) != len(items):
        raise ValueError(f"{name} must be sorted and unique")
    return items


def _validated_content_id(
    value: Mapping[str, Any],
    *,
    id_field: str,
    name: str,
) -> str:
    supplied = sha256_digest(value[id_field], name=f"{name} {id_field}")
    descriptor = dict(value)
    descriptor.pop(id_field)
    if content_id(descriptor) != supplied:
        raise ValueError(f"{name} {id_field} does not match content")
    return supplied


def _unit_record(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    require_exact_fields(value, expected=_COHORT_UNIT_FIELDS, name=name)
    object_id = nonempty_string(value["object_id"], name=f"{name}.object_id")
    stratum = nonempty_string(value["stratum"], name=f"{name}.stratum")
    if stratum not in {"sheet", "volumetric"}:
        raise ValueError(f"{name}.stratum must be sheet or volumetric")
    episode_id = genuine_integer(value["episode_id"], name=f"{name}.episode_id", minimum=0)
    metadata_path = nonempty_string(value["metadata_path"], name=f"{name}.metadata_path")
    if metadata_path != f"raw/{object_id}/metadata.json":
        raise ValueError(f"{name}.metadata_path changed")
    return {
        "object_id": object_id,
        "stratum": stratum,
        "episode_id": episode_id,
        "metadata_path": metadata_path,
        "metadata_sha256": sha256_digest(
            value["metadata_sha256"],
            name=f"{name}.metadata_sha256",
        ),
    }


def _stage0_records(
    selection: Deform360Stage0SelectionV1,
    *,
    split: str,
) -> tuple[dict[str, object], ...]:
    units = (
        selection.calibration_units
        if split == "calibration"
        else selection.confirmation_units
    )
    return tuple(
        sorted(
            (
                {
                    "object_id": unit.object_id,
                    "stratum": unit.stratum,
                    "episode_id": unit.episode_id,
                    "metadata_path": unit.metadata_path,
                    "metadata_sha256": unit.metadata_sha256,
                }
                for unit in units
            ),
            key=lambda item: str(item["object_id"]),
        )
    )


def _validated_units(value: object, *, name: str) -> tuple[dict[str, object], ...]:
    raw = _json_array(value, name=name)
    units = tuple(_unit_record(item, name=f"{name}[{index}]") for index, item in enumerate(raw))
    if units != tuple(sorted(units, key=lambda item: str(item["object_id"]))):
        raise ValueError(f"{name} must be sorted by object_id")
    object_ids = tuple(str(item["object_id"]) for item in units)
    if len(set(object_ids)) != len(object_ids):
        raise ValueError(f"{name} repeats an object")
    return units


def _validate_prob4d_cohort_binding(
    value: Mapping[str, Any],
    *,
    stage0_selection: Deform360Stage0SelectionV1,
) -> Mapping[str, Any]:
    require_exact_fields(value, expected=_COHORT_FIELDS, name="Prob4D cohort binding")
    if value["schema_name"] != PROB4D_COHORT_BINDING_SCHEMA:
        raise ValueError("Prob4D cohort binding schema changed")
    if (
        genuine_integer(value["schema_version"], name="cohort schema_version", minimum=1)
        != PROB4D_COHORT_BINDING_VERSION
    ):
        raise ValueError("Prob4D cohort binding version changed")
    if value["claim_boundary"] != PROB4D_COHORT_BINDING_CLAIM_BOUNDARY:
        raise ValueError("Prob4D cohort binding claim boundary changed")
    if value["source_repository"] != BAYESIAN_PHYSTWIN_REPOSITORY:
        raise ValueError("Prob4D cohort binding uses another source repository")
    exact_revision(value["source_revision"], name="cohort source_revision")
    if value["source_path"] != DEFORM360_SELECTION_PATH:
        raise ValueError("Prob4D cohort binding uses another Stage-0 path")
    if value["selection_schema"] != "bayesian-phystwin/deform360-official-hub-selection-v1":
        raise ValueError("Prob4D cohort binding selection schema changed")
    if genuine_integer(
        value["selection_schema_version"],
        name="selection_schema_version",
        minimum=1,
    ) != 1:
        raise ValueError("Prob4D cohort binding selection version changed")
    comparisons = {
        "selection_artifact_sha256": stage0_selection.selection_artifact_sha256,
        "content_selection_sha256": stage0_selection.content_selection_sha256,
        "selection_sha256": stage0_selection.selection_sha256,
        "selection_implementation_revision": stage0_selection.implementation_revision,
        "protocol_id": stage0_selection.protocol_id,
        "protocol_sha256": stage0_selection.protocol_sha256,
        "dataset_resolved_revision": stage0_selection.dataset_revision,
        "processing_revision": stage0_selection.processing_revision,
    }
    for key, expected in comparisons.items():
        if value[key] != expected:
            raise ValueError(f"Prob4D cohort binding changed {key}")
    if value["dataset_repository"] != "brownu/deform360":
        raise ValueError("Prob4D cohort binding dataset repository changed")
    if value["processing_repository"] != "lhy0807/deform360":
        raise ValueError("Prob4D cohort binding processing repository changed")
    nonempty_string(value["dataset_requested_revision"], name="dataset_requested_revision")
    if value["statistical_unit"] != "physical-object":
        raise ValueError("Prob4D cohort binding statistical unit changed")
    if genuine_boolean(
        value["replacement_allowed_after_payload_access"],
        name="replacement_allowed_after_payload_access",
    ):
        raise ValueError("Prob4D cohort binding permits cohort replacement")
    boundary = value["information_boundary"]
    if not isinstance(boundary, Mapping):
        raise ValueError("Prob4D cohort information boundary must be a JSON object")
    expected_boundary = {
        "object_directory_names_opened": True,
        "object_metadata_json_opened": True,
        "camera_media_opened": False,
        "tactile_arrays_opened": False,
        "robot_arrays_opened": False,
        "geometry_annotations_opened": False,
        "target_outcomes_opened": False,
    }
    if set(boundary) != set(expected_boundary):
        raise ValueError("Prob4D cohort information-boundary fields changed")
    for key, expected in expected_boundary.items():
        if genuine_boolean(boundary[key], name=f"information_boundary.{key}") is not expected:
            raise ValueError(f"Prob4D cohort information boundary changed: {key}")

    calibration_units = _validated_units(value["calibration_units"], name="calibration_units")
    target_units = _validated_units(value["target_units"], name="target_units")
    if calibration_units != _stage0_records(stage0_selection, split="calibration"):
        raise ValueError("Prob4D cohort calibration units differ from Stage 0")
    if target_units != _stage0_records(stage0_selection, split="target"):
        raise ValueError("Prob4D cohort target units differ from Stage 0")
    expected_calibration_ids = tuple(str(item["object_id"]) for item in calibration_units)
    expected_target_ids = tuple(str(item["object_id"]) for item in target_units)
    if _canonical_json_strings(
        value["calibration_group_ids"],
        name="calibration_group_ids",
    ) != expected_calibration_ids:
        raise ValueError("Prob4D cohort calibration group IDs changed")
    if _canonical_json_strings(value["target_group_ids"], name="target_group_ids") != expected_target_ids:
        raise ValueError("Prob4D cohort target group IDs changed")
    _validated_content_id(
        value,
        id_field="cohort_binding_id",
        name="Prob4D cohort binding",
    )
    return value


def _validate_promotion_arms(value: object) -> None:
    raw = _json_array(value, name="promotion arms")
    if not raw:
        raise ValueError("promotion arms must not be empty")
    arm_ids: list[str] = []
    roles: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"promotion arms[{index}] must be a JSON object")
        require_exact_fields(item, expected=_PROMOTION_ARM_FIELDS, name=f"promotion arms[{index}]")
        arm_id = nonempty_string(item["arm_id"], name=f"promotion arms[{index}].arm_id")
        role = nonempty_string(item["role"], name=f"promotion arms[{index}].role")
        nonempty_string(
            item["query_method_id"],
            name=f"promotion arms[{index}].query_method_id",
        )
        provider_method = item["provider_method_id"]
        sensor_assisted = genuine_boolean(
            item["sensor_assisted"],
            name=f"promotion arms[{index}].sensor_assisted",
        )
        if not isinstance(item["metadata"], Mapping):
            raise ValueError(f"promotion arms[{index}].metadata must be a JSON object")
        if role == "physical_fallback":
            if provider_method is not None or sensor_assisted:
                raise ValueError("physical fallback arm semantics changed")
        else:
            nonempty_string(
                provider_method,
                name=f"promotion arms[{index}].provider_method_id",
            )
        if (role == "sensor_assisted") is not sensor_assisted:
            raise ValueError("sensor-assisted promotion arm semantics changed")
        arm_ids.append(arm_id)
        roles.append(role)
    if tuple(arm_ids) != tuple(sorted(arm_ids)) or len(set(arm_ids)) != len(arm_ids):
        raise ValueError("promotion arms must be sorted by unique arm_id")
    missing = sorted(_REQUIRED_PROMOTION_ROLES - set(roles))
    if missing:
        raise ValueError(f"promotion arms are missing required roles: {missing}")
    for role in _REQUIRED_PROMOTION_ROLES:
        if roles.count(role) != 1:
            raise ValueError(f"promotion role {role!r} must occur exactly once")


def _required_metadata(
    value: object,
    *,
    confirmation_authorization: Deform360ConfirmationOpeningAuthorizationV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    include_provider_revision: bool,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Prob4D metadata must be a JSON object")
    if value.get(BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY) != (
        confirmation_authorization.authorization_id
    ):
        raise ValueError("Prob4D metadata binds another confirmation authorization")
    if value.get(BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY) != visual_provider_lock.artifact_id:
        raise ValueError("Prob4D metadata binds another visual-provider lock")
    if include_provider_revision:
        if value.get(PREDICTION_PROVIDER_REVISION_METADATA_KEY) != (
            visual_provider_lock.provider_revision
        ):
            raise ValueError("Prob4D admission binds another prediction-provider revision")
        if genuine_boolean(
            value.get(CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY),
            name=CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY,
        ) is not True:
            raise ValueError("Prob4D admission does not declare provider-input-only access")
    return value


def _validate_prob4d_promotion_lock(
    value: Mapping[str, Any],
    *,
    cohort_binding: Mapping[str, Any],
    stage0_selection: Deform360Stage0SelectionV1,
    confirmation_authorization: Deform360ConfirmationOpeningAuthorizationV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
) -> Mapping[str, Any]:
    require_exact_fields(value, expected=_PROMOTION_LOCK_FIELDS, name="Prob4D promotion lock")
    if value["schema_name"] != PROB4D_PROMOTION_LOCK_SCHEMA:
        raise ValueError("Prob4D promotion lock schema changed")
    if genuine_integer(
        value["schema_version"],
        name="promotion lock schema_version",
        minimum=1,
    ) != PROB4D_PROMOTION_LOCK_VERSION:
        raise ValueError("Prob4D promotion lock version changed")
    if value["claim_boundary"] != PROB4D_PROMOTION_LOCK_CLAIM_BOUNDARY:
        raise ValueError("Prob4D promotion lock claim boundary changed")
    if value["source_repository"] != PROB4D_REPOSITORY:
        raise ValueError("Prob4D promotion lock source repository changed")
    exact_revision(value["source_revision"], name="Prob4D source_revision")
    if value["bayesian_phystwin_repository"] != BAYESIAN_PHYSTWIN_REPOSITORY:
        raise ValueError("Prob4D promotion lock BayesianPhysTwin repository changed")
    exact_revision(
        value["bayesian_phystwin_revision"],
        name="BayesianPhysTwin promotion revision",
    )
    if value["motioncrafter_revision"] != visual_provider_lock.motioncrafter_revision:
        raise ValueError("Prob4D promotion lock MotionCrafter revision changed")
    if value["model_set_id"] != visual_provider_lock.model_set_id:
        raise ValueError("Prob4D promotion lock model-set identity changed")
    sha256_digest(value["prediction_run_spec_id"], name="prediction_run_spec_id")
    sha256_digest(
        value["provider_evaluation_manifest_sha256"],
        name="provider_evaluation_manifest_sha256",
    )
    frozen = value["frozen_artifact_ids"]
    if not isinstance(frozen, Mapping):
        raise ValueError("Prob4D frozen_artifact_ids must be a JSON object")
    for key, item in frozen.items():
        nonempty_string(key, name="frozen_artifact_ids key")
        sha256_digest(item, name=f"frozen_artifact_ids[{key!r}]")
    if frozen.get("cohort_binding") != cohort_binding["cohort_binding_id"]:
        raise ValueError("Prob4D promotion lock binds another cohort")
    if frozen.get("provider_configuration") != visual_provider_lock.artifact_id:
        raise ValueError("Prob4D promotion lock binds another provider configuration")
    expected_calibration = tuple(
        sorted(unit.object_id for unit in stage0_selection.calibration_units)
    )
    expected_target = tuple(
        sorted(unit.object_id for unit in stage0_selection.confirmation_units)
    )
    if _canonical_json_strings(
        value["calibration_group_ids"],
        name="promotion calibration_group_ids",
    ) != expected_calibration:
        raise ValueError("Prob4D promotion lock calibration groups changed")
    if _canonical_json_strings(
        value["target_group_ids"],
        name="promotion target_group_ids",
    ) != expected_target:
        raise ValueError("Prob4D promotion lock target groups changed")
    _canonical_json_strings(
        value["development_group_ids"],
        name="promotion development_group_ids",
    )
    _validate_promotion_arms(value["arms"])
    nonempty_string(value["provider_reference_arm_id"], name="provider_reference_arm_id")
    nonempty_string(value["primary_query_arm_id"], name="primary_query_arm_id")
    genuine_integer(value["bootstrap_resamples"], name="bootstrap_resamples", minimum=100)
    genuine_integer(value["bootstrap_seed"], name="bootstrap_seed", minimum=0)
    minimum_groups = genuine_integer(
        value["minimum_target_group_count"],
        name="minimum_target_group_count",
        minimum=1,
    )
    if minimum_groups != len(expected_target):
        raise ValueError("Prob4D promotion lock does not require the complete target cohort")
    for name in (
        "query_superiority_margin_mm",
        "harmful_update_margin_mm",
        "maximum_worst_group_regression_mm",
    ):
        observed = value[name]
        if type(observed) not in {int, float} or float(observed) < 0.0:
            raise ValueError(f"{name} must be a nonnegative finite number")
    for name in (
        "maximum_harmful_accepted_updates",
        "maximum_technical_failures",
    ):
        genuine_integer(value[name], name=name, minimum=0)
    coverage = value["minimum_mean_accepted_coverage"]
    if coverage is not None and (
        type(coverage) not in {int, float} or not 0.0 <= float(coverage) <= 1.0
    ):
        raise ValueError("minimum_mean_accepted_coverage must lie in [0, 1]")
    _required_metadata(
        value["metadata"],
        confirmation_authorization=confirmation_authorization,
        visual_provider_lock=visual_provider_lock,
        include_provider_revision=False,
    )
    _validated_content_id(
        value,
        id_field="promotion_lock_id",
        name="Prob4D promotion lock",
    )
    return value


def _validate_admitted_payload(value: object, *, cutoff: int, name: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    require_exact_fields(value, expected=_ADMISSION_PAYLOAD_FIELDS, name=name)
    payload_id = sha256_digest(value["payload_id"], name=f"{name}.payload_id")
    nonempty_string(value["window_id"], name=f"{name}.window_id")
    output_frames = _json_array(value["output_frame_ids"], name=f"{name}.output_frame_ids")
    normalized_frames = tuple(
        genuine_integer(frame, name=f"{name}.output_frame_ids[{index}]", minimum=0)
        for index, frame in enumerate(output_frames)
    )
    if not normalized_frames or normalized_frames != tuple(sorted(set(normalized_frames))):
        raise ValueError(f"{name}.output_frame_ids must be nonempty, sorted, and unique")
    start = genuine_integer(
        value["source_frame_start"],
        name=f"{name}.source_frame_start",
        minimum=0,
    )
    stop = genuine_integer(
        value["source_frame_stop_exclusive"],
        name=f"{name}.source_frame_stop_exclusive",
        minimum=1,
    )
    if stop <= start or stop > cutoff:
        raise ValueError(f"{name} crosses its causal cutoff")
    dependence = _canonical_json_strings(
        value["dependence_group_ids"],
        name=f"{name}.dependence_group_ids",
    )
    if not dependence:
        raise ValueError(f"{name}.dependence_group_ids must not be empty")
    return payload_id


def _validate_target_entries(
    value: object,
    *,
    stage0_selection: Deform360Stage0SelectionV1,
) -> tuple[dict[str, object], ...]:
    raw = _json_array(value, name="target admission entries")
    expected = {
        unit.object_id: unit for unit in stage0_selection.confirmation_units
    }
    summaries: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        name = f"target admission entries[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(item, expected=_ADMISSION_ENTRY_FIELDS, name=name)
        group_id = nonempty_string(item["group_id"], name=f"{name}.group_id")
        if group_id not in expected:
            raise ValueError(f"{name} identifies an unregistered target group")
        unit = expected[group_id]
        episode_id = genuine_integer(item["episode_id"], name=f"{name}.episode_id", minimum=0)
        if episode_id != unit.episode_id or item["stratum"] != unit.stratum:
            raise ValueError(f"{name} object/episode/stratum differs from Stage 0")
        sequence_id = nonempty_string(item["sequence_id"], name=f"{name}.sequence_id")
        manifest_sha256 = sha256_digest(
            item["manifest_sha256"],
            name=f"{name}.manifest_sha256",
        )
        manifest_artifact_id = sha256_digest(
            item["manifest_artifact_id"],
            name=f"{name}.manifest_artifact_id",
        )
        provider_run_id = sha256_digest(
            item["provider_run_id"],
            name=f"{name}.provider_run_id",
        )
        cutoff = genuine_integer(
            item["causal_frame_stop"],
            name=f"{name}.causal_frame_stop",
            minimum=1,
        )
        payloads = _json_array(item["admitted_payloads"], name=f"{name}.admitted_payloads")
        if not payloads:
            raise ValueError(f"{name}.admitted_payloads must not be empty")
        payload_ids = tuple(
            _validate_admitted_payload(
                payload,
                cutoff=cutoff,
                name=f"{name}.admitted_payloads[{payload_index}]",
            )
            for payload_index, payload in enumerate(payloads)
        )
        if payload_ids != tuple(sorted(set(payload_ids))):
            raise ValueError(f"{name}.admitted_payloads must be sorted by unique payload_id")
        summaries.append(
            {
                "group_id": group_id,
                "sequence_id": sequence_id,
                "manifest_sha256": manifest_sha256,
                "manifest_artifact_id": manifest_artifact_id,
                "provider_run_id": provider_run_id,
                "payload_ids": payload_ids,
            }
        )
    summaries.sort(key=lambda item: str(item["group_id"]))
    if tuple(str(item["group_id"]) for item in summaries) != tuple(sorted(expected)):
        raise ValueError("target admission does not cover the exact confirmation cohort")
    return tuple(summaries)


def _validate_target_admission(
    value: Mapping[str, Any],
    *,
    promotion_lock: Mapping[str, Any],
    cohort_binding: Mapping[str, Any],
    stage0_selection: Deform360Stage0SelectionV1,
    confirmation_authorization: Deform360ConfirmationOpeningAuthorizationV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
) -> tuple[Mapping[str, Any], tuple[dict[str, object], ...]]:
    require_exact_fields(value, expected=_ADMISSION_FIELDS, name="Prob4D target admission")
    if value["schema_name"] != PROB4D_TARGET_ADMISSION_SCHEMA:
        raise ValueError("Prob4D target admission schema changed")
    if genuine_integer(
        value["schema_version"],
        name="target admission schema_version",
        minimum=1,
    ) != PROB4D_TARGET_ADMISSION_VERSION:
        raise ValueError("Prob4D target admission version changed")
    if value["claim_boundary"] != PROB4D_TARGET_ADMISSION_CLAIM_BOUNDARY:
        raise ValueError("Prob4D target admission claim boundary changed")
    comparisons = {
        "promotion_lock_id": promotion_lock["promotion_lock_id"],
        "cohort_binding_id": cohort_binding["cohort_binding_id"],
        "source_repository": promotion_lock["source_repository"],
        "source_revision": promotion_lock["source_revision"],
        "prediction_run_spec_id": promotion_lock["prediction_run_spec_id"],
        "provider_repository": visual_provider_lock.motioncrafter_repository,
        "provider_revision": visual_provider_lock.motioncrafter_revision,
        "model_set_id": visual_provider_lock.model_set_id,
    }
    for key, expected in comparisons.items():
        if value[key] != expected:
            raise ValueError(f"Prob4D target admission changed {key}")
    nonempty_string(value["provider_family"], name="provider_family")
    sha256_digest(value["loader_id"], name="loader_id")
    for key in (
        "coordinate_semantics",
        "point_semantics",
        "flow_semantics",
        "ray_semantics",
        "source_dependency_semantics",
    ):
        nonempty_string(value[key], name=key)
    if value["source_dependency_semantics"] != (
        "per-output-exclusive-source-frame-interval-v1"
    ):
        raise ValueError("Prob4D target admission source-dependency semantics changed")
    if genuine_boolean(value["target_outcomes_used"], name="target_outcomes_used"):
        raise ValueError("Prob4D target admission used target outcomes")
    _required_metadata(
        value["metadata"],
        confirmation_authorization=confirmation_authorization,
        visual_provider_lock=visual_provider_lock,
        include_provider_revision=True,
    )
    summaries = _validate_target_entries(
        value["entries"],
        stage0_selection=stage0_selection,
    )
    _validated_content_id(
        value,
        id_field="target_provider_admission_id",
        name="Prob4D target admission",
    )
    return value, summaries


def _exact_source_artifacts(value: Mapping[str, str]) -> Mapping[str, Any]:
    sources = source_artifact_mapping(value, name="source_artifacts")
    missing = sorted(_REQUIRED_SOURCE_KEYS - set(sources))
    extra = sorted(set(sources) - _REQUIRED_SOURCE_KEYS)
    if missing or extra:
        raise ValueError(
            f"source_artifacts changed: missing={missing}, extra={extra}"
        )
    return sources


def _digest_by_group(
    value: Mapping[str, str],
    *,
    name: str,
    expected_groups: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if tuple(sorted(value)) != expected_groups:
        raise ValueError(f"{name} group IDs changed")
    return frozen_finite_json_mapping(
        {
            group_id: sha256_digest(value[group_id], name=f"{name}[{group_id!r}]")
            for group_id in expected_groups
        },
        name=name,
    )


def _payload_ids_by_group(
    value: Mapping[str, Sequence[str]],
    *,
    expected_groups: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != expected_groups:
        raise ValueError("target_payload_ids_by_group group IDs changed")
    normalized: dict[str, list[str]] = {}
    for group_id in expected_groups:
        raw = value[group_id]
        if isinstance(raw, (str, bytes)):
            raise ValueError("target payload IDs must be a sequence")
        payloads = tuple(
            sha256_digest(item, name=f"target payload ID for {group_id}")
            for item in raw
        )
        if not payloads or payloads != tuple(sorted(set(payloads))):
            raise ValueError("target payload IDs must be nonempty, sorted, and unique")
        normalized[group_id] = list(payloads)
    return frozen_finite_json_mapping(normalized, name="target_payload_ids_by_group")


@dataclass(frozen=True)
class Deform360Prob4DTargetOutcomeAuthorizationV1:
    """One exact post-provider authorization to open target outcomes."""

    confirmation_opening_authorization_id: str
    confirmation_opening_token: str
    stage0_selection_artifact_sha256: str
    visual_provider_lock_id: str
    prob4d_cohort_binding_id: str
    prob4d_promotion_lock_id: str
    target_provider_admission_id: str
    prob4d_source_revision: str
    bayesian_phystwin_revision: str
    prediction_provider_revision: str
    motioncrafter_revision: str
    model_set_id: str
    prediction_run_spec_id: str
    confirmation_group_ids: Sequence[str]
    target_manifest_sha256_by_group: Mapping[str, str]
    target_manifest_artifact_id_by_group: Mapping[str, str]
    target_provider_run_id_by_group: Mapping[str, str]
    target_payload_ids_by_group: Mapping[str, Sequence[str]]
    source_artifacts: Mapping[str, str]
    confirmation_provider_inputs_opened: bool
    target_outcomes_opened: bool = False
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_STATUS
    authorization_id: str | None = None

    def __post_init__(self) -> None:
        digests = {
            name: sha256_digest(value, name=name)
            for name, value in (
                (
                    "confirmation_opening_authorization_id",
                    self.confirmation_opening_authorization_id,
                ),
                ("confirmation_opening_token", self.confirmation_opening_token),
                (
                    "stage0_selection_artifact_sha256",
                    self.stage0_selection_artifact_sha256,
                ),
                ("visual_provider_lock_id", self.visual_provider_lock_id),
                ("prob4d_cohort_binding_id", self.prob4d_cohort_binding_id),
                ("prob4d_promotion_lock_id", self.prob4d_promotion_lock_id),
                (
                    "target_provider_admission_id",
                    self.target_provider_admission_id,
                ),
                ("model_set_id", self.model_set_id),
                ("prediction_run_spec_id", self.prediction_run_spec_id),
            )
        }
        revisions = {
            name: exact_revision(value, name=name)
            for name, value in (
                ("prob4d_source_revision", self.prob4d_source_revision),
                (
                    "bayesian_phystwin_revision",
                    self.bayesian_phystwin_revision,
                ),
                (
                    "prediction_provider_revision",
                    self.prediction_provider_revision,
                ),
                ("motioncrafter_revision", self.motioncrafter_revision),
            )
        }
        groups = canonical_sorted_strings(
            self.confirmation_group_ids,
            name="confirmation_group_ids",
        )
        if not groups:
            raise ValueError("confirmation_group_ids must not be empty")
        manifest_sha = _digest_by_group(
            self.target_manifest_sha256_by_group,
            name="target_manifest_sha256_by_group",
            expected_groups=groups,
        )
        manifest_ids = _digest_by_group(
            self.target_manifest_artifact_id_by_group,
            name="target_manifest_artifact_id_by_group",
            expected_groups=groups,
        )
        run_ids = _digest_by_group(
            self.target_provider_run_id_by_group,
            name="target_provider_run_id_by_group",
            expected_groups=groups,
        )
        payload_ids = _payload_ids_by_group(
            self.target_payload_ids_by_group,
            expected_groups=groups,
        )
        sources = _exact_source_artifacts(self.source_artifacts)
        inputs_opened = genuine_boolean(
            self.confirmation_provider_inputs_opened,
            name="confirmation_provider_inputs_opened",
        )
        outcomes_opened = genuine_boolean(
            self.target_outcomes_opened,
            name="target_outcomes_opened",
        )
        outcomes_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not inputs_opened:
            raise ValueError("provider inputs must be opened before target admission exists")
        if outcomes_opened or outcomes_used:
            raise ValueError("target outcomes must remain closed during authorization")
        status = nonempty_string(self.status, name="status")
        if status != DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_STATUS:
            raise ValueError("target-outcome authorization status changed")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="target-outcome authorization metadata",
        )
        for name, value in {**digests, **revisions}.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "confirmation_group_ids", groups)
        object.__setattr__(self, "target_manifest_sha256_by_group", manifest_sha)
        object.__setattr__(
            self,
            "target_manifest_artifact_id_by_group",
            manifest_ids,
        )
        object.__setattr__(self, "target_provider_run_id_by_group", run_ids)
        object.__setattr__(self, "target_payload_ids_by_group", payload_ids)
        object.__setattr__(self, "source_artifacts", sources)
        object.__setattr__(self, "confirmation_provider_inputs_opened", inputs_opened)
        object.__setattr__(self, "target_outcomes_opened", outcomes_opened)
        object.__setattr__(self, "target_outcomes_used", outcomes_used)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.identity_record())
        if self.authorization_id is not None:
            supplied_id = sha256_digest(self.authorization_id, name="authorization_id")
            if supplied_id != expected_id:
                raise ValueError("target-outcome authorization_id does not match content")
        object.__setattr__(self, "authorization_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SCHEMA,
            "schema_version": DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_VERSION,
            "semantics": DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SEMANTICS,
            "status": self.status,
            "confirmation_opening_authorization_id": (
                self.confirmation_opening_authorization_id
            ),
            "confirmation_opening_token": self.confirmation_opening_token,
            "stage0_selection_artifact_sha256": (
                self.stage0_selection_artifact_sha256
            ),
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "prob4d_cohort_binding_id": self.prob4d_cohort_binding_id,
            "prob4d_promotion_lock_id": self.prob4d_promotion_lock_id,
            "target_provider_admission_id": self.target_provider_admission_id,
            "prob4d_source_revision": self.prob4d_source_revision,
            "bayesian_phystwin_revision": self.bayesian_phystwin_revision,
            "prediction_provider_revision": self.prediction_provider_revision,
            "motioncrafter_revision": self.motioncrafter_revision,
            "model_set_id": self.model_set_id,
            "prediction_run_spec_id": self.prediction_run_spec_id,
            "confirmation_group_ids": list(self.confirmation_group_ids),
            "target_manifest_sha256_by_group": plain_json(
                self.target_manifest_sha256_by_group
            ),
            "target_manifest_artifact_id_by_group": plain_json(
                self.target_manifest_artifact_id_by_group
            ),
            "target_provider_run_id_by_group": plain_json(
                self.target_provider_run_id_by_group
            ),
            "target_payload_ids_by_group": plain_json(
                self.target_payload_ids_by_group
            ),
            "source_artifacts": plain_json(self.source_artifacts),
            "confirmation_provider_inputs_opened": (
                self.confirmation_provider_inputs_opened
            ),
            "target_outcomes_opened": self.target_outcomes_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": (
                DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_CLAIM_BOUNDARY
            ),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "authorization_id": self.authorization_id}

    def query_result_metadata(self) -> dict[str, str]:
        """Return the exact metadata required on BayesianPhysTwin query results."""

        assert self.authorization_id is not None
        return {
            "target_provider_admission_id": self.target_provider_admission_id,
            "deform360_confirmation_opening_authorization_id": (
                self.confirmation_opening_authorization_id
            ),
            "deform360_prob4d_target_outcome_authorization_id": self.authorization_id,
            "prob4d_promotion_lock_id": self.prob4d_promotion_lock_id,
            "prob4d_cohort_binding_id": self.prob4d_cohort_binding_id,
        }

    def summary(self) -> dict[str, object]:
        return {
            "authorization_id": self.authorization_id,
            "status": self.status,
            "confirmation_opening_authorization_id": (
                self.confirmation_opening_authorization_id
            ),
            "prob4d_promotion_lock_id": self.prob4d_promotion_lock_id,
            "target_provider_admission_id": self.target_provider_admission_id,
            "confirmation_group_count": len(self.confirmation_group_ids),
            "confirmation_provider_inputs_opened": True,
            "target_outcomes_opened": False,
            "target_outcomes_used": False,
            "claim_boundary": (
                DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_CLAIM_BOUNDARY
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 Prob4D target-outcome authorization",
    ) -> Deform360Prob4DTargetOutcomeAuthorizationV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_AUTHORIZATION_FIELDS, name=name)
        if value["schema"] != DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        ) != DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_VERSION:
            raise ValueError(f"{name} schema_version changed")
        if value["semantics"] != DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SEMANTICS:
            raise ValueError(f"{name} semantics changed")
        if value["claim_boundary"] != (
            DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_CLAIM_BOUNDARY
        ):
            raise ValueError(f"{name} claim boundary changed")
        mapping_fields = (
            "target_manifest_sha256_by_group",
            "target_manifest_artifact_id_by_group",
            "target_provider_run_id_by_group",
            "target_payload_ids_by_group",
            "source_artifacts",
            "metadata",
        )
        for field_name in mapping_fields:
            if not isinstance(value[field_name], Mapping):
                raise ValueError(f"{name}.{field_name} must be a JSON object")
        groups = _canonical_json_strings(
            value["confirmation_group_ids"],
            name=f"{name}.confirmation_group_ids",
        )
        return cls(
            confirmation_opening_authorization_id=value[
                "confirmation_opening_authorization_id"
            ],
            confirmation_opening_token=value["confirmation_opening_token"],
            stage0_selection_artifact_sha256=value[
                "stage0_selection_artifact_sha256"
            ],
            visual_provider_lock_id=value["visual_provider_lock_id"],
            prob4d_cohort_binding_id=value["prob4d_cohort_binding_id"],
            prob4d_promotion_lock_id=value["prob4d_promotion_lock_id"],
            target_provider_admission_id=value["target_provider_admission_id"],
            prob4d_source_revision=value["prob4d_source_revision"],
            bayesian_phystwin_revision=value["bayesian_phystwin_revision"],
            prediction_provider_revision=value["prediction_provider_revision"],
            motioncrafter_revision=value["motioncrafter_revision"],
            model_set_id=value["model_set_id"],
            prediction_run_spec_id=value["prediction_run_spec_id"],
            confirmation_group_ids=groups,
            target_manifest_sha256_by_group=value[
                "target_manifest_sha256_by_group"
            ],
            target_manifest_artifact_id_by_group=value[
                "target_manifest_artifact_id_by_group"
            ],
            target_provider_run_id_by_group=value[
                "target_provider_run_id_by_group"
            ],
            target_payload_ids_by_group=value["target_payload_ids_by_group"],
            source_artifacts=value["source_artifacts"],
            confirmation_provider_inputs_opened=value[
                "confirmation_provider_inputs_opened"
            ],
            target_outcomes_opened=value["target_outcomes_opened"],
            target_outcomes_used=value["target_outcomes_used"],
            metadata=value["metadata"],
            status=value["status"],
            authorization_id=value["authorization_id"],
        )


def build_deform360_prob4d_target_outcome_authorization(
    *,
    stage0_selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    confirmation_opening_authorization: Deform360ConfirmationOpeningAuthorizationV1,
    prob4d_cohort_binding: Mapping[str, Any],
    prob4d_promotion_lock: Mapping[str, Any],
    target_provider_admission: Mapping[str, Any],
    source_artifacts: Mapping[str, str],
    confirmation_provider_inputs_opened: bool,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360Prob4DTargetOutcomeAuthorizationV1:
    """Validate the complete cross-repository chain before target outcomes."""

    if not isinstance(stage0_selection, Deform360Stage0SelectionV1):
        raise TypeError("stage0_selection must be a Deform360Stage0SelectionV1")
    if not isinstance(visual_provider_lock, Deform360VisualProviderLockV1):
        raise TypeError("visual_provider_lock must be a Deform360VisualProviderLockV1")
    if not isinstance(
        confirmation_opening_authorization,
        Deform360ConfirmationOpeningAuthorizationV1,
    ):
        raise TypeError(
            "confirmation_opening_authorization must be a "
            "Deform360ConfirmationOpeningAuthorizationV1"
        )
    if confirmation_opening_authorization.stage0_selection_artifact_sha256 != (
        stage0_selection.selection_artifact_sha256
    ):
        raise ValueError("confirmation authorization Stage-0 identity changed")
    if confirmation_opening_authorization.visual_provider_lock_id != (
        visual_provider_lock.artifact_id
    ):
        raise ValueError("confirmation authorization visual-provider identity changed")
    if confirmation_opening_authorization.confirmation_payloads_opened:
        raise ValueError("base confirmation authorization reports payload access")
    if confirmation_opening_authorization.target_outcomes_used:
        raise ValueError("base confirmation authorization used target outcomes")
    cohort = _validate_prob4d_cohort_binding(
        prob4d_cohort_binding,
        stage0_selection=stage0_selection,
    )
    lock = _validate_prob4d_promotion_lock(
        prob4d_promotion_lock,
        cohort_binding=cohort,
        stage0_selection=stage0_selection,
        confirmation_authorization=confirmation_opening_authorization,
        visual_provider_lock=visual_provider_lock,
    )
    admission, entries = _validate_target_admission(
        target_provider_admission,
        promotion_lock=lock,
        cohort_binding=cohort,
        stage0_selection=stage0_selection,
        confirmation_authorization=confirmation_opening_authorization,
        visual_provider_lock=visual_provider_lock,
    )
    groups = tuple(str(entry["group_id"]) for entry in entries)
    return Deform360Prob4DTargetOutcomeAuthorizationV1(
        confirmation_opening_authorization_id=(
            confirmation_opening_authorization.authorization_id
        ),
        confirmation_opening_token=(
            confirmation_opening_authorization.confirmation_opening_token
        ),
        stage0_selection_artifact_sha256=(
            stage0_selection.selection_artifact_sha256
        ),
        visual_provider_lock_id=visual_provider_lock.artifact_id,
        prob4d_cohort_binding_id=cohort["cohort_binding_id"],
        prob4d_promotion_lock_id=lock["promotion_lock_id"],
        target_provider_admission_id=admission[
            "target_provider_admission_id"
        ],
        prob4d_source_revision=lock["source_revision"],
        bayesian_phystwin_revision=lock["bayesian_phystwin_revision"],
        prediction_provider_revision=visual_provider_lock.provider_revision,
        motioncrafter_revision=visual_provider_lock.motioncrafter_revision,
        model_set_id=visual_provider_lock.model_set_id,
        prediction_run_spec_id=lock["prediction_run_spec_id"],
        confirmation_group_ids=groups,
        target_manifest_sha256_by_group={
            str(entry["group_id"]): str(entry["manifest_sha256"])
            for entry in entries
        },
        target_manifest_artifact_id_by_group={
            str(entry["group_id"]): str(entry["manifest_artifact_id"])
            for entry in entries
        },
        target_provider_run_id_by_group={
            str(entry["group_id"]): str(entry["provider_run_id"])
            for entry in entries
        },
        target_payload_ids_by_group={
            str(entry["group_id"]): tuple(entry["payload_ids"])
            for entry in entries
        },
        source_artifacts=source_artifacts,
        confirmation_provider_inputs_opened=confirmation_provider_inputs_opened,
        metadata={} if metadata is None else metadata,
    )


def verify_deform360_prob4d_target_outcome_authorization(
    observed: Deform360Prob4DTargetOutcomeAuthorizationV1,
    **arguments: Any,
) -> None:
    """Rebuild one authorization and require exact deterministic equality."""

    if not isinstance(observed, Deform360Prob4DTargetOutcomeAuthorizationV1):
        raise TypeError(
            "observed must be a Deform360Prob4DTargetOutcomeAuthorizationV1"
        )
    replayed = build_deform360_prob4d_target_outcome_authorization(**arguments)
    if observed.to_record() != replayed.to_record():
        raise ValueError("target-outcome authorization differs from deterministic replay")


def save_deform360_prob4d_target_outcome_authorization(
    value: Deform360Prob4DTargetOutcomeAuthorizationV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist one validated post-provider authorization."""

    if not isinstance(value, Deform360Prob4DTargetOutcomeAuthorizationV1):
        raise TypeError(
            "value must be a Deform360Prob4DTargetOutcomeAuthorizationV1"
        )
    write_atomic_json(value.to_record(), path, overwrite=overwrite)


def load_deform360_prob4d_target_outcome_authorization(
    path: str | Path,
) -> Deform360Prob4DTargetOutcomeAuthorizationV1:
    """Strictly load and independently revalidate one authorization."""

    return Deform360Prob4DTargetOutcomeAuthorizationV1.from_mapping(
        load_strict_json_object(
            path,
            label="Deform360 Prob4D target-outcome authorization",
        )
    )


__all__ = [
    "BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY",
    "BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY",
    "BPT_STAGE0_SOURCE_KEY",
    "BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY",
    "BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY",
    "CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY",
    "DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_CLAIM_BOUNDARY",
    "DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SCHEMA",
    "DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SEMANTICS",
    "DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_STATUS",
    "DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_VERSION",
    "PREDICTION_PROVIDER_REVISION_METADATA_KEY",
    "PROB4D_COHORT_BINDING_SOURCE_KEY",
    "PROB4D_PROMOTION_LOCK_SOURCE_KEY",
    "PROB4D_TARGET_ADMISSION_SOURCE_KEY",
    "Deform360Prob4DTargetOutcomeAuthorizationV1",
    "build_deform360_prob4d_target_outcome_authorization",
    "load_deform360_prob4d_target_outcome_authorization",
    "save_deform360_prob4d_target_outcome_authorization",
    "verify_deform360_prob4d_target_outcome_authorization",
]
