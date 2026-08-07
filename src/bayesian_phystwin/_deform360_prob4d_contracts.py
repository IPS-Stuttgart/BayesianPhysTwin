"""Strict frozen Prob4D contracts used by the Deform360 outcome gate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._canonical_contracts import genuine_boolean, genuine_integer
from ._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
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
PROB4D_PROMOTION_LOCK_SCHEMA = "prob4d.heldout-provider-promotion-lock"
PROB4D_TARGET_ADMISSION_SCHEMA = "prob4d.heldout-target-provider-admission"

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
REQUIRED_SOURCE_KEYS = frozenset(
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
_UNIT_FIELDS = frozenset(
    {"object_id", "stratum", "episode_id", "metadata_path", "metadata_sha256"}
)
_LOCK_FIELDS = frozenset(
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
_ARM_FIELDS = frozenset(
    {
        "arm_id",
        "role",
        "query_method_id",
        "provider_method_id",
        "sensor_assisted",
        "metadata",
    }
)
_REQUIRED_ROLES = frozenset(
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
_ENTRY_FIELDS = frozenset(
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
_PAYLOAD_FIELDS = frozenset(
    {
        "payload_id",
        "window_id",
        "output_frame_ids",
        "source_frame_start",
        "source_frame_stop_exclusive",
        "dependence_group_ids",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _array(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return value


def _sorted_strings(value: object, *, name: str) -> tuple[str, ...]:
    values = tuple(
        nonempty_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(_array(value, name=name))
    )
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise ValueError(f"{name} must be sorted and unique")
    return values


def _finite_nonnegative(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _content_identity(
    value: Mapping[str, Any],
    *,
    id_field: str,
    name: str,
) -> str:
    supplied = sha256_digest(value[id_field], name=f"{name}.{id_field}")
    descriptor = dict(value)
    descriptor.pop(id_field)
    if content_id(descriptor) != supplied:
        raise ValueError(f"{name}.{id_field} does not match content")
    return supplied


def _unit(value: object, *, name: str) -> dict[str, object]:
    mapping = _mapping(value, name=name)
    require_exact_fields(mapping, expected=_UNIT_FIELDS, name=name)
    object_id = nonempty_string(mapping["object_id"], name=f"{name}.object_id")
    stratum = nonempty_string(mapping["stratum"], name=f"{name}.stratum")
    if stratum not in {"sheet", "volumetric"}:
        raise ValueError(f"{name}.stratum must be sheet or volumetric")
    episode_id = genuine_integer(
        mapping["episode_id"],
        name=f"{name}.episode_id",
        minimum=0,
    )
    metadata_path = nonempty_string(
        mapping["metadata_path"],
        name=f"{name}.metadata_path",
    )
    if metadata_path != f"raw/{object_id}/metadata.json":
        raise ValueError(f"{name}.metadata_path changed")
    return {
        "object_id": object_id,
        "stratum": stratum,
        "episode_id": episode_id,
        "metadata_path": metadata_path,
        "metadata_sha256": sha256_digest(
            mapping["metadata_sha256"],
            name=f"{name}.metadata_sha256",
        ),
    }


def _units(value: object, *, name: str) -> tuple[dict[str, object], ...]:
    result = tuple(
        _unit(item, name=f"{name}[{index}]")
        for index, item in enumerate(_array(value, name=name))
    )
    ordered = tuple(sorted(result, key=lambda item: str(item["object_id"])))
    if result != ordered:
        raise ValueError(f"{name} must be sorted by object_id")
    if len({str(item["object_id"]) for item in result}) != len(result):
        raise ValueError(f"{name} repeats an object")
    return result


def _stage0_units(
    selection: Deform360Stage0SelectionV1,
    *,
    confirmation: bool,
) -> tuple[dict[str, object], ...]:
    source = (
        selection.confirmation_units if confirmation else selection.calibration_units
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
                for unit in source
            ),
            key=lambda item: str(item["object_id"]),
        )
    )


def _cohort(
    value: Mapping[str, Any],
    *,
    stage0: Deform360Stage0SelectionV1,
) -> Mapping[str, Any]:
    require_exact_fields(value, expected=_COHORT_FIELDS, name="Prob4D cohort binding")
    if value["schema_name"] != PROB4D_COHORT_BINDING_SCHEMA:
        raise ValueError("Prob4D cohort binding schema changed")
    if genuine_integer(
        value["schema_version"],
        name="cohort schema_version",
        minimum=1,
    ) != 1:
        raise ValueError("Prob4D cohort binding version changed")
    if value["claim_boundary"] != PROB4D_COHORT_BINDING_CLAIM_BOUNDARY:
        raise ValueError("Prob4D cohort binding claim boundary changed")
    if repository_name(
        value["source_repository"],
        name="cohort source_repository",
    ) != BAYESIAN_PHYSTWIN_REPOSITORY:
        raise ValueError("Prob4D cohort binding uses another source repository")
    exact_revision(value["source_revision"], name="cohort source_revision")
    if value["source_path"] != DEFORM360_SELECTION_PATH:
        raise ValueError("Prob4D cohort binding uses another Stage-0 path")
    if value["selection_schema"] != (
        "bayesian-phystwin/deform360-official-hub-selection-v1"
    ):
        raise ValueError("Prob4D cohort binding selection schema changed")
    if genuine_integer(
        value["selection_schema_version"],
        name="selection_schema_version",
        minimum=1,
    ) != 1:
        raise ValueError("Prob4D cohort binding selection version changed")
    expected_scalars = {
        "selection_artifact_sha256": stage0.selection_artifact_sha256,
        "content_selection_sha256": stage0.content_selection_sha256,
        "selection_sha256": stage0.selection_sha256,
        "selection_implementation_revision": stage0.implementation_revision,
        "protocol_id": stage0.protocol_id,
        "protocol_sha256": stage0.protocol_sha256,
        "dataset_resolved_revision": stage0.dataset_revision,
        "processing_revision": stage0.processing_revision,
    }
    for key, expected in expected_scalars.items():
        if value[key] != expected:
            raise ValueError(f"Prob4D cohort binding changed {key}")
    if value["dataset_repository"] != "brownu/deform360":
        raise ValueError("Prob4D cohort binding dataset repository changed")
    if value["processing_repository"] != "lhy0807/deform360":
        raise ValueError("Prob4D cohort binding processing repository changed")
    nonempty_string(
        value["dataset_requested_revision"],
        name="dataset_requested_revision",
    )
    if value["statistical_unit"] != "physical-object":
        raise ValueError("Prob4D cohort binding statistical unit changed")
    if genuine_boolean(
        value["replacement_allowed_after_payload_access"],
        name="replacement_allowed_after_payload_access",
    ):
        raise ValueError("Prob4D cohort binding permits cohort replacement")
    boundary = _mapping(
        value["information_boundary"],
        name="cohort information_boundary",
    )
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
        observed = genuine_boolean(
            boundary[key],
            name=f"information_boundary.{key}",
        )
        if observed != expected:
            raise ValueError(f"Prob4D cohort information boundary changed: {key}")
    calibration = _units(value["calibration_units"], name="calibration_units")
    target = _units(value["target_units"], name="target_units")
    if calibration != _stage0_units(stage0, confirmation=False):
        raise ValueError("Prob4D cohort calibration units differ from Stage 0")
    if target != _stage0_units(stage0, confirmation=True):
        raise ValueError("Prob4D cohort target units differ from Stage 0")
    calibration_ids = tuple(str(item["object_id"]) for item in calibration)
    target_ids = tuple(str(item["object_id"]) for item in target)
    if _sorted_strings(
        value["calibration_group_ids"],
        name="calibration_group_ids",
    ) != calibration_ids:
        raise ValueError("Prob4D cohort calibration group IDs changed")
    if _sorted_strings(
        value["target_group_ids"],
        name="target_group_ids",
    ) != target_ids:
        raise ValueError("Prob4D cohort target group IDs changed")
    _content_identity(
        value,
        id_field="cohort_binding_id",
        name="Prob4D cohort binding",
    )
    return value


def _metadata(
    value: object,
    *,
    confirmation: Deform360ConfirmationOpeningAuthorizationV1,
    provider: Deform360VisualProviderLockV1,
    require_producer: bool,
) -> Mapping[str, Any]:
    mapping = _mapping(value, name="Prob4D metadata")
    if mapping.get(BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY) != (
        confirmation.authorization_id
    ):
        raise ValueError("Prob4D metadata binds another confirmation authorization")
    if mapping.get(BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY) != provider.artifact_id:
        raise ValueError("Prob4D metadata binds another visual-provider lock")
    if require_producer:
        if mapping.get(PREDICTION_PROVIDER_REVISION_METADATA_KEY) != (
            provider.provider_revision
        ):
            raise ValueError("Prob4D admission binds another prediction-provider revision")
        if not genuine_boolean(
            mapping.get(CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY),
            name=CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY,
        ):
            raise ValueError("Prob4D admission does not declare provider-input-only access")
    return mapping


def _arms(value: object) -> None:
    raw = _array(value, name="promotion arms")
    if not raw:
        raise ValueError("promotion arms must not be empty")
    arm_ids: list[str] = []
    roles: list[str] = []
    for index, raw_arm in enumerate(raw):
        name = f"promotion arms[{index}]"
        arm = _mapping(raw_arm, name=name)
        require_exact_fields(arm, expected=_ARM_FIELDS, name=name)
        arm_id = nonempty_string(arm["arm_id"], name=f"{name}.arm_id")
        role = nonempty_string(arm["role"], name=f"{name}.role")
        nonempty_string(arm["query_method_id"], name=f"{name}.query_method_id")
        provider_method = arm["provider_method_id"]
        sensor_assisted = genuine_boolean(
            arm["sensor_assisted"],
            name=f"{name}.sensor_assisted",
        )
        _mapping(arm["metadata"], name=f"{name}.metadata")
        if role == "physical_fallback":
            if provider_method is not None or sensor_assisted:
                raise ValueError("physical fallback arm semantics changed")
        else:
            nonempty_string(provider_method, name=f"{name}.provider_method_id")
        if (role == "sensor_assisted") != sensor_assisted:
            raise ValueError("sensor-assisted promotion arm semantics changed")
        arm_ids.append(arm_id)
        roles.append(role)
    if tuple(arm_ids) != tuple(sorted(arm_ids)) or len(set(arm_ids)) != len(
        arm_ids
    ):
        raise ValueError("promotion arms must be sorted by unique arm_id")
    missing = sorted(_REQUIRED_ROLES - set(roles))
    if missing:
        raise ValueError(f"promotion arms are missing required roles: {missing}")
    for role in _REQUIRED_ROLES:
        if roles.count(role) != 1:
            raise ValueError(f"promotion role {role!r} must occur exactly once")


def _lock(
    value: Mapping[str, Any],
    *,
    cohort: Mapping[str, Any],
    stage0: Deform360Stage0SelectionV1,
    confirmation: Deform360ConfirmationOpeningAuthorizationV1,
    provider: Deform360VisualProviderLockV1,
) -> Mapping[str, Any]:
    require_exact_fields(value, expected=_LOCK_FIELDS, name="Prob4D promotion lock")
    if value["schema_name"] != PROB4D_PROMOTION_LOCK_SCHEMA:
        raise ValueError("Prob4D promotion lock schema changed")
    if genuine_integer(
        value["schema_version"],
        name="promotion lock schema_version",
        minimum=1,
    ) != 1:
        raise ValueError("Prob4D promotion lock version changed")
    if value["claim_boundary"] != PROB4D_PROMOTION_LOCK_CLAIM_BOUNDARY:
        raise ValueError("Prob4D promotion lock claim boundary changed")
    if repository_name(
        value["source_repository"],
        name="promotion source_repository",
    ) != PROB4D_REPOSITORY:
        raise ValueError("Prob4D promotion lock source repository changed")
    exact_revision(value["source_revision"], name="Prob4D source_revision")
    if repository_name(
        value["bayesian_phystwin_repository"],
        name="bayesian_phystwin_repository",
    ) != BAYESIAN_PHYSTWIN_REPOSITORY:
        raise ValueError("Prob4D promotion lock BayesianPhysTwin repository changed")
    exact_revision(
        value["bayesian_phystwin_revision"],
        name="BayesianPhysTwin promotion revision",
    )
    if value["motioncrafter_revision"] != provider.motioncrafter_revision:
        raise ValueError("Prob4D promotion lock MotionCrafter revision changed")
    if value["model_set_id"] != provider.model_set_id:
        raise ValueError("Prob4D promotion lock model-set identity changed")
    sha256_digest(value["prediction_run_spec_id"], name="prediction_run_spec_id")
    sha256_digest(
        value["provider_evaluation_manifest_sha256"],
        name="provider_evaluation_manifest_sha256",
    )
    frozen = _mapping(value["frozen_artifact_ids"], name="frozen_artifact_ids")
    for key, artifact_id in frozen.items():
        nonempty_string(key, name="frozen_artifact_ids key")
        sha256_digest(artifact_id, name=f"frozen_artifact_ids[{key!r}]")
    if frozen.get("cohort_binding") != cohort["cohort_binding_id"]:
        raise ValueError("Prob4D promotion lock binds another cohort")
    if frozen.get("provider_configuration") != provider.artifact_id:
        raise ValueError("Prob4D promotion lock binds another provider configuration")
    calibration_ids = tuple(
        sorted(unit.object_id for unit in stage0.calibration_units)
    )
    target_ids = tuple(sorted(unit.object_id for unit in stage0.confirmation_units))
    if _sorted_strings(
        value["calibration_group_ids"],
        name="promotion calibration_group_ids",
    ) != calibration_ids:
        raise ValueError("Prob4D promotion lock calibration groups changed")
    if _sorted_strings(
        value["target_group_ids"],
        name="promotion target_group_ids",
    ) != target_ids:
        raise ValueError("Prob4D promotion lock target groups changed")
    _sorted_strings(
        value["development_group_ids"],
        name="promotion development_group_ids",
    )
    _arms(value["arms"])
    nonempty_string(
        value["provider_reference_arm_id"],
        name="provider_reference_arm_id",
    )
    nonempty_string(value["primary_query_arm_id"], name="primary_query_arm_id")
    genuine_integer(value["bootstrap_resamples"], name="bootstrap_resamples", minimum=100)
    genuine_integer(value["bootstrap_seed"], name="bootstrap_seed", minimum=0)
    minimum_groups = genuine_integer(
        value["minimum_target_group_count"],
        name="minimum_target_group_count",
        minimum=1,
    )
    if minimum_groups != len(target_ids):
        raise ValueError("Prob4D promotion lock does not require the complete target cohort")
    for field_name in (
        "query_superiority_margin_mm",
        "harmful_update_margin_mm",
        "maximum_worst_group_regression_mm",
    ):
        _finite_nonnegative(value[field_name], name=field_name)
    for field_name in (
        "maximum_harmful_accepted_updates",
        "maximum_technical_failures",
    ):
        genuine_integer(value[field_name], name=field_name, minimum=0)
    coverage = value["minimum_mean_accepted_coverage"]
    if coverage is not None:
        observed = _finite_nonnegative(
            coverage,
            name="minimum_mean_accepted_coverage",
        )
        if observed > 1.0:
            raise ValueError("minimum_mean_accepted_coverage must lie in [0, 1]")
    _metadata(
        value["metadata"],
        confirmation=confirmation,
        provider=provider,
        require_producer=False,
    )
    _content_identity(
        value,
        id_field="promotion_lock_id",
        name="Prob4D promotion lock",
    )
    return value


def _payload(value: object, *, cutoff: int, name: str) -> str:
    payload = _mapping(value, name=name)
    require_exact_fields(payload, expected=_PAYLOAD_FIELDS, name=name)
    payload_id = sha256_digest(payload["payload_id"], name=f"{name}.payload_id")
    nonempty_string(payload["window_id"], name=f"{name}.window_id")
    frames = tuple(
        genuine_integer(frame, name=f"{name}.output_frame_ids[{index}]", minimum=0)
        for index, frame in enumerate(
            _array(payload["output_frame_ids"], name=f"{name}.output_frame_ids")
        )
    )
    if not frames or frames != tuple(sorted(set(frames))):
        raise ValueError(f"{name}.output_frame_ids must be nonempty, sorted, and unique")
    start = genuine_integer(
        payload["source_frame_start"],
        name=f"{name}.source_frame_start",
        minimum=0,
    )
    stop = genuine_integer(
        payload["source_frame_stop_exclusive"],
        name=f"{name}.source_frame_stop_exclusive",
        minimum=1,
    )
    if stop <= start or stop > cutoff:
        raise ValueError(f"{name} crosses its causal cutoff")
    if not _sorted_strings(
        payload["dependence_group_ids"],
        name=f"{name}.dependence_group_ids",
    ):
        raise ValueError(f"{name}.dependence_group_ids must not be empty")
    return payload_id


def _entries(
    value: object,
    *,
    stage0: Deform360Stage0SelectionV1,
) -> tuple[dict[str, Any], ...]:
    expected = {unit.object_id: unit for unit in stage0.confirmation_units}
    summaries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(_array(value, name="target entries")):
        name = f"target entries[{index}]"
        entry = _mapping(raw_entry, name=name)
        require_exact_fields(entry, expected=_ENTRY_FIELDS, name=name)
        group_id = nonempty_string(entry["group_id"], name=f"{name}.group_id")
        if group_id not in expected:
            raise ValueError(f"{name} identifies an unregistered target group")
        unit = expected[group_id]
        episode_id = genuine_integer(
            entry["episode_id"],
            name=f"{name}.episode_id",
            minimum=0,
        )
        if episode_id != unit.episode_id or entry["stratum"] != unit.stratum:
            raise ValueError(f"{name} object/episode/stratum differs from Stage 0")
        sequence_id = nonempty_string(
            entry["sequence_id"],
            name=f"{name}.sequence_id",
        )
        cutoff = genuine_integer(
            entry["causal_frame_stop"],
            name=f"{name}.causal_frame_stop",
            minimum=1,
        )
        raw_payloads = _array(
            entry["admitted_payloads"],
            name=f"{name}.admitted_payloads",
        )
        if not raw_payloads:
            raise ValueError(f"{name}.admitted_payloads must not be empty")
        payload_ids = tuple(
            _payload(
                payload,
                cutoff=cutoff,
                name=f"{name}.admitted_payloads[{payload_index}]",
            )
            for payload_index, payload in enumerate(raw_payloads)
        )
        if payload_ids != tuple(sorted(set(payload_ids))):
            raise ValueError(f"{name}.admitted_payloads must be sorted by payload_id")
        summaries.append(
            {
                "group_id": group_id,
                "sequence_id": sequence_id,
                "manifest_sha256": sha256_digest(
                    entry["manifest_sha256"],
                    name=f"{name}.manifest_sha256",
                ),
                "manifest_artifact_id": sha256_digest(
                    entry["manifest_artifact_id"],
                    name=f"{name}.manifest_artifact_id",
                ),
                "provider_run_id": sha256_digest(
                    entry["provider_run_id"],
                    name=f"{name}.provider_run_id",
                ),
                "payload_ids": payload_ids,
            }
        )
    summaries.sort(key=lambda item: str(item["group_id"]))
    if tuple(str(item["group_id"]) for item in summaries) != tuple(sorted(expected)):
        raise ValueError("target admission does not cover the exact confirmation cohort")
    return tuple(summaries)


def _admission(
    value: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    cohort: Mapping[str, Any],
    stage0: Deform360Stage0SelectionV1,
    confirmation: Deform360ConfirmationOpeningAuthorizationV1,
    provider: Deform360VisualProviderLockV1,
) -> tuple[Mapping[str, Any], tuple[dict[str, Any], ...]]:
    require_exact_fields(value, expected=_ADMISSION_FIELDS, name="Prob4D target admission")
    if value["schema_name"] != PROB4D_TARGET_ADMISSION_SCHEMA:
        raise ValueError("Prob4D target admission schema changed")
    if genuine_integer(
        value["schema_version"],
        name="target admission schema_version",
        minimum=1,
    ) != 1:
        raise ValueError("Prob4D target admission version changed")
    if value["claim_boundary"] != PROB4D_TARGET_ADMISSION_CLAIM_BOUNDARY:
        raise ValueError("Prob4D target admission claim boundary changed")
    expected_values = {
        "promotion_lock_id": lock["promotion_lock_id"],
        "cohort_binding_id": cohort["cohort_binding_id"],
        "source_repository": lock["source_repository"],
        "source_revision": lock["source_revision"],
        "prediction_run_spec_id": lock["prediction_run_spec_id"],
        "provider_repository": provider.motioncrafter_repository,
        "provider_revision": provider.motioncrafter_revision,
        "model_set_id": provider.model_set_id,
    }
    for key, expected in expected_values.items():
        if value[key] != expected:
            raise ValueError(f"Prob4D target admission changed {key}")
    nonempty_string(value["provider_family"], name="provider_family")
    sha256_digest(value["loader_id"], name="loader_id")
    for field_name in (
        "coordinate_semantics",
        "point_semantics",
        "flow_semantics",
        "ray_semantics",
        "source_dependency_semantics",
    ):
        nonempty_string(value[field_name], name=field_name)
    if value["source_dependency_semantics"] != (
        "per-output-exclusive-source-frame-interval-v1"
    ):
        raise ValueError("Prob4D target admission source-dependency semantics changed")
    if genuine_boolean(value["target_outcomes_used"], name="target_outcomes_used"):
        raise ValueError("Prob4D target admission used target outcomes")
    _metadata(
        value["metadata"],
        confirmation=confirmation,
        provider=provider,
        require_producer=True,
    )
    summaries = _entries(value["entries"], stage0=stage0)
    _content_identity(
        value,
        id_field="target_provider_admission_id",
        name="Prob4D target admission",
    )
    return value, summaries


@dataclass(frozen=True)
class ValidatedProb4DTargetChainV1:
    """Validated identities and per-object summaries from the foreign contracts."""

    cohort_binding_id: str
    promotion_lock_id: str
    target_provider_admission_id: str
    prob4d_source_revision: str
    bayesian_phystwin_revision: str
    prediction_run_spec_id: str
    groups: tuple[str, ...]
    manifest_sha256_by_group: Mapping[str, str]
    manifest_artifact_id_by_group: Mapping[str, str]
    provider_run_id_by_group: Mapping[str, str]
    payload_ids_by_group: Mapping[str, tuple[str, ...]]


def validate_prob4d_target_chain(
    *,
    prob4d_cohort_binding: Mapping[str, Any],
    prob4d_promotion_lock: Mapping[str, Any],
    target_provider_admission: Mapping[str, Any],
    stage0_selection: Deform360Stage0SelectionV1,
    confirmation_opening_authorization: Deform360ConfirmationOpeningAuthorizationV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
) -> ValidatedProb4DTargetChainV1:
    """Validate exact Prob4D v1 artifacts against the frozen BPT protocol."""

    cohort = _cohort(prob4d_cohort_binding, stage0=stage0_selection)
    lock = _lock(
        prob4d_promotion_lock,
        cohort=cohort,
        stage0=stage0_selection,
        confirmation=confirmation_opening_authorization,
        provider=visual_provider_lock,
    )
    admission, summaries = _admission(
        target_provider_admission,
        lock=lock,
        cohort=cohort,
        stage0=stage0_selection,
        confirmation=confirmation_opening_authorization,
        provider=visual_provider_lock,
    )
    groups = tuple(str(summary["group_id"]) for summary in summaries)
    return ValidatedProb4DTargetChainV1(
        cohort_binding_id=str(cohort["cohort_binding_id"]),
        promotion_lock_id=str(lock["promotion_lock_id"]),
        target_provider_admission_id=str(
            admission["target_provider_admission_id"]
        ),
        prob4d_source_revision=str(lock["source_revision"]),
        bayesian_phystwin_revision=str(lock["bayesian_phystwin_revision"]),
        prediction_run_spec_id=str(lock["prediction_run_spec_id"]),
        groups=groups,
        manifest_sha256_by_group={
            str(summary["group_id"]): str(summary["manifest_sha256"])
            for summary in summaries
        },
        manifest_artifact_id_by_group={
            str(summary["group_id"]): str(summary["manifest_artifact_id"])
            for summary in summaries
        },
        provider_run_id_by_group={
            str(summary["group_id"]): str(summary["provider_run_id"])
            for summary in summaries
        },
        payload_ids_by_group={
            str(summary["group_id"]): tuple(summary["payload_ids"])
            for summary in summaries
        },
    )


__all__ = [
    "BAYESIAN_PHYSTWIN_REPOSITORY",
    "BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY",
    "BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY",
    "BPT_STAGE0_SOURCE_KEY",
    "BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY",
    "BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY",
    "CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY",
    "PREDICTION_PROVIDER_REVISION_METADATA_KEY",
    "PROB4D_COHORT_BINDING_CLAIM_BOUNDARY",
    "PROB4D_COHORT_BINDING_SCHEMA",
    "PROB4D_COHORT_BINDING_SOURCE_KEY",
    "PROB4D_PROMOTION_LOCK_CLAIM_BOUNDARY",
    "PROB4D_PROMOTION_LOCK_SCHEMA",
    "PROB4D_PROMOTION_LOCK_SOURCE_KEY",
    "PROB4D_REPOSITORY",
    "PROB4D_TARGET_ADMISSION_CLAIM_BOUNDARY",
    "PROB4D_TARGET_ADMISSION_SCHEMA",
    "PROB4D_TARGET_ADMISSION_SOURCE_KEY",
    "REQUIRED_SOURCE_KEYS",
    "ValidatedProb4DTargetChainV1",
    "validate_prob4d_target_chain",
]
