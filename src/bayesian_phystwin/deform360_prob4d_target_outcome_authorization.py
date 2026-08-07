"""Bind admitted Prob4D target manifests before Deform360 outcome access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from ._deform360_prob4d_contracts import (
    BAYESIAN_PHYSTWIN_REPOSITORY,
    BPT_CONFIRMATION_AUTHORIZATION_METADATA_KEY,
    BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY,
    BPT_STAGE0_SOURCE_KEY,
    BPT_VISUAL_PROVIDER_LOCK_METADATA_KEY,
    BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY,
    CONFIRMATION_PROVIDER_INPUTS_ONLY_METADATA_KEY,
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
    REQUIRED_SOURCE_KEYS,
    validate_prob4d_target_chain,
)
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
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


def _confirmation_authorization_id(
    value: Deform360ConfirmationOpeningAuthorizationV1,
) -> str:
    authorization_id = value.authorization_id
    if authorization_id is None:
        raise ValueError("confirmation opening authorization lacks an identity")
    return sha256_digest(
        authorization_id,
        name="confirmation_opening_authorization_id",
    )


def _frozen_source_artifacts(value: Mapping[str, str]) -> Mapping[str, Any]:
    sources = source_artifact_mapping(value, name="source_artifacts")
    missing = sorted(REQUIRED_SOURCE_KEYS - set(sources))
    extra = sorted(set(sources) - REQUIRED_SOURCE_KEYS)
    if missing or extra:
        raise ValueError(
            f"source_artifacts changed: missing={missing}, extra={extra}"
        )
    return sources


def _digest_mapping(
    value: Mapping[str, str],
    *,
    name: str,
    groups: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if tuple(sorted(value)) != groups:
        raise ValueError(f"{name} group IDs changed")
    return frozen_finite_json_mapping(
        {
            group_id: sha256_digest(value[group_id], name=f"{name}[{group_id!r}]")
            for group_id in groups
        },
        name=name,
    )


def _payload_mapping(
    value: Mapping[str, Sequence[str]],
    *,
    groups: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != groups:
        raise ValueError("target_payload_ids_by_group group IDs changed")
    normalized: dict[str, list[str]] = {}
    for group_id in groups:
        raw = value[group_id]
        if isinstance(raw, (str, bytes)):
            raise ValueError("target payload IDs must be a sequence")
        payload_ids = tuple(
            sha256_digest(item, name=f"target payload ID for {group_id}")
            for item in raw
        )
        if not payload_ids or payload_ids != tuple(sorted(set(payload_ids))):
            raise ValueError(
                "target payload IDs must be nonempty, sorted, and unique"
            )
        normalized[group_id] = list(payload_ids)
    return frozen_finite_json_mapping(
        normalized,
        name="target_payload_ids_by_group",
    )


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
        manifest_sha256 = _digest_mapping(
            self.target_manifest_sha256_by_group,
            name="target_manifest_sha256_by_group",
            groups=groups,
        )
        manifest_artifact_ids = _digest_mapping(
            self.target_manifest_artifact_id_by_group,
            name="target_manifest_artifact_id_by_group",
            groups=groups,
        )
        provider_run_ids = _digest_mapping(
            self.target_provider_run_id_by_group,
            name="target_provider_run_id_by_group",
            groups=groups,
        )
        payload_ids = _payload_mapping(
            self.target_payload_ids_by_group,
            groups=groups,
        )
        sources = _frozen_source_artifacts(self.source_artifacts)
        provider_inputs_opened = genuine_boolean(
            self.confirmation_provider_inputs_opened,
            name="confirmation_provider_inputs_opened",
        )
        target_outcomes_opened = genuine_boolean(
            self.target_outcomes_opened,
            name="target_outcomes_opened",
        )
        target_outcomes_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not provider_inputs_opened:
            raise ValueError(
                "provider inputs must be opened before target admission exists"
            )
        if target_outcomes_opened or target_outcomes_used:
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
        object.__setattr__(
            self,
            "target_manifest_sha256_by_group",
            manifest_sha256,
        )
        object.__setattr__(
            self,
            "target_manifest_artifact_id_by_group",
            manifest_artifact_ids,
        )
        object.__setattr__(
            self,
            "target_provider_run_id_by_group",
            provider_run_ids,
        )
        object.__setattr__(self, "target_payload_ids_by_group", payload_ids)
        object.__setattr__(self, "source_artifacts", sources)
        object.__setattr__(
            self,
            "confirmation_provider_inputs_opened",
            provider_inputs_opened,
        )
        object.__setattr__(self, "target_outcomes_opened", target_outcomes_opened)
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.identity_record())
        if self.authorization_id is not None:
            supplied_id = sha256_digest(
                self.authorization_id,
                name="authorization_id",
            )
            if supplied_id != expected_id:
                raise ValueError(
                    "target-outcome authorization_id does not match content"
                )
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

        authorization_id = self.authorization_id
        if authorization_id is None:
            raise AssertionError("target-outcome authorization lacks an identity")
        return {
            "target_provider_admission_id": self.target_provider_admission_id,
            "deform360_confirmation_opening_authorization_id": (
                self.confirmation_opening_authorization_id
            ),
            "deform360_prob4d_target_outcome_authorization_id": authorization_id,
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
        if value["schema_version"] != (
            DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_VERSION
        ):
            raise ValueError(f"{name} schema_version changed")
        if value["semantics"] != (
            DEFORM360_PROB4D_TARGET_OUTCOME_AUTHORIZATION_SEMANTICS
        ):
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
        group_ids = value["confirmation_group_ids"]
        if type(group_ids) is not list:
            raise ValueError(f"{name}.confirmation_group_ids must be an array")
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
            confirmation_group_ids=group_ids,
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

    chain = validate_prob4d_target_chain(
        prob4d_cohort_binding=prob4d_cohort_binding,
        prob4d_promotion_lock=prob4d_promotion_lock,
        target_provider_admission=target_provider_admission,
        stage0_selection=stage0_selection,
        confirmation_opening_authorization=confirmation_opening_authorization,
        visual_provider_lock=visual_provider_lock,
    )
    return Deform360Prob4DTargetOutcomeAuthorizationV1(
        confirmation_opening_authorization_id=_confirmation_authorization_id(
            confirmation_opening_authorization
        ),
        confirmation_opening_token=(
            confirmation_opening_authorization.confirmation_opening_token
        ),
        stage0_selection_artifact_sha256=(
            stage0_selection.selection_artifact_sha256
        ),
        visual_provider_lock_id=visual_provider_lock.artifact_id,
        prob4d_cohort_binding_id=chain.cohort_binding_id,
        prob4d_promotion_lock_id=chain.promotion_lock_id,
        target_provider_admission_id=chain.target_provider_admission_id,
        prob4d_source_revision=chain.prob4d_source_revision,
        bayesian_phystwin_revision=chain.bayesian_phystwin_revision,
        prediction_provider_revision=visual_provider_lock.provider_revision,
        motioncrafter_revision=visual_provider_lock.motioncrafter_revision,
        model_set_id=visual_provider_lock.model_set_id,
        prediction_run_spec_id=chain.prediction_run_spec_id,
        confirmation_group_ids=chain.groups,
        target_manifest_sha256_by_group=chain.manifest_sha256_by_group,
        target_manifest_artifact_id_by_group=(
            chain.manifest_artifact_id_by_group
        ),
        target_provider_run_id_by_group=chain.provider_run_id_by_group,
        target_payload_ids_by_group=chain.payload_ids_by_group,
        source_artifacts=source_artifacts,
        confirmation_provider_inputs_opened=confirmation_provider_inputs_opened,
        metadata={} if metadata is None else metadata,
    )


def verify_deform360_prob4d_target_outcome_authorization(
    observed: Deform360Prob4DTargetOutcomeAuthorizationV1,
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
) -> None:
    """Rebuild one authorization and require exact deterministic equality."""

    if not isinstance(observed, Deform360Prob4DTargetOutcomeAuthorizationV1):
        raise TypeError(
            "observed must be a Deform360Prob4DTargetOutcomeAuthorizationV1"
        )
    replayed = build_deform360_prob4d_target_outcome_authorization(
        stage0_selection=stage0_selection,
        visual_provider_lock=visual_provider_lock,
        confirmation_opening_authorization=confirmation_opening_authorization,
        prob4d_cohort_binding=prob4d_cohort_binding,
        prob4d_promotion_lock=prob4d_promotion_lock,
        target_provider_admission=target_provider_admission,
        source_artifacts=source_artifacts,
        confirmation_provider_inputs_opened=confirmation_provider_inputs_opened,
        metadata=metadata,
    )
    if observed.to_record() != replayed.to_record():
        raise ValueError(
            "target-outcome authorization differs from deterministic replay"
        )


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
    "BAYESIAN_PHYSTWIN_REPOSITORY",
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
    "Deform360Prob4DTargetOutcomeAuthorizationV1",
    "build_deform360_prob4d_target_outcome_authorization",
    "load_deform360_prob4d_target_outcome_authorization",
    "save_deform360_prob4d_target_outcome_authorization",
    "verify_deform360_prob4d_target_outcome_authorization",
]
