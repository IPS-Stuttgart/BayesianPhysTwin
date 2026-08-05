# ruff: noqa: F403, F405
"""Internal implementation slice for Deform360 calibration acquisition."""

from __future__ import annotations

from ._deform360_calibration_acquisition_common import *

@dataclass(frozen=True)
class Deform360CalibrationAcquisitionPlanV1:
    """Exact calibration-only payload allowlist fixed before access."""

    selection_artifact_sha256: str
    content_selection_sha256: str
    visual_provider_lock_id: str
    dataset_revision: str
    processing_revision: str
    implementation_revision: str
    calibration_units: Sequence[Deform360CohortUnitV1]
    forbidden_confirmation_object_ids: Sequence[str]
    camera_policy: str = "all-official-cameras-sorted-v1"
    tactile_policy: str = "all-exact-npy-timestamp-sensors-sorted-v1"
    audio_policy: str = "exclude-wav-and-flac-v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_id: str = DEFORM360_CALIBRATION_PROTOCOL_ID
    dataset_repository: str = DEFORM360_DATASET_REPOSITORY
    processing_repository: str = DEFORM360_PROCESSING_REPOSITORY
    calibration_payloads_opened: bool = False
    confirmation_payloads_opened: bool = False
    target_outcomes_used: bool = False

    def __post_init__(self) -> None:
        protocol_id = nonempty_string(self.protocol_id, name="protocol_id")
        _require(
            protocol_id == DEFORM360_CALIBRATION_PROTOCOL_ID,
            "calibration acquisition protocol changed",
        )
        dataset_repository = nonempty_string(
            self.dataset_repository,
            name="dataset_repository",
        )
        _require(
            dataset_repository == DEFORM360_DATASET_REPOSITORY,
            "calibration acquisition dataset repository changed",
        )
        processing_repository = nonempty_string(
            self.processing_repository,
            name="processing_repository",
        )
        _require(
            processing_repository == DEFORM360_PROCESSING_REPOSITORY,
            "calibration acquisition processing repository changed",
        )
        digests = {
            "selection_artifact_sha256": sha256_digest(
                self.selection_artifact_sha256,
                name="selection_artifact_sha256",
            ),
            "content_selection_sha256": sha256_digest(
                self.content_selection_sha256,
                name="content_selection_sha256",
            ),
            "visual_provider_lock_id": sha256_digest(
                self.visual_provider_lock_id,
                name="visual_provider_lock_id",
            ),
        }
        revisions = {
            "dataset_revision": exact_revision(
                self.dataset_revision,
                name="dataset_revision",
            ),
            "processing_revision": exact_revision(
                self.processing_revision,
                name="processing_revision",
            ),
            "implementation_revision": exact_revision(
                self.implementation_revision,
                name="implementation_revision",
            ),
        }
        units = _calibration_units(self.calibration_units)
        calibration_ids = {unit.object_id for unit in units}
        forbidden = canonical_sorted_strings(
            self.forbidden_confirmation_object_ids,
            name="forbidden_confirmation_object_ids",
        )
        _require(
            len(forbidden)
            == 2 * DEFORM360_CONFIRMATION_OBJECTS_PER_STRATUM,
            "forbidden confirmation cohort must contain exactly 12 objects",
        )
        _require(
            calibration_ids.isdisjoint(forbidden),
            "calibration and forbidden confirmation objects overlap",
        )
        camera_policy = nonempty_string(self.camera_policy, name="camera_policy")
        tactile_policy = nonempty_string(self.tactile_policy, name="tactile_policy")
        audio_policy = nonempty_string(self.audio_policy, name="audio_policy")
        _require(
            camera_policy == "all-official-cameras-sorted-v1",
            "camera policy changed after provider freeze",
        )
        _require(
            tactile_policy == "all-exact-npy-timestamp-sensors-sorted-v1",
            "tactile policy changed after provider freeze",
        )
        _require(
            audio_policy == "exclude-wav-and-flac-v1",
            "audio exclusion policy changed",
        )
        calibration_opened = genuine_boolean(
            self.calibration_payloads_opened,
            name="calibration_payloads_opened",
        )
        confirmation_opened = genuine_boolean(
            self.confirmation_payloads_opened,
            name="confirmation_payloads_opened",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        _require(
            not calibration_opened,
            "acquisition plan must precede calibration access",
        )
        _require(
            not confirmation_opened,
            "acquisition plan opened confirmation payloads",
        )
        _require(not target_used, "acquisition plan used target outcomes")

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "dataset_repository", dataset_repository)
        object.__setattr__(self, "processing_repository", processing_repository)
        for name, value in {**digests, **revisions}.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "calibration_units", units)
        object.__setattr__(self, "forbidden_confirmation_object_ids", forbidden)
        object.__setattr__(self, "camera_policy", camera_policy)
        object.__setattr__(self, "tactile_policy", tactile_policy)
        object.__setattr__(self, "audio_policy", audio_policy)
        object.__setattr__(self, "calibration_payloads_opened", calibration_opened)
        object.__setattr__(self, "confirmation_payloads_opened", confirmation_opened)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="acquisition plan metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_ACQUISITION_PLAN_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_ACQUISITION_VERSION,
            "semantics": DEFORM360_CALIBRATION_ACQUISITION_SEMANTICS,
            "protocol_id": self.protocol_id,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "content_selection_sha256": self.content_selection_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "dataset_repository": self.dataset_repository,
            "dataset_revision": self.dataset_revision,
            "processing_repository": self.processing_repository,
            "processing_revision": self.processing_revision,
            "implementation_revision": self.implementation_revision,
            "calibration_units": [unit.to_record() for unit in self.calibration_units],
            "forbidden_confirmation_object_ids": (
                self.forbidden_confirmation_object_ids
            ),
            "camera_policy": self.camera_policy,
            "tactile_policy": self.tactile_policy,
            "audio_policy": self.audio_policy,
            "calibration_payloads_opened": self.calibration_payloads_opened,
            "confirmation_payloads_opened": self.confirmation_payloads_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY,
        }

    @property
    def plan_id(self) -> str:
        return content_id(self.descriptor())

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "plan_id": self.plan_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 calibration acquisition plan",
    ) -> Deform360CalibrationAcquisitionPlanV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_PLAN_FIELDS, name=name)
        if value["schema"] != DEFORM360_CALIBRATION_ACQUISITION_PLAN_SCHEMA:
            raise ValueError(f"{name} schema changed")
        if value["semantics"] != DEFORM360_CALIBRATION_ACQUISITION_SEMANTICS:
            raise ValueError(f"{name} semantics changed")
        if value["claim_boundary"] != DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY:
            raise ValueError(f"{name} claim boundary changed")
        if (
            genuine_integer(
                value["schema_version"],
                name=f"{name} schema_version",
                minimum=1,
            )
            != DEFORM360_CALIBRATION_ACQUISITION_VERSION
        ):
            raise ValueError(f"{name} schema_version changed")
        units_raw = value["calibration_units"]
        if not isinstance(units_raw, Sequence) or isinstance(units_raw, (str, bytes)):
            raise ValueError(f"{name} calibration_units must be a sequence")
        result = cls(
            selection_artifact_sha256=cast(
                str,
                value["selection_artifact_sha256"],
            ),
            content_selection_sha256=cast(str, value["content_selection_sha256"]),
            visual_provider_lock_id=cast(str, value["visual_provider_lock_id"]),
            dataset_revision=cast(str, value["dataset_revision"]),
            processing_revision=cast(str, value["processing_revision"]),
            implementation_revision=cast(str, value["implementation_revision"]),
            calibration_units=tuple(
                _unit_from_record(item, name=f"{name} calibration unit")
                for item in units_raw
            ),
            forbidden_confirmation_object_ids=cast(
                Sequence[str],
                value["forbidden_confirmation_object_ids"],
            ),
            camera_policy=cast(str, value["camera_policy"]),
            tactile_policy=cast(str, value["tactile_policy"]),
            audio_policy=cast(str, value["audio_policy"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            protocol_id=cast(str, value["protocol_id"]),
            dataset_repository=cast(str, value["dataset_repository"]),
            processing_repository=cast(str, value["processing_repository"]),
            calibration_payloads_opened=cast(
                bool,
                value["calibration_payloads_opened"],
            ),
            confirmation_payloads_opened=cast(
                bool,
                value["confirmation_payloads_opened"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
        )
        declared = sha256_digest(value["plan_id"], name=f"{name} plan_id")
        if declared != result.plan_id:
            raise ValueError(f"{name} plan_id does not match content")
        return result

def build_calibration_acquisition_plan(
    *,
    stage0_selection_path: str | Path,
    visual_provider_lock_path: str | Path,
    implementation_revision: str,
    protocol_path: str | Path | None = None,
) -> Deform360CalibrationAcquisitionPlanV1:
    """Build the exact calibration allowlist from committed target-blind locks."""

    stage0 = load_deform360_stage0_selection(
        stage0_selection_path,
        protocol_path=protocol_path,
    )
    provider = load_deform360_visual_provider_lock(visual_provider_lock_path)
    if provider.protocol_id != stage0.protocol_id:
        raise ValueError("visual-provider and Stage-0 protocols differ")
    if provider.selected_raw_payloads_opened:
        raise ValueError("visual-provider lock was created after raw payload access")
    if provider.target_outcomes_used:
        raise ValueError("visual-provider lock used target outcomes")
    return Deform360CalibrationAcquisitionPlanV1(
        selection_artifact_sha256=stage0.selection_artifact_sha256,
        content_selection_sha256=stage0.content_selection_sha256,
        visual_provider_lock_id=provider.artifact_id,
        dataset_revision=stage0.dataset_revision,
        processing_revision=stage0.processing_revision,
        implementation_revision=implementation_revision,
        calibration_units=stage0.calibration_units,
        forbidden_confirmation_object_ids=tuple(
            unit.object_id for unit in stage0.confirmation_units
        ),
        metadata={
            "stage0_snapshot_id": stage0.snapshot_id,
            "visual_provider_repository": provider.provider_repository,
            "visual_provider_revision": provider.provider_revision,
            "model_set_id": provider.model_set_id,
        },
    )
