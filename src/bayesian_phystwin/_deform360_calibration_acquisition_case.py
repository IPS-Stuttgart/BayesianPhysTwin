# ruff: noqa: F403, F405
"""Internal implementation slice for Deform360 calibration acquisition."""

from __future__ import annotations

from ._deform360_calibration_acquisition_common import *

@dataclass(frozen=True)
class Deform360CalibrationAcquisitionCaseV1:
    """One selected calibration unit after official source preparation."""

    plan_id: str
    object_id: str
    episode_id: int
    stratum: Literal["sheet", "volumetric"]
    status: Deform360CalibrationAcquisitionStatus
    raw_factor_artifacts: Mapping[str, str]
    output_artifacts: Mapping[str, str] = field(default_factory=dict)
    aligned_frame_count: int | None = None
    camera_count: int = 0
    tactile_sensor_count: int = 0
    bimanual: bool | None = None
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    calibration_payloads_opened: bool = True
    confirmation_payloads_opened: bool = False
    target_outcomes_used: bool = False

    def __post_init__(self) -> None:
        plan_id = sha256_digest(self.plan_id, name="plan_id")
        object_id = nonempty_string(self.object_id, name="object_id")
        episode_id = genuine_integer(self.episode_id, name="episode_id", minimum=0)
        if type(self.stratum) is not str or self.stratum not in {"sheet", "volumetric"}:
            raise ValueError("stratum must be sheet or volumetric")
        stratum = cast(Literal["sheet", "volumetric"], self.stratum)
        if type(self.status) is not str or self.status not in {
            "prepared",
            "technical_failure",
        }:
            raise ValueError("status must be prepared or technical_failure")
        status = cast(Deform360CalibrationAcquisitionStatus, self.status)
        raw_factor_artifacts = source_artifact_mapping(
            self.raw_factor_artifacts,
            name="raw_factor_artifacts",
        )
        _require(raw_factor_artifacts, "raw_factor_artifacts must be nonempty")
        output_artifacts = source_artifact_mapping(
            self.output_artifacts,
            name="output_artifacts",
            allow_empty=True,
        )
        camera_count = genuine_integer(
            self.camera_count,
            name="camera_count",
            minimum=0,
        )
        tactile_count = genuine_integer(
            self.tactile_sensor_count,
            name="tactile_sensor_count",
            minimum=0,
        )
        bimanual = self.bimanual
        if bimanual is not None:
            bimanual = genuine_boolean(bimanual, name="bimanual")
        frame_count = self.aligned_frame_count
        if frame_count is not None:
            frame_count = genuine_integer(
                frame_count,
                name="aligned_frame_count",
                minimum=1,
            )
        failure_stage = self.failure_stage
        failure_type = self.failure_type
        failure_digest = self.failure_message_sha256
        if status == "prepared":
            _require(frame_count is not None, "prepared case needs aligned_frame_count")
            _require(camera_count >= 1, "prepared case needs at least one camera")
            _require(output_artifacts, "prepared case needs output_artifacts")
            _require(bimanual is not None, "prepared case needs bimanual status")
            _require(
                failure_stage is None
                and failure_type is None
                and failure_digest is None,
                "prepared case cannot carry failure fields",
            )
        else:
            failure_stage = nonempty_string(failure_stage, name="failure_stage")
            failure_type = nonempty_string(failure_type, name="failure_type")
            failure_digest = sha256_digest(
                failure_digest,
                name="failure_message_sha256",
            )
            _require(
                frame_count is None,
                "failed case cannot claim aligned_frame_count",
            )
            _require(not output_artifacts, "failed case cannot claim complete outputs")

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
        _require(calibration_opened, "case must record calibration payload access")
        _require(not confirmation_opened, "case opened confirmation payloads")
        _require(not target_used, "case used target outcomes")

        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "raw_factor_artifacts", raw_factor_artifacts)
        object.__setattr__(self, "output_artifacts", output_artifacts)
        object.__setattr__(self, "aligned_frame_count", frame_count)
        object.__setattr__(self, "camera_count", camera_count)
        object.__setattr__(self, "tactile_sensor_count", tactile_count)
        object.__setattr__(self, "bimanual", bimanual)
        object.__setattr__(self, "failure_stage", failure_stage)
        object.__setattr__(self, "failure_type", failure_type)
        object.__setattr__(self, "failure_message_sha256", failure_digest)
        object.__setattr__(self, "calibration_payloads_opened", calibration_opened)
        object.__setattr__(self, "confirmation_payloads_opened", confirmation_opened)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="acquisition case metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_ACQUISITION_CASE_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_ACQUISITION_VERSION,
            "semantics": DEFORM360_CALIBRATION_ACQUISITION_SEMANTICS,
            "plan_id": self.plan_id,
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "stratum": self.stratum,
            "status": self.status,
            "aligned_frame_count": self.aligned_frame_count,
            "camera_count": self.camera_count,
            "tactile_sensor_count": self.tactile_sensor_count,
            "bimanual": self.bimanual,
            "raw_factor_artifacts": self.raw_factor_artifacts,
            "output_artifacts": self.output_artifacts,
            "failure_stage": self.failure_stage,
            "failure_type": self.failure_type,
            "failure_message_sha256": self.failure_message_sha256,
            "calibration_payloads_opened": self.calibration_payloads_opened,
            "confirmation_payloads_opened": self.confirmation_payloads_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY,
        }

    @property
    def case_id(self) -> str:
        return content_id(self.descriptor())

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "case_id": self.case_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "Deform360 calibration acquisition case",
    ) -> Deform360CalibrationAcquisitionCaseV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        require_exact_fields(value, expected=_CASE_FIELDS, name=name)
        if value["schema"] != DEFORM360_CALIBRATION_ACQUISITION_CASE_SCHEMA:
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
        result = cls(
            plan_id=cast(str, value["plan_id"]),
            object_id=cast(str, value["object_id"]),
            episode_id=cast(int, value["episode_id"]),
            stratum=cast(Literal["sheet", "volumetric"], value["stratum"]),
            status=cast(Deform360CalibrationAcquisitionStatus, value["status"]),
            raw_factor_artifacts=cast(
                Mapping[str, str],
                value["raw_factor_artifacts"],
            ),
            output_artifacts=cast(Mapping[str, str], value["output_artifacts"]),
            aligned_frame_count=cast(int | None, value["aligned_frame_count"]),
            camera_count=cast(int, value["camera_count"]),
            tactile_sensor_count=cast(int, value["tactile_sensor_count"]),
            bimanual=cast(bool | None, value["bimanual"]),
            failure_stage=cast(str | None, value["failure_stage"]),
            failure_type=cast(str | None, value["failure_type"]),
            failure_message_sha256=cast(
                str | None,
                value["failure_message_sha256"],
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
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
        declared = sha256_digest(value["case_id"], name=f"{name} case_id")
        if declared != result.case_id:
            raise ValueError(f"{name} case_id does not match content")
        return result
