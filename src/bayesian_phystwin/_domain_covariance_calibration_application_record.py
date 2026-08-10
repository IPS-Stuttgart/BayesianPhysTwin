"""Content-addressed record for one covariance apply-or-fallback decision."""

from __future__ import annotations

from dataclasses import dataclass

from ._canonical_contracts import genuine_boolean
from ._domain_covariance_calibration_common import (
    DOMAIN_COVARIANCE_APPLICATION_SCHEMA,
    DOMAIN_COVARIANCE_APPLICATION_VERSION,
    _canonical_string,
)
from ._portable_contracts import content_id, sha256_digest


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationApplicationV1:
    """Content-addressed record of one apply-or-exact-fallback decision."""

    certificate_id: str
    domain_id: str
    domain_decision_id: str | None
    transform_id: str | None
    input_covariance_id: str
    output_covariance_id: str
    calibration_applied: bool
    reason: str
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        certificate_id = sha256_digest(self.certificate_id, name="certificate_id")
        domain_id = _canonical_string(self.domain_id, name="domain_id")
        decision_id = self.domain_decision_id
        if decision_id is not None:
            decision_id = sha256_digest(decision_id, name="domain_decision_id")
        transform_id = self.transform_id
        if transform_id is not None:
            transform_id = sha256_digest(transform_id, name="transform_id")
        input_id = sha256_digest(
            self.input_covariance_id,
            name="input_covariance_id",
        )
        output_id = sha256_digest(
            self.output_covariance_id,
            name="output_covariance_id",
        )
        applied = genuine_boolean(
            self.calibration_applied,
            name="calibration_applied",
        )
        reason = _canonical_string(self.reason, name="reason")
        if applied:
            if decision_id is None or transform_id is None:
                raise ValueError(
                    "applied calibration requires decision and transform IDs"
                )
        elif output_id != input_id:
            raise ValueError("rejected calibration must preserve covariance ID")
        object.__setattr__(self, "certificate_id", certificate_id)
        object.__setattr__(self, "domain_id", domain_id)
        object.__setattr__(self, "domain_decision_id", decision_id)
        object.__setattr__(self, "transform_id", transform_id)
        object.__setattr__(self, "input_covariance_id", input_id)
        object.__setattr__(self, "output_covariance_id", output_id)
        object.__setattr__(self, "calibration_applied", applied)
        object.__setattr__(self, "reason", reason)
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match covariance application")
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DOMAIN_COVARIANCE_APPLICATION_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_APPLICATION_VERSION,
            "certificate_id": self.certificate_id,
            "domain_id": self.domain_id,
            "domain_decision_id": self.domain_decision_id,
            "transform_id": self.transform_id,
            "input_covariance_id": self.input_covariance_id,
            "output_covariance_id": self.output_covariance_id,
            "calibration_applied": self.calibration_applied,
            "reason": self.reason,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
