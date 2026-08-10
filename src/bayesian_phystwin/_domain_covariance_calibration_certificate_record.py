"""Content-addressed domain covariance calibration certificate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._canonical_contracts import plain_json
from ._domain_covariance_calibration_certificate_validation import (
    normalize_certificate_fields,
    validate_guard_binding,
)
from ._domain_covariance_calibration_common import (
    DOMAIN_COVARIANCE_CALIBRATION_SCHEMA,
    DOMAIN_COVARIANCE_CALIBRATION_VERSION,
    DomainCovarianceCalibrationConfigV1,
    _canonical_string,
)
from ._domain_covariance_calibration_fold import DomainCovarianceFoldV1
from ._domain_covariance_calibration_transform import DomainCovarianceTransformV1
from ._portable_contracts import content_id, sha256_digest
from .calibration_domain_guard import CalibrationDomainGuardCertificateV1


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationCertificateV1:
    """Content-addressed covariance transforms plus cross-fitted domain guard."""

    calibration_partition_id: str
    statistical_unit: str
    residual_definition: str
    covariance_definition: str
    dimension: int
    config: DomainCovarianceCalibrationConfigV1
    calibration_data_id: str
    transforms: Sequence[DomainCovarianceTransformV1]
    fold_records: Sequence[DomainCovarianceFoldV1]
    guard_certificate: CalibrationDomainGuardCertificateV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        normalized = normalize_certificate_fields(self)
        names = (
            "calibration_partition_id",
            "statistical_unit",
            "residual_definition",
            "covariance_definition",
            "dimension",
            "calibration_data_id",
            "transforms",
            "fold_records",
            "metadata",
        )
        for name, value in zip(names, normalized, strict=True):
            object.__setattr__(self, name, value)
        validate_guard_binding(self)
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError(
                    "artifact_id does not match domain covariance calibration"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def deployment_admissible(self) -> bool:
        return self.guard_certificate.deployment_admissible

    @property
    def supported_domains(self) -> tuple[str, ...]:
        return self.guard_certificate.supported_domains

    def transform_for_domain(
        self,
        domain_id: str,
    ) -> DomainCovarianceTransformV1 | None:
        canonical = _canonical_string(domain_id, name="domain_id")
        return next(
            (
                transform
                for transform in self.transforms
                if transform.domain_id == canonical
            ),
            None,
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DOMAIN_COVARIANCE_CALIBRATION_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_CALIBRATION_VERSION,
            "calibration_partition_id": self.calibration_partition_id,
            "statistical_unit": self.statistical_unit,
            "residual_definition": self.residual_definition,
            "covariance_definition": self.covariance_definition,
            "dimension": self.dimension,
            "config": self.config.descriptor(),
            "calibration_data_id": self.calibration_data_id,
            "transforms": [item.to_record() for item in self.transforms],
            "fold_records": [item.to_record() for item in self.fold_records],
            "guard_certificate": self.guard_certificate.to_record(),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
