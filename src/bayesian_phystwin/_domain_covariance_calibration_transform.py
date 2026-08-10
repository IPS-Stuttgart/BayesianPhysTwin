"""Final domain covariance transform record."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._canonical_contracts import genuine_integer
from ._domain_covariance_calibration_common import (
    DOMAIN_COVARIANCE_TRANSFORM_SCHEMA,
    DOMAIN_COVARIANCE_TRANSFORM_VERSION,
    _bounded_float,
    _canonical_string,
    _canonical_strings,
    _finite_float,
)
from ._portable_contracts import content_id, sha256_digest

@dataclass(frozen=True, slots=True)
class DomainCovarianceTransformV1:
    """One final all-calibration-groups transform for a declared domain."""

    domain_id: str
    dimension: int
    group_ids: Sequence[str]
    scale: float
    isotropic_floor_variance: float
    reference_variance: float
    raw_group_balanced_nll_per_dimension: float
    calibrated_group_balanced_nll_per_dimension: float
    raw_mean_normalized_energy: float
    calibrated_mean_normalized_energy: float
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        domain_id = _canonical_string(self.domain_id, name="domain_id")
        dimension = genuine_integer(
            self.dimension,
            name="dimension",
            minimum=1,
        )
        groups = _canonical_strings(self.group_ids, name="group_ids")
        if len(set(groups)) != len(groups):
            raise ValueError("group_ids must not contain duplicates")
        groups = tuple(sorted(groups))
        scale = _bounded_float(self.scale, name="scale", minimum=1.0)
        floor = _bounded_float(
            self.isotropic_floor_variance,
            name="isotropic_floor_variance",
            minimum=0.0,
        )
        reference = _bounded_float(
            self.reference_variance,
            name="reference_variance",
            minimum=np.finfo(np.float64).tiny,
        )
        raw_nll = _finite_float(
            self.raw_group_balanced_nll_per_dimension,
            name="raw_group_balanced_nll_per_dimension",
        )
        calibrated_nll = _finite_float(
            self.calibrated_group_balanced_nll_per_dimension,
            name="calibrated_group_balanced_nll_per_dimension",
        )
        raw_energy = _bounded_float(
            self.raw_mean_normalized_energy,
            name="raw_mean_normalized_energy",
            minimum=0.0,
        )
        calibrated_energy = _bounded_float(
            self.calibrated_mean_normalized_energy,
            name="calibrated_mean_normalized_energy",
            minimum=0.0,
        )
        object.__setattr__(self, "domain_id", domain_id)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "isotropic_floor_variance", floor)
        object.__setattr__(self, "reference_variance", reference)
        object.__setattr__(
            self,
            "raw_group_balanced_nll_per_dimension",
            raw_nll,
        )
        object.__setattr__(
            self,
            "calibrated_group_balanced_nll_per_dimension",
            calibrated_nll,
        )
        object.__setattr__(self, "raw_mean_normalized_energy", raw_energy)
        object.__setattr__(
            self,
            "calibrated_mean_normalized_energy",
            calibrated_energy,
        )
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match covariance transform")
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DOMAIN_COVARIANCE_TRANSFORM_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_TRANSFORM_VERSION,
            "domain_id": self.domain_id,
            "dimension": self.dimension,
            "group_ids": list(self.group_ids),
            "scale": self.scale,
            "isotropic_floor_variance": self.isotropic_floor_variance,
            "reference_variance": self.reference_variance,
            "raw_group_balanced_nll_per_dimension": (
                self.raw_group_balanced_nll_per_dimension
            ),
            "calibrated_group_balanced_nll_per_dimension": (
                self.calibrated_group_balanced_nll_per_dimension
            ),
            "raw_mean_normalized_energy": self.raw_mean_normalized_energy,
            "calibrated_mean_normalized_energy": (
                self.calibrated_mean_normalized_energy
            ),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
