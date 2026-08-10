"""Leave-one-group-out predictive covariance calibration record."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._domain_covariance_calibration_common import (
    DOMAIN_COVARIANCE_FOLD_SCHEMA,
    DOMAIN_COVARIANCE_FOLD_VERSION,
    _bounded_float,
    _canonical_string,
    _canonical_strings,
    _finite_float,
)
from ._portable_contracts import content_id, sha256_digest


@dataclass(frozen=True, slots=True)
class DomainCovarianceFoldV1:
    """One leave-one-group-out predictive calibration record."""

    domain_id: str
    held_out_group_id: str
    training_group_ids: Sequence[str]
    scale: float
    isotropic_floor_variance: float
    reference_variance: float
    raw_nll_per_dimension: float
    calibrated_nll_per_dimension: float
    log_loss_ratio: float
    loss_ratio: float
    raw_normalized_energy: float
    calibrated_normalized_energy: float
    raw_coordinate_coverage_90: float
    calibrated_coordinate_coverage_90: float
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        domain_id = _canonical_string(self.domain_id, name="domain_id")
        held_out = _canonical_string(
            self.held_out_group_id,
            name="held_out_group_id",
        )
        training = _canonical_strings(
            self.training_group_ids,
            name="training_group_ids",
        )
        if len(set(training)) != len(training):
            raise ValueError("training_group_ids must not contain duplicates")
        if held_out in training:
            raise ValueError("held-out group must not occur in training_group_ids")
        training = tuple(sorted(training))
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
            self.raw_nll_per_dimension,
            name="raw_nll_per_dimension",
        )
        calibrated_nll = _finite_float(
            self.calibrated_nll_per_dimension,
            name="calibrated_nll_per_dimension",
        )
        log_ratio = _finite_float(self.log_loss_ratio, name="log_loss_ratio")
        if not math.isclose(
            log_ratio,
            calibrated_nll - raw_nll,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("log_loss_ratio must equal calibrated NLL minus raw NLL")
        loss_ratio = _bounded_float(
            self.loss_ratio,
            name="loss_ratio",
            minimum=np.finfo(np.float64).tiny,
        )
        raw_energy = _bounded_float(
            self.raw_normalized_energy,
            name="raw_normalized_energy",
            minimum=0.0,
        )
        calibrated_energy = _bounded_float(
            self.calibrated_normalized_energy,
            name="calibrated_normalized_energy",
            minimum=0.0,
        )
        raw_coverage = _bounded_float(
            self.raw_coordinate_coverage_90,
            name="raw_coordinate_coverage_90",
            minimum=0.0,
            maximum=1.0,
        )
        calibrated_coverage = _bounded_float(
            self.calibrated_coordinate_coverage_90,
            name="calibrated_coordinate_coverage_90",
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "domain_id", domain_id)
        object.__setattr__(self, "held_out_group_id", held_out)
        object.__setattr__(self, "training_group_ids", training)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "isotropic_floor_variance", floor)
        object.__setattr__(self, "reference_variance", reference)
        object.__setattr__(self, "raw_nll_per_dimension", raw_nll)
        object.__setattr__(self, "calibrated_nll_per_dimension", calibrated_nll)
        object.__setattr__(self, "log_loss_ratio", log_ratio)
        object.__setattr__(self, "loss_ratio", loss_ratio)
        object.__setattr__(self, "raw_normalized_energy", raw_energy)
        object.__setattr__(
            self,
            "calibrated_normalized_energy",
            calibrated_energy,
        )
        object.__setattr__(self, "raw_coordinate_coverage_90", raw_coverage)
        object.__setattr__(
            self,
            "calibrated_coordinate_coverage_90",
            calibrated_coverage,
        )
        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match covariance fold")
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DOMAIN_COVARIANCE_FOLD_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_FOLD_VERSION,
            "domain_id": self.domain_id,
            "held_out_group_id": self.held_out_group_id,
            "training_group_ids": list(self.training_group_ids),
            "scale": self.scale,
            "isotropic_floor_variance": self.isotropic_floor_variance,
            "reference_variance": self.reference_variance,
            "raw_nll_per_dimension": self.raw_nll_per_dimension,
            "calibrated_nll_per_dimension": self.calibrated_nll_per_dimension,
            "log_loss_ratio": self.log_loss_ratio,
            "loss_ratio": self.loss_ratio,
            "raw_normalized_energy": self.raw_normalized_energy,
            "calibrated_normalized_energy": self.calibrated_normalized_energy,
            "raw_coordinate_coverage_90": self.raw_coordinate_coverage_90,
            "calibrated_coordinate_coverage_90": (
                self.calibrated_coordinate_coverage_90
            ),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}
