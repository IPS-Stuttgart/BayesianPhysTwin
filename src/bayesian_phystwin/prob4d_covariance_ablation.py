"""Controlled attribution of Prob4D covariance structure inside BayesianPhysTwin."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    literal_lower_hex,
    plain_json,
)
from .decisive_evidence import analyze_decisive_evidence, parse_decisive_evidence

PROB4D_COVARIANCE_ABLATION_SCHEMA: Final = (
    "bayesian_phystwin.prob4d_covariance_ablation"
)
PROB4D_COVARIANCE_ABLATION_VERSION: Final = 1
PROB4D_COVARIANCE_ABLATION_REPORT_SCHEMA: Final = (
    "bayesian_phystwin.prob4d_covariance_ablation_report"
)
PROB4D_COVARIANCE_ABLATION_REPORT_VERSION: Final = 1
PROB4D_COVARIANCE_ABLATION_BOUNDARY: Final = (
    "Controlled covariance-attribution diagnostic only. It verifies matched units, "
    "common observation means, physical linearizations, fallback and risk policies, "
    "calibration partitions, and software stacks. It does not by itself establish "
    "real-provider competence, calibrated uncertainty, physical-query benefit, "
    "Causal4D intervention benefit, deployment safety, or state of the art."
)
ALLOWED_VARIANT_DIFFERENCE: Final = "prob4d-covariance-treatment-only"

COVARIANCE_TREATMENTS: Final[tuple[str, ...]] = (
    "full_joint",
    "block_diagonal",
    "independent_rows",
    "shared_uncertainty_removed",
    "shared_uncertainty_underreported",
)

INVARIANT_DIGEST_FIELDS: Final[tuple[str, ...]] = (
    "observation_mean_sha256",
    "scored_units_sha256",
    "physical_linearization_sha256",
    "fallback_policy_sha256",
    "risk_policy_sha256",
    "calibration_partition_sha256",
    "software_stack_sha256",
)

_VARIANT_FIELDS: Final = frozenset(
    {
        "method",
        "treatment",
        "shared_uncertainty_scale",
        "gauge_factors_enabled",
        "run_manifest_sha256",
        "covariance_artifact_sha256",
        *INVARIANT_DIGEST_FIELDS,
    }
)
_TOP_LEVEL_REQUIRED: Final = frozenset(
    {
        "schema",
        "schema_version",
        "ablation_id",
        "reference_treatment",
        "locked_factors",
        "variants",
        "evidence",
    }
)
_TOP_LEVEL_OPTIONAL: Final = frozenset({"metadata"})
_LOCKED_FACTOR_KEYS: Final = frozenset(
    {
        "dataset_id",
        "split_id",
        "registered_statistical_unit",
        "source_or_calibration_policy_frozen",
        "allowed_variant_difference",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return cast(bool, value)


def _exact_fields(
    values: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    name: str,
) -> None:
    keys = frozenset(values)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{name} is missing fields {missing}")
    if unknown:
        raise ValueError(f"{name} contains unknown fields {unknown}")


def _digest(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CovarianceVariantV1:
    """One matched run in the fixed five-way covariance ablation."""

    method: str
    treatment: str
    shared_uncertainty_scale: float
    gauge_factors_enabled: bool
    run_manifest_sha256: str
    covariance_artifact_sha256: str
    observation_mean_sha256: str
    scored_units_sha256: str
    physical_linearization_sha256: str
    fallback_policy_sha256: str
    risk_policy_sha256: str
    calibration_partition_sha256: str
    software_stack_sha256: str

    def __post_init__(self) -> None:
        _text(self.method, name="method")
        if self.treatment not in COVARIANCE_TREATMENTS:
            raise ValueError(f"unsupported covariance treatment {self.treatment!r}")
        scale = _number(
            self.shared_uncertainty_scale,
            name="shared_uncertainty_scale",
            minimum=0.0,
            maximum=1.0,
        )
        _boolean(self.gauge_factors_enabled, name="gauge_factors_enabled")
        for field in fields(self):
            if field.name.endswith("_sha256"):
                _digest(getattr(self, field.name), name=field.name)
        if self.treatment == "full_joint":
            if scale != 1.0 or not self.gauge_factors_enabled:
                raise ValueError(
                    "full_joint requires scale 1 and enabled gauge factors"
                )
        elif self.treatment == "shared_uncertainty_underreported":
            if not 0.0 < scale < 1.0 or not self.gauge_factors_enabled:
                raise ValueError(
                    "shared_uncertainty_underreported requires a scale inside "
                    "(0, 1) and enabled gauge factors"
                )
        elif self.treatment == "block_diagonal":
            if scale != 1.0 or self.gauge_factors_enabled:
                raise ValueError(
                    "block_diagonal requires scale 1 and disabled gauge factors"
                )
        elif scale != 0.0 or self.gauge_factors_enabled:
            raise ValueError(
                f"{self.treatment} requires scale 0 and disabled gauge factors"
            )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        index: int,
    ) -> CovarianceVariantV1:
        name = f"variants[{index}]"
        mapping = _mapping(value, name=name)
        _exact_fields(mapping, required=_VARIANT_FIELDS, name=name)
        arguments: dict[str, object] = {
            "method": _text(mapping["method"], name=f"{name}.method"),
            "treatment": _text(mapping["treatment"], name=f"{name}.treatment"),
            "shared_uncertainty_scale": _number(
                mapping["shared_uncertainty_scale"],
                name=f"{name}.shared_uncertainty_scale",
                minimum=0.0,
                maximum=1.0,
            ),
            "gauge_factors_enabled": _boolean(
                mapping["gauge_factors_enabled"],
                name=f"{name}.gauge_factors_enabled",
            ),
        }
        for digest_name in (
            "run_manifest_sha256",
            "covariance_artifact_sha256",
            *INVARIANT_DIGEST_FIELDS,
        ):
            arguments[digest_name] = _digest(
                mapping[digest_name],
                name=f"{name}.{digest_name}",
            )
        return cls(**cast(dict[str, Any], arguments))

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class Prob4DCovarianceAblationV1:
    """Validated controlled ablation plus matched decisive-evidence records."""

    ablation_id: str
    reference_treatment: str
    locked_factors: Mapping[str, Any]
    variants: tuple[CovarianceVariantV1, ...]
    evidence: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def variants_by_treatment(self) -> dict[str, CovarianceVariantV1]:
        return {variant.treatment: variant for variant in self.variants}

    @classmethod
    def from_mapping(cls, value: object) -> Prob4DCovarianceAblationV1:
        payload = _mapping(value, name="payload")
        _exact_fields(
            payload,
            required=_TOP_LEVEL_REQUIRED,
            optional=_TOP_LEVEL_OPTIONAL,
            name="payload",
        )
        if payload["schema"] != PROB4D_COVARIANCE_ABLATION_SCHEMA:
            raise ValueError("unsupported schema")
        if isinstance(payload["schema_version"], bool) or (
            payload["schema_version"] != PROB4D_COVARIANCE_ABLATION_VERSION
        ):
            raise ValueError("schema_version must be the integer 1")
        ablation_id = _text(payload["ablation_id"], name="ablation_id")
        reference_treatment = _text(
            payload["reference_treatment"],
            name="reference_treatment",
        )
        if reference_treatment not in COVARIANCE_TREATMENTS:
            raise ValueError("reference_treatment is unsupported")
        if reference_treatment == "full_joint":
            raise ValueError("reference_treatment must be an ablated comparator")

        locked_factors = frozen_finite_json_mapping(
            _mapping(payload["locked_factors"], name="locked_factors"),
            name="locked_factors",
        )
        missing_locked = sorted(_LOCKED_FACTOR_KEYS - frozenset(locked_factors))
        if missing_locked:
            raise ValueError(f"locked_factors is missing fields {missing_locked}")
        _text(locked_factors["dataset_id"], name="locked_factors.dataset_id")
        _text(locked_factors["split_id"], name="locked_factors.split_id")
        registered_unit = _text(
            locked_factors["registered_statistical_unit"],
            name="locked_factors.registered_statistical_unit",
        )
        if locked_factors["source_or_calibration_policy_frozen"] is not True:
            raise ValueError(
                "locked_factors.source_or_calibration_policy_frozen must be true"
            )
        if locked_factors["allowed_variant_difference"] != ALLOWED_VARIANT_DIFFERENCE:
            raise ValueError(
                "locked_factors.allowed_variant_difference must permit only the "
                "Prob4D covariance treatment"
            )

        variants = tuple(
            CovarianceVariantV1.from_mapping(raw, index=index)
            for index, raw in enumerate(_sequence(payload["variants"], name="variants"))
        )
        if len(variants) != len(COVARIANCE_TREATMENTS):
            raise ValueError("variants must contain the complete five-way ablation")
        methods = [variant.method for variant in variants]
        treatments = [variant.treatment for variant in variants]
        if len(set(methods)) != len(methods):
            raise ValueError("variant method names must be unique")
        if set(treatments) != set(COVARIANCE_TREATMENTS):
            raise ValueError(
                "variants must contain each canonical covariance treatment exactly once"
            )
        for digest_name in INVARIANT_DIGEST_FIELDS:
            values = {getattr(variant, digest_name) for variant in variants}
            if len(values) != 1:
                raise ValueError(f"variants changed invariant digest {digest_name!r}")
        if len({variant.run_manifest_sha256 for variant in variants}) != len(variants):
            raise ValueError("each variant must bind a distinct run manifest")
        covariance_artifacts = {
            variant.covariance_artifact_sha256 for variant in variants
        }
        if len(covariance_artifacts) != len(variants):
            raise ValueError("each variant must bind a distinct covariance artifact")

        evidence = frozen_finite_json_mapping(
            _mapping(payload["evidence"], name="evidence"),
            name="evidence",
        )
        bundle = parse_decisive_evidence(evidence)
        if bundle.protocol_id != ablation_id:
            raise ValueError("evidence.protocol_id must equal ablation_id")
        if bundle.statistical_unit != registered_unit:
            raise ValueError(
                "evidence statistical_unit differs from the registered locked factor"
            )
        variant_methods = frozenset(methods)
        evidence_methods = frozenset(record.method for record in bundle.records)
        if evidence_methods != variant_methods:
            raise ValueError(
                "evidence methods must exactly match the covariance variants"
            )
        reference_method = next(
            variant.method
            for variant in variants
            if variant.treatment == reference_treatment
        )
        if (
            bundle.reference_method is not None
            and bundle.reference_method != reference_method
        ):
            raise ValueError(
                "evidence reference_method differs from reference_treatment"
            )
        metadata = frozen_finite_json_mapping(
            cast(Mapping[str, Any] | None, payload.get("metadata")),
            name="metadata",
        )
        return cls(
            ablation_id=ablation_id,
            reference_treatment=reference_treatment,
            locked_factors=locked_factors,
            variants=variants,
            evidence=evidence,
            metadata=metadata,
        )

    def canonical_input(self) -> dict[str, object]:
        order = {name: index for index, name in enumerate(COVARIANCE_TREATMENTS)}
        return {
            "schema": PROB4D_COVARIANCE_ABLATION_SCHEMA,
            "schema_version": PROB4D_COVARIANCE_ABLATION_VERSION,
            "ablation_id": self.ablation_id,
            "reference_treatment": self.reference_treatment,
            "locked_factors": plain_json(self.locked_factors),
            "variants": [
                variant.to_dict()
                for variant in sorted(
                    self.variants,
                    key=lambda item: order[item.treatment],
                )
            ],
            "evidence": plain_json(self.evidence),
            "metadata": plain_json(self.metadata),
        }


def _full_joint_pairwise_attribution(
    ablation: Prob4DCovarianceAblationV1,
) -> dict[str, object]:
    variants = ablation.variants_by_treatment
    full_joint = variants["full_joint"]
    comparisons: dict[str, object] = {}
    for treatment in COVARIANCE_TREATMENTS:
        if treatment == "full_joint":
            continue
        comparator = variants[treatment]
        analysis = analyze_decisive_evidence(
            ablation.evidence,
            reference_method=comparator.method,
        )
        metrics = cast(Mapping[str, Any], analysis["metrics"])
        metric_results: dict[str, object] = {}
        for metric, metric_summary in sorted(metrics.items()):
            methods = cast(Mapping[str, Any], metric_summary["methods"])
            full_joint_summary = cast(Mapping[str, Any], methods[full_joint.method])
            metric_results[metric] = {
                "raw": plain_json(full_joint_summary["raw_vs_reference_method"]),
                "operational": plain_json(
                    full_joint_summary["operational_vs_reference_method"]
                ),
            }
        comparisons[treatment] = {
            "comparator_method": comparator.method,
            "shared_uncertainty_scale": comparator.shared_uncertainty_scale,
            "gauge_factors_enabled": comparator.gauge_factors_enabled,
            "metrics": metric_results,
        }
    return {
        "full_joint_method": full_joint.method,
        "comparisons": comparisons,
    }


def analyze_prob4d_covariance_ablation(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate the controlled ablation and build paired attribution evidence."""

    ablation = Prob4DCovarianceAblationV1.from_mapping(payload)
    variants = ablation.variants_by_treatment
    reference_method = variants[ablation.reference_treatment].method
    canonical_input = ablation.canonical_input()
    input_sha256 = _canonical_json_sha256(canonical_input)
    locked_factors_sha256 = _canonical_json_sha256(
        cast(Mapping[str, Any], plain_json(ablation.locked_factors))
    )
    evidence_sha256 = _canonical_json_sha256(
        cast(Mapping[str, Any], plain_json(ablation.evidence))
    )
    invariant_digests = {
        digest_name: getattr(ablation.variants[0], digest_name)
        for digest_name in INVARIANT_DIGEST_FIELDS
    }
    report: dict[str, object] = {
        "schema": PROB4D_COVARIANCE_ABLATION_REPORT_SCHEMA,
        "schema_version": PROB4D_COVARIANCE_ABLATION_REPORT_VERSION,
        "source_schema": PROB4D_COVARIANCE_ABLATION_SCHEMA,
        "source_schema_version": PROB4D_COVARIANCE_ABLATION_VERSION,
        "ablation_id": ablation.ablation_id,
        "reference_treatment": ablation.reference_treatment,
        "reference_method": reference_method,
        "complete_five_way_ablation": True,
        "only_covariance_treatment_varied": True,
        "matched_fallback_required": True,
        "source_or_calibration_policy_frozen": True,
        "claim_authorized": False,
        "scientific_boundary": PROB4D_COVARIANCE_ABLATION_BOUNDARY,
        "input_content_sha256": input_sha256,
        "locked_factors_sha256": locked_factors_sha256,
        "evidence_content_sha256": evidence_sha256,
        "locked_factors": plain_json(ablation.locked_factors),
        "invariant_digests": invariant_digests,
        "variants": {
            treatment: variants[treatment].to_dict()
            for treatment in COVARIANCE_TREATMENTS
        },
        "decisive_evidence_summary": analyze_decisive_evidence(
            ablation.evidence,
            reference_method=reference_method,
        ),
        "full_joint_attribution": _full_joint_pairwise_attribution(ablation),
        "metadata": plain_json(ablation.metadata),
    }
    report["report_id"] = _canonical_json_sha256(report)
    return report


__all__ = [
    "ALLOWED_VARIANT_DIFFERENCE",
    "COVARIANCE_TREATMENTS",
    "INVARIANT_DIGEST_FIELDS",
    "PROB4D_COVARIANCE_ABLATION_BOUNDARY",
    "PROB4D_COVARIANCE_ABLATION_REPORT_SCHEMA",
    "PROB4D_COVARIANCE_ABLATION_REPORT_VERSION",
    "PROB4D_COVARIANCE_ABLATION_SCHEMA",
    "PROB4D_COVARIANCE_ABLATION_VERSION",
    "CovarianceVariantV1",
    "Prob4DCovarianceAblationV1",
    "analyze_prob4d_covariance_ablation",
]
