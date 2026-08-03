"""Fail-closed freeze and decision contracts for Prob4D-to-BPT evidence.

The contract separates observation-provider competence from guarded physical
prediction.  It freezes disjoint development, calibration, and target groups;
software and calibration identities; the complete method matrix; exact fallback;
and all quantitative gate criteria before any target outcome is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, cast

PROTOCOL_SCHEMA: Final = "bayesian_phystwin.prob4d-prospective-protocol"
PROTOCOL_VERSION: Final = 1
READINESS_SCHEMA: Final = "bayesian_phystwin.prob4d-prospective-readiness"
READINESS_VERSION: Final = 1
RESULT_SCHEMA: Final = "bayesian_phystwin.prob4d-prospective-result"
RESULT_VERSION: Final = 1
DECISION_SCHEMA: Final = "bayesian_phystwin.prob4d-prospective-decision"
DECISION_VERSION: Final = 1
CLAIM_ID: Final = "prob4d.improves_bayesian_physical_twin"

Stage = Literal["development", "calibration", "target"]
CriterionStage = Literal["provider", "physical"]
Comparison = Literal["<=", ">="]
Statistic = Literal["estimate", "ci_lower", "ci_upper"]

REQUIRED_METHODS: Final = frozenset(
    {
        "physical_baseline",
        "simple_visual",
        "prob4d_fused_gauge_marginalized",
        "prob4d_framewise_joint_gauge",
        "prob4d_tracklet_joint_gauge",
    }
)
PROB4D_INTERFACES: Final = frozenset(
    {"fused", "framewise_factors", "tracklet_factors"}
)
ALLOWED_ROLES: Final = frozenset(
    {"physical_baseline", "visual_reference", "prob4d_candidate"}
)
ALLOWED_INTERFACES: Final = frozenset(
    {"none", "simple_visual", *PROB4D_INTERFACES}
)
ALLOWED_GAUGE_TREATMENTS: Final = frozenset(
    {"none", "fixed", "marginalized", "explicit_joint_nuisance"}
)
ALLOWED_SEED_POLICIES: Final = frozenset({"legacy-common", "derived-per-call"})
ALLOWED_ARTIFACT_STAGES: Final = frozenset(
    {"development", "calibration", "source_only"}
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any], *, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be Boolean")
    return value


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_revision(value: object, *, name: str) -> str:
    revision = _require_string(value, name=name)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact 40- or 64-character revision")
    return revision


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _safe_relative_path(value: object, *, name: str) -> str:
    path_text = _require_string(value, name=name)
    if "\\" in path_text:
        raise ValueError(f"{name} must be a POSIX relative path")
    path = PurePosixPath(path_text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe relative path")
    return path.as_posix()


def _validate_timestamp(value: object, *, name: str) -> str:
    text = _require_string(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return text


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ProtocolUnitV1:
    """One indivisible development, calibration, or target unit."""

    unit_id: str
    group_id: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, name: str) -> ProtocolUnitV1:
        _require_exact_fields(
            value,
            frozenset({"unit_id", "group_id"}),
            name=name,
        )
        return cls(
            unit_id=_require_string(value.get("unit_id"), name=f"{name}.unit_id"),
            group_id=_require_string(
                value.get("group_id"),
                name=f"{name}.group_id",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {"unit_id": self.unit_id, "group_id": self.group_id}


@dataclass(frozen=True, slots=True)
class ProtocolMethodV1:
    """One predeclared observation or physical-baseline interface."""

    method_id: str
    role: str
    observation_interface: str
    gauge_treatment: str
    sensor_assisted: bool
    exact_fallback: bool

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        name: str,
    ) -> ProtocolMethodV1:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "method_id",
                    "role",
                    "observation_interface",
                    "gauge_treatment",
                    "sensor_assisted",
                    "exact_fallback",
                }
            ),
            name=name,
        )
        method = cls(
            method_id=_require_string(
                value.get("method_id"),
                name=f"{name}.method_id",
            ),
            role=_require_string(value.get("role"), name=f"{name}.role"),
            observation_interface=_require_string(
                value.get("observation_interface"),
                name=f"{name}.observation_interface",
            ),
            gauge_treatment=_require_string(
                value.get("gauge_treatment"),
                name=f"{name}.gauge_treatment",
            ),
            sensor_assisted=_require_bool(
                value.get("sensor_assisted"),
                name=f"{name}.sensor_assisted",
            ),
            exact_fallback=_require_bool(
                value.get("exact_fallback"),
                name=f"{name}.exact_fallback",
            ),
        )
        method._validate(name=name)
        return method

    def _validate(self, *, name: str) -> None:
        if self.role not in ALLOWED_ROLES:
            raise ValueError(f"{name}.role is unsupported")
        if self.observation_interface not in ALLOWED_INTERFACES:
            raise ValueError(f"{name}.observation_interface is unsupported")
        if self.gauge_treatment not in ALLOWED_GAUGE_TREATMENTS:
            raise ValueError(f"{name}.gauge_treatment is unsupported")
        if self.role == "physical_baseline":
            if self.observation_interface != "none" or self.gauge_treatment != "none":
                raise ValueError("physical baseline cannot consume observations")
            if not self.exact_fallback or self.sensor_assisted:
                raise ValueError("physical baseline must be the unassisted exact fallback")
        elif self.role == "visual_reference":
            if self.observation_interface != "simple_visual":
                raise ValueError("visual reference must use simple_visual")
            if self.exact_fallback:
                raise ValueError("visual reference cannot be the exact fallback")
        else:
            if self.observation_interface not in PROB4D_INTERFACES:
                raise ValueError("Prob4D candidate must use a Prob4D interface")
            if self.exact_fallback:
                raise ValueError("Prob4D candidate cannot be the exact fallback")
            if (
                self.observation_interface == "fused"
                and self.gauge_treatment != "marginalized"
            ):
                raise ValueError("fused Prob4D candidate must marginalize gauge")
            if (
                self.observation_interface in {"framewise_factors", "tracklet_factors"}
                and self.gauge_treatment != "explicit_joint_nuisance"
            ):
                raise ValueError("unfused Prob4D factors require explicit joint gauge")

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "role": self.role,
            "observation_interface": self.observation_interface,
            "gauge_treatment": self.gauge_treatment,
            "sensor_assisted": self.sensor_assisted,
            "exact_fallback": self.exact_fallback,
        }


@dataclass(frozen=True, slots=True)
class GateCriterionV1:
    """One frozen scalar decision rule on a paired evaluation statistic."""

    criterion_id: str
    stage: CriterionStage
    method_id: str
    reference_method_id: str
    metric: str
    statistic: Statistic
    comparison: Comparison
    threshold: float

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        name: str,
    ) -> GateCriterionV1:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "criterion_id",
                    "stage",
                    "method_id",
                    "reference_method_id",
                    "metric",
                    "statistic",
                    "comparison",
                    "threshold",
                }
            ),
            name=name,
        )
        stage = _require_string(value.get("stage"), name=f"{name}.stage")
        statistic = _require_string(
            value.get("statistic"),
            name=f"{name}.statistic",
        )
        comparison = _require_string(
            value.get("comparison"),
            name=f"{name}.comparison",
        )
        if stage not in {"provider", "physical"}:
            raise ValueError(f"{name}.stage is unsupported")
        if statistic not in {"estimate", "ci_lower", "ci_upper"}:
            raise ValueError(f"{name}.statistic is unsupported")
        if comparison not in {"<=", ">="}:
            raise ValueError(f"{name}.comparison is unsupported")
        return cls(
            criterion_id=_require_string(
                value.get("criterion_id"),
                name=f"{name}.criterion_id",
            ),
            stage=cast(CriterionStage, stage),
            method_id=_require_string(
                value.get("method_id"),
                name=f"{name}.method_id",
            ),
            reference_method_id=_require_string(
                value.get("reference_method_id"),
                name=f"{name}.reference_method_id",
            ),
            metric=_require_string(value.get("metric"), name=f"{name}.metric"),
            statistic=cast(Statistic, statistic),
            comparison=cast(Comparison, comparison),
            threshold=_require_finite(
                value.get("threshold"),
                name=f"{name}.threshold",
            ),
        )

    def passes(self, value: float) -> bool:
        if self.comparison == "<=":
            return value <= self.threshold
        return value >= self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "criterion_id": self.criterion_id,
            "stage": self.stage,
            "method_id": self.method_id,
            "reference_method_id": self.reference_method_id,
            "metric": self.metric,
            "statistic": self.statistic,
            "comparison": self.comparison,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class FrozenArtifactV1:
    """One source-side file bound into the pre-target freeze."""

    role: str
    path: str
    sha256: str
    access_stage: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        name: str,
    ) -> FrozenArtifactV1:
        _require_exact_fields(
            value,
            frozenset({"role", "path", "sha256", "access_stage"}),
            name=name,
        )
        stage = _require_string(
            value.get("access_stage"),
            name=f"{name}.access_stage",
        )
        if stage not in ALLOWED_ARTIFACT_STAGES:
            raise ValueError("target artifacts cannot be bound before target opening")
        role = _require_string(value.get("role"), name=f"{name}.role")
        if "target" in role.casefold() or "outcome" in role.casefold():
            raise ValueError("frozen artifact role cannot contain target outcomes")
        return cls(
            role=role,
            path=_safe_relative_path(value.get("path"), name=f"{name}.path"),
            sha256=_require_sha256(value.get("sha256"), name=f"{name}.sha256"),
            access_stage=stage,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "access_stage": self.access_stage,
        }


@dataclass(frozen=True, slots=True)
class Prob4DProspectiveProtocolV1:
    """Immutable, content-addressed pre-target experimental contract."""

    protocol_id: str
    frozen_at: str
    frozen_by: str
    split: Mapping[str, tuple[ProtocolUnitV1, ...]]
    methods: tuple[ProtocolMethodV1, ...]
    software: Mapping[str, object]
    calibration: Mapping[str, object]
    analysis: Mapping[str, object]
    criteria: tuple[GateCriterionV1, ...]
    frozen_artifacts: tuple[FrozenArtifactV1, ...]
    protocol_sha256: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> Prob4DProspectiveProtocolV1:
        expected = frozenset(
            {
                "schema_name",
                "schema_version",
                "protocol_id",
                "claim_id",
                "frozen_at",
                "frozen_by",
                "split",
                "methods",
                "software",
                "calibration",
                "analysis",
                "criteria",
                "frozen_artifacts",
                "protocol_sha256",
            }
        )
        _require_exact_fields(value, expected, name="Prob4D prospective protocol")
        if value.get("schema_name") != PROTOCOL_SCHEMA:
            raise ValueError("unsupported Prob4D prospective protocol schema")
        if value.get("schema_version") != PROTOCOL_VERSION:
            raise ValueError("unsupported Prob4D prospective protocol version")
        if value.get("claim_id") != CLAIM_ID:
            raise ValueError("Prob4D prospective protocol claim changed")
        declared_id = _require_sha256(
            value.get("protocol_sha256"),
            name="protocol_sha256",
        )
        if _content_id(value, field="protocol_sha256") != declared_id:
            raise ValueError("Prob4D prospective protocol content address mismatch")

        split_value = _require_mapping(value.get("split"), name="split")
        _require_exact_fields(
            split_value,
            frozenset({"development", "calibration", "target"}),
            name="split",
        )
        split: dict[str, tuple[ProtocolUnitV1, ...]] = {}
        for stage in ("development", "calibration", "target"):
            units = _require_list(split_value.get(stage), name=f"split.{stage}")
            if not units:
                raise ValueError(f"split.{stage} must not be empty")
            split[stage] = tuple(
                ProtocolUnitV1.from_dict(
                    _require_mapping(unit, name=f"split.{stage}[{index}]"),
                    name=f"split.{stage}[{index}]",
                )
                for index, unit in enumerate(units)
            )

        methods_value = _require_list(value.get("methods"), name="methods")
        methods = tuple(
            ProtocolMethodV1.from_dict(
                _require_mapping(item, name=f"methods[{index}]"),
                name=f"methods[{index}]",
            )
            for index, item in enumerate(methods_value)
        )
        criteria_value = _require_list(value.get("criteria"), name="criteria")
        criteria = tuple(
            GateCriterionV1.from_dict(
                _require_mapping(item, name=f"criteria[{index}]"),
                name=f"criteria[{index}]",
            )
            for index, item in enumerate(criteria_value)
        )
        artifacts_value = _require_list(
            value.get("frozen_artifacts"),
            name="frozen_artifacts",
        )
        artifacts = tuple(
            FrozenArtifactV1.from_dict(
                _require_mapping(item, name=f"frozen_artifacts[{index}]"),
                name=f"frozen_artifacts[{index}]",
            )
            for index, item in enumerate(artifacts_value)
        )
        instance = cls(
            protocol_id=_require_string(
                value.get("protocol_id"),
                name="protocol_id",
            ),
            frozen_at=_validate_timestamp(value.get("frozen_at"), name="frozen_at"),
            frozen_by=_require_string(value.get("frozen_by"), name="frozen_by"),
            split=MappingProxyType(split),
            methods=methods,
            software=cast(
                Mapping[str, object],
                _freeze_json(
                    dict(_require_mapping(value.get("software"), name="software"))
                ),
            ),
            calibration=cast(
                Mapping[str, object],
                _freeze_json(
                    dict(
                        _require_mapping(
                            value.get("calibration"),
                            name="calibration",
                        )
                    )
                ),
            ),
            analysis=cast(
                Mapping[str, object],
                _freeze_json(
                    dict(_require_mapping(value.get("analysis"), name="analysis"))
                ),
            ),
            criteria=criteria,
            frozen_artifacts=artifacts,
            protocol_sha256=declared_id,
        )
        instance._validate()
        return instance

    def _validate(self) -> None:
        unit_ids: dict[str, str] = {}
        group_ids: dict[str, str] = {}
        for stage, units in self.split.items():
            for unit in units:
                previous = unit_ids.setdefault(unit.unit_id, stage)
                if previous != stage:
                    raise ValueError("unit IDs must be disjoint across split stages")
                previous_group = group_ids.setdefault(unit.group_id, stage)
                if previous_group != stage:
                    raise ValueError("group IDs must be disjoint across split stages")
        method_ids = [method.method_id for method in self.methods]
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("protocol method IDs must be unique")
        if not REQUIRED_METHODS.issubset(method_ids):
            missing = sorted(REQUIRED_METHODS - set(method_ids))
            raise ValueError(f"protocol method matrix is incomplete: {missing}")
        fallback = [method for method in self.methods if method.exact_fallback]
        if len(fallback) != 1 or fallback[0].method_id != "physical_baseline":
            raise ValueError("physical_baseline must be the sole exact fallback")
        by_method = {method.method_id: method for method in self.methods}

        self._validate_software()
        self._validate_calibration()
        primary = self._validate_analysis(by_method)

        criterion_ids = [criterion.criterion_id for criterion in self.criteria]
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("gate criterion IDs must be unique")
        if not self.criteria:
            raise ValueError("protocol must define provider and physical gates")
        stages = {criterion.stage for criterion in self.criteria}
        if stages != {"provider", "physical"}:
            raise ValueError("protocol requires separate provider and physical criteria")
        for criterion in self.criteria:
            if criterion.method_id not in by_method:
                raise ValueError("gate criterion references an unknown method")
            if criterion.reference_method_id not in by_method:
                raise ValueError("gate criterion references an unknown reference")
            if criterion.method_id == criterion.reference_method_id:
                raise ValueError("gate criterion method and reference must differ")
            expected_reference = (
                "simple_visual" if criterion.stage == "provider" else "physical_baseline"
            )
            if criterion.reference_method_id != expected_reference:
                raise ValueError(
                    f"{criterion.stage} criterion has the wrong reference method"
                )
            if criterion.method_id not in primary:
                raise ValueError("gate criteria may select only frozen primary candidates")
        for method_id in primary:
            method_stages = {
                criterion.stage
                for criterion in self.criteria
                if criterion.method_id == method_id
            }
            if method_stages != {"provider", "physical"}:
                raise ValueError(
                    "every primary candidate requires provider and physical criteria"
                )

        roles = [artifact.role for artifact in self.frozen_artifacts]
        paths = [artifact.path for artifact in self.frozen_artifacts]
        if len(set(roles)) != len(roles):
            raise ValueError("frozen artifact roles must be unique")
        if len(set(paths)) != len(paths):
            raise ValueError("frozen artifact paths must be unique")
        required_roles = {
            "provider_evaluation_manifest",
            "analysis_manifest",
            "method_freeze",
        }
        if not required_roles.issubset(roles):
            missing_roles = sorted(required_roles - set(roles))
            raise ValueError(f"frozen artifacts are incomplete: {missing_roles}")

    def _validate_software(self) -> None:
        expected = frozenset(
            {
                "prob4d_revision",
                "prob4d_wheel_sha256",
                "bayesian_phystwin_revision",
                "bayesian_phystwin_wheel_sha256",
                "motioncrafter_revision",
                "motioncrafter_model_set_id",
                "seed_policy",
                "python_version",
                "numpy_version",
            }
        )
        _require_exact_fields(self.software, expected, name="software")
        _require_revision(self.software.get("prob4d_revision"), name="prob4d_revision")
        _require_sha256(
            self.software.get("prob4d_wheel_sha256"),
            name="prob4d_wheel_sha256",
        )
        _require_revision(
            self.software.get("bayesian_phystwin_revision"),
            name="bayesian_phystwin_revision",
        )
        _require_sha256(
            self.software.get("bayesian_phystwin_wheel_sha256"),
            name="bayesian_phystwin_wheel_sha256",
        )
        _require_revision(
            self.software.get("motioncrafter_revision"),
            name="motioncrafter_revision",
        )
        _require_sha256(
            self.software.get("motioncrafter_model_set_id"),
            name="motioncrafter_model_set_id",
        )
        seed_policy = _require_string(
            self.software.get("seed_policy"),
            name="seed_policy",
        )
        if seed_policy not in ALLOWED_SEED_POLICIES:
            raise ValueError("unsupported MotionCrafter seed policy")
        _require_string(self.software.get("python_version"), name="python_version")
        _require_string(self.software.get("numpy_version"), name="numpy_version")

    def _validate_calibration(self) -> None:
        expected = frozenset(
            {
                "gauge_artifact_id",
                "point_artifact_id",
                "reliability_artifact_id",
                "tracklet_policy_id",
                "calibration_split_id",
                "grouping_definition",
            }
        )
        _require_exact_fields(self.calibration, expected, name="calibration")
        for field in (
            "gauge_artifact_id",
            "point_artifact_id",
            "reliability_artifact_id",
            "tracklet_policy_id",
        ):
            _require_sha256(self.calibration.get(field), name=field)
        _require_string(
            self.calibration.get("calibration_split_id"),
            name="calibration_split_id",
        )
        _require_string(
            self.calibration.get("grouping_definition"),
            name="grouping_definition",
        )

    def _validate_analysis(
        self,
        by_method: Mapping[str, ProtocolMethodV1],
    ) -> tuple[str, ...]:
        expected = frozenset(
            {
                "statistical_unit",
                "primary_candidate_method_ids",
                "bootstrap_resamples",
                "bootstrap_seed",
                "exact_fallback_method_id",
                "harmful_update_definition",
                "rejection_treatment",
                "target_outcomes_opened_before_freeze",
                "target_method_selection_allowed",
                "provider_and_physical_gates_separate",
                "causal4d_evaluation_before_bpt_gate",
            }
        )
        _require_exact_fields(self.analysis, expected, name="analysis")
        if self.analysis.get("statistical_unit") != "group_id":
            raise ValueError("prospective analysis must aggregate by group_id")
        primary_value = self.analysis.get("primary_candidate_method_ids")
        if not isinstance(primary_value, tuple) or not primary_value:
            raise ValueError("primary_candidate_method_ids must be a nonempty list")
        primary = tuple(
            _require_string(value, name="primary candidate method ID")
            for value in primary_value
        )
        if len(set(primary)) != len(primary):
            raise ValueError("primary candidate method IDs must be unique")
        for method_id in primary:
            method = by_method.get(method_id)
            if method is None or method.role != "prob4d_candidate":
                raise ValueError("primary methods must be Prob4D candidates")
        _require_integer(
            self.analysis.get("bootstrap_resamples"),
            name="bootstrap_resamples",
            minimum=1000,
        )
        _require_integer(
            self.analysis.get("bootstrap_seed"),
            name="bootstrap_seed",
        )
        if self.analysis.get("exact_fallback_method_id") != "physical_baseline":
            raise ValueError("exact fallback method must be physical_baseline")
        _require_string(
            self.analysis.get("harmful_update_definition"),
            name="harmful_update_definition",
        )
        if (
            self.analysis.get("rejection_treatment")
            != "exact_physical_baseline_fallback"
        ):
            raise ValueError("rejection must use exact physical-baseline fallback")
        required_false = (
            "target_outcomes_opened_before_freeze",
            "target_method_selection_allowed",
            "causal4d_evaluation_before_bpt_gate",
        )
        for field in required_false:
            if self.analysis.get(field) is not False:
                raise ValueError(f"analysis must declare {field}=false")
        if self.analysis.get("provider_and_physical_gates_separate") is not True:
            raise ValueError("provider and physical gates must remain separate")
        return primary

    @property
    def primary_candidate_method_ids(self) -> tuple[str, ...]:
        value = self.analysis["primary_candidate_method_ids"]
        assert isinstance(value, tuple)
        return tuple(str(item) for item in value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": PROTOCOL_SCHEMA,
            "schema_version": PROTOCOL_VERSION,
            "protocol_id": self.protocol_id,
            "claim_id": CLAIM_ID,
            "frozen_at": self.frozen_at,
            "frozen_by": self.frozen_by,
            "split": {
                stage: [unit.to_dict() for unit in units]
                for stage, units in self.split.items()
            },
            "methods": [method.to_dict() for method in self.methods],
            "software": _plain_json(self.software),
            "calibration": _plain_json(self.calibration),
            "analysis": _plain_json(self.analysis),
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "frozen_artifacts": [
                artifact.to_dict() for artifact in self.frozen_artifacts
            ],
            "protocol_sha256": self.protocol_sha256,
        }


def build_prob4d_prospective_protocol(
    configuration: Mapping[str, Any],
) -> Prob4DProspectiveProtocolV1:
    """Normalize an unhashed configuration and bind its content address."""

    if "protocol_sha256" in configuration:
        raise ValueError("unfrozen configuration must not declare protocol_sha256")
    payload = dict(configuration)
    payload.setdefault("schema_name", PROTOCOL_SCHEMA)
    payload.setdefault("schema_version", PROTOCOL_VERSION)
    payload.setdefault("claim_id", CLAIM_ID)
    payload["protocol_sha256"] = _content_id(payload, field="protocol_sha256")
    return Prob4DProspectiveProtocolV1.from_dict(payload)


def load_prob4d_prospective_protocol(
    path: str | Path,
) -> Prob4DProspectiveProtocolV1:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prospective protocol {path}") from error
    return Prob4DProspectiveProtocolV1.from_dict(
        _require_mapping(payload, name="prospective protocol")
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_prob4d_prospective_protocol(
    path: str | Path,
    protocol: Prob4DProspectiveProtocolV1,
) -> None:
    _atomic_write_json(Path(path), protocol.to_dict())


def _resolve_frozen_artifact(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("artifact root must be a regular directory")
    candidate = root_resolved.joinpath(*PurePosixPath(relative).parts)
    if candidate.is_symlink():
        raise ValueError(f"frozen artifact must not be a symlink: {relative}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"frozen artifact escapes root: {relative}") from error
    return resolved


def check_prob4d_prospective_readiness(
    protocol: Prob4DProspectiveProtocolV1,
    artifact_root: str | Path,
) -> dict[str, object]:
    """Hash-verify every pre-target artifact and emit a content-addressed status."""

    root = Path(artifact_root)
    records: list[dict[str, object]] = []
    ready = True
    for artifact in protocol.frozen_artifacts:
        record: dict[str, object] = {
            "role": artifact.role,
            "path": artifact.path,
            "expected_sha256": artifact.sha256,
            "access_stage": artifact.access_stage,
        }
        try:
            path = _resolve_frozen_artifact(root, artifact.path)
            if not path.is_file():
                raise ValueError("file is missing")
            observed = _sha256_file(path)
            matched = observed == artifact.sha256
            record.update(
                observed_sha256=observed,
                matched=matched,
                error=None if matched else "SHA-256 mismatch",
            )
            ready = ready and matched
        except ValueError as error:
            record.update(observed_sha256=None, matched=False, error=str(error))
            ready = False
        records.append(record)
    payload: dict[str, Any] = {
        "schema_name": READINESS_SCHEMA,
        "schema_version": READINESS_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "artifact_root": str(root.resolve()),
        "target_outcomes_opened_before_freeze": False,
        "provider_and_physical_gates_separate": True,
        "causal4d_evaluation_admissible": False,
        "artifacts": records,
        "ready_for_target_opening": ready,
    }
    payload["readiness_sha256"] = _content_id(payload, field="readiness_sha256")
    return payload


def _criterion_statistics(
    value: Mapping[str, Any],
) -> dict[str, float]:
    raw = _require_mapping(
        value.get("criterion_statistics"),
        name="criterion_statistics",
    )
    return {
        _require_string(key, name="criterion ID"): _require_finite(
            item,
            name=f"criterion_statistics[{key!r}]",
        )
        for key, item in raw.items()
    }


def decide_prob4d_prospective_gates(
    protocol: Prob4DProspectiveProtocolV1,
    result: Mapping[str, Any],
) -> dict[str, object]:
    """Apply the frozen provider gate before admitting the physical gate."""

    expected_fields = frozenset(
        {
            "schema_name",
            "schema_version",
            "protocol_sha256",
            "criterion_statistics",
            "physical_update",
            "target_access",
        }
    )
    _require_exact_fields(result, expected_fields, name="prospective result")
    if result.get("schema_name") != RESULT_SCHEMA:
        raise ValueError("unsupported Prob4D prospective result schema")
    if result.get("schema_version") != RESULT_VERSION:
        raise ValueError("unsupported Prob4D prospective result version")
    if result.get("protocol_sha256") != protocol.protocol_sha256:
        raise ValueError("prospective result references a different protocol")
    target_access = _require_mapping(result.get("target_access"), name="target_access")
    _require_exact_fields(
        target_access,
        frozenset({"opened_after_freeze", "selection_performed_after_opening"}),
        name="target_access",
    )
    if target_access.get("opened_after_freeze") is not True:
        raise ValueError("target outcomes must be opened only after the freeze")
    if target_access.get("selection_performed_after_opening") is not False:
        raise ValueError("target-informed method selection is forbidden")

    statistics = _criterion_statistics(result)
    criteria_by_stage = {
        stage: [criterion for criterion in protocol.criteria if criterion.stage == stage]
        for stage in ("provider", "physical")
    }

    def evaluate(
        criteria: Sequence[GateCriterionV1],
        *,
        require_all: bool,
    ) -> tuple[list[dict[str, object]], bool | None]:
        records: list[dict[str, object]] = []
        missing: list[str] = []
        for criterion in criteria:
            if criterion.criterion_id not in statistics:
                missing.append(criterion.criterion_id)
                continue
            observed = statistics[criterion.criterion_id]
            records.append(
                {
                    **criterion.to_dict(),
                    "observed": observed,
                    "passed": criterion.passes(observed),
                }
            )
        if missing and require_all:
            raise ValueError(f"prospective result lacks gate criteria: {sorted(missing)}")
        if missing:
            return records, None
        return records, all(bool(record["passed"]) for record in records)

    provider_records, provider_passed = evaluate(
        criteria_by_stage["provider"],
        require_all=True,
    )
    assert provider_passed is not None
    physical_records: list[dict[str, object]] = []
    physical_passed: bool | None = None
    physical_update_value = result.get("physical_update")
    if provider_passed:
        physical_records, physical_passed = evaluate(
            criteria_by_stage["physical"],
            require_all=True,
        )
        physical_update = _require_mapping(
            physical_update_value,
            name="physical_update",
        )
        _require_exact_fields(
            physical_update,
            frozenset(
                {
                    "fallback_method_id",
                    "fallback_exact",
                    "evaluated_group_count",
                    "accepted_update_count",
                    "harmful_accepted_update_count",
                }
            ),
            name="physical_update",
        )
        if physical_update.get("fallback_method_id") != "physical_baseline":
            raise ValueError("physical result changed the frozen fallback")
        if physical_update.get("fallback_exact") is not True:
            raise ValueError("physical result did not preserve exact fallback")
        group_count = _require_integer(
            physical_update.get("evaluated_group_count"),
            name="evaluated_group_count",
            minimum=1,
        )
        accepted = _require_integer(
            physical_update.get("accepted_update_count"),
            name="accepted_update_count",
        )
        harmful = _require_integer(
            physical_update.get("harmful_accepted_update_count"),
            name="harmful_accepted_update_count",
        )
        if harmful > accepted or accepted > group_count:
            raise ValueError("physical update counts are inconsistent")
    else:
        if physical_update_value is not None:
            raise ValueError(
                "physical_update must be null when provider competence fails"
            )
        _, omitted = evaluate(criteria_by_stage["physical"], require_all=False)
        if omitted is not None:
            raise ValueError(
                "physical criteria must remain unopened after provider failure"
            )

    overall = bool(provider_passed and physical_passed)
    payload: dict[str, Any] = {
        "schema_name": DECISION_SCHEMA,
        "schema_version": DECISION_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "provider_gate": {
            "passed": provider_passed,
            "criteria": provider_records,
        },
        "physical_gate": {
            "admissible": provider_passed,
            "passed": physical_passed,
            "criteria": physical_records,
        },
        "prob4d_supported_feeder": overall,
        "causal4d_evaluation_admissible": overall,
        "exact_fallback_method_id": "physical_baseline",
        "target_informed_method_selection": False,
    }
    payload["decision_sha256"] = _content_id(payload, field="decision_sha256")
    return payload


def load_json_mapping(path: str | Path, *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} {path}") from error
    return _require_mapping(payload, name=name)


def write_json_mapping(path: str | Path, value: Mapping[str, Any]) -> None:
    _atomic_write_json(Path(path), value)


__all__ = [
    "CLAIM_ID",
    "DECISION_SCHEMA",
    "DECISION_VERSION",
    "GateCriterionV1",
    "Prob4DProspectiveProtocolV1",
    "ProtocolMethodV1",
    "ProtocolUnitV1",
    "READINESS_SCHEMA",
    "READINESS_VERSION",
    "REQUIRED_METHODS",
    "RESULT_SCHEMA",
    "RESULT_VERSION",
    "build_prob4d_prospective_protocol",
    "check_prob4d_prospective_readiness",
    "decide_prob4d_prospective_gates",
    "load_json_mapping",
    "load_prob4d_prospective_protocol",
    "save_prob4d_prospective_protocol",
    "write_json_mapping",
]
