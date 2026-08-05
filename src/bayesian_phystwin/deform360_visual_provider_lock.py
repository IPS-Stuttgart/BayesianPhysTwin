"""Target-blind locks for the Deform360 visual provider and Stage-1 calibration.

The official-Hub Deform360 cohort is selected before camera, tactile, robot, or
geometry payload access.  These contracts make the remaining visual-observation
choices equally explicit: the exact Prob4D/MotionCrafter producer is frozen
before calibration payload access, and calibration-derived quantities are frozen
before confirmation payload access.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)

DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA = (
    "bayesian-phystwin.deform360-visual-provider-lock"
)
DEFORM360_VISUAL_PROVIDER_LOCK_VERSION = 1
DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS = (
    "target-blind-prob4d-motioncrafter-producer-lock-v1"
)
DEFORM360_VISUAL_CALIBRATION_LOCK_SCHEMA = (
    "bayesian-phystwin.deform360-visual-calibration-lock"
)
DEFORM360_VISUAL_CALIBRATION_LOCK_VERSION = 1
DEFORM360_VISUAL_CALIBRATION_LOCK_SEMANTICS = (
    "calibration-only-visual-contact-guard-interval-finite-group-lock-v1"
)
DEFORM360_VISUOTACTILE_PROTOCOL_ID = "deform360-official-hub-visuotactile-v1"
DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID = (
    "deform360-official-hub-visuotactile-v1-visual-provider-lock"
)
DEFORM360_PROB4D_REPOSITORY = "IPS-Stuttgart/Prob4D"
DEFORM360_MOTIONCRAFTER_REPOSITORY = "TencentARC/MotionCrafter"
DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID = (
    "697261131ae83a08a0ce437c7b837f32ab4870c6c0cc4256784373b3ce19f1c8"
)
DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT = 10
DEFORM360_FINITE_GROUP_CONFORMAL_RANK = 10

_VISUAL_PROVIDER_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "artifact_id",
        "protocol_id",
        "amendment_id",
        "provider_repository",
        "provider_revision",
        "provider_api_version",
        "provider_manifest_id",
        "provider_attestation_sha256",
        "motioncrafter_repository",
        "motioncrafter_revision",
        "model_set_id",
        "root_seed",
        "seed_policy",
        "window_size",
        "overlap",
        "height",
        "width",
        "storage_dtype",
        "initial_metric_frame_prior_id",
        "additional_metric_anchor_policy",
        "max_gauge_rank",
        "minimum_retained_gauge_trace",
        "stream_contract_version",
        "full_joint_gauge_covariance",
        "persistent_material_identities",
        "causal_cutoff_convention",
        "selected_raw_payloads_opened",
        "target_outcomes_used",
        "metadata",
    }
)
_VISUAL_CALIBRATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "artifact_id",
        "protocol_id",
        "amendment_id",
        "visual_provider_lock_id",
        "selection_lock_id",
        "calibration_object_ids",
        "visual_calibration_id",
        "contact_anchor_calibration_id",
        "guard_calibration_id",
        "interval_calibration_id",
        "calibration_design_id",
        "calibration_group_count",
        "conformal_rank",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "metadata",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _sha256(value: object, *, name: str) -> str:
    try:
        return literal_lower_hex(value, name=name, lengths={64})
    except ValueError as error:
        raise ValueError(
            f"{name} must be a literal lowercase SHA-256 digest"
        ) from error


def _revision(value: object, *, name: str) -> str:
    try:
        return literal_lower_hex(value, name=name, lengths={40})
    except ValueError as error:
        raise ValueError(
            f"{name} must be an exact literal lowercase Git commit"
        ) from error


def _finite_probability(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number in (0, 1]")
    result = float(value)
    if not np.isfinite(result) or not (0.0 < result <= 1.0):
        raise ValueError(f"{name} must be a real number in (0, 1]")
    return result


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _load_strict_json(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root must be a JSON object")
    return value


@dataclass(frozen=True)
class Deform360VisualProviderLockV1:
    """Exact target-blind Prob4D/MotionCrafter producer configuration."""

    provider_revision: str
    provider_manifest_id: str
    provider_attestation_sha256: str
    motioncrafter_revision: str
    model_set_id: str
    root_seed: int
    seed_policy: str
    window_size: int
    overlap: int
    height: int
    width: int
    storage_dtype: Literal["float32", "float64"]
    initial_metric_frame_prior_id: str
    additional_metric_anchor_policy: Literal["none", "independent_sparse"]
    max_gauge_rank: int | None
    minimum_retained_gauge_trace: float
    selected_raw_payloads_opened: bool = False
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_id: str = DEFORM360_VISUOTACTILE_PROTOCOL_ID
    amendment_id: str = DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID
    provider_repository: str = DEFORM360_PROB4D_REPOSITORY
    motioncrafter_repository: str = DEFORM360_MOTIONCRAFTER_REPOSITORY

    def __post_init__(self) -> None:
        protocol_id = _literal_string(self.protocol_id, name="protocol_id")
        if protocol_id != DEFORM360_VISUOTACTILE_PROTOCOL_ID:
            raise ValueError("visual provider lock changed protocol_id")
        amendment_id = _literal_string(self.amendment_id, name="amendment_id")
        if amendment_id != DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID:
            raise ValueError("visual provider lock changed amendment_id")
        provider_repository = _literal_string(
            self.provider_repository,
            name="provider_repository",
        )
        if provider_repository != DEFORM360_PROB4D_REPOSITORY:
            raise ValueError("visual provider lock changed Prob4D repository")
        motioncrafter_repository = _literal_string(
            self.motioncrafter_repository,
            name="motioncrafter_repository",
        )
        if motioncrafter_repository != DEFORM360_MOTIONCRAFTER_REPOSITORY:
            raise ValueError("visual provider lock changed MotionCrafter repository")

        provider_revision = _revision(
            self.provider_revision,
            name="provider_revision",
        )
        provider_manifest_id = _sha256(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        provider_attestation_sha256 = _sha256(
            self.provider_attestation_sha256,
            name="provider_attestation_sha256",
        )
        motioncrafter_revision = _revision(
            self.motioncrafter_revision,
            name="motioncrafter_revision",
        )
        model_set_id = _sha256(self.model_set_id, name="model_set_id")
        root_seed = genuine_integer(self.root_seed, name="root_seed", minimum=0)
        seed_policy = _literal_string(self.seed_policy, name="seed_policy")
        window_size = genuine_integer(
            self.window_size,
            name="window_size",
            minimum=2,
        )
        overlap = genuine_integer(self.overlap, name="overlap", minimum=0)
        if overlap >= window_size:
            raise ValueError("overlap must be smaller than window_size")
        height = genuine_integer(self.height, name="height", minimum=1)
        width = genuine_integer(self.width, name="width", minimum=1)
        if self.storage_dtype not in {"float32", "float64"}:
            raise ValueError("storage_dtype must be float32 or float64")
        initial_metric_frame_prior_id = _sha256(
            self.initial_metric_frame_prior_id,
            name="initial_metric_frame_prior_id",
        )
        if self.additional_metric_anchor_policy not in {
            "none",
            "independent_sparse",
        }:
            raise ValueError("additional_metric_anchor_policy is unsupported")
        max_gauge_rank = self.max_gauge_rank
        if max_gauge_rank is not None:
            max_gauge_rank = genuine_integer(
                max_gauge_rank,
                name="max_gauge_rank",
                minimum=1,
            )
        minimum_retained_gauge_trace = _finite_probability(
            self.minimum_retained_gauge_trace,
            name="minimum_retained_gauge_trace",
        )
        selected_raw_payloads_opened = genuine_boolean(
            self.selected_raw_payloads_opened,
            name="selected_raw_payloads_opened",
        )
        target_outcomes_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if selected_raw_payloads_opened:
            raise ValueError("visual provider lock requires unopened selected payloads")
        if target_outcomes_used:
            raise ValueError("visual provider lock must be target blind")

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "amendment_id", amendment_id)
        object.__setattr__(self, "provider_repository", provider_repository)
        object.__setattr__(self, "provider_revision", provider_revision)
        object.__setattr__(self, "provider_manifest_id", provider_manifest_id)
        object.__setattr__(
            self,
            "provider_attestation_sha256",
            provider_attestation_sha256,
        )
        object.__setattr__(
            self,
            "motioncrafter_repository",
            motioncrafter_repository,
        )
        object.__setattr__(self, "motioncrafter_revision", motioncrafter_revision)
        object.__setattr__(self, "model_set_id", model_set_id)
        object.__setattr__(self, "root_seed", root_seed)
        object.__setattr__(self, "seed_policy", seed_policy)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)
        object.__setattr__(
            self,
            "initial_metric_frame_prior_id",
            initial_metric_frame_prior_id,
        )
        object.__setattr__(self, "max_gauge_rank", max_gauge_rank)
        object.__setattr__(
            self,
            "minimum_retained_gauge_trace",
            minimum_retained_gauge_trace,
        )
        object.__setattr__(
            self,
            "selected_raw_payloads_opened",
            selected_raw_payloads_opened,
        )
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="visual provider lock metadata",
            ),
        )

    def descriptor(self) -> dict[str, object]:
        """Return the timestamp-free content-addressed descriptor."""

        return {
            "schema": DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA,
            "schema_version": DEFORM360_VISUAL_PROVIDER_LOCK_VERSION,
            "semantics": DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS,
            "protocol_id": self.protocol_id,
            "amendment_id": self.amendment_id,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "provider_api_version": 2,
            "provider_manifest_id": self.provider_manifest_id,
            "provider_attestation_sha256": self.provider_attestation_sha256,
            "motioncrafter_repository": self.motioncrafter_repository,
            "motioncrafter_revision": self.motioncrafter_revision,
            "model_set_id": self.model_set_id,
            "root_seed": self.root_seed,
            "seed_policy": self.seed_policy,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "height": self.height,
            "width": self.width,
            "storage_dtype": self.storage_dtype,
            "initial_metric_frame_prior_id": self.initial_metric_frame_prior_id,
            "additional_metric_anchor_policy": (self.additional_metric_anchor_policy),
            "max_gauge_rank": self.max_gauge_rank,
            "minimum_retained_gauge_trace": self.minimum_retained_gauge_trace,
            "stream_contract_version": 2,
            "full_joint_gauge_covariance": True,
            "persistent_material_identities": True,
            "causal_cutoff_convention": "exclusive",
            "selected_raw_payloads_opened": self.selected_raw_payloads_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        """Return the SHA-256 content identity of the complete lock."""

        return _content_id(self.descriptor())

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> Deform360VisualProviderLockV1:
        if not isinstance(value, Mapping):
            raise ValueError("visual provider lock must be a JSON object")
        _require_exact_fields(
            value,
            expected=_VISUAL_PROVIDER_FIELDS,
            name="visual provider lock",
        )
        if value["schema"] != DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA:
            raise ValueError("unsupported visual provider lock schema")
        schema_version = genuine_integer(
            value["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if schema_version != DEFORM360_VISUAL_PROVIDER_LOCK_VERSION:
            raise ValueError("unsupported visual provider lock version")
        if value["semantics"] != DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS:
            raise ValueError("visual provider lock semantics changed")
        provider_api_version = genuine_integer(
            value["provider_api_version"],
            name="provider_api_version",
            minimum=1,
        )
        if provider_api_version != 2:
            raise ValueError("visual provider lock changed provider_api_version")
        stream_contract_version = genuine_integer(
            value["stream_contract_version"],
            name="stream_contract_version",
            minimum=1,
        )
        if stream_contract_version != 2:
            raise ValueError("visual provider lock changed stream_contract_version")
        for field_name in (
            "full_joint_gauge_covariance",
            "persistent_material_identities",
        ):
            if not genuine_boolean(value[field_name], name=field_name):
                raise ValueError(f"visual provider lock changed {field_name}")
        cutoff = _literal_string(
            value["causal_cutoff_convention"],
            name="causal_cutoff_convention",
        )
        if cutoff != "exclusive":
            raise ValueError("visual provider lock changed causal_cutoff_convention")
        lock = cls(
            provider_revision=cast(str, value["provider_revision"]),
            provider_manifest_id=cast(str, value["provider_manifest_id"]),
            provider_attestation_sha256=cast(
                str,
                value["provider_attestation_sha256"],
            ),
            motioncrafter_revision=cast(str, value["motioncrafter_revision"]),
            model_set_id=cast(str, value["model_set_id"]),
            root_seed=cast(int, value["root_seed"]),
            seed_policy=cast(str, value["seed_policy"]),
            window_size=cast(int, value["window_size"]),
            overlap=cast(int, value["overlap"]),
            height=cast(int, value["height"]),
            width=cast(int, value["width"]),
            storage_dtype=cast(Literal["float32", "float64"], value["storage_dtype"]),
            initial_metric_frame_prior_id=cast(
                str,
                value["initial_metric_frame_prior_id"],
            ),
            additional_metric_anchor_policy=cast(
                Literal["none", "independent_sparse"],
                value["additional_metric_anchor_policy"],
            ),
            max_gauge_rank=cast(int | None, value["max_gauge_rank"]),
            minimum_retained_gauge_trace=cast(
                float,
                value["minimum_retained_gauge_trace"],
            ),
            selected_raw_payloads_opened=cast(
                bool,
                value["selected_raw_payloads_opened"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            protocol_id=cast(str, value["protocol_id"]),
            amendment_id=cast(str, value["amendment_id"]),
            provider_repository=cast(str, value["provider_repository"]),
            motioncrafter_repository=cast(
                str,
                value["motioncrafter_repository"],
            ),
        )
        declared_id = _sha256(value["artifact_id"], name="artifact_id")
        if declared_id != lock.artifact_id:
            raise ValueError("visual provider lock artifact_id does not match content")
        return lock


@dataclass(frozen=True)
class Deform360VisualCalibrationLockV1:
    """Calibration-only choices frozen before confirmation payload access."""

    visual_provider_lock_id: str
    selection_lock_id: str
    calibration_object_ids: Sequence[str]
    visual_calibration_id: str
    contact_anchor_calibration_id: str
    guard_calibration_id: str
    interval_calibration_id: str
    calibration_design_id: str
    calibration_group_count: int
    conformal_rank: int
    confirmation_payloads_opened: bool = False
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_id: str = DEFORM360_VISUOTACTILE_PROTOCOL_ID
    amendment_id: str = DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID

    def __post_init__(self) -> None:
        protocol_id = _literal_string(self.protocol_id, name="protocol_id")
        if protocol_id != DEFORM360_VISUOTACTILE_PROTOCOL_ID:
            raise ValueError("visual calibration lock changed protocol_id")
        amendment_id = _literal_string(self.amendment_id, name="amendment_id")
        if amendment_id != DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID:
            raise ValueError("visual calibration lock changed amendment_id")
        visual_provider_lock_id = _sha256(
            self.visual_provider_lock_id,
            name="visual_provider_lock_id",
        )
        selection_lock_id = _sha256(
            self.selection_lock_id,
            name="selection_lock_id",
        )
        calibration_object_ids = canonical_string_tuple(
            self.calibration_object_ids,
            name="calibration_object_ids",
            allow_empty=False,
        )
        if len(set(calibration_object_ids)) != len(calibration_object_ids):
            raise ValueError("calibration_object_ids must be unique")
        visual_calibration_id = _sha256(
            self.visual_calibration_id,
            name="visual_calibration_id",
        )
        contact_anchor_calibration_id = _sha256(
            self.contact_anchor_calibration_id,
            name="contact_anchor_calibration_id",
        )
        guard_calibration_id = _sha256(
            self.guard_calibration_id,
            name="guard_calibration_id",
        )
        interval_calibration_id = _sha256(
            self.interval_calibration_id,
            name="interval_calibration_id",
        )
        calibration_design_id = _sha256(
            self.calibration_design_id,
            name="calibration_design_id",
        )
        if calibration_design_id != DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID:
            raise ValueError(
                "calibration_design_id changed the registered finite-group design"
            )
        calibration_group_count = genuine_integer(
            self.calibration_group_count,
            name="calibration_group_count",
            minimum=1,
        )
        if calibration_group_count != len(calibration_object_ids):
            raise ValueError(
                "calibration_group_count must equal the number of calibration objects"
            )
        if calibration_group_count != DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT:
            raise ValueError(
                "calibration_group_count must equal the registered "
                "finite-group design count"
            )
        conformal_rank = genuine_integer(
            self.conformal_rank,
            name="conformal_rank",
            minimum=1,
        )
        if conformal_rank != DEFORM360_FINITE_GROUP_CONFORMAL_RANK:
            raise ValueError(
                "conformal_rank must equal the registered finite-group design rank"
            )
        confirmation_payloads_opened = genuine_boolean(
            self.confirmation_payloads_opened,
            name="confirmation_payloads_opened",
        )
        target_outcomes_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if confirmation_payloads_opened:
            raise ValueError(
                "visual calibration lock requires unopened confirmation payloads"
            )
        if target_outcomes_used:
            raise ValueError("visual calibration lock must be target blind")

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "amendment_id", amendment_id)
        object.__setattr__(
            self,
            "visual_provider_lock_id",
            visual_provider_lock_id,
        )
        object.__setattr__(self, "selection_lock_id", selection_lock_id)
        object.__setattr__(
            self,
            "calibration_object_ids",
            calibration_object_ids,
        )
        object.__setattr__(
            self,
            "visual_calibration_id",
            visual_calibration_id,
        )
        object.__setattr__(
            self,
            "contact_anchor_calibration_id",
            contact_anchor_calibration_id,
        )
        object.__setattr__(self, "guard_calibration_id", guard_calibration_id)
        object.__setattr__(
            self,
            "interval_calibration_id",
            interval_calibration_id,
        )
        object.__setattr__(
            self,
            "calibration_design_id",
            calibration_design_id,
        )
        object.__setattr__(
            self,
            "calibration_group_count",
            calibration_group_count,
        )
        object.__setattr__(self, "conformal_rank", conformal_rank)
        object.__setattr__(
            self,
            "confirmation_payloads_opened",
            confirmation_payloads_opened,
        )
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="visual calibration lock metadata",
            ),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_VISUAL_CALIBRATION_LOCK_SCHEMA,
            "schema_version": DEFORM360_VISUAL_CALIBRATION_LOCK_VERSION,
            "semantics": DEFORM360_VISUAL_CALIBRATION_LOCK_SEMANTICS,
            "protocol_id": self.protocol_id,
            "amendment_id": self.amendment_id,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "selection_lock_id": self.selection_lock_id,
            "calibration_object_ids": self.calibration_object_ids,
            "visual_calibration_id": self.visual_calibration_id,
            "contact_anchor_calibration_id": self.contact_anchor_calibration_id,
            "guard_calibration_id": self.guard_calibration_id,
            "interval_calibration_id": self.interval_calibration_id,
            "calibration_design_id": self.calibration_design_id,
            "calibration_group_count": self.calibration_group_count,
            "conformal_rank": self.conformal_rank,
            "confirmation_payloads_opened": self.confirmation_payloads_opened,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        return _content_id(self.descriptor())

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> Deform360VisualCalibrationLockV1:
        if not isinstance(value, Mapping):
            raise ValueError("visual calibration lock must be a JSON object")
        _require_exact_fields(
            value,
            expected=_VISUAL_CALIBRATION_FIELDS,
            name="visual calibration lock",
        )
        if value["schema"] != DEFORM360_VISUAL_CALIBRATION_LOCK_SCHEMA:
            raise ValueError("unsupported visual calibration lock schema")
        schema_version = genuine_integer(
            value["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if schema_version != DEFORM360_VISUAL_CALIBRATION_LOCK_VERSION:
            raise ValueError("unsupported visual calibration lock version")
        if value["semantics"] != DEFORM360_VISUAL_CALIBRATION_LOCK_SEMANTICS:
            raise ValueError("visual calibration lock semantics changed")
        lock = cls(
            visual_provider_lock_id=cast(str, value["visual_provider_lock_id"]),
            selection_lock_id=cast(str, value["selection_lock_id"]),
            calibration_object_ids=cast(
                Sequence[str],
                value["calibration_object_ids"],
            ),
            visual_calibration_id=cast(str, value["visual_calibration_id"]),
            contact_anchor_calibration_id=cast(
                str,
                value["contact_anchor_calibration_id"],
            ),
            guard_calibration_id=cast(str, value["guard_calibration_id"]),
            interval_calibration_id=cast(str, value["interval_calibration_id"]),
            calibration_design_id=cast(str, value["calibration_design_id"]),
            calibration_group_count=cast(int, value["calibration_group_count"]),
            conformal_rank=cast(int, value["conformal_rank"]),
            confirmation_payloads_opened=cast(
                bool,
                value["confirmation_payloads_opened"],
            ),
            target_outcomes_used=cast(bool, value["target_outcomes_used"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            protocol_id=cast(str, value["protocol_id"]),
            amendment_id=cast(str, value["amendment_id"]),
        )
        declared_id = _sha256(value["artifact_id"], name="artifact_id")
        if declared_id != lock.artifact_id:
            raise ValueError(
                "visual calibration lock artifact_id does not match content"
            )
        return lock


def save_deform360_visual_provider_lock(
    path: str | Path,
    lock: Deform360VisualProviderLockV1,
) -> None:
    """Serialize a canonical human-readable visual-provider lock."""

    Path(path).write_text(
        json.dumps(lock.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_deform360_visual_provider_lock(
    path: str | Path,
) -> Deform360VisualProviderLockV1:
    """Strictly load and independently revalidate a visual-provider lock."""

    value = _load_strict_json(Path(path), name="visual provider lock")
    return Deform360VisualProviderLockV1.from_mapping(value)


def save_deform360_visual_calibration_lock(
    path: str | Path,
    lock: Deform360VisualCalibrationLockV1,
) -> None:
    """Serialize a canonical human-readable Stage-1 calibration lock."""

    Path(path).write_text(
        json.dumps(lock.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_deform360_visual_calibration_lock(
    path: str | Path,
) -> Deform360VisualCalibrationLockV1:
    """Strictly load and independently revalidate a Stage-1 calibration lock."""

    value = _load_strict_json(Path(path), name="visual calibration lock")
    return Deform360VisualCalibrationLockV1.from_mapping(value)


__all__ = [
    "DEFORM360_MOTIONCRAFTER_REPOSITORY",
    "DEFORM360_PROB4D_REPOSITORY",
    "DEFORM360_VISUAL_CALIBRATION_LOCK_SCHEMA",
    "DEFORM360_VISUAL_CALIBRATION_LOCK_SEMANTICS",
    "DEFORM360_VISUAL_CALIBRATION_LOCK_VERSION",
    "DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID",
    "DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA",
    "DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS",
    "DEFORM360_VISUAL_PROVIDER_LOCK_VERSION",
    "DEFORM360_VISUOTACTILE_PROTOCOL_ID",
    "Deform360VisualCalibrationLockV1",
    "Deform360VisualProviderLockV1",
    "load_deform360_visual_calibration_lock",
    "load_deform360_visual_provider_lock",
    "save_deform360_visual_calibration_lock",
    "save_deform360_visual_provider_lock",
]
