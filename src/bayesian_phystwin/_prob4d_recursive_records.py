"""Content-addressed recursive Prob4D step and run records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import content_id
from ._prob4d_stream_common import (
    CLAIM_BEARING_PROB4D_STREAM_RUN_SCHEMA,
    CLAIM_BEARING_PROB4D_STREAM_RUN_VERSION,
    CLAIM_BEARING_PROB4D_STREAM_STEP_SCHEMA,
    CLAIM_BEARING_PROB4D_STREAM_STEP_VERSION,
    _RUN_FIELDS,
    _STEP_FIELDS,
    _calibration_ids,
    _nonempty_literal_string,
    _optional_calibration_ids,
    _sha256,
)


@dataclass(frozen=True, slots=True)
class ClaimBearingProb4DStreamStepV1:
    """One immutable recursive claim-bearing update and complete-belief route."""

    stream_artifact_id: str
    stream_update_id: str
    observation_binding_id: str
    update_index: int
    admitted_frame_start: int
    causal_frame_stop: int
    prior_belief_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    claim_update_id: str
    candidate_belief_id: str
    guard_decision_id: str
    selection_id: str
    selected_belief_id: str
    selected_candidate: bool
    exact_fallback: bool
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    runtime_revision_independently_verified: bool
    covariance_semantics_id: str
    covariance_policy_id: str
    recursive_nuisance_policy_id: str
    previous_step_id: str | None
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    step_id: str | None = None

    def __post_init__(self) -> None:
        digest_fields = (
            "stream_artifact_id",
            "stream_update_id",
            "observation_binding_id",
            "prior_belief_id",
            "observation_artifact_id",
            "linearization_artifact_id",
            "claim_update_id",
            "candidate_belief_id",
            "guard_decision_id",
            "selection_id",
            "selected_belief_id",
            "provider_manifest_id",
            "covariance_semantics_id",
            "covariance_policy_id",
            "recursive_nuisance_policy_id",
        )
        for name in digest_fields:
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        update_index = genuine_integer(
            self.update_index,
            name="update_index",
            minimum=0,
        )
        admitted_start = genuine_integer(
            self.admitted_frame_start,
            name="admitted_frame_start",
            minimum=0,
        )
        causal_stop = genuine_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        if causal_stop <= admitted_start:
            raise ValueError("step causal_frame_stop must exceed admitted start")
        selected_candidate = genuine_boolean(
            self.selected_candidate,
            name="selected_candidate",
        )
        exact_fallback = genuine_boolean(
            self.exact_fallback,
            name="exact_fallback",
        )
        expected_selected = (
            self.candidate_belief_id
            if selected_candidate
            else self.prior_belief_id
        )
        if self.selected_belief_id != expected_selected:
            raise ValueError("selected_belief_id contradicts selected_candidate")
        if exact_fallback == selected_candidate:
            raise ValueError(
                "exact_fallback must be true exactly when the candidate is rejected"
            )
        calibration_ids = _calibration_ids(self.calibration_artifact_ids)
        runtime_source = _nonempty_literal_string(
            self.runtime_revision_source,
            name="runtime_revision_source",
        )
        runtime_verified = genuine_boolean(
            self.runtime_revision_independently_verified,
            name="runtime_revision_independently_verified",
        )
        if not runtime_verified:
            raise ValueError(
                "runtime_revision_independently_verified must be true"
            )
        previous = self.previous_step_id
        if previous is not None:
            previous = _sha256(previous, name="previous_step_id")
        if update_index == 0 and previous is not None:
            raise ValueError("the first recursive step cannot have a predecessor")
        if update_index > 0 and previous is None:
            raise ValueError("a later recursive step must bind its predecessor")
        reason = _nonempty_literal_string(self.reason, name="reason")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="claim-bearing stream step metadata",
        )
        object.__setattr__(self, "update_index", update_index)
        object.__setattr__(self, "admitted_frame_start", admitted_start)
        object.__setattr__(self, "causal_frame_stop", causal_stop)
        object.__setattr__(self, "selected_candidate", selected_candidate)
        object.__setattr__(self, "exact_fallback", exact_fallback)
        object.__setattr__(self, "calibration_artifact_ids", calibration_ids)
        object.__setattr__(self, "runtime_revision_source", runtime_source)
        object.__setattr__(
            self,
            "runtime_revision_independently_verified",
            runtime_verified,
        )
        object.__setattr__(self, "previous_step_id", previous)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        supplied_id = self.step_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="step_id")
            if supplied_id != expected_id:
                raise ValueError("claim-bearing stream step_id does not match content")
        object.__setattr__(self, "step_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CLAIM_BEARING_PROB4D_STREAM_STEP_SCHEMA,
            "schema_version": CLAIM_BEARING_PROB4D_STREAM_STEP_VERSION,
            "stream_artifact_id": self.stream_artifact_id,
            "stream_update_id": self.stream_update_id,
            "observation_binding_id": self.observation_binding_id,
            "update_index": self.update_index,
            "admitted_frame_start": self.admitted_frame_start,
            "causal_frame_stop": self.causal_frame_stop,
            "prior_belief_id": self.prior_belief_id,
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "claim_update_id": self.claim_update_id,
            "candidate_belief_id": self.candidate_belief_id,
            "guard_decision_id": self.guard_decision_id,
            "selection_id": self.selection_id,
            "selected_belief_id": self.selected_belief_id,
            "selected_candidate": self.selected_candidate,
            "exact_fallback": self.exact_fallback,
            "provider_manifest_id": self.provider_manifest_id,
            "calibration_artifact_ids": dict(self.calibration_artifact_ids),
            "runtime_revision_source": self.runtime_revision_source,
            "runtime_revision_independently_verified": (
                self.runtime_revision_independently_verified
            ),
            "covariance_semantics_id": self.covariance_semantics_id,
            "covariance_policy_id": self.covariance_policy_id,
            "recursive_nuisance_policy_id": (
                self.recursive_nuisance_policy_id
            ),
            "previous_step_id": self.previous_step_id,
            "reason": self.reason,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "step_id": self.step_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "claim-bearing Prob4D stream step",
    ) -> ClaimBearingProb4DStreamStepV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if set(value) != _STEP_FIELDS:
            raise ValueError(f"{name} fields changed")
        if value["schema"] != CLAIM_BEARING_PROB4D_STREAM_STEP_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != CLAIM_BEARING_PROB4D_STREAM_STEP_VERSION:
            raise ValueError(f"{name} version changed")
        return cls(
            stream_artifact_id=cast(str, value["stream_artifact_id"]),
            stream_update_id=cast(str, value["stream_update_id"]),
            observation_binding_id=cast(str, value["observation_binding_id"]),
            update_index=cast(int, value["update_index"]),
            admitted_frame_start=cast(int, value["admitted_frame_start"]),
            causal_frame_stop=cast(int, value["causal_frame_stop"]),
            prior_belief_id=cast(str, value["prior_belief_id"]),
            observation_artifact_id=cast(
                str,
                value["observation_artifact_id"],
            ),
            linearization_artifact_id=cast(
                str,
                value["linearization_artifact_id"],
            ),
            claim_update_id=cast(str, value["claim_update_id"]),
            candidate_belief_id=cast(str, value["candidate_belief_id"]),
            guard_decision_id=cast(str, value["guard_decision_id"]),
            selection_id=cast(str, value["selection_id"]),
            selected_belief_id=cast(str, value["selected_belief_id"]),
            selected_candidate=cast(bool, value["selected_candidate"]),
            exact_fallback=cast(bool, value["exact_fallback"]),
            provider_manifest_id=cast(str, value["provider_manifest_id"]),
            calibration_artifact_ids=cast(
                Mapping[str, str],
                value["calibration_artifact_ids"],
            ),
            runtime_revision_source=cast(
                str,
                value["runtime_revision_source"],
            ),
            runtime_revision_independently_verified=cast(
                bool,
                value["runtime_revision_independently_verified"],
            ),
            covariance_semantics_id=cast(
                str,
                value["covariance_semantics_id"],
            ),
            covariance_policy_id=cast(str, value["covariance_policy_id"]),
            recursive_nuisance_policy_id=cast(
                str,
                value["recursive_nuisance_policy_id"],
            ),
            previous_step_id=cast(str | None, value["previous_step_id"]),
            reason=cast(str, value["reason"]),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            step_id=cast(str, value["step_id"]),
        )


@dataclass(frozen=True, slots=True)
class ClaimBearingProb4DStreamRunV1:
    """Append-only complete-belief history for one exact Prob4D stream."""

    stream_artifact_id: str
    initial_belief_id: str
    recursive_nuisance_policy_id: str
    steps: Sequence[ClaimBearingProb4DStreamStepV1] = ()
    provider_manifest_id: str | None = None
    calibration_artifact_ids: Mapping[str, str] = field(default_factory=dict)
    runtime_revision_source: str | None = None
    runtime_revision_independently_verified: bool = False
    covariance_policy_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def __post_init__(self) -> None:
        stream_id = _sha256(self.stream_artifact_id, name="stream_artifact_id")
        initial_id = _sha256(self.initial_belief_id, name="initial_belief_id")
        nuisance_policy_id = _sha256(
            self.recursive_nuisance_policy_id,
            name="recursive_nuisance_policy_id",
        )
        steps = tuple(self.steps)
        if any(
            not isinstance(step, ClaimBearingProb4DStreamStepV1) for step in steps
        ):
            raise ValueError(
                "steps must contain ClaimBearingProb4DStreamStepV1 objects"
            )
        provider_id = self.provider_manifest_id
        calibration_ids = _optional_calibration_ids(self.calibration_artifact_ids)
        runtime_source = self.runtime_revision_source
        runtime_verified = genuine_boolean(
            self.runtime_revision_independently_verified,
            name="runtime_revision_independently_verified",
        )
        covariance_policy_id = self.covariance_policy_id
        if steps:
            provider_id = _sha256(provider_id, name="provider_manifest_id")
            if not calibration_ids:
                raise ValueError("a populated run must bind calibration artifacts")
            runtime_source = _nonempty_literal_string(
                runtime_source,
                name="runtime_revision_source",
            )
            if not runtime_verified:
                raise ValueError("a populated run must bind verified runtime evidence")
            covariance_policy_id = _sha256(
                covariance_policy_id,
                name="covariance_policy_id",
            )
        elif any(
            value is not None
            for value in (provider_id, runtime_source, covariance_policy_id)
        ) or calibration_ids or runtime_verified:
            raise ValueError("an empty run cannot predeclare update-derived locks")

        previous: ClaimBearingProb4DStreamStepV1 | None = None
        expected_prior = initial_id
        for index, step in enumerate(steps):
            if step.update_index != index:
                raise ValueError("run step indices must be contiguous from zero")
            if step.stream_artifact_id != stream_id:
                raise ValueError("run step identifies a different Prob4D stream")
            if step.prior_belief_id != expected_prior:
                raise ValueError("run belief chain is broken")
            expected_previous = None if previous is None else previous.step_id
            if step.previous_step_id != expected_previous:
                raise ValueError("run step hash chain is broken")
            if step.provider_manifest_id != provider_id:
                raise ValueError("run provider manifest changed between updates")
            if dict(step.calibration_artifact_ids) != dict(calibration_ids):
                raise ValueError("run calibration artifacts changed between updates")
            if step.runtime_revision_source != runtime_source:
                raise ValueError("run runtime revision source changed")
            if not step.runtime_revision_independently_verified:
                raise ValueError("run step lacks verified runtime evidence")
            if step.covariance_policy_id != covariance_policy_id:
                raise ValueError("run covariance interpretation policy changed")
            if step.recursive_nuisance_policy_id != nuisance_policy_id:
                raise ValueError("run recursive nuisance policy changed")
            expected_prior = step.selected_belief_id
            previous = step

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="claim-bearing Prob4D stream run metadata",
        )
        object.__setattr__(self, "stream_artifact_id", stream_id)
        object.__setattr__(self, "initial_belief_id", initial_id)
        object.__setattr__(
            self,
            "recursive_nuisance_policy_id",
            nuisance_policy_id,
        )
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "provider_manifest_id", provider_id)
        object.__setattr__(self, "calibration_artifact_ids", calibration_ids)
        object.__setattr__(self, "runtime_revision_source", runtime_source)
        object.__setattr__(
            self,
            "runtime_revision_independently_verified",
            runtime_verified,
        )
        object.__setattr__(self, "covariance_policy_id", covariance_policy_id)
        object.__setattr__(self, "metadata", metadata)
        expected_id = content_id(self.descriptor())
        supplied_id = self.run_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="run_id")
            if supplied_id != expected_id:
                raise ValueError("claim-bearing stream run_id does not match content")
        object.__setattr__(self, "run_id", expected_id)

    @property
    def final_belief_id(self) -> str:
        if not self.steps:
            return self.initial_belief_id
        return self.steps[-1].selected_belief_id

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CLAIM_BEARING_PROB4D_STREAM_RUN_SCHEMA,
            "schema_version": CLAIM_BEARING_PROB4D_STREAM_RUN_VERSION,
            "stream_artifact_id": self.stream_artifact_id,
            "initial_belief_id": self.initial_belief_id,
            "recursive_nuisance_policy_id": (
                self.recursive_nuisance_policy_id
            ),
            "provider_manifest_id": self.provider_manifest_id,
            "calibration_artifact_ids": dict(self.calibration_artifact_ids),
            "runtime_revision_source": self.runtime_revision_source,
            "runtime_revision_independently_verified": (
                self.runtime_revision_independently_verified
            ),
            "covariance_policy_id": self.covariance_policy_id,
            "steps": [step.to_record() for step in self.steps],
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "run_id": self.run_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "claim-bearing Prob4D stream run",
    ) -> ClaimBearingProb4DStreamRunV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if set(value) != _RUN_FIELDS:
            raise ValueError(f"{name} fields changed")
        if value["schema"] != CLAIM_BEARING_PROB4D_STREAM_RUN_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != CLAIM_BEARING_PROB4D_STREAM_RUN_VERSION:
            raise ValueError(f"{name} version changed")
        raw_steps = value["steps"]
        if not isinstance(raw_steps, list):
            raise ValueError(f"{name} steps must be an array")
        return cls(
            stream_artifact_id=cast(str, value["stream_artifact_id"]),
            initial_belief_id=cast(str, value["initial_belief_id"]),
            recursive_nuisance_policy_id=cast(
                str,
                value["recursive_nuisance_policy_id"],
            ),
            provider_manifest_id=cast(str | None, value["provider_manifest_id"]),
            calibration_artifact_ids=cast(
                Mapping[str, str],
                value["calibration_artifact_ids"],
            ),
            runtime_revision_source=cast(
                str | None,
                value["runtime_revision_source"],
            ),
            runtime_revision_independently_verified=cast(
                bool,
                value["runtime_revision_independently_verified"],
            ),
            covariance_policy_id=cast(
                str | None,
                value["covariance_policy_id"],
            ),
            steps=tuple(
                ClaimBearingProb4DStreamStepV1.from_mapping(
                    step,
                    name=f"{name} step {index}",
                )
                for index, step in enumerate(raw_steps)
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            run_id=cast(str, value["run_id"]),
        )
