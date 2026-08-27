"""Bind the frozen Causal4D acquisition to cross-action transport evidence.

The Causal4D physical design is a balanced incomplete block over four actions:
each independent grasp session contains exactly one unordered action pair, every
pair is repeated once at each of three contact regions, and both directions are
scored within the session.  This module validates that design, builds the
session-specific transport and placebo protocols before target access, and
combines their independently fail-closed decisions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, cast

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.cross_action_placebo_v1 import (
    PlaceboArm,
    PlaceboDecision,
)
from bayesian_phystwin_experiments.cross_action_placebo_v2 import (
    CrossActionPlaceboProtocolV2,
    CrossActionPlaceboResultV2,
)
from bayesian_phystwin_experiments.cross_action_transport_contracts_v1 import (
    TransportArm,
    TransportDecision,
    _commit,
    _digest,
)
from bayesian_phystwin_experiments.cross_action_transport_v2 import (
    CrossActionProtocolV2,
    CrossActionTransportResultV2,
    SessionActionSetV2,
)

CAUSAL4D_CROSS_ACTION_SCHEMA: Final = (
    "bayesian_phystwin.causal4d_cross_action_registration"
)
CAUSAL4D_CROSS_ACTION_VERSION: Final = 1
CAUSAL4D_SLOTH_PROTOCOL_ID: Final = "causal4d-sloth-multi-action-v1"
CAUSAL4D_SLOTH_DESIGN_SHA256: Final = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
CAUSAL4D_SLOTH_ACTION_IDS: Final = (
    "lateral_low",
    "lift_high",
    "lift_low",
    "lower_high",
)
CAUSAL4D_SLOTH_CONTACT_IDS: Final = (
    "left_forepaw",
    "right_forepaw",
    "upper_torso",
)
CAUSAL4D_SLOTH_SESSION_COUNT: Final = 18
CAUSAL4D_SLOTH_EXECUTION_COUNT: Final = 36
CAUSAL4D_CROSS_ACTION_CLAIM_BOUNDARY: Final = (
    "A positive joint result establishes bounded transport of one exact guarded "
    "physical candidate across the frozen paired-action Causal4D acquisition, "
    "including separation from the registered deterministic, discrepancy, and "
    "broken-mechanism controls. It does not establish a unique physical cause, "
    "arbitrary-action or unseen-object generalization, calibrated raw posterior "
    "uncertainty, real Prob4D competence, deployment safety, or general "
    "deformable-object state of the art."
)


class JointTransportDecision(str, Enum):
    """Conjunctive decision over transport and placebo evidence."""

    SUPPORTED = "causal4d_physical_transport_supported"
    NOT_SUPPORTED = "causal4d_physical_transport_not_supported"
    INSUFFICIENT_SESSIONS = "causal4d_physical_transport_insufficient_sessions"


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be literal strings")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return cast(Sequence[Any], value)


def _label(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a literal integer of at least {minimum}")
    return value


def _true(value: object, *, name: str) -> bool:
    result = genuine_boolean(value, name=name)
    if not result:
        raise ValueError(f"{name} must be true")
    return result


def causal4d_protocol_design_sha256(protocol: Mapping[str, Any]) -> str:
    """Reproduce Causal4D's canonical self-digest calculation."""

    payload = plain_json(_mapping(protocol, name="protocol"))
    if not isinstance(payload, dict):
        raise TypeError("protocol must serialize to a JSON object")
    payload.pop("design_sha256", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class Causal4DCrossActionDesignV1:
    """Validated paired-action view of one immutable Causal4D protocol."""

    protocol_id: str
    design_sha256: str
    session_action_sets: tuple[SessionActionSetV2, ...]
    session_contact_ids: Mapping[str, str]
    execution_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    contact_ids: tuple[str, ...]
    analysis_split_unit: str
    target_outcomes_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_id",
            _label(self.protocol_id, name="protocol_id"),
        )
        object.__setattr__(
            self,
            "design_sha256",
            _digest(self.design_sha256, name="design_sha256"),
        )
        session_sets = tuple(
            sorted(self.session_action_sets, key=lambda value: value.object_session_id)
        )
        if not session_sets or any(
            not isinstance(value, SessionActionSetV2) for value in session_sets
        ):
            raise TypeError(
                "session_action_sets must contain SessionActionSetV2 values"
            )
        session_ids = tuple(value.object_session_id for value in session_sets)
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("session_action_sets must not repeat a session")
        object.__setattr__(self, "session_action_sets", session_sets)

        contacts = _mapping(self.session_contact_ids, name="session_contact_ids")
        if set(contacts) != set(session_ids):
            raise ValueError("session_contact_ids must cover the exact session roster")
        frozen_contacts = frozen_finite_json_mapping(
            {
                session: _label(contacts[session], name="contact id")
                for session in session_ids
            },
            name="session_contact_ids",
        )
        object.__setattr__(self, "session_contact_ids", frozen_contacts)

        executions = tuple(
            sorted(_label(value, name="execution_id") for value in self.execution_ids)
        )
        if not executions or len(executions) != len(set(executions)):
            raise ValueError("execution_ids must be nonempty and unique")
        object.__setattr__(self, "execution_ids", executions)

        actions = tuple(
            sorted(_label(value, name="action_id") for value in self.action_ids)
        )
        if len(actions) < 2 or len(actions) != len(set(actions)):
            raise ValueError("action_ids must contain at least two unique values")
        object.__setattr__(self, "action_ids", actions)
        contact_ids = tuple(
            sorted(_label(value, name="contact_id") for value in self.contact_ids)
        )
        if not contact_ids or len(contact_ids) != len(set(contact_ids)):
            raise ValueError("contact_ids must be nonempty and unique")
        object.__setattr__(self, "contact_ids", contact_ids)
        object.__setattr__(
            self,
            "analysis_split_unit",
            _label(self.analysis_split_unit, name="analysis_split_unit"),
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if target_used:
            raise ValueError("design extraction must not use target outcomes")
        object.__setattr__(self, "target_outcomes_used", target_used)

    @property
    def target_session_ids(self) -> tuple[str, ...]:
        return tuple(value.object_session_id for value in self.session_action_sets)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CAUSAL4D_CROSS_ACTION_SCHEMA,
            "schema_version": CAUSAL4D_CROSS_ACTION_VERSION,
            "artifact_kind": "Causal4DCrossActionDesignV1",
            "protocol_id": self.protocol_id,
            "design_sha256": self.design_sha256,
            "session_action_sets": [
                value.descriptor() for value in self.session_action_sets
            ],
            "session_contact_ids": plain_json(self.session_contact_ids),
            "execution_ids": list(self.execution_ids),
            "action_ids": list(self.action_ids),
            "contact_ids": list(self.contact_ids),
            "analysis_split_unit": self.analysis_split_unit,
            "target_outcomes_used": self.target_outcomes_used,
        }

    @property
    def design_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


def extract_causal4d_cross_action_design_v1(
    protocol: Mapping[str, Any],
    *,
    expected_protocol_id: str = CAUSAL4D_SLOTH_PROTOCOL_ID,
    expected_design_sha256: str = CAUSAL4D_SLOTH_DESIGN_SHA256,
    expected_action_ids: Sequence[str] = CAUSAL4D_SLOTH_ACTION_IDS,
    expected_contact_ids: Sequence[str] = CAUSAL4D_SLOTH_CONTACT_IDS,
    expected_session_count: int = CAUSAL4D_SLOTH_SESSION_COUNT,
    expected_execution_count: int = CAUSAL4D_SLOTH_EXECUTION_COUNT,
) -> Causal4DCrossActionDesignV1:
    """Validate and extract the frozen balanced paired-action acquisition."""

    document = _mapping(protocol, name="protocol")
    protocol_id = _label(document.get("protocol_id"), name="protocol_id")
    if protocol_id != expected_protocol_id:
        raise ValueError("Causal4D protocol_id does not match the registered design")
    if _integer(document.get("schema_version"), name="schema_version") != 1:
        raise ValueError("Causal4D schema_version must be 1")
    stored_digest = _digest(document.get("design_sha256"), name="design_sha256")
    calculated_digest = causal4d_protocol_design_sha256(document)
    if stored_digest != calculated_digest:
        raise ValueError("Causal4D protocol design_sha256 is not self-consistent")
    if stored_digest != _digest(
        expected_design_sha256,
        name="expected_design_sha256",
    ):
        raise ValueError(
            "Causal4D protocol does not match the registered design digest"
        )

    analysis_lock = _mapping(document.get("analysis_lock"), name="analysis_lock")
    split_unit = _label(analysis_lock.get("split_unit"), name="analysis split_unit")
    if split_unit != "grasp_session":
        raise ValueError("Causal4D independent unit must remain grasp_session")
    for key in (
        "exclusions_locked_before_target_evaluation",
        "no_held_contact_or_profile_in_fold_source_sessions",
        "no_session_shared_between_fit_and_calibration",
        "target_outcomes_may_not_select_hyperparameters",
    ):
        _true(analysis_lock.get(key), name=f"analysis_lock[{key!r}]")

    expected_actions = tuple(
        sorted(_label(value, name="action id") for value in expected_action_ids)
    )
    profiles = _sequence(document.get("command_profiles"), name="command_profiles")
    profile_ids = tuple(
        sorted(
            _label(
                _mapping(value, name="command profile").get("id"),
                name="command profile id",
            )
            for value in profiles
        )
    )
    if profile_ids != expected_actions or len(profile_ids) != len(set(profile_ids)):
        raise ValueError("Causal4D command profile roster changed")

    expected_contacts = tuple(
        sorted(_label(value, name="contact id") for value in expected_contact_ids)
    )
    contacts = _sequence(document.get("contact_regions"), name="contact_regions")
    contact_ids = tuple(
        sorted(
            _label(
                _mapping(value, name="contact region").get("id"),
                name="contact region id",
            )
            for value in contacts
        )
    )
    if contact_ids != expected_contacts or len(contact_ids) != len(set(contact_ids)):
        raise ValueError("Causal4D contact-region roster changed")

    session_order_values = _sequence(
        document.get("acquisition_session_order"),
        name="acquisition_session_order",
    )
    session_order = tuple(
        _label(value, name="acquisition session id") for value in session_order_values
    )
    if len(session_order) != expected_session_count:
        raise ValueError("Causal4D session count changed")
    if len(session_order) != len(set(session_order)):
        raise ValueError("Causal4D acquisition_session_order repeats a session")

    execution_values = _sequence(document.get("executions"), name="executions")
    if len(execution_values) != expected_execution_count:
        raise ValueError("Causal4D execution count changed")
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    execution_ids: list[str] = []
    execution_indices: list[int] = []
    for value in execution_values:
        row = _mapping(value, name="execution")
        session_id = _label(row.get("session_id"), name="execution session_id")
        if session_id not in set(session_order):
            raise ValueError("execution references an unregistered session")
        execution_id = _label(row.get("execution_id"), name="execution_id")
        action_id = _label(row.get("command_profile_id"), name="command_profile_id")
        contact_id = _label(row.get("contact_region_id"), name="contact_region_id")
        if action_id not in expected_actions:
            raise ValueError("execution references an unregistered action")
        if contact_id not in expected_contacts:
            raise ValueError("execution references an unregistered contact")
        pair_order = _integer(row.get("pair_order"), name="pair_order")
        if pair_order not in {0, 1}:
            raise ValueError("pair_order must be exactly 0 or 1")
        execution_index = _integer(
            row.get("acquisition_execution_index"),
            name="acquisition_execution_index",
        )
        by_session[session_id].append(row)
        execution_ids.append(execution_id)
        execution_indices.append(execution_index)
    if len(execution_ids) != len(set(execution_ids)):
        raise ValueError("Causal4D execution_id values must be unique")
    if set(execution_indices) != set(range(expected_execution_count)):
        raise ValueError("Causal4D acquisition execution indices changed")
    if set(by_session) != set(session_order):
        raise ValueError("execution table does not cover the frozen session roster")

    session_sets: list[SessionActionSetV2] = []
    session_contacts: dict[str, str] = {}
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_contact_counts: Counter[tuple[tuple[str, str], str]] = Counter()
    action_session_counts: Counter[str] = Counter()
    contact_session_counts: Counter[str] = Counter()
    for session_id in session_order:
        rows = by_session[session_id]
        if len(rows) != 2:
            raise ValueError(
                "every Causal4D session must contain exactly two executions"
            )
        if {_integer(row.get("pair_order"), name="pair_order") for row in rows} != {
            0,
            1,
        }:
            raise ValueError("every Causal4D session must contain pair orders 0 and 1")
        actions = tuple(
            sorted(
                _label(row.get("command_profile_id"), name="command_profile_id")
                for row in rows
            )
        )
        if len(actions) != 2 or len(set(actions)) != 2:
            raise ValueError("each Causal4D session must contain two distinct actions")
        contacts_for_session = {
            _label(row.get("contact_region_id"), name="contact_region_id")
            for row in rows
        }
        if len(contacts_for_session) != 1:
            raise ValueError("both executions in a session must share one contact")
        contact_id = next(iter(contacts_for_session))
        session_sets.append(
            SessionActionSetV2(
                object_session_id=session_id,
                action_ids=actions,
            )
        )
        session_contacts[session_id] = contact_id
        pair_counts[actions] += 1
        pair_contact_counts[(actions, contact_id)] += 1
        contact_session_counts[contact_id] += 1
        for action in actions:
            action_session_counts[action] += 1

    expected_pairs = set(itertools.combinations(expected_actions, 2))
    if set(pair_counts) != expected_pairs:
        raise ValueError("Causal4D action-pair coverage is incomplete")
    expected_repetitions = len(expected_contacts)
    if any(pair_counts[pair] != expected_repetitions for pair in expected_pairs):
        raise ValueError("each action pair must repeat once per contact region")
    if any(
        pair_contact_counts[(pair, contact)] != 1
        for pair in expected_pairs
        for contact in expected_contacts
    ):
        raise ValueError("each action pair must occur exactly once at each contact")
    expected_sessions_per_contact = len(expected_pairs)
    if any(
        contact_session_counts[contact] != expected_sessions_per_contact
        for contact in expected_contacts
    ):
        raise ValueError("contact regions must contain all six action pairs")
    expected_sessions_per_action = (len(expected_actions) - 1) * len(
        expected_contacts
    )
    if any(
        action_session_counts[action] != expected_sessions_per_action
        for action in expected_actions
    ):
        raise ValueError("the balanced action incidence pattern changed")

    return Causal4DCrossActionDesignV1(
        protocol_id=protocol_id,
        design_sha256=stored_digest,
        session_action_sets=tuple(session_sets),
        session_contact_ids=session_contacts,
        execution_ids=tuple(execution_ids),
        action_ids=expected_actions,
        contact_ids=expected_contacts,
        analysis_split_unit=split_unit,
    )


@dataclass(frozen=True, slots=True)
class Causal4DCrossActionRegistrationV1:
    """Claim-bearing pre-target binding of Causal4D and BayesianPhysTwin."""

    design: Causal4DCrossActionDesignV1
    transport_protocol: CrossActionProtocolV2
    placebo_protocol: CrossActionPlaceboProtocolV2
    causal4d_revision: str
    bayesian_phystwin_revision: str
    causal4d_amendment_id: str
    causal4d_method_freeze_id: str
    causal4d_method_freeze_attestation_id: str
    causal4d_source_panel_id: str
    causal4d_readiness_id: str
    causal4d_primary_analysis_id: str
    causal4d_target_access_policy_id: str
    bayesian_phystwin_distribution_id: str
    prob4d_usage_declaration_id: str
    prediction_batch_policy_id: str
    registration_frozen_before_first_execution: bool = True
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.design, Causal4DCrossActionDesignV1):
            raise TypeError("design must be Causal4DCrossActionDesignV1")
        if not isinstance(self.transport_protocol, CrossActionProtocolV2):
            raise TypeError("transport_protocol must be CrossActionProtocolV2")
        if not isinstance(self.placebo_protocol, CrossActionPlaceboProtocolV2):
            raise TypeError("placebo_protocol must be CrossActionPlaceboProtocolV2")
        object.__setattr__(
            self,
            "causal4d_revision",
            _commit(self.causal4d_revision, name="causal4d_revision"),
        )
        object.__setattr__(
            self,
            "bayesian_phystwin_revision",
            _commit(
                self.bayesian_phystwin_revision,
                name="bayesian_phystwin_revision",
            ),
        )
        for name in (
            "causal4d_amendment_id",
            "causal4d_method_freeze_id",
            "causal4d_method_freeze_attestation_id",
            "causal4d_source_panel_id",
            "causal4d_readiness_id",
            "causal4d_primary_analysis_id",
            "causal4d_target_access_policy_id",
            "bayesian_phystwin_distribution_id",
            "prob4d_usage_declaration_id",
            "prediction_batch_policy_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))

        design_sets = tuple(
            value.descriptor() for value in self.design.session_action_sets
        )
        transport_sets = tuple(
            value.descriptor() for value in self.transport_protocol.session_action_sets
        )
        placebo_sets = tuple(
            value.descriptor() for value in self.placebo_protocol.session_action_sets
        )
        if transport_sets != design_sets or placebo_sets != design_sets:
            raise ValueError(
                "transport and placebo protocols must bind the exact design"
            )
        if self.transport_protocol.acquisition_binding_id != self.design.design_id:
            raise ValueError(
                "transport protocol must bind the exact design certificate"
            )
        if (
            self.transport_protocol.target_roster_id
            != self.placebo_protocol.target_roster_id
        ):
            raise ValueError("transport and placebo target rosters must match")
        if (
            self.placebo_protocol.parent_transport_protocol_id
            != self.transport_protocol.protocol_id
        ):
            raise ValueError("placebo protocol must bind the transport protocol")
        if (
            self.transport_protocol.target_access_policy_id
            != self.causal4d_target_access_policy_id
        ):
            raise ValueError("target-access policy must match the Causal4D freeze")

        frozen = genuine_boolean(
            self.registration_frozen_before_first_execution,
            name="registration_frozen_before_first_execution",
        )
        target_used = genuine_boolean(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        if not frozen or target_used:
            raise ValueError(
                "registration must be frozen before execution and target-outcome free"
            )
        object.__setattr__(self, "registration_frozen_before_first_execution", frozen)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="registration metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CAUSAL4D_CROSS_ACTION_SCHEMA,
            "schema_version": CAUSAL4D_CROSS_ACTION_VERSION,
            "artifact_kind": "Causal4DCrossActionRegistrationV1",
            "design_id": self.design.design_id,
            "transport_protocol_id": self.transport_protocol.protocol_id,
            "placebo_protocol_id": self.placebo_protocol.protocol_id,
            "causal4d_revision": self.causal4d_revision,
            "bayesian_phystwin_revision": self.bayesian_phystwin_revision,
            "causal4d_amendment_id": self.causal4d_amendment_id,
            "causal4d_method_freeze_id": self.causal4d_method_freeze_id,
            "causal4d_method_freeze_attestation_id": (
                self.causal4d_method_freeze_attestation_id
            ),
            "causal4d_source_panel_id": self.causal4d_source_panel_id,
            "causal4d_readiness_id": self.causal4d_readiness_id,
            "causal4d_primary_analysis_id": self.causal4d_primary_analysis_id,
            "causal4d_target_access_policy_id": (
                self.causal4d_target_access_policy_id
            ),
            "bayesian_phystwin_distribution_id": (
                self.bayesian_phystwin_distribution_id
            ),
            "prob4d_usage_declaration_id": self.prob4d_usage_declaration_id,
            "prediction_batch_policy_id": self.prediction_batch_policy_id,
            "registration_frozen_before_first_execution": (
                self.registration_frozen_before_first_execution
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CAUSAL4D_CROSS_ACTION_CLAIM_BOUNDARY,
        }

    @property
    def registration_id(self) -> str:
        return cast(str, content_id(self.descriptor()))


def build_causal4d_cross_action_registration_v1(
    protocol: Mapping[str, Any],
    *,
    causal4d_revision: str,
    bayesian_phystwin_revision: str,
    causal4d_amendment_id: str,
    causal4d_method_freeze_id: str,
    causal4d_method_freeze_attestation_id: str,
    causal4d_source_panel_id: str,
    causal4d_readiness_id: str,
    causal4d_primary_analysis_id: str,
    causal4d_target_access_policy_id: str,
    bayesian_phystwin_distribution_id: str,
    prob4d_usage_declaration_id: str,
    prediction_batch_policy_id: str,
    development_roster_id: str,
    calibration_roster_id: str,
    target_roster_id: str,
    query_id: str,
    query_jacobian_id: str,
    identifiability_certificate_id: str,
    nonlinear_closure_certificate_id: str,
    score_definition_id: str,
    grouping_rule_id: str,
    interval_method_id: str,
    model_stack_id: str,
    numerical_environment_id: str,
    technical_failure_policy_id: str,
    placebo_arm_construction_ids: Mapping[str, str],
    minimum_sessions: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
    minimum_off_diagonal_gain: float,
    minimum_discrepancy_contrast: float,
    minimum_comparator_contrast: float,
    maximum_harmful_session_fraction: float,
    minimum_placebo_contrast: float,
    harmful_gain_margin: float = 0.0,
    expected_protocol_id: str = CAUSAL4D_SLOTH_PROTOCOL_ID,
    expected_design_sha256: str = CAUSAL4D_SLOTH_DESIGN_SHA256,
    expected_action_ids: Sequence[str] = CAUSAL4D_SLOTH_ACTION_IDS,
    expected_contact_ids: Sequence[str] = CAUSAL4D_SLOTH_CONTACT_IDS,
    expected_session_count: int = CAUSAL4D_SLOTH_SESSION_COUNT,
    expected_execution_count: int = CAUSAL4D_SLOTH_EXECUTION_COUNT,
    metadata: Mapping[str, Any] | None = None,
) -> Causal4DCrossActionRegistrationV1:
    """Build the complete target-blind registration from frozen identities."""

    design = extract_causal4d_cross_action_design_v1(
        protocol,
        expected_protocol_id=expected_protocol_id,
        expected_design_sha256=expected_design_sha256,
        expected_action_ids=expected_action_ids,
        expected_contact_ids=expected_contact_ids,
        expected_session_count=expected_session_count,
        expected_execution_count=expected_execution_count,
    )
    registered_arms = (
        TransportArm.PHYSICAL_FALLBACK,
        TransportArm.LAST_RESIDUAL,
        TransportArm.DISCREPANCY_ONLY,
        TransportArm.STATE_ONLY,
        TransportArm.STATE_PARAMETER,
        TransportArm.GUARDED_PHYSICAL,
    )
    shared_metadata = {
        **dict(metadata or {}),
        "causal4d_protocol_id": design.protocol_id,
        "causal4d_design_sha256": design.design_sha256,
        "balanced_incomplete_action_design": True,
    }
    transport = CrossActionProtocolV2(
        development_roster_id=development_roster_id,
        calibration_roster_id=calibration_roster_id,
        target_roster_id=target_roster_id,
        acquisition_binding_id=design.design_id,
        query_id=query_id,
        query_jacobian_id=query_jacobian_id,
        identifiability_certificate_id=identifiability_certificate_id,
        nonlinear_closure_certificate_id=nonlinear_closure_certificate_id,
        score_definition_id=score_definition_id,
        grouping_rule_id=grouping_rule_id,
        interval_method_id=interval_method_id,
        target_access_policy_id=causal4d_target_access_policy_id,
        model_stack_id=model_stack_id,
        numerical_environment_id=numerical_environment_id,
        technical_failure_policy_id=technical_failure_policy_id,
        session_action_sets=design.session_action_sets,
        registered_arms=registered_arms,
        physical_transport_arm=TransportArm.GUARDED_PHYSICAL,
        discrepancy_reference_arm=TransportArm.DISCREPANCY_ONLY,
        matched_comparator_arm=TransportArm.LAST_RESIDUAL,
        minimum_sessions=minimum_sessions,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        minimum_off_diagonal_gain=minimum_off_diagonal_gain,
        minimum_discrepancy_contrast=minimum_discrepancy_contrast,
        minimum_comparator_contrast=minimum_comparator_contrast,
        maximum_harmful_session_fraction=maximum_harmful_session_fraction,
        harmful_gain_margin=harmful_gain_margin,
        metadata=shared_metadata,
    )
    placebos = (
        PlaceboArm.WRONG_ACTION,
        PlaceboArm.WRONG_OBJECT,
        PlaceboArm.PHASE_SHIFTED,
        PlaceboArm.IDENTITY_PERMUTED,
    )
    placebo = CrossActionPlaceboProtocolV2(
        parent_transport_protocol_id=transport.protocol_id,
        target_roster_id=target_roster_id,
        session_action_sets=design.session_action_sets,
        physical_arm_label=TransportArm.GUARDED_PHYSICAL.value,
        placebo_arms=placebos,
        arm_construction_ids=placebo_arm_construction_ids,
        minimum_sessions=minimum_sessions,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 100_000,
        confidence_level=confidence_level,
        minimum_placebo_contrast=minimum_placebo_contrast,
        metadata=shared_metadata,
    )
    return Causal4DCrossActionRegistrationV1(
        design=design,
        transport_protocol=transport,
        placebo_protocol=placebo,
        causal4d_revision=causal4d_revision,
        bayesian_phystwin_revision=bayesian_phystwin_revision,
        causal4d_amendment_id=causal4d_amendment_id,
        causal4d_method_freeze_id=causal4d_method_freeze_id,
        causal4d_method_freeze_attestation_id=(
            causal4d_method_freeze_attestation_id
        ),
        causal4d_source_panel_id=causal4d_source_panel_id,
        causal4d_readiness_id=causal4d_readiness_id,
        causal4d_primary_analysis_id=causal4d_primary_analysis_id,
        causal4d_target_access_policy_id=causal4d_target_access_policy_id,
        bayesian_phystwin_distribution_id=bayesian_phystwin_distribution_id,
        prob4d_usage_declaration_id=prob4d_usage_declaration_id,
        prediction_batch_policy_id=prediction_batch_policy_id,
        metadata=shared_metadata,
    )


@dataclass(frozen=True, slots=True)
class Causal4DJointTransportResultV1:
    """Conjunctive transport-plus-placebo result for one registration."""

    registration: Causal4DCrossActionRegistrationV1
    transport_result: CrossActionTransportResultV2
    placebo_result: CrossActionPlaceboResultV2

    decision: JointTransportDecision = field(init=False)
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.registration, Causal4DCrossActionRegistrationV1):
            raise TypeError("registration must be Causal4DCrossActionRegistrationV1")
        if not isinstance(self.transport_result, CrossActionTransportResultV2):
            raise TypeError("transport_result must be CrossActionTransportResultV2")
        if not isinstance(self.placebo_result, CrossActionPlaceboResultV2):
            raise TypeError("placebo_result must be CrossActionPlaceboResultV2")
        if (
            self.transport_result.protocol.protocol_id
            != self.registration.transport_protocol.protocol_id
        ):
            raise ValueError("transport result does not match the registration")
        if (
            self.placebo_result.protocol.protocol_id
            != self.registration.placebo_protocol.protocol_id
        ):
            raise ValueError("placebo result does not match the registration")
        if (
            self.transport_result.target_accounting_id
            != self.placebo_result.target_accounting_id
        ):
            raise ValueError("transport and placebo target accounting must match")
        if (
            self.transport_result.excluded_session_ids
            != self.placebo_result.excluded_session_ids
        ):
            raise ValueError("transport and placebo exclusions must match")

        transport_attestations = {
            row.target_access_attestation_id for row in self.transport_result.score_rows
        }
        placebo_attestations = {
            row.target_access_attestation_id for row in self.placebo_result.score_rows
        }
        if transport_attestations != placebo_attestations or len(
            transport_attestations
        ) != 1:
            raise ValueError("transport and placebo rows must share one target opening")
        transport_scorers = {row.scorer_id for row in self.transport_result.score_rows}
        placebo_scorers = {row.scorer_id for row in self.placebo_result.score_rows}
        if transport_scorers != placebo_scorers or len(transport_scorers) != 1:
            raise ValueError("transport and placebo rows must share one frozen scorer")
        transport_commits = {
            row.prediction.commit_id for row in self.transport_result.score_rows
        }
        placebo_commits = {
            row.prediction.commit_id for row in self.placebo_result.score_rows
        }
        expected_revision = self.registration.bayesian_phystwin_revision
        if transport_commits != {expected_revision} or placebo_commits != {
            expected_revision
        }:
            raise ValueError("all predictions must bind the registered BPT revision")

        physical_rows = {
            (
                row.prediction.object_session_id,
                row.prediction.source_action_id,
                row.prediction.target_action_id,
            ): row
            for row in self.transport_result.score_rows
            if row.prediction.arm
            is self.registration.transport_protocol.physical_transport_arm
            and row.prediction.source_action_id != row.prediction.target_action_id
        }
        for row in self.placebo_result.score_rows:
            prediction = row.prediction
            key = (
                prediction.object_session_id,
                prediction.source_action_id,
                prediction.target_action_id,
            )
            parent_row = physical_rows.get(key)
            if parent_row is None or prediction.parent_transport_prediction_id != (
                parent_row.prediction.prediction_id
            ):
                raise ValueError(
                    "placebo predictions must bind the exact physical transport "
                    "prediction"
                )
            if row.target_outcome_id != parent_row.target_outcome_id:
                raise ValueError(
                    "placebo and transport rows must score the same outcome"
                )
            parent = parent_row.prediction
            expected_selected = (
                parent.disposition.value == "candidate_selected"
            )
            expected_fallback = parent.disposition.value == "exact_fallback"
            if (
                prediction.candidate_selected != expected_selected
                or prediction.exact_fallback != expected_fallback
            ):
                raise ValueError(
                    "placebo predictions must preserve the parent disposition"
                )

        if (
            self.transport_result.decision
            is TransportDecision.INSUFFICIENT_SESSIONS
            or self.placebo_result.decision is PlaceboDecision.INSUFFICIENT_SESSIONS
        ):
            decision = JointTransportDecision.INSUFFICIENT_SESSIONS
        elif (
            self.transport_result.decision is TransportDecision.SUPPORTED
            and self.placebo_result.decision is PlaceboDecision.SUPPORTED
        ):
            decision = JointTransportDecision.SUPPORTED
        else:
            decision = JointTransportDecision.NOT_SUPPORTED
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "result_id", cast(str, content_id(self.descriptor())))

    @property
    def supports_physical_transport(self) -> bool:
        return self.decision is JointTransportDecision.SUPPORTED

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CAUSAL4D_CROSS_ACTION_SCHEMA,
            "schema_version": CAUSAL4D_CROSS_ACTION_VERSION,
            "artifact_kind": "Causal4DJointTransportResultV1",
            "registration_id": self.registration.registration_id,
            "transport_result_id": self.transport_result.result_id,
            "placebo_result_id": self.placebo_result.result_id,
            "target_accounting_id": self.transport_result.target_accounting_id,
            "excluded_session_ids": list(
                self.transport_result.excluded_session_ids
            ),
            "independent_session_count": (
                self.transport_result.independent_session_count
            ),
            "decision": self.decision.value,
            "supports_physical_transport": self.supports_physical_transport,
            "claim_boundary": CAUSAL4D_CROSS_ACTION_CLAIM_BOUNDARY,
        }


__all__ = [
    "CAUSAL4D_CROSS_ACTION_CLAIM_BOUNDARY",
    "CAUSAL4D_CROSS_ACTION_SCHEMA",
    "CAUSAL4D_CROSS_ACTION_VERSION",
    "CAUSAL4D_SLOTH_ACTION_IDS",
    "CAUSAL4D_SLOTH_CONTACT_IDS",
    "CAUSAL4D_SLOTH_DESIGN_SHA256",
    "CAUSAL4D_SLOTH_EXECUTION_COUNT",
    "CAUSAL4D_SLOTH_PROTOCOL_ID",
    "CAUSAL4D_SLOTH_SESSION_COUNT",
    "Causal4DCrossActionDesignV1",
    "Causal4DCrossActionRegistrationV1",
    "Causal4DJointTransportResultV1",
    "JointTransportDecision",
    "build_causal4d_cross_action_registration_v1",
    "causal4d_protocol_design_sha256",
    "extract_causal4d_cross_action_design_v1",
]
