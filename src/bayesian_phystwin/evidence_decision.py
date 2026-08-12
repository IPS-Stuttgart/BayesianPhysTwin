"""Version-dispatching loader for Bayesian-PhysTwin evidence decisions."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from .evidence_decision_v1 import (
    EVIDENCE_DECISION_SCHEMA as EVIDENCE_DECISION_SCHEMA_V1,
    EVIDENCE_DECISION_SCHEMA_VERSION as EVIDENCE_DECISION_SCHEMA_VERSION_V1,
    EvidenceDecisionV1,
    _load_json_mapping,
    _require_integer,
    load_evidence_decision as load_evidence_decision_v1,
)
from .evidence_decision_v2 import (
    EVIDENCE_DECISION_SCHEMA as EVIDENCE_DECISION_SCHEMA_V2,
    EVIDENCE_DECISION_SCHEMA_VERSION as EVIDENCE_DECISION_SCHEMA_VERSION_V2,
    EvidenceDecisionV2,
    load_evidence_decision_v2,
)

EvidenceDecision: TypeAlias = EvidenceDecisionV1 | EvidenceDecisionV2


def load_evidence_decision(path: str | Path) -> EvidenceDecision:
    """Load a supported evidence decision without guessing its wire version."""

    payload = _load_json_mapping(path, name="evidence decision")
    schema_name = payload.get("schema_name")
    if schema_name not in {
        EVIDENCE_DECISION_SCHEMA_V1,
        EVIDENCE_DECISION_SCHEMA_V2,
    }:
        raise ValueError("unsupported evidence-decision schema")
    version = _require_integer(payload.get("schema_version"), name="schema_version")
    if version == EVIDENCE_DECISION_SCHEMA_VERSION_V1:
        return load_evidence_decision_v1(path)
    if version == EVIDENCE_DECISION_SCHEMA_VERSION_V2:
        return load_evidence_decision_v2(path)
    raise ValueError("unsupported evidence-decision schema version")


__all__ = ["EvidenceDecision", "load_evidence_decision"]
