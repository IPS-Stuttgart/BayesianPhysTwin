"""Portable component evidence and joint portfolio-certificate assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from ._portable_contracts import content_id
from .query_portfolio_replication_v1 import QUERY_IDS, QueryOutcomeV1
from .query_portfolio_replication_v4 import protocol, score

SCHEMA = "bayesian_phystwin.query_portfolio_component_evidence"
VERSION = 1


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("SHA-256/content identity must be a 64-character string")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("invalid hexadecimal SHA-256/content identity") from error
    return value


def component_evidence(
    outcome: QueryOutcomeV1,
    *,
    component_result_id: str,
    component_result_sha256: str,
) -> dict[str, Any]:
    """Serialize one complete, independently verified query denominator."""

    value: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "portfolio_protocol_id": protocol()["protocol_id"],
        "query_id": outcome.query_id,
        "component_result_id": _digest(component_result_id),
        "component_result_sha256": _digest(component_result_sha256),
        "gain": outcome.gain.tolist(),
        "candidate_deployed": outcome.candidate_deployed.tolist(),
        "ordinary_success": outcome.ordinary_success.tolist(),
        "worlds": len(outcome.gain),
        "partial_result": False,
    }
    value["artifact_id"] = content_id(value)
    return value


def load_component_evidence(value: Mapping[str, Any]) -> QueryOutcomeV1:
    """Validate a component record and recover its immutable query outcome."""

    expected_keys = {
        "schema",
        "version",
        "portfolio_protocol_id",
        "query_id",
        "component_result_id",
        "component_result_sha256",
        "gain",
        "candidate_deployed",
        "ordinary_success",
        "worlds",
        "partial_result",
        "artifact_id",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != SCHEMA
        or value.get("version") != VERSION
        or value.get("portfolio_protocol_id") != protocol()["protocol_id"]
        or value.get("query_id") not in QUERY_IDS
        or value.get("worlds") != 320
        or value.get("partial_result") is not False
    ):
        raise ValueError("invalid portfolio component evidence")
    identity = value.get("artifact_id")
    body = {key: item for key, item in value.items() if key != "artifact_id"}
    if identity != content_id(body):
        raise ValueError("portfolio component evidence identity changed")
    _digest(value.get("component_result_id"))
    _digest(value.get("component_result_sha256"))
    return QueryOutcomeV1(
        query_id=cast(str, value["query_id"]),
        gain=np.asarray(value["gain"], dtype=np.float64),
        candidate_deployed=np.asarray(value["candidate_deployed"], dtype=np.bool_),
        ordinary_success=np.asarray(value["ordinary_success"], dtype=np.bool_),
    )


def assemble(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Assemble the complete two-query familywise certificate."""

    if set(records) != set(QUERY_IDS):
        raise ValueError("exact registered component set required")
    outcomes = {key: load_component_evidence(records[key]) for key in QUERY_IDS}
    result = score(outcomes)
    value: dict[str, Any] = {
        **result,
        "schema": "bayesian_phystwin.query_portfolio_replication.result_evidence_v1",
        "component_evidence": {
            key: {
                "artifact_id": records[key]["artifact_id"],
                "component_result_id": records[key]["component_result_id"],
                "component_result_sha256": records[key]["component_result_sha256"],
            }
            for key in QUERY_IDS
        },
        "partial_results_used": False,
    }
    value.pop("artifact_id")
    value["artifact_id"] = content_id(value)
    return value
