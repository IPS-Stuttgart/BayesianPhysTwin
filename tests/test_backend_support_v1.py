from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
from typing import Any, cast

import pytest

from bayesian_phystwin.backend_support_v1 import (
    FIVE_BACKEND_IDS,
    FIVE_BACKEND_SUPPORT_RESOURCE_SHA256,
    describe_five_backend_support,
    load_five_backend_support,
    validate_five_backend_support,
    verify_five_backend_source_tree,
)

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = (
    ROOT
    / "src"
    / "bayesian_phystwin"
    / "contract_data"
    / "backend_support_v1"
    / "five_backend_support_v1.json"
)


def _backends() -> list[dict[str, Any]]:
    value = describe_five_backend_support()
    return cast(list[dict[str, Any]], value["backends"])


def test_five_backend_support_resource_is_hash_bound_and_exact() -> None:
    assert hashlib.sha256(RESOURCE.read_bytes()).hexdigest() == (
        FIVE_BACKEND_SUPPORT_RESOURCE_SHA256
    )
    support = load_five_backend_support()

    assert support["schema"] == "bayesian-phystwin.five-backend-support"
    assert support["schema_version"] == 1
    assert tuple(item["backend_id"] for item in support["backends"]) == (
        FIVE_BACKEND_IDS
    )
    assert all(
        item["support_status"] == "fully-supported" for item in support["backends"]
    )
    assert all(
        all(item["support_capabilities"].values()) for item in support["backends"]
    )


def test_full_support_is_not_predictive_promotion() -> None:
    backends = {item["backend_id"]: item for item in _backends()}

    assert backends["deform-dlo-v7"]["evidence"]["recommendation_authorized"] is True
    assert backends["deform-dlo-v7"]["evidence"]["stage"] == (
        "benchmark-value-qualified"
    )
    for backend_id in FIVE_BACKEND_IDS[1:]:
        assert backends[backend_id]["support_status"] == "fully-supported"
        assert backends[backend_id]["evidence"]["recommendation_authorized"] is False
        assert backends[backend_id]["evidence"]["latest_gate_decision"] == "rejected"


def test_protected_boundaries_and_exact_fallback_are_universal() -> None:
    for backend in _backends():
        evidence = backend["evidence"]
        assert evidence["protected_target_outcomes_opened"] is False
        assert evidence["held_v8_accessed"] is False
        assert evidence["dlo4_or_dlo5_accessed"] is False
        assert backend["support_capabilities"]["exact_fallback"] is True
        assert "exact" in backend["fallback_semantics"]


def test_repository_support_files_and_evidence_verify() -> None:
    assert verify_five_backend_source_tree(ROOT) == {
        "schema": "bayesian-phystwin.five-backend-support",
        "schema_version": 1,
        "backend_count": 5,
        "fully_supported_backend_count": 5,
        "implementation_file_count": 14,
        "test_file_count": 14,
        "documentation_file_count": 13,
        "evidence_artifact_count": 8,
        "recommendation_authorized_backend_ids": ["deform-dlo-v7"],
        "status": "verified",
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "bayesian_phystwin.backend_support_v1",
        "bayesian_phystwin.jax_fem_hyperelastic_v2",
        "bayesian_phystwin.matphys_native_source_v1",
        "bayesian_phystwin.mujoco_flex_source_v1",
        "bayesian_phystwin.sofa_fem_canonical_source_v3",
        "bayesian_phystwin_experiments.deform_dlo_local_residual",
    ],
)
def test_backend_interfaces_import_without_loading_optional_engines(
    module_name: str,
) -> None:
    assert importlib.import_module(module_name).__name__ == module_name


def test_validator_rejects_incomplete_support() -> None:
    value = copy.deepcopy(describe_five_backend_support())
    value["backends"][2]["support_capabilities"]["exact_fallback"] = False

    with pytest.raises(ValueError, match="full-support contract"):
        validate_five_backend_support(value)


def test_validator_rejects_support_promotion_conflation() -> None:
    value = copy.deepcopy(describe_five_backend_support())
    value["backends"][1]["evidence"]["recommendation_authorized"] = True

    with pytest.raises(ValueError, match="recommendation must follow"):
        validate_five_backend_support(value)


def test_validator_rejects_roster_reordering() -> None:
    value = copy.deepcopy(describe_five_backend_support())
    value["backends"][0], value["backends"][1] = (
        value["backends"][1],
        value["backends"][0],
    )

    with pytest.raises(ValueError, match="roster or order changed"):
        validate_five_backend_support(value)


def test_loaded_support_is_immutable() -> None:
    support = load_five_backend_support()

    with pytest.raises(TypeError, match="immutable"):
        support["schema"] = "changed"
