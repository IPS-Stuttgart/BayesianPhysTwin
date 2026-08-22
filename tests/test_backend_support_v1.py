from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
from typing import Any, cast

import pytest

import bayesian_phystwin.backend_support_v1 as backend_support
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


def _replace(
    value: dict[str, Any], path: tuple[str | int, ...], replacement: Any
) -> None:
    cursor: Any = value
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = replacement


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("schema",), "wrong", "schema changed"),
        (("schema_version",), 2, "schema version changed"),
        (("support_definition",), [], "support_definition must be a JSON object"),
        (("support_definition", "status"), "partial", "status changed"),
        (
            ("support_definition", "required_capabilities"),
            ["discoverable"],
            "capability roster changed",
        ),
        (("backends",), "not-an-array", "backends must be a JSON array"),
        (("backends", 0, "display_name"), "", "nonempty canonical string"),
        (("backends", 0, "display_name"), "bad\nname", "control character"),
        (("backends", 0, "category"), "unknown", "unsupported category"),
        (
            ("backends", 0, "distribution_scope"),
            "unknown",
            "unsupported distribution scope",
        ),
        (
            ("backends", 1, "canonical_material_profile"),
            "unknown-profile",
            "unknown material profile",
        ),
        (("backends", 0, "support_status"), "partial", "not fully supported"),
        (
            ("backends", 0, "support_capabilities", "discoverable"),
            1,
            "literal boolean",
        ),
        (
            ("backends", 0, "implementation_paths"),
            ["/absolute.py"],
            "canonical relative POSIX path",
        ),
        (
            ("backends", 0, "implementation_paths"),
            ["z.py", "a.py"],
            "canonical lexical order",
        ),
        (
            ("backends", 0, "implementation_paths"),
            ["same.py", "same.py"],
            "unique values",
        ),
        (("backends", 0, "evidence", "stage"), "unknown", "unsupported evidence stage"),
        (
            ("backends", 0, "evidence", "latest_gate_decision"),
            "maybe",
            "invalid gate decision",
        ),
        (
            ("backends", 0, "evidence", "protected_target_outcomes_opened"),
            True,
            "protected evidence",
        ),
        (
            ("backends", 0, "evidence", "public_benchmark_outcomes_opened"),
            False,
            "public benchmark access",
        ),
        (
            ("backends", 0, "evidence", "artifacts"),
            [],
            "at least one evidence artifact",
        ),
        (
            ("backends", 0, "evidence", "artifacts", 0, "sha256"),
            "BAD",
            "lowercase SHA-256 digest",
        ),
    ),
)
def test_validator_rejects_malformed_support_descriptor_fields(
    path: tuple[str | int, ...], replacement: Any, message: str
) -> None:
    value = copy.deepcopy(describe_five_backend_support())
    _replace(value, path, replacement)

    with pytest.raises(ValueError, match=message):
        validate_five_backend_support(value)


def test_validator_requires_exact_backend_count() -> None:
    value = copy.deepcopy(describe_five_backend_support())
    value["backends"].pop()

    with pytest.raises(ValueError, match="exactly five backends"):
        validate_five_backend_support(value)


def test_validator_rejects_duplicate_evidence_artifacts() -> None:
    value = copy.deepcopy(describe_five_backend_support())
    artifacts = value["backends"][0]["evidence"]["artifacts"]
    artifacts.append(copy.deepcopy(artifacts[0]))

    with pytest.raises(ValueError, match="duplicate evidence artifacts"):
        validate_five_backend_support(value)


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b'{"key": 1, "key": 2}', "duplicate JSON object key"),
        (b'{"key": NaN}', "non-finite JSON constant"),
        (b"{", "resource is malformed"),
    ),
)
def test_strict_resource_parser_rejects_ambiguous_json(
    raw: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        backend_support._strict_json_object(raw)


def test_loader_rejects_mutated_installed_resource_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_support, "FIVE_BACKEND_SUPPORT_RESOURCE_SHA256", "0" * 64
    )

    with pytest.raises(RuntimeError, match="resource changed"):
        load_five_backend_support()


def test_source_tree_verifier_rejects_invalid_root(tmp_path: Path) -> None:
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("not a repository", encoding="utf-8")

    with pytest.raises(ValueError, match="ordinary directory"):
        verify_five_backend_source_tree(not_a_directory)


def test_source_tree_verifier_rejects_missing_declared_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = copy.deepcopy(describe_five_backend_support())
    descriptor["backends"][0]["implementation_paths"] = ["missing.py"]
    monkeypatch.setattr(
        backend_support, "load_five_backend_support", lambda: descriptor
    )

    with pytest.raises(RuntimeError, match="support file is unavailable"):
        verify_five_backend_source_tree(ROOT)


def test_source_tree_verifier_rejects_mutated_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = copy.deepcopy(describe_five_backend_support())
    descriptor["backends"][0]["evidence"]["artifacts"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(
        backend_support, "load_five_backend_support", lambda: descriptor
    )

    with pytest.raises(RuntimeError, match="retained backend evidence changed"):
        verify_five_backend_source_tree(ROOT)


def test_loaded_support_is_immutable() -> None:
    support = load_five_backend_support()

    with pytest.raises(TypeError, match="immutable"):
        support["schema"] = "changed"
