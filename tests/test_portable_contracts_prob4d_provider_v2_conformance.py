from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from bayesian_phystwin import prob4d_api_v2 as bridge


def _identity() -> dict[str, object]:
    return {
        "project_id": bridge.PROB4D_REQUIRED_PROJECT_ID,
        "canonical_repository": "IPS-Stuttgart/Prob4D",
        "frozen_artifact_repository": "FlorianPfaff/Prob4D",
    }


def _api() -> SimpleNamespace:
    return SimpleNamespace(
        API_VERSION=2,
        PROVIDER_API_VERSION=2,
        PROVIDER_FACTOR_API_VERSION=2,
        prob4d_project_identity=_identity,
        validate_prob4d_project_identity=lambda value: value,
        load_claim_bearing_tree_sparse_observation=lambda path: path,
    )


def _contract_module(**changes: object) -> SimpleNamespace:
    vector = object()
    stack = object()
    prior = SimpleNamespace(
        prior_id=bridge.PROB4D_REQUIRED_PROVIDER_V2_MINIMAL_PRIOR_ID
    )
    materialization = SimpleNamespace(
        gauge_tree_prior=prior,
        tree_sparse_stack=stack,
    )
    values: dict[str, object] = {
        "PROVIDER_V2_CONTRACT_NUMERICAL_ATOL": (
            bridge.PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_ATOL
        ),
        "PROVIDER_V2_CONTRACT_NUMERICAL_RTOL": (
            bridge.PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_RTOL
        ),
        "provider_v2_contract_bundle_manifest": lambda: {
            "bundle_name": bridge.PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE,
            "bundle_sha256": (
                bridge.PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE_SHA256
            ),
        },
        "provider_v2_contract_schema": lambda: {"valid_vectors": ["minimal"]},
        "provider_v2_contract_vector": lambda name="minimal": vector,
        "materialize_provider_v2_contract_vector": (lambda supplied: materialization),
        "validate_provider_v2_contract_materialization": (
            lambda supplied, produced: None
        ),
        "provider_v2_contract_stack_semantic_sha256": (
            lambda supplied: bridge.PROB4D_REQUIRED_PROVIDER_V2_STACK_SEMANTIC_SHA256
        ),
        "invalid_provider_v2_contract_vectors": (
            lambda: tuple(object() for _ in range(10))
        ),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _install_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract: object | None = None,
) -> list[str]:
    imports: list[str] = []
    corpus = _contract_module() if contract is None else contract

    def import_module(name: str) -> Any:
        imports.append(name)
        if name == "prob4d.api.v2":
            return _api()
        if name == "prob4d.provider_v2_contract_bundle":
            return corpus
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(bridge.importlib, "import_module", import_module)
    return imports


def test_inspect_prob4d_provider_v2_contract_validates_portable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = _install_modules(monkeypatch)

    result = bridge.inspect_prob4d_provider_v2_contract()

    assert imports == [
        "prob4d.api.v2",
        "prob4d.provider_v2_contract_bundle",
    ]
    assert result.bundle_name == bridge.PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE
    assert result.bundle_sha256 == (
        bridge.PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE_SHA256
    )
    assert result.valid_vector_count == 1
    assert result.invalid_vector_count == 10
    assert result.minimal_prior_id == (
        bridge.PROB4D_REQUIRED_PROVIDER_V2_MINIMAL_PRIOR_ID
    )
    assert result.minimal_stack_semantic_sha256 == (
        bridge.PROB4D_REQUIRED_PROVIDER_V2_STACK_SEMANTIC_SHA256
    )
    assert result.numerical_atol == bridge.PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_ATOL
    assert result.numerical_rtol == bridge.PROB4D_REQUIRED_PROVIDER_V2_NUMERICAL_RTOL


def test_inspect_prob4d_provider_v2_contract_reports_missing_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def import_module(name: str) -> object:
        if name == "prob4d.api.v2":
            return _api()
        raise ImportError(name)

    monkeypatch.setattr(bridge.importlib, "import_module", import_module)

    with pytest.raises(ImportError, match="requires an installed Prob4D contract"):
        bridge.inspect_prob4d_provider_v2_contract()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "provider_v2_contract_bundle_manifest": lambda: {
                    "bundle_name": "changed.bundle",
                    "bundle_sha256": (
                        bridge.PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE_SHA256
                    ),
                }
            },
            "bundle changed name",
        ),
        (
            {
                "provider_v2_contract_bundle_manifest": lambda: {
                    "bundle_name": (
                        bridge.PROB4D_REQUIRED_PROVIDER_V2_CONTRACT_BUNDLE
                    ),
                    "bundle_sha256": "0" * 64,
                }
            },
            "bundle changed bytes",
        ),
        (
            {"provider_v2_contract_bundle_manifest": lambda: []},
            "contract bundle manifest must be a mapping",
        ),
        (
            {"provider_v2_contract_schema": lambda: {"valid_vectors": "minimal"}},
            "valid_vectors must be strings",
        ),
        (
            {"provider_v2_contract_schema": lambda: {"valid_vectors": [1]}},
            "valid_vectors must be strings",
        ),
        (
            {
                "provider_v2_contract_schema": lambda: {
                    "valid_vectors": ["minimal", "extra"]
                }
            },
            "valid-vector roster changed",
        ),
        (
            {"provider_v2_contract_stack_semantic_sha256": (lambda supplied: 7)},
            "semantic SHA-256 must be a nonempty string",
        ),
        (
            {"provider_v2_contract_stack_semantic_sha256": (lambda supplied: "0" * 64)},
            "stack semantic identity changed",
        ),
        (
            {"invalid_provider_v2_contract_vectors": lambda: []},
            "invalid contract vectors must be a tuple",
        ),
        (
            {
                "invalid_provider_v2_contract_vectors": (
                    lambda: tuple(object() for _ in range(9))
                )
            },
            "invalid-vector roster changed",
        ),
        (
            {"PROVIDER_V2_CONTRACT_NUMERICAL_ATOL": True},
            "numerical_atol must be a finite number",
        ),
        (
            {"PROVIDER_V2_CONTRACT_NUMERICAL_ATOL": math.inf},
            "numerical_atol must be a finite number",
        ),
        (
            {"PROVIDER_V2_CONTRACT_NUMERICAL_ATOL": 1e-9},
            "numerical atol changed",
        ),
        (
            {"PROVIDER_V2_CONTRACT_NUMERICAL_RTOL": object()},
            "numerical_rtol must be a finite number",
        ),
        (
            {"PROVIDER_V2_CONTRACT_NUMERICAL_RTOL": math.nan},
            "numerical_rtol must be a finite number",
        ),
        (
            {"PROVIDER_V2_CONTRACT_NUMERICAL_RTOL": 1e-8},
            "numerical rtol changed",
        ),
    ],
)
def test_inspect_prob4d_provider_v2_contract_rejects_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    _install_modules(monkeypatch, contract=_contract_module(**changes))

    with pytest.raises(ImportError, match=message):
        bridge.inspect_prob4d_provider_v2_contract()


def test_inspect_prob4d_provider_v2_contract_rejects_prior_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization = SimpleNamespace(
        gauge_tree_prior=SimpleNamespace(prior_id="0" * 64),
        tree_sparse_stack=object(),
    )
    contract = _contract_module(
        materialize_provider_v2_contract_vector=lambda supplied: materialization
    )
    _install_modules(monkeypatch, contract=contract)

    with pytest.raises(ImportError, match="minimal prior identity changed"):
        bridge.inspect_prob4d_provider_v2_contract()


def test_inspect_prob4d_provider_v2_contract_rejects_missing_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization = SimpleNamespace(
        gauge_tree_prior=None,
        tree_sparse_stack=object(),
    )
    contract = _contract_module(
        materialize_provider_v2_contract_vector=lambda supplied: materialization
    )
    _install_modules(monkeypatch, contract=contract)

    with pytest.raises(ImportError, match="prior_id must be a nonempty string"):
        bridge.inspect_prob4d_provider_v2_contract()


def test_inspect_prob4d_provider_v2_contract_requires_corpus_callables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_modules(
        monkeypatch,
        contract=_contract_module(provider_v2_contract_schema=None),
    )

    with pytest.raises(ImportError, match="lacks callable provider_v2_contract_schema"):
        bridge.inspect_prob4d_provider_v2_contract()
