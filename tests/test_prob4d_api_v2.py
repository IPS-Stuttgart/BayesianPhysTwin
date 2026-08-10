from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from bayesian_phystwin import prob4d_api_v2 as bridge


def _identity() -> dict[str, object]:
    return {
        "project_id": bridge.PROB4D_REQUIRED_PROJECT_ID,
        "canonical_repository": "IPS-Stuttgart/Prob4D",
        "frozen_artifact_repository": "FlorianPfaff/Prob4D",
    }


def _api(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "API_VERSION": 2,
        "PROVIDER_API_VERSION": 2,
        "PROVIDER_FACTOR_API_VERSION": 2,
        "prob4d_project_identity": _identity,
        "validate_prob4d_project_identity": lambda value: value,
        "load_claim_bearing_tree_sparse_observation": lambda path: path,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_inspect_prob4d_api_v2_validates_versions_and_project_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge.importlib, "import_module", lambda name: _api())

    result = bridge.inspect_prob4d_api_v2()

    assert result.api_version == 2
    assert result.provider_api_version == 2
    assert result.provider_factor_api_version == 2
    assert result.project_id == bridge.PROB4D_REQUIRED_PROJECT_ID
    assert result.canonical_repository == "IPS-Stuttgart/Prob4D"
    assert result.frozen_artifact_repository == "FlorianPfaff/Prob4D"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("API_VERSION", 3, "stable API version"),
        ("PROVIDER_API_VERSION", 3, "provider API version"),
        ("PROVIDER_FACTOR_API_VERSION", 3, "factor API version"),
        ("API_VERSION", True, "must be an integer"),
    ],
)
def test_inspect_prob4d_api_v2_rejects_incompatible_versions(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        bridge.importlib,
        "import_module",
        lambda name: _api(**{field: value}),
    )

    with pytest.raises(ImportError, match=message):
        bridge.inspect_prob4d_api_v2()


def test_inspect_prob4d_api_v2_rejects_project_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    identity["project_id"] = "github-repository-id:1"
    monkeypatch.setattr(
        bridge.importlib,
        "import_module",
        lambda name: _api(prob4d_project_identity=lambda: identity),
    )

    with pytest.raises(ImportError, match="not the supported project"):
        bridge.inspect_prob4d_api_v2()


def test_tree_sparse_loader_uses_only_the_stable_prob4d_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    imports: list[str] = []
    paths: list[str] = []

    def load(path: object) -> object:
        paths.append(str(path))
        return sentinel

    def import_module(name: str) -> object:
        imports.append(name)
        assert name == "prob4d.api.v2"
        return _api(load_claim_bearing_tree_sparse_observation=load)

    monkeypatch.setattr(bridge.importlib, "import_module", import_module)

    assert bridge.load_claim_bearing_tree_sparse_prob4d("claim.json") is sentinel
    assert imports == ["prob4d.api.v2"]
    assert paths == ["claim.json"]


def test_bridge_import_does_not_eagerly_import_prob4d() -> None:
    code = """
import sys
import bayesian_phystwin.prob4d_api_v2
loaded = sorted(
    name
    for name in sys.modules
    if name == "prob4d" or name.startswith("prob4d.")
)
if loaded:
    raise SystemExit(f"Prob4D imported eagerly: {loaded}")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
