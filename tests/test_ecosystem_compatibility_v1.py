from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, cast

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib

import bayesian_phystwin.ecosystem_compatibility_v1 as compatibility
from bayesian_phystwin.causal4d_belief_provider_v1 import (
    CAUSAL4D_BELIEF_ARTIFACT_SCHEMA_VERSIONS,
    CAUSAL4D_BELIEF_PROVIDER_API_VERSION,
)
from bayesian_phystwin.causal4d_belief_provider_v2 import (
    CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION,
)
from bayesian_phystwin.causal4d_provider_v1 import (
    CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS as CAUSAL4D_V1_ARTIFACT_SCHEMAS,
)
from bayesian_phystwin.causal4d_provider_v1 import (
    CAUSAL4D_PROVIDER_API_VERSION as CAUSAL4D_PROVIDER_V1_API_VERSION,
)
from bayesian_phystwin.causal4d_provider_v2 import (
    CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS as CAUSAL4D_V2_ARTIFACT_SCHEMAS,
)
from bayesian_phystwin.causal4d_provider_v2 import (
    CAUSAL4D_PROVIDER_API_VERSION as CAUSAL4D_PROVIDER_V2_API_VERSION,
)
from bayesian_phystwin.prob4d_provider_attestation import (
    PROB4D_PROVIDER_API_VERSION,
    PROB4D_PROVIDER_ATTESTATION_VERSION,
    PROB4D_PROVIDER_IMPORT_BOUNDARY,
)

ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_TABLE_ID = "7c59d154fda374aef21ee3ff7c141ca6dab1abb4963d28e97d29b20718eeba45"
_EXPECTED_CAUSAL4D_MODULES = (
    ("bayesian_phystwin.causal4d_artifacts_v1", 1, "frozen_compatibility"),
    ("bayesian_phystwin.causal4d_artifacts_v2", 2, "production_additive"),
    (
        "bayesian_phystwin.causal4d_belief_provider_v1",
        CAUSAL4D_BELIEF_PROVIDER_API_VERSION,
        "frozen_compatibility",
    ),
    (
        "bayesian_phystwin.causal4d_belief_provider_v2",
        CAUSAL4D_BELIEF_PROVIDER_V2_API_VERSION,
        "additive_development",
    ),
    ("bayesian_phystwin.causal4d_graph_provider_v1", 1, "production"),
    (
        "bayesian_phystwin.causal4d_provider_v1",
        CAUSAL4D_PROVIDER_V1_API_VERSION,
        "frozen_compatibility",
    ),
    (
        "bayesian_phystwin.causal4d_provider_v2",
        CAUSAL4D_PROVIDER_V2_API_VERSION,
        "production",
    ),
    ("bayesian_phystwin.causal4d_public_provider_v1", 1, "diagnostic"),
    (
        "bayesian_phystwin.causal4d_tree_block_provider_v1",
        1,
        "production_additive",
    ),
)


def _table() -> compatibility.EcosystemCompatibilityTableV1:
    return compatibility.load_ecosystem_compatibility_table_v1()


def _payload() -> dict[str, Any]:
    return copy.deepcopy(_table().descriptor())


def _interfaces(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], payload["interfaces"])


def _components(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], payload["components"])


def _set_top_unknown(payload: dict[str, Any]) -> None:
    payload["unexpected"] = True


def _drop_schema(payload: dict[str, Any]) -> None:
    del payload["schema_name"]


def _bad_schema_name(payload: dict[str, Any]) -> None:
    payload["schema_name"] = "other"


def _boolean_schema_version(payload: dict[str, Any]) -> None:
    payload["schema_version"] = True


def _bad_status(payload: dict[str, Any]) -> None:
    payload["status"] = "claim_bearing_release"


def _swap_components(payload: dict[str, Any]) -> None:
    components = _components(payload)
    components[0], components[1] = components[1], components[0]


def _unknown_component(payload: dict[str, Any]) -> None:
    _components(payload)[0]["component_id"] = "unknown"


def _wrong_distribution(payload: dict[str, Any]) -> None:
    _components(payload)[0]["distribution_name"] = "other"


def _bad_repository(payload: dict[str, Any]) -> None:
    _components(payload)[0]["repository"] = "missing-owner"


def _bad_component_range(payload: dict[str, Any]) -> None:
    _components(payload)[0]["supported_versions"] = "0.4.x"


def _bad_python_range(payload: dict[str, Any]) -> None:
    _components(payload)[0]["requires_python"] = ">= 3.10"


def _empty_dependencies(payload: dict[str, Any]) -> None:
    _components(payload)[0]["required_dependencies"] = {}


def _bad_dependency_name(payload: dict[str, Any]) -> None:
    _components(payload)[0]["required_dependencies"] = {"Num Py": ">=1.23"}


def _bad_dependency_range(payload: dict[str, Any]) -> None:
    _components(payload)[0]["required_dependencies"] = {"numpy": True}


def _swap_interfaces(payload: dict[str, Any]) -> None:
    interfaces = _interfaces(payload)
    interfaces[0], interfaces[1] = interfaces[1], interfaces[0]


def _duplicate_interface(payload: dict[str, Any]) -> None:
    _interfaces(payload).append(copy.deepcopy(_interfaces(payload)[-1]))


def _participants_are_text(payload: dict[str, Any]) -> None:
    _interfaces(payload)[0]["participants"] = "bayesian_phystwin"


def _unknown_participant(payload: dict[str, Any]) -> None:
    interface = _interfaces(payload)[0]
    interface["participants"] = ["bayesian_phystwin", "unknown"]
    interface["distribution_ranges"] = {
        "bayesian_phystwin": ">=0.4,<0.5",
        "unknown": ">=1.0",
    }


def _range_roster_mismatch(payload: dict[str, Any]) -> None:
    del _interfaces(payload)[0]["distribution_ranges"]["causal4d"]


def _bad_interface_range(payload: dict[str, Any]) -> None:
    _interfaces(payload)[0]["distribution_ranges"]["causal4d"] = "~0.5"


def _contradictory_interface_range(payload: dict[str, Any]) -> None:
    _interfaces(payload)[0]["distribution_ranges"]["causal4d"] = ">=0.5,<0.7"


def _empty_provider_modules(payload: dict[str, Any]) -> None:
    _interfaces(payload)[0]["provider_modules"] = []


def _unsorted_provider_modules(payload: dict[str, Any]) -> None:
    modules = _interfaces(payload)[0]["provider_modules"]
    modules[0], modules[1] = modules[1], modules[0]


def _duplicate_provider_module(payload: dict[str, Any]) -> None:
    modules = _interfaces(payload)[1]["provider_modules"]
    modules.append(copy.deepcopy(modules[0]))


def _bad_provider_module_path(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["provider_modules"][0]["module"] = "Prob4D provider"


def _provider_module_outside_participants(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["provider_modules"][0]["module"] = "causal4d.provider_v1"


def _bad_provider_api(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["provider_modules"][0]["api_version"] = 0


def _bad_module_lifecycle(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["provider_modules"][0]["lifecycle"] = "retired"


def _empty_schema_mapping(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["required_artifact_schema_versions"] = {}


def _empty_schema_name(payload: dict[str, Any]) -> None:
    schemas = _interfaces(payload)[1]["required_artifact_schema_versions"]
    schemas[""] = schemas.pop("ObservationBeliefV1")


def _unsorted_schema_versions(payload: dict[str, Any]) -> None:
    schemas = _interfaces(payload)[1]["required_artifact_schema_versions"]
    schemas["ObservationBeliefV1"] = [2, 1]


def _empty_schema_versions(payload: dict[str, Any]) -> None:
    schemas = _interfaces(payload)[1]["required_artifact_schema_versions"]
    schemas["ObservationBeliefV1"] = []


def _boolean_schema_entry(payload: dict[str, Any]) -> None:
    schemas = _interfaces(payload)[1]["required_artifact_schema_versions"]
    schemas["ObservationBeliefV1"] = [True]


def _bad_interface_lifecycle(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["lifecycle"] = "retired"


def _coerced_claim_flag(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["supports_claim_bearing_admission"] = 1


def _coerced_revision_flag(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["exact_revisions_required_for_evidence"] = 1


def _claim_without_revision_lock(payload: dict[str, Any]) -> None:
    interface = _interfaces(payload)[0]
    interface["supports_claim_bearing_admission"] = True
    interface["exact_revisions_required_for_evidence"] = False


def _notes_are_text(payload: dict[str, Any]) -> None:
    _interfaces(payload)[1]["notes"] = "note"


def _duplicate_note(payload: dict[str, Any]) -> None:
    notes = _interfaces(payload)[1]["notes"]
    notes.append(notes[0])


def _weaken_evidence_boundary(payload: dict[str, Any]) -> None:
    payload["evidence_boundary"]["exact_revisions_required_for_claim_bearing_runs"] = (
        False
    )


_MUTATIONS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("top unknown", _set_top_unknown),
    ("missing schema", _drop_schema),
    ("schema name", _bad_schema_name),
    ("schema version", _boolean_schema_version),
    ("status", _bad_status),
    ("component order", _swap_components),
    ("component ID", _unknown_component),
    ("distribution", _wrong_distribution),
    ("repository", _bad_repository),
    ("component range", _bad_component_range),
    ("Python range", _bad_python_range),
    ("empty dependencies", _empty_dependencies),
    ("dependency name", _bad_dependency_name),
    ("dependency range", _bad_dependency_range),
    ("interface order", _swap_interfaces),
    ("duplicate interface", _duplicate_interface),
    ("participants type", _participants_are_text),
    ("participant", _unknown_participant),
    ("range roster", _range_roster_mismatch),
    ("interface range", _bad_interface_range),
    ("interface/component range", _contradictory_interface_range),
    ("empty provider modules", _empty_provider_modules),
    ("provider module order", _unsorted_provider_modules),
    ("duplicate provider module", _duplicate_provider_module),
    ("provider module path", _bad_provider_module_path),
    ("provider participant", _provider_module_outside_participants),
    ("provider API", _bad_provider_api),
    ("module lifecycle", _bad_module_lifecycle),
    ("empty schemas", _empty_schema_mapping),
    ("schema name", _empty_schema_name),
    ("schema order", _unsorted_schema_versions),
    ("empty schema versions", _empty_schema_versions),
    ("schema Boolean", _boolean_schema_entry),
    ("interface lifecycle", _bad_interface_lifecycle),
    ("claim Boolean", _coerced_claim_flag),
    ("revision Boolean", _coerced_revision_flag),
    ("claim revision lock", _claim_without_revision_lock),
    ("notes type", _notes_are_text),
    ("duplicate note", _duplicate_note),
    ("evidence boundary", _weaken_evidence_boundary),
)


def test_installed_table_is_content_addressed_and_immutable() -> None:
    table = _table()

    assert table.table_id == _EXPECTED_TABLE_ID
    assert table.as_dict()["table_id"] == _EXPECTED_TABLE_ID
    assert table.table_name == "bpt-prob4d-causal4d-development-compatibility-v1"
    assert table.status == "development_interoperability"

    with pytest.raises(TypeError, match="immutable"):
        cast(dict[str, Any], table.payload)["status"] = "changed"
    component = table.component("bayesian_phystwin")
    with pytest.raises(TypeError, match="immutable"):
        cast(dict[str, Any], component)["supported_versions"] = ">=0.5,<0.6"


def test_resource_and_documentation_are_declared_for_wheel_and_sdist() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    package_data = metadata["tool"]["setuptools"]["package-data"]["bayesian_phystwin"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "contract_data/ecosystem_compatibility_v1/*.json" in package_data
    assert "include docs/ecosystem_compatibility_v1.md" in manifest
    assert (
        "recursive-include "
        "src/bayesian_phystwin/contract_data/ecosystem_compatibility_v1 "
        "*.json"
    ) in manifest


def test_component_lines_match_current_package_contracts() -> None:
    table = _table()

    assert table.component("bayesian_phystwin") == {
        "component_id": "bayesian_phystwin",
        "distribution_name": "bayesian-phystwin",
        "repository": "IPS-Stuttgart/BayesianPhysTwin",
        "supported_versions": ">=0.4,<0.5",
        "requires_python": ">=3.10",
        "required_dependencies": {"numpy": ">=1.23"},
        "role": "Bayesian physical-twin inference and compatibility bridge",
    }
    assert table.component("prob4d")["supported_versions"] == ">=0.4,<0.5"
    causal4d = table.component("causal4d")
    assert causal4d["supported_versions"] == ">=0.5,<0.6"
    assert causal4d["required_dependencies"] == {
        "numpy": ">=1.24",
        "packaging": ">=23",
        "scipy": ">=1.10",
    }


def test_causal4d_registry_and_core_schemas_are_exact() -> None:
    interface = _table().interface("bayesian-phystwin-to-causal4d")
    modules = cast(Sequence[Mapping[str, Any]], interface["provider_modules"])
    observed = tuple(
        (
            module["module"],
            module["api_version"],
            module["lifecycle"],
        )
        for module in modules
    )

    assert observed == _EXPECTED_CAUSAL4D_MODULES
    for module_name, _, _ in _EXPECTED_CAUSAL4D_MODULES:
        assert importlib.util.find_spec(module_name) is not None

    schemas = cast(
        Mapping[str, Sequence[int]],
        interface["required_artifact_schema_versions"],
    )
    assert schemas["GraphBelief"] == [CAUSAL4D_V1_ARTIFACT_SCHEMAS["GraphBelief"]]
    assert schemas["TwinBelief"] == [CAUSAL4D_V1_ARTIFACT_SCHEMAS["TwinBelief"]]
    assert schemas["ReplayRequest"] == [CAUSAL4D_V2_ARTIFACT_SCHEMAS["ReplayRequest"]]
    assert schemas["ReplayTrajectory"] == [
        CAUSAL4D_V2_ARTIFACT_SCHEMAS["ReplayTrajectory"]
    ]
    assert schemas["FixedBayesianAnchorConfig"] == [
        CAUSAL4D_BELIEF_ARTIFACT_SCHEMA_VERSIONS["FixedBayesianAnchorConfig"]
    ]
    assert schemas["RobustEndpointPosterior"] == [
        CAUSAL4D_BELIEF_ARTIFACT_SCHEMA_VERSIONS["RobustEndpointPosterior"]
    ]
    assert interface["supports_claim_bearing_admission"] is True
    assert interface["exact_revisions_required_for_evidence"] is True


def test_prob4d_provider_rows_separate_historical_and_claim_bearing_use() -> None:
    table = _table()
    provider_v1 = table.interface("prob4d-provider-v1-to-bayesian-phystwin")
    provider_v2 = table.interface("prob4d-provider-v2-to-bayesian-phystwin")

    v1_modules = cast(Sequence[Mapping[str, Any]], provider_v1["provider_modules"])
    assert [(item["module"], item["api_version"]) for item in v1_modules] == [
        ("prob4d.provider_v1", 1)
    ]
    assert provider_v1["supports_claim_bearing_admission"] is False

    v2_modules = cast(Sequence[Mapping[str, Any]], provider_v2["provider_modules"])
    assert [(item["module"], item["api_version"]) for item in v2_modules] == [
        (PROB4D_PROVIDER_IMPORT_BOUNDARY, PROB4D_PROVIDER_API_VERSION)
    ]
    v2_schemas = cast(
        Mapping[str, Sequence[int]],
        provider_v2["required_artifact_schema_versions"],
    )
    assert v2_schemas["ProviderAttestation"] == [PROB4D_PROVIDER_ATTESTATION_VERSION]
    assert v2_schemas["ObservationBeliefV1"] == [1]
    assert v2_schemas["Prob4DCausalObservationStream"] == [2]
    assert v2_schemas["ObservationFactorBundle"] == [4]
    assert provider_v2["supports_claim_bearing_admission"] is True
    assert provider_v2["exact_revisions_required_for_evidence"] is True


def test_resource_bytes_round_trip_through_strict_validation() -> None:
    resource = resources.files("bayesian_phystwin").joinpath(
        *compatibility.ECOSYSTEM_COMPATIBILITY_RESOURCE.split("/")
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))

    restored = compatibility.validate_ecosystem_compatibility_table_v1(payload)

    assert restored.descriptor() == _table().descriptor()
    assert restored.table_id == _EXPECTED_TABLE_ID


@pytest.mark.parametrize(("name", "mutate"), _MUTATIONS)
def test_malformed_or_weakened_tables_fail_closed(
    name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError):
        compatibility.validate_ecosystem_compatibility_table_v1(payload)


@pytest.mark.parametrize(
    "text",
    (
        '{"schema_name": "a", "schema_name": "b"}',
        '{"value": NaN}',
        "[]",
        "{",
    ),
)
def test_strict_resource_parser_rejects_ambiguous_json(text: str) -> None:
    with pytest.raises(ValueError):
        compatibility._strict_json_object(text)


def test_unknown_component_and_interface_lookups_are_explicit() -> None:
    table = _table()

    with pytest.raises(KeyError, match="unknown"):
        table.component("unknown")
    with pytest.raises(KeyError, match="unknown"):
        table.interface("unknown")
