from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
QUALITY_SCRIPT = ROOT / "tools/quality/changed_python_quality.py"
PYPROJECT = ROOT / "pyproject.toml"

STRICT_ARTIFACT_PATHS = (
    "src/bayesian_phystwin/run_manifest.py",
    "src/bayesian_phystwin/repository_provenance.py",
    "src/bayesian_phystwin/run_manifest_v2.py",
    "src/bayesian_phystwin/claim_bundle_v1.py",
    "src/bayesian_phystwin/evidence_decision_v1.py",
    "src/bayesian_phystwin/physical_query_v1.py",
    "src/bayesian_phystwin/material_backend_evidence_v1.py",
)
STRICT_INTEGRATION_PATHS = (
    "src/bayesian_phystwin/v1/__init__.py",
    "src/bayesian_phystwin/causal4d_provider_v1.py",
    "src/bayesian_phystwin/prob4d_causal_lineage.py",
)
STRICT_MODULES = (
    "bayesian_phystwin.run_manifest",
    "bayesian_phystwin.repository_provenance",
    "bayesian_phystwin.run_manifest_v2",
    "bayesian_phystwin.claim_bundle_v1",
    "bayesian_phystwin.evidence_decision_v1",
    "bayesian_phystwin.physical_query_v1",
    "bayesian_phystwin.material_backend_evidence_v1",
    "bayesian_phystwin.v1",
    "bayesian_phystwin.causal4d_provider_v1",
    "bayesian_phystwin.prob4d_causal_lineage",
)
PER_MODULE_STRICT_OPTIONS = (
    "check_untyped_defs",
    "disallow_any_generics",
    "disallow_incomplete_defs",
    "disallow_subclassing_any",
    "disallow_untyped_calls",
    "disallow_untyped_decorators",
    "disallow_untyped_defs",
    "extra_checks",
    "no_implicit_reexport",
    "strict_equality",
    "warn_return_any",
    "warn_unused_ignores",
)


def _load_quality_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "bpt_changed_python_quality_policy",
        QUALITY_SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _integration_override_block() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    blocks = text.split("[[tool.mypy.overrides]]")[1:]
    matches = [block for block in blocks if '"bayesian_phystwin.v1"' in block]
    assert len(matches) == 1
    return matches[0]


def test_strict_paths_are_unconditional_and_deduplicated() -> None:
    quality = _load_quality_module()

    assert quality._STRICT_ARTIFACT_TYPE_TARGETS == STRICT_ARTIFACT_PATHS
    assert quality._STRICT_INTEGRATION_TYPE_TARGETS == STRICT_INTEGRATION_PATHS

    always = tuple(quality._ALWAYS_TYPE_TARGETS)
    strict = tuple(quality._STRICT_TYPE_TARGETS)
    required = set(STRICT_ARTIFACT_PATHS + STRICT_INTEGRATION_PATHS)

    assert len(always) == len(set(always))
    assert len(strict) == len(set(strict))
    assert required <= set(always)
    assert required <= set(strict)


def test_pyproject_applies_per_module_strict_options() -> None:
    block = _integration_override_block()

    for module in STRICT_MODULES:
        assert f'"{module}"' in block
    for option in PER_MODULE_STRICT_OPTIONS:
        assert f"{option} = true" in block
