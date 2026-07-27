"""Extend the versioned Causal4D facade to installed-wheel integration APIs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "src/bayesian_phystwin/causal4d_scientific_provider_v1.py"
TEST = ROOT / "tests/test_causal4d_scientific_provider_v1.py"

_EXPORTS = '''
    "ArtifactDigest": ("run_manifest", "ArtifactDigest"),
    "GaugeAwareBeliefConfig": ("gauge_aware_belief", "GaugeAwareBeliefConfig"),
    "RepositoryState": ("repository_provenance", "RepositoryState"),
    "RunManifestV2": ("run_manifest_v2", "RunManifestV2"),
    "build_gauge_aware_batch_from_observation_belief": (
        "observation_belief_gauge_adapter",
        "build_gauge_aware_batch_from_observation_belief",
    ),
    "installed_package_versions": ("run_manifest", "installed_package_versions"),
    "load_observation_belief": ("observation_belief", "load_observation_belief"),
    "load_run_manifest_v2": ("run_manifest_v2", "load_run_manifest_v2"),
    "sha256_file": ("run_manifest", "sha256_file"),
    "update_gauge_aware_belief": (
        "gauge_aware_belief",
        "update_gauge_aware_belief",
    ),
    "validate_prob4d_causal_observation_belief": (
        "prob4d_causal_lineage",
        "validate_prob4d_causal_observation_belief",
    ),
    "verify_run_manifest_artifacts": (
        "run_manifest_v2",
        "verify_run_manifest_artifacts",
    ),
    "write_run_manifest": ("run_manifest_v2", "write_run_manifest"),
'''

_EXPECTED = '''
    "ArtifactDigest",
    "GaugeAwareBeliefConfig",
    "RepositoryState",
    "RunManifestV2",
    "build_gauge_aware_batch_from_observation_belief",
    "installed_package_versions",
    "load_observation_belief",
    "load_run_manifest_v2",
    "sha256_file",
    "update_gauge_aware_belief",
    "validate_prob4d_causal_observation_belief",
    "verify_run_manifest_artifacts",
    "write_run_manifest",
'''

_RESOLUTION_TEST = '''


def test_installed_wheel_contract_exports_resolve() -> None:
    assert provider.GaugeAwareBeliefConfig.__name__ == "GaugeAwareBeliefConfig"
    assert provider.RunManifestV2.__name__ == "RunManifestV2"
    assert provider.RepositoryState.__name__ == "RepositoryState"
    assert callable(provider.load_observation_belief)
    assert callable(provider.update_gauge_aware_belief)
    assert callable(provider.write_run_manifest)
'''


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    replace_once(
        FACADE,
        '_EXPORTS: Final[dict[str, tuple[str, str]]] = {\n',
        '_EXPORTS: Final[dict[str, tuple[str, str]]] = {\n' + _EXPORTS,
    )
    replace_once(
        TEST,
        '_EXPECTED_CAUSAL4D_EXPORTS = {\n',
        '_EXPECTED_CAUSAL4D_EXPORTS = {\n' + _EXPECTED,
    )
    text = TEST.read_text(encoding="utf-8")
    marker = "test_installed_wheel_contract_exports_resolve"
    if marker in text:
        raise RuntimeError("integration facade test already present")
    TEST.write_text(text.rstrip() + _RESOLUTION_TEST, encoding="utf-8")


if __name__ == "__main__":
    main()
