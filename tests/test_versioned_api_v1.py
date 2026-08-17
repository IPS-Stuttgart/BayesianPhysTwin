from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "api/versioned-public-api-v1.json"


def _isolated(script: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_versioned_api_is_deliberately_small_and_frozen() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    v1 = __import__("bayesian_phystwin.v1", fromlist=["*"])

    assert manifest["schema"] == "bayesian-phystwin.versioned-public-api-snapshot"
    assert manifest["schema_version"] == 1
    assert manifest["package"] == "bayesian_phystwin.v1"
    assert manifest["compatibility_line"] == "0.4"
    assert manifest["policy"] == "exact-versioned-integration-export-surface"
    assert list(v1.__all__) == manifest["symbols"]
    assert len(v1.__all__) == len(set(v1.__all__))
    for name in manifest["symbols"]:
        assert getattr(v1, name) is not None


def test_versioned_api_manifest_is_part_of_the_source_distribution() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "include api/versioned-public-api-v1.json" in manifest


def test_package_root_import_is_research_module_free() -> None:
    report = _isolated(
        """
        import json
        import sys

        import bayesian_phystwin

        package_modules = sorted(
            name
            for name in sys.modules
            if name == "bayesian_phystwin"
            or name.startswith("bayesian_phystwin.")
        )
        print(
            json.dumps(
                {
                    "package_modules": package_modules,
                    "dir_contains_all": all(
                        name in dir(bayesian_phystwin)
                        for name in bayesian_phystwin.__all__
                    ),
                }
            )
        )
        """
    )

    assert report == {
        "package_modules": ["bayesian_phystwin"],
        "dir_contains_all": True,
    }


def test_versioned_api_import_does_not_load_experiment_or_optional_modules() -> None:
    report = _isolated(
        """
        import json
        import sys

        import bayesian_phystwin.v1

        forbidden_package_prefixes = (
            "bayesian_phystwin.deform360_",
            "bayesian_phystwin.experiments",
            "bayesian_phystwin_experiments",
            "bayesian_phystwin.phystwin_",
            "bayesian_phystwin.synthetic_benchmark",
        )
        forbidden_external = {"cv2", "h5py", "numpyro", "pymc", "scipy", "torch"}
        leaked = sorted(
            name
            for name in sys.modules
            if name in forbidden_external
            or name.startswith(forbidden_package_prefixes)
        )
        print(json.dumps({"leaked": leaked}))
        """
    )

    assert report == {"leaked": []}


def test_legacy_root_export_loads_only_its_owning_module_and_is_cached() -> None:
    report = _isolated(
        """
        import json
        import sys

        import bayesian_phystwin

        before = sorted(
            name
            for name in sys.modules
            if name == "bayesian_phystwin"
            or name.startswith("bayesian_phystwin.")
        )
        first = bayesian_phystwin.BinaryCalibrationMetrics
        from bayesian_phystwin.calibration import BinaryCalibrationMetrics

        second = bayesian_phystwin.BinaryCalibrationMetrics
        after = sorted(
            name
            for name in sys.modules
            if name == "bayesian_phystwin"
            or name.startswith("bayesian_phystwin.")
        )
        print(
            json.dumps(
                {
                    "before": before,
                    "calibration_loaded": (
                        "bayesian_phystwin.calibration" in after
                    ),
                    "deform360_loaded": any(
                        name.startswith("bayesian_phystwin.deform360_")
                        for name in after
                    ),
                    "identity_preserved": (
                        first is second is BinaryCalibrationMetrics
                    ),
                }
            )
        )
        """
    )

    assert report == {
        "before": ["bayesian_phystwin"],
        "calibration_loaded": True,
        "deform360_loaded": False,
        "identity_preserved": True,
    }


def test_unknown_root_attribute_remains_an_attribute_error() -> None:
    import bayesian_phystwin

    assert set(bayesian_phystwin.__all__).issubset(dir(bayesian_phystwin))

    missing_name = "definitely_not_a_public_export"
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(bayesian_phystwin, missing_name)
