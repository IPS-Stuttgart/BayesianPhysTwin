from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "scripts/science/run_cloth_sim2real_prob4d_covariance_locked_v1.py"
)
EXPECTED_SHA256 = (
    "268d07d94396f6f4ca277b6da0e8acf43512747fea6d40327eb33166da972c7f"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "cloth_covariance_dataset_lock",
        LAUNCHER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_binds_authoritative_dataset_sha256() -> None:
    module = _module()

    assert module.DATASET_SHA256 == EXPECTED_SHA256
    assert len(module.DATASET_SHA256) == 64
    assert module._IMPLEMENTATION.DATASET_SHA256 == EXPECTED_SHA256
    assert module.CovariancePolicy.from_treatment("full_joint").construction == (
        "persistent"
    )


def test_authoritative_identity_differs_from_known_truncated_value() -> None:
    module = _module()

    assert module.LEGACY_MALFORMED_DATASET_SHA256 != EXPECTED_SHA256
    assert len(module.LEGACY_MALFORMED_DATASET_SHA256) == 63
