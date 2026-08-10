from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "science"
    / "run_full22_discrepancy_candidate_tournament_v1.py"
)
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "protocols" / "full22_discrepancy_candidate_tournament_v1.json"
)
PROTOCOL_SHA256 = "22746172fb6d207e80a2ebfeb2a003332317f3895317ff12988275acc75e16bd"


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "full22_discrepancy_candidate_tournament_v1",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_protocol_is_frozen_source_only_and_complete() -> None:
    module = _module()
    protocol, protocol_id = module._load_protocol(PROTOCOL_PATH)

    assert protocol_id == PROTOCOL_SHA256
    assert protocol["information_boundary"] == {
        "admission_manifest_sealed_before_future_scoring": True,
        "claim_authorized": False,
        "confirmation_payload_used": False,
        "prediction_manifests_sealed_before_admission": True,
        "replacement_allowed": False,
        "target_outcome_used": False,
    }
    assert module._candidate_ids(protocol) == (
        "physical_fallback",
        "last_residual",
        "independent_endpoint_v1",
        "dynamic_endpoint_v2",
        "structured_kernel_rank4_v1",
        "graph_dynamic_kernel_rank4_v1",
    )
    revisions = {
        row["candidate_id"]: row["source_revision"] for row in protocol["candidates"]
    }
    assert revisions["structured_kernel_rank4_v1"] == (
        "265cfe8488fdb40f2c1c72c67385e0e88bab2595"
    )
    assert revisions["graph_dynamic_kernel_rank4_v1"] == (
        "f49eecdbbbe31fc36b1a64ab6284a9e69d8851b3"
    )
    assert protocol["selection"]["runtime_tie_break"].startswith("disabled")


def test_geometry_kernel_basis_is_deterministic_and_orthonormal() -> None:
    module = _module()
    geometry = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
            [0.2, 0.2, 0.1],
            [0.1, 0.3, 0.2],
        ]
    )

    first = module._geometry_kernel_basis(
        geometry,
        rank=4,
        diagonal_tie_break=1e-12,
    )
    second = module._geometry_kernel_basis(
        geometry,
        rank=4,
        diagonal_tie_break=1e-12,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(4), atol=1e-12)
    for column in range(first.shape[1]):
        pivot = int(np.argmax(np.abs(first[:, column])))
        assert first[pivot, column] >= 0.0


def test_last_residual_and_horizon_partition_are_causal() -> None:
    module = _module()
    residual = np.arange(36, dtype=float).reshape(4, 3, 3)
    valid = np.asarray(
        [
            [True, False, True],
            [False, True, True],
            [True, False, False],
            [True, True, True],
        ],
        dtype=bool,
    )

    endpoint = module._last_valid_residual(
        residual,
        valid,
        end_frame=3,
    )

    np.testing.assert_array_equal(endpoint[0], residual[2, 0])
    np.testing.assert_array_equal(endpoint[1], residual[1, 1])
    np.testing.assert_array_equal(endpoint[2], residual[1, 2])
    groups = module._horizon_groups(8)
    assert tuple(groups) == ("early", "middle", "late")
    np.testing.assert_array_equal(groups["early"], [0, 1, 2])
    np.testing.assert_array_equal(groups["middle"], [3, 4, 5])
    np.testing.assert_array_equal(groups["late"], [6, 7])


def test_metric_guard_uses_higher_finite_sample_quantile() -> None:
    module = _module()
    values = np.asarray([-0.3, -0.2, 0.1, 0.4])

    assert module._higher_quantile(values, 0.5) == pytest.approx(0.1)
    assert module._higher_quantile(values, 0.9) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="finite nonempty"):
        module._higher_quantile(np.asarray([]), 0.9)


def test_common_floor_proper_score_is_finite_for_zero_covariance() -> None:
    module = _module()
    error = np.asarray([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    covariance = np.zeros((2, 3, 3))

    score = module._regularized_gaussian_nll(
        error,
        covariance,
        observation_std_m=0.005,
        eigenvalue_floor_m2=1e-12,
    )

    assert np.all(np.isfinite(score))
    assert score[1] > score[0]


def _report(selected: str, passed: bool) -> dict[str, object]:
    return {
        "selected_candidate": selected,
        "source_gate_passed": passed,
        "report_id": selected.encode().hex().ljust(64, "0")[:64],
    }


def test_metric_arbitration_requires_same_passing_challenger() -> None:
    module = _module()
    advanced = module._metric_arbitration(
        {
            "track": _report("dynamic", True),
            "chamfer": _report("dynamic", True),
        },
        reference_candidate="last_residual",
    )
    assert advanced["source_gate_passed"] is True
    assert advanced["selected_candidate"] == "dynamic"
    assert advanced["claim_authorized"] is False

    disagreement = module._metric_arbitration(
        {
            "track": _report("dynamic", True),
            "chamfer": _report("structured", True),
        },
        reference_candidate="last_residual",
    )
    assert disagreement["source_gate_passed"] is False
    assert disagreement["selected_candidate"] == "last_residual"

    failed = module._metric_arbitration(
        {
            "track": _report("dynamic", True),
            "chamfer": _report("dynamic", False),
        },
        reference_candidate="last_residual",
    )
    assert failed["source_gate_passed"] is False
    assert failed["selected_candidate"] == "last_residual"


def test_atomic_publication_refuses_implicit_replacement(
    tmp_path: Path,
) -> None:
    module = _module()
    target = tmp_path / "artifact"

    with module._atomic_output_directory(target, force=False) as temporary:
        (temporary / "payload.json").write_text("{}\n", encoding="utf-8")
    assert (target / "payload.json").exists()

    with pytest.raises(FileExistsError, match="output already exists"):
        with module._atomic_output_directory(target, force=False):
            pass


def test_prefix_manifest_recomputes_content_identity(
    tmp_path: Path,
) -> None:
    module = _module()
    descriptor = {
        "contract": module.PREFIX_MANIFEST_CONTRACT,
        "schema_version": 1,
        "protocol_id": "a" * 64,
        "cases": [],
    }
    descriptor["prefix_manifest_id"] = module._canonical_sha256(descriptor)
    module._write_json(tmp_path / "prefix_manifest.json", descriptor)

    loaded = module._load_prefix_manifest(tmp_path)
    assert loaded["prefix_manifest_id"] == descriptor["prefix_manifest_id"]

    payload = json.loads(
        (tmp_path / "prefix_manifest.json").read_text(encoding="utf-8")
    )
    payload["protocol_id"] = "b" * 64
    module._write_json(tmp_path / "prefix_manifest.json", payload)
    with pytest.raises(ValueError, match="identity changed"):
        module._load_prefix_manifest(tmp_path)


def test_registered_statistical_unit_is_deployment_valid() -> None:
    from bayesian_phystwin.discrepancy_candidate_tournament import (
        ALLOWED_DISCREPANCY_TOURNAMENT_STATISTICAL_UNITS,
    )

    module = _module()
    assert module.STATISTICAL_UNIT == "physical-object-session"
    assert module.STATISTICAL_UNIT in ALLOWED_DISCREPANCY_TOURNAMENT_STATISTICAL_UNITS


def test_metric_arbitration_reports_completion_status() -> None:
    module = _module()
    selected = module._metric_arbitration(
        {
            "track": _report("dynamic", True),
            "chamfer": _report("dynamic", True),
        },
        reference_candidate="last_residual",
    )
    no_selection = module._metric_arbitration(
        {
            "track": _report("dynamic", True),
            "chamfer": _report("structured", True),
        },
        reference_candidate="last_residual",
    )

    assert selected["status"] == "selected"
    assert no_selection["status"] == "completed_no_selection"
    assert no_selection["decision"] == "retain-reference-candidate"
    assert no_selection["claim_authorized"] is False
