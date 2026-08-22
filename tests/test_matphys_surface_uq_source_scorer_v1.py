from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    array_sha256,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/remote/score_matphys_surface_uq_source_v1.py"
PROTOCOL = ROOT / "configs/sota/matphys_surface_uq_source_v1.json"
PROTOCOL_V2 = ROOT / "configs/sota/matphys_surface_uq_source_v2.json"


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("matphys_uq_scorer", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _case_manifest(
    root: Path,
    case_id: str,
    *,
    seed: int,
    uncertainty_policy: str = "matphys",
    protocol_path: Path = PROTOCOL,
) -> Path:
    generator = np.random.default_rng(seed)
    covariance = np.broadcast_to(np.diag([9e-6, 1e-6, 4e-6]), (400, 3, 3)).copy()
    residual = generator.multivariate_normal(
        np.zeros(3),
        4.0 * covariance[0] + np.eye(3) * 1e-6,
        size=400,
    )
    arrays = {
        "residual_m": residual,
        "covariance_m2": (
            covariance if uncertainty_policy == "matphys" else np.zeros_like(covariance)
        ),
        "frame_index": np.repeat(np.arange(58, 76), 23)[:400],
        "node_index": np.arange(400),
        "nearest_distance_m": np.linalg.norm(residual, axis=1),
    }
    case_root = root / case_id
    case_root.mkdir()
    event_path = case_root / "matphys_surface_events.npz"
    np.savez_compressed(event_path, **arrays)
    manifest = {
        "schema": "bayesian-phystwin.matphys-surface-uq-source-case-v1",
        "schema_version": 1,
        "case_id": case_id,
        "protocol_sha256": file_sha256(protocol_path),
        "status": "scorable",
        "uncertainty_policy": uncertainty_policy,
        "events": {
            "path": event_path.name,
            "sha256": file_sha256(event_path),
            "array_sha256": {
                name: array_sha256(value) for name, value in sorted(arrays.items())
            },
        },
    }
    manifest["artifact_id"] = content_id(manifest)
    manifest_path = case_root / "matphys_surface_uq_case.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_aggregate_requires_and_scores_complete_source_denominator(
    tmp_path: Path,
) -> None:
    module = _module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    case_ids = protocol["source_panel"]["case_ids"]
    manifests = [
        _case_manifest(tmp_path, case_id, seed=260823 + index)
        for index, case_id in enumerate(case_ids)
    ]

    result = module.aggregate_source(
        protocol_path=PROTOCOL,
        case_manifest_paths=manifests,
        output_dir=tmp_path / "aggregate",
    )

    assert result["source_denominator_count"] == 10
    assert result["ordinary_scorable_count"] == 10
    assert result["leave_one_case_out"]["case_count"] == 10
    assert result["information_boundary"]["target_or_confirmation_data_read"] is False


def test_aggregate_rejects_missing_source_case(tmp_path: Path) -> None:
    module = _module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifests = [
        _case_manifest(tmp_path, case_id, seed=260900 + index)
        for index, case_id in enumerate(protocol["source_panel"]["case_ids"][:-1])
    ]

    with pytest.raises(ValueError, match="denominator is incomplete"):
        module.aggregate_source(
            protocol_path=PROTOCOL,
            case_manifest_paths=manifests,
            output_dir=tmp_path / "aggregate",
        )


def test_guarded_aggregate_scores_fallback_as_an_exact_isotropic_tie(
    tmp_path: Path,
) -> None:
    module = _module()
    protocol = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))
    retained = {"058-roll-napkin-ep0001", "167-glove-gray-cloth-ep0000"}
    fallback = "186-monster-ep0006"
    manifests: list[Path] = []
    evidence = tmp_path / "physical_prediction_manifest.json"
    evidence.write_text("{}", encoding="utf-8")
    for index, case_id in enumerate(protocol["source_panel"]["case_ids"]):
        if case_id in retained:
            result = module.retain_case(
                protocol_path=PROTOCOL_V2,
                case_id=case_id,
                status="unavailable-physical-carrier",
                evidence_manifest_paths=[evidence],
                output_dir=tmp_path / case_id,
            )
            assert (
                result["information_boundary"]["source_scoring_outcome_read"] is False
            )
            manifests.append(tmp_path / case_id / "matphys_surface_uq_case.json")
            continue
        manifests.append(
            _case_manifest(
                tmp_path,
                case_id,
                seed=261000 + index,
                uncertainty_policy=(
                    "isotropic-fallback" if case_id == fallback else "matphys"
                ),
                protocol_path=PROTOCOL_V2,
            )
        )

    result = module.aggregate_source(
        protocol_path=PROTOCOL_V2,
        case_manifest_paths=manifests,
        output_dir=tmp_path / "aggregate",
    )
    evaluation = result["leave_one_case_out"]

    assert result["source_denominator_count"] == 10
    assert result["ordinary_scorable_count"] == 8
    assert evaluation["matphys_case_count"] == 7
    assert evaluation["isotropic_fallback_case_count"] == 1
    fallback_row = next(
        row for row in evaluation["case_rows"] if row["case_id"] == fallback
    )
    assert fallback_row["candidate"] == fallback_row["isotropic"]
    assert fallback_row["candidate_nll_win"] is False
