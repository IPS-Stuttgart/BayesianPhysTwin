import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_external_backbone import (
    EXTERNAL_COORDINATE_FRAME,
    EXTERNAL_VERTEX_CONTRACT,
    _development_comparison,
    _load_cached_summary,
    _stage_trajectory,
    validate_external_backbone_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    case_root = data_root / "case_a"
    case_root.mkdir(parents=True)
    observed = np.zeros((8, 3, 3), dtype=np.float32)
    observed[:, :, 0] = np.arange(8, dtype=np.float32)[:, None] * 0.001
    with (case_root / "final_data.pkl").open("wb") as handle:
        pickle.dump({"object_points": observed}, handle)
    (case_root / "split.json").write_text(
        json.dumps({"train": [0, 5], "test": [5, 8], "frame_len": 8})
    )
    trajectory = np.concatenate(
        (observed, np.zeros((8, 2, 3), dtype=np.float32)), axis=1
    )
    trajectory_path = tmp_path / "external" / "case_a.pkl"
    trajectory_path.parent.mkdir()
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle)
    manifest_path = tmp_path / "external" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backbone": {
                    "name": "fixture",
                    "source_repository": "https://example.test/fixture",
                    "source_commit": "a" * 40,
                    "future_observations_used": False,
                    "coordinate_frame": EXTERNAL_COORDINATE_FRAME,
                    "vertex_contract": EXTERNAL_VERTEX_CONTRACT,
                },
                "cases": [
                    {
                        "name": "case_a",
                        "trajectory": "case_a.pkl",
                        "sha256": _sha256(trajectory_path),
                        "evidence_end_frame_exclusive": 5,
                    }
                ],
            }
        )
    )
    return data_root, manifest_path, trajectory_path


def test_external_manifest_accepts_causal_aligned_trajectory(tmp_path: Path) -> None:
    data_root, manifest_path, trajectory_path = _fixture(tmp_path)

    result = validate_external_backbone_manifest(
        data_root, manifest_path, require_full_cohort=False
    )

    assert result["cases"][0]["trajectory"] == str(trajectory_path.resolve())
    assert result["cases"][0]["maximum_initial_alignment_error_m"] == 0.0
    assert result["backbone"]["future_observations_used"] is False


def test_external_manifest_rejects_future_observation_evidence(tmp_path: Path) -> None:
    data_root, manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["evidence_end_frame_exclusive"] = 6
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="released training interval"):
        validate_external_backbone_manifest(
            data_root, manifest_path, require_full_cohort=False
        )


def test_external_manifest_requires_explicit_future_blind_claim(tmp_path: Path) -> None:
    data_root, manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["backbone"]["future_observations_used"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="forbid future observations"):
        validate_external_backbone_manifest(
            data_root, manifest_path, require_full_cohort=False
        )


def test_external_manifest_rejects_hash_or_material_identity_mismatch(
    tmp_path: Path,
) -> None:
    data_root, manifest_path, trajectory_path = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256"):
        validate_external_backbone_manifest(
            data_root, manifest_path, require_full_cohort=False
        )

    manifest["cases"][0]["sha256"] = _sha256(trajectory_path)
    manifest_path.write_text(json.dumps(manifest))
    with trajectory_path.open("rb") as handle:
        trajectory = pickle.load(handle)
    trajectory[0, 0, 0] = 0.01
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle)
    manifest["cases"][0]["sha256"] = _sha256(trajectory_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="material-vertex alignment"):
        validate_external_backbone_manifest(
            data_root, manifest_path, require_full_cohort=False
        )


def test_merged_manifest_revalidates_component_provenance(tmp_path: Path) -> None:
    data_root, component_path, _ = _fixture(tmp_path)
    checkpoint = tmp_path / "external" / "checkpoint.pth"
    audit = tmp_path / "external" / "causal_audit.json"
    checkpoint.write_bytes(b"checkpoint")
    audit.write_text("{}\n", encoding="utf-8")
    component = json.loads(component_path.read_text())
    component["backbone"].update(
        {
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "causal_training_audit": {
                "path": str(audit),
                "sha256": _sha256(audit),
            },
        }
    )
    component_path.write_text(json.dumps(component), encoding="utf-8")

    merged_path = tmp_path / "external" / "merged.json"
    merged_backbone = dict(component["backbone"])
    merged_backbone.pop("checkpoint")
    merged_backbone.pop("causal_training_audit")
    merged_backbone["component_manifests"] = [
        {
            "path": str(component_path),
            "sha256": _sha256(component_path),
            "cases": ["case_a"],
            "checkpoint": component["backbone"]["checkpoint"],
            "causal_training_audit": component["backbone"][
                "causal_training_audit"
            ],
        }
    ]
    merged_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backbone": merged_backbone,
                "cases": component["cases"],
            }
        ),
        encoding="utf-8",
    )

    validate_external_backbone_manifest(
        data_root, merged_path, require_full_cohort=False
    )

    component["backbone"]["claim_boundary"] = "changed after merge"
    component_path.write_text(json.dumps(component), encoding="utf-8")
    with pytest.raises(ValueError, match=r"component_manifests\[0\] SHA-256"):
        validate_external_backbone_manifest(
            data_root, merged_path, require_full_cohort=False
        )


def test_staging_is_exact_and_rejects_existing_different_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.pkl"
    destination = tmp_path / "staged" / "trajectory.pkl"
    source.write_bytes(b"source")

    _stage_trajectory(source, destination, _sha256(source))
    assert destination.read_bytes() == b"source"

    destination.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="differs from its manifest"):
        _stage_trajectory(source, destination, _sha256(source))


def test_cached_summary_normalizes_json_lists_against_protocol_tuples(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": {"candidates": [0.0, 0.1]},
                "inputs": {"baseline_trajectory": {"sha256": "a" * 64}},
            }
        )
    )

    summary = _load_cached_summary(
        summary_path,
        expected_config={"candidates": (0.0, 0.1)},
        expected_baseline_sha256="a" * 64,
    )

    assert summary is not None


def _case_result(
    baseline: tuple[float, float],
    bayesian: tuple[float, float],
    last: tuple[float, float],
    selected: str,
) -> dict[str, object]:
    def metrics(values: tuple[float, float]) -> dict[str, float]:
        return {
            "chamfer_distance_m": values[0],
            "track_error_m": values[1],
        }

    return {
        "selector": {"selected_method": selected},
        "test": {
            "bayesian_anchor": {
                "baseline_official_evaluation": metrics(baseline),
                "corrected_official_evaluation": metrics(bayesian),
            },
            "last_residual": {
                "corrected_official_evaluation": metrics(last),
            },
        },
    }


def test_development_comparison_is_labeled_and_uses_validation_selection() -> None:
    result = _development_comparison(
        {
            "single_lift_sloth": _case_result(
                (0.02, 0.03),
                (0.015, 0.02),
                (0.014, 0.021),
                "bayesian_anchor",
            ),
            "double_lift_sloth": _case_result(
                (0.01, 0.02),
                (0.009, 0.018),
                (0.008, 0.017),
                "last_residual",
            ),
        }
    )

    assert result["status"].startswith("development-only")
    selected = result["methods"]["external_validation_selected"]
    assert selected["per_case"]["single_lift_sloth"]["track_error_m"] == 0.02
    assert selected["per_case"]["double_lift_sloth"]["track_error_m"] == 0.017
    assert selected["equal_case_mean"]["chamfer_distance_m"] == pytest.approx(
        0.0115
    )


def test_development_comparison_rejects_non_development_case() -> None:
    with pytest.raises(ValueError, match="ordered subset"):
        _development_comparison(
            {"not_a_development_case": _case_result((1, 1), (1, 1), (1, 1), "backbone")}
        )
