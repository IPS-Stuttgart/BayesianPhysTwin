from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.contracts.fixed_anchor import FixedBayesianAnchorConfigV1
from bayesian_phystwin.endpoint_model_average import (
    ModelAveragedEndpointConfigV1,
    infer_model_averaged_endpoint,
)

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "run_deform360_public_evaluation.py"
)
_SPEC = importlib.util.spec_from_file_location("_deform360_public_evaluation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_selection = _MODULE.build_selection
calibrate = _MODULE.calibrate
confirm = _MODULE.confirm
normalized_evidence_prediction = _MODULE.normalized_evidence_prediction
_parse_selected_archive = _MODULE._parse_selected_archive


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _protocol(
    path: Path,
    *,
    calibration_per_stratum: int = 2,
    confirmation_per_stratum: int = 2,
) -> Path:
    payload = {
        "schema": "bayesian-phystwin/deform360-independent-protocol-v1",
        "schema_version": 1,
        "protocol_id": "test-deform360-independent",
        "status": "locked-before-numerical-payload-access",
        "locked_at_utc": "2026-08-04T00:00:00Z",
        "claim_boundary": "synthetic contract test",
        "strata": ["sheet", "volumetric"],
        "horizons_frames": [1, 2, 4, 8],
        "methods": [
            "persistence",
            "last_displacement",
            "cumulative_evidence",
            "normalized_evidence",
        ],
        "information_boundary": {
            "selection_uses_names_only": True,
            "historical_reserved_targets_must_remain_unopened": True,
            "calibration_payloads_open_before_confirmation": True,
            "calibration_serialized_before_confirmation": True,
            "confirmation_parameters_refit": False,
            "future_used_for_scoring_only": True,
            "replacement_after_payload_access_allowed": False,
        },
        "limits": {
            "max_frames_per_archive": 32,
            "max_tracks_per_archive": 16,
            "max_evaluation_prefixes_per_horizon": 4,
            "minimum_prefix_displacements": 3,
        },
        "selection": {
            "seed": "test-selection",
            "calibration_objects_per_stratum": calibration_per_stratum,
            "confirmation_objects_per_stratum": confirmation_per_stratum,
            "episodes_per_object": 1,
            "maximum_locked_archive_paths_per_object": 2,
            "historical_exclusions": "test",
            "object_rule": "test",
            "target_replacement": "forbidden",
        },
        "group_calibration": {
            "across_object_coverage": 0.75,
            "within_object_event_quantile": 0.9,
            "target_reference": "chi-square",
            "object_q90_interpretation": "test",
            "simultaneous_interpretation": "test",
        },
        "support_gate": {
            "minimum_supported_calibration_objects": 3,
            "minimum_supported_calibration_objects_per_stratum": 1,
            "minimum_supported_confirmation_objects": 3,
            "minimum_supported_confirmation_objects_per_stratum": 1,
            "unsupported_object_action": "retain",
            "confirmation_action_if_calibration_fails": "do not open",
        },
        "normalized_evidence": {
            "component_bank": "test",
            "mean_rule": "mean evidence",
            "prior": "uniform",
            "target_outcome_tuning": False,
            "weight_rule": "test",
        },
        "bootstrap": {
            "samples": 200,
            "seed": 7,
            "statistical_unit": "object",
        },
        "success_gates": {
            "normalized_effective_components_exceed_cumulative_each_horizon": True,
            "normalized_raw_nll_better_than_cumulative_each_horizon": True,
            "object_q90_calibrated_coverage_range": [0.0, 1.0],
            "point_results_reported_against_last_displacement": True,
        },
    }
    _write_json(path, payload)
    return path


def _inventory(path: Path, object_ids: list[str]) -> Path:
    objects = []
    for index, object_id in enumerate(object_ids):
        episode = index % 3
        objects.append(
            {
                "object_id": object_id,
                "episode_ids_from_names": [episode],
                "sample_paths": [
                    f"processed/{object_id}/episode_{episode:04d}/trajectory.npz"
                ],
                "numeric_paths": [
                    f"processed/{object_id}/episode_{episode:04d}/trajectory.npz"
                ],
            }
        )
    payload = {
        "schema": "bayesian-phystwin/deform360-name-inventory-v1",
        "schema_version": 1,
        "repository_revision": "test",
        "dataset_root": "/not-opened",
        "information_boundary": {
            "dataset_payload_opened": False,
            "file_contents_hashed": False,
            "names_and_directory_structure_only": True,
            "reserved_target_outcomes_opened": False,
        },
        "total_files": len(objects),
        "total_directories": len(objects),
        "object_count": len(objects),
        "top_level": [],
        "extension_counts": {".npz": len(objects)},
        "objects": objects,
    }
    payload["inventory_sha256"] = _MODULE._canonical_sha256(payload)
    _write_json(path, payload)
    return path


def _config_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root
        / "configs"
        / "sota"
        / "deform360_bias_aware_guarded_belief_prospective_v1.json",
        root
        / "configs"
        / "sota"
        / "deform360_bias_aware_guarded_belief_prospective_v2.json",
    )


def _fresh_ids() -> list[str]:
    return [
        "010-orange-cloth",
        "012-hat-cloth",
        "013-glove-cloth",
        "014-glove-vinyl-cloth",
        "045-cat",
        "047-rectangle-sponge",
        "048-butter-sponge",
        "050-boxing",
    ]


def _trajectory(offset: float, *, frames: int = 32) -> np.ndarray:
    base = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.02, 0.02, 0.00],
        ]
    )
    result = []
    for frame in range(frames):
        velocity = np.array([0.001 + offset + 0.00002 * frame, -0.0004, 0.0003])
        result.append(base + frame * velocity)
    return np.asarray(result)


def _materialize_selection_data(root: Path, selection: dict[str, object]) -> None:
    for partition in ("calibration", "confirmation"):
        for index, record in enumerate(selection[partition]):
            assert isinstance(record, dict)
            relative = record["archive_paths"][0]
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            trajectory = _trajectory(index * 1e-5)
            np.savez_compressed(
                path,
                positions_world_m=trajectory,
                valid_mask=np.ones(trajectory.shape[:2], dtype=bool),
            )


def test_selection_is_deterministic_and_excludes_reserved_objects(
    tmp_path: Path,
) -> None:
    protocol = _protocol(tmp_path / "protocol.json")
    object_ids = _fresh_ids() + ["075-leather", "139-rubber-ball"]
    inventory = _inventory(tmp_path / "inventory.json", object_ids)
    v1, v2 = _config_paths()

    first = build_selection(inventory, protocol, v1, v2)
    second = build_selection(inventory, protocol, v1, v2)

    assert first == second
    assert first["selection_complete"] is True
    selected = {
        record["object_id"]
        for partition in ("calibration", "confirmation")
        for record in first[partition]
    }
    assert "075-leather" not in selected
    assert "139-rubber-ball" not in selected
    assert len(selected) == 8


def test_normalized_evidence_retains_more_components() -> None:
    frames = 80
    residual = np.zeros((frames, 1, 3), dtype=float)
    residual[:, 0, 0] = 0.002 + 0.0002 * np.sin(np.arange(frames) / 4)
    valid = np.ones((frames, 1), dtype=bool)
    config = ModelAveragedEndpointConfigV1(
        components=(
            FixedBayesianAnchorConfigV1(
                process_std_m=0.0,
                observation_std_m=0.001,
            ),
            FixedBayesianAnchorConfigV1(
                process_std_m=0.0005,
                observation_std_m=0.0025,
            ),
            FixedBayesianAnchorConfigV1(
                process_std_m=0.001,
                observation_std_m=0.005,
            ),
        )
    )
    posterior = infer_model_averaged_endpoint(
        residual, valid, end_frame=frames, config=config
    )
    normalized = normalized_evidence_prediction(posterior)
    cumulative_effective = 1.0 / np.sum(np.square(posterior.component_weights[0]))
    normalized_effective = 1.0 / np.sum(np.square(normalized.component_weights[0]))

    assert normalized_effective > cumulative_effective
    assert normalized_effective > 1.05
    assert np.all(np.linalg.eigvalsh(normalized.covariance_m2) >= -1e-12)


def test_calibration_serializes_before_confirmation(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path / "protocol.json")
    inventory = _inventory(tmp_path / "inventory.json", _fresh_ids())
    v1, v2 = _config_paths()
    selection = build_selection(inventory, protocol, v1, v2)
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    data_root = tmp_path / "data"
    _materialize_selection_data(data_root, selection)

    calibration = calibrate(
        data_root,
        protocol,
        selection_path,
        revision="revision-test",
    )
    assert calibration["confirmation_authorized"] is True
    assert calibration["information_boundary"]["confirmation_payloads_opened"] is False
    calibration_path = tmp_path / "calibration.json"
    _write_json(calibration_path, calibration)

    result = confirm(
        data_root,
        protocol,
        selection_path,
        calibration_path,
        revision="revision-test",
    )
    assert result["support_gate"]["passed"] is True
    assert result["information_boundary"]["historical_reserved_targets_opened"] is False
    assert len(result["result_sha256"]) == 64
    assert set(result["point"]) == {"h1", "h2", "h4", "h8"}


def test_confirmation_rejects_unauthorized_calibration(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path / "protocol.json")
    inventory = _inventory(tmp_path / "inventory.json", _fresh_ids())
    v1, v2 = _config_paths()
    selection = build_selection(inventory, protocol, v1, v2)
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    protocol_payload = json.loads(protocol.read_text())
    protocol_sha = _MODULE._canonical_sha256(protocol_payload)
    calibration = {
        "schema": "bayesian-phystwin/deform360-independent-calibration-v1",
        "schema_version": 1,
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection["selection_sha256"],
        "confirmation_authorized": False,
        "information_boundary": {"confirmation_payloads_opened": False},
    }
    calibration["calibration_sha256"] = _MODULE._canonical_sha256(calibration)
    calibration_path = tmp_path / "calibration.json"
    _write_json(calibration_path, calibration)

    with pytest.raises(PermissionError, match="did not authorize"):
        confirm(
            tmp_path / "missing-data",
            protocol,
            selection_path,
            calibration_path,
            revision=None,
        )


def test_selected_paths_are_object_and_episode_scoped() -> None:
    with pytest.raises(ValueError, match="escapes"):
        _parse_selected_archive(
            {
                "object_id": "010-orange-cloth",
                "stratum": "sheet",
                "episode_id": 1,
                "archive_paths": ["../010-orange-cloth/episode_0001/x.npz"],
            }
        )
    with pytest.raises(ValueError, match="episode changed"):
        _parse_selected_archive(
            {
                "object_id": "010-orange-cloth",
                "stratum": "sheet",
                "episode_id": 1,
                "archive_paths": ["processed/010-orange-cloth/episode_0002/x.npz"],
            }
        )


def test_packed_hull_contract_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "processed/010-orange-cloth/episode_0001/hulls.npz"
    path.parent.mkdir(parents=True)
    base = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    hulls = [base + np.array([frame * 0.001, 0.0, 0.0]) for frame in range(16)]
    offsets = np.arange(0, (len(hulls) + 1) * len(base), len(base))
    np.savez_compressed(
        path,
        frame_indices=np.arange(len(hulls)),
        point_offsets=offsets,
        points_world_m=np.concatenate(hulls),
    )
    protocol_path = _protocol(
        tmp_path / "protocol.json",
        calibration_per_stratum=2,
        confirmation_per_stratum=2,
    )
    protocol = _MODULE._load_protocol(protocol_path)
    selected = _parse_selected_archive(
        {
            "object_id": "010-orange-cloth",
            "stratum": "sheet",
            "episode_id": 1,
            "archive_paths": ["processed/010-orange-cloth/episode_0001/hulls.npz"],
        }
    )
    case, attempts = _MODULE._open_selected_case(tmp_path, selected, protocol)

    assert case is not None
    assert case.representation == "packed_visual_hulls"
    assert attempts[-1]["status"] == "selected"
