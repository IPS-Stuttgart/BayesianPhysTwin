from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_reusable_dynamics import (
    CANONICAL_REUSABLE_DYNAMICS_CONFIG_SHA256,
    load_reusable_dynamics_config,
    load_reusable_dynamics_pipeline_config,
    select_reusable_dynamics_source_grid,
    validate_reusable_dynamics_association_evidence,
    validate_reusable_dynamics_calibration_request,
    validate_reusable_dynamics_config,
    validate_reusable_dynamics_source_request,
    validate_reusable_dynamics_source_selection,
    validate_reusable_dynamics_source_trust_compatibility,
)


ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
)
PIPELINE_CONFIG = (
    ROOT
    / "configs/causal4d_public/deform360_reusable_dynamics_pipeline_081_v1.json"
)
MASK_SUMMARY = (
    ROOT
    / "milestones/deform360-reusable-association-v2-calibration-mask"
    / "calibration_mask_summary.json"
)
PREFIX_SUMMARY = (
    ROOT
    / "milestones/deform360-reusable-association-v2-calibration-prefix"
    / "calibration_prefix_summary.json"
)
SOURCE_SELECTION = (
    ROOT
    / "milestones/deform360-reusable-dynamics-081-v1"
    / "artifacts/source_selection.json"
)
SOURCE_TRUST = (
    ROOT
    / "milestones/deform360-reusable-dynamics-081-v1"
    / "artifacts/source_trust_compatibility.json"
)


def test_canonical_reusable_dynamics_protocol_is_locked() -> None:
    payload = load_reusable_dynamics_config(CONFIG)

    result = validate_reusable_dynamics_config(payload)

    assert result["config_sha256"] == CANONICAL_REUSABLE_DYNAMICS_CONFIG_SHA256
    assert result["physical_candidate_count"] == 24
    assert result["source_episodes"] == [1, 4, 6]
    assert result["calibration_episodes"] == [0, 2, 8]
    assert result["sealed_target_episodes"] == [5]


def test_observation_pipeline_is_locked_before_reconstruction() -> None:
    parent = load_reusable_dynamics_config(CONFIG)

    pipeline = load_reusable_dynamics_pipeline_config(
        PIPELINE_CONFIG, parent=parent
    )

    assert pipeline["config"]["reconstruction"]["voxel_resolution"] == 120
    assert pipeline["config"]["reconstruction"]["warm_start_iterations"] == 250
    assert pipeline["config"]["point_cloud"]["expected_frame_count"] == 76
    assert pipeline["config"]["information_boundary"]["target_media_read"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("frame_protocol", "future_frame_allowed_for_initial_association", True),
        (
            "information_boundary",
            "calibration_future_allowed_for_method_or_hyperparameter_changes",
            True,
        ),
        (
            "information_boundary",
            "target_media_allowed_before_all_calibration_gates_pass",
            True,
        ),
        ("fixed_action_trust", "base_action_response", 0.5),
    ],
)
def test_reusable_dynamics_rejects_protocol_mutations(
    section: str, key: str, value: object
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["config"][section][key] = value

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_reusable_dynamics_config(changed)


def test_association_evidence_unlocks_dynamics_staging() -> None:
    payload = load_reusable_dynamics_config(CONFIG)

    result = validate_reusable_dynamics_association_evidence(
        payload,
        mask_summary_path=MASK_SUMMARY,
        prefix_summary_path=PREFIX_SUMMARY,
    )

    assert result["passed"] is True
    assert result["calibration_episodes"] == [0, 2, 8]


def test_source_request_is_limited_to_source_partition() -> None:
    payload = load_reusable_dynamics_config(CONFIG)

    result = validate_reusable_dynamics_source_request(
        payload, object_id="081-stripe-rope", episode_id=6
    )

    assert result["allowed_raw_frame_range"] == [110, 191]
    with pytest.raises(ValueError, match="not in the source partition"):
        validate_reusable_dynamics_source_request(
            payload, object_id="081-stripe-rope", episode_id=0
        )


def test_calibration_request_has_operation_specific_frame_boundaries() -> None:
    payload = load_reusable_dynamics_config(CONFIG)

    initial = validate_reusable_dynamics_calibration_request(
        payload,
        object_id="081-stripe-rope",
        episode_id=2,
        operation="initial-association",
    )
    staged = validate_reusable_dynamics_calibration_request(
        payload,
        object_id="081-stripe-rope",
        episode_id=2,
        operation="staging",
    )
    scoring = validate_reusable_dynamics_calibration_request(
        payload,
        object_id="081-stripe-rope",
        episode_id=2,
        operation="one-shot-scoring",
    )

    assert initial["allowed_frame_range"] == [110, 111]
    assert staged["allowed_frame_range"] == [110, 191]
    assert scoring["allowed_frame_range"] == [1, 76]
    assert scoring["method_or_hyperparameter_changes_allowed"] is False


def test_sealed_target_is_rejected_by_every_open_request() -> None:
    payload = load_reusable_dynamics_config(CONFIG)

    with pytest.raises(ValueError, match="not in the independent calibration"):
        validate_reusable_dynamics_calibration_request(
            payload,
            object_id="081-stripe-rope",
            episode_id=5,
            operation="initial-association",
        )
    with pytest.raises(ValueError, match="not in the source partition"):
        validate_reusable_dynamics_source_request(
            payload, object_id="081-stripe-rope", episode_id=5
        )


def test_source_grid_selection_is_pooled_and_source_only(tmp_path: Path) -> None:
    payload = load_reusable_dynamics_config(CONFIG)
    config = payload["config"]
    simulator = config["official_phystwin"]
    source = {int(row["episode_id"]): row for row in config["source_inputs"]}
    candidates = list(
        itertools.product(
            simulator["source_parameter_grid"]["init_spring_Y"],
            simulator["source_parameter_grid"]["drag_damping"],
            simulator["source_parameter_grid"]["dashpot_damping"],
        )
    )
    ideal_by_episode = {1: 0, 4: 7, 6: 23}
    for index, (spring, drag, dashpot) in enumerate(candidates):
        label = f"y{int(spring)}-drag{int(drag)}-dash{int(dashpot)}"
        for episode_id, ideal in ideal_by_episode.items():
            output = tmp_path / f"ep{episode_id}" / label
            output.mkdir(parents=True)
            trajectory = output / "official_phystwin_trajectory.npz"
            trajectory.write_bytes(f"{episode_id}:{index}".encode("ascii"))
            trajectory_sha256 = hashlib.sha256(trajectory.read_bytes()).hexdigest()
            relative = 1.0 + 0.01 * abs(index - ideal)
            metrics = {
                "frame_range": [1, 60],
                "prediction_finite": True,
                "track_rmse_m": relative,
                "chamfer_m": relative,
                "persistence_track_rmse_m": 1.0,
                "persistence_chamfer_m": 1.0,
            }
            result = {
                "passed": True,
                "source_only_smoke": True,
                "official_phystwin_revision": simulator["upstream_revision"],
                "config_sha256": simulator["real_config_sha256"],
                "split_sha256": simulator["source_split_sha256"],
                "data_sha256": source[episode_id]["controller_bundle_sha256"],
                "support_dynamics": {"mode": simulator["support_mode"]},
                "config_overrides": {
                    **simulator["fixed_overrides"],
                    "init_spring_Y": float(spring),
                    "drag_damping": float(drag),
                    "dashpot_damping": float(dashpot),
                },
                "num_controller_points": source[episode_id]["controller_count"],
                "trajectory_sha256": trajectory_sha256,
                "metrics": {
                    "intervals": {
                        "train": metrics,
                        "test": {
                            **metrics,
                            "frame_range": [60, 76],
                        },
                    }
                },
            }
            (output / "official_phystwin_smoke.json").write_text(
                json.dumps(result), encoding="utf-8"
            )

    failed_path = (
        tmp_path
        / "ep4/y80000-drag10-dash50/official_phystwin_smoke.json"
    )
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    failed["passed"] = False
    failed["first_nonfinite_frame"] = 20
    failed["metrics"]["intervals"]["train"]["prediction_finite"] = False
    failed["metrics"]["intervals"]["train"]["track_rmse_m"] = None
    failed["metrics"]["intervals"]["train"]["chamfer_m"] = None
    failed_path.write_text(json.dumps(failed), encoding="utf-8")

    selected = select_reusable_dynamics_source_grid(payload, grid_root=tmp_path)
    validated = validate_reusable_dynamics_source_selection(
        selected, config=payload
    )

    assert selected["candidate_count"] == 24
    assert selected["eligible_candidate_count"] == 23
    assert selected["rejected_candidate_labels"] == ["y80000-drag10-dash50"]
    assert selected["selected_pooled_physical_parameters"] == {
        "init_spring_Y": 30000.0,
        "drag_damping": 1.0,
        "dashpot_damping": 100.0,
    }
    assert selected["selected_single_source_physical_parameters"]["1"] == {
        "init_spring_Y": 10000.0,
        "drag_damping": 1.0,
        "dashpot_damping": 50.0,
    }
    assert selected["selected_single_source_physical_parameters"]["6"] == {
        "init_spring_Y": 80000.0,
        "drag_damping": 10.0,
        "dashpot_damping": 100.0,
    }
    assert selected["information_boundary"]["calibration_episode_read"] is False
    assert validated["passed"] is True


def test_source_selection_and_fixed_trust_unlock_calibration_staging() -> None:
    config = load_reusable_dynamics_config(CONFIG)
    selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    trust = json.loads(SOURCE_TRUST.read_text(encoding="utf-8"))

    selected = validate_reusable_dynamics_source_selection(
        selection, config=config
    )
    compatible = validate_reusable_dynamics_source_trust_compatibility(
        trust, config=config, source_selection=selection
    )

    assert selected["selected_pooled_physical_parameters"] == {
        "dashpot_damping": 100.0,
        "drag_damping": 10.0,
        "init_spring_Y": 10000.0,
    }
    assert compatible["passed"] is True
    assert trust["roles"]["pooled"][
        "execution_balanced_untouched_tail"
    ]["track_improvement_fraction_vs_persistence"] == pytest.approx(
        0.12386025039153435
    )
