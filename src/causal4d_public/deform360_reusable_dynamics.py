"""Frozen boundary for the first Deform360 reusable-twin dynamics test."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping


REUSABLE_DYNAMICS_SCHEMA_VERSION = 1
REUSABLE_DYNAMICS_PROTOCOL_ID = "deform360-reusable-dynamics-081-v1"
CANONICAL_REUSABLE_DYNAMICS_CONFIG_SHA256 = (
    "9fceb7a417822b979a22313285bbaf02cb0f4e5506fef953c38a8e4ae5566d6c"
)
REUSABLE_DYNAMICS_PIPELINE_PROTOCOL_ID = (
    "deform360-reusable-dynamics-pipeline-081-v1"
)
CANONICAL_REUSABLE_DYNAMICS_PIPELINE_CONFIG_SHA256 = (
    "e32c20e98442e7112a79c1d54de3f58a4608d9c382739f05a10085df53d42039"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reusable_dynamics_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reusable_dynamics_result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_reusable_dynamics_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == REUSABLE_DYNAMICS_SCHEMA_VERSION,
        "unsupported reusable-dynamics schema",
    )
    observed = reusable_dynamics_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "reusable-dynamics checksum mismatch",
    )
    _require(
        observed == CANONICAL_REUSABLE_DYNAMICS_CONFIG_SHA256,
        "reusable-dynamics protocol differs from the canonical lock",
    )
    config = payload.get("config")
    _require(isinstance(config, Mapping), "reusable-dynamics config is missing")
    _require(
        config.get("protocol_id") == REUSABLE_DYNAMICS_PROTOCOL_ID,
        "reusable-dynamics protocol id changed",
    )
    _require(
        config.get("status") == "locked-before-calibration-dynamics-media-read",
        "reusable-dynamics protocol was not frozen prospectively",
    )

    partition = config.get("episode_partition", {})
    source = tuple(int(value) for value in partition.get("source_selection", ()))
    calibration = tuple(
        int(value) for value in partition.get("independent_calibration", ())
    )
    target = tuple(int(value) for value in partition.get("sealed_target", ()))
    _require(source == (1, 4, 6), "source episode partition changed")
    _require(calibration == (0, 2, 8), "calibration episode partition changed")
    _require(target == (5,), "sealed target episode changed")
    _require(
        not (set(source) & set(calibration) or set(source) & set(target)),
        "source episodes overlap evaluation episodes",
    )
    _require(
        not set(calibration) & set(target),
        "calibration episodes overlap the sealed target",
    )
    _require(
        partition.get("target_may_open_only_after_all_calibration_gates_pass")
        is True,
        "target opening is not gated",
    )

    frames = config.get("frame_protocol", {})
    _require(
        frames.get("raw_aligned_range_half_open") == [110, 191],
        "raw dynamics frame range changed",
    )
    _require(frames.get("raw_frame_count") == 81, "raw frame count changed")
    _require(
        frames.get("tracking_tail_frames_skipped") == 5,
        "tracking-tail policy changed",
    )
    _require(
        frames.get("processed_frame_count") == 76,
        "processed frame count changed",
    )
    _require(
        frames.get("independent_calibration_prediction_range_half_open") == [1, 76],
        "calibration prediction horizon changed",
    )
    _require(
        frames.get("future_frame_allowed_for_initial_association") is False,
        "future frame leaked into initial association",
    )

    simulator = config.get("official_phystwin", {})
    grid = simulator.get("source_parameter_grid", {})
    candidates = tuple(
        itertools.product(
            grid.get("init_spring_Y", ()),
            grid.get("drag_damping", ()),
            grid.get("dashpot_damping", ()),
        )
    )
    selection = config.get("source_selection", {})
    _require(len(candidates) == 24, "physical source grid must contain 24 tuples")
    _require(
        selection.get("candidate_count") == len(candidates),
        "declared source candidate count disagrees with the grid",
    )
    _require(
        selection.get("candidate_generation_uses_source_episodes_only") is True,
        "physical candidates may use evaluation episodes",
    )
    _require(
        selection.get("candidate_prediction_arm")
        == "raw_official_phystwin_driven",
        "physical selection arm changed",
    )
    _require(
        selection.get("invalid_candidate_policy")
        == (
            "retain the attempted candidate in the audit and reject it jointly "
            "if any source rollout is non-finite; never substitute or tune around "
            "the failure"
        ),
        "invalid physical-candidate policy changed",
    )
    _require(
        selection.get("selection_artifact_must_be_frozen_before_calibration_scoring")
        is True,
        "source selection need not freeze before calibration",
    )
    _require(
        selection.get("calibration_or_target_outcomes_allowed") is False,
        "source selection may inspect evaluation outcomes",
    )

    trust = config.get("fixed_action_trust", {})
    _require(
        trust.get("base_action_response") == 0.4,
        "source-frozen action response changed",
    )
    _require(
        trust.get("autonomous_drift") == 0.1,
        "source-frozen autonomous drift changed",
    )
    _require(
        trust.get("weights_may_change_after_freeze") is False,
        "action-trust weights may change after evaluation",
    )
    source_gates = config.get("source_compatibility_gates", {})
    _require(
        source_gates.get("minimum_untouched_tail_track_improvement_fraction")
        == 0.0
        and source_gates.get("minimum_untouched_tail_cd_improvement_fraction")
        == 0.0
        and source_gates.get("minimum_joint_win_episode_count") == 2
        and source_gates.get("maximum_per_episode_degradation_fraction_per_metric")
        == 0.25
        and source_gates.get("all_gates_conjunctive") is True,
        "source compatibility gate changed",
    )

    gates = config.get("calibration_gates", {})
    _require(gates.get("all_gates_conjunctive") is True, "gates are not conjunctive")
    _require(
        gates.get("conformal", {}).get("order_statistic_rank") == 3,
        "conformal rank is inconsistent with three calibration executions",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("calibration_future_allowed_for_method_or_hyperparameter_changes")
        is False,
        "calibration futures may tune the method",
    )
    _require(
        boundary.get("target_media_allowed_before_all_calibration_gates_pass")
        is False,
        "target media may open before calibration passes",
    )
    _require(
        boundary.get("target_outcomes_allowed_for_method_selection") is False,
        "target outcomes may select the method",
    )
    claims = config.get("claim_boundary", {})
    _require(
        claims.get("state_of_the_art_claim") is False,
        "prospective protocol already claims state of the art",
    )
    _require(
        claims.get("full_dense_phystwin_required_for_state_of_the_art_claim") is True,
        "sparse diagnostic backend may claim state of the art",
    )
    return {
        "passed": True,
        "protocol_id": REUSABLE_DYNAMICS_PROTOCOL_ID,
        "config_sha256": observed,
        "source_episodes": list(source),
        "calibration_episodes": list(calibration),
        "sealed_target_episodes": list(target),
        "physical_candidate_count": len(candidates),
    }


def load_reusable_dynamics_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "reusable-dynamics file must be an object")
    validate_reusable_dynamics_config(payload)
    return payload


def validate_reusable_dynamics_pipeline_config(
    payload: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    parent_validated = validate_reusable_dynamics_config(parent)
    _require(payload.get("schema_version") == 1, "unsupported pipeline schema")
    observed = reusable_dynamics_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "reusable-dynamics pipeline checksum mismatch",
    )
    _require(
        observed == CANONICAL_REUSABLE_DYNAMICS_PIPELINE_CONFIG_SHA256,
        "reusable-dynamics pipeline differs from the canonical lock",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == REUSABLE_DYNAMICS_PIPELINE_PROTOCOL_ID,
        "reusable-dynamics pipeline id changed",
    )
    _require(
        config.get("parent_config_sha256") == parent_validated["config_sha256"],
        "reusable-dynamics pipeline uses another parent protocol",
    )
    reconstruction = config.get("reconstruction", {})
    _require(
        reconstruction
        == {
            "minimum_visual_hull_points": 512,
            "voxel_resolution": 120,
            "cube_half_extent_m": 0.5,
            "first_frame_iterations": 500,
            "warm_start_iterations": 250,
            "warm_start_from_previous_frame": True,
        },
        "reconstruction settings changed",
    )
    tracking = config.get("tracking", {})
    _require(
        tracking.get("checkpoint_sha256")
        == "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834",
        "tracking checkpoint changed",
    )
    point_cloud = config.get("point_cloud", {})
    _require(
        point_cloud.get("rng_seed") == 0
        and point_cloud.get("tracking_tail_frames_skipped") == 5
        and point_cloud.get("expected_frame_count") == 76
        and point_cloud.get("correlation_aware_variant_is_primary") is False,
        "point-cloud pipeline changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("calibration_reconstruction_or_prediction_metrics_computed_before_this_addendum")
        is False,
        "pipeline was not frozen before reconstruction outcomes",
    )
    _require(
        boundary.get("calibration_outcomes_allowed_for_pipeline_changes") is False,
        "calibration outcomes may change the observation pipeline",
    )
    _require(
        boundary.get("target_media_read") is False,
        "pipeline addendum read target media",
    )
    return {
        "passed": True,
        "protocol_id": REUSABLE_DYNAMICS_PIPELINE_PROTOCOL_ID,
        "config_sha256": observed,
        "parent_config_sha256": parent_validated["config_sha256"],
    }


def load_reusable_dynamics_pipeline_config(
    path: str | Path,
    *,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "pipeline config must be an object")
    validate_reusable_dynamics_pipeline_config(payload, parent=parent)
    return payload


def validate_reusable_dynamics_association_evidence(
    payload: Mapping[str, Any],
    *,
    mask_summary_path: str | Path,
    prefix_summary_path: str | Path,
) -> dict[str, Any]:
    """Verify the independent association evidence that unlocks dynamics staging."""

    validate_reusable_dynamics_config(payload)
    parent = payload["config"]["parent_association"]
    mask_path = Path(mask_summary_path)
    prefix_path = Path(prefix_summary_path)
    _require(
        _sha256_file(mask_path) == parent["mask_summary_file_sha256"],
        "association mask summary hash mismatch",
    )
    _require(
        _sha256_file(prefix_path) == parent["prefix_summary_file_sha256"],
        "association prefix summary hash mismatch",
    )
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    expected_parent_hash = parent["config_sha256"]
    _require(
        mask.get("config_sha256") == expected_parent_hash,
        "mask summary uses another association protocol",
    )
    _require(
        prefix.get("config_sha256") == expected_parent_hash,
        "prefix summary uses another association protocol",
    )
    _require(
        mask.get("conjunctive_mask_gate_passed") is True,
        "association mask gate did not pass",
    )
    _require(
        prefix.get("conjunctive_reusable_association_gate_passed") is True,
        "association prefix gate did not pass",
    )
    expected_episodes = payload["config"]["episode_partition"][
        "independent_calibration"
    ]
    _require(
        sorted(int(item["episode_id"]) for item in mask.get("episodes", ()))
        == expected_episodes,
        "mask summary covers another calibration partition",
    )
    _require(
        sorted(int(item["episode_id"]) for item in prefix.get("episodes", ()))
        == expected_episodes,
        "prefix summary covers another calibration partition",
    )
    _require(
        mask.get("information_boundary", {}).get("future_prediction_metrics_computed")
        is False,
        "association mask gate inspected prediction outcomes",
    )
    _require(
        prefix.get("information_boundary", {}).get(
            "future_prediction_metrics_computed"
        )
        is False,
        "association prefix gate inspected prediction outcomes",
    )
    _require(
        mask.get("information_boundary", {}).get("target_media_read") is False,
        "association mask gate read the sealed target",
    )
    _require(
        prefix.get("information_boundary", {}).get("target_media_read") is False,
        "association prefix gate read the sealed target",
    )
    return {
        "passed": True,
        "mask_summary_sha256": parent["mask_summary_file_sha256"],
        "prefix_summary_sha256": parent["prefix_summary_file_sha256"],
        "calibration_episodes": list(expected_episodes),
    }


def validate_reusable_dynamics_source_request(
    payload: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    validated = validate_reusable_dynamics_config(payload)
    config = payload["config"]
    _require(object_id == config["object_id"], "object is outside this protocol")
    _require(
        int(episode_id) in config["episode_partition"]["source_selection"],
        "episode is not in the source partition",
    )
    return {
        **validated,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "split": "source",
        "allowed_raw_frame_range": list(
            config["frame_protocol"]["raw_aligned_range_half_open"]
        ),
    }


def validate_reusable_dynamics_calibration_request(
    payload: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    operation: str,
) -> dict[str, Any]:
    """Authorize only the declared calibration read for one operation."""

    validated = validate_reusable_dynamics_config(payload)
    config = payload["config"]
    _require(object_id == config["object_id"], "object is outside this protocol")
    _require(
        int(episode_id) in config["episode_partition"]["independent_calibration"],
        "episode is not in the independent calibration partition",
    )
    allowed = {
        "initial-association": [110, 111],
        "staging": list(config["frame_protocol"]["raw_aligned_range_half_open"]),
        "one-shot-scoring": list(
            config["frame_protocol"][
                "independent_calibration_prediction_range_half_open"
            ]
        ),
    }
    _require(operation in allowed, "unsupported calibration operation")
    return {
        **validated,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "split": "independent-calibration",
        "operation": operation,
        "allowed_frame_range": allowed[operation],
        "method_or_hyperparameter_changes_allowed": False,
        "target_media_allowed": False,
    }


def _parameter_label(parameters: Mapping[str, float]) -> str:
    return (
        f"y{int(parameters['init_spring_Y'])}"
        f"-drag{int(parameters['drag_damping'])}"
        f"-dash{int(parameters['dashpot_damping'])}"
    )


def _relative_raw_score(metrics: Mapping[str, Any]) -> float:
    track = float(metrics["track_rmse_m"])
    chamfer = float(metrics["chamfer_m"])
    persistence_track = float(metrics["persistence_track_rmse_m"])
    persistence_chamfer = float(metrics["persistence_chamfer_m"])
    _require(
        all(
            math.isfinite(value)
            for value in (track, chamfer, persistence_track, persistence_chamfer)
        ),
        "source-grid score is non-finite",
    )
    _require(
        persistence_track > 0.0 and persistence_chamfer > 0.0,
        "source-grid persistence denominator is zero",
    )
    return 0.5 * (
        track / persistence_track + chamfer / persistence_chamfer
    )


def select_reusable_dynamics_source_grid(
    payload: Mapping[str, Any],
    *,
    grid_root: str | Path,
) -> dict[str, Any]:
    """Select pooled and single-source physical tuples without evaluation data."""

    validated = validate_reusable_dynamics_config(payload)
    config = payload["config"]
    root = Path(grid_root)
    simulator = config["official_phystwin"]
    grid = simulator["source_parameter_grid"]
    candidates = [
        {
            "init_spring_Y": float(init_spring_y),
            "drag_damping": float(drag_damping),
            "dashpot_damping": float(dashpot_damping),
        }
        for init_spring_y, drag_damping, dashpot_damping in itertools.product(
            grid["init_spring_Y"],
            grid["drag_damping"],
            grid["dashpot_damping"],
        )
    ]
    source_records = {
        int(record["episode_id"]): record for record in config["source_inputs"]
    }
    source_ids = [int(value) for value in validated["source_episodes"]]
    selection_range = config["source_selection"]["score_range_half_open"]
    table: list[dict[str, Any]] = []
    persistence_reference: dict[int, tuple[float, float]] = {}
    for parameters in candidates:
        label = _parameter_label(parameters)
        by_episode: dict[str, Any] = {}
        scores = []
        candidate_eligible = True
        for episode_id in source_ids:
            result_path = (
                root
                / f"ep{episode_id}"
                / label
                / "official_phystwin_smoke.json"
            )
            trajectory_path = result_path.with_name("official_phystwin_trajectory.npz")
            _require(result_path.is_file(), f"missing source-grid result: {result_path}")
            _require(
                trajectory_path.is_file(),
                f"missing source-grid trajectory: {trajectory_path}",
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rollout_passed = result.get("passed") is True
            _require(
                isinstance(result.get("passed"), bool),
                "source-grid rollout has no pass/fail status",
            )
            _require(
                result.get("source_only_smoke") is True,
                "source-grid rollout is not source-only",
            )
            _require(
                result.get("official_phystwin_revision")
                == simulator["upstream_revision"],
                "source-grid PhysTwin revision changed",
            )
            _require(
                result.get("config_sha256") == simulator["real_config_sha256"],
                "source-grid PhysTwin config changed",
            )
            _require(
                result.get("split_sha256") == simulator["source_split_sha256"],
                "source-grid split changed",
            )
            _require(
                result.get("data_sha256")
                == source_records[episode_id]["controller_bundle_sha256"],
                "source-grid controller bundle changed",
            )
            _require(
                result.get("support_dynamics", {}).get("mode")
                == simulator["support_mode"],
                "source-grid support mode changed",
            )
            overrides = result.get("config_overrides", {})
            expected_overrides = {
                **simulator["fixed_overrides"],
                "init_spring_Y": parameters["init_spring_Y"],
                "drag_damping": parameters["drag_damping"],
                "dashpot_damping": parameters["dashpot_damping"],
            }
            _require(
                overrides == expected_overrides,
                "source-grid physical parameters changed",
            )
            _require(
                int(result.get("num_controller_points", -1))
                == int(source_records[episode_id]["controller_count"]),
                "source-grid controller count changed",
            )
            _require(
                result.get("trajectory_sha256") == _sha256_file(trajectory_path),
                "source-grid trajectory checksum changed",
            )
            metrics = result.get("metrics", {})
            train = metrics.get("intervals", {}).get("train", {})
            tail = metrics.get("intervals", {}).get("test", {})
            _require(
                train.get("frame_range") == selection_range,
                "source-grid train interval changed",
            )
            reference = (
                float(train["persistence_track_rmse_m"]),
                float(train["persistence_chamfer_m"]),
            )
            if episode_id in persistence_reference:
                _require(
                    all(
                        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
                        for a, b in zip(reference, persistence_reference[episode_id])
                    ),
                    "persistence baseline changes across physical candidates",
                )
            else:
                persistence_reference[episode_id] = reference
            if rollout_passed and train.get("prediction_finite") is True:
                score: float | None = _relative_raw_score(train)
                scores.append(score)
            else:
                score = None
                candidate_eligible = False
            by_episode[str(episode_id)] = {
                "relative_score_vs_persistence": score,
                "rollout_passed": rollout_passed,
                "first_nonfinite_frame": result.get("first_nonfinite_frame"),
                "train_metrics": train,
                "untouched_tail_metrics": tail,
                "result_file_sha256": _sha256_file(result_path),
                "trajectory_sha256": result["trajectory_sha256"],
            }
        table.append(
            {
                "physical_parameters": parameters,
                "candidate_label": label,
                "eligible": candidate_eligible,
                "pooled_relative_score_vs_persistence": (
                    float(sum(scores) / len(scores)) if candidate_eligible else None
                ),
                "by_episode": by_episode,
            }
        )

    def tie_key(row: Mapping[str, Any], *, score: float) -> tuple[float, ...]:
        parameters = row["physical_parameters"]
        return (
            score,
            float(parameters["init_spring_Y"]),
            float(parameters["drag_damping"]),
            float(parameters["dashpot_damping"]),
        )

    eligible_table = [row for row in table if row["eligible"]]
    _require(
        len(eligible_table) >= 2,
        "fewer than two physical candidates are finite on every source episode",
    )
    pooled = min(
        eligible_table,
        key=lambda row: tie_key(
            row, score=float(row["pooled_relative_score_vs_persistence"])
        ),
    )
    single = {
        str(episode_id): min(
            eligible_table,
            key=lambda row: tie_key(
                row,
                score=float(
                    row["by_episode"][str(episode_id)][
                        "relative_score_vs_persistence"
                    ]
                ),
            ),
        )["physical_parameters"]
        for episode_id in source_ids
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableDynamicsSourceGridSelection",
        "protocol_id": REUSABLE_DYNAMICS_PROTOCOL_ID,
        "config_sha256": validated["config_sha256"],
        "object_id": config["object_id"],
        "source_episode_ids": source_ids,
        "candidate_count": len(table),
        "eligible_candidate_count": len(eligible_table),
        "rejected_candidate_labels": [
            row["candidate_label"] for row in table if not row["eligible"]
        ],
        "selection_arm": config["source_selection"]["candidate_prediction_arm"],
        "selection_frame_range": selection_range,
        "score": config["source_selection"]["score"],
        "selected_pooled_physical_parameters": pooled["physical_parameters"],
        "selected_pooled_relative_score_vs_persistence": pooled[
            "pooled_relative_score_vs_persistence"
        ],
        "selected_single_source_physical_parameters": single,
        "candidate_table": table,
        "information_boundary": {
            "source_episodes_read": source_ids,
            "source_train_outcomes_used_for_selection": True,
            "source_untouched_tails_reported_but_not_selected": True,
            "calibration_episode_read": False,
            "target_episode_read": False,
        },
        "claim_boundary": (
            "source-only physical selection; no reusable-dynamics transfer or SOTA claim"
        ),
    }
    result["result_sha256"] = reusable_dynamics_result_sha256(result)
    return result


def validate_reusable_dynamics_source_selection(
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_reusable_dynamics_config(config)
    _require(
        payload.get("artifact_kind")
        == "Deform360ReusableDynamicsSourceGridSelection",
        "unexpected source-selection artifact",
    )
    _require(
        payload.get("config_sha256") == validated["config_sha256"],
        "source selection uses another dynamics protocol",
    )
    _require(
        payload.get("result_sha256") == reusable_dynamics_result_sha256(payload),
        "source-selection checksum mismatch",
    )
    _require(payload.get("candidate_count") == 24, "source grid is incomplete")
    _require(
        isinstance(payload.get("eligible_candidate_count"), int)
        and 2 <= int(payload["eligible_candidate_count"]) <= 24,
        "source grid has invalid eligible-candidate support",
    )
    _require(
        payload.get("source_episode_ids") == validated["source_episodes"],
        "source selection uses another episode partition",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("calibration_episode_read") is False,
        "source selection read calibration data",
    )
    _require(
        boundary.get("target_episode_read") is False,
        "source selection read target data",
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "selected_pooled_physical_parameters": payload[
            "selected_pooled_physical_parameters"
        ],
    }


def validate_reusable_dynamics_source_trust_compatibility(
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    source_selection: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_reusable_dynamics_config(config)
    selection = validate_reusable_dynamics_source_selection(
        source_selection, config=config
    )
    _require(
        payload.get("artifact_kind")
        == "Deform360ReusableDynamicsSourceTrustCompatibility",
        "unexpected source-trust compatibility artifact",
    )
    _require(
        payload.get("config_sha256") == validated["config_sha256"],
        "source-trust compatibility uses another dynamics protocol",
    )
    _require(
        payload.get("source_selection_result_sha256")
        == selection["result_sha256"],
        "source-trust compatibility uses another physical selection",
    )
    _require(
        payload.get("result_sha256") == reusable_dynamics_result_sha256(payload),
        "source-trust compatibility checksum mismatch",
    )
    gates = payload.get("gates", {})
    _require(
        isinstance(gates, Mapping)
        and bool(gates)
        and all(value is True for value in gates.values()),
        "source-trust compatibility gates did not all pass",
    )
    _require(payload.get("passed") is True, "source-trust compatibility failed")
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("calibration_episode_read") is False,
        "source-trust compatibility read calibration data",
    )
    _require(
        boundary.get("target_episode_read") is False,
        "source-trust compatibility read target data",
    )
    _require(
        boundary.get("method_or_hyperparameter_changes_allowed") is False,
        "source-trust compatibility permits post-gate tuning",
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "source_selection_result_sha256": selection["result_sha256"],
    }


__all__ = [
    "CANONICAL_REUSABLE_DYNAMICS_CONFIG_SHA256",
    "CANONICAL_REUSABLE_DYNAMICS_PIPELINE_CONFIG_SHA256",
    "REUSABLE_DYNAMICS_PROTOCOL_ID",
    "REUSABLE_DYNAMICS_PIPELINE_PROTOCOL_ID",
    "load_reusable_dynamics_config",
    "load_reusable_dynamics_pipeline_config",
    "reusable_dynamics_config_sha256",
    "reusable_dynamics_result_sha256",
    "select_reusable_dynamics_source_grid",
    "validate_reusable_dynamics_association_evidence",
    "validate_reusable_dynamics_calibration_request",
    "validate_reusable_dynamics_config",
    "validate_reusable_dynamics_pipeline_config",
    "validate_reusable_dynamics_source_request",
    "validate_reusable_dynamics_source_selection",
    "validate_reusable_dynamics_source_trust_compatibility",
]
