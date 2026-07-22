#!/usr/bin/env python3
"""Diagnose a direct metric D405 graph-state update on open PokeFlex sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_bayesian_registration import (  # noqa: E402
    PokeFlexBayesianRegistrationConfig,
    register_pokeflex_graph_posterior,
)
from bayesian_phystwin.pokeflex_independent_depth import (  # noqa: E402
    anchor_fit_scores_mm,
)
from bayesian_phystwin.pokeflex_independent_depth_protocol import (  # noqa: E402
    load_pokeflex_independent_depth_protocol,
)
from bayesian_phystwin.pokeflex_registration_protocol import (  # noqa: E402
    load_pokeflex_registration_protocol,
)
from bayesian_phystwin.pokeflex_released_checkpoint import (  # noqa: E402
    PokeFlexReleasedCheckpoint,
)
from run_pokeflex_bayesian_registration_smoke import (  # noqa: E402
    _cd_ul1_mm,
    _load_mesh,
    _surface_sample,
    _template_frame,
    _view_points,
)
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    _load_official_template,
    _realsense_anchor_inventory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_CD_UL1_mm": float(np.mean(array)),
        "median_CD_UL1_mm": float(np.median(array)),
        "p90_CD_UL1_mm": float(np.quantile(array, 0.9)),
    }


def run_direct_d405_state_diagnostic(
    take_root: Path,
    source_artifact_path: Path,
    independent_protocol_path: Path,
    registration_protocol_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
) -> dict[str, object]:
    """Fit one D405-only graph displacement and score its one-step transfer."""

    independent_protocol = load_pokeflex_independent_depth_protocol(
        independent_protocol_path
    )
    registration_protocol = load_pokeflex_registration_protocol(
        registration_protocol_path
    )
    payload = independent_protocol["payload"]
    method = payload["method_lock"]
    if (
        registration_protocol["protocol_sha256"]
        != payload["parent_protocol"]["protocol_sha256"]
    ):
        raise ValueError("independent-depth parent protocol changed")

    source_artifact = json.loads(source_artifact_path.read_text(encoding="utf-8"))
    if (
        source_artifact.get("artifact_kind")
        != "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke"
    ):
        raise ValueError("unexpected frozen source artifact kind")
    if source_artifact.get("future_observation_used") is not False:
        raise ValueError("frozen source artifact used future observations")
    if source_artifact.get("take", {}).get("id") != take_root.name:
        raise ValueError("source artifact take identity changed")
    source_anchor = source_artifact.get("independent_depth_anchor", {})
    if source_anchor.get("protocol_sha256") != independent_protocol["protocol_sha256"]:
        raise ValueError("source artifact protocol checksum changed")

    robot_path = take_root / "robot_data.json"
    robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(record["frame"]): record for record in robot_records}
    active = [
        frame
        for frame, record in sorted(robot_by_frame.items())
        if float(record["forces"][1]) > 3.0
    ]
    template_frame = _template_frame(active)
    template_path = take_root / "meshes" / f"mesh-f{template_frame:05d}.obj"
    template_vertices, template_faces, template_preprocessing = _load_official_template(
        template_path
    )
    frame_limit = int(source_artifact["take"]["maximum_frame"])

    checkpoint = PokeFlexReleasedCheckpoint.load(
        template_vertices,
        upstream_checkout=upstream_checkout,
        checkpoint_root=checkpoint_root,
    )
    features_by_frame: dict[int, object] = {}
    preprocessing_by_frame: dict[int, object] = {}
    for frame in range(1, frame_limit):
        views = tuple(
            _view_points(take_root, frame, camera, template_vertices)
            for camera in (0, 1)
        )
        feature, preprocessing = checkpoint.encode_frame(views)
        features_by_frame[frame] = feature
        preprocessing_by_frame[frame] = preprocessing
    predictions_by_frame = {}
    for frame in range(6, frame_limit + 1):
        history = range(frame - checkpoint.history_frame_count, frame)
        predictions_by_frame[frame] = checkpoint.predict_from_encoded_history(
            [features_by_frame[index] for index in history],
            [preprocessing_by_frame[index] for index in history],
        )

    support_radius_m = float(method["static_template_support_radius_mm"]) / 1000.0
    anchors, calibrations, calibration_hashes = _realsense_anchor_inventory(
        take_root,
        template_frame,
        template_vertices,
        frame_limit,
        support_radius_m,
    )
    calibration_residual_mm = np.asarray(
        [1000.0 * value.median_residual_m for value in calibrations],
        dtype=np.float64,
    )
    eligible = calibration_residual_mm <= float(
        method["maximum_calibration_median_residual_mm"]
    )

    config = PokeFlexBayesianRegistrationConfig(
        residual_geometry="point_to_point",
        observation_variance_m2=0.004**2,
        camera_bias_variance_m2=0.010**2,
        minimum_independent_view_count=2,
    )
    sample_count = int(
        registration_protocol["payload"]["evaluation"]["sampling"]["surface_points"]
    )
    base_seed = int(
        registration_protocol["payload"]["evaluation"]["sampling"]["seed"]
    )

    baseline_errors: list[float] = []
    candidate_errors: list[float] = []
    target_records = []
    for frozen_target in source_artifact["targets"]:
        target_frame = int(frozen_target["target_frame"])
        source_frame = target_frame - 1
        baseline_error = float(frozen_target["released_checkpoint_CD_UL1_mm"])
        target_prior = predictions_by_frame[target_frame].vertices_m
        candidate = target_prior
        accepted = False
        reason = "exact-baseline-fallback"
        update_diagnostics: dict[str, object] = {}
        source_fit_regret_mm: list[float] = []
        camera_biases_m: list[list[float]] = [[0.0, 0.0, 0.0]] * len(eligible)

        if source_frame in predictions_by_frame and np.all(eligible):
            source_prior = predictions_by_frame[source_frame].vertices_m
            action_supported = float(robot_by_frame[source_frame]["forces"][1]) > 3.0
            anchor = anchors[source_frame]
            sensor_views = tuple(
                anchor.points_m[anchor.sensor_index == index]
                for index in range(len(anchor.sensor_names))
            )
            update = register_pokeflex_graph_posterior(
                source_prior,
                sensor_views,
                action_supported=action_supported,
                prior_faces=template_faces,
                source_reliabilities=eligible.astype(np.float64),
                config=config,
            )
            posterior_scores = anchor_fit_scores_mm(update.posterior_vertices_m, anchor)
            prior_scores = anchor_fit_scores_mm(source_prior, anchor)
            source_fit_regret = posterior_scores - prior_scores
            source_fit_regret_mm = source_fit_regret.tolist()
            update_diagnostics = dict(update.diagnostics)
            camera_biases_m = update.camera_biases_m.tolist()
            if update.accepted and action_supported and np.all(source_fit_regret < 0.0):
                correction = update.posterior_vertices_m - source_prior
                candidate = target_prior + correction
                accepted = True
                reason = "two-d405-direct-graph-update"
            elif not action_supported:
                reason = "no-source-frame-action-support"
            elif not update.accepted:
                reason = update.reason
            else:
                reason = "source-fit-not-supported-by-both-d405s"
        elif not np.all(eligible):
            reason = "fewer-than-two-calibration-qualified-d405s"
        else:
            reason = "no-five-frame-source-prior"

        if not accepted and candidate.tobytes() != target_prior.tobytes():
            raise AssertionError("rejected direct D405 update changed checkpoint bytes")
        target_mesh = _load_mesh(
            take_root / "meshes" / f"mesh-f{target_frame:05d}.obj"
        )
        target_sample = _surface_sample(
            np.asarray(target_mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(target_mesh.faces, dtype=np.int64),
            sample_count,
            base_seed + target_frame,
        )
        baseline_sample = _surface_sample(
            target_prior,
            template_faces,
            sample_count,
            base_seed + target_frame,
        )
        recomputed_baseline = _cd_ul1_mm(baseline_sample, target_sample)
        if not np.isclose(recomputed_baseline, baseline_error, atol=1e-10, rtol=0.0):
            raise ValueError("released-checkpoint target score changed")
        candidate_sample = _surface_sample(
            candidate,
            template_faces,
            sample_count,
            base_seed + target_frame,
        )
        candidate_error = _cd_ul1_mm(candidate_sample, target_sample)
        baseline_errors.append(baseline_error)
        candidate_errors.append(candidate_error)
        target_records.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "accepted": accepted,
                "reason": reason,
                "source_fit_regret_mm": source_fit_regret_mm,
                "camera_biases_m": camera_biases_m,
                "update_diagnostics": update_diagnostics,
                "released_checkpoint_CD_UL1_mm": baseline_error,
                "direct_d405_state_CD_UL1_mm": candidate_error,
                "hidden_difference_mm": candidate_error - baseline_error,
            }
        )

    baseline = np.asarray(baseline_errors, dtype=np.float64)
    candidate = np.asarray(candidate_errors, dtype=np.float64)
    difference = candidate - baseline
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexDirectD405StateDiagnostic",
        "claim_status": "post-open source-only mechanism diagnostic",
        "take": {
            "id": take_root.name,
            "maximum_frame": frame_limit,
            "robot_sha256": _sha256(robot_path),
            "template_sha256": _sha256(template_path),
            "template_preprocessing": template_preprocessing,
        },
        "source_artifact": {
            "path": str(source_artifact_path.resolve()),
            "sha256": _sha256(source_artifact_path),
        },
        "protocol_sha256": independent_protocol["protocol_sha256"],
        "causal_semantics": (
            "D405 observations through f-1 fit a metric graph displacement at f-1; "
            "the displacement is carried once onto the released prediction for f"
        ),
        "future_observation_used": False,
        "target_mesh_used_only_after_prediction_for_source_scoring": True,
        "method": {
            "candidate_count": 1,
            "correction_scale": 1.0,
            "action_support_required": True,
            "both_calibration_qualified_d405s_required": True,
            "both_source_fit_regrets_must_be_negative": True,
            "state_update": config.as_dict(),
            "exact_fallback": "released Kinect checkpoint vertices byte-for-byte",
        },
        "independent_depth_anchor": {
            "sensor_family": "eye-in-hand RealSense D405 depth",
            "calibration_sha256": list(calibration_hashes),
            "calibration_median_residual_mm": calibration_residual_mm.tolist(),
            "eligible_sensor_mask": eligible.tolist(),
            "maximum_template_distance_m": support_radius_m,
        },
        "aggregates": {
            "released_checkpoint": _summary(baseline_errors),
            "direct_d405_state": _summary(candidate_errors),
            "relative_improvement": (baseline_mean - candidate_mean) / baseline_mean,
            "wins": int(np.sum(difference < -1e-12)),
            "losses": int(np.sum(difference > 1e-12)),
            "fallback_ties": int(np.sum(np.abs(difference) <= 1e-12)),
            "accepted_target_count": int(
                np.sum([record["accepted"] for record in target_records])
            ),
        },
        "targets": target_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("source_artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--independent-protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_independent_depth_source_validation_v2.json"
        ),
    )
    parser.add_argument(
        "--registration-protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_bayesian_registration_v1.json"
        ),
    )
    args = parser.parse_args()
    result = run_direct_d405_state_diagnostic(
        args.take_root.resolve(),
        args.source_artifact.resolve(),
        args.independent_protocol.resolve(),
        args.registration_protocol.resolve(),
        args.upstream_checkout.resolve(),
        args.checkpoint_root.resolve(),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing diagnostic differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "take_id": result["take"]["id"],
                "aggregates": result["aggregates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
