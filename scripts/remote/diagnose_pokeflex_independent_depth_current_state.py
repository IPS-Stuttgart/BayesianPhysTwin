#!/usr/bin/env python3
"""Score current PokeFlex state corrections with same-time D405 evidence."""

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
from bayesian_phystwin.pokeflex_released_checkpoint import (  # noqa: E402
    PokeFlexReleasedCheckpoint,
)
from run_pokeflex_bayesian_registration_smoke import (  # noqa: E402
    _template_frame,
    _view_points,
)
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    _candidate_name,
    _correction_field_variants,
    _load_official_template,
    _realsense_anchor_inventory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_current_state_diagnostic(
    take_root: Path,
    source_artifact_path: Path,
    protocol_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
) -> dict[str, object]:
    """Evaluate same-time anchor evidence without reopening target meshes."""

    protocol = load_pokeflex_independent_depth_protocol(protocol_path)
    payload = protocol["payload"]
    method = payload["method_lock"]
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
    anchor_metadata = source_artifact.get("independent_depth_anchor", {})
    if anchor_metadata.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("source artifact protocol checksum changed")
    support_radius_m = float(method["static_template_support_radius_mm"]) / 1000.0
    if float(anchor_metadata.get("maximum_template_distance_m", -1.0)) != support_radius_m:
        raise ValueError("source artifact template support radius changed")
    expected_fields = tuple(map(str, method["correction_fields"]))
    if tuple(source_artifact.get("correction_fields", ())) != expected_fields:
        raise ValueError("source artifact correction fields changed")

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
    template_vertices, template_faces, _ = _load_official_template(template_path)
    frame_limit = int(source_artifact["take"]["maximum_frame"])

    checkpoint = PokeFlexReleasedCheckpoint.load(
        template_vertices,
        upstream_checkout=upstream_checkout,
        checkpoint_root=checkpoint_root,
    )
    views_by_frame: dict[int, tuple[np.ndarray, ...]] = {}
    features_by_frame: dict[int, object] = {}
    preprocessing_by_frame: dict[int, object] = {}
    for frame in range(1, frame_limit):
        views = tuple(
            _view_points(take_root, frame, camera, template_vertices)
            for camera in (0, 1)
        )
        feature, preprocessing = checkpoint.encode_frame(views)
        views_by_frame[frame] = views
        features_by_frame[frame] = feature
        preprocessing_by_frame[frame] = preprocessing
    predictions_by_frame = {}
    for frame in range(6, frame_limit + 1):
        history = range(frame - checkpoint.history_frame_count, frame)
        predictions_by_frame[frame] = checkpoint.predict_from_encoded_history(
            [features_by_frame[index] for index in history],
            [preprocessing_by_frame[index] for index in history],
        )

    anchors, calibrations, calibration_hashes = _realsense_anchor_inventory(
        take_root,
        template_frame,
        template_vertices,
        frame_limit,
        support_radius_m,
    )
    config = PokeFlexBayesianRegistrationConfig(residual_geometry="point_to_point")
    updates_by_frame = {}
    corrections_by_frame: dict[int, np.ndarray] = {}
    for source_frame in range(6, frame_limit):
        source_prior = predictions_by_frame[source_frame].vertices_m
        update = register_pokeflex_graph_posterior(
            source_prior,
            views_by_frame[source_frame],
            action_supported=float(robot_by_frame[source_frame]["forces"][1]) > 3.0,
            prior_faces=template_faces,
            config=config,
        )
        updates_by_frame[source_frame] = update
        corrections_by_frame[source_frame] = update.posterior_vertices_m - source_prior

    target_records = []
    scales = tuple(map(float, method["correction_scales"]))
    for frozen_target in source_artifact["targets"]:
        target_frame = int(frozen_target["target_frame"])
        source_frame = target_frame - 1
        if source_frame not in predictions_by_frame or source_frame not in updates_by_frame:
            target_errors = {
                "released_checkpoint_CD_UL1_mm": float(
                    frozen_target["released_checkpoint_CD_UL1_mm"]
                )
            }
            for field in expected_fields:
                for scale in scales:
                    if scale > 0.0:
                        name = _candidate_name(field, scale)
                        target_errors[name] = float(frozen_target[name])
            target_records.append(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "action_supported": False,
                    "update_accepted": False,
                    "current_state_anchor_regret": {},
                    **target_errors,
                }
            )
            continue
        source_prior = predictions_by_frame[source_frame].vertices_m
        target_prior = predictions_by_frame[target_frame].vertices_m
        update = updates_by_frame[source_frame]
        correction = corrections_by_frame[source_frame]
        action_supported = float(robot_by_frame[source_frame]["forces"][1]) > 3.0
        variants = _correction_field_variants(
            source_prior,
            target_prior,
            correction,
            expected_fields,
            previous_correction=corrections_by_frame.get(source_frame - 1),
            tool_positions=np.asarray(
                [
                    robot_by_frame[frame]["T_WT"]
                    for frame in range(max(1, source_frame - 3), source_frame + 1)
                ],
                dtype=np.float64,
            )[:, :3, 3],
            end_effector_positions=np.asarray(
                [
                    robot_by_frame[frame]["T_WE"]
                    for frame in range(max(1, source_frame - 3), source_frame + 1)
                ],
                dtype=np.float64,
            )[:, :3, 3],
            force_vectors=np.asarray(
                [
                    robot_by_frame[frame]["forces"][:3]
                    for frame in range(max(1, source_frame - 3), source_frame + 1)
                ],
                dtype=np.float64,
            ),
        )
        if not update.accepted or not action_supported:
            for field in variants:
                if field.startswith(("action_", "force_")):
                    variants[field] = np.zeros_like(source_prior)

        anchor = anchors[source_frame]
        baseline_scores = anchor_fit_scores_mm(source_prior, anchor)
        current_regret = {}
        target_errors = {
            "released_checkpoint_CD_UL1_mm": float(
                frozen_target["released_checkpoint_CD_UL1_mm"]
            )
        }
        for field, field_correction in variants.items():
            for scale in scales:
                name = _candidate_name(field, scale)
                if scale == 0.0:
                    continue
                source_candidate = source_prior + scale * field_correction
                scores = anchor_fit_scores_mm(source_candidate, anchor)
                regret = scores - baseline_scores
                current_regret[name] = {
                    "evidence_frame": source_frame,
                    "target_prediction_frame": target_frame,
                    "per_sensor_mm": regret.tolist(),
                    "mean_mm": float(np.mean(regret)),
                    "covariance_intersection_upper_mm": float(np.max(regret)),
                }
                target_errors[name] = float(frozen_target[name])
        target_records.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "action_supported": action_supported,
                "update_accepted": bool(update.accepted),
                "current_state_anchor_regret": current_regret,
                **target_errors,
            }
        )

    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexIndependentDepthCurrentStateDiagnostic",
        "claim_status": "post-open source-only mechanism diagnostic",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_artifact": {
            "path": str(source_artifact_path.resolve()),
            "sha256": _sha256(source_artifact_path),
        },
        "take": {
            "id": take_root.name,
            "maximum_frame": frame_limit,
            "robot_sha256": _sha256(robot_path),
            "template_sha256": _sha256(template_path),
        },
        "causal_semantics": (
            "D405 frame f-1 scores correction candidates for state f-1; the same "
            "material correction is propagated once to predict f"
        ),
        "future_observation_used": False,
        "target_mesh_used_by_diagnostic_runner": False,
        "independent_depth_anchor": {
            "sensor_family": "eye-in-hand RealSense D405 depth",
            "calibration_sha256": list(calibration_hashes),
            "median_residual_mm": [
                1000.0 * value.median_residual_m for value in calibrations
            ],
            "p90_residual_mm": [
                1000.0 * value.p90_residual_m for value in calibrations
            ],
            "maximum_template_distance_m": support_radius_m,
        },
        "correction_fields": list(expected_fields),
        "correction_scales": list(scales),
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
        "--protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_independent_depth_source_validation_v2.json"
        ),
    )
    args = parser.parse_args()
    result = run_current_state_diagnostic(
        args.take_root.resolve(),
        args.source_artifact.resolve(),
        args.protocol.resolve(),
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
                "target_count": len(result["targets"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
