#!/usr/bin/env python3
"""Run and score the locked physical grid on one Deform360 fit episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_reusable_sota_method import (
    load_reusable_sota_method,
    reusable_sota_physical_candidates,
)
from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_reusable_sota_selection import (
    score_reusable_sota_trajectory,
)
from causal4d_public.deform360_reusable_sota_window import (
    authorize_development_fit_window,
    load_reusable_sota_window,
)
from causal4d_public.deform360_sota_processing import (
    authorize_development_processing,
    validate_development_final_data_input,
)
from run_deform360_reusable_sota_prediction_bank import (
    _git_revision,
    _run_rollout,
    _sha256_file,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--window-addendum", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--development-observations", type=Path, required=True)
    parser.add_argument("--source-final-data", type=Path, required=True)
    parser.add_argument("--simulator-final-data", type=Path, required=True)
    parser.add_argument("--episode-graph", type=Path, required=True)
    parser.add_argument("--state-artifact", type=Path, required=True)
    parser.add_argument("--twin-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _nonfinite_rollout_failure(
    *,
    output_dir: Path,
    data_path: Path,
    candidate: Mapping[str, Any],
    controller_scale: float,
    warp: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a validated numerical-instability record, if one exists."""

    result_path = output_dir / "official_phystwin_smoke.json"
    trajectory_path = output_dir / "official_phystwin_trajectory.npz"
    if not result_path.is_file() or not trajectory_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_overrides = {
        "controller_max_neighbours": int(warp["controller_max_neighbours"]),
        "controller_radius": float(warp["controller_radius_m"]),
        "dashpot_damping": float(candidate["dashpot_damping"]),
        "drag_damping": float(candidate["drag_damping"]),
        "init_spring_Y": float(candidate["init_spring_y"]),
    }
    metrics = result.get("metrics", {})
    if not (
        result.get("passed") is False
        and result.get("official_phystwin_revision") == warp["revision"]
        and result.get("config_sha256") == warp["real_config_sha256"]
        and result.get("config_overrides") == expected_overrides
        and result.get("data_sha256") == _sha256_file(data_path)
        and result.get("trajectory_sha256") == _sha256_file(trajectory_path)
        and float(
            result.get("realized_actuation", {}).get(
                "controller_displacement_scale", -1.0
            )
        )
        == controller_scale
        and metrics.get("reason") == "nonfinite_rollout"
    ):
        return None
    return {
        "failure_kind": "nonfinite_rollout",
        "first_nonfinite_frame": int(result["first_nonfinite_frame"]),
        "finite_vertex_fraction": float(result["finite_vertex_fraction"]),
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256_file(result_path),
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": _sha256_file(trajectory_path),
    }


def _load_twin_summary(
    path: Path,
    *,
    object_id: str,
    episode_id: int,
    processing_authorization: Mapping[str, Any],
    window_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    boundary = payload.get("information_boundary", {})
    sota = payload.get("sota_authorization", {})
    _require(
        payload.get("artifact_kind") == "Deform360AutomaticEpisodeTwin"
        and payload.get("result_sha256") == _canonical_sha256(payload)
        and payload.get("object_id") == object_id
        and int(payload.get("episode_id", -1)) == episode_id
        and payload.get("phase") == "source"
        and payload.get("passed") is True
        and sota.get("processing") == dict(processing_authorization)
        and sota.get("window") == dict(window_authorization),
        "source twin summary is incompatible",
    )
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("post_initial_object_observation_used") is False
        and boundary.get("simulator_residual_used") is False
        and boundary.get("target_access") is False
        and boundary.get("prediction_only_input_required") is False,
        "source twin used post-initial evidence",
    )
    return payload


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_sota_config(args.protocol)
    window = load_reusable_sota_window(args.window_addendum)
    method = load_reusable_sota_method(args.method)
    processing_authorization = authorize_development_processing(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        role="fit",
    )
    window_authorization = authorize_development_fit_window(
        protocol,
        window,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    _require(
        method["config"]["parent_config_sha256"] == protocol["config_sha256"]
        and method["config"]["window_config_sha256"] == window["config_sha256"],
        "source bank uses another method lock",
    )
    observations = json.loads(
        args.development_observations.read_text(encoding="utf-8")
    )
    final_data_validation = validate_development_final_data_input(
        observations,
        authorization=processing_authorization,
        final_data_path=args.source_final_data,
    )
    twin = _load_twin_summary(
        args.twin_summary,
        object_id=args.object_id,
        episode_id=args.episode_id,
        processing_authorization=processing_authorization,
        window_authorization=window_authorization,
    )
    _require(
        twin.get("input_sha256", {}).get("episode_final_data")
        == _sha256_file(args.source_final_data)
        and twin.get("input_sha256", {}).get("development_observations")
        == _sha256_file(args.development_observations)
        and twin.get("output_sha256", {}).get("simulator_final_data")
        == _sha256_file(args.simulator_final_data)
        and twin.get("output_sha256", {}).get("episode_graph")
        == _sha256_file(args.episode_graph)
        and twin.get("output_sha256", {}).get("state_artifact")
        == _sha256_file(args.state_artifact),
        "source twin inputs or outputs changed",
    )

    warp = method["config"]["official_warp"]
    real_config = args.official_phystwin_repo / "configs" / "real.yaml"
    _require(
        _git_revision(args.official_phystwin_repo) == warp["revision"]
        and _sha256_file(real_config) == warp["real_config_sha256"],
        "official PhysTwin checkout changed",
    )
    with args.source_final_data.open("rb") as stream:
        source = pickle.load(stream)
    target = np.asarray(source["object_points"], dtype=np.float64)
    frame_count = int(method["config"]["prediction_bank"]["frame_count"])
    _require(
        target.shape[0] == frame_count
        == int(final_data_validation["point_frame_count"]),
        "source trajectory differs from the locked horizon",
    )
    persistence = np.repeat(target[:1], frame_count, axis=0)
    with np.load(args.state_artifact, allow_pickle=False) as archive:
        readout_weights = np.asarray(archive["readout_weights"], dtype=np.float64)
    _require(
        readout_weights.shape[0] == target.shape[1],
        "source readout differs from the material identities",
    )
    frame_protocol = window["config"]["frame_protocol"]
    ranges = {
        "full": frame_protocol["evaluation_range_half_open"],
        **frame_protocol["horizon_ranges_half_open"],
    }
    persistence_metrics = score_reusable_sota_trajectory(
        target,
        persistence,
        horizon_ranges_half_open=ranges,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    runner = args.repo / "scripts/remote/run_deform360_official_phystwin_smoke.py"
    records = []
    for candidate in reusable_sota_physical_candidates(method):
        candidate_root = args.output_root / candidate["label"]
        try:
            driven = _run_rollout(
                runner=runner,
                official_repo=args.official_phystwin_repo,
                data_path=args.simulator_final_data,
                graph_path=args.episode_graph,
                real_config=real_config,
                output_dir=candidate_root / "driven",
                device=args.device,
                candidate=candidate,
                controller_scale=1.0,
                warp=warp,
            )
            zero = _run_rollout(
                runner=runner,
                official_repo=args.official_phystwin_repo,
                data_path=args.simulator_final_data,
                graph_path=args.episode_graph,
                real_config=real_config,
                output_dir=candidate_root / "zero",
                device=args.device,
                candidate=candidate,
                controller_scale=0.0,
                warp=warp,
            )
        except (RuntimeError, ValueError):
            failures = {
                phase: failure
                for phase, scale in (("driven", 1.0), ("zero", 0.0))
                if (
                    failure := _nonfinite_rollout_failure(
                        output_dir=candidate_root / phase,
                        data_path=args.simulator_final_data,
                        candidate=candidate,
                        controller_scale=scale,
                        warp=warp,
                    )
                )
                is not None
            }
            if not failures:
                raise
            records.append(
                {
                    **candidate,
                    "valid": False,
                    "failure": {
                        "claim_boundary": (
                            "locked numerical-instability result; candidate is "
                            "retained in the grid but ineligible for selection"
                        ),
                        "rollouts": failures,
                    },
                    "metrics": None,
                    "absolute_driven_metrics": None,
                    "absolute_zero_metrics": None,
                    "prediction_path": None,
                    "prediction_sha256": None,
                    "maximum_dense_response_m": None,
                }
            )
            continue
        with np.load(driven["trajectory_path"], allow_pickle=False) as archive:
            driven_graph = np.asarray(archive["vertices"], dtype=np.float64)
        with np.load(zero["trajectory_path"], allow_pickle=False) as archive:
            zero_graph = np.asarray(archive["vertices"], dtype=np.float64)
        _require(
            driven_graph.shape == zero_graph.shape
            and driven_graph.shape[0] == frame_count
            and driven_graph.shape[1] == readout_weights.shape[1]
            and np.all(np.isfinite(driven_graph))
            and np.all(np.isfinite(zero_graph)),
            f"invalid source graph response for {candidate['label']}",
        )
        dense_driven = np.einsum(
            "mn,tnc->tmc", readout_weights, driven_graph, optimize=True
        )
        dense_zero = np.einsum(
            "mn,tnc->tmc", readout_weights, zero_graph, optimize=True
        )
        graph_response = driven_graph - zero_graph
        dense_response = dense_driven - dense_zero
        prediction = persistence + dense_response
        metrics = score_reusable_sota_trajectory(
            target,
            prediction,
            horizon_ranges_half_open=ranges,
        )
        absolute_driven_metrics = score_reusable_sota_trajectory(
            target,
            dense_driven,
            horizon_ranges_half_open=ranges,
        )
        absolute_zero_metrics = score_reusable_sota_trajectory(
            target,
            dense_zero,
            horizon_ranges_half_open=ranges,
        )
        prediction_path = candidate_root / "prediction.npz"
        np.savez_compressed(
            prediction_path,
            prediction_m=prediction.astype(np.float32),
            persistence_m=persistence.astype(np.float32),
            absolute_driven_prediction_m=dense_driven.astype(np.float32),
            absolute_zero_prediction_m=dense_zero.astype(np.float32),
            graph_response_m=graph_response.astype(np.float32),
            candidate_label=np.asarray(candidate["label"]),
        )
        records.append(
            {
                **candidate,
                "valid": True,
                "failure": None,
                "driven_result_sha256": driven["result_sha256"],
                "driven_trajectory_sha256": driven["trajectory_sha256"],
                "zero_result_sha256": zero["result_sha256"],
                "zero_trajectory_sha256": zero["trajectory_sha256"],
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": _sha256_file(prediction_path),
                "metrics": metrics,
                "absolute_driven_metrics": absolute_driven_metrics,
                "absolute_zero_metrics": absolute_zero_metrics,
                "maximum_dense_response_m": float(
                    np.max(np.linalg.norm(dense_response, axis=-1))
                ),
            }
        )

    valid_candidate_count = sum(record["valid"] for record in records)
    invalid_candidate_labels = [
        record["label"] for record in records if not record["valid"]
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableSotaSourceCandidateBank",
        "protocol_id": method["config"]["protocol_id"],
        "method_config_sha256": method["config_sha256"],
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "processing_authorization": processing_authorization,
        "window_authorization": window_authorization,
        "candidate_count": len(records),
        "valid_candidate_count": valid_candidate_count,
        "invalid_candidate_count": len(invalid_candidate_labels),
        "invalid_candidate_labels": invalid_candidate_labels,
        "candidate_order": [record["label"] for record in records],
        "persistence_metrics": persistence_metrics,
        "records": records,
        "input_sha256": {
            "development_observations": _sha256_file(
                args.development_observations
            ),
            "source_final_data": _sha256_file(args.source_final_data),
            "simulator_final_data": _sha256_file(args.simulator_final_data),
            "episode_graph": _sha256_file(args.episode_graph),
            "state_artifact": _sha256_file(args.state_artifact),
            "twin_summary": _sha256_file(args.twin_summary),
        },
        "information_boundary": {
            "source_future_outcome_used_for_candidate_scoring": True,
            "source_future_outcome_used_for_twin_initialization": False,
            "held_development_outcome_read": False,
            "confirmatory_object_read": False,
        },
        "report_only_controls": {
            "absolute_driven_phystwin": True,
            "absolute_zero_action_phystwin": True,
            "used_for_candidate_selection": False,
        },
        "passed": len(records) == 18 and valid_candidate_count >= 1,
        "claim_boundary": (
            "source-only candidate scoring under the independent metric contract; "
            "no held or direct Deform360 Table 4 claim"
        ),
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    manifest_path = args.output_root / "source_candidate_bank.json"
    _require(not manifest_path.exists(), f"source bank exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "object_id": args.object_id,
                "episode_id": args.episode_id,
                "candidate_count": len(records),
                "result_sha256": manifest["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
