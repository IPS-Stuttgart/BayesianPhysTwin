#!/usr/bin/env python3
"""Generate causal-transport banks on the exhausted Deform360 source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_causal_transport import (
    CausalContactTransportConfig,
    causal_contact_transport_prediction,
)
from causal4d_public.deform360_causal_transport_method import (
    causal_transport_candidates,
    load_causal_transport_method,
)
from causal4d_public.deform360_reusable_sota_selection import (
    score_reusable_sota_trajectory,
)
from causal4d_public.deform360_reusable_trust import (
    build_deform360_trust_features,
)


FRAME_COUNT = 76
HORIZON_RANGES = {
    "full": [1, 76],
    "early": [1, 26],
    "middle": [26, 51],
    "late": [51, 76],
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-diagnosis", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    _require(isinstance(payload, Mapping), f"pickle is not a mapping: {path}")
    return payload


def _load_robot(path: Path, *, group_count: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        actions = np.asarray(stored["actions"], dtype=np.float64)
        openings = np.asarray(stored["openings"], dtype=np.float64)
    if actions.ndim == 3:
        actions = actions[:, None]
    if openings.ndim == 1:
        openings = openings[:, None]
    _require(
        actions.ndim == 4
        and actions.shape[0] >= FRAME_COUNT
        and actions.shape[1:] == (group_count, 5, 3)
        and openings.shape == actions.shape[:2]
        and np.all(np.isfinite(actions))
        and np.all(np.isfinite(openings)),
        f"robot state is incompatible: {path}",
    )
    return actions[:FRAME_COUNT].copy(), openings[:FRAME_COUNT].copy()


def _episode_paths(
    *,
    result_root: Path,
    stage_root: Path,
    object_id: str,
    episode_id: int,
) -> tuple[Path, Path]:
    key = f"{object_id}-ep{episode_id:04d}"
    return (
        result_root / key / "target_data.pkl",
        stage_root / key / "episode_0000" / "robot" / "robot.npz",
    )


def main() -> int:
    args = _parse_args()
    diagnosis = json.loads(args.failure_diagnosis.read_text(encoding="utf-8"))
    _require(
        diagnosis.get("artifact_kind") == "Deform360IndependentSourceFailureDiagnosis"
        and isinstance(diagnosis.get("episodes"), list),
        "failure diagnosis is incompatible",
    )
    method = load_causal_transport_method(args.method)
    candidates = causal_transport_candidates(method)
    contact = method["config"]["contact_policy"]
    group_size = int(contact["controller_group_size"])
    outputs = []
    for source_record in diagnosis["episodes"]:
        object_id = str(source_record["object_id"])
        episode_id = int(source_record["episode_id"])
        target_path, robot_path = _episode_paths(
            result_root=args.result_root,
            stage_root=args.stage_root,
            object_id=object_id,
            episode_id=episode_id,
        )
        source = _load_pickle(target_path)
        target = np.asarray(source["object_points"], dtype=np.float64)
        controllers = np.asarray(source["controller_points"], dtype=np.float64)
        _require(
            target.ndim == controllers.ndim == 3
            and target.shape[0] == controllers.shape[0] == FRAME_COUNT
            and target.shape[2] == controllers.shape[2] == 3
            and controllers.shape[1] % group_size == 0
            and np.all(np.isfinite(target))
            and np.all(np.isfinite(controllers)),
            f"source trajectory is incompatible: {object_id} episode {episode_id}",
        )
        group_count = controllers.shape[1] // group_size
        actions, openings = _load_robot(robot_path, group_count=group_count)
        persistence = np.repeat(target[:1], FRAME_COUNT, axis=0)
        persistence_metrics = score_reusable_sota_trajectory(
            target,
            persistence,
            horizon_ranges_half_open=HORIZON_RANGES,
        )
        records = []
        for candidate in candidates:
            config = CausalContactTransportConfig(
                controller_group_size=group_size,
                maximum_contact_distance_m=float(contact["maximum_contact_distance_m"]),
                opening_contact_threshold_m=float(
                    contact["opening_contact_threshold_m"]
                ),
                confirmation_frames=int(contact["confirmation_frames"]),
                base_support_scale_m=float(candidate["base_support_scale_m"]),
                support_growth_per_travel=float(candidate["support_growth_per_travel"]),
                initial_contact_gain=float(candidate["initial_contact_gain"]),
                acquired_contact_gain=float(candidate["acquired_contact_gain"]),
                transform_mode=str(candidate["transform_mode"]),
            )
            result = causal_contact_transport_prediction(
                target[0], controllers, openings, config=config
            )
            metrics = score_reusable_sota_trajectory(
                target,
                result.prediction_m,
                horizon_ranges_half_open=HORIZON_RANGES,
            )
            response = result.prediction_m - persistence
            records.append(
                {
                    **candidate,
                    "valid": True,
                    "metrics": metrics,
                    "diagnostics": result.diagnostics(),
                    "trust_features": build_deform360_trust_features(
                        actions, openings, response, persistence
                    ),
                    "prediction_array_sha256": _array_sha256(result.prediction_m),
                }
            )
        _require(
            len(records) == len(candidates) == 49
            and records[0]["label"] == "persistence"
            and records[0]["diagnostics"]["exact_persistence"] is True
            and records[0]["metrics"] == persistence_metrics,
            "persistence control is not exact",
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalTransportExhaustedSourceBank",
            "protocol_id": "deform360-causal-expert-router-exhausted-source-v1",
            "method_config_sha256": method["config_sha256"],
            "object_id": object_id,
            "episode_id": episode_id,
            "episode_key": str(source_record["episode_key"]),
            "candidate_count": len(records),
            "candidate_order": [record["label"] for record in records],
            "persistence_metrics": persistence_metrics,
            "records": records,
            "input_sha256": {
                "failure_diagnosis": _sha256_file(args.failure_diagnosis),
                "target_data": _sha256_file(target_path),
                "robot_state": _sha256_file(robot_path),
                "initial_object_points": _array_sha256(target[0]),
                "controller_points": _array_sha256(controllers),
                "aligned_actions": _array_sha256(actions),
                "aligned_openings": _array_sha256(openings),
            },
            "information_boundary": {
                "source_panel_previously_exhausted": True,
                "source_future_outcome_used_for_candidate_scoring": True,
                "source_future_outcome_used_for_prediction": False,
                "prediction_inputs": [
                    "frame-zero object geometry",
                    "known robot trajectory",
                    "known gripper openness",
                ],
                "tactile_used": False,
                "symbolic_action_label_used": False,
                "fresh_or_confirmatory_data_read": False,
                "pokeflex_target_read": False,
            },
            "passed": True,
            "claim_boundary": (
                "post-failure source discovery only; requires object-held-out "
                "cross-fitting and a fresh prospective transfer gate"
            ),
        }
        payload["result_sha256"] = _canonical_sha256(payload)
        output = args.output_root / f"{object_id}-ep{episode_id:04d}.json"
        _require(not output.exists(), f"source bank exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        outputs.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "path": str(output.resolve()),
                "file_sha256": _sha256_file(output),
                "result_sha256": payload["result_sha256"],
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalTransportExhaustedSourceManifest",
        "episode_count": len(outputs),
        "object_count": len({item["object_id"] for item in outputs}),
        "method_config_sha256": method["config_sha256"],
        "failure_diagnosis_sha256": _sha256_file(args.failure_diagnosis),
        "outputs": outputs,
        "information_boundary": {
            "source_panel_previously_exhausted": True,
            "fresh_or_confirmatory_data_read": False,
            "pokeflex_target_read": False,
        },
        "passed": True,
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    manifest_path = args.output_root / "manifest.json"
    _require(not manifest_path.exists(), f"source manifest exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "episode_count": len(outputs),
                "object_count": manifest["object_count"],
                "result_sha256": manifest["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
