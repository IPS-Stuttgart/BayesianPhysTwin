#!/usr/bin/env python3
"""Extract opened PokeFlex action/discrepancy rows without touching targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))
sys.path.insert(0, str(_repository_root() / "scripts" / "remote"))

from run_pokeflex_bayesian_registration_smoke import (  # noqa: E402
    _cd_ul1_mm,
    _load_mesh,
    _surface_sample,
    _template_frame,
    _view_points,
)
from run_pokeflex_checkpoint_registration_smoke import (  # noqa: E402
    _load_official_template,
)

from bayesian_phystwin.pokeflex_action_discrepancy import (  # noqa: E402
    causal_action_features,
    robust_nearest_translation_m,
)
from bayesian_phystwin.pokeflex_registration_protocol import (  # noqa: E402
    load_pokeflex_registration_protocol,
)
from bayesian_phystwin.pokeflex_released_checkpoint import (  # noqa: E402
    PokeFlexReleasedCheckpoint,
)

ARTIFACT_KIND = "PokeFlexActionDiscrepancyOpenedTakeV1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _checkpoint_hashes(root: Path) -> dict[str, str]:
    return {
        name: _sha256(root / name)
        for name in (
            "pointcloud_encoder.pth",
            "attention_model.pth",
            "decoder.pth",
        )
    }


def _run(arguments: argparse.Namespace) -> int:
    _require(
        arguments.acknowledge_opened_outcome,
        "extraction requires --acknowledge-opened-outcome",
    )
    take_root = arguments.take_root.resolve()
    protocol_path = arguments.protocol.resolve()
    upstream = arguments.upstream_checkout.resolve()
    checkpoint_root = arguments.checkpoint_root.resolve()
    output_json = arguments.output_json.resolve()
    output_npz = arguments.output_npz.resolve()
    _require(take_root.is_dir(), f"take root does not exist: {take_root}")
    _require(not output_npz.exists(), f"refusing to overwrite {output_npz}")

    protocol = load_pokeflex_registration_protocol(protocol_path)
    payload = protocol["payload"]
    object_name, separator, take_number = take_root.name.rpartition("_T")
    _require(separator and take_number.isdigit(), "take directory identity is invalid")
    development = set(payload["cohort"]["development_objects"])
    calibration = set(payload["cohort"]["calibration_objects"])
    target = set(payload["cohort"]["target_objects"])
    _require(object_name not in target, "sealed target object supplied to extractor")
    _require(
        object_name in development or object_name in calibration,
        "take is outside the opened development/calibration cohorts",
    )
    role = "development" if object_name in development else "opened_calibration"
    if role == "opened_calibration":
        _require(
            arguments.acknowledge_opened_calibration,
            "opened calibration extraction requires explicit acknowledgement",
        )

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
    template_vertices, template_faces, preprocessing = _load_official_template(
        template_path
    )
    frame_limit = arguments.maximum_frame or max(robot_by_frame)
    valid_targets = sorted(frame for frame in active if 6 <= frame <= frame_limit)
    _require(valid_targets, "take has no causal target frames")

    checkpoint = PokeFlexReleasedCheckpoint.load(
        template_vertices,
        upstream_checkout=upstream,
        checkpoint_root=checkpoint_root,
        device=arguments.device,
    )
    features_by_frame: dict[int, object] = {}
    preprocessing_by_frame: dict[int, object] = {}
    for frame in range(1, frame_limit):
        views = tuple(
            _view_points(take_root, frame, camera, template_vertices)
            for camera in (0, 1)
        )
        feature, frame_preprocessing = checkpoint.encode_frame(views)
        features_by_frame[frame] = feature
        preprocessing_by_frame[frame] = frame_preprocessing

    sample_count = int(payload["evaluation"]["sampling"]["surface_points"])
    base_seed = int(payload["evaluation"]["sampling"]["seed"])
    frame_ids: list[int] = []
    causal_features: list[np.ndarray] = []
    oracle_translation: list[np.ndarray] = []
    baseline_samples: list[np.ndarray] = []
    target_samples: list[np.ndarray] = []
    baseline_errors: list[float] = []
    oracle_errors: list[float] = []

    for target_frame in valid_targets:
        history_frames = tuple(range(target_frame - 5, target_frame))
        prediction = checkpoint.predict_from_encoded_history(
            [features_by_frame[frame] for frame in history_frames],
            [preprocessing_by_frame[frame] for frame in history_frames],
        ).vertices_m
        feature = causal_action_features(
            [robot_by_frame[frame] for frame in history_frames],
            template_vertices_m=template_vertices,
            predicted_vertices_m=prediction,
        )
        baseline_sample = _surface_sample(
            prediction,
            template_faces,
            sample_count,
            base_seed + target_frame,
        )

        # Outcome geometry is loaded only after the candidate inputs are frozen.
        target_mesh = _load_mesh(take_root / "meshes" / f"mesh-f{target_frame:05d}.obj")
        target_sample = _surface_sample(
            np.asarray(target_mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(target_mesh.faces, dtype=np.int64),
            sample_count,
            base_seed + target_frame,
        )
        translation = robust_nearest_translation_m(
            baseline_sample,
            target_sample,
            retained_fraction=arguments.retained_fraction,
        )
        frame_ids.append(target_frame)
        causal_features.append(feature)
        oracle_translation.append(translation)
        baseline_samples.append(np.asarray(baseline_sample, dtype=np.float32))
        target_samples.append(np.asarray(target_sample, dtype=np.float32))
        baseline_errors.append(_cd_ul1_mm(baseline_sample, target_sample))
        oracle_errors.append(
            _cd_ul1_mm(baseline_sample + translation[None], target_sample)
        )

    arrays = {
        "target_frame": np.asarray(frame_ids, dtype=np.int64),
        "causal_features": np.asarray(causal_features, dtype=np.float64),
        "oracle_translation_m": np.asarray(oracle_translation, dtype=np.float64),
        "baseline_samples_m": np.asarray(baseline_samples, dtype=np.float32),
        "target_samples_m": np.asarray(target_samples, dtype=np.float32),
        "baseline_cd_ul1_mm": np.asarray(baseline_errors, dtype=np.float64),
        "oracle_translation_cd_ul1_mm": np.asarray(
            oracle_errors,
            dtype=np.float64,
        ),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    checkpoint_hashes = _checkpoint_hashes(checkpoint_root)
    expected_hashes = {
        name: value["sha256"]
        for name, value in payload["upstream"]["released_kinect_checkpoint"].items()
    }
    _require(
        checkpoint_hashes == expected_hashes,
        "released checkpoint hashes changed",
    )
    result = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "claim_boundary": (
            "Post-open development/calibration outcome extraction. Sealed target "
            "objects are rejected and no target claim is authorized."
        ),
        "take": {
            "id": take_root.name,
            "object": object_name,
            "take": f"T{take_number}",
            "role": role,
            "target_frame_count": len(frame_ids),
            "template_frame": template_frame,
            "frame_limit": frame_limit,
        },
        "causal_boundary": {
            "history": "target f uses only frames f-5 through f-1",
            "feature_is_residual_independent": True,
            "target_geometry_loaded_after_feature_construction": True,
            "sealed_target_object_accessed": False,
        },
        "configuration": {
            "retained_nearest_neighbor_fraction": arguments.retained_fraction,
            "surface_sample_count": sample_count,
            "surface_sample_seed": base_seed,
            "device": checkpoint.device,
        },
        "input_sha256": {
            "protocol": protocol["protocol_sha256"],
            "robot_data": _sha256(robot_path),
            "template": _sha256(template_path),
            "checkpoint": checkpoint_hashes,
        },
        "template_preprocessing": preprocessing,
        "arrays": {
            "path": str(output_npz),
            "sha256": _sha256(output_npz),
            "keys": sorted(arrays),
        },
        "diagnostic": {
            "baseline_mean_cd_ul1_mm": float(np.mean(baseline_errors)),
            "per_frame_oracle_translation_mean_cd_ul1_mm": float(
                np.mean(oracle_errors)
            ),
            "per_frame_oracle_relative_improvement": float(
                (np.mean(baseline_errors) - np.mean(oracle_errors))
                / max(np.mean(baseline_errors), 1e-15)
            ),
            "median_oracle_translation_norm_mm": float(
                1000.0
                * np.median(np.linalg.norm(arrays["oracle_translation_m"], axis=1))
            ),
        },
    }
    _write_json_once(output_json, result)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-frame", type=int)
    parser.add_argument("--retained-fraction", type=float, default=0.9)
    parser.add_argument("--acknowledge-opened-outcome", action="store_true")
    parser.add_argument("--acknowledge-opened-calibration", action="store_true")
    return parser


def main() -> int:
    return _run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
