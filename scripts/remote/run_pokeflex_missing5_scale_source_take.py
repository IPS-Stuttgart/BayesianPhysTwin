#!/usr/bin/env python3
"""Run the frozen six-scale bank on one missing-five PokeFlex source take."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "remote"))

from bayesian_phystwin.pokeflex_action_robust_all18 import (  # noqa: E402
    SOURCE_FIELD,
)
from bayesian_phystwin.pokeflex_missing5_scale import (  # noqa: E402
    BASE_EFFECTIVE_SCALE,
    CANDIDATE_MULTIPLIERS,
    file_sha256,
    validate_source_protocol,
)


def _manifest_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _valid_pose(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return array.shape == (4, 4) and bool(np.all(np.isfinite(array)))


def _apply_missing_twe_fallback(
    result: dict[str, object],
    *,
    missing_twe_frames: set[int],
    summary,
) -> dict[str, object]:
    """Discard every correction whose action history lacks measured T_WE."""

    prefix = f"checkpoint_{SOURCE_FIELD}_residual_scale_"
    fallback_targets = set()
    for target in result["targets"]:
        target_frame = int(target["target_frame"])
        source_frame = target_frame - 1
        history = range(max(1, source_frame - 3), source_frame + 1)
        if not missing_twe_frames.intersection(history):
            continue
        fallback_targets.add(target_frame)
        checkpoint = float(target["released_checkpoint_CD_UL1_mm"])
        for key in tuple(target):
            if key.startswith(prefix):
                target[key] = checkpoint

    for update in result["updates"]:
        if int(update["target_frame"]) in fallback_targets:
            update["observation_update_accepted_before_action_fallback"] = bool(
                update["accepted"]
            )
            update["accepted"] = False
            update["action_supported"] = False
            update["reason"] = "missing-required-T_WE-exact-checkpoint-fallback"

    candidate_keys = tuple(
        key for key in result["aggregates"] if key.startswith(prefix)
    )
    for key in candidate_keys:
        values = [float(target[key]) for target in result["targets"]]
        result["aggregates"][key] = summary(values)
    result["best_development_candidate"] = min(
        candidate_keys,
        key=lambda key: result["aggregates"][key]["mean_CD_UL1_mm"],
    )
    result["missing_required_T_WE"] = {
        "source_frame_count": len(missing_twe_frames),
        "source_frames": sorted(missing_twe_frames),
        "fallback_target_count": len(fallback_targets),
        "fallback_target_frames": sorted(fallback_targets),
        "sentinel_role": (
            "nonphysical in-memory execution placeholder; every affected action "
            "candidate is discarded and replaced by the exact checkpoint score"
        ),
        "source_robot_bytes_modified": False,
        "pose_imputation_used_by_prediction": False,
    }
    return result


def _run_smoke(
    runner_module,
    *,
    take_root: Path,
    registration_protocol: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
):
    """Authorize one source object without changing the frozen legacy runner."""

    original_loader = runner_module.load_pokeflex_registration_protocol
    original_json_loads = runner_module.json.loads
    raw_robot = original_json_loads(
        (take_root / "robot_data.json").read_text(encoding="utf-8")
    )
    missing_twe_frames = {
        int(record["frame"])
        for record in raw_robot
        if not _valid_pose(record.get("T_WE"))
    }

    def load_with_source_authorization(path: Path):
        registration = deepcopy(original_loader(path))
        development = registration["payload"]["cohort"]["development_objects"]
        registration["payload"]["cohort"]["development_objects"] = sorted(
            set(development) | {take_root.name.rpartition("_T")[0]}
        )
        return registration

    def load_with_discarded_twe_sentinels(value: str, *args, **kwargs):
        decoded = original_json_loads(value, *args, **kwargs)
        if (
            missing_twe_frames
            and isinstance(decoded, list)
            and decoded
            and all(isinstance(record, dict) for record in decoded)
            and all("frame" in record and "forces" in record for record in decoded)
        ):
            decoded = deepcopy(decoded)
            for record in decoded:
                if int(record["frame"]) in missing_twe_frames:
                    record["T_WE"] = np.eye(4, dtype=np.float64).tolist()
        return decoded

    runner_module.load_pokeflex_registration_protocol = load_with_source_authorization
    runner_module.json.loads = load_with_discarded_twe_sentinels
    try:
        scales = (0.0,) + tuple(
            BASE_EFFECTIVE_SCALE * value for value in CANDIDATE_MULTIPLIERS
        )
        result = runner_module.run_smoke(
            take_root,
            registration_protocol,
            upstream_checkout,
            checkpoint_root,
            correction_scales=scales,
            correction_fields=(SOURCE_FIELD,),
            residual_geometry="point_to_point",
            maximum_frame=None,
            include_frozen_action_guard=False,
            record_online_observation_regret=False,
        )
        return _apply_missing_twe_fallback(
            result,
            missing_twe_frames=missing_twe_frames,
            summary=runner_module._summary,
        )
    finally:
        runner_module.load_pokeflex_registration_protocol = original_loader
        runner_module.json.loads = original_json_loads


def _validate_projection(
    manifest: dict[str, object],
    *,
    take_id: str,
    projected_archive: Path,
    protocol: dict[str, object],
) -> None:
    expected = protocol["archive_inventory"]["takes"][take_id]
    expected_runner_hash = protocol["implementation"][
        "source_projection_runner_file_sha256"
    ]
    if manifest.get("artifact_kind") != "PokeFlexMissingFiveScaleSourceProjection":
        raise ValueError("projection manifest kind changed")
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("projection manifest digest changed")
    if manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("projection protocol binding changed")
    if manifest.get("projection_runner_file_sha256") != expected_runner_hash:
        raise ValueError("projection runner binding changed")
    if manifest.get("take_id") != take_id:
        raise ValueError("projection take changed")
    if manifest.get("source_archive_sha256") != expected["sha256"]:
        raise ValueError("projection source archive changed")
    if int(manifest.get("source_archive_bytes", -1)) != int(expected["bytes"]):
        raise ValueError("projection source archive size changed")
    if manifest.get("target_geometry_decoded") is not False:
        raise ValueError("projection decoded target geometry")
    if manifest.get("outcome_metric_computed") is not False:
        raise ValueError("projection computed an outcome metric")
    if manifest.get("held_v8_accessed") is not False:
        raise ValueError("projection crossed the held-v8 boundary")
    if projected_archive.stat().st_size != int(manifest["projected_archive_bytes"]):
        raise ValueError("projected archive size changed")
    if file_sha256(projected_archive) != manifest["projected_archive_sha256"]:
        raise ValueError("projected archive bytes changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--projected-archive", type=Path, required=True)
    parser.add_argument(
        "--registration-protocol",
        type=Path,
        default=(ROOT / "configs" / "sota" / "pokeflex_bayesian_registration_v1.json"),
    )
    args = parser.parse_args()

    import run_pokeflex_checkpoint_registration_smoke as runner_module

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validation = validate_source_protocol(protocol)
    implementation = protocol["implementation"]
    if file_sha256(Path(__file__)) != implementation["source_runner_file_sha256"]:
        raise ValueError("source runner bytes changed")
    legacy_path = ROOT / implementation["legacy_runner"]
    if file_sha256(legacy_path) != implementation["legacy_runner_file_sha256"]:
        raise ValueError("legacy runner bytes changed")
    registration_protocol = args.registration_protocol.resolve()
    if (
        file_sha256(registration_protocol)
        != implementation["registration_protocol_file_sha256"]
    ):
        raise ValueError("registration protocol bytes changed")

    take_root = args.take_root.resolve()
    take_id = take_root.name
    if take_id not in validation["source_take_ids"]:
        raise ValueError("take is outside the frozen source cohort")
    projected_archive = args.projected_archive.resolve()
    projection_manifest = json.loads(
        args.projection_manifest.read_text(encoding="utf-8")
    )
    _validate_projection(
        projection_manifest,
        take_id=take_id,
        projected_archive=projected_archive,
        protocol=protocol,
    )

    result = _run_smoke(
        runner_module,
        take_root=take_root,
        registration_protocol=registration_protocol,
        upstream_checkout=args.upstream_checkout.resolve(),
        checkpoint_root=args.checkpoint_root.resolve(),
    )
    result["missing5_source_protocol_sha256"] = protocol["protocol_sha256"]
    result["source_runner_file_sha256"] = implementation["source_runner_file_sha256"]
    result["legacy_runner_file_sha256"] = implementation["legacy_runner_file_sha256"]
    result["projection_manifest_sha256"] = projection_manifest["manifest_sha256"]
    result["source_archive_sha256"] = projection_manifest["source_archive_sha256"]
    result["source_prediction_role"] = (
        "previously opened public non-target action; scale calibration only"
    )
    result["official_target_outcome_used"] = False
    result["held_v8_accessed"] = False
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing source artifact differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "take_id": take_id,
                "protocol_sha256": protocol["protocol_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
