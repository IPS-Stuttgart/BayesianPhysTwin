#!/usr/bin/env python3
"""Generate delayed Kinect plus D405 evidence on opened PokeFlex takes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_independent_depth_protocol import (  # noqa: E402
    load_pokeflex_independent_depth_protocol,
)
import run_pokeflex_checkpoint_registration_independent_depth as candidate_runner  # noqa: E402


FROZEN_CANDIDATE_RUNNER_SHA256 = (
    "7927deb862dac8783b5415197ff65854ec3c0235a01db88689997c9b97f22e25"
)
OPENED_TAKES = tuple(
    f"{object_name}_T{take}"
    for object_name in (
        "3dPrintedHeart",
        "FoamDice",
        "MemoryFoam",
        "PlushOctopus",
        "ToiletPaperRoll",
    )
    for take in (1, 3, 4, 5, 6)
) + (
    "3dPrintedPyramid_T2",
    "Beanbag_T2",
    "FoamCylinder_T2",
    "PlushMoon_T2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--independent-depth-protocol",
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
    take_root = args.take_root.resolve()
    if take_root.name not in OPENED_TAKES:
        raise ValueError("take is outside the explicitly opened development cohort")
    candidate_path = (
        _repository_root()
        / "scripts"
        / "remote"
        / "run_pokeflex_checkpoint_registration_independent_depth.py"
    )
    if _sha256(candidate_path) != FROZEN_CANDIDATE_RUNNER_SHA256:
        raise ValueError("frozen candidate runner checksum changed")

    registration_path = args.registration_protocol.resolve()
    canonical_registration = candidate_runner.load_pokeflex_registration_protocol(
        registration_path
    )
    authorized_registration = copy.deepcopy(canonical_registration)
    authorized_registration["payload"]["cohort"]["development_objects"] = sorted(
        {take.rsplit("_T", 1)[0] for take in OPENED_TAKES}
    )
    original_loader = candidate_runner.load_pokeflex_registration_protocol

    def opened_development_adapter(path: str | Path) -> dict[str, object]:
        if Path(path).resolve() != registration_path:
            raise ValueError("registration protocol path changed")
        return copy.deepcopy(authorized_registration)

    independent_path = args.independent_depth_protocol.resolve()
    independent = load_pokeflex_independent_depth_protocol(independent_path)
    method = independent["payload"]["method_lock"]
    candidate_runner.load_pokeflex_registration_protocol = opened_development_adapter
    try:
        result = candidate_runner.run_smoke(
            take_root,
            registration_path,
            args.upstream_checkout.resolve(),
            args.checkpoint_root.resolve(),
            correction_scales=tuple(map(float, method["correction_scales"])),
            correction_fields=tuple(map(str, method["correction_fields"])),
            residual_geometry="point_to_point",
            maximum_frame=None,
            include_frozen_action_guard=False,
            record_online_observation_regret=True,
            record_independent_anchor_regret=True,
            independent_depth_protocol_path=independent_path,
            independent_anchor_maximum_template_distance_m=(
                float(method["static_template_support_radius_mm"]) / 1000.0
            ),
        )
    finally:
        candidate_runner.load_pokeflex_registration_protocol = original_loader
    result["dual_sensor_regret_development"] = {
        "claim_status": "post-open source/calibration method development",
        "candidate_runner_sha256": FROZEN_CANDIDATE_RUNNER_SHA256,
        "d405_evidence": "frame f-1 only",
        "kinect_evidence": "frame f-1 only",
        "prediction_target": "frame f",
        "future_observation_used": False,
        "target_objects_opened": False,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing development artifact differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "take_id": take_root.name,
                "target_count": len(result["targets"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
