#!/usr/bin/env python3
"""Build one target-free AllTracker prefix measurement for the sealed cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_raw_camera_observation import (  # noqa: E402
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    build_raw_camera_measurement_case_with_contract,
)
from bayesian_phystwin.deform360_tactile_guard_outcome_sealed import (  # noqa: E402
    CLAIM_LABEL,
    PROTOCOL_ID,
    load_outcome_sealed_protocol,
    validate_backbone_seal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--backbone-case-dir", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--alltracker-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = load_outcome_sealed_protocol(
        args.protocol,
        repository_root=args.repo,
    )
    config = RawCameraObservationConfig(**protocol["method"]["camera_observation"])
    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source,
        args.checkpoint,
        device=args.device,
        config=config,
    )

    def validate(seal: dict[str, object]) -> None:
        validate_backbone_seal(seal, protocol=protocol)

    try:
        result = build_raw_camera_measurement_case_with_contract(
            args.backbone_case_dir,
            args.processed_episode_dir,
            args.output_dir,
            runtime,
            protocol_id=PROTOCOL_ID,
            expected_case_names=tuple(
                str(row["case"]) for row in protocol["cohort"]["cases"]
            ),
            prediction_seal_validator=validate,
            claim_boundary=(
                f"{CLAIM_LABEL}; causal RGB prefix only; no future outcome read"
            ),
            config=config,
        )
    finally:
        runtime.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
