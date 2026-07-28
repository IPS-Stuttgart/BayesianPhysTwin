#!/usr/bin/env python3
"""Build one frozen causal AllTracker measurement for a fresh locked case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_fresh_pairwise_protocol import (
    file_sha256,
    load_bound_cohort,
    load_fresh_pairwise_protocol,
    validate_backbone_seal,
)
from bayesian_phystwin.deform360_fresh_camera_observation import (
    build_fresh_raw_camera_measurement_case_with_contract,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--backbone-case-dir", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alltracker-source", type=Path, required=True)
    parser.add_argument("--alltracker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    protocol = load_fresh_pairwise_protocol(
        args.protocol,
        repository_root=repo,
    )
    cohort = load_bound_cohort(args.cohort_lock, protocol)
    expected_cases = tuple(str(case["case"]) for case in cohort["cases"])
    protocol_sha256 = file_sha256(args.protocol)
    cohort_sha256 = str(cohort["cohort_lock_sha256"])

    def validator(seal):
        validate_backbone_seal(
            seal,
            protocol_config_sha256=protocol_sha256,
            cohort_lock_sha256=cohort_sha256,
        )

    config = RawCameraObservationConfig()
    processed = args.processed_episode_dir.resolve()
    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source,
        args.alltracker_checkpoint,
        device=args.device,
        config=config,
    )
    try:
        manifest = build_fresh_raw_camera_measurement_case_with_contract(
            args.backbone_case_dir,
            processed,
            args.output_dir,
            runtime,
            protocol_id=protocol["protocol_id"],
            expected_case_names=expected_cases,
            prediction_seal_validator=validator,
            claim_boundary=(
                "frozen fresh-object causal RGB-prefix measurement; no target or "
                "outcome artifact is available to this process"
            ),
            minimum_eligible_camera_count=protocol["observation"][
                "minimum_eligible_camera_count"
            ],
            config=config,
        )
    finally:
        runtime.close()
    print(
        json.dumps(
            {
                "case": manifest["case"],
                "measurement_result_sha256": manifest["result_sha256"],
                "accepted_measurement_count_by_update": manifest["output"][
                    "accepted_measurement_count_by_update"
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
