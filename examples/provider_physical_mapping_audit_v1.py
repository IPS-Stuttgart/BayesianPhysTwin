"""Run a deterministic, target-blind provider-to-physical mapping audit."""

from __future__ import annotations

import json

import numpy as np

from bayesian_phystwin.provider_physical_mapping_audit_v1 import (
    ProviderPhysicalMappingCaseV1,
    ProviderPhysicalMappingPolicyV1,
    audit_provider_physical_mapping,
)


def main() -> int:
    """Create and print one self-contained admissible mapping certificate."""

    case = ProviderPhysicalMappingCaseV1(
        case_id="synthetic-source-mapping-v1",
        provider_artifact_id="0" * 64,
        physical_query_id="1" * 64,
        mapping_protocol_id="2" * 64,
        provider_frame="camera_native",
        physical_frame="robot_world",
        points_native=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [900.0, 900.0, 900.0],
            ]
        ),
        valid_mask=np.array([True, True, True, False]),
        provider_unit_scale_m=0.01,
        provider_to_physical=np.eye(4),
        query_bounds_m=np.array(
            [
                [-0.001, -0.001, -0.001],
                [0.021, 0.001, 0.001],
            ]
        ),
        timestamps_s=np.array([0.0, 0.1, 0.2, np.nan]),
        query_time_window_s=np.array([0.0, 0.2]),
        covariances_native=np.repeat(np.eye(3)[None], 4, axis=0),
        metadata={"information_split": "synthetic-source-only"},
    )
    policy = ProviderPhysicalMappingPolicyV1(
        minimum_valid_point_count=3,
        minimum_valid_fraction=0.75,
        minimum_mapped_point_count=3,
        minimum_mapped_fraction=1.0,
        require_timestamps=True,
        require_covariance=True,
    )
    audit = audit_provider_physical_mapping(case, policy)
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0 if audit.mapping_admissible else 1


if __name__ == "__main__":
    raise SystemExit(main())
