"""Source-only residual-history contract for registered Deform360 studies.

The public experimental module combines a deterministic whole-recorder camera
partition, a no-fill ``(T, N, 3)`` residual-history adapter with explicit
``(T, N)`` validity, and a covariance-only candidate that returns exact physical
fallback objects whenever support or covariance admission fails. It contains no
target roster and grants no target-access or claim authorization.
"""

from ._deform360_covariance_residual_history_adapter_v1 import (
    ResidualHistoryAdapterV1,
    build_residual_history_adapter,
)
from ._deform360_covariance_residual_history_common_v1 import (
    CAMERA_PARTITION_NAMESPACE,
    CLAIM_BOUNDARY,
    FALLBACK_SEMANTICS,
    HORIZON_LABELS,
    REFERENCE_MEAN_SEMANTICS,
    RESIDUAL_STORAGE_SEMANTICS,
    TARGET_QUARANTINE_ROOT,
    DisjointCameraPartitionV1,
    ResidualHistoryDryRunPolicyV1,
    assert_outside_target_quarantine,
    camera_hardware_family,
    deterministic_disjoint_camera_partition,
)
from ._deform360_covariance_residual_history_decision_v1 import (
    ResidualHistoryDryRunDecisionV1,
    ResidualHistoryDryRunResultV1,
)
from ._deform360_covariance_residual_history_last_valid_v1 import (
    run_source_only_residual_history_dry_run,
)

__all__ = [
    "CAMERA_PARTITION_NAMESPACE",
    "CLAIM_BOUNDARY",
    "DisjointCameraPartitionV1",
    "FALLBACK_SEMANTICS",
    "HORIZON_LABELS",
    "REFERENCE_MEAN_SEMANTICS",
    "RESIDUAL_STORAGE_SEMANTICS",
    "ResidualHistoryAdapterV1",
    "ResidualHistoryDryRunDecisionV1",
    "ResidualHistoryDryRunPolicyV1",
    "ResidualHistoryDryRunResultV1",
    "TARGET_QUARANTINE_ROOT",
    "assert_outside_target_quarantine",
    "build_residual_history_adapter",
    "camera_hardware_family",
    "deterministic_disjoint_camera_partition",
    "run_source_only_residual_history_dry_run",
]
