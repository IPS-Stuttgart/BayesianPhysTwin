"""Source-only residual-history contracts for registered Deform360 studies.

The public module binds explicit camera-recorder provenance, disjoint
reconstruction manifests, a no-fill ``(T, N, 3)`` residual history with exact
``(T, N)`` validity, the caller-owned registered ``last_residual`` mean, and
exact physical fallback. The registered entry point constructs the frozen
``independent_endpoint_v1`` covariance internally; callers cannot inject a
substitute covariance. The module contains no target roster, target path,
target workflow, or claim authorization.
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
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_COVARIANCE_SCALES,
    REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    RESIDUAL_STORAGE_SEMANTICS,
    CameraRecorderFamilyMapV1,
    DisjointCameraPartitionV1,
    ReconstructionManifestV1,
    ResidualHistoryDryRunPolicyV1,
    deterministic_disjoint_camera_partition,
    validate_reconstruction_separation,
)
from ._deform360_covariance_residual_history_decision_v1 import (
    ResidualHistoryDryRunDecisionV1,
    ResidualHistoryDryRunResultV1,
)
from ._deform360_covariance_residual_history_last_valid_v1 import (
    run_source_only_residual_history_dry_run,
)
from ._deform360_covariance_residual_history_registered_v1 import (
    REGISTERED_DONOR_RECORD_SCHEMA,
    REGISTERED_EXECUTION_SCHEMA,
    REGISTERED_EXECUTION_VERSION,
    IndependentEndpointCovarianceDonorV1,
    RegisteredResidualHistoryExecutionV1,
    run_registered_source_only_residual_history,
)

__all__ = [
    "CAMERA_PARTITION_NAMESPACE",
    "CLAIM_BOUNDARY",
    "CameraRecorderFamilyMapV1",
    "DisjointCameraPartitionV1",
    "FALLBACK_SEMANTICS",
    "HORIZON_LABELS",
    "IndependentEndpointCovarianceDonorV1",
    "REFERENCE_MEAN_SEMANTICS",
    "REGISTERED_COVARIANCE_DONOR_ID",
    "REGISTERED_COVARIANCE_SCALES",
    "REGISTERED_DONOR_RECORD_SCHEMA",
    "REGISTERED_EXECUTION_SCHEMA",
    "REGISTERED_EXECUTION_VERSION",
    "REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_MATERIAL",
    "REGISTERED_REFERENCE_PREDICTOR_ID",
    "RESIDUAL_STORAGE_SEMANTICS",
    "ReconstructionManifestV1",
    "RegisteredResidualHistoryExecutionV1",
    "ResidualHistoryAdapterV1",
    "ResidualHistoryDryRunDecisionV1",
    "ResidualHistoryDryRunPolicyV1",
    "ResidualHistoryDryRunResultV1",
    "build_residual_history_adapter",
    "deterministic_disjoint_camera_partition",
    "run_registered_source_only_residual_history",
    "run_source_only_residual_history_dry_run",
    "validate_reconstruction_separation",
]
