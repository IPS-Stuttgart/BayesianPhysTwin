"""Dynamics-consistent latent process discrepancy for deformable PhysTwins.

The public API represents nodal discrepancy force as ``f_t = B_force c_t``.
The force basis is graph-localized and may impose hard internal net-force and
net-torque constraints. Coefficients follow a stable Gaussian AR(1) process and
can be conditioned on reliability-weighted inverse-dynamics force evidence with
an optional local mechanical-work prior.
"""

from ._process_discrepancy_basis import (
    ProcessDiscrepancyBasisV1,
    build_process_discrepancy_basis,
)
from ._process_discrepancy_inference import (
    apply_process_discrepancy_force,
    process_discrepancy_step,
    update_process_discrepancy,
)
from ._process_discrepancy_model import (
    PROCESS_DISCREPANCY_SCHEMA_VERSION,
    ProcessDiscrepancyDynamicsV1,
    ProcessDiscrepancyFitBoundaryV1,
    ProcessDiscrepancyModelV1,
    ProcessDiscrepancyStateV1,
    ProcessDiscrepancyUpdateV1,
    initial_process_discrepancy_state,
    predict_process_discrepancy,
    process_discrepancy_force_moments,
)

__all__ = [
    "PROCESS_DISCREPANCY_SCHEMA_VERSION",
    "ProcessDiscrepancyBasisV1",
    "ProcessDiscrepancyDynamicsV1",
    "ProcessDiscrepancyFitBoundaryV1",
    "ProcessDiscrepancyModelV1",
    "ProcessDiscrepancyStateV1",
    "ProcessDiscrepancyUpdateV1",
    "apply_process_discrepancy_force",
    "build_process_discrepancy_basis",
    "initial_process_discrepancy_state",
    "predict_process_discrepancy",
    "process_discrepancy_force_moments",
    "process_discrepancy_step",
    "update_process_discrepancy",
]
