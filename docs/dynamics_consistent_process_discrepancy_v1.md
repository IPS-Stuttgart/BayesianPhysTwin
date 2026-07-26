# Dynamics-consistent latent process discrepancy v1

This module is an **experimental, disabled-by-default** model extension. It is
intended for use only after source-side and prospective evidence localizes a
material PhysTwin error to the process dynamics. It does not change the released
baseline, the stable Causal4D provider-v1 contract, or any existing frozen result.

## Model

The discrepancy is a low-rank nodal force field

\[
    f_k = B_f c_k,
\]

where `B_f` is constructed from scalar graph modes and `c_k` is a Gaussian latent
coefficient vector in Newtons. The temporal model is strictly stable,

\[
    c_{k+1} = A c_k + w_k, \qquad \rho(A) < 1,
\]

with explicit coefficient covariance. The convenience constructor uses a
discrete Ornstein--Uhlenbeck process parameterized by a half-life and stationary
standard deviation.

## Physical support and conservation

`build_dynamics_consistent_force_basis()` takes explicit masks for eligible,
contact, and attached nodes. Attached nodes are always excluded. Contact handling
is declared as one of:

- `all_supported`: use all eligible, unattached nodes;
- `contact_only`: permit discrepancy only at declared contact nodes;
- `exclude_contact`: permit discrepancy only away from declared contact nodes.

Zero net force and zero net torque are optional because they are appropriate for
an internal unresolved force field, but not for every external contact model. The
constraints are imposed in coefficient space before the final orthonormal basis
is formed. Constraint residuals and active-node counts are recorded in basis
diagnostics.

## Bayesian coefficient inference

`LatentForceBelief` carries coefficient mean and covariance.
`predict_latent_force_belief()` propagates both through the stable temporal model.
`condition_latent_force_belief()` performs a linear-Gaussian update using a
caller-provided physical response matrix. It supports two explicit quadratic
regularizers:

1. coefficient-force precision, and
2. instantaneous mechanical-power precision.

The power penalty is implemented as a zero-valued pseudo-observation of

\[
    p(c) = \sum_i v_i^\mathsf{T} f_i(c).
\]

This makes the work assumption visible and testable instead of hiding it in an
ad hoc state correction. `forecast_latent_force_belief()` exposes both
coefficient covariance and per-node 3-by-3 force covariance for every forecast
frame.

## Exact zero-force parity

`replay_with_process_force_schedule()` examines the complete schedule. If every
entry is exactly zero, it first clears any stale external force and calls the
provider's unchanged `replay_restart()` method. It does not enter the
force-schedule implementation.

The official Warp backend already contains an opt-in external-force tensor guarded
by an integer enable flag. `OfficialProcessDiscrepancyReplayAdapter` uses that
owned hook for non-zero schedules and clears it in a `finally` block. It is not
added to `PhysTwinReplayProvider` v1; existing consumers therefore remain
unchanged.

A release or paper experiment must still record and check byte-level or numeric
baseline parity in its own pinned runtime. Unit tests establish dispatch parity,
not cross-GPU floating-point identity.

## Required controls and source-frozen selection

Every candidate is compared with both:

- the unchanged physical baseline; and
- the existing readout-only discrepancy.

`compare_process_discrepancy_rollouts()` computes the same trajectory metrics for
all three on one common finite support.
`build_process_discrepancy_candidate_configuration()` binds the force-basis ID,
stable-process ID, physical-response model digest, both regularization strengths,
and the schedule policy. `select_source_frozen_process_candidate()` consumes only
source-case summaries and verifies that this complete candidate configuration is
content-addressed. It requires disjoint source and held-out case identifiers, binds
every source case to a SHA-256 digest, and fails if target outcomes are declared as
used for selection. The resulting JSON artifact is itself content-addressed.

The intended sequence is:

1. freeze graph rank, support policy, temporal half-life, stationary force scale,
   physical-response digest, work precision, schedule policy, and acceptance
   thresholds on source cases;
2. write and archive the source-frozen selection artifact;
3. open held-out continuation outcomes only after that artifact is fixed;
4. report baseline, readout-only, and process-discrepancy results together,
   including harmful-update and worst-case regression summaries;
5. retain the unchanged baseline whenever the source gate rejects the process
   candidate.

## Non-goals

This module does not claim that a process discrepancy improves the current
PhysTwin benchmark. It supplies the constrained model and evidence boundary
needed to test that hypothesis without repeating the previously harmful direct
state-injection design.
