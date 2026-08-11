# Registered source-only covariance path v1

## Purpose

`bayesian_phystwin.deform360_registered_covariance_source_v1` contains the
single source-side execution path for the frozen Deform360 covariance-only
candidate. It is deliberately narrower than the abandoned multi-branch adapter
stack.

The contract answers one implementation question:

> Given a caller-owned registered `last_residual` future mean, do the opened
> causal source residuals reproduce that mean exactly, and can the frozen
> `independent_endpoint_v1` donor provide the preregistered covariance without
> changing the mean?

It does not define or open a target cohort.

## Frozen candidate

The public function
`run_registered_deform360_covariance_source_v1` hard-binds:

- reference predictor: `last_residual`;
- covariance donor: `independent_endpoint_v1`;
- donor implementation: the same evidence-weighted endpoint model average used
  by the frozen full-22 tournament;
- future propagation: consecutive horizons `1, ..., H`;
- early/middle/late covariance scales: `[8, 16, 16]`;
- minimum support: two valid causal observations for every material identity;
- missingness: residual entries outside the Boolean validity mask must be exact
  zero; and
- fallback: the exact caller-owned physical mean and covariance arrays.

The caller cannot supply another donor identity or covariance scale.

## Registered mean verification

For each material identity, the implementation finds the last valid residual in
the causal prefix. It adds that residual to every registered physical-future
frame and compares the resulting array digest with the caller-owned registered
`last_residual` mean.

The accepted result returns the caller's registered mean object by identity. A
mean mismatch returns the exact physical fallback; the function never repairs or
reconstructs the registered comparator.

## Covariance construction

After mean and support admission, the contract fits the frozen endpoint model
average on the prefix and requests predictions at horizons `1` through `H`.
Only the resulting covariance is used. `compose_covariance_only_hybrid` applies
the frozen horizon schedule and verifies that point prediction is unchanged.

The content-addressed source record binds:

- source-unit and source-artifact identities;
- registered-mean and physical-fallback identities;
- prefix cutoff and horizon bins;
- residual, validity, support, mean, fallback, and deployed-covariance digests;
- the covariance-only hybrid artifact identity;
- admission or fallback reason; and
- the implementation-only claim boundary.

## Failure behavior

Structural violations such as malformed shapes, non-real data, nonzero hidden
residual values, invalid identifiers, or non-PSD fallback covariance raise an
error.

The following scientific admission failures return exact whole-case fallback:

- any material has fewer than two valid prefix observations;
- the registered mean differs from the independently reconstructed
  `last_residual` mean; or
- internal donor/covariance construction fails numerically.

Unsupported materials are not silently filled and do not receive a partial
accepted correction.

## Information boundary

This module contains no target roster, target path, target workflow, download
plan, prediction authorization, outcome access, or promotion rule. Passing its
tests is source-only implementation evidence. It does not establish provider
competence, fresh-object calibration, physical-query benefit, Causal4D benefit,
deployment safety, benchmark parity, or state of the art.
