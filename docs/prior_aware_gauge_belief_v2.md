# Strict prior-aware grouped-mixture admission v2

## Purpose

`bayesian_phystwin.prior_aware_gauge_belief_v2` is a prospective, versioned
admission layer for the existing dense and native-sparse prior-aware grouped
Student-t mixture solvers.

The version-1 solver implementations remain unchanged for historical evidence.
A caller must select v2 explicitly. The module is not exported from the package
root and does not redirect an existing command, protocol, or result.

## Additional admission conditions

Version 1 records whether its robust mixture reached a fixed point and reports
the minimum and maximum eigenvalues of the exact reduced observed-information
matrix. Those diagnostics were previously available to a downstream guard, but
an otherwise numerically admissible update could still be returned when:

- the iteration budget ended before the robust-mixture fixed point converged;
- a precision floor changed the objective from the exact grouped mixture;
- exact observed curvature was non-positive; or
- exact observed curvature was positive but excessively ill-conditioned.

Version 2 calls the unchanged v1 solver and admits the returned update only when
all of the following hold:

1. the objective is `exact-group-mixture-gradient`;
2. the robust-mixture fixed point converged;
3. solution-delta and stationarity diagnostics are present, finite, and
   nonnegative;
4. the exact reduced Hessian diagnostics are present and internally consistent;
5. the exact reduced Hessian is positive definite; and
6. its eigenvalue condition number does not exceed the frozen v2 limit.

Any failure returns the same prior-valued, zero-update fallback used by the
owning dense or native-sparse solver. V2 performs no jitter, eigenvalue clipping,
pseudoinverse, or silent fallback to the working Gauss-Newton covariance.

## API

```python
from bayesian_phystwin.prior_aware_gauge_belief_v2 import (
    PriorAwareGaugeAdmissionConfigV2,
    update_prior_aware_gauge_belief_v2,
    update_sparse_prior_aware_gauge_belief_v2,
)

result = update_prior_aware_gauge_belief_v2(
    batch,
    config=solver_config,
    admission_config=PriorAwareGaugeAdmissionConfigV2(
        maximum_exact_hessian_condition_number=1.0e14,
    ),
)
```

The native-sparse function accepts the existing `SparseGaugeDesignV1` as its
second positional argument. Both functions return
`PriorAwareGaugeBeliefResultV2`, which retains the normal gauge-aware result
arrays and adds an explicit implementation identity in its diagnostics.

## Failure reasons

The strict prospective layer uses stable reasons:

- `strict-v2-non-exact-mixture-objective`;
- `strict-v2-fixed-point-not-converged`;
- `strict-v2-invalid-admission-diagnostics`;
- `strict-v2-non-positive-exact-mixture-curvature`; and
- `strict-v2-ill-conditioned-exact-mixture-curvature`.

An update already rejected by v1 keeps its original reason. V2 records that the
underlying inference rejected the update rather than relabeling it as a strict
post-solver failure.

## Promotion boundary

This is numerical and decision-boundary hardening only. It does not change the
frozen Deform360 10+12-object method, open confirmation payloads, establish
provider competence, calibrate local covariance, demonstrate physical-query
benefit, establish deployment safety, or establish state of the art.

A claim-bearing protocol may use v2 only after freezing the exact implementation,
solver settings, strict condition limit, guard, calibration evidence, and target
information order before outcome access.
