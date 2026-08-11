# Numerical compatibility profile v1

## Purpose

[`NumericalEnvironmentV1`](numerical_environment_v1.md) remains the exact replay
identity for an evidence-producing process. It binds the complete installed
package inventory, Python patch/compiler details, logical CPU count, NumPy build,
execution controls, and optional dependency lock.

Exact replay and numerical comparability are different questions. An unrelated
installed package or a Python patch/compiler change can alter the exact profile
even when the registered numerical solver state is unchanged. A different
logical CPU count is harmless only when numerical thread counts are explicitly
and completely pinned. `bayesian_phystwin.numerical_compatibility_v1` derives a
second, narrower content identity without weakening the exact profile.

## Compatibility descriptor

Version 1 binds:

- Python implementation and major/minor version;
- NumPy version and normalized NumPy build-configuration digest;
- optional SciPy version;
- byte order;
- the complete registered numerical execution-control mapping;
- an implicit-parallelism record; and
- the dependency-lock digest and byte size.

It deliberately excludes:

- Python patch version and compiler string; and
- the complete installed-distribution inventory.

Logical CPU count is excluded only when all registered CPU thread-count controls
are explicit positive integers and `OMP_DYNAMIC` explicitly disables dynamic
teams. Otherwise the logical CPU count is part of the compatibility descriptor.
Those exact fields remain available and content-addressed through the
`NumericalEnvironmentV1.profile_id` in every case.

## Implicit-parallelism boundary

An unset thread-count variable is not equivalent to a fixed thread count. BLAS,
OpenMP, and expression runtimes may derive their worker count from available
logical CPUs, which can change reduction order and floating-point results.
Therefore the descriptor contains:

```text
implicit_parallelism:
  thread_counts_fully_pinned: bool
  logical_cpu_count: int | null
```

The following registered controls must all contain positive integer values before
CPU count can be omitted:

- `BLIS_NUM_THREADS`;
- `MKL_NUM_THREADS`;
- `NUMEXPR_NUM_THREADS`;
- `OMP_NUM_THREADS`;
- `OPENBLAS_NUM_THREADS`; and
- `VECLIB_MAXIMUM_THREADS`.

In addition, `OMP_DYNAMIC` must be explicitly one of `0`, `false`, `no`, or
`off`, compared case-insensitively. Partial pinning does not suppress CPU-count
binding. When the controls are not fully pinned and the exact profile lacks a
logical CPU count, compatibility derivation fails closed.

This rule is intentionally conservative. A protocol may impose a smaller,
backend-specific control set through a future compatibility-contract version,
but version 1 does not infer the active threading backend from free-form build
text.

## Dependency-lock boundary

A dependency lock is required by default when deriving compatibility identity.
This prevents a claim-bearing comparison from describing two environments as
compatible while leaving resolver input unbound. Diagnostic use may explicitly
set `require_dependency_lock=False`; the resulting descriptor contains a null
lock and must not be presented as a locked evidence environment.

The compatibility descriptor uses lock content and byte size, not its local
basename. Renaming an identical resolver input therefore does not manufacture a
numerical difference, while any byte change does.

## Usage

```python
from bayesian_phystwin.numerical_compatibility_v1 import (
    numerical_compatibility_id_v1,
    numerical_compatibility_record_v1,
    numerically_compatible_v1,
)
from bayesian_phystwin.numerical_environment_v1 import (
    capture_numerical_environment_v1,
)

profile = capture_numerical_environment_v1(
    dependency_lock="requirements.lock",
)
compatibility_id = numerical_compatibility_id_v1(profile)
record = numerical_compatibility_record_v1(profile)

same_numerics = numerically_compatible_v1(profile, another_profile)
```

The record binds the derived compatibility identity to the exact source
`profile_id`. Two records may have the same compatibility identity while keeping
different exact source identities. Record validation requires the exact source
profile, so one profile cannot silently claim another profile's replay identity.

## Interpretation

Matching compatibility identities mean only that the fields declared by this
contract version agree. They do not establish bitwise equality, statistical
equivalence, calibrated uncertainty, provider competence, physical-query
benefit, or reproducibility of code and data that are not separately bound.
Protocols remain free to require exact profile equality or additional hardware
identity when their solver makes those fields relevant.

This module is additive and is not exported from the frozen package-root API.
Existing `NumericalEnvironmentV1` bytes and `profile_id` values are unchanged.
