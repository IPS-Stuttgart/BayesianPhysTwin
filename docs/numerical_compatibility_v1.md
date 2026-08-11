# Numerical compatibility profile v1

## Purpose

[`NumericalEnvironmentV1`](numerical_environment_v1.md) remains the exact replay
identity for an evidence-producing process. It binds the complete installed
package inventory, Python patch/compiler details, logical CPU count, NumPy build,
execution controls, and optional dependency lock.

Exact replay and numerical comparability are different questions. An unrelated
installed package, a different logical CPU count, or a Python patch/compiler
change can alter the exact profile even when the registered numerical solver
state is unchanged. `bayesian_phystwin.numerical_compatibility_v1` derives a
second, narrower content identity without weakening the exact profile.

## Compatibility descriptor

Version 1 binds:

- Python implementation and major/minor version;
- NumPy version and normalized NumPy build-configuration digest;
- optional SciPy version;
- byte order;
- the complete registered numerical execution-control mapping; and
- the dependency-lock digest and byte size.

It deliberately excludes:

- Python patch version and compiler string;
- logical CPU count; and
- the complete installed-distribution inventory.

Those fields remain available and content-addressed through the exact
`NumericalEnvironmentV1.profile_id`.

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
