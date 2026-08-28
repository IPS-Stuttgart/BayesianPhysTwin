# Slingshot Contact Mechanism: Retained Result

All three frozen source arms completed under implementation `205e248a`.
Five of six checks pass; the overall gate remains **FAIL**, because one
unchanged repeat exceeds the 1 micrometre full-position replay tolerance.

| Arm | Native reward | Cube progress | Sphere progress |
|---|---:|---:|---:|
| Unchanged repeat 0 | 7.019126892 | 53.995914 mm | 183.777601 mm |
| Unchanged repeat 1 | 7.019134045 | 54.003090 mm | 183.781508 mm |
| Sphere-to-rod contact disabled | 6.900000095 | 8.31e-9 mm | 0 mm |

The registered intervention changes only sphere collision geometry index 3
from non-rigid-coupled to uncoupled; all other coupling flags and all rigid
collision filters remain unchanged. Post-build solver flags verify this
intervention. The selected Cartesian robot command arrays are byte-identical
across all arms. Disabling this contact removes effectively all task response,
supporting rod-to-projectile dependence for this source controller. It does
not provide a full energy accounting or validate real-world elasticity.

The two unchanged repeats differ from the earlier isolated reference by
5.2719228e-5 m and 2.3260833e-7 m in maximum position coordinates. Their native
reward changes remain below 1e-5. The largest discrepancy is about 0.053 mm,
not a macroscopic task failure, but it fails the frozen 0.001 mm check. That
gate and the earlier full-memory failure are not reclassified. No hidden-state
restart or Bayesian comparison is authorized by this audit.

All 99 arrays and metric/gate arithmetic were separately replayed from the
archive without native execution. Result artifact ID:
`9b7caad55329e59d97c6da1010b3e42e9beb98125e793984ca00817df7a9c9aa`.
File SHA-256:
`3a61484640fd0afe314c1c77fbbd97374fa5a6696cc51107a1db2107a5ca746d`.
The 116 relevant tests, Ruff, and focused MyPy passed before execution.

Further task-level source studies must model or bound numerical replay error
explicitly and use fresh full rollouts. They cannot claim exact determinism,
reinterpret this failure as a pass, or transfer a hidden-state snapshot based
on the failed memory gate. Public simulation only; no targets or recordings.
