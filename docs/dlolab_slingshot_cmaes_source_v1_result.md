# Native Slingshot CMA-ES Source Result

The bounded standard CMA-ES search completed all 64 candidate evaluations and
the one selected isolated replay under runtime-binding implementation
`038c1cc4`. No candidate was retried or replaced. The original zero-execution
runtime failure is retained separately. The frozen competence gate is **FAIL**:
six of seven checks pass, but the all-native-memory replay tolerance does not.

| Controller | Native cumulative reward | Cube forward progress | Sphere forward progress |
|---|---:|---:|---:|
| Zero | 6.900000095 | 0 mm | 0 mm |
| Best fixed-bank action | 6.901790142 | 1.790 mm | 108.795 mm |
| Selected CMA-ES action, isolated replay | 7.019133568 | 54.003 mm | 183.781 mm |

Candidate 29 was selected on native reward alone. Its isolated replay reproduces
the reward exactly, and all position traces agree within 5.2273e-9 m. The tracked
gripper remains at least 202.926 mm from the cube. These are useful nominal
task-response observations, not a passing frozen qualification or Bayesian gain.

Eleven final memory fields fail `rtol=1e-6, atol=1e-9`. Maximum differences
include rigid generalized velocity 2.249e-6, generalized acceleration 4.807e-5,
rigid quaternion 5.433e-8, rod velocity 8.769e-8 m/s, and rod directors about
8.507e-8. Generalized coordinates mix translational/angular components, so
their maxima must not be labeled uniformly in metres. No tolerance is changed.

A standalone NumPy-only archive implementation rehashed the native bundles,
recomputed all 64 rewards, checked the projected controls and selected action,
and reproduced the failed seven-check result. It is a second implementation
by the same agent, not an independent human review. The runtime amendment's
112 relevant unit tests, Ruff, and focused MyPy passed; its real-import child
environment preflight also passed before the native run.

Result artifact ID:
`15bc747efc578586ec3916b14617d297271c0cf92c6f1a37866e9b1d56866c45`.
Result file SHA-256:
`8918c51ba2a073dfdb37145058c6950eee8046a275587e57ebde5fe1b7744398`.
Selection file SHA-256:
`e0070719643003bb4e86a122f5894727858b90237f50c16fefaf52f224b4a3c3`.

There is no Bayesian comparison, elastic-launch validation, fresh target,
hardware recording, or SOTA claim. A new source-only mechanism/numerical audit
may examine whether rod-to-projectile coupling causes this response and whether
task-relevant quantities are reproducible under fresh execution. Such an audit
must not rewrite this failed full-memory gate or authorize hidden-state restart.
