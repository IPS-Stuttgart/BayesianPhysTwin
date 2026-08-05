# Deform360 official-Hub MotionCrafter Stage 1 result

Date: 2026-08-05

Status: provider generation complete; calibration and confirmation remain
sealed.

## Result

The process-isolated v6 runner completed all 30 frozen camera jobs for the ten
calibration objects. Each job produced one disjoint MotionCrafter control, one
latent-linear control, and two independently decoded overlap windows. This is
30 prediction manifests and 120 NPZ members in total.

All prediction-manifest hashes and all declared member hashes were reverified
after copying the bundle from volatile GPU-local shared memory to durable
source-only storage. The durable bundle contains 181 files and
17,811,595,372 bytes. No score, fitted policy, confirmation payload, target
artifact, or outcome artifact exists in the provider output tree.

| Check | Result |
| --- | ---: |
| Frozen camera jobs | 30/30 complete |
| Prediction manifests reverified | 30/30 |
| Declared NPZ members reverified | 120/120 |
| Distinct job IDs / run specifications | 30 / 30 |
| Forbidden score/policy/target artifacts | 0 |
| Future frames used for prediction | No |

The earlier v4 and v5 continuations each stopped on their third camera because
CUDA allocator state survived between jobs in the shared model process. V6
runs every camera in a separate child process, while the parent verifies the
frozen schedule and sealed outputs. It crossed the prior failure point and
completed without changing any camera, frame window, model, seed, or provider
product.

## Information Boundary

Integrity inspection of calibration provider outputs was authorized. Numeric
calibration errors remain unopened, no calibration policy has been fit, and no
confirmation payload or target outcome was accessed. This result establishes
provider execution and provenance only. It does not establish observation
competence, covariance calibration, Bayesian-PhysTwin improvement,
confirmation, or state-of-the-art performance.

The next authorized operation is the separately frozen calibration-object
processing and source gate. Confirmation remains prohibited unless that gate
passes exactly as registered.

## Provenance

- Bayesian-PhysTwin runtime revision:
  `f186ca7bf844fc9f61a0c1e39fd763b2db9b134c`
- process-isolation implementation revision:
  `5b7f1c60546b814c1c34e56db397e4a0877dd36f`
- Prob4D revision:
  `25d90ef7f78ba4307f4555cb636d666004e1bf66`
- MotionCrafter revision:
  `1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257`
- v6 job-manifest ID:
  `9726e7ae12d442956ff81376fe52cdc2f8360fdcd3e5cccbc12543ca584b30f9`
- v6 job-manifest file SHA-256:
  `b9302a27d779a6de619baffc04e624eee629a226a140b90278fa9dd06b213fe2`
- run-report SHA-256:
  `db94d78c9b5acd2c1290976f1ff9647c525df0bad7ab62f4621175ec0fc75383`
- run ID:
  `61ace928b719ff1343b9e6656e5dd49a9e9012bf4e62715051baa9a31fd5726a`
- compact summary SHA-256:
  `fa1be1ff15756db6fd0fc16d335f9021eda13ba337e9e357d0324cc577d5ce0b`
- durable provider root:
  `/home/florianpfaff/source-only/deform360-official-hub-motioncrafter-calibration-v6`

Focused Bayesian-PhysTwin verification passed 31 tests locally and 31 tests
on the exact remote checkout; focused Prob4D verification passed 21 remote
tests. Changed-file Ruff checks passed. The broad Bayesian-PhysTwin run passed
2,099 tests with 31 skipped; its sole failure was the known isolated-import
harness invocation, and that exact test passed when the package was installed
editable.
