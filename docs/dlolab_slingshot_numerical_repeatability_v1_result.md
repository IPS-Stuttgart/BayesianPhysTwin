# Native Repeatability: Observed Budget Passed

The CPU source implementation was frozen at
`2d02c80bfa8352daf4328feff15c0f978d4434dc`. All 15 registered fresh-process
batches completed and passed admission, with 120 trajectories, 465 verified
force commands, and 15 verified release commands. There were no retries,
replacements, unrun batches, or retained technical failures.

Only previously known controls were executed. Five batches per contact world
used two permutations of the same policy multiset, with duplicate slots.
These are numerical repeats in three simulated worlds, not 120 independent
physical executions. No new recovery path, posterior, action selection,
control-value comparison, or protected data was evaluated.

| Registered diagnostic | Observed maximum | Preset budget | Decision |
|---|---:|---:|---|
| Same-policy native reward range | 0.0000834465 | 0.00025 | Pass |
| Batch-mean paired reward-contrast range | 0.0000649691 | 0.00050 | Pass |
| Same-policy coordinate range | 0.280590 mm | 1.0 mm | Pass |

The largest reward range is 1.67% of the preceding studies' minimum useful
gain of 0.005. The paired-contrast range is 1.30% of that gain. This makes a
prospective numerical allowance plausible, but it is not a calibrated bound
on future errors or evidence of improved control.

Variation was concentrated in the middle coupling world (0.6), for the
already-known force-screen policy 6. Its maximum projectile and target-cube
coordinate ranges were 0.280590 and 0.155450 mm. Its same-layout/same-slot
cross-process reward range was also 0.0000834465, so slot permutation alone
does not explain the variation. Layout A minus B mean reward was 0.00000830491;
this descriptive contrast is not a randomized estimate of a layout effect.
The other two coupling worlds had identical rewards across their repeats,
but their position arrays were not generally byte-identical.

Each paired contrast uses policy means over duplicate slots within a batch.
The smaller paired range is therefore not, by itself, evidence that shared
noise cancellation improves decisions: duplicate averaging also changes this
comparison. The recorded five-batch reward and contrast covariances are
descriptive and are not fitted into a controller or treated as calibrated UQ.

## Scientific Boundary

This passes only the registered **observed numerical-budget screen**.
Five batches in each of three known worlds do not establish a population
repeatability bound, a simulator-wide guarantee, or robustness for different
actions/materials. No controller study is authorized automatically.

The preceding contact-path study remains closed at its original
1 micrometre/exact-reward reference gate. Its failed result is neither
rescored nor relabeled. A new study must specify its own reference and
numerical-error contract prospectively. Exact software fallback should mean
returning the frozen incumbent artifact unchanged, not assuming that a
fresh native replay is bit-identical to a previous simulation.

The positive DEFORM result and all earlier negative studies are unchanged.
No new recording, GPU, robot, protected target, held-v8, or DLO4/DLO5 access
occurred. Nothing is pushed or merged to main.

## Verification and Preservation

The implementation passed 251 relevant tests, Ruff, focused MyPy, and the
exact CPU/runtime/source preflight. A separate arithmetic implementation
rehashed all native and source records, recomputed 120 native rewards,
verified all force/release commands, and reproduced the ranges, covariance
matrices, and budget decision. It enumerates pairwise differences rather
than using the production range operation and forms covariance contrasts
entrywise. Six synthetic arithmetic cases also agreed. This is a second
implementation by the same agent, not independent human review.

- Lock ID: `b5598b27b174a0eb5e62635d008c1ecbb8b13f4199f2f771b44d898fcb1d0ca8`.
- Lock file SHA-256: `d54e43ceca2bc4ecaa291e00dbda3023840b58787562151573ddc817bf0d0078`.
- Result ID: `f9cf9969ae66d244408971700f6f1d9c0b6b6167df1e78afc24cb16bcd45b0fe`.
- Result file SHA-256: `7a981d7a408a4562a4bed5172510a6e23ea91c01219b1ca4409e48b371728b66`.

Compact lock, result, and all 15 admission records are under
`results/source/dlolab_slingshot_numerical_repeatability_v1/`. The full native
root remains write-once at
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/numerical-repeatability-v1`.
