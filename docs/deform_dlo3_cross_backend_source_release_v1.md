# DEFORM-to-PyElastica: verified source-only transfer

## Finding

**Unchanged DEFORM-fitted local-residual models improve a different physical
backend, PyElastica, on all eight registered DLO3 source-test trajectories.**
The equal-seed correction reduces mean coordinate L1 from **22.854730 mm to
22.172621 mm**, a **2.984544%** improvement, without fitting any parameter to the
PyElastica transfer outcomes. All three individual DEFORM seed models improve
the mean. The registered direct-transfer gate passes.

This is a recovery and verification of an already completed source experiment,
not a new run, new target opening, or retrospective revision of its gate.

## Complete comparison

Lower mean coordinate L1 is better. Rows share eight complete DLO3
`train/source_test` trajectories, 498 forecast frames per trajectory, and 12
nodes. The transfer runner uses the first two states and known clamped actions;
future free-node observations are used for evaluation, not to form predictions.

| PyElastica prediction | Mean L1 (mm) | Gain vs raw | Wins / 8 | Registered result |
|---|---:|---:|---:|---|
| Raw, sealed backend | 22.854730 | Reference | - | Reference |
| DEFORM correction, unchanged equal-seed coefficients | 22.172621 | 2.9845% | 8 | Direct-transfer gate passes |
| Same correction, one leave-one-trajectory-out scalar | 22.340514 | 2.2499% | 6 | Scalar gate passes, but weaker than direct |
| Previously fitted PyElastica-specific correction | 17.744533 | 22.3595% | 8 | Backend-specific reference |

The direct arm's trajectory-level paired bootstrap interval for candidate minus
raw is **[-0.793157, -0.566640] mm**. Its worst trajectory ratio is **0.983117**.
Seed 42/43/44 means are respectively 22.154278/22.231110/22.139672 mm; the
equal-seed arm was frozen, not selected from the best seed.

The scalar arm's corresponding interval is **[-1.166038, +0.286564] mm**, and it
loses on two trajectories. Its worst ratio is 1.089442. Although it passes the
registered direction/magnitude gate, it does **not** provide stronger evidence
than the direct arm or a confidence-interval-based improvement claim. Its
least-squares scalar is fitted to coordinate L2 on seven other trajectories,
whereas evaluation uses coordinate L1. Additional scalar fitting was not a
reliable improvement here.

All eight transferred correction fields have positive alignment with the
PyElastica residual; the median cosine is 0.370749. The exact no-refit arm retains
**13.3480%** of the backend-specific correction gain. Thus some predictive
correction transfers, but most of the available gain remains backend-specific.

## What this adds

The supported progression is now:

1. The same correction procedure can be fitted separately for different backends.
2. A limited component of the **fitted correction itself** also transfers from
   DEFORM to PyElastica without coefficient refitting on this source panel.

DEFORM here is the deformable-linear-object backend, not DeformMaster.

This directly supports a narrowly scoped backend-relative accuracy claim. It
does not make the correction universally backend-independent, show that the
shared component is a uniquely identified physical mechanism, or establish that
Bayesian uncertainty caused this point-accuracy gain. Shared observation or
representation effects could also contribute.

The eight trajectories belong to one DLO, not eight independent physical
objects. The archived JSON calls its interval `object_bootstrap_95_interval_m`;
the actual resampling unit is the **complete source-test trajectory**. The
scalar interval is additionally conditional on overlapping fixed LOO fits.
Both intervals are descriptive source evidence, not fresh-object confirmation.
No SOTA, target-transfer, calibration, or deployment-safety claim follows.

## Provenance and verification

- Existing Actions run: [33536420739, attempt 1](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33536420739), completed successfully on 2026-09-01.
- Workflow head: `9eec2835d328dc9a059154e836fa484c51d835d6`.
- Frozen execution implementation: `c124296f754377ed72459945674d6b0e88dbf9b8`.
- Archive ID: `9811886089`; archive name: `deform-dlo3-hierarchical-transfer-33536420739-1`.
- Source-only export receipt: `f0d4ca0ff98bcfd730a8a4036ae7a6165dd3667552724a28b3f5f5a9039e2116`.

The mixed archive also contains protected evaluations. The export read only ZIP
directory metadata, checksum metadata, and the ten explicitly allowed
`cross-backend/` and `cross-backend-scalar/` members using exact HTTP byte ranges.
It read 20,906 of 67,379 archive bytes. Each source member passed its ZIP CRC,
length check, and SHA-256 comparison against the archive checksum manifest.
No protected member payload, combined outcome report, or mixed workflow log was
read. The GitHub-reported whole-ZIP digest is retained as an API attestation;
it was **not** recomputed by downloading the mixed archive.

The ten source artifacts are preserved byte-for-byte in
`results/sota/deform_dlo3_cross_backend_source_release_v1/`, with the exact export
script as a text provenance record. Regression checks independently recompute
every paired summary and bootstrap interval from the released per-trajectory
errors, rederive both registered gates, check CSV agreement and protocol/seal
identities, and verify the export receipt and all member hashes. They do not
claim to rerun native simulation, reconstruct prediction tensors, or refit
models. The six executed source/protocol/runner files match the current
`9d7383ea56a0a9e3ad6753d1c42fe653cd7e615d` source bytes exactly.

Local verification: 29 source-transfer/release tests pass; changed-test Ruff
and strict MyPy pass. These are focused checks, not a full repository-suite run.

## Next experiment, not authorized by this release

Keep the successful DEFORM implementation and the exact direct-transfer arm
unchanged. A stronger study would prospectively compare raw PyElastica, this
unchanged transfer, and the backend-specific reference on a separately
authorized untouched set, with a simple persistent/bias-correction control.
Decide that protocol before outcomes; do not choose a new scalar or tune this
eight-trajectory panel. This release grants no access to DLO3 official evaluation,
DLO4/DLO5, held-v8, or other protected targets.
