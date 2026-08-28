# Native Slingshot Numerical-Repeatability Audit

This source-only audit follows a retained-reference mismatch in the stopped
contact-path screen. It does not rerun that recovery bank, open its value
comparison, relax its 1 micrometre/exact-reward gate, or select a new action.
Only the already-known force-screen policies 2, 6, and 5 are executed.

Use the same three coupling worlds (0.9, 0.3, 0.6 execution order), nominal
placement/material, seed 0, native arm controller, and 900-step horizon.
For each world run five fresh CPU processes, each with eight environments:

| Layout | Force-screen policy indices by native slot | Fresh processes |
|---|---|---:|
| A | 2, 6, 5, 2, 6, 5, 5, 5 | 3 |
| B | 6, 2, 6, 2, 5, 5, 5, 5 | 2 |

This is 15 batches / 120 trajectories, not 120 independent physical worlds.
Layouts have identical policy multisets and differ only by slot permutation.
Policies 2 and 6 use -24 N per finger after step 300; policy 5 uses -3 N.
Reset force is -1 N and all first-macro forces are -3 N. Every actual solver
command must match. Release remains position 0.08 m at step 700. No material,
geometry, control limits, observations, hidden-state restart, or upstream
source is changed. No new vertical-path candidate is executed.

Admission verifies hashes, exact command bytes, complete finite native arrays,
material/contact realization, actuator schedule, reward arithmetic, and fixed
endpoints. Duplicate, cross-process, and cross-layout differences are the
measurements in this audit; they are not used to censor batches. Their older
QA results remain recorded as diagnostics. A technical/physical admission
failure stops the audit, is retained, and permits no retry or replacement.

Report per-world/per-policy reward ranges and maximum coordinate ranges across
all observations, within each batch, and between fresh processes at the same
layout and slot. Report the layout-A minus layout-B mean reward difference.
For each batch compute each policy's mean over its duplicate slots, then form
the two strong-policy-minus-fallback reward contrasts. Report their ranges
and the descriptive reward/contrast covariance across the five batches.
Do not treat duplicate slots or coordinate frames as independent replicates.

The source engineering budget is tied to the earlier minimum scientific gain
of 0.005: an observed per-policy reward range no greater than 0.00025 (5%),
paired-regret range no greater than 0.0005, and coordinate range no greater
than 1 mm. All three checks are required for an observed-budget pass. These
budgets are new audit criteria, not amendments to the failed study. Finite
observed ranges are not statistical upper bounds for future runs. Five
batches do not establish a calibrated stochastic simulator model, and batch
layouts are not exchangeable random physical executions. A pass authorizes
no controller-value study by itself and cannot revive the stopped bank.

The write-once root is
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/numerical-repeatability-v1`.
The lock binds the clean implementation, source/test/doc bytes, runtime and
public asset lineage, exact schedules, known-control references, and the
stopped path result. Each native task is claimed before initialization; each
later task rederives prior admission from sealed inputs. Record all 15 planned
tasks, completed/admitted counts, and unrun dispositions even on failure.

No GPU, new recording, robot, protected target, calibration/evaluation world,
held-v8, DLO4/DLO5, public push, or main merge is authorized. The positive
DEFORM result and all earlier studies remain unchanged.
