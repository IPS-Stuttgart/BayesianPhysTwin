# Later-Branch Source Value: Small Gain, Gate Failed

The implementation was frozen at
`8d87a3f0139027dae7881c589cfed20085010047`. All 11 CPU batches / 88 native
trajectories completed and passed every source admission check. Three
nominal repetitions qualified the new final-action context before the eight
remaining material worlds ran. There were no retries, replacements, unrun
batches, technical failures, or protected-data accesses.

The source bank was sealed before the frozen expected-reward integration.
This is a nine-world finite-model feasibility screen, not an independent
controller test or real-world performance claim.

## Results

| Policy / diagnostic | Expected native reward | Gain over new best fixed |
|---|---:|---:|
| New bank's best fixed action | 7.003344059 | 0 |
| Bias-aware MAP | 7.004792208 | +0.001448149 |
| Bias-aware posterior-mean action | **7.008826125** | **+0.005482066** |
| Posterior ignoring shared bias | 6.998770764 | -0.004573295 |
| Original bank's best fixed action | 7.008321106 | +0.004977047 |
| Perfect-world-information oracle, new bank | 7.014640570 | +0.011296511 |

The bias-aware Bayesian action gains 0.004033917 over MAP and 0.010055361
over the ignored-bias control. These are small positive finite-model source
signals. They are not a new general Bayesian decision principle, a calibrated
safety result, or confirmation on fresh worlds.

After the preset 0.0005 paired numerical margin, the gain over the new best
fixed action is **0.004982066**, below 0.005. It also falls well below the
separate 10%-of-fixed-excess threshold of **0.010334396**. The latter fails
by a substantial margin, not a rounding edge. The primary source gate is
therefore **FAIL**. No new evaluation is authorized, and no threshold or
noise model is adjusted afterward.

The posterior exceeds the original action bank's best fixed reward by only
0.000505019. Thus the apparent gain versus this new bank must not be presented
as a substantial improvement over the strongest preceding control. The old
bank allows different earlier motions; it is an additional strategy-level
comparator, not a prefix-matched ablation. Within the new bank, all policies
have exactly the same first two commanded motions.

The Bayesian mean's Monte Carlo integration standard error is 0.000031432.
This is uncertainty from numerical integration of the declared nine-world
and sensor-noise model, not a confidence interval over physical executions.
The stored per-world expectations and action frequencies remain source-only.

## What Changed Scientifically

The fixed later prefix has whitened stretching secant norm 4.673232, versus
the source gate of 1. Its bending secant norm is only 0.395742. Allowing new
final motions creates multiple world-optimal actions and a small Bayesian
benefit; simply waiting for more informative response does not create a
large enough task gain under the frozen budget and comparators.

The posterior-mean decision already maximizes conditional expected reward
for the declared finite source bank and likelihood. A more elaborate
selector using exactly those same assumptions is not a credible way to
remove the relative-value shortfall. This does not reject different sensors,
actions, tasks, or model classes, and the finite-model result does not bound
real-world performance. It does close this fixed source design.

This is not a pure timing ablation: both the observation times and remaining
action bank differ from the earlier study. It demonstrates their combined
source value only. The -3 N native force, physics, controller, release time,
native reward and 900-step horizon are unchanged. No new contact-path or
strong-force recovery action from the stopped screen is rerun.

## Numerical Qualification

- Three nominal repeats: maximum coordinate range 0.00318999 mm and native
  reward range 0.000000476837, both below their preset budgets.
- Across all batches: maximum retained-reference position error 0.0297978 mm,
  retained-reference reward error 0.0000290871, and duplicate position error
  0.0297152 mm. All fixed endpoint errors are zero.
- Entire 500-frame prefix versus the frozen source reference: maximum error
  4.05648e-13 m. No future observation enters the prefix likelihood.

These are observed engineering qualifications, not population error bounds.
The preceding contact-path failure remains unchanged under its original
1 micrometre/exact-reward rule. This new screen was governed by its separate
prospectively frozen 1 mm / 0.00025 reward contract throughout.

## Verification and Preservation

All 285 relevant tests, Ruff, focused MyPy, and the exact CPU/source/runtime
preflight passed before native execution. The tests include causal frame
boundaries, native corruption, terminal failure accounting, worker gating,
no-information and informative controls, and a MAP-equivalent control that
correctly fails the full promotion gate.

`verify_late.py` rehashes all code and source records, recomputes 88 native
rewards and every native/repeatability gate, and reconstructs the sealed
source bank. Its second likelihood implementation uses a full Kronecker
covariance quadratic form instead of the production Cholesky whitening;
five synthetic comparisons and the native evidence agree. This is a second
implementation by the same agent, not independent human review. It runs no
simulator and changes no evidence.

- Lock ID: `26b58c8e47450eb1d1f59fea039697f563d4f307b52039b5d2ad46f2cdec85b5`.
- Lock file SHA-256: `06b53cc80e671505e1fb1fdc1088fd2b62fdd955ee4f0c3813a681a2fd1b5c83`.
- Result ID: `c424c7b2cf1224f7dca0e6da7a193a5cab95833afecb4a44b7be5246f7dde700`.
- Result file SHA-256: `0219575265c42778a2343f309a84296868da878fe954713e64a7f5301afd1b00`.

Compact records are under `results/source/dlolab_slingshot_late_branch_v1/`.
The complete write-once native root is
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/late-branch-value-source-v1`.
The good DEFORM results and all prior studies remain unchanged. No new
recording, GPU, robot, protected target, held-v8, DLO4/DLO5, public push, or
main merge occurred.
