# Coupled Action Regret: Native Source Study Did Not Pass

## Decision

Do not promote the coupled-regret method from this experiment. All implementation
and evidence-order checks passed, but the scientific gate failed. The joint arm
beats hold and the deliberately uncoupled guard; it does not beat the much
simpler nominal model or calibrated mean-only guard. All 64 episodes have the
same oracle-optimal action. This task therefore supplies no evidence that
Bayesian inference improves action choice.

The source was frozen at `b53e58d523d581dbdd10406374231c3736e89082` before the
native model bank, calibration outcomes, or evaluation predictions were run.
The exact protocol is `docs/dlolab_coupled_action_regret_source_v1.md` and the
compact evidence is `results/sota/dlolab_coupled_action_regret_source_v1/`.
No thresholds, action choices, or method parameters were changed after outcomes.

## Results

39 simulated calibration episodes; 64 distinct simulated evaluation episodes;
one procedural object family. Equal-episode mean task loss includes terminal
tip error and the registered action-effort cost. Values below are in mm^2
(the stored scientific arrays use m^2).

| Arm | Mean task loss (mm^2) | Change vs hold | Non-hold choices | Harmful choices |
|---|---:|---:|---:|---:|
| Hold | 1101.637 | 0.00% | 0/64 | 0/64 |
| Nominal point model | 905.156 | -17.84% | 64/64 | 0/64 |
| Posterior, iid sensor | 905.156 | -17.84% | 64/64 | 0/64 |
| Posterior, shared bias | 905.156 | -17.84% | 64/64 | 0/64 |
| Calibrated mean-regret guard | 905.156 | -17.84% | 64/64 | 0/64 |
| Independent-marginal guard | 1021.787 | -7.25% | 23/64 | 0/64 |
| Coupled-regret guard, primary | 905.156 | -17.84% | 64/64 | 0/64 |
| Score-time oracle ceiling | 905.156 | -17.84% | Not a policy | Not a policy |

The primary's paired mean-gain 95% interval is [181.669, 211.408] mm^2 versus
hold, [92.286, 141.504] mm^2 versus the independent-marginal guard, and exactly
[0, 0] versus the calibrated mean guard. These are 10,000-replicate paired
bootstrap intervals with the frozen seed, not physical-object cluster intervals.

Simultaneous nine-action bound coverage is 90.625% for joint and mean guards,
98.4375% for the independent guard. The independent arm is more conservative in
this trial. Calibration offsets are 19.493, 39.735, and 0 mm^2 respectively
for joint, mean, and independent. The 90% calibration order statistic is rank
36 of 39. Zero observed harmful actions gives a one-sided 95% binomial upper
bound of 4.573% over these simulated episodes; it is not evidence of robot
safety, population physical calibration, or a guarantee under model shift.

Eight of eleven registered gate checks passed. Three failed:

- No 10% gain advantage over the calibrated mean guard.
- No positive paired gain lower bound against that same guard.
- Only one distinct oracle action, below the required three.

The last check is important: even a full-state oracle cannot improve on the
nominal action in this task. The dominant action is also selected by the
posterior and joint/mean guards. The held-out data do not support a claim of
decision-relevant state uncertainty. This does not reject coupled uncertainty
generally; it rejects this 80 ms, contact-free task as evidence of its value.

## Execution And Verification

Native three-world qualification passed all nine checks, including exact replay
of all 15 memory fields. Maximum root error was zero; maximum relative segment
length error was 4.89e-5. Different actions moved free nodes by up to 36.49 mm.
All 15 model worlds, 39 calibration episodes, and 64 evaluation episodes
completed normally. There were zero technical failures, unsealable episodes,
replacements, or retries in the scientific run.

The model bank and calibration were sealed before evaluation. All 64 seven-arm
decisions were sealed and separately copied before evaluation futures were
generated. The scorer rechecked inference, complete native prefix replay, and
action hashes before opening the future stage. A separate arithmetic
implementation passed 10,108 numeric/gate checks, including a dense Gaussian
likelihood, Cartesian quantiles, exact calibration, decisions, losses, intervals,
and primary-gate reconstruction. This is independent arithmetic, not an
independent human review.

73 focused new tests passed; the expanded suite with existing DEFORM, DEFT, and
harm-risk contracts passed 248/248. Ruff and focused MyPy passed. The full
repository suite was not run. Native CPU stage wall times were 19.41 s for the
model bank, 18.91 s for calibration, 10.96 s for decision sealing, and 18.28 s
for evaluation/scoring; these include setup and are not per-policy latency.

Three earlier setup attempts are retained separately: missing sparse-checkout
Python source files, missing EGL import support, and failure to create an EGL
offscreen context after native warm-up. Local software OSMesa then passed the
single-world qualification; the subsequently extended world bank passed too.
These are environment qualification attempts, not discarded method outcomes.
No native physics was changed to resolve them.

## Evidence Identities

- Source lock: `f606edb9b58083d342aa2804108047a078fa884c827e71ccc30aaaf75fee588a`.
- Bank: `638ce8de9dde6b8434777dd066b0c50c34781090ef385fdf6c559fa2b9c82d97`.
- Calibration: `7b1dfbfbb251af254758119bfa0f410d183687fed1082b38931c2658c3f6f382`.
- Predictions: `c044892eb3b24793dec335bfff2deb2502345a7b093bac5be97e9120a201825e`.
- Score: `b9950119eb6f3107834b74fb23d0e318e9e7a8b6edf97f89409c28ccee8edc00`.
- Arithmetic verification: `5ffb6a4f9c37668747ac60761ee6789056e1906c92bf06013093323d37d3c853`.

The original DEFORM state-update evidence remains unchanged and is still the
stronger positive result. Do not tune this frozen decision task on these
outcomes. A distinct future control study should establish source-only
decision sensitivity before investing in a confirmation: a change in hidden
state or physical regime must change the optimal action. Native contact tasks
or force constraints would be a different study, not a relabeling of this one.
No new physical recordings, GPU job, protected target, held-v8, DLO4/DLO5,
official DEFORM DLO3 evaluation, or physical Causal4D record was used. Code and
evidence remain on an isolated local/private branch; nothing was pushed or
merged into the successful backend.
