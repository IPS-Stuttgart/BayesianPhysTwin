# Recursive sparse-observation belief field

Run date: 2026-07-19

Status: implemented and independently transferred; development evidence only
for the current observation gate. This is not an open-loop state-of-the-art
claim.

## Verdict

A small recursive discrepancy belief can turn sparse material-point
observations into a correction for every point in a deformable physical twin.
The fixed implementation transfers across the released 22-case PhysTwin
cohort, an additional 11-case cloth cohort, and an open 27-episode Deform360
source panel.

The strongest result is the independent-source Deform360 transfer. On 27
episodes spanning five physical objects, the risk-limited RBF field reduces
hidden-identity RMSE by 25.77% and symmetric hidden-point Chamfer by 26.59%
relative to the sealed physical prior. It improves both metrics in all 27
episodes. The local field also beats a matched global-translation update in all
27 episodes.

A causal continuation selector now removes the frozen-state failure in this
open panel. It estimates, from sparse displacement up to the current update,
whether the physical future is actually continuing. The selected hybrid
reduces the two errors by 30.83% and 30.39% versus the physical prior and is
0.90% and 1.21% better than frozen current state. The 0.25 decision threshold
was chosen after the open outcomes were inspected, so this is a paper
hypothesis rather than confirmation evidence.

Three limitations prevent a stronger claim:

1. The Deform360 measurements are sparse identities from a multiview-fused,
   track-assisted material trajectory, not raw single-camera observations.
2. This is an online-assimilation protocol and not parity with the official
   Deform360 open-loop Table-4 evaluator.
3. Correspondence safety and the causal continuation selector were developed
   after these cohorts were opened. Both require a newly held-out transfer.
4. Five-millimetre measurement noise still produces harmful tail cases. The
   current correspondence-safe gate is an exact mismatch control, not a
   general sensor-noise guarantee.

The defensible paper direction is **selective full-field virtual sensing for
deformable digital twins from sparse partial observations**. The remaining
research work is now a preregistered held-out transfer of the causal selector
and risk-coverage family, followed by a raw causal observation operator.

## Method

At each online update, the estimator observes the discrepancy between a
physical rollout and a fixed set of sparse material identities. Its state is
one global translation plus one local residual vector and diagonal variance per
selected centre. A robust Student-t update suppresses individual outliers. A
Gaussian RBF decoder evaluates the same posterior state at arbitrary physical
points:

```text
sparse residual measurements
          |
          v
robust recursive global + local belief
          |
          v
shared RBF discrepancy field at any physical point
          |
          v
physical prior + posterior correction
```

Unavailable centres retain their posterior state. Prediction variance grows
with elapsed forecast frames. Correction magnitude is capped. A rejected
update leaves the corresponding interval bit-for-bit equal to the physical
prior. The continuation selector freezes when fewer than three usable
overlapping material tracks support its causal projection.

This implementation is deliberately simulator-agnostic: it consumes registered
point trajectories, not PhysTwin internals. It therefore also evaluates an
already sealed Deform360 physical prediction without refitting that predictor.
The array evaluator accepts separate measurement and scoring-target streams;
gate decisions, filtering, continuation selection, and interval widths cannot
read the clean scoring target when a noisy observation stream is supplied.

## Information boundary

The PhysTwin evaluator selects 16 geometry-spanning centres using only
frame-zero geometry and pre-test visibility/validity. It observes those centres
at three fixed update fractions. Frames strictly after each update are scored,
and every assimilation identity is permanently excluded from identity error,
manual-track error, and both the observed and predicted Chamfer sets.

The Deform360 adapter is fixed to 27 already-open independent-source episodes:

- `002-rope-silk`: 5 episodes;
- `083-blanket-cloth`: 6 episodes;
- `085-scarf-cloth`: 5 episodes;
- `092-squirrel`: 5 episodes; and
- `170-spider`: 6 episodes.

For each episode, the physical trajectory was checksummed before its source
future was opened. Sixteen centres are selected by deterministic frame-zero
farthest-point sampling. Observations arrive at frames 19, 38, and 57. Scoring
uses later frames only and permanently removes all centre identities from both
directions of the symmetric Chamfer calculation as well as identity RMSE.
Before any target pickle is loaded, the evaluator verifies that its SHA-256,
episode identifiers, and prediction-seal SHA-256 match the opened outcome
manifest.

The current risk rule requires a strict majority, at least 9 of 16 centres, and
a current median radial residual no larger than

```text
max(10 mm, 1.5 * pre-update-history p95 dispersion).
```

The history statistic is frozen to frames `[0, 19)` in Deform360 and to the
pre-test prefix in PhysTwin. Later observations cannot enlarge it.

A stricter correspondence-safe control additionally requires at least 13 of
the 16 current radial residuals to lie below that frozen threshold. It trades
coverage for an executable exact-fallback guarantee under the tested identity
mismatches. It is reported as a risk-coverage endpoint, not as the default
estimator under occlusion.

## Deform360 independent-source transfer

Lower is better. Values are equal-episode means in millimetres.

| Ownership | Arm | Hidden identity RMSE | Symmetric hidden Chamfer |
|---|---|---:|---:|
| External prior | Sealed physical prediction | 10.602 | 9.064 |
| Ours, control | Persistence | 9.361 | 8.246 |
| Ours, control | Recursive global translation | 8.366 | 7.027 |
| Ours, control | Frozen corrected current state | 7.399 | 6.387 |
| Ours | Risk-limited recursive RBF | 7.869 | 6.654 |
| Ours, post-hoc development | Causally selected continue/freeze | 7.333 | 6.310 |
| Ours, exact-safety control | 13-of-16 correspondence-safe RBF | 9.346 | 8.002 |
| Ours, non-selective diagnostic | Ungated recursive RBF | 6.597 | 5.649 |

The risk-limited field accepts 70 of 81 updates.

Against the sealed physical prior:

- hidden identity RMSE changes by -25.77%, with 27/27 episode wins and an
  equal-physical-object cluster-bootstrap mean-difference interval of
  `[-4.226, -1.424]` mm;
- symmetric hidden Chamfer changes by -26.59%, with 27/27 wins and interval
  `[-3.997, -1.143]` mm.

Against recursive global translation:

- hidden identity RMSE changes by -5.94%, with 27/27 wins and interval
  `[-0.667, -0.325]` mm;
- symmetric hidden Chamfer changes by -5.31%, with 27/27 wins and interval
  `[-0.541, -0.208]` mm.

Against persistence, aggregate changes are -15.94% and -19.31%, but only 18
and 17 episodes improve and the physical-object cluster intervals cross zero.
The field is also 6.35% and 4.18% worse than the frozen-current-state control.
Those controls are part of the main result, not optional ablations.

The audited causal selector chooses continuation for 24 and freezing for 46 of the 70
accepted updates. It beats the original risk-limited field by 6.81% identity
and 5.18% Chamfer and beats frozen state by 0.90% and 1.21%. All four
equal-physical-object cluster intervals exclude zero. It retains 27/27 joint
wins versus the physical prior, but only 9 episodes beat frozen state on both
metrics. On released PhysTwin22 and the additional 11 cloth cases, every
accepted update selects continuation, exactly preserving the stronger
physical-continuation endpoint. This domain-adaptive behaviour is promising,
but the threshold remains post-hoc.

Two fresh runs from the audited frozen source reproduced every aggregate, comparison,
artifact checksum, and gate count bit-for-bit. Their summary SHA-256 is:

```text
a52118f37b9bcd04e749cadb5ab1d005499e956024e9b917294c061c95b86052
```

## PhysTwin transfer diagnostics

All PhysTwin numbers below are development or fixed-transfer evidence rather
than a new confirmation claim. Lower is better.

On the released 22 cases, the current support-and-dispersion gate accepts 45 of
66 updates. Relative to the physical prior, it changes hidden non-centre point
error by -8.15% and one-sided L1 Chamfer by -7.27%. Seventeen of 22 cases win
both metrics, and the physical-object cluster intervals exclude zero. The
maximum Chamfer regression is +10.57%, so the predeclared no-large-regression
gate does not pass.

On the additional 11 cloth cases, it accepts 13 of 33 updates. Hidden
non-centre error changes by -7.59%, with cluster interval
`[-2.811, -0.206]` mm. Chamfer changes by -5.74%, but its cluster interval
`[-1.127, +0.023]` mm narrowly crosses zero. Four cases win both metrics. This
cohort was inspected before the present dispersion rule was chosen and is not
an untouched confirmation set.

An earlier fixed method without the support gate improved hidden identities by
10.41% on the additional cohort but made Chamfer 5.93% worse. This negative
transfer motivated the exact-fallback path and is retained as evidence against
reporting only favourable identity metrics.

## Uncertainty audit

The raw coordinate-wise Gaussian posterior is not generally calibrated. On
accepted intervals only, nominal-90% hidden-coordinate coverage is:

| Cohort | Raw coverage | Mean full width |
|---|---:|---:|
| Released PhysTwin 22 | 74.57% | 24.68 mm |
| Additional PhysTwin 11 | 90.79% | 16.89 mm |
| Deform360 27 | 94.23% | 30.16 mm |

A no-scalar conformal-style interval is now an explicit evaluator output. At
an accepted update, it flattens the absolute coordinate residuals of the
currently observed sensor identities and selects order statistic
`min(n, ceil((n + 1) * p))`. The unchanged nominal-90% recipe obtains:

| Cohort | Coverage | Mean full width |
|---|---:|---:|
| Released PhysTwin 22 | 91.12% | 55.88 mm |
| Additional PhysTwin 11 | 96.30% | 24.56 mm |
| Deform360 27 | 94.60% | 31.18 mm |

It repairs Original22 undercoverage but is wide and conservative on the two
transfer cohorts. Coordinates share frames and identities, so this is
conformal-style rather than a formal iid guarantee. Its q90 value is useful as
a selective risk score: error rises monotonically as higher-q intervals are
retained on Original22 and Deform360. Only 5 of 11 additional cases contain an
accepted interval, so that cohort remains selection-biased. The defensible
claim is “causal uncertainty supports selective risk control,” not “a fully
calibrated predictive distribution.”

## Temporal ablation

A fixed residual-velocity term gives a small, consistent improvement over the
static discrepancy state. A conservative fixed coefficient of 0.25 improves
the current field by 0.16% on both released-PhysTwin metrics and by 1.55% RMSE
and 1.40% Chamfer on Deform360, with object-cluster intervals excluding zero in
all four comparisons. It still loses to frozen current state on Deform360 by
4.71% and 2.73%. Residual velocity is therefore an ablation, not the missing
continuation model.

## Corruption stress

The exact frozen gate was rerun with eight deterministic corruption seeds per
episode. In the primary stress, the calibration prefix stays clean and only the
three scheduled sparse updates are corrupted.

| Scheduled-update condition | Accepted updates | Identity change vs prior | Chamfer change vs prior | Joint episode wins |
|---|---:|---:|---:|---:|
| Clean | 70/81 (86.42%) | -25.77% | -26.59% | 27/27 |
| Independent 5 mm Gaussian noise | 393/648 (60.65%) | -14.76% | -15.28% | 20/27 |
| 25% identity mismatch | 475/648 (73.30%) | -15.70% | -15.97% | 21/27 |
| 50% identity mismatch | 1/648 (0.15%) | -0.13% | -0.20% | 1/27 |

The 50% mismatch result demonstrates the executable fallback contract: 647 of
648 intervals return the physical prior exactly, and the maximum regression is
zero. It does not establish general safety. Under 5 mm noise, the worst
episode-mean regressions are +17.67% identity and +43.00% Chamfer; under 25%
mismatch they are +31.56% and +34.29%. A median-dispersion detector is
necessarily insensitive below its breakdown point.

A secondary stress corrupts the frozen history as well as the scheduled
updates. It demonstrates threshold poisoning rather than a deployable
condition: 50% mismatch is accepted in 98.46% of updates, and worst
episode-mean regressions exceed 100%. The history prefix must therefore come
from a trusted calibration channel, or the scale must be externally capped.
For the default median gate, “risk-limited” therefore refers only to exact
rejection behaviour and the tested majority-corruption regime.

The implemented 13-of-16 correspondence-safe endpoint accepts 49/81 clean
updates, improves identity/Chamfer by 11.84%/11.72%, wins both metrics on 24/27
episodes, and has zero clean regression. It rejects every scheduled update in
all eight seeds at both 25% and 50% identity mismatch, returning the physical
prior exactly. On PhysTwin, natural occlusion reduces acceptance to 9/66 on
Original22 and 1/33 on Additional11, retaining less than 1% improvement. This
is why it is a safety endpoint in a risk-coverage family rather than the new
default. Under 5 mm Gaussian noise it accepts only 7.56% of updates and still
has small tail regressions; noise-aware reliability remains open.

## Original versus ours

| Component | Origin |
|---|---|
| PhysTwin simulation, released trajectories, data, splits, and released metric definitions | Original PhysTwin |
| Deform360 recordings and external sealed physical predictions | External Deform360/source-backend work |
| Recursive global/local Bayesian state, robust update, RBF arbitrary-point decoder, covariance, and exact fallback | Ours |
| Centre selection, online protocol, hidden-identity and symmetric-Chamfer evaluator, cluster bootstrap, hashes, and CLI | Ours |
| Persistence, global translation, frozen-current-state, and ungated matched controls | Ours |
| Causal continuation gain/selector, measurement-target separation, and correspondence-safe risk-coverage endpoint | Ours, post-hoc development |
| Update-only finite-sample conformal-style interval | Ours |

No MolmoMotion prediction enters these results. The reusable idea imported from
MolmoMotion-Field is the shared arbitrary-point belief and recursive update;
this implementation tests that idea as an observation-only correction layer on
physical-twin trajectories.

## Reproduce

On `gpuserver6000`, with the repository staged at
`/mnt/corsair/florianpfaff/bpt-online-belief-v1`:

```bash
cd /mnt/corsair/florianpfaff/bpt-online-belief-v1
source /home/florianpfaff/.venvs/motioncrafter-v1/bin/activate

PYTHONPATH=src python -m pytest -q \
  tests/test_phystwin_online_belief.py \
  tests/test_phystwin_online_belief_evaluation.py \
  tests/test_deform360_online_belief_evaluation.py \
  tests/test_online_belief_diagnostics.py

PYTHONPATH=src python -m bayesian_phystwin.cli.deform360_online_belief \
  /mnt/corsair/florianpfaff/deform360-dense-reusable-panel-v1/independent-source-v1 \
  runs/deform360-online-belief-open27-v2-audited-source-development

PYTHONPATH=src python -m bayesian_phystwin.cli.online_belief_diagnostics \
  corruption-stress \
  --output runs/online-belief-diagnostics-v1/corruption-stress.json

PYTHONPATH=src python -m bayesian_phystwin.cli.online_belief_diagnostics \
  tail-gates \
  --output runs/online-belief-diagnostics-v1/tail-gates.json

PYTHONPATH=src python -m bayesian_phystwin.cli.online_belief_diagnostics \
  causal-continuation \
  --output runs/online-belief-diagnostics-v1/causal-continuation.json

PYTHONPATH=src python -m bayesian_phystwin.cli.online_belief_diagnostics \
  residual-velocity \
  --output runs/online-belief-diagnostics-v1/residual-velocity-current-scorer.json
```

The focused test suite contains 27 tests. It checks immutable belief state,
occlusion retention, robust update behaviour, arbitrary-point decoding, causal
centre selection, measurement/target separation, causal continuation,
conformal-style widths, mismatch-safe exact fallback, and permanent centre
exclusion from both evaluators. It also covers manifest binding, unique FPS
identities under coincident geometry, declared gate-candidate routing, and the
four deterministic diagnostic runners.

## Artifacts

- Deform360 clean run:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/deform360-online-belief-open27-v2-audited-source-development`
- Deform360 v2 summary SHA-256:
  `a52118f37b9bcd04e749cadb5ab1d005499e956024e9b917294c061c95b86052`
- Released-22 observation-gated run:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-original22-observation-gated-v3`
- Additional-11 observation-gated run:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-additional11-observation-gated-v3`
- Residual-velocity diagnostic with the current centre-excluding scorer:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-diagnostics-v1/residual-velocity-current-scorer.json`
- Residual-velocity SHA-256:
  `35913d27291e0bfcb773561939a6245b4aecd2e1c8ca43c4c21c1ae5da019038`
- Deform360 corruption stress:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-diagnostics-v1/corruption-stress.json`
- Corruption-stress SHA-256:
  `8c6c50f177870ec380d3e429bb38eae8063372644ec0d1dcea548e54b4462b11`
- Correspondence-safe tail-gate stress:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-diagnostics-v1/tail-gates.json`
- Tail-gate SHA-256:
  `1fbef0979fcbf431fa63d2f8d4cb4efe4679a26a04580bf324a711bc47844abd`
- Causal continuation diagnostic:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-diagnostics-v1/causal-continuation.json`
- Causal continuation SHA-256:
  `f35fe04863f536e2e993ada872f382baa9c640d366fc392f70b4d1214cb494e4`
- Causal conformal-style calibration audit:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/causal-conformal-calibration-audit-v1/summary.json`
- Calibration-audit SHA-256:
  `8e8eb6a223e4f450129db81ae5adb83fa48eab479e809f1d082d020669885fc1`
- Original22 causal-selector run:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-original22-causal-continuation-v4-audited-development`
- Original22 causal-selector summary SHA-256:
  `80979de59465b31b88b48a479d1b447f95ed87bd87cc2630455a9fb774cb467c`
- Additional11 causal-selector run:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-additional11-causal-continuation-v4-audited-development`
- Additional11 causal-selector summary SHA-256:
  `e4459b3309deb53269faaf6a2f95acad5aaeb5f12ef501d142ae90ab82946af6`
- Original22 correspondence-safe run:
  `gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/online-belief-original22-correspondence-safe-v5-development`
- Original22 correspondence-safe summary SHA-256:
  `898a05b31c785453e865ab30ddf8faef730da81406084b5e92a299f3716dd3b8`
