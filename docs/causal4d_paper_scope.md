# Causal4D Paper Scope

## Core claim

The first Causal4D paper is about:

> **Bayesian abduction of realized interventions for counterfactual prediction
> of deformable-object dynamics.**

The paper studies the causal distinction

```text
command u != realized intervention z = (phi, kappa)
```

where `phi` contains persistent actuation variables such as gain, delay, and
controller-frame bias, while `kappa` contains event-specific contact and slip.
An uncertain physical twin supplies state, parameter, and discrepancy beliefs.

## Contribution chain

The main paper must establish this chain, in order:

1. define commanded and realized interventions as distinct variables;
2. infer `z` jointly with uncertainty over the physical twin;
3. implement explicit abduction, `do(u_cf)`, and posterior prediction;
4. demonstrate held-out contact/action gains in controlled experiments;
5. validate factual and held-out interventional prediction on a same-object,
   multi-action real protocol;
6. report calibrated uncertainty, or state a precise empirical calibration
   boundary when nominal coverage is not attained.

No later component can substitute for a missing earlier link. In particular,
language ranking and software planning cannot compensate for weak real
intervention-abduction evidence.

## Claim hierarchy

| Tier | Component | Paper role |
| --- | --- | --- |
| Core | `u` versus realized `z=(phi,kappa)` | central problem and contribution |
| Core | joint twin/intervention posterior | central method |
| Core | abduction-intervention-prediction | central causal operator |
| Core | controlled held-out contacts/actions | causal validation |
| Required real evidence | same-object multi-action protocol | external validation |
| Required boundary | independent-execution calibration | probabilistic claim or explicit limitation |
| Supporting backend | Bayesian-PhysTwin and PhysTwin/Warp | uncertain physical model, not a new reconstruction claim |
| Optional experiment | MolmoMotion task posterior | appendix only after its independent acceptance gate passes |
| Application | constrained closed-loop planning | application/software demonstration without robot execution |
| Out of scope | MotionCrafter association and dense perception assimilation | separate Bayesian-PhysTwin/Hao work |

## Evidence status

As of 2026-07-12:

| Required link | Status | Evidence |
| --- | --- | --- |
| command/realization decomposition | implemented | typed `u_obs`, `phi`, `kappa_obs`, and `u_cf` artifacts |
| joint abduction | implemented | prefix-only posterior over twin and realized intervention |
| explicit causal operator | implemented | separate abduction, action, and prediction stages |
| controlled held-out gains | passed | shifted-contact RMSE `4.132 -> 0.805 mm`; coverage `77.9% -> 90.8%` |
| same-object multi-action real validation | pending acquisition | locked 36-execution protocol exists; no physical collection claimed |
| cross-action calibration boundary | pending real protocol | graph persistence reaches `67.78%` on one target; rejected transfer reaches `43.03%` |

**Current decision:** the method and controlled result justify continuing the
paper, but the complete first-paper claim is not yet ready. The next decisive
evidence is the same-object multi-action real protocol, including either
successful held-out calibration or a well-powered cross-action bound on its
failure, not another architecture component.

## Main-paper experiment matrix

The minimum comparison is:

| Method | Twin uncertainty | Realized intervention inference |
| --- | ---: | ---: |
| nominal PhysTwin | no | no |
| Bayesian-PhysTwin | yes | no |
| Causal4D | yes | yes |
| intervention oracle | fixed diagnostic only | oracle |

The main metrics are factual continuation error, held-out interventional
prediction error, contact/gain/delay/slip recovery where ground truth exists,
coverage, interval width, NLL or energy score, NEES, and worst-group coverage.
Oracle results diagnose inference, proposal, and model gaps; they are not
deployable baselines.

MolmoMotion is excluded from this matrix. The current corrected checkpoint does
not beat zero or constant velocity and ranks the true action fifth of five, so
`beta=0` remains the only admitted setting. Closed-loop planning may illustrate
use of the posterior, but it is not a robotics contribution without genuine
hardware execution.

## Language discipline

Use the following terms precisely:

- **Controlled counterfactual prediction:** valid when simulator exogenous
  conditions are shared across factual and alternative interventions.
- **Held-out interventional prediction from matched initial conditions:** the
  correct description for repeated real executions.
- **Realized-intervention posterior:** appropriate on real data even when the
  true contact node is not directly observed.
- **Calibrated:** reserved for held-out independent-execution coverage that
  passes the locked protocol.

Do not claim individual-level real counterfactual ground truth, real contact
recovery without instrumentation, a calibrated real posterior at current
coverage, language-conditioned world modeling, or robot control.

## Paper structure

1. Problem: commanded actions do not identify realized physical interventions.
2. Model: uncertain deformable twin plus persistent and event-specific
   intervention variables.
3. Inference: factual abduction followed by explicit intervention and
   prediction.
4. Controlled evaluation: recovery and held-out counterfactual prediction.
5. Real evaluation: same-object factual, same-grasp, and new-contact protocols.
6. Calibration and limitations: independent executions, discrepancy, replay
   variance, and bounded claims.

MolmoMotion and the closed-loop runner belong in an appendix or application
section only if they clarify robustness without enlarging the headline claim.
