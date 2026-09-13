# Anytime-valid simulator admission

## Scientific question

A physical twin may run indefinitely, evaluate a learned correction after each
delayed physical outcome, and be inspected whenever the evidence looks
favorable. A fixed-sample confidence interval does not generally preserve its
nominal error rate under this continuous monitoring.

The proposed interface treats correction admission as a sequential test between
one frozen shadow candidate and the caller-owned physical fallback. It asks for
two pieces of evidence:

1. the candidate improves a registered capped loss by more than a fixed material
   margin; and
2. the probability of a registered materially harmful outcome is below a fixed
   ceiling.

The candidate is deployed only when both evidence processes cross. Before that,
the exact physical fallback is returned.

## Registered bounded gain

For one outcome revealed in the active epoch, let

\[
L_i^c=\min\{\ell_i^c,B\},\qquad
L_i^0=\min\{\ell_i^0,B\},
\]

where `c` denotes the candidate, `0` the fallback, and the loss cap `B` is fixed
before the target outcome is revealed. For a material-gain margin `delta`, define

\[
X_i = \frac{L_i^0-L_i^c-\delta}{B+\delta}\in[-1,1].
\]

The gain null is

\[
H_G:\quad \mathbb E[X_i\mid\mathcal F_{i-1}]\le 0
\quad\text{for every revealed trial }i.
\]

For every fixed or predictable bet `lambda` in `[0,1)`,

\[
M_n^{\lambda}=\prod_{i=1}^n(1+\lambda X_i)
\]

is a nonnegative supermartingale under `H_G`, because its next conditional
multiplicative factor has expectation at most one. Any convex mixture

\[
E_n^G=\sum_j w_j M_n^{\lambda_j}
\]

is therefore also an e-process.

## Registered harmful-outcome rate

Let `H_i` be one when the candidate loss exceeds the fallback loss by more than
the registered harm margin. For a ceiling `q`, the bad-rate null is

\[
H_H:\quad \Pr(H_i=1\mid\mathcal F_{i-1})\ge q
\quad\text{for every revealed trial }i.
\]

Choose fixed alternatives `p_j<q`. The component process

\[
M_n^{p_j}=\prod_{i=1}^n
\left(\frac{p_j}{q}\right)^{H_i}
\left(\frac{1-p_j}{1-q}\right)^{1-H_i}
\]

is a nonnegative supermartingale under `H_H`. Its one-step expectation is affine
and decreasing in the conditional harm probability and equals one at `q`.
Consequently the mixture

\[
E_n^H=\sum_j v_j M_n^{p_j}
\]

is an e-process for the harmful-rate null.

## Anytime and restart guarantee

In epoch `k`, admit only after

\[
E_{k,n}^G\ge 1/\alpha_k^G
\quad\text{and}\quad
E_{k,n}^H\ge 1/\alpha_k^H.
\]

Ville's inequality gives

\[
\Pr_{H_G}\!\left(\sup_n E_{k,n}^G\ge 1/\alpha_k^G\right)
\le \alpha_k^G,
\]

and analogously for the harm process. To support indefinitely many externally
declared restarts, use

\[
\alpha_k=\alpha(1-\rho)\rho^k,\qquad 0<\rho<1,
\]

so that `sum_k alpha_k = alpha`.

Therefore:

- if the gain null holds through every epoch, the probability of ever passing
  the gain gate is at most the total gain budget;
- if the harm null holds through every epoch, the same statement holds for the
  harm budget; and
- if at least one bad null holds in every epoch but its identity may change by
  epoch, authorization is bounded by the sum of the two lifetime budgets.

The registered benchmark assigns `0.025` to each process, yielding a `0.05`
combined lifetime false-authorization bound for that last composite regime.

This is a conditional statistical statement. It requires the candidate,
fallback, trial inclusion, loss cap, harm definition, bet mixtures, epoch, and
reveal order to be fixed or predictable before the corresponding target loss is
seen. Multiple correction candidates require a separate alpha allocation or a
valid e-value merger.

## Delayed outcomes

A shadow trial is issued before its target-dependent losses exist. The trial
stores only an identifier, issuing epoch, issue step, and registered maturity
step. Its losses update the e-process only after maturity. If an externally
declared restart occurs first, the late outcome is retained for audit but cannot
be imported into the new epoch.

The reveal filtration, rather than wall-clock issue order, indexes the
supermartingale. Predictable or exogenous delays therefore do not invalidate the
construction. Outcome-dependent suppression, target-dependent reordering, or
using a late old-epoch result to initialize a new epoch would violate the
contract.

## Fresh controlled experiment

The first claim-bearing experiment is frozen on recursive-corruption v2 seed
roster `200000:200400`, disjoint from the `100000:100200` development roster.
Each seed-domain is one independent trial and aggregates the eleven registered
stress conditions before entering the evidence stream.

- Candidate: `guarded_recursive`.
- Exact fallback: `physical_baseline`.
- Loss: equal-condition mean full-sequence RMSE.
- Loss cap: `15 mm`.
- Material gain margin: `0.25 mm`.
- Harm event: candidate episode loss exceeds fallback episode loss.
- Harm ceiling: `10%`.
- Outcome delay: uniformly frozen integer from 1 to 12 later episodes.
- Minimum matured outcomes: 25.
- Gain and harm lifetime budgets: `2.5%` each.

At episode `t`, all outcomes whose registered delays have matured are first
revealed. The deployment choice for episode `t` is then made from the current
e-values. Only after the shadow comparison and reveal time are registered is the
fresh seed-domain generated. The result reports the first authorization time,
selected cumulative loss, selected harmful episodes, candidate regret, and
byte-exact fallback identity.

A separate implementation calibration simulates 5,000 worlds under each null,
monitors after every outcome, and permits four geometrically alpha-spent epochs.
Those simulations are a software check, not the source of the mathematical
guarantee.

## Why this could be a large contribution

The current guard is a fixed source-qualified selector. This extension changes
the claim from

> the threshold worked on a frozen cohort

into

> the system may monitor indefinitely and stop when evidence is sufficient,
> while preserving a registered lifetime false-promotion budget under delayed
> feedback and optional stopping.

That is operationally relevant to a physical twin that accumulates experience
over months rather than one benchmark split. It also creates a clean interface
between Bayesian physical belief revision and frequentist sequential evidence:
the Bayesian candidate carries the physical belief, while the e-process governs
whether accumulated observable losses justify promoting that candidate.

## Claim boundary

A successful controlled experiment supports an anytime-valid **admission
mechanism** under its registered score and conditional nulls. It does not imply:

- that all real robot episodes satisfy those null assumptions;
- robustness to arbitrary nonstationarity inside an epoch;
- calibration of the physical-state posterior;
- a bound on unobserved physical damage or general deployment safety;
- validity after outcome-dependent candidate changes; or
- state of the art in deformable prediction.

A real-data successor should freeze one candidate on source trajectories and
stream independent held objects, sessions, or trajectories as delayed blocks.
The physical outcome used to update evidence must be defined before deployment,
and each block may enter only once.

## Statistical background

The construction uses standard nonnegative-supermartingale and e-process
principles for anytime-valid inference. The contribution pursued here is their
physical-twin admission contract: complete candidate versus exact physical
fallback, a separate harmful-update gate, delayed-outcome bookkeeping, and
lifetime alpha spending over simulator-correction epochs.
