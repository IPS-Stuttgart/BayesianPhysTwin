# Joint anytime-valid admission v2

## Why version 2 is stronger

Version 1 accumulated two independent evidence streams before authorizing a
learned simulator correction:

1. bounded evidence that the candidate improves a registered capped loss; and
2. Bernoulli evidence that materially harmful updates occur below a registered
   ceiling.

A naive implementation assigns half of an epoch's error budget to each stream.
That is valid but unnecessarily conservative because the candidate is admitted
only when **both** requirements pass.

Version 2 treats admissibility as an intersection--union test.

## Registered hypotheses

Let \(X_t\in[-1,1]\) be the bounded fallback-minus-candidate gain after subtracting
the required mean-gain margin. Let \(H_t\in\{0,1\}\) indicate a materially harmful
candidate outcome under the frozen harm definition. The invalid-candidate null
is

\[
\mathcal H_0
=
\mathcal H_{\mathrm{gain}}
\cup
\mathcal H_{\mathrm{harm}},
\]

where

\[
\mathcal H_{\mathrm{gain}}:
\mathbb E[X_t\mid\mathcal F_{t-1}]\le 0
\]

throughout the epoch, and

\[
\mathcal H_{\mathrm{harm}}:
\Pr(H_t=1\mid\mathcal F_{t-1})\ge \rho_{\max}
\]

throughout the epoch. The admissible alternative requires both positive mean
gain and harm rate below the ceiling.

For epoch \(j\), let \(E^{g}_{j,t}\) and \(E^{h}_{j,t}\) be valid component
e-processes. Version 2 latches the events

\[
\sup_t E^{g}_{j,t}\ge 1/\alpha_j,
\qquad
\sup_t E^{h}_{j,t}\ge 1/\alpha_j,
\]

and admits only after both have occurred and the minimum sample count is met.
If the gain null is true, joint admission is a subset of the first event. If the
harm null is true, it is a subset of the second. Thus, for every distribution in
the union null,

\[
\Pr(\text{false admission in epoch }j)\le \alpha_j.
\]

No independence between the component evidence streams is required. No
Bonferroni division by two is required. A geometric schedule with

\[
\sum_{j\ge0}\alpha_j\le\alpha
\]

therefore controls the probability of one or more false admissions across an
unbounded sequence of declared epochs.

The qualification is important: one fixed component null must hold throughout
the epoch. The theorem does not cover arbitrary within-epoch alternation between
incompatible null regimes.

## Content-addressed decision contract

Every controller instance binds the following identifiers before any outcome is
revealed:

- candidate belief or correction;
- exact physical fallback;
- paired score;
- harmful-outcome definition;
- information set;
- reveal policy; and
- all statistical thresholds and betting grids.

The canonical JSON descriptor is hashed with SHA-256. Pending trials retain this
digest and cannot update a controller created under another contract. Selection
also verifies the candidate and fallback identities, then returns the exact
registered object rather than reconstructing a numerically similar belief.

## Delayed outcomes and epochs

A trial is registered before its outcome matures. It records:

- issuing epoch;
- issue and maturity steps;
- whether it belongs to admission or post-admission revocation evidence; and
- the decision-contract digest.

An outcome from a closed epoch is retained for audit but cannot update the new
epoch. External domain-shift declarations and evidence-based revocations both
return deployment to the exact fallback and allocate a fresh summable error
budget.

## Change-point revocation

A monitor started once at promotion can be slow after a late shift because it
carries all favorable pre-shift evidence. Version 2 instead mixes reverse-gain
e-processes over every deterministic post-admission start time \(s\), using

\[
\pi_s=\frac{1}{s(s+1)}.
\]

The unresolved future-start mass after \(t\) outcomes is \(1/(t+1)\), so the
weights sum to one. This heavy-tailed start-time mixture remains an e-process
under the registered reverse-gain null and can detect degradation beginning at
an unknown time without choosing that time after observing the outcomes.

Revocation bounds false evidence-based revocation under its own null and beta
budget. It does not guarantee zero harmful deployments after an abrupt shift.

## Controlled efficiency comparison

The frozen protocol `protocols/anytime_joint_admission_v2.json` compares:

- shared-alpha intersection--union admission; and
- an otherwise identical Bonferroni-split gate.

The first epoch receives alpha 0.025. The shared rule therefore uses component
threshold 40, while the split rule uses threshold 80. The study contains two
union-null distributions—one on the mean-gain boundary and one on the harm-rate
boundary—and two beneficial low-harm alternatives. It reports Monte Carlo false
admission, Wilson intervals, power, and first-crossing time.

## Paper-facing claim

A positive controlled result supports:

> A learned correction can be admitted under continuous monitoring with a
> lifetime false-admission budget while simultaneously requiring positive mean
> utility and a bounded harmful-update rate; exploiting the intersection--union
> structure improves admission efficiency without weakening that bound.

The real DLO sequence remains retrospective. Neither the controlled study nor
the replay establishes fresh deployment validity, physical safety, arbitrary
nonstationarity, universal transport, or causal identification.
