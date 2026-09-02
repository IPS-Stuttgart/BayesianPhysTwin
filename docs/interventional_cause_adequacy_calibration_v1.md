# Finite-group calibration of cause-family adequacy

The residual-span test in `interventional_cause_adequacy_v1` requires a radius
for variation that should not be interpreted as evidence of an omitted cause.
Choosing that radius on the target residual would invalidate the
`none-of-the-above` decision.

`CauseFamilyAdequacyCalibrationV1` freezes the radius from complete independent
source groups. For group `g`, let

\[
s_g=\left\|(I-P_S)r_g\right\|_2
\]

be the norm of the source residual outside the complete registered cause span.
For `n` exchangeable source groups and target miscoverage `alpha`, define

\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil,
\qquad
\hat\rho=s_{(k)},
\]

where `s_(k)` is the `k`-th ordered source score. A finite radius is issued only
when `k <= n`; otherwise the requested guarantee is unsupported by the available
number of source groups.

Under exchangeability of the `n` source groups and one future group generated
under the same registered cause-family assumptions,

\[
\Pr\{s_{n+1}\le\hat\rho\}\ge\frac{k}{n+1}\ge1-\alpha.
\]

This is a false-`unmodeled_cause` control under the registered source population.
It does not give power against omitted mechanisms, prove that the cause family is
physically complete, or transfer automatically to another object, material,
action family, sensor, or noise law.

The calibration artifact binds:

- the exact cause-family identity;
- intervention and whitening identities;
- the independent-group rule;
- every source-group score;
- the miscoverage level and finite-sample order statistic;
- evidence that the candidate family was frozen before scores; and
- evidence that no target outcome entered calibration.

A target adequacy certificate records the exact calibration identity in its
metadata and uses the frozen radius without modification.
