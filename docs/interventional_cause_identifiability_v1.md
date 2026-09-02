# Why Is the Twin Wrong? Interventional cause identifiability v1

## Scientific question

A residual around a physical twin can be explained by several operationally
different causes:

- physical state error;
- physical parameter error;
- realized intervention or contact error;
- observation-frame bias; or
- source-local model/readout discrepancy.

A good fit under one action does not distinguish these explanations.  The new
certificate asks which registered cause-specific quantity is identifiable from
its **response across changed interventions**, after every competing cause is
projected out.

## Model

For interventions `u in U`, stack the whitened response residuals and local
response signatures:

\[
r = \sum_{c\in\mathcal C} S_c\beta_c + N\nu + \epsilon,
\qquad
S_c = \operatorname{vstack}_{u\in\mathcal U} S_{c,u}.
\]

`N` contains declared nuisance directions that are not themselves candidate
causes.  For cause `c`, define the competing design

\[
N_c = [N, S_1,\ldots,S_{c-1},S_{c+1},\ldots,S_C]
\]

and residualized signature

\[
A_c=(I-P_{N_c})S_c.
\]

Let the registered cause-specific query be

\[
q_c=B_c\beta_c.
\]

## Theorem: cause-query identifiability

The registered cause query is uniquely determined by the stacked response for
all compatible competitor coefficients if and only if

\[
\ker(A_c)\subseteq\ker(B_c).
\]

Equivalently, there exists a linear map `M_c` such that

\[
B_c=M_cA_c.
\]

### Proof

Two values `beta_c` and `beta'_c` are observationally indistinguishable after
allowing arbitrary competing-cause and nuisance coefficients precisely when

\[
A_c(\beta_c-\beta'_c)=0.
\]

The query is invariant over every such indistinguishable pair precisely when

\[
B_c(\beta_c-\beta'_c)=0
\]

for every vector in `ker(A_c)`.  This is the kernel-inclusion condition.  In
finite-dimensional linear algebra, `ker(A_c) subseteq ker(B_c)` is equivalent to
every row of `B_c` lying in the row space of `A_c`, hence to the existence of
`M_c` with `B_c=M_cA_c`.

For `B_c=I`, the complete coefficient is identifiable if and only if

\[
\operatorname{rank}([N_c,S_c])
=\operatorname{rank}(N_c)+\dim(\beta_c).
\]

## Why intervention changes matter

Under one action, several cause signatures can occupy the same response
subspace. Stacking interventions can rotate or scale physical-state, material,
and contact responses differently while observation bias remains fixed and a
source-local discrepancy fails to transport. The implementation reports:

- source/single-intervention status;
- joint status;
- the identifiable fraction of the registered query;
- pairwise canonical correlations and principal angles;
- leave-one-intervention-out losses; and
- every smallest intervention subset that identifies the query, up to a frozen
  search limit.

This makes the output constructive: it states not only that a cause is
confounded, but which intervention changes are sufficient to separate it.

## Controlled falsification result

The checked-in controlled study registers five scalar causes. All five have the
same response under the source action, but distinct multi-intervention
signatures. With 10,000 frozen Monte Carlo trials:

- source-action cause classification is 20.00%, i.e. chance for five causes;
- all five causes are identifiable after intervention changes;
- multi-action attribution achieves 100.00% classification in the registered
  signal-to-noise regime;
- confirmation RMSE falls from 0.4644 to 0.0538, equal to the oracle cause;
- a wrong-action transport control has RMSE 0.6406; and
- declaring a nuisance exactly aligned with the material signature changes the
  material result to `confounded` with residualized rank zero.

The result identity is
`edebf0f9720cf56f08ac62307bec8e34f848284f14363186bc6b8824bf5cdea8`.

## Public real-data continuation

The first real-data test should use the complete Tracking Cloth data because its
maintained model already exposes distinct operational hypotheses:

- `last_residual`: persistent readout/model discrepancy;
- `nominal_state_injection`: physical state correction;
- `map_physics` / `bayesian_physics`: state plus material-parameter correction;
- `nominal_physics`: unchanged physical fallback.

A development audit can fit on shaking and test transport to twisting.  The
claim-bearing confirmation should use the three self-collision repetitions:

1. repetition 1 fits candidate response signatures;
2. repetition 2 freezes cause selection and the intervention set;
3. all repetition-3 predictions are jointly sealed;
4. repetition-3 outcomes are opened once for scoring.

The primary contrast is not candidate versus persistence.  It is the selected
physical cause versus `last_residual`, together with wrong-intervention,
wrong-material, temporal-shift, and marker-identity controls.  A physical-cause
claim requires both nonlinear replay closure and held-intervention transport.

## Claim boundary

A passing certificate proves identifiability only relative to the registered
finite cause family, supplied local signatures, nuisance model, interventions,
query, whitening, and tolerances.  An omitted nuisance can still imitate a
cause.  The certificate does not establish a unique data-generating mechanism,
global nonlinear identifiability, unseen-object transfer, calibrated predictive
uncertainty, safe control, deployment safety, or state of the art.
