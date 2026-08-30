# Controlled query-conditional competence experiment

## Purpose

This experiment isolates the proposed mechanism before any physical
confirmation is attempted. It asks whether a source-trained score can decide,
before a future trajectory is generated, when a query-specific contact-model
simulator may replace an exact nominal fallback.

The design descends from the user-supplied controlled active-causal bundle with
SHA-256
`6e1dc500f0f982827005d216d36c30e46051d61a8a80d100042c00d0ed5aa738`.
The new experiment asks a different question and uses a fresh implementation,
fresh partitions, and a separate one-shot confirmation roster. The source
instrument is
`src/bayesian_phystwin_experiments/controlled_query_competence_v1.py`.

## Outcome-unopened route

Each independently seeded episode contains exactly one query context:

`topology x future action x horizon x query functional`.

The topology is one of `rope`, `cloth`, and `soft_block`; there are seven
future actions, three horizons, and four dimensionless query functionals. A
weak `centre_pulse` screen is observed first. The screen updates a posterior
over four contact hypotheses:

1. nominal attachment;
2. shifted attachment;
3. compliant slip; and
4. shifted compliant slip.

For the preassigned future query, every contact-model simulator produces a
forecast without seeing future truth. The candidate is the simulator with
minimum posterior expected squared query disagreement. Model 0 is always the
exact fallback. If the candidate is model 0, the route falls back without
claiming coverage.

The implementation fixes, in this order:

1. the screen observation and posterior;
2. action, horizon, and query identity;
3. all four model forecasts;
4. candidate identity;
5. the complete pre-outcome risk feature vector; and only then
6. the nonlinear, parameter-perturbed future truth.

No true hypothesis, future state, future loss, harm label, or selected outcome
enters the risk feature function.

## Practical harm

All query values are dimensionless. Position queries are normalized by the
object's characteristic edge length; peak strain is already dimensionless.
For candidate loss `L_S` and fallback loss `L_B`, harmful use is

`L_S > L_B + 0.025`.

The margin corresponds to 2.5% characteristic-length RMSE or 2.5 percentage
points of peak edge strain. It was frozen from the scientific interpretation,
not selected on confirmation outcomes.

## Frozen partitions

There is one query per episode, so an episode is the independent group for the
exact binomial certificate.

| Partition | Episodes per topology | Seed base | Permitted use |
| --- | ---: | ---: | --- |
| Screen calibration | 96 | 202608100 | Prior and screen residual variance |
| Risk training | 192 | 202608400 | Logistic harm-model coefficients |
| Threshold selection | 384 | 202608700 | Largest eligible threshold on a fixed grid |
| Source gate | 384 | 202608950 | Independent pre-confirmation qualification |
| Confirmation | 512 | 202609300 | Exactly one claim-bearing run after a passing source gate |

Query assignment uses an independent random stream offset by `100000`. The
source stage cannot generate the confirmation partition. A later execution
request must bind the exact source result and explicitly authorize the single
confirmation attempt.

## Models and ablations

The primary score is deterministic L2 logistic regression over 32 pre-outcome
features: posterior mass, posterior uncertainty, query-projected model
disagreement, predicted harm and regret, action, horizon, query, candidate, and
in-population topology identity.

Three source-trained ablations use the same candidate, partitions, threshold
grid, harm definition, and fallback:

- context-agnostic posterior features;
- posterior uncertainty only; and
- query-projected model disagreement only.

Always fallback, always query-select, and the globally best fixed source model
are nonselective references.

## Gates

The threshold is the largest member of the frozen grid satisfying all of the
following on threshold-selection episodes:

- at least 100 accepted groups;
- at least 25% total coverage;
- at least 20 accepted groups for every query functional;
- exact one-sided 95% Clopper-Pearson harmful-use upper bound no greater than
  10%; and
- nonpositive mean selected-policy regret relative to fallback.

The independent source gate and the one-shot confirmation additionally require
the paired 2,000-resample 95% interval for selected-policy regret to lie below
zero. Every rejected query uses model 0. A failed source gate leaves
confirmation closed; a failed confirmation is frozen without retry or method
change.

## Source disposition

The exactly-once source execution passed all registered checks. The primary
full query-conditional arm selected threshold `0.2` before the independent
source-gate partition. It accepted 709 of 1,152 queries (61.55% coverage), with
50 practically harmful accepts (7.05%) and an exact one-sided 95% upper bound
of 8.84%, below the frozen 10% limit. Mean selected regret was `-0.0041345`;
its paired 95% interval, `[-0.0054021, -0.0029569]`, lay below zero.

The context-agnostic posterior and uncertainty-only ablations could not select
an eligible threshold. The full arm and the query-projected
model-disagreement-only arm both passed the source gate, so this stage does not
by itself establish that every contextual feature in the full score is
necessary. Detailed outcome evidence and its independent verifier are retained
in the private paper repository. The public hash-only receipt is
`evidence/controlled_query_competence_source_receipt_v1.json`.

The source stage authorized registration of one controlled confirmation
attempt. It did not open confirmation outcomes and did not authorize physical
confirmation.

## Scope boundary

Development auditing found that training on two topology classes and
certifying on an unseen rope topology did not preserve the 10% harm bound. The
registered claim is therefore intentionally in-population over the three
declared topology classes and independent episodes. Unseen-topology transfer
is not claimed, and the negative stress test must remain visible in the paper.

This controlled experiment uses no Prob4D data, physical outcomes, held-v8,
DLO4/DLO5, Deform360 target, or 4D-DRESS artifact. Even a passing result is
mechanism evidence only. Public physical confirmation remains a separate,
currently closed gate.
