# Active decision acquisition for physical twins

A physical twin may be unable to identify its complete latent state while still
having enough information for one registered action. The complementary failure
mode is also common: the current evidence is insufficient, but one additional
camera, tactile contact, or diagnostic observation would resolve exactly the
ambiguity that matters to the decision.

`bayesian_phystwin.active_decision_acquisition_v1` adds the finite, exact
**certify-or-probe** layer for that setting.

## Contract

The caller supplies:

- finite physical hypotheses and their positive prior support;
- posterior masses over a registered physical query quotient;
- a finite action loss matrix;
- deterministic candidate probe partitions and positive acquisition costs;
- an optional simultaneous elementwise loss-error radius; and
- a separately declared worst-case-regret tolerance.

The module never constructs a within-class point belief. After an observed probe
history, each original quotient class contributes either its complete fixed
mass, no mass, or an arbitrary amount between those extremes. The exact
post-history pairwise loss gap is the resulting box-constrained
linear-fractional maximum. This gives the exact worst-case regret over every
complete belief compatible with the original quotient masses and the observed
probe outcomes.

Optional loss radii replace each nominal pairwise loss difference by its
simultaneous upper endpoint. They therefore provide a finite-data outer
certificate when the caller can justify the radii. The module does not estimate
or validate those radii.

## Active policy

For a modest registered probe set, exact dynamic programming returns the policy
with minimum worst-case acquisition cost. Every node either:

1. returns a tolerance-certified action;
2. acquires one selected probe and branches on its outcome; or
3. reports the task as infeasible under the available probes, requiring exact
   physical fallback or a different sensing modality.

The solver is exponential in the number of candidate probes and refuses an
oversized set. This is intentional: the exact implementation is a reference and
a mechanism verifier for small sensor portfolios. Larger portfolios can use a
registered approximation while retaining the exact node certificate.

The companion weighted set-cover solver finds the minimum-cost nonadaptive probe
set whose common refinement separates every currently confounded pair with a
different action-loss-difference signature. If no available probe separates one
such pair, global decision identification is impossible under the registered
sensor family.

## Controlled evidence

The registered controlled study uses 24 physical hypotheses, three optimal
action groups, two decision-relevant probes, and four nuisance probes. The
nuisance probes are more attractive to greedy hypothesis-entropy acquisition,
but they do not resolve the action efficiently.

| Method | Worst-case cost | Uniform expected cost |
| --- | ---: | ---: |
| Exact adaptive decision certificate | 2.000 | 1.333 |
| Greedy hypothesis entropy | 4.000 | 3.333 |
| Global decision-identifying set | 2.000 | n/a |
| Full-state-identifying set | 6.000 | n/a |

The adaptive policy uses one probe for the 16/24 branch whose action is then
identified and a second probe only on the remaining branch. Removing the second
decision probe leaves an indistinguishable pair with opposing optimal actions;
the policy reports infeasibility rather than inventing confidence.

The exact result is under
`results/science/active_decision_acquisition_v1/controlled-v1/`.

## Public-data route

The intended real-data companion is sequential normal-view acquisition on DOT
rope. Development must use already-open `R11--R20`; normal images and 3-D
outcomes for `R21--R70` remain outside this source stage. The earlier fixed
rank-six model is not reused: the routed CUT3R source result observed rank seven
on nine of ten sequences. Each camera must instead emit a rank-agnostic query
belief or prior-anchored query message, with unknown cross-camera dependence
handled conservatively.

The source study should compare certificate-directed camera acquisition against
best fixed camera, random order, hypothesis entropy, query-variance reduction,
and all-view reference. The primary outcome is camera cost at matched physical
decision quality. Complete sequence is the statistical unit. Provider outputs
must seal before 3-D marker truth is opened, and source completion cannot
implicitly authorize `R21--R30`.

## Claim boundary

The exact active certificate is conditional on the registered finite hypotheses,
prior support, quotient masses, deterministic probe model, action set, loss,
costs, and tolerance. It does not validate those objects, establish a learned
provider, identify a complete physical state, certify continuous control,
guarantee held-out transport, authorize deployment, or establish safety.
