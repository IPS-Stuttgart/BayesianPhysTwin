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
a mechanism verifier for small sensor portfolios. It memoizes each compatible-
hypothesis mask and remaining-probe subset exactly once and resolves equal-cost
plans by stable registered order. Larger portfolios can use a registered
approximation while retaining the exact node certificate.

The companion weighted set-cover solver finds the minimum-cost nonadaptive probe
set whose common refinement separates every currently confounded pair with a
different action-loss-difference signature. If no available probe separates one
such pair, global decision identification is impossible under the registered
sensor family.

The exact dynamic program and the minimum nonadaptive set are complementary
certificates. The first proves the least contingent sensing cost for the supplied
predictive prior and chosen worst-case/expected objective. The second proves the
least fixed sensor-set cost required to identify every represented decision.
Their gap quantifies the value of adapting later measurements to earlier
outcomes; neither objective requires complete state identification.

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

## Existing-data validation route

No new recording is required. The primary real-data companion is the existing
DEFORM DLO4/DLO5 virtual-sensing protocol: internal-node position and velocity
readouts are masked and revealed one at a time, while future internal-node
trajectories remain unavailable until every sensing path and action is frozen.
The practical decision-regret policy can be compared with the exact dynamic
program on a deliberately bounded probe subset, giving both an empirical result
and an optimality-gap certificate.

A complementary Tracking Cloth replay can treat already recorded marker prefixes
as diagnostic probes and the logged release configurations as terminal physical
actions. That study must preserve repetition-level source/confirmation custody
and must be described as resettable logged-action replay rather than online robot
control.

The central evaluation quantities are sensing cost, certified-action coverage,
realized regret, exact-fallback frequency, physical hypotheses remaining at the
time of action, and the ratio between practical and exact minimum sensing cost.
The desired mechanism is action certification while multiple physical states
remain compatible—not reconstruction of a unique state.

## Claim boundary

The exact active certificate is conditional on the registered finite hypotheses,
prior support, quotient masses, deterministic probe model, action set, loss,
costs, and tolerance. It does not validate those objects, establish a learned
provider, identify a complete physical state, certify continuous control,
guarantee held-out transport, authorize deployment, or establish safety.
