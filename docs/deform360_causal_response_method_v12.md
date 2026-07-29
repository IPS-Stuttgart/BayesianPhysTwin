# Deform360 Causal-Response Direct-Depth V12

## Status

The method is implementation-locked before any fresh cohort is selected. It
is not yet a source experiment, a target result, or a state-of-the-art claim.
Fresh selection remains prohibited until an independently produced hash-only
manifest covers every object touched by held-v8 and all of its attempts.

This dependency is intentional. Selecting objects while one sealed campaign's
object identities are unknown would make a later "fresh-object" claim
unverifiable.

## Why V12 Is a New Information Contract

V4 planned queries from projected physical response and admitted zero of seven
source cases. V9 used fixed direct-depth schedules and admitted zero of seven.
V10 found target-free physical motion for only four of eight cases. V11
replaced motion with action support, but causal TAPNext++ retained only 5 of
64 endpoint identities and regressed badly against a nearly static
persistence baseline.

V12 does not change those thresholds and retry. It changes what constitutes
evidence:

1. wait for the earliest *observed* causal response inside the allowed prefix;
2. require released tactile contact and measured actuator displacement;
3. build direct metric endpoint observations independently in two disjoint
   six-camera panels;
4. use one panel to propose a sparse state/discrepancy update;
5. use the other panel only to test current-prefix transfer;
6. remove per-endpoint Sim(3) nuisance before state inference;
7. admit only nonrigid pairwise response that transfers across panels; and
8. otherwise return the selected physical-or-persistence baseline byte for
   byte.

The update therefore cannot pass because an action was merely commanded. It
must have observable object response, independent contact support, spatial
coverage, action alignment, cross-panel consistency, uncertainty consistency,
and held-out-prefix improvement.

## Statistical Boundary

Association probabilities come from candidate geometry and depth/mask support.
Prior reliability comes only from view redundancy and view scatter. The state
innovation changes neither quantity. Association remains a separate event in
the existing robust mixture likelihood, where the innovation magnitude is
processed once to obtain posterior inlier probabilities.

The two camera panels are unknown-correlation groups. Their pixels and cameras
are not multiplied as independent precision. Direct-depth views are fused
conservatively, effective evidence is capped, and a shared 5 mm bias variance
is retained. The validation panel never forms the update.

Pairwise distance changes remove rigid frame errors. A two-dimensional
nuisance span formed by physical pair distances at both endpoints removes a
time-varying global metric scale. The accepted proposal is then aligned to the
physical graph separately at birth and update using a bounded weighted Sim(3)
fit. This is deliberately conservative: global pose and scale changes are not
claimed as observed deformation.

## Candidate Belief

The proposal panel produces birth-anchored metric measurements with covariance
in square metres. Endpoint covariance, association-mixture spread, fit
residual, temporal unknown-correlation inflation, and shared-bias variance all
reach the recursive RBF belief.

The current implementation keeps the successful readout-discrepancy semantics:
it updates the future observable trajectory, not the internal Warp state. That
claim boundary follows the matched localization audit, where readout
persistence outperformed one state reset and constant-force injection.

An admitted innovation enters `robust_mixture_likelihood` once. Rejection at
the event, nuisance, support, or later regret gate leaves the selected baseline
bit-identical.

## Prospective Evaluation

After the complete exclusion union exists, select 12 fresh physical objects
using metadata only and one episode per object. Seal every V12 candidate and
fallback before opening disjoint hidden identities or future geometry.

The source result must improve object-balanced hidden-identity RMSE and
Chamfer distance by at least 5%, jointly win on at least 8 of 12 objects, keep
every object regression below 5%, and keep false-safe admissions below 10%.
A three-fold object-level cross-fit must also produce the registered negative
upper regret bound. Failure closes V12 without threshold changes.

Only a complete source pass permits a separately frozen target protocol. The
existing V1 and held-v8 cohorts remain unavailable under this method lock.

## Current Deliverable

The code now provides:

- a typed, checksummed causal-response admission artifact;
- disjoint-panel and causal-support enforcement;
- translation, rotation, and scale nuisance controls;
- metric covariance and residual-independent reliability;
- a robust recursive RBF candidate;
- physical-prefix and observation hash binding; and
- executable bit-exact fallback.

The remaining critical-path input is the held-v8 all-attempt hash-only
exclusion manifest. No GPU run or object selection is justified before it is
available.
