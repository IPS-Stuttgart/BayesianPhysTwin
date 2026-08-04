# Controlled Prob4D-to-BayesianPhysTwin decision study v1

## Scientific question

This study asks whether Prob4D's joint-gauge observation factors and persistent
causal point identities improve a guarded Bayesian physical-state query relative
to the unchanged physical fallback and simpler visual interfaces.

The experiment is calibration/target separated. Target seeds are disjoint from
calibration seeds, the deployment guard is frozen before target outcomes are
opened, and every rejected update must reproduce the physical fallback exactly.
A valid negative result is retained without target-informed retuning.

## Why this is the admissible next study

Static endpoint persistence, one-dimensional action scaling, and the first
three-case action-propagated state diagnostic are already exhausted negative or
non-promotable results. Reopening the previously inspected PhysTwin cohorts would
violate their recorded stop rules. The present study therefore uses fresh
controlled physical-object/session groups to test the new cross-repository
factor and nuisance-inference path before consuming a genuinely fresh physical
cohort.

## Compared methods

The frozen protocol compares:

- `B0_physical_fallback`: exact zero visual update;
- `B1_naive_last_frame_state`: a simple last-frame visual state estimate;
- `P1_marginal_gauge_persistent`: persistent identities with rowwise gauge
  covariance marginalized into observation covariance;
- `P2_explicit_gauge_framewise`: framewise identities with explicit joint
  `Sim(3)` gauge nuisance;
- `P3_explicit_gauge_persistent`: the primary method, combining explicit joint
  gauge nuisance and identities repeated over several causal observation times;
- `P4_explicit_gauge_persistent_metric_anchor`: the primary method with a
  separately generated metric-gauge anchor.

The Prob4D side is exercised through real `ObservationFactorBundle`,
`GaugeEstimate`, `Sim3`, and sparse-stack implementations. BayesianPhysTwin uses
its real prior-aware gauge-and-bias solver. The explicit-gauge arms consume
conditional point covariance and the complete joint gauge prior exactly once.
The marginal arm is deliberately separate and cannot recover shared cross-row
or cross-window gauge dependence.

## Frozen information boundary

Forty-eight source/calibration groups fit one risk threshold per method. The
risk score may use only posterior query width, posterior nominal probability,
prior-aware identifiable fraction, query sensitivity, and fixed-point
convergence. It cannot use target truth.

The target panel contains 384 independently seeded groups across six registered
conditions:

1. nominal correlated observations;
2. coherent common-mode visual bias;
3. corrupted observation groups;
4. weak physical-state identifiability;
5. large gauge uncertainty; and
6. a mixed stress condition.

The statistical unit is the complete synthetic physical object/session group,
not a frame, window, factor row, or point.

## Registered decision

The primary `P3` arm passes only if all criteria hold:

- at least 10% equal-group deployed RMSE improvement over the physical fallback;
- the upper endpoint of the paired 95% group-bootstrap interval is below zero;
- harmful accepted updates are at most 5%;
- no registered scenario regresses by more than 2%;
- every rejection is exact fallback; and
- the explicit persistent arm is noninferior to the marginal persistent arm by
  the registered 2% margin.

Exit code 3 denotes a completed valid negative result. It must not be converted
into a red infrastructure failure or used to reopen the frozen target seeds.

## Reproducible self-hosted execution

`.github/workflows/prob4d-bpt-controlled-decisive.yml` runs on
`[self-hosted, Linux, X64, nvidia-smi]`. It checks out the BayesianPhysTwin head,
resolves the exact frozen Prob4D revision, verifies the protocol digest before
execution, runs focused regressions, executes all calibration and target groups,
independently recomputes the registered decision, verifies the evidence
checksums, and uploads the complete evidence directory.

The workflow has read-only repository permissions. A current private Prob4D
checkout must be available either through the registered repository credential
or as an exact local source revision on the self-hosted runner. It fails closed
rather than substituting a different producer revision.

## Claim boundary

This is controlled calibration/target-separated synthetic evidence. A passing
result authorizes—not replaces—a fresh physical-object or acquisition-session
experiment. It does not establish real-world Prob4D provider competence,
calibrated deployment uncertainty, BayesianPhysTwin physical benefit on an
independent physical cohort, or Causal4D intervention benefit.
