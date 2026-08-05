# Prob4D real-camera validation v1

## Purpose

The controlled Prob4D-to-BayesianPhysTwin experiment established a large
synthetic mechanism result, but it did not establish that real camera factors
are competent. This experiment tests the same central distinction on public
PhysTwin recordings:

- marginalizing shared gauge uncertainty independently in every row; versus
- retaining the complete cross-window `Sim(3)` gauge as an explicit nuisance.

The endpoint is a physical-state correction at the final causal camera frame,
scored on real manually tracked material identities that are reserved at frame
zero and excluded from direct camera-to-graph association.

## Information boundary

The first two complete, independently decoded MotionCrafter windows are the
only prediction payloads opened. Their last source frame is the query frame.
No later MotionCrafter, VGGT, point-cloud, or manual-track frame enters candidate
construction. The released manual-track pickle is monolithic, so this is a
retrospective dependency-controlled test rather than a prospective custody
claim. The implementation writes and hashes each candidate before evaluating
the query-frame manual coordinates.

The camera path uses rolling within-window scene-flow tracklets. Tracklet and
graph-association probabilities enter as generalized-Bayes likelihood power,
not as perception reliability. Prior reliability uses only overlap
disagreement. Physical innovation is processed once by the robust mixture
likelihood. Assignment-mixture spread is added to conditional metric covariance.
Dense rows in one absolute frame share a fixed effective-sample cap.

## Comparisons

1. `B0_physical_fallback`: unchanged released PhysTwin state.
2. `P1_marginal_gauge_persistent`: persistent identities, row-marginal gauge.
3. `P2_explicit_gauge_framewise`: query-frame identities, explicit joint gauge.
4. `P3_explicit_gauge_persistent`: persistent identities, explicit joint gauge.

The real-camera guard thresholds are copied unchanged from the controlled
calibration. Rejected candidates must reproduce the physical fallback exactly.

## Claim boundary

The 19 interactions were examined by earlier MotionCrafter/Prob4D work. The run
is therefore a retrospective real-camera transfer test. A positive result can
justify a fresh object/session protocol; it cannot itself confirm deployment
calibration, future prediction, Causal4D benefit, or state of the art.

## Result

The implementation was locked at `231737ddeedc8a9c258e960f6e827d12b80008ed`
before the complete 19-case run. Prob4D was fixed at
`364f216c14f7770c1b360bb1b836b11ecf0c18b8`. All 19 cases scored, no case was
replaced, and there were no technical failures.

| Method | Raw RMSE | Deployed RMSE | Deployed change | Raw wins | Accepted | Accepted 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Physical fallback | 6.899 mm | 6.899 mm | 0.00% | 0/19 | 0/19 | n/a |
| Marginal gauge, persistent | 7.054 mm | 6.942 mm | +0.62% | 7/19 | 11/19 | 37.3% |
| Explicit gauge, framewise | 6.899 mm | 6.899 mm | 0.00% | 0/19 | 0/19 | n/a |
| Explicit gauge, persistent | 7.035 mm | 6.899 mm | 0.00% | 3/19 | 0/19 | n/a |

Positive percentages in the deployed-change column denote regression. The
marginal-gauge arm admitted two harmful updates among eleven accepted cases
(18.2%), and its paired deployed-minus-physical 95% interval was
`[-0.069, +0.176]` mm. The framewise arm had no query-frame row that passed the
frozen graph-association support threshold. The primary explicit persistent arm
was inference-admissible in 17 cases, but every risk score exceeded its
controlled-calibration threshold; all 19 deployments therefore reproduced the
physical fallback exactly.

The registered decision is
`do-not-advance-from-retrospective-real-camera-transfer`. The cohort-completion
and exact-fallback checks passed. The improvement, paired interval, accepted
case count, and coverage checks did not pass. The bound report SHA-256 is
`63d933e01d4f26c186ed78c086b06f30a97d8b1badbca751418f24b91d3f5f99`.

## Interpretation

The controlled 91.33% RMSE reduction remains valid evidence that explicit
correlated-gauge inference can work when its observation model is correct. It
does not transfer through the present real MotionCrafter camera factors. The
explicit model is safer than marginalization here because it abstains instead
of overcounting uncertain gauge evidence, but abstention alone is not an
accuracy improvement. The marginal arm's low coverage and harmful-admission
rate show that its controlled uncertainty scale is not calibrated for these
recordings.

This result does not justify a larger preregistered run. The next admissible
step is source-only observation-provider work with genuinely independent
evidence against coherent camera bias, followed by a new lock. Candidate
directions include material-identity tracks that pass a causal prefix
competence gate or an independent sparse depth, LiDAR, or tactile anchor. The
current 19 cases must not be reused to tune that provider.
