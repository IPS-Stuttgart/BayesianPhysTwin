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

Results will be appended only after the protocol, implementation, and tests are
committed.
