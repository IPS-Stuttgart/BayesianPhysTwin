# Deform360 Active-Query Feasibility V10

## Question

Can the frozen physics-guided active-query planner supply eight moving graph
identities with independently supported frame-zero query locations before a
tracker or Bayesian state update is run?

This is a source-only feasibility test. It is not an accuracy experiment and
cannot support a state-of-the-art claim.

## Why This Test Comes First

Earlier fixed active/sentinel schedules failed mainly because their complete
query budget could not be materialized. Running another expensive tracker
before checking the replacement planner would repeat that failure mode.

V10 therefore evaluates only the front-end contract:

1. select four cameras using frame-zero physical coverage and angular
   diversity;
2. associate each frame-zero graph projection with local metric depth and the
   released object mask;
3. retain an identity only when at least two selected cameras provide an
   association with probability at least 0.5;
4. require at least 2 mm of action-conditioned physical motion;
5. greedily choose eight identities using predicted motion, visibility,
   low-rank graph information, and spatial diversity.

Association probability decides whether a query can be placed. It is not
perception reliability and is not used as a state-update weight. No state
innovation exists in this audit.

## Locked Source Gate

The panel contains eight already-open source cases whose physical artifacts are
hash-bound in
`configs/sota/deform360_active_query_feasibility_source_v10.json`.

The gate passes only if at least 6 of 8 cases supply all eight frame-zero
queries. A short budget is an abstention. Thresholds, camera support, or query
count must not be relaxed after seeing the source result.

Passing this gate authorizes only a new, separately frozen tracker-provider
protocol. It does not authorize a state update, future scoring, held-v8 access,
or opening the sealed V1 target cohort.

## Information Boundary

Allowed inputs:

- the sealed source physical rollout through frame 57;
- the physical graph basis;
- calibrated camera geometry;
- frame-zero depth and object masks;
- known robot action already used by the physical rollout.

Forbidden inputs:

- RGB, depth, or masks after frame zero;
- any tracker output;
- manual or hidden future identities;
- candidate state updates or future metrics;
- held-v8 artifacts, processes, identities, or outcomes;
- the sealed V1 target cohort.

Each result stores the selected camera panel, candidate and selected graph
identities, independent camera support counts, exact numeric array hashes, the
camera-certificate digest, and the physical/protocol file digests.
