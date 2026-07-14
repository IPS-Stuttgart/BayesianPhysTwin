# Deform360 Replication Source Backend V1

This milestone records the terminal source-only branch of the preregistered
six-object Deform360 replication. The locked object-level competence gate
failed, so no target prefix, target geometry, or target tactile was opened.

## Locked Inputs

- Replication protocol config SHA-256:
  `f0aab308345807b2183f653306a062d4ad0295584b6b283deb99d29b3c247934`
- Backend policy config SHA-256:
  `96e9a99d4b1052e97e53c28281af0af45b4c5cd3fee9b07dac06b722b460d478`
- Source-QA result SHA-256:
  `2d7f0f4be5d27af1c2d6abb87168e0bf1a07c335287a6d15541c266adf22290f`
- Dataset revision:
  `7fea8e20231a47641d1d2bc8791920ec4e62ec5e`
- Official PhysTwin commit:
  `2b6630528141b9cba5a7677c8b88b2129b4a8390`
- Fail-closed outcome implementation commit:
  `403d8c97132c6c17a33a89aa3a815d5a5e72bc8d`

The backend policy was locked and tagged as
`deform360-replication-backend-lock-v1` before any target-prefix access. It
requires every object to pass both at least 5% pooled source Chamfer improvement
over persistence and at least a 60% leave-one-source win rate.

## Result

The six-object target phase was not admitted. Zero of six objects passed.

| Object | Source outcome | Warp Chamfer | Persistence | Relative gain | LOO wins |
| --- | --- | ---: | ---: | ---: | ---: |
| `002-rope-silk` | pooled fit | 51.73 mm | 39.69 mm | -30.34% | 3/6 |
| `081-stripe-rope` | pooled fit | 37.88 mm | 38.31 mm | +1.13% | 3/6 |
| `085-scarf-cloth` | pooled fit | 79.43 mm | 37.00 mm | -114.66% | 0/6 |
| `083-blanket-cloth` | source-pooling failure | n/a | n/a | n/a | n/a |
| `092-squirrel` | source-geometry failure | n/a | n/a | n/a | n/a |
| `170-spider` | pooled fit | 61.89 mm | 24.30 mm | -154.72% | 0/6 |

Across the four objects with pooled fits, the diagnostic mean is 57.73 mm for
the official-Warp adapter and 34.83 mm for persistence, a -65.77% relative
gain, with 6/24 leave-one-source wins. This partial-cohort aggregate is
diagnostic only; admission is defined per object.

The blanket produced all six source grids, but no candidate satisfied the
locked p99 edge-strain limit on every episode. Episodes 8 and 9 had no eligible
candidate. The squirrel failed strict source geometry on episode 7 because
fewer than 90% of future hull observations were available. Neither failure was
repaired by changing thresholds, relaxing consensus, substituting an object,
or adding a fallback.

The typed source-stage failure path was added after these failures became
observable solely to record terminal rejection without fabricating a pooled
fit. It cannot admit an object, alter a score, or permit target access.

## Information Boundary

The sealed decision states:

- `target_prefix_access_permitted: false`;
- `target_future_access_permitted: false`;
- `target_prefix_read: false`;
- `target_future_geometry_read: false`;
- `target_future_tactile_read: false`.

The metadata-only boundary audit additionally confirms that none of the six
target episode directories exists in the aligned, observation, or fit staging
trees. Calibration streams were preprocessed operationally, but calibration
outcomes were not used for this source-backend decision.

## Claim Boundary

This is a negative feasibility result for the preregistered sparse graph
adapter driven by the official PhysTwin Warp simulator. Its filament, sheet,
and volumetric graphs are deterministically built from public multiview visual
hulls. It is not a dense instance reconstruction from the full PhysTwin
pipeline, and it is not a test of Bayesian state or parameter inference.

The earlier single-object `001-rope` source gate therefore did not generalize
to the selected six-object cohort. The result blocks target evaluation of this
backend; it does not establish that PhysTwin or Bayesian-PhysTwin fails on
Deform360 under a stronger reconstruction and registration interface.

## Archived Evidence

- `artifacts/source_backend_decision.json`: checksummed terminal decision;
- `artifacts/pooled-fits/`: four complete pooled and leave-one-source fits;
- `artifacts/source-grids/`: all 30 completed 200-candidate source grids;
- `artifacts/stage-failures/`: typed failures, logs, and squirrel mask evidence;
- `artifacts/contact-models/`: six source-fitted opening/contact models;
- `verification/`: environment, tests, lint, and target-boundary audit.

Decision result SHA-256:
`3603d273f6263dfe682631b9d9c72ae73b11ec2707a649f6d065cb6855e343a2`.
