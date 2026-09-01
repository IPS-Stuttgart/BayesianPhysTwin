# DLO3 residual coefficient transfer to DLO4 and DLO5

## Scientific question

The frozen DLO4/DLO5 study tests whether the complete **procedure** replicates:
each object receives its own all-train physical model and its own source-fitted
local residual model, while the recipe and hyperparameters remain unchanged.
That is already a strong object-level replication, but it does not show that the
fitted discrepancy itself is reusable.

This secondary experiment asks a sharper question:

> Do local-residual coefficients fitted only on DLO3 improve DLO4 and DLO5
> target predictions without any DLO4/DLO5 residual refit?

A positive result would support a bounded coefficient-level transfer claim
across deformable-object operators. The object-specific physical backbone still
uses the matching DLO's 56 training trajectories; only the residual coefficients
are transferred unchanged. This is therefore not transfer of the complete twin.

## Registration boundary

The complete arm, three-seed aggregation, shrinkage, gate, and diagnostics were
frozen while protected Actions run `33361441865` was still in its target
prediction stage. The DLO4 and DLO5 source-gate outcomes were already open, but
no target scores had been opened and no target prediction or outcome was used to
design this experiment.

The experiment is consequently classified as an **outcome-blind pre-score
secondary diagnostic**, not as the original prospective primary endpoint.

## Frozen arms

For each of DLO4 and DLO5, the comparison contains:

1. the matching-object all-train physical prediction at update 6400;
2. the protected study's matching-object source-qualified residual prediction;
3. unchanged DLO3 residual models fitted under seeds 42, 43, and 44; and
4. the arithmetic prediction mean of those three DLO3 no-refit arms.

All transferred arms retain shrinkage `0.25`. There is no object-side residual
refit, target-dependent seed weighting, seed selection, shrinkage selection,
threshold selection, case replacement, or retry.

## Registered decision

Each target DLO must independently satisfy all of the following:

- at least 1% mean coordinate-L1 improvement over its matching physical model;
- at least 8 of 14 complete-trajectory wins;
- no trajectory candidate/physical ratio above 1.10; and
- positive mean improvement for at least two of the three DLO3 seed models.

The overall decision is positive only if **both DLO4 and DLO5 pass**. The
reported equal-DLO aggregate cannot override failure on either object.

## Support-shift diagnostic

The runner standardizes every DLO4/DLO5 causal feature with each DLO3 model's
frozen feature location and scale. It reports absolute-z quantiles, the fractions
above 3, 5, and 10, and the maximum absolute z value. These diagnostics explain
whether transfer succeeds or fails under feature-support shift; they cannot
alter the registered gate.

## Information order

The execution is bound to the exact successful parent workflow, protocol,
joint prediction seal, per-DLO prediction seals, and three DLO3 model hashes.
It then:

1. writes a preflight record without semantically reading parent target scores;
2. writes the complete transfer method seal;
3. only then opens the parent score result, target prediction archives, and
   target trajectory payloads;
4. reproduces the protected study's matching-object point metrics exactly;
5. evaluates all three DLO3 seeds and the equal-seed arm; and
6. retains trajectory-level results, feature-support diagnostics, reports, and
   SHA-256 identities.

## Implementation provenance

The three Python implementation files were rewritten by the repository's exact
Ruff 0.16.5 check-and-format configuration. The self-deleting formatting helper
was removed in the same commit, so this experiment adds no formatter workflow or
additional persistent execution surface. Numerical validation remains separate
from formatting and is performed by the ordinary pull-request test matrix.

## Claim boundary

A positive decision supports unchanged **DLO3 local-residual coefficient**
transfer to the exact released DLO4 and DLO5 target operators, on top of
separately fitted matching-object physical backbones. It does not establish:

- transfer of the complete physical twin;
- arbitrary-object or arbitrary-topology generalization;
- backend independence by itself;
- physical-parameter identification;
- deployment safety; or
- universal state of the art.

Combined with a positive DEFORM-to-PyElastica no-refit result, it would support
the stronger but still bounded conclusion that a component of the fitted
missing dynamics transfers unchanged across both object operators and an
independently implemented physical backend.
