# Deform360 covariance-only target v1.3

## Why this amendment exists

The sealed names-only plan found no official processed Deform360 annotations for
the 24 locked physical objects. The study therefore cannot report official
Deform360 track or Chamfer parity. Protocol v1.3 keeps the v1.2 target roster,
point mean, covariance donor, horizon scales, fallback, and no-replacement rule
unchanged, but freezes the executable provider and the custom evaluation before
any target media are decoded.

The claim remains narrow: does a source-frozen covariance donor improve a proper
predictive-distribution score while leaving every predicted point byte-identical
to `last_residual`? This is a custom fresh-object calibration study, not an
official benchmark or point-accuracy result.

## Causal history adapter

`independent_endpoint_v1` requires residual history with shape `(T,N,3)` and a
validity mask with shape `(T,N)`. The new adapter builds that history over every
permitted causal prefix frame in Deform360 world coordinates. Positions use
metres and covariance uses square metres. Material identity order is shared by
the physical prefix, residual history, future mean, covariance, and scoring
events.

Missing identity-frame observations stay invalid and are stored as zero. They
are never nearest-filled and then counted as evidence. This intentionally differs
from the frozen v5 `last_residual` point estimator, which spatially completes one
endpoint for use as a mean.

Every visual row is associated with the frozen v5 geometry-only rule: at most
four candidates, a 10 mm association scale, a 40 mm maximum nearest-candidate
distance, and entropy strength 0.5. Rows beyond the distance bound have zero
association support. A graph identity is observed in a frame only when one
candidate contribution reaches 0.05. Arbitrarily distant triangulations therefore
cannot manufacture endpoint updates.

Association and perception reliability are separate quantities. Geometry controls
candidate probabilities and admission; it never lowers the stored prior perception
reliability. Reliability uses only residual-independent source confidence, mask
distance, overlap disagreement, and a conservative `1/sqrt(contributor_count)`
factor when overlap correlation is unknown. Duplicate windows cannot increase the
stored reliability or reduce the retained row covariance.

For candidate identity `j`, the innovation is `y - x_j`, weighted by its assignment
probability. The observation covariance retains the source covariance in square
metres and the assignment-mixture spread, so an ambiguous midpoint produces
opposite-signed candidate innovations instead of a false zero residual. Admitted
innovations are not clipped: the source-frozen robust endpoint mixture handles each
innovation exactly once.

The endpoint donor's registered process/observation-noise grid remains the sole
likelihood model in this study. Per-row metric covariance and cue reliability are
preserved in the history artifact for audit and control admission, but are not
silently combined with that already calibrated likelihood. Testing a
heteroscedastic endpoint filter is a separate method change and is not part of this
frozen target protocol.

## Empirical support and fallback

A material identity supplies empirical covariance only after at least two valid
prefix updates. A case admits the donor only when at least two prefix frames have
observations and at least half of its identities meet that update threshold.

`infer_model_averaged_endpoint` can return prior/process covariance for an
identity with zero observations. V1.3 labels that output `prior_only` and does
not treat it as empirical donor evidence. Unsupported identities retain exact
fallback covariance. If the case-level gate fails, both mean and covariance are
exact fallback bytes.

The point mean is never taken from the covariance provider. Before scoring, the
candidate and registered `last_residual` arrays must match in dtype, shape,
bytes, and content digest.

The registered reference-mean digest is supplied independently to the forecast
builder and verified; it is not computed from whichever array the caller happens
to pass. A history is likewise bound to the observation-split artifact, exact
provider camera panel, provider reconstruction digest, and canonical ordering of
unique `(camera_id, window_id)` inputs. The forecast carries the same split and
the independently constructed scoring-reconstruction digest.

## Custom outcome separation

Available camera names are ranked by a source-frozen SHA-256 rule using only the
object-session hash and camera ID. Even ranks form the provider panel and odd
ranks form the scoring panel. Each panel contains at least two cameras. The two
panels are disjoint, and their reconstructions must be built as distinct
artifacts without shared pixel, depth, or reconstruction products.

This separation prevents a shared camera or reconstruction bias from making its
own covariance appear calibrated. The partition cannot change after any target
decode, reconstruction, or score.

## Registered estimands

Each event is one 3D material identity at one untouched future frame. Candidate
covariance includes the registered 5 mm observation floor. V1.3 reports:

- 3D marginal Gaussian negative log score;
- 3D marginal NEES, whose calibrated expectation is three;
- 90% marginal Gaussian ellipsoid coverage;
- 90% marginal ellipsoid volume; and
- the same summaries by early, middle, and late horizon.

Events are averaged within physical-object-session first, then sessions receive
equal weight. No cross-identity joint covariance is available, so no joint
energy-score interpretation is registered.

## Source-only gate

The synthetic dry run passed before target processing. It checks history and
validity alignment, metre and square-metre units, world coordinates, PSD
covariance, deterministic camera partitioning, distinct reconstruction
artifacts, byte-identical point means, prior-only handling, and exact case-level
fallback. It also reproduces the two pre-decode blockers: 17 m-distant rows yield
zero updates and exact fallback, while midpoint ambiguity yields opposite-signed
candidate innovations with retained mixture covariance. It is implementation
evidence only.

The full lock is
`protocols/locks/deform360_covariance_only_target_v1_3.json`; the dry-run artifact
is
`results/science/deform360_covariance_only_target_v1/provider_source_dry_run_v1.json`.

## Remaining boundary

No target media, robot/tactile arrays, geometry, tracks, predictions, or outcomes
have been opened. The next permissible step is names-only materialization of the
24 camera partitions, followed by independent provider/scoring reconstruction
and prediction sealing. All 24 locked sessions remain in the denominator,
including technical failures.
