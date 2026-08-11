# Deform360 covariance-only target v1.4

## Why this amendment exists

Independent source-only review reproduced two blockers in the v1.3 provider
before any target media, endpoint, or outcome was decoded. First, arbitrarily
distant rows could obtain soft graph assignments and empirical update counts.
Second, an ambiguous midpoint used `y - E[x]` for every candidate, creating a
false zero residual for both identities. V1.4 corrects those implementation and
provenance contracts while retaining the v1.2 roster and every v1.3 scientific
choice: point mean, covariance donor, horizon scales, support thresholds,
fallback, estimands, no-replacement rule, and claim boundary.

The v1.3 lock remains immutable and is superseded before target decode. No sealed
selection, acquisition, verification, quarantine, or exclusion artifact is
rewritten.

## Admissible association

Every visual row uses the frozen v5 geometry-only association rule with at most
four candidates, a 10 mm scale, a 40 mm maximum nearest-candidate distance, and
entropy strength 0.5. Rows beyond the distance bound have zero association
support. A graph identity is observed in a frame only when one candidate
contribution reaches 0.05. Thus a distant but internally coherent triangulation
cannot manufacture endpoint updates.

Geometry controls association probabilities and admission only. Stored prior
perception reliability uses residual-independent source confidence, mask distance,
overlap disagreement, and a conservative `1/sqrt(contributor_count)` factor when
overlap correlation is unknown. Increasing the PhysTwin innovation does not lower
that cue-derived reliability for an otherwise admitted row. Canonically equivalent
duplicate evidence cannot reduce covariance or raise reliability, and repeated
`(camera_id, window_id)` keys are rejected.

## Conditional innovations and uncertainty

For candidate identity `j`, the innovation is `y - x_j`, weighted by its
assignment probability. Source point covariance in square metres and
assignment-mixture spread remain in the history artifact. An ambiguous midpoint
therefore produces opposite-signed candidate innovations and nonzero covariance,
not a shared zero residual.

Admitted innovations are passed unclipped to a provider-specific heteroscedastic
robust endpoint path. For endpoint component `k`, the inlier observation
covariance is

```text
R_tn,k = (R_tn + sigma_obs,k^2 I) / reliability_tn.
```

The outlier component applies its registered variance multiplier to this same
matrix. The full `3x3` covariance enters the Gaussian mixture likelihood and
Kalman update, then remains anisotropic through component averaging and horizon
process-noise propagation. Cue reliability scales covariance exactly once. The
state innovation affects only the robust inlier/outlier responsibility and is not
reused to set reliability or clipped beforehand.

This provider-specific path leaves the stable public endpoint API and all prior
experiments byte/behavior compatible when they do not call the new provider.

## Split and baseline provenance

The residual history is bound to the observation-split artifact, exact provider
camera panel, provider reconstruction digest, and canonical ordering of unique
windows. The forecast carries that split and the independent scoring-reconstruction
digest. Reconstruction IDs must be lowercase SHA-256 values.

The registered `last_residual` mean digest is supplied independently and verified
against the exact input array. It is no longer derived tautologically from whatever
array a caller provides. Candidate mean dtype, shape, C-order bytes, and digest
remain identical to the registered mean, including every exact fallback.

## Source-only gate

The regenerated synthetic gate covers the complete v1.3 contract plus the two
reproduced blockers. Two 17 m-distant rows yield zero valid entries, zero endpoint
updates, and exact fallback covariance. Two midpoint rows between identities at
0 and 10 mm yield +5 mm and -5 mm conditional innovations with retained assignment
spread that survives into the forecast. Additional regressions prove that larger
metric row covariance increases or preserves forecast covariance, lower cue
reliability cannot make the forecast more confident, and gross innovations are
robustified once. They also cover duplicate correlation, canonical window order,
split/reconstruction binding, baseline digest verification, prior-only handling,
PSD covariance, byte-identical means, and exact fallback.

This is source-only implementation evidence. It authorizes no target, official
benchmark, calibration, point-accuracy, Causal4D, Prob4D, or state-of-the-art
claim. Target decode remains prohibited until the v1.4 lock and its source-only
verification are committed and independently reviewed.
