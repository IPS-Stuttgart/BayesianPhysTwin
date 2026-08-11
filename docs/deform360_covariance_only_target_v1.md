# Deform360 covariance-only target v1

## Purpose

The opened 22-case evidence supports a narrow Bayesian claim: endpoint-model
covariance can improve a proper predictive-distribution score even when no new
point mean beats exact last-residual persistence. This protocol tests that claim
on previously untouched Deform360 physical objects.

The candidate changes covariance only. Its point mean is byte-identical to
`last_residual`; its covariance is donated by `independent_endpoint_v1` and
multiplied by `8`, `16`, and `16` in the early, middle, and late horizons. These
numbers multiply covariance, not standard deviation. If no causal residual is
available, the unchanged physical prediction remains the exact fallback.

## Target selection

All previously opened, reserved, selected, or technically dispositioned objects
are represented by a hash-only exclusion artifact. From the remaining official
object names, the protocol fixes 16 candidates per object stratum before reading
metadata. Only `metadata.json` may then be opened.

The final target contains 24 distinct physical objects and one session per
object. It has exactly two sessions in every cell of:

- sheet versus volumetric object naming stratum;
- unimanual versus bimanual interaction; and
- elevation, planar/contact, versus shape-change action family.

If the fixed candidate panel cannot satisfy the factorial design, the protocol
stops before target payload access. Once the roster is frozen, missing streams,
missing released annotations, reconstruction failures, and unsupported sessions
remain in the denominator and are never replaced.

## Information boundary

Before roster lock, code may read only official object names, selected
`metadata.json` files, repository filenames/sizes/digests, and hash-only prior
exclusions. It may not decode camera media, load tactile or robot arrays, open
geometry or track annotations, run reconstruction or tracking, visualize a
target, or inspect any future score.

After roster lock, exact payload bytes may be downloaded and rehashed into a
quarantined cache. Released Deform360 annotations are preferred where they exist;
their absence is a retained target disposition, never a reason to substitute a
different object.

## Evaluation

The primary endpoint is the equal-object mean Gaussian negative-log-score
difference between the covariance-only candidate and `last_residual`, using the
registered common 5 mm observation-noise floor. Physical object sessions are the
independent resampling units. Coverage, interval width, NEES, energy score, and
horizon-resolved behavior are secondary outcomes. Track and Chamfer predictions
must remain exactly identical because the mean is unchanged.

This protocol cannot support a point-accuracy, physical-state, Causal4D, Prob4D,
or state-of-the-art claim. A paper claim requires the independently locked target
gate to pass with complete 24-session failure accounting.
