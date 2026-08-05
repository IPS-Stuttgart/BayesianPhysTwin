# Deform360 conservative two-view tactile gauge validation

## Question

Can the source-locked three-camera MotionCrafter panel support a conservative
two-view tactile metric gauge on a fresh calibration object when unknown
cross-view correlation is forbidden from increasing precision?

This is an observation-artifact admission test. It does not authorize a state
update, opening calibration prediction scores, or a confirmation or SOTA claim.

## Selection and information boundary

The prior three-camera validation on `036-napkin-cloth` ended in exact fallback
because only two of its three provider cameras passed the frozen target-free
visibility gate. This protocol changes the model class and therefore selects a
fresh object rather than weakening that failed gate in place.

Using only the locked calibration order and bimanual metadata, the source rule
is:

> Choose the first unopened bimanual calibration object after the prior
> validation object.

The selected case is `153-cake`, source episode 5 and processed episode 0. The
permitted causal prefix is `[0,42)` and untouched future is `[42,66)`. The
provider panel remains the three already-generated Stage 1 cameras. No provider
array, image pixel, tactile value, or prediction score from this case was read
before this protocol and its all-camera robot lock were frozen.

## Conservative two-view change

The scientific held-prefix errors remain unchanged: median error at most 5 mm
and 90th-percentile error at most 15 mm for both direct and swapped tactile
assignments. A camera is eligible only with full assignment coverage and a
minimum 64 px image margin. The selected pair must be separated by at least 30
degrees.

Relative to the old three-view gauge, the two-view arm is deliberately less
confident:

- per-camera covariance floor increases from 5 mm to 10 mm;
- a separate 10 mm shared-bias floor is added;
- cross-view covariance uses a conservative union that Loewner-dominates every
  input covariance;
- two correlated cameras never create precision gain;
- either assignment failing produces exact fallback.

The candidate scope is restricted to the three existing parent-provider
cameras, so this experiment consumes the archived Stage 1 v6 bundle and does
not rerun MotionCrafter.

## Compatibility and execution order

Before freezing this case, the old three-view gauge was replayed under revision
`4f7a1572`. After removing only runtime provenance fields and the derived
artifact ID, the complete scientific payload was exactly equal to the frozen
old result. This establishes behavior compatibility for the old schema.

Execution remains staged:

1. recover the robot prefix from all 32 calibrated cameras;
2. freeze and run the unchanged direct/swapped tactile geometry stage;
3. compute target-free contact visibility only on the three provider cameras;
4. apply the frozen two-view selection policy;
5. freeze and evaluate the metric gauge;
6. proceed to carrier admission only if every source gate passes.

Every failure terminates in exact baseline fallback. Held-v8, confirmation
payloads, prediction scores, and future frames remain outside this protocol.
