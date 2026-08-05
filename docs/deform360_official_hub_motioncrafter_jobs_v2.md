# Deform360 official-Hub MotionCrafter jobs v2

## Status

The calibration inference schedule is frozen before MotionCrafter inference or
calibration scoring. Its content ID is
`bba14f34b884dcc214273e54fb0a9bbb190acfc49633c54bd4e596665385ba22`.

The schedule contains 30 jobs: three pose-selected cameras for each of the ten
calibration objects. Every job reads exactly the 42 authorized source frames and
binds two independently decoded 25-frame windows with eight-frame overlap. No
frame at or after the exclusive causal cutoff can enter a prediction.

## Complete bundle

The causal Prob4D provider consumes only the independently decoded overlap
windows. Nevertheless, every job retains the disjoint and latent-linear control
products because Prob4D provider-v2 validates their complete deterministic seed
schedule before a calibration artifact can become claim-bearing. The remote
runner delegates those products to Prob4D's crash-safe runner and verifies all
member hashes.

One pinned model instance is reused across jobs. This changes startup cost only;
each job retains its own immutable run configuration, source-video digest,
output directory, and prediction manifest.

## Execution order

The lexicographically first frozen job is the smoke job. It must pass source
custody, model-source identity, prediction integrity, frame-lineage, and seed
checks before the remaining 29 jobs run. Infrastructure retries may only resume
the identical hash-bound job. Cases, cameras, windows, models, and seeds cannot
be replaced.

## Information boundary

This is a post-payload, pre-score calibration schedule. Integrity inspection of
generated provider artifacts is allowed. Calibration errors, policy fits,
confirmation payloads, target outcomes, and future frames remain unopened. The
artifact therefore establishes execution provenance only, not provider
competence, calibrated uncertainty, physical improvement, confirmation, or a
state-of-the-art result.
