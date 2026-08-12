# Deform360 v6 source camera reuse

## Purpose

The frozen v6 public-source run sealed 100 predictions, but all ten objects
used technical fallback because a camera-local metric-gauge support failure was
handled as an object-wide provider failure. The source suffix was not opened.
This additive protocol repairs only that failure granularity and emits a new,
versioned source prediction batch before any outcome access.

Deform360 contains real-world robot manipulation recordings. Its released RGB,
camera calibration, and robot-state streams are existing physical
measurements. This recovery therefore requires no new recording and no human
approval. A person cannot choose cameras, waive a failed check, or change a
threshold.

## Frozen inputs

The amendment binds all inputs by content identity and file SHA-256:

- the successful v6 source run, its compact GitHub artifact, execution lock,
  source plan, prediction batch, prediction receipt, and execution receipt;
- the 324-camera visual-production result and its existing integrity-bound
  disjoint MotionCrafter archives;
- the target-free robot-metric prefix plan and the exact materialization result
  produced before source-suffix access.

The similarly named later metric audit is not interchangeable with the bound
materialization result. Its implementation revision and content identity are
different, so the amendment rejects it.

## Deterministic recovery

1. Audit each originally attempted camera independently with the unchanged
   32-pixel cluster definition and eight-cluster threshold.
2. Trigger recovery only when fewer than two original cameras pass.
3. Consider only cameras already present in the locked metric plan, excluding
   endpoint-reserved and previously attempted cameras.
4. Rank by independent-cluster count, qualifying causal-frame count, projected
   robot-point count, then camera ID. Reuse at most four cameras.
5. Reuse only the exact disjoint baseline named and hashed by each existing
   prediction manifest. No provider inference is run. Prob4D is explicitly
   unused; this is not the decoded-uniform overlap-fusion arm.
6. Re-audit all attempted cameras independently. Admit an object only with at
   least two passing cameras; otherwise use exact `B0` physical fallback.
7. Seal a new 10-by-10 prediction panel and a content-addressed execution
   receipt before any source endpoint or outcome is named.

The original batch remains immutable. No threshold, candidate rank, physical
baseline, discrepancy rule, or target cohort changes.

## Information and claim boundary

The workflow may read public source prefixes and integrity metadata. It may not
read the development suffix, endpoint plan, confirmation payload, target
payload, or any outcome. It does not run endpoint planning or scoring.

The resulting receipt explicitly records that no new measurement, human
approval, provider inference, source suffix, confirmation payload, or target
outcome was used. It authorizes neither independent confirmation nor a claim.
Only a later, separately registered source scoring gate may decide whether a
larger independent public evaluation is justified.

The protected execution is one-shot. Once its durable run marker is created,
a technical failure is retained in a content-addressed receipt and is not
replaced or retried. The amendment has one durable run root across source
revisions. A fixed `O_EXCL` ownership file claims it atomically before the run
root is created; the bounded failure step may finish only the matching claim.
Cleanup never writes into an existing, skipped, or differently owned run.
