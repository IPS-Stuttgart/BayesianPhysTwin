# Deform360 calibration visual-execution admission

## Purpose

The official-Hub calibration pipeline now has two independent, frozen metadata
artifacts:

1. the prepared-source inventory records the exact retained video, timestamp,
   tactile, robot, and alignment bytes for all ten calibration objects; and
2. the visual-production plan fixes the target-blind camera roster, nested frame
   windows, MotionCrafter/Prob4D seeds, dependence groups, call namespaces, and
   output locations.

Neither artifact alone proves that every planned visual job refers to the exact
retained media bytes that were inventoried. The visual-execution admission closes
that handoff before any MotionCrafter or Prob4D call.

## Contract

The admission independently validates both source artifacts and binds:

- the exact plan and inventory IDs, file SHA-256 values, and byte counts;
- the Stage-0 selection, visual-provider lock, successful calibration-source run
  record, and calibration-source result identities;
- the exact BayesianPhysTwin, Prob4D, MotionCrafter, calibration-source, and
  processing revisions;
- all ten physical objects and their sheet/volumetric strata;
- every planned object/camera job in canonical order;
- the selected 81-frame, prediction 76-frame, and prefix 58-frame ranges;
- object and view seeds, call namespaces, and dependence groups;
- each retained video and timestamp path, SHA-256 value, and byte count;
- the aligned frame count, camera dimensions, frame rate, and timeline identity;
  and
- collision-free output paths.

Each job receives its own content identity. The complete admission receives a
second content identity over the source artifacts and every job.

## Fail-closed behavior

Admission fails if any of the following occurs:

- object, episode, stratum, or camera substitution;
- camera omission, duplication, or order drift;
- selected, prediction, or prefix frame-window drift;
- video or timestamp path substitution;
- retained file digest or byte-count drift;
- selection, provider-lock, run-record, or source-result drift;
- duplicate JSON keys, non-finite JSON, Boolean schema aliases, or ID tampering;
- output-path, namespace, or view-seed collisions; or
- any attempt to overwrite an existing admission.

The plan and inventory files are opened through stable no-follow descriptors.
Their exact verified bytes are parsed after device, inode, size, modification
time, and change time are checked before and after reading.

## Command

```bash
python scripts/science/admit_deform360_calibration_visual_execution.py \
  --visual-production-plan calibration-visual-production-plan.json \
  --prepared-source-inventory prepared-source-inventory.json \
  --implementation-revision "$(git rev-parse HEAD)" \
  --output calibration-visual-execution-admission.json
```

Exit code `0` means the metadata-only handoff is admitted. Exit code `2` means a
source, identity, schema, information-order, or publication contract failed.

## Information boundary

The admission reads only the two content-addressed JSON metadata artifacts. It
does not open retained videos, timestamps, tactile arrays, robot state, geometry
annotations, confirmation payloads, or target outcomes. It does not execute
MotionCrafter or Prob4D and does not change the frozen provider, estimator,
comparison arms, guard, fallback, object split, or confirmation rule.

A valid admission authorizes only the declared calibration-only work list. It is
not evidence of provider competence, physical-query benefit, uncertainty
calibration, tactile benefit, deployment safety, Causal4D benefit, or state of
the art.
