# V14 Physical Runtime V2

## Trigger

Before the first physical carrier was executed, static preflight found that the
runner passed the lawful 58-frame prefix-geometry robot archive to a physical
path whose registered horizon is 76 frames. The prefix archive is deliberately
short because object-facing geometry is forbidden after frame 57; it is not
the registered known-action source.

No physical carrier, source outcome, identity target, future metric, target
artifact, or held-v8 artifact was opened before this correction.

## Correction

The child runtime lock binds, for queue ranks 3 through 14:

- the exact action-only window-stage result;
- the exact 81-frame staged robot archive;
- the first 76 frames as the physical rollout action;
- the unchanged frame-zero geometry;
- the amended runner and runtime-validation module.

The last five staged frames retain their pre-existing role in point-cloud tail
construction and are not added to the 76-frame physical horizon.

The runner now keeps the two inputs explicit:

```text
58-frame prefix geometry robot -> geometry custody only
81-frame staged robot [0,76) -> known physical action
```

## Claim Boundary

This is a pre-execution input-contract repair, not a method or gate change and
not empirical evidence. The original physical pre-lock remains immutable. The
runtime-v2 child lock records why the old input was inoperable and exactly
which already-staged action replaces it.
