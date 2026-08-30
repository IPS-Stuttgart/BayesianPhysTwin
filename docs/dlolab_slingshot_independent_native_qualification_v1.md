# DLO-Lab Slingshot Independent Native Qualification v1

The prospective v2 policy certificate passed its causal-prefix admission gate
but could not be scored: 286 of 288 evaluation futures sealed ordinarily and
two registered worlds failed the eight-environment native QA. The complete
denominator remains unscored, and those worlds are not retried or replaced.

This development-only qualification tests a different execution interface.
Each registered world/action pair runs in its own fresh Python process with a
single Genesis environment. Eight new continuous worlds are generated with
seed `262060`, disjoint from every earlier Slingshot roster. Each world runs
the frozen eight-slot action bank, including the independently repeated action
in slots 5 and 7, for 64 total processes. No v2 world is rerun.

For each world, the qualification reconstructs the standard eight-slot array
only after all isolated executions have sealed. It rederives exact controls,
world placement and material realization, native reward arithmetic, a common
300-step prefix within 0.5 mm, duplicate-action positions within 0.5 mm,
duplicate rewards within 0.001, and fixed endpoints within 1 nm. All 64
processes and all eight world-level checks must pass. Failures remain in the
denominator; no retry or replacement is allowed.

A pass qualifies only this executor and permits freezing a new v3 protocol on
new disjoint simulator worlds. It does not authorize that scientific run by
itself and provides no policy-value, coverage, calibration, SOTA, robot-safety,
or real-data claim. The run uses public DLO-Lab code and procedural worlds on
CPU; it reads no protected target, held-v8, DLO4/DLO5, or new recording.

The registered write-once output root is
`/home/fpfaff/source-only/dlolab-slingshot-independent-native-qualification-v1`.
Its separate attempt ledger is written before any native child process starts.
