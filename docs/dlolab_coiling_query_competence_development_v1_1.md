# DLO-Lab Coiling Development v1.1 Replacement

V1.1 is the sole implementation replacement for the terminal zero-step v1
failure. The parent lock and both failure identities are embedded in the
machine protocol and the compact parent summary is hash-bound by the runner.
The v1 output root remains terminal and is never reused.

The only implementation change is environment-specific: coiling has one native
rod, so its bending and twisting fields have shape `(1, 8)`, not wiring's
`(3, 8)`. V1.1 requires that exact coiling layout before and after each setter.
No action, world, observation, reward, noise, threshold, tie-break, denominator,
or information boundary changes. `protocol_v1_1()` is mechanically identical
to v1 after removing its schema and replacement lineage block, and this is unit
tested.

The fresh registered root is
`/home/fpfaff/source-only/dlolab-coiling-query-competence-development-v1-1`.
It permits one execution only. Every world still writes its claim before native
initialization, and any runtime or qualification failure is terminal without a
further replacement. A completed development pass still cannot authorize a
prospective replication automatically.
