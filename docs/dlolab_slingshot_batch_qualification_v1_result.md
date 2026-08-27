# Native Eight-Environment Slingshot: Parity Passed

The frozen batch at `98c8591ad9fed6ae1d1a9efbbab25a965378091f` passed
8/8 environment comparisons with the isolated zero/pull references. Maximum
position discrepancy is 3.08771e-12 m, all 23 final native memory fields pass
the unchanged tolerances, fixed endpoints remain exact, and all native rewards
match exactly. This is tolerance parity, not a bit-identical-state claim.

The batch took 38.993 s within the native wrapper for eight trajectories,
compared with about 29 s per isolated trajectory in the earlier qualification.
Those single-run timings are descriptive, not a throughput benchmark. The
expanded relevant suite passes 282 tests; Ruff and focused MyPy pass.

Canonical result: `732cca6a79d6a33dd112fe039a3b532a8eb66308be4b2f0023ca44faf450d9fe`.
Result SHA-256: `21c31c97882b80c8f191cf266aeaaf9620be81324a55d4bcca011d617bcef3cf`.
Generation SHA-256: `4c1e4620e80eb54bea33aa214fa8707d5d8b4d8fa413abd8f5c9441984b73498`.
Array SHA-256: `0b430194338f35ef8c236897bc7e3368790235fd0696076e2e5442e37992691c`.

Only this execution shape is qualified. The separate 24-action task-competence
bank remains failed. A standard optimizer can now be tested prospectively as a
source-development baseline; no Bayesian or published-controller gain exists.
