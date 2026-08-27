# Native Slingshot Task-Competence Bank: Failed

The fixed 24-action source bank ran at
`8916f99e1175af3228dfbf576087776c523e61b2` with 24 ordinary successful
executions, zero retained runtime failures, and zero replacements. Every action
was run in its own qualified CPU process. All 792 arrays and native reward
arithmetic were checked using a separate standalone implementation after the
full generation seal. This is not an independent human review.

No candidate passed the frozen 10 mm sphere-and-cube progress and 0.01 native
reward-gain gate. Twenty-three actions leave the sphere unmoved and have the
same native reward as zero control, 6.900000095367432. The remaining fixed
random action (index 21) moves the sphere 108.795 mm and cube 1.790 mm, with
reward 6.901790142059326. Its small contact response is real in the retained
native traces, but it is below the registered task-competence threshold.

This closes the fixed action bank, not the native simulator or all possible
controllers. It is not a failed test of a Bayesian controller: none was run.
There is no task-competence promotion, new Bayesian gain, published-controller
parity, or real-world claim. DEFORM remains unchanged. A separately frozen
nominal controller-optimization development study could use this opened source
information, but cannot call the reused task independent confirmation or revise
this result. It must establish competence before a belief-dependent comparison.

Canonical result: `5c4ec4a1188a0cd96f0ec90aedc1b1800537a0a060b06b7e2977a50635f34946`.
Result SHA-256: `fcbf55f54f71269cd48ac211df051820cf1e902f5f03fe0eaa5408fe4f3e6f29`.
Lock SHA-256: `e51f74b1b58693f4d9b9ea521e4eb9feeaef8f6185702da707acdfc0e469be99`.
Generation SHA-256: `3f9a9aea2a37893b9a0413d4c8a47bec3b8afdf7fab6ef6c7961618e6660eb67`.

The focused pre-run suite passed 53 tests; the expanded relevant suite passed
278. Ruff, focused MyPy, and diff checks passed. All evidence and implementation
remain local/private. No GPU, protected data, recordings, push, or main merge.
