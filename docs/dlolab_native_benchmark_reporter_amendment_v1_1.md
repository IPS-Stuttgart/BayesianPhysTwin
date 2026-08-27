# Slingshot Qualification v1.1: Reporting Repair Only

This prospective amendment permits one new deterministic source-qualification
execution after the retained v1 post-rollout reporting failure. It does not
retroactively pass or overwrite v1, and it is not another independent sample.
No method scores or usable numeric trajectories were available from v1.

The implementation now compares boolean state by inequality, preserves its
boolean dtype in NPZ transport, seals each completed native rollout before any
cross-rollout statistic, and seals the complete generation before analysis.
It also distinguishes a reporting-stage failure from the last native loop label.
Regression tests reproduce the boolean-subtraction failure and lossless transport.

Everything scientific is unchanged: official task and source, asset bytes,
runtime packages, simulated robot/controller, material/contact settings, reward,
zero/pull/pull order, 900 steps, control amplitudes, and all qualification gates.
The runner requires the exact prior failure SHA-256
`aecc4225c9e8c06998d4e339df28a53d32ca12ae26fde8a42f9bd34680819db3`,
validates its canonical identity, and compares native source/runtime/protocol
with the parent before initialization. Only reporting-version/custody fields
may differ. The original failed directory remains immutable.

Use a fresh `qualification-v1-1-reporter-repair` output directory. This amendment
authorizes no parameter search, Bayesian method comparison, task redesign,
protected data, GPU job, target evaluation, public push, or main-branch merge.
Any additional failure remains retained and blocks downstream method work.
