# Native Slingshot Qualification v1.1: Replay Gate Failed

The reporting-only execution at `7d7ffee9ef756a5c63367809f878f8386eba252f`
completed and sealed all three native rollouts. It does not replace the retained
v1 reporting failure. The official task, controller, contact physics, source
assets, reward, actions, and thresholds were unchanged.

Six of seven qualification checks pass. The pull moves the gripper 102.537 mm
and the band 10.519 mm, with zero fixed-endpoint displacement. Repeated pull
positions differ by at most 3.03215e-8 m. The final 23-field native state is
not byte-identical, and two rod fields fail the registered allclose test
(`rtol=1e-6`, `atol=1e-9`): velocity (maximum difference 5.56166e-9 m/s)
and twist (maximum difference 4.40991e-8 in the native field's units).
These are small numerical discrepancies, not evidence of a large physical
error. Nevertheless, this exact reused-environment qualification fails.

All three native cumulative rewards are 6.900000095367432. Neither the zero
action nor the simple pull moved the target cube. This is a control smoke,
not a trained-controller comparison or a Bayesian gain.

The aggregate NPZ and all 99 array identities were rehashed successfully.
Canonical result: `d3631c30851c0efe8436a1bfcdf388b9ce7e4d46df46534c3a60ba0351a3daae`.
Result file SHA-256: `33b293c808812f97fcd6833f0cf8daff3607d08a4bd1ed78767ed361e4efea65`.
Array file SHA-256: `e55bffa903af1dbf4c85a321fb490d8676f0b2258b0f12e2d71d435cde80d4cf`.

No threshold is relaxed and no method comparison is authorized by this result.
A separately frozen fresh-process execution contract may test whether process
isolation removes reset-history effects. It must retain these failed receipts
and use the same native physics, actions, and numerical tolerances.
No protected data, GPU work, new recording, public push, or main merge occurred.
