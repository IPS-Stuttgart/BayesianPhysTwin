# Native Slingshot Decision-Value Screen: PASS

This is a public-simulator source diagnostic, not a Bayesian-controller result,
an independent target evaluation, or published benchmark parity. The frozen
implementation was `348d3bbff128b4151cf87684d9b733ffd82e402c`.

All nine worlds and 72 native trajectories completed without technical failure.
The seven unique actions share the first 300 native frames. Each world includes
an exact incumbent duplicate; all world QA checks passed.

| Fixed comparison | Mean native reward |
|---|---:|
| Zero control | 6.900000095 |
| Original nominal CMA-ES incumbent | 6.949683242 |
| Best single action over the nine worlds (action 5) | 6.950281355 |
| World-conditioned oracle | 6.969645924 |

The raw oracle gain over the best single action is 0.019364569 native reward.
After subtracting the registered 0.002 numerical margin, it is 0.017364569,
or 34.53% of the best fixed action's gain over zero. Five worlds improve by
more than 0.01; five distinct actions are optimal across the bank. All six
registered decision-value checks pass.

Maximum common-prefix position discrepancy is 4.314e-14 m. Maximum duplicate
position discrepancy is 2.678e-9 m. The nominal incumbent differs from its
previous isolated reference by at most 1.184e-8 m. These are observations from
this screen, not universal simulator-determinism or coverage guarantees.

The earlier full-state CMA-ES replay and tight mechanism-audit replay gates
remain failed. This screen uses a separately frozen task-level numerical
envelope, fresh processes, and no hidden-state restart; it does not revise
those prior decisions. The original native reward, robot, and contact physics
are unchanged. No protected data, new recordings, or GPU were used.

## Provenance

- Result artifact ID: `472b2a34928671451207bc9bf6485a367ec2612342b40003ab2bf8608b57e2bb`.
- Result file SHA-256: `e4097d1be73321573ef3dd1ecb309e9d77101207cbc0ffbdc90c8eed3b5d165b`.
- Lock artifact ID: `305f31645a90faaec41429009ff36058c0352779162bc2829ecb5d922a16821d`.
- Lock file SHA-256: `cfa1038159a8696f52001143b95de308b672deec93683bf9945ec5d0777cdc93`.
- Generation artifact ID: `dcc0f64923bd177c21cf3e5ae13748e0612dd253041beb0ba91e92f26c71d5ad`.
- Source root: `/home/fpfaff/source-only/dlolab-benchmark-source-v1/decision-value-source-v1`.

The standalone archive verifier rehashed every native array and recomputed the
rewards, QA checks, and gate with a second arithmetic implementation. It is
not independent human review. Before execution, 119 relevant tests, Ruff,
focused MyPy, and diff checks passed.

## Next Test

The screen establishes that state-dependent decisions have useful headroom.
It does not establish that an allowed noisy prefix identifies those decisions,
that Bayesian uncertainty helps beyond a point estimate, or that a calibrated
guard improves the risk/reward tradeoff. Those require a separately frozen
comparison on fresh simulated conditions, including the best source-selected
fixed action, MAP/point control, posterior-mean control, mean-only calibrated
control, and matched joint-versus-independent regret guards.
