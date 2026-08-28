# Native Contact-Realization Screen: No Decision Headroom

The frozen three-world source screen completed and failed its advancement
gate. There were three ordinary native successes, zero technical failures,
zero replacements, and no retry. The implementation is local commit
`7899457e959746691970c8bb4a0fd9de595da596`; no production or DEFORM code changed.

The nominal 0.9 adapter replay passed: maximum position difference from the
registered reference was 5.45879031e-9 m, below 1e-6 m, with exact native
rewards. All three batches passed common-prefix, duplicate-action, endpoint,
and reward checks. The native solver coefficients were checked before action
execution, and nonrobot contact coefficients were unchanged.

| Source-model decision | Expected native reward | Gain over best fixed action |
|---|---:|---:|
| Best fixed action 6 | 7.00116316 | 0 |
| Bias-aware posterior mean | 7.00116316 | 0 (numerical 8.9e-16) |
| Bias-aware MAP | 6.99829616 | -0.00286701 |
| Posterior ignoring shared bias | 7.00087803 | -0.00028513 |
| Perfect information | 7.00116316 | 0 |

Action 6 is optimal or tied in every source world. At coupling 0.3, all tested
actions tie the zero-control reward. Consequently, even perfect knowledge of
contact coupling cannot improve on the strongest fixed action in this bank.
The posterior beats MAP and the ignored-bias arm, but that is not useful
information value against the stronger control.

The causal prefix does change with coupling: pairwise Mahalanobis distances
under the registered observation model are 2.31987 (0.3/0.6), 3.19563
(0.3/0.9), and 1.63357 (0.6/0.9). This is an action-bank limitation rather than
zero observable response. The existing actions only vary future Cartesian
motion and rotation; they do not include a contact-recovery or grip-force
decision. This distinction motivates a different source question, not a
revision or automatic rerun of this failed protocol.

These values integrate the three-point source model with hypothetical Gaussian
sensing. They are not independent evaluation rewards, a calibration result,
measured friction identification, or published benchmark parity. The native
coefficient implements tangential gripper coupling, not validated Coulomb slip.

## Verification and Custody

- 181 relevant tests, Ruff, focused MyPy, and source/real-constructor preflight
  passed before execution.
- A second arithmetic implementation recomputed all 24 native rewards, nominal
  replay, source-prefix distances, three decision arms, and every gate. It
  used a dense 36-dimensional covariance inverse rather than the production
  Cholesky whitening. This is not independent human review.
- Source lock ID:
  `362175cce80bce8eb5409a02bcdd0476b376241525293d5aa7e9d05316631f67`.
- Result ID:
  `09cd8e3f9ab6e879a872c06fb9ee039dd61b1d0b597ef656f4137114ee0c5d1a`.
- Result file SHA-256:
  `2894aed878de6da56fa84d089e58a9dfaa522ed1dee90cb5579f1f15fd2189cd`.

No new recordings, GPU work, calibration/evaluation worlds, protected targets,
or robot executions were used. No new method is promoted.
