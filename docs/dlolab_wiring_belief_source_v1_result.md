# Wiring-Post Source Result: Insufficient Decision Value

**Disposition: complete source-screen failure, not a runtime failure.**
All 11 native CPU batches / 88 trajectories passed the frozen qualification.
The seven-action, nine-material-world source bank failed six of eight value
checks. No larger evaluation, method promotion, or retry is authorized.

Implementation and protocol were committed before execution at
`a00e4807ab5c16c8f1ba4dd8410ed3ebde65be3c`. The native robot, task, target shape,
reward, constraints, and contact of
[DLO-Lab](https://github.com/UMass-Embodied-AGI/DLO-Lab) were retained. The material
prior and fixed action bank are ours, not its published controller benchmark.
See the [source protocol](dlolab_wiring_belief_source_v1.md).

## Complete Source Comparison

Equal weight over nine specified simulator material settings; larger native
final reward is better. The Bayesian/MAP rows integrate the frozen synthetic
observation-noise model, not new real sensor recordings.

| Decision | Expected native final reward | Gain over best fixed |
|---|---:|---:|
| Hold after shared prefix | 0.193107107 | -0.721975560 |
| Nominal-world best action | 0.911795497 | -0.003287170 |
| Best fixed action: overshoot then return | 0.915082667 | 0 |
| Posterior ignoring shared observation bias | 0.914309104 | -0.000773563 |
| Bias-aware MAP material, then best action | 0.915246347 | +0.000163680 |
| Bias-aware posterior expected-reward action | 0.915250884 | +0.000168217 |
| Perfect-information finite-bank oracle | 0.915252705 | +0.000170039 |

The Bayesian raw gain is only 0.01838% of the best fixed reward. Subtracting
the frozen numerical pair margin of 0.002 gives -0.001831783. Its raw gain over
MAP is 0.000004537, far below the registered useful-gain threshold.

Two different actions are optimal: nominal routing in the three E=10000 worlds,
and overshoot-and-return in the other six. But the best fixed action is already
nearly optimal everywhere. Bayesian selection captures 98.93% of the tiny raw
perfect-information headroom; this ratio is descriptive, not a gain of 98.93%
in task reward or a certified numerical bound.

The passing checks are improvement over prefix hold and multiple oracle actions.
The oracle magnitude, per-world useful-headroom count, Bayesian magnitude,
relative deficit reduction, MAP advantage, and adjusted ignored-bias advantage
all fail. More accurate inference cannot create substantial value when the
registered finite action bank has so little oracle headroom.

## Native Qualification

All ordinary native trajectories completed; technical failures, missing seals,
unsealable cases, replacements, and unrun batches are all zero. Every trace and
full rigid/rod memory was finite.

| Check | Maximum observed value |
|---|---:|
| Three-repeat same-action final reward span | 0 |
| Three-repeat coordinate span | 7.3669e-13 m |
| Within-batch duplicate coordinate error | 0 |
| All-action shared-prefix error | 1.2465e-13 m |
| Fixed-post movement | 0 |
| Material-point attachment distance | 2.0457e-6 m |
| Segment-length relative error | 0.6284% |
| Native final reward reconstruction error | 2.9269e-8 |

These are finite source observations, not population repeatability bounds.
Native cumulative float32 reward recomputes exactly. The native solver ran on
CPU with Python 3.12.13, Torch 2.8.0+cpu, NumPy 2.5.2, and pinned DLO-Lab
`c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`.

## Verification and Custody

Before execution, 802 DLO-Lab/DEFORM regression tests, Ruff, and focused MyPy
passed. Five additional verifier tests exercise a separate reward calculation,
Sherman-Morrison likelihood versus Cholesky whitening, record tampering, gate
booleans, and dtype-bound array identity.
The combined post-run DLO-Lab/DEFORM suite passes 807/807 tests; changed-file
Ruff and four-source-file MyPy checks also pass.

The standalone verifier imports no production experiment code. It rechecks
committed source hashes, all native array hashes, 88 trajectories, 7,920 reward
rows, repeatability, source-prefix extraction, posterior/MAP decisions, and all
gates. Maximum arithmetic difference is 1.22125e-15. This is a second arithmetic
implementation, not an independent human review.

- Source-lock ID:
  `544267169a4dd5a6cc6d072806c6480a7e612cbdc016ad216f5f654719a78b93`.
- Source-bank seal ID:
  `270cf0350741651be470ad56afa30c68fbdc12b3980689f7d5280ed7f1e4aa99`.
- Result ID:
  `9c5ba88c377ca55c4ac1b731ef7d009befcf19e65d05b25216ea12306ef98c91`.
- Result file SHA-256:
  `fa5361f80fd6952a5efae20605ffbe66f9388135005071546aa53cad30ebb82e`.
- Verification ID:
  `973760fd7d68497b91f2d8fcaa25294c81b26ad77253b1aaba262d9d48d451e8`.

Full local run: `/home/fpfaff/source-only/dlolab-wiring-belief-source-v1`.
Compact evidence is in `results/source/dlolab_wiring_belief_source_v1/`.
No public push or main merge was performed.

## Scientific Decision

Do not scale or retune this fixed bank to obtain a headline Bayesian-control
result. It provides a functioning, numerically qualified public native task and
a useful negative control: strong task performance does not imply useful value
of uncertainty-aware action selection. It does not rule out richer actions,
other observation settings, other material priors, or other tasks.

The successful DEFORM point forecasts and all earlier frozen failures remain
unchanged. No new recordings, real robot execution, GPU work, protected targets,
held-v8, DLO4/DLO5, or official DLO3 evaluation were accessed.
