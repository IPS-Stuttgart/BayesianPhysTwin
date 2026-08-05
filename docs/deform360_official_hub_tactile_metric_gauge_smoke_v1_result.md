# Deform360 official-Hub tactile metric-gauge smoke result

## Result

The frozen source-only metric-gauge gate passed. All three geometry-selected
cameras passed for both retained tactile-to-gripper assignment hypotheses.
Every camera supplied all 46 active taxel rows across the six causal contact
frames.

| Camera | Provider | Assignment 0 median / p90 / max | Assignment 1 median / p90 / max |
| --- | --- | ---: | ---: |
| `019_cam1` | Reused Stage 1 | 2.22 / 4.77 / 8.05 mm | 3.30 / 7.81 / 14.85 mm |
| `013_cam1` | Supplemental | 3.77 / 6.31 / 55.90 mm | 2.06 / 4.65 / 17.72 mm |
| `006_cam0` | Supplemental | 3.93 / 7.06 / 12.14 mm | 2.64 / 4.70 / 6.78 mm |

The locked admission thresholds were at most 5 mm median and 15 mm 90th
percentile held-frame error, with three cameras required to pass both
assignments. The two assignment hypotheses remain at prior probabilities
`0.5/0.5`. Unknown cross-view correlation is handled with equal-weight
covariance intersection, not independent-view precision multiplication.

The `55.90 mm` maximum residual for `013_cam1`, assignment 0, is deliberately
reported. It did not violate the preregistered robust median/p90 gate, but it
precludes treating every taxel row as a Gaussian measurement. A later
association/update stage must retain robust innovation handling.

## Meaning

This result establishes that known causal contact geometry can identify a
stable metric Sim(3) gauge for the MotionCrafter point maps in this source
case. It does not establish which tactile group belongs to which gripper, which
visual point belongs to which deformable-object material point, or that a
Bayesian-PhysTwin state update improves prediction.

Accordingly, the result records:

- `metric_gauge_authorized=true`;
- `contact_anchor_authorized=false`;
- calibration scores unopened;
- confirmation and target payloads unopened;
- no held-v8 access;
- no SOTA claim.

The next admissible method step is a source-only object-association gate that
preserves the assignment mixture and metric covariance. Only after that gate
passes may the already registered calibration score be opened for the guarded
Bayesian update.

## Provenance

- Provider lock ID:
  `f1259e69bb7f1cc7e5f3923376e8a6e6b3b26d40913098471f3cf550d6564bd3`
- Supplemental provider run ID:
  `7dea7487b68f6aec98abd1859a9033035acae2446f863abeb3e472a84252670a`
- Supplemental bundle: 13 files, 1,193,009,914 bytes; both manifests and all
  eight declared members independently rehashed.
- Evaluator revision:
  `5369ab3317a01871577d048a8b413709b81cfd7d`
- Full result artifact ID:
  `96dee634e43a869cf9a794fee14cd3ac4fae1825c988a7a3e008552061fca41f`
- Full result SHA-256:
  `b6bbbfe118288ca6d75a2577cd679024876c166c048fdc87c9bfd3bbaf420f97`
- Original and replay outputs are byte-identical.
- Focused verification passed 30 tests locally and 30 tests on the exact
  remote checkout; Ruff and bytecode compilation passed.

The full result remains durably stored at
`/home/florianpfaff/source-only/deform360-tactile-metric-gauge-smoke-v1-5369ab33/metric_gauge_result.json`.
