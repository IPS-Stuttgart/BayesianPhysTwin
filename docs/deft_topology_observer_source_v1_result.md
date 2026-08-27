# Topology-Supported DEFT Source Test: Not Promoted

## Result

All three registered recordings and all 24 forecast arms completed normally.
Independent verification reproduced 512 metrics and all 36 gate decisions.
The primary transfer gate **failed**. Measuring a point on each child did not
make the fixed paired physical pose/velocity update transferable: hidden-child
RMSE increased **2.88%**, and late RMSE increased **5.49%**, versus native DEFT.
There were zero joint RMSE/L1 wins over native DEFT across the three recordings.

Retain unchanged native DEFT and the separate successful DEFORM method. Do not
retune this test's observations, window, interpolation, gain, or velocity rule
using the opened outcomes. Secondary arms cannot rescue the failed primary.

## Matched Results

All numbers are millimetres. First average the two child branches equally within
each recording, then average the three recordings equally. Five child identities
are disjoint from both observation policies. Duplicate roots, tips observed in
the prefix, and padded identities are excluded. Coordinate L1 is not Chamfer
distance. These are three recordings of one object, not three object replicates.

| Arm | Child RMSE | Coordinate L1 | Late RMSE | FDE |
|---|---:|---:|---:|---:|
| Unchanged full DEFT | 24.433 | 10.934 | **29.842** | 34.548 |
| Parent-only paired pose + velocity | **24.384** | **10.852** | 30.002 | 34.885 |
| Topology readout persistence | 26.344 | 11.093 | 33.625 | 37.945 |
| Topology linear-velocity readout | 48.116 | 19.609 | 67.561 | 77.023 |
| Topology paired pose | 24.666 | 11.002 | 31.267 | 35.465 |
| Topology paired pose + velocity, primary | 25.137 | 11.220 | 31.480 | 35.065 |
| Topology direct full-model update | 25.032 | 11.124 | 31.414 | 34.179 |
| Topology periodic native pose correction | 25.307 | 11.313 | 31.250 | **33.765** |

Every corrected arm uses eight additional known material-point observations:
four identities at two prefix times. The native comparator uses none. All arms
share two initial full states and the prescribed future four-point clamp motion.
The topology policy observes two parent junctions and two child tips; the parent
policy uses its unchanged four parent identities. No arm sees their union.

| Arm | Child 1 RMSE | Child 2 RMSE | Hidden parent RMSE |
|---|---:|---:|---:|
| Unchanged full DEFT | 21.088 | 27.777 | 19.865 |
| Parent-only paired update | 21.065 | 27.704 | 19.526 |
| Topology readout persistence | 29.997 | **22.691** | **15.656** |
| Topology paired update, primary | 22.555 | 27.719 | 19.724 |

The primary wins against the linear-velocity readout on all three recordings,
and against persistent readout and periodic correction on two of three. It
nevertheless fails against the strongest comparators and fails the registered
per-child improvement requirements. The parent-only arm's 0.20% aggregate RMSE
gain is small, lacks the primary status, and comes with worse late error. It is
not a promoted replacement.

## Interpretation

The topology observations identify all four coefficients of the chosen spatial
interpolation basis in the synthetic test, while the parent-only policy
identifies two. That is not full nonlinear state observability: twist, internal
forces, model discrepancy, and unobserved spatial directions remain unresolved.
The real-data result rejects the narrower prediction that filling this basis's
measurement null space is enough for the unchanged physical update to transfer.

Readout correction helps the hidden parent and child 2 but harms child 1. Physical
transport does not turn those local measurements into consistent hidden-child
gains. The experiment does not determine which internal mechanism caused this,
and it does not reject Bayesian state estimation generally. No calibration,
independent generalization, or official benchmark/SOTA claim follows.

## Provenance and Verification

- Method and independent verifier frozen before decoding at
  `c585e3c2358e5e5003a8d6152ed8721100354f40`.
- Source receipt SHA-256:
  `f66190c58917c4a6427b050651a28095edddf293fc9de091219d6a52b5f910c8`.
- Synthetic qualification: six checks pass, including exact recovery in the
  declared interpolation space, clamp/root consistency, and zero increments.
  File SHA-256:
  `16f1c102f2b77829a4df4935b0cdf67a000d94a372828a8f83d1808f0f1cc88a`.
- Input barrier SHA-256:
  `8bd20ff4dba3cd9f95003652801be7c746d2f60f2a5ebc6a74a8715dd964dad9`.
- Prediction barrier SHA-256:
  `29c3bd38b358208883703b933b804fd0e8e82a60661047692b71ff3cbe0bd0f5`.
  All 24 forecasts were sealed before any future free-node scoring.
- Result SHA-256:
  `e7acd4ddea57d6dd22594b01225894fd289be7653ceb5feb000603f4c5bf0c4f`.
- Independent verification SHA-256:
  `0e4c04c94966b8ed50974ebf7e589210b0f01705b4823f9efc77e97faef22ca7`.
  The independent raw-identity implementation verifies all 512 metrics and all
  36 gate decisions from the sealed arrays.
- Relevant restart, belief, native-DEFT, and topology suites: **255 passed**.
  All five new Python files pass Ruff; the module, runner, and verifier pass
  focused MyPy. This is not a claim to have run the full repository suite.
- Accounting: three ordinary successful recordings, zero retained technical
  failures, zero unsealable recordings; no empirical retry or replacement.

The pinned DEFT code/checkpoint and qualified CPU compatibility runtime are
unchanged. The three filename-selected BDLO1 training recordings have checkpoint
training exposure. The earlier failed track1244 recording is excluded. This is
a new source-capacity test, not a held-out-object confirmation.

An incomplete acquisition archive is retained as documented in the protocol;
the exact three files were subsequently downloaded and hashed before decode.
No unselected trajectory, public evaluation/test content, protected DEFORM DLO3
evaluation or DLO4/DLO5, held-v8, Deform360 target, or physical Causal4D data was
inspected. No new physical recordings were made. Existing successful and failed
experiments are unchanged. Evidence remains local/private-paper only, with no
automatic authorization for a larger evaluation.
