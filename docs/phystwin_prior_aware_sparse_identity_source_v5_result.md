# Prior-aware sparse-identity state update: source result v5

## Result

The frozen v5 smoke completed prediction and scoring on the opened
`single_lift_cloth` development case. Exact Warp replay passed at zero error,
the four prefix sensors were fully observed, and five disjoint identities
remained hidden until the prediction seal.

The propagated state-plus-bias update failed its untouched prefix validation
gate:

| Prefix comparator | Vector RMSE |
| --- | ---: |
| Persistent graph readout | 35.685 mm |
| Propagated state plus bias | 38.590 mm |
| Difference | +2.905 mm |

The nominal state update was therefore rejected before nonlinear closure. The
sealed candidate, state-only trajectory, and persistence trajectory are
byte-identical.

After sealing, hidden-identity scoring gave:

| Future method | Chamfer | Hidden track |
| --- | ---: | ---: |
| Unchanged physical baseline | 24.625 mm | 40.609 mm |
| Sealed persistence fallback | 24.293 mm | 41.433 mm |
| Change | -0.332 mm (-1.35%) | +0.825 mm (+2.03%) |

Persistence helps CD slightly but worsens the disjoint material-identity
metric. Its early hidden-track benefit also reverses in the middle and late
horizon. The joint advancement gate fails.

## Decision

Stop this rank-four propagated state-plus-bias parameterization. Do not expand
it to a source panel or independent cohort, and do not tune its rank, priors,
caps, response length, or sensor count on this opened case.

The useful conclusions are narrower:

1. Exact branching positions and velocities can be recovered from the selected
   MatPhys replay.
2. Sparse prefix sensors must be selected from identities actually supported
   in the declared response window; frame-zero geometry alone is insufficient.
3. In this case, a linearized low-rank state update does not transfer even to
   held-out prefix frames, so the regret guard correctly falls back.
4. Readout persistence is metric-dependent: a modest point-cloud improvement
   can coexist with worse hidden material correspondence.

This removes action-propagated rank-four state correction from the immediate
SOTA path. The next method should use the already stronger automatic online
belief evidence rather than tuning this rejected manual-sensor smoke.

## Evidence

The compact result is
`results/sota/phystwin_prior_aware_sparse_identity_source_v5/summary.json`.
The complete per-frame score, prediction manifest, replay parity, and
correction manifest are archived beside it.

This is one previously opened source interaction with manual prefix sensors. It
is not open-loop, independent, confirmatory, or SOTA evidence. No held-v8
artifact was accessed.
