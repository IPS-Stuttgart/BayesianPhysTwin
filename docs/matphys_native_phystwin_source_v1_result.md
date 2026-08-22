# Native PhysTwin MatPhys source result v1

## Decision

The registered source gate **failed**. The target-excluded MatPhys fold
ensemble does not provide useful covariance around the unchanged released
PhysTwin mean, and no fresh evaluation is authorized. DEFORM and every frozen
Causal4D result remain unchanged.

This is an opened-source mechanism result, not a fresh, confirmatory, real-data,
or state-of-the-art comparison.

## Execution

- Implementation revision: `43df7905edbc1d23c5156a2c9a794b05440292d2`
- Official PhysTwin revision: `2b6630528141b9cba5a7677c8b88b2129b4a8390`
- MatPhys revision: `c16b858dfb79bf21024ead24b45a710600de7b4f`
- Registered source cases: 11
- Ordinary parity-valid cases: 8
- Retained native-parity failures: 3
- Result ID: `0ca947f6329ff2cdc60302e33afc21af340d2bd3c3c2c9a08b7d8e2db709a06a`
- Result file SHA-256: `505a191a51abf52bb8ec57c4fbd6331745c56e769bb29c17c7fe83bf44faea99`

The initial deterministic replay smoke was discarded before scoring because it
made the registered incumbent-replay covariance exactly zero. The final run
uses the official atomic spring-force path, preserves its numerical variation,
and writes each replay to a new content-addressed evidence root.

## Native parity

The frozen identity threshold was 2 mm coordinate RMSE against the released
trajectory.

| Case | Status | Replay-to-release RMSE (mm) | Maximum error (mm) |
| --- | --- | ---: | ---: |
| `cloth_blue_fold` | retained failure | 2.414 | 63.233 |
| `cloth_blue_lift` | ordinary | 0.313 | 4.625 |
| `cloth_pant_fold` | ordinary | 0.440 | 5.455 |
| `cloth_pant_lift` | retained failure | 6.284 | 693.852 |
| `cloth_red_fold` | ordinary | 0.727 | 5.698 |
| `cloth_red_lift` | ordinary | 0.494 | 3.083 |
| `cloth_shirt_fold` | retained failure | 3.673 | 36.094 |
| `cloth_shirt_lift` | ordinary | 0.089 | 1.065 |
| `cloth_skirt_1_fold` | ordinary | 0.649 | 6.734 |
| `cloth_skirt_1_lift` | ordinary | 0.902 | 6.875 |
| `cloth_skirt_2_fold` | ordinary | 0.769 | 13.560 |

The failures are deterministic accounting outcomes, not failed MatPhys point
predictions. They show that long released self-collision trajectories are not
uniformly reproducible under the pinned current runtime. For example, the four
official-atomic incumbent replays for `cloth_blue_fold` differ pairwise by
1.756--2.973 mm coordinate RMSE.

## Covariance result

Lower NLL and volume are better. Coverage is for the nominal 90% marginal
three-dimensional ellipsoid. Aggregates are equal-event within case and then
equal-case over the eight parity-valid cases.

| Metric | MatPhys-shaped covariance | LOO isotropic comparator |
| --- | ---: | ---: |
| Gaussian NLL (nats/event) | -7.3234 | **-7.8486** |
| NLL wins | 0/8 | 8/8 or tie |
| 90% coverage | 91.85% | not an advancement endpoint |
| Mean-volume ratio | 1.0463 | 1.0000 |

The registered NLL improvement is `-0.5253` nats/event: the candidate is worse,
not better. Its ellipsoids are also 4.63% larger. Only the coverage interval and
complete-denominator accounting checks pass. Case-win, aggregate-NLL, volume,
and zero-parity-failure checks all fail.

## Post-open headroom

Two non-claim-bearing diagnostics test whether the native members should be
promoted as a point predictor instead:

- their direct ensemble mean changes equal-case coordinate RMSE from 14.6352 to
  14.5143 mm (`-0.83%`, 6/8 point-error wins), but separately calibrated
  isotropic NLL improves by only 0.0247 nats/event with 4/8 NLL wins;
- selecting one member using only its prefix error changes coordinate RMSE by
  `-0.82%` with 5/8 wins, while a future oracle over all 11 members has only
  `-2.08%` headroom.

These checks are exploratory and were computed after the source result opened.
They do not authorize a new selector or threshold. Their small and inconsistent
headroom supports closing this particular MatPhys family rather than tuning it.

## Consequence

MatPhys remains a supported, opt-in material-proposal interface and a useful
negative backend control. It is not promoted as the default point predictor or
uncertainty donor. A future MatPhys study would require genuinely new evidence,
such as an independently released native trajectory cohort or a materially new
proposal family, not another scale grid on these 11 interactions.

The exact machine-readable result is
[`results/sota/matphys_native_phystwin_source_v1/result.json`](../results/sota/matphys_native_phystwin_source_v1/result.json).
