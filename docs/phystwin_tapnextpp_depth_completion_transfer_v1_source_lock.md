# TAPNext++ depth-completion transfer source lock

The source-transfer staging step completed from implementation commit
`3feff08fb33a4c20fde29735842bab1c9632ed29` before any new TAPNext++
prediction or withheld-prefix score was produced.

## Accounting

- Fixed source cases: 8
- Prediction-ready cases: 8
- Technical staging failures: 0
- Replacements: 0
- Held-v8 artifacts accessed: no
- Future simulator outcomes read: no

| Case | Prefix window | Query identity IDs |
| --- | ---: | --- |
| `double_lift_cloth_1` | `[61,81)` | `0,8,6,2` |
| `double_stretch_sloth` | `[2,22)` | `0,7,5,4` |
| `double_lift_zebra` | `[20,40)` | `0,6,4,1` |
| `rope_double_hand` | `[22,42)` | `0,8,4,2` |
| `single_lift_rope` | `[15,35)` | `0,8,4,6` |
| `single_push_rope_4` | `[29,49)` | `0,8,4,2` |
| `single_lift_dinosor` | `[26,46)` | `0,2,8,7` |
| `weird_package` | `[5,25)` | `0,6,4,3` |

## Checksums

- Cohort protocol SHA-256:
  `42df862177a63b19995d9c09f6f464494854868ae6953f4af8298c1947fef8f9`
- Source manifest file SHA-256:
  `590196fbe2aedc8982ddd5ef269e264173cdca39aa9f2d3a4c357dc1f4a41b3c`
- Source manifest canonical result SHA-256:
  `c15864cf057fee8ba03f5929fc32660c1c8265810ad866821aa0865008707a52`

The source manifest and eight per-case tracker protocols are archived under
`results/sota/phystwin_tapnextpp_depth_completion_transfer_v1/source_lock/`.
Each case protocol binds the staged prediction input, separately withheld
prefix target, physical selector input, fixed tracker configuration, completion
configuration, and per-case gates.

## Authorization boundary

This lock authorizes the eight causal prefix tracker predictions, their strict
prediction seals, and target-free depth completion. The withheld manual prefix
may be opened only after both provider artifacts are sealed. It does not
authorize future Bayesian-PhysTwin scoring, target access, or a SOTA claim.
