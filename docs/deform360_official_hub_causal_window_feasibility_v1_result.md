# Deform360 official-Hub causal-window feasibility v1

## Result

The frozen post-payload, pre-score visual execution path failed its registered
calibration-support requirement and is closed before provider inference.

| Quantity | Result |
|---|---:|
| Locked calibration objects | 10 |
| Successful causal windows | 4 |
| Retained technical failures | 6 |
| Required supported objects | 8 |
| Replacements | 0 |
| Calibration prediction scores opened | 0 |
| Confirmation payloads opened | 0 |

All six failures had the same predeclared cause: tactile contact occurred before
the frozen 42-frame observed history could be placed while retaining exactly six
post-contact frames. The three-camera pose-only selection and input custody did
not produce a recorded failure. This is therefore a schedule-feasibility failure,
not evidence for or against MotionCrafter, Prob4D, visual calibration, tactile
fusion, or Bayesian-PhysTwin accuracy.

## Frozen execution

- Implementation revision:
  `521d23f5433071cc322c6c3fd63e1e63ab4306de`
- Visual execution lock:
  `87b1efe7dc7e9a8f1fd4163e0a4164b5ba45e6bcc47d66e2fba2a2526c6f51e9`
- Provider recovery lock:
  `02d90e58f4f21d052073e098469ff8a7cd991f48b895d7496cebdb84dd10cb3d`
- Camera-panel policy:
  `ec8d7f56bb59731b6c5eec03fd627f79ef5864a447257f198a5c8d3b4869ffb5`
- Causal-window manifest ID:
  `9a23701c8181b89ba5b09cb545ca750513fb1e3f9a8d360099d00582d8b9f406`
- Manifest file SHA-256:
  `3367947f2c6d78c67d5c0c2691f302a4a1838d9b069ffd30773da5d9c8e04a8b`
- Server artifact:
  `/home/florianpfaff/source-only/deform360-official-hub-causal-windows-521d23f5.json`

The exact retained records are in
`results/sota/deform360_official_hub_causal_window_feasibility_v1/causal_window_manifest.json`.
Every claim-bearing input file in that manifest was rehashed against the Stage 1
inventory. The operator used tactile values only to locate first contact and
camera-to-world poses only to select the frozen three-view panel. It did not
decode camera frames, run the provider, inspect future geometry, or compute an
endpoint.

## Decision

Do not run the v1 provider on only the four surviving objects, replace failed
objects, shorten individual windows, or move individual cutoffs. Any such action
would change the registered campaign after observing its feasibility result.

A successor can remain scientifically valid because these ten objects are the
declared calibration cohort and no score has been opened. It must be a separately
identified v2 schedule committed before provider inference. The natural
target-free candidate is the earliest cutoff satisfying both constraints:

```text
cutoff = max(first_contact + 6, 42)
source = [cutoff - 42, cutoff)
future = [cutoff, cutoff + 24)
```

This keeps two 25-frame windows with eight-frame overlap, gives every arm the
same per-object prefix and future, and avoids synthetic padding. It does expose
more than six post-contact frames when contact occurs early, so that information
difference must be stated explicitly and checked only on calibration before any
fresh confirmation payload is opened.
