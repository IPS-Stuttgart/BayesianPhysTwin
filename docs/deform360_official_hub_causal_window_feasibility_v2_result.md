# Deform360 official-Hub causal-window feasibility v2

## Result

The separately locked v2 schedule passes the target-free calibration-support
gate.

| Quantity | Result |
|---|---:|
| Locked calibration objects | 10 |
| Successful causal windows | 10 |
| Retained technical failures | 0 |
| Required supported objects | 8 |
| Replacements | 0 |
| Calibration provider outputs opened | 0 |
| Calibration prediction scores opened | 0 |
| Confirmation payloads opened | 0 |

Six objects have first tactile contact at frame 0. For those objects the tactile
event indicates an already-established grasp rather than identifying action onset,
and v2 uses frames `[0, 42)` as the observed prefix. The other four objects have
later first-contact frames and retain exactly six post-contact observations. This
difference was anticipated by the v2 lock and is now recorded as a calibration
limitation: persistence may be especially strong after the 42-frame prefixes.

## Frozen execution

- Implementation revision:
  `781e73168489d401bb1f90343d95410925d960d8`
- V2 causal-schedule recovery lock:
  `56a6ebc0ac65e19c098ccfa83cda1be9990c579b396d46cfdd13c52bc5f0530e`
- Parent visual execution lock:
  `87b1efe7dc7e9a8f1fd4163e0a4164b5ba45e6bcc47d66e2fba2a2526c6f51e9`
- Causal-window manifest ID:
  `9fe5fdf4ae6449182d2e5064ad99417b4252dd04b76831df722b44614c2351dd`
- Manifest file SHA-256:
  `7398575f32ea8868f241da9356264d5b44b815f41425f6f259afcbbd10f336de`
- Server artifact:
  `/home/florianpfaff/source-only/deform360-official-hub-causal-windows-v2-781e7316.json`

The exact window, camera panel, and input-file hashes are archived in
`results/sota/deform360_official_hub_causal_window_feasibility_v2/causal_window_manifest.json`.

## Decision

The minimum 8-object support gate passes, so the locked Prob4D/MotionCrafter
provider may now run on all ten calibration objects. This result authorizes only
provider inference and registered calibration comparisons. It does not authorize
confirmation-payload access, which remains conditional on the later accuracy,
regret, covariance, finite-group, and tactile-ablation gates.
