# Genesis MPM source qualification result v1

## Decision

The exact Genesis World 1.3.3 MPM runtime passed the two-action source-physics
gate and failed the separately frozen source-value gate. Genesis is therefore
recorded as `source-physics-qualified`, but it is not source-value-qualified,
recommended, or allowed to replace an incumbent backend.

Both selected output archives are byte-identical copies of the registered
incumbent predictions. The source future outcomes were not opened. No DEFORM,
target, confirmation, Causal4D, or held-v8 artifact was read or modified.

## Frozen runtime and custody

The native runtime is Genesis revision
`0796d27667087d0087fe09d903f8aadf7fa9adeb`, package version 1.3.3, with
runtime ID
`aecd2a170f974a166495da0c8692631acebf09d7b605c4ec0f9621f49434132a`.
The source-physics runner was frozen at BayesianPhysTwin commit `f00b6b1e`.
The source-value implementation and protocol were frozen before outcome access
at commit `82353b14212598d9d0f5bec609117ffe84f0f77c`.

| Artifact | SHA-256 |
| --- | --- |
| Source-physics result | `e7e3a8172a4760a8ebc8f9cda16812c811037674dc08a9e2dc0b4810d826b0da` |
| Backend qualification | `cc263bb7890af19c1f7bdae40f6c5f701d90f105a1e9c70bf386d2accb39561d` |
| Sealed value grid | `caf35f48bd570ebcac836b5ccd37b9a22dd559ec810460710f567042afa3e2db` |
| Prefix decision | `657a3c2d72395f33a33e6dacdff2e619db4a12959a533a6f425ec572b6cf58d9` |
| Future no-open result | `3eacaa761b0ee4148f9600a4212c3e714e9d173fa7f3b55e0fcf9488dcbd8e0d` |

The native process emitted Genesis's upstream warning that Torch versions below
2.8 are unsupported. The exact pinned Torch 2.4 runtime is retained rather than
silently changed after the run; all registered deterministic and numerical
checks passed under that identity.

## Source-physics result

The final grid-aware source-physics gate passed every registered check.

| Source group | Action response | Parameter sensitivity | Time-step relative error | Grid-aligned equivariance | Off-lattice error | Zero drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `double_lift_zebra` | 1.170 mm | 0.496 mm | 4.50% | 0.000 pm | 0.312 mm | 0.000 mm |
| `double_stretch_zebra` | 0.086 mm | 0.048 mm | 6.75% | 0.000 pm | 0.015 mm | 0.000 mm |

Topology identity, source-query parity, deterministic replay, deformation
bounds, particle-step bounds, and exact incumbent fallback also passed for both
groups. Arbitrary sub-grid translation is reported as MPM discretization error,
not misrepresented as continuous rigid equivariance.

## Source-value result

The value arm was fixed to an equal-weight 25/100/500 kPa ensemble. All six
full-horizon trajectories were sealed before either prefix was opened. The table
reports the final third of each allowed prefix in millimetres; ES is the
equal-event 3D marginal energy score.

| Group | Method | Identity RMSE | Chamfer | ES |
| --- | --- | ---: | ---: | ---: |
| `double_lift_zebra` | Genesis ensemble | 13.675 | 10.795 | 17.695 |
|  | Registered incumbent | 7.710 | 3.535 | 6.867 |
|  | MatPhys comparator | 7.405 | 3.502 | 6.441 |
|  | Persistence | 22.856 | 17.246 | 25.047 |
| `double_stretch_zebra` | Genesis ensemble | 28.996 | 21.745 | 34.779 |
|  | Registered incumbent | 9.772 | 5.199 | 10.990 |
|  | MatPhys comparator | 9.093 | 4.538 | 8.695 |
|  | Persistence | 29.803 | 17.723 | 25.602 |

The equal-group balanced point ratio versus persistence was `0.856`, but the
worst group ratio was `1.100`, so transfer failed. The equal-group energy-score
ratio was `1.032`, also a failure. Relative to the incumbent, equal-group
identity and Chamfer ratios were `2.370` and `3.618`. Ensemble spread was finite
and nondegenerate at 7.20 mm and 3.80 mm, but uncertainty did not rescue the
poor mean or proper score.

## Consequence

Genesis remains useful as a real, simulator-neutral MPM integration and a
numerically qualified research backend. This particular frame-boundary
attachment bridge and fixed elastic ensemble are rejected as a competitive
source predictor. The negative value result does not justify tuning on these
two opened actions. A genuinely new Genesis arm would need a different physical
mechanism or contact/attachment formulation and a newly frozen source protocol.

DEFORM remains the protected strong result. The Genesis path shares only the
portable rollout contract and adds no branch to DEFORM's implementation,
configuration, selection, or evidence artifacts.
