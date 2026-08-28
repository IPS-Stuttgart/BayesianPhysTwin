# Reference-centered DEFORM transport: failed source screen

## Decision

Do not promote or retry this arm. The sole primary, reference-centered paired
transport, fails all six registered value checks on the already-open DLO2
source object. The existing incumbent and successful paired update are unchanged.
No DLO1/DLO3 transfer, protected evaluation, new recording, or GPU run follows.

The pre-outcome implementation is
`c1e31520b43d7ccccf65b719941dcf2e3c776c37`; the protocol remains unchanged at
`configs/sota/deform_reference_transport_source_v1.json`. This result note and
the additional arithmetic audit are explicitly post-outcome additions, not a
retroactive repair of that protocol.

## Matched point results

All arms use the same two initial full states, frozen model/readout, prescribed
clamp actions, and eight 3D material-point prefix observations. Four disjoint
hidden identities are scored. These are released metric identities, not an
automatic RGB sensing result. Fourteen already-open trajectories completed;
the registered design case is excluded, leaving thirteen equally weighted
trajectories. Forecast frames 50:170 are partitioned into three 40-frame bins.

| Arm | Coordinate L1 (mm) | Point RMSE (mm) | Late RMSE (mm) | FDE (mm) |
|---|---:|---:|---:|---:|
| Unchanged incumbent | 10.673 | 25.614 | 27.524 | 19.379 |
| Previous paired update | 9.594 | 23.066 | 27.089 | 19.519 |
| Reference initialization only, diagnostic | 9.593 | 23.041 | 27.034 | 19.497 |
| Repeated reference centering, primary | 10.927 | 26.693 | 31.389 | 24.002 |

The primary increases mean point RMSE by **15.72%** and late RMSE by **15.87%**
relative to the previous paired update. It wins jointly on L1/RMSE in only
3/13 trajectories; its worst trajectory RMSE ratio is 1.711916. The frozen
10,000-replicate paired trajectory bootstrap gives an RMSE-difference interval
of **[0.827971, 6.904887] mm**, entirely on the worsening side. This is a
conditional interval on one already-open object, not population confirmation.

The initialization-only diagnostic improves RMSE by only 0.11%. It cannot
replace the preregistered primary or rescue the failed gate. Full per-trajectory,
horizon, and arm metrics are retained in the compact result artifact.

## Accounting and programmed controls

- Ordinary prediction-generation successes: 14/14.
- Retained prediction technical failures: 0.
- Unsealable predictions, replacements, or omitted source trajectories: 0.
- Scored non-design trajectories: 13/13.
- Programmed pre-score controls: 6/6 pass.
- Registered primary value checks: 0/6 pass.
- Original post-score arithmetic verifier: **FAIL**, separately retained below.

The incumbent, previous paired predictions, and zero-reference reduction are
byte-identical to their registered references. Zero innovation returns the
original incumbent object. Nominal CPU replay differs from the archived replay
by at most 0.003636 mm, with coordinate RMSE 0.000407 mm, within the predeclared
limits. The fourteen forecasts were sealed before source future truth scoring.

## Commanded-clamp contract failure

The frozen checker `scripts/verify_deform_reference_transport.py` stops at line
189: it requires every native clamp position to equal the supplied position
command. The maximum difference is **10.869026 mm**, also present in the
unchanged native replay. This is not the much smaller CPU/archive replay error.

Read-only inspection of pinned upstream `DEFORM_sim.py` explains a mechanism:
evaluation assigns the clamp commands before its length-constraint projection.
In `applyInternalConstraintsIteration`, when the first endpoint of an edge is
clamped, the loop can move the second endpoint without first testing whether
that endpoint is also clamped. This applies to adjacent prescribed clamp pairs.
The returned native trace therefore does not establish literal command equality.
The exact upstream file SHA-256 is
`3a8fe59ec4500f0b882099673df37650c9fc26396f273dfc53930c8412d3bd08`.

The original method note's statement that clamp positions are exact is too
strong if read as exact agreement with supplied commands. It is retained as
frozen provenance, with this correction recorded separately. No upstream code,
command, tolerance, prediction, or original verifier is changed. This is an
additional qualification failure, not grounds for a rescue run or a claim of
fully qualified physical-model falsification.

## Explicitly post-result audit

`scripts/audit_deform_reference_transport_posthoc.py` reproduces the original
checker failure and then checks a different, clearly labeled invariant: all
native clamp positions in both propagated branches remain byte-identical to
the unchanged native replay. It does not substitute this weaker invariant for
the registered exact-command check.

The audit verifies all 672 bound source files and 43 sealed array members,
all recorded centering/readout identities, 624 case/arm/horizon metric values,
the bootstrap, win count, and failed decision. Maximum metric-arithmetic
difference is 7.11e-15. It records `original_second_arithmetic_passed=false`,
`literal_command_exact_check_passed=false`, and `promotion_authorized=false`.
It is a second arithmetic implementation, **not independent human review**.
Neither it nor the transfer verification runs a new native prediction.

Verification: 528 DEFORM tests pass locally, including six new synthetic
posthoc-audit regressions. The exact frozen server runtime passed 23 focused
tests before the native run. Ruff, focused MyPy, and diff checks pass.

## Evidence and interpretation

Compact evidence is in `results/sota/deform_reference_transport_source_v1/`.
The full immutable native run remains on gpuserver4090 at
`/home/florianpfaff/source-only/deform-reference-transport-source-v1/run-v1`.
Its downloaded copy and source snapshot were rehashed without changing that
run root. The separate posthoc receipt lives outside the frozen run root.

| Artifact | Identity |
|---|---|
| Lock canonical ID | `07f17d1895c63c8f944c7febb1068f7f17e1c9306c168f01afac16b09167e1c8` |
| Prediction seal file SHA-256 | `50463f7908f63b737e1e6b9f51e7947068f84f65ba4a016d6ae4a730c563ace0` |
| Prediction NPZ file SHA-256 | `c7b7003110bcd11a62fe53d13cffe5d7ca4a1444fa0ab4a85145bab5950cd4a3` |
| Original result canonical ID | `efafc2177c453f1822e59ac6e19fa3d58e7361d39410c83ae23af4c7be194ed4` |
| Posthoc audit canonical ID | `4a21cde4f09ff6f0efa4e7cc2c10a3753bb31c537425855523b664dee597ead9` |

For this fixed implementation, initializing propagation at the learned readout
has little point-metric effect, while repeated common position/velocity
re-centering is harmful. This does not reject all lifted-state, render-aware,
or Bayesian online-update models. It supplies a negative ablation and a native
control-contract diagnostic, not a larger positive contribution, calibrated UQ,
counterfactual validation, fresh confirmation, or SOTA evidence. The previous
paired update remains the retained method. Nothing is pushed or merged to main.
