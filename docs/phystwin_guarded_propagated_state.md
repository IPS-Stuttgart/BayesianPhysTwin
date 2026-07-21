# Guarded Action-Propagated State Diagnostic

## Purpose

This diagnostic tests whether the successful rank-4 graph-persistence
correction should instead enter PhysTwin as an uncertain state update. It is a
separate Bayesian-PhysTwin development branch and does not alter the frozen
Causal4D claim or the Prob4D observation feeder.

The correction is inferred at the pre-intervention endpoint. Position and
velocity perturbations are propagated through official Warp under the recorded
action, yielding finite-response columns

\[
R_k(t)=x_t(\xi_0+\Delta\xi_k,u)-x_t(\xi_0,u).
\]

The allowed response prefix is modeled as

\[
O_t-H(x_t)=\sum_k R_k(t)w_k+U_4c+\epsilon_t.
\]

The state weights and persistent graph bias are inferred jointly with a robust
Student-t likelihood. Perception reliability is fixed before the innovation is
evaluated, dense nodes and frames have effective-sample caps, and the state
branch is rejected when its response is confounded with the persistent bias.
The raw coefficient covariance is propagated to the readout for diagnostics,
but is explicitly not called calibrated.

## Forecast-Blind Guard

Only the endpoint and six permitted post-intervention prefix frames are used.
The first four frames fit the candidate. The final three prefix frames compare
it with graph persistence fitted at the fit-subset endpoint. The state branch
must improve prefix RMSE by both 5% and 0.25 mm. It is then refitted on the
complete prefix. Rejection reuses the frozen graph-persistence particle and
global trajectories byte-for-byte.

This is an exhausted-case implementation diagnostic. The guard is not a
source-calibrated regret certificate and no prospective mechanism promotion is
claimed.

## Development Result

Official Warp was rerun for all 24 rank-4 position/velocity perturbations and
four Bayesian particles on the three previously exhausted sloth interactions.
Values are future means in millimetres; coverage is coordinate-wise nominal
90% coverage with the existing variance floor.

| Case | Method | CD | Track | Coverage |
|---|---|---:|---:|---:|
| `single_lift_sloth` | BPT | 19.087 | 27.247 | 37.1% |
|  | Graph persistence | **14.770** | **24.851** | **43.8%** |
|  | Raw propagated state + bias | 17.215 | 25.933 | 41.6% |
|  | Guarded | **14.770** | **24.851** | **43.8%** |
| `double_lift_sloth` | BPT | 20.861 | 30.708 | 32.3% |
|  | Graph persistence | **14.958** | **25.280** | **44.1%** |
|  | Raw propagated state + bias | 20.009 | 29.721 | 34.4% |
|  | Guarded | **14.958** | **25.280** | **44.1%** |
| `double_stretch_sloth` | BPT | 5.886 | 9.468 | 89.1% |
|  | Graph persistence | 5.807 | **8.135** | 87.9% |
|  | Raw propagated state + bias | **5.726** | 9.133 | **90.2%** |
|  | Guarded | 5.807 | **8.135** | 87.9% |

Case-balanced, graph persistence is 11.845 mm CD and 19.422 mm track error.
The raw propagated branch is 14.317 mm and 21.596 mm, respectively: a 20.87%
CD regression and an 11.19% track regression against persistence. The guarded
branch remains exactly at the persistence values.

The raw branch shrinks the fitted persistence coefficient energy by
99.55%, 98.66%, and 98.52%, respectively, but its held-out-prefix changes are
-25.94%, -18.50%, and -12.71% relative to persistence. All three state updates
are therefore rejected. The guarded result is exactly the frozen persistence
result in all three cases.

## Interpretation

The experiment separates explanatory shrinkage from predictive value. A state
model can absorb almost the entire fitted readout correction and still predict
worse after physical propagation. Consequently, correction shrinkage is a
necessary mechanism diagnostic, not a sufficient promotion criterion. This
supports the existing v3 rule that shrinkage and held-out prediction must both
pass.

The result also complements the camera-only common-mode impossibility result:
neither a coherent camera update nor an unanchored state reinterpretation can
be trusted solely because it explains the prefix. A credible next state update
needs independent state/contact evidence or a genuinely fresh, source-calibrated
panel. The 19-case exploratory cohort is not run for this branch because zero
of three development cases passes the forecast-blind gate.

## Commands

Run one case from its frozen localization directory:

```bash
bpt-diagnose-phystwin-propagated-state \
  /path/to/PhysTwin \
  /path/to/phystwin-discrepancy-localization-v1/single_lift_sloth \
  /path/to/output/single_lift_sloth
```

Aggregate case summaries:

```bash
bpt-aggregate-phystwin-propagated-state \
  /path/to/output/aggregate.json \
  /path/to/output/*/summary.json
```
