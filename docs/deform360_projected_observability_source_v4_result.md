# Deform360 Projected-Observability Source V4 Result

## Frozen question

Can a target-free planner select enough graph identities and cameras whose
prefix response is physically measurable to support a guarded online state
update?

The implementation and seven-case source panel were frozen at
`3401e1b05ed32a86804b95fed4f91f1bf2e67335`. The protocol required at least
five of seven source cases to pass admission. No prediction outcome was
permitted before admission.

## Result

The source gate failed:

| Result | Count |
| --- | ---: |
| Locked source cases | 7 |
| Admitted | 0 |
| Projected-observability multicover infeasible | 4 |
| Too few cameras with measurable projected response | 2 |
| No globally eligible physical-response point | 1 |
| Tracker runs | 0 |
| Future outcome evaluations | 0 |

All failures occurred in the target-free physical/camera planning stage. The
runner never reached AllTracker inference, candidate state updating, hidden
identity scoring, or future prediction metrics.

## Interpretation

The negative is stronger than the earlier one-case V3 rejection. Under the
frozen short prefix and 0.5 mm response requirement, camera-tangent
observability does not transfer across this seven-object source panel. Global
3-D physical response is therefore not enough: it may be too small in the
available camera tangents, concentrated in too few views, or unable to support
the required disjoint spatial multicover.

The fixed-threshold projected-observability family is closed. Lowering its
minimum response, camera count, or spatial support after these results would
turn an admission failure into post-outcome method selection.

## Consequence

This result does not evaluate the accuracy of a guarded Bayesian state update;
no update was admitted. It instead shows that the proposed camera-only
evidence interface cannot be applied broadly enough under the current action
windows.

The next credible SOTA attempt must change the information source or the
prospectively frozen acquisition/query design. Strong candidates are:

1. an independent metric modality such as sparse depth, tactile contact, or
   measured actuation that can break coherent camera-bias ambiguity;
2. a fresh-object, sequential query provider that waits for target-free
   physical excitation and preserves exact baseline fallback;
3. an action-conditioned latent common-mode bias model validated on source
   objects before any new target is opened.

The result does not authorize a SOTA claim, target evaluation, or any change to
held-v8.

## Provenance

- Config SHA-256:
  `39c6665bc148e0f5c05a2fec66a32f7ca35e81623ebdeaee090188a4f39c3ac0`
- Protocol note SHA-256:
  `90deee1ff53076131ed22c4ce87c9f22f4f17e59f976dcc4535890d377d9e2e0`
- Runner SHA-256:
  `e7f89f243b27198f0ea3265d1a2a8629c884ac3b3b517e3b2baafe688baf5ca0`
- Planner SHA-256:
  `772a6d885fda0c7e66c264553b216e25a8a06b389c33335fcdb154ef154af9eb`
- Compact result:
  `results/sota/deform360_projected_observability_source_v4/source_gate_summary.json`
