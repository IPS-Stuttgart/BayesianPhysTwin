# Deform360 tactile-only development diagnostic

**Status:** completed retrospective public-real-data development run. The result is
negative for the exact generic tactile forecasting route. It does not authorize a
paper claim, fresh confirmation, target retuning, or a causal interpretation.

## Execution identity

- source revision: `5404860a581d799a0628e720e2e4b012a26a4268`;
- request revision: `aecbf40f3fcb884c527d1d651f234de18bd78e75`;
- workflow run: `33323425375`;
- self-hosted job: `99289313698`;
- runner: `[self-hosted, Linux, X64, gpuserver4090]` / `workstation1`;
- artifact: `9735595047`;
- artifact SHA-256:
  `8b42866d21e6c1f071e685151de352c4d6001577c57493c4d7402877b9eca4dd`;
- compact retained record:
  `evidence/deform360/deform360_real_evaluation_gpuserver4090_2026-08-30.json`.

The request bound the exact reviewed source revision. The hosted authorization,
causal-prefix tests, future-poisoning test, reserved-object exclusion, Ruff checks,
and request-shape checks passed before the self-hosted job started.

## What the mounted release supported

The registered name-only inventory found:

| Carrier | Candidate count |
| --- | ---: |
| cleaned point-cloud sequence | 0 |
| fixed-identity trajectory archive | 0 |
| raw tactile stream | 7,123 |

The pilot therefore evaluated twelve normalized tactile-field streams from six
physical objects. It did not evaluate reconstructed geometry, fixed material
points, robot actions, physical parameters, or interventions.

## Result

| Method | Mean field RMSE |
| --- | ---: |
| persistence | **0.110624** |
| causal-prefix Bayesian lag mixture | 0.141185 |
| last residual | 0.200134 |

The Bayesian-minus-best-baseline difference was `+0.0305611`; lower is better.
The case bootstrap 95% interval was `[+0.0280810, +0.0329088]`, and the
object-balanced interval was `[+0.0275658, +0.0336689]`. The Bayesian candidate
beat persistence in `0/12` cases and beat the best registered baseline in `0/12`
cases.

The uncertainty output is also unusable as calibration evidence. Mean normalized
joint NEES was `95,396.68` and NLL per dimension was `47,691.24`, despite
`98.72%` marginal nominal-90% coverage. High marginal coverage therefore did not
indicate a credible joint predictive distribution.

## Scientific disposition

This is a useful stop result. It rules out the tempting but weak paper direction
of presenting a generic mixture of tactile temporal increments as a Bayesian
physical twin. The same twelve streams must not be used to tune lag sets,
temperatures, covariance floors, activity windows, or object selection and then
be re-labelled as confirmation.

The result does **not** test the main ecosystem hypothesis. Prob4D contributes
most when it preserves joint spatial or gauge dependence; BayesianPhysTwin
contributes most when observations revise a complete physical belief with an
exact fallback; Causal4D contributes most when that belief changes a held-out
interventional query or decision. A normalized single-sensor field forecast
contains none of those decisive links.

## Contribution that would be materially larger

The stronger paper question is:

> Does a dependence-preserving observation belief from one physical action revise
> an object-specific Bayesian twin in a way that improves a registered query or
> decision under a different action, while a prospective risk gate returns the
> exact physical fallback when the update is unsupported?

This connects the three repositories end to end:

1. **Prob4D:** produce matched joint draws or a shared low-rank factor for the
   processed source episode. Preserve cross-point and shared-gauge dependence;
   retain a diagonalized version as an ablation.
2. **BayesianPhysTwin:** abduct object-specific physical/contact parameters from
   the source action while keeping the target episode's initial state separate.
   Audit the proposed Gaussian information update and use byte-exact fallback for
   unsupported queries.
3. **Causal4D:** evaluate candidate target actions or safe probes by expected
   reduction in one preregistered physical-query or finite-decision Bayes risk,
   not by entropy over the complete latent label.
4. **Held-out target:** seal every target forecast before opening the future of a
   different-action episode from the same object.

The contribution is not the classical value-of-information formula or another
small average point-error gain. It is the demonstrated composition of a
**dependence-bearing 4-D observation, complete uncertain physical twin,
cross-action transport, task-conditioned intervention value, prospective risk,
and exact fallback**.

## Decisive Deform360 experiment

Use the already retained deterministic eight-object development design in
`evidence/deform360/query_validation_readiness_gpuserver4090_2026-08-30.json`.
Each object has one source episode and one target episode from distinct action
families. Keep all target futures closed while completing the following stages.

### Stage A — official source-only processing

Pin `lhy0807/deform360` revision
`d8522a4403b766aeb387510c04e89032a56fdf35`. Materialize only the allowed source
episodes through the official alignment and annotation contracts needed for:

- synchronized robot/gripper trajectories and registered action identity;
- cleaned point-cloud sequences or PhysTwin control points;
- track validity and frame provenance; and
- tactile alignment only when it contributes to the source belief.

No target future is needed to establish whether this source adapter is complete,
numerically stable, and compatible with the physical hypothesis bank.

### Stage B — frozen source qualification

Before target access, freeze:

- physical/contact hypothesis bank and prior;
- source-prefix cutoff and target forecast horizon;
- query functionals and units;
- Prob4D joint-belief representation and diagonalized ablation;
- BayesianPhysTwin information audit, competence score, threshold, risk cap, and
  exact fallback;
- Causal4D candidate-action or probe set;
- all baselines, promotion margins, and object-level bootstrap procedure.

The source gate must require usable geometry on a fixed minimum number of sheet
and volumetric objects, finite and nondegenerate joint uncertainty, no future
leakage, and no source evidence that the guarded policy is materially harmful.
Failure is terminal for the exact adapter and protocol.

### Stage C — target-closed prediction sealing

For every eligible source-target pair, generate and hash all target predictions
without opening the target continuation. The primary arm must use the complete
joint belief. Register these comparisons:

- physical or released-model fallback;
- persistence;
- last residual / constant velocity;
- source-local discrepancy without cross-action transport;
- diagonalized Prob4D belief;
- complete joint Prob4D belief;
- generic-information probe selection;
- task-conditioned risk-gated selection;
- fixed-safe and random cost-matched probing; and
- wrong-object and action-label-permutation negative controls.

A marginal-preserving dependence control should independently permute matched
probe-outcome and held-out-query draws within source-frozen exchangeability
strata. It must preserve both marginals while destroying only their pairing.

### Stage D — held-out evaluation

Open each target future once and evaluate at the physical-object level. Primary
outcomes should be registered downstream quantities, not an undifferentiated
state average:

- held-out query log score or energy score;
- query RMSE in physical units;
- finite decision regret;
- accepted-update or accepted-probe harm rate with an exact upper bound;
- coverage, interval width, and joint NEES;
- fallback/abstention rate; and
- worst-object and worst-action-family regret.

Frames, points, coordinates, cameras, and taxels are nested observations, not
independent statistical units. Report paired object-level intervals and preserve
all negative subgroups.

## Promotion rule

A larger paper claim is warranted only when the complete, risk-gated chain:

1. improves the registered held-out query or decision against the strongest
   simple and physical fallbacks;
2. retains a bounded harmful-accept rate and exact fallback custody;
3. outperforms the diagonalized and source-local ablations;
4. loses its advantage under the dependence-destruction control; and
5. preserves the direction across sheet and volumetric object strata.

A result that only improves raw tactile one-step prediction, only changes
uncertainty width, or only wins against a weak deterministic reference is not the
intended contribution.

## Claim boundary

The completed pilot establishes only a negative normalized-tactile development
result and the absence of registered processed geometry on the mounted tree. The
cross-action experiment above remains a staged protocol direction until official
source processing passes, predictions are sealed, and target outcomes are opened
under a separately reviewed authorization. Existing bounded BayesianPhysTwin,
Prob4D, and Causal4D results remain unchanged.
