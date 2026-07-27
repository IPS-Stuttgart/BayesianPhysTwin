# ObservationBeliefV1

`ObservationBeliefV1` is the narrow, versioned interface between a 4-D
perception feeder and Bayesian-PhysTwin. It is a content-addressed, non-pickled
NPZ artifact. The same schema is emitted by Prob4D and can be validated by
Causal4D without importing either provider.

Descriptor metadata is copied, normalized through finite canonical JSON, and
exposed as recursively immutable dict/list-compatible containers. Mutating
caller-owned containers after construction, or attempting nested mutation
through the belief, therefore cannot change an existing artifact content
address.

## Information boundary

Every row carries an absolute source-frame ID, and the descriptor declares an
exclusive `causal_frame_stop`. Construction and loading fail when any declared
or observed frame reaches that boundary. The artifact therefore cannot silently
mix reconstruction-only future frames into a predictive update.

The identity tuple `(frame_id, entity_id, view_index, window_index)` must be
unique. Association probability and prior reliability are stored separately.
The prior nominal probability used by the robust likelihood is supplied per
effective group and is not recomputed from the physical innovation.

## Structured covariance

For observation row `i`, the represented covariance is

```text
C_i,local + U_i U_i^T
```

with cross-row covariance

```text
Cov(i, j) = U_i U_j^T
```

when rows share a `factor_group_id`. Prob4D uses this to retain the
seven-dimensional uncertain `Sim(3)` gauge of each overlap window. Local 3x3
blocks remain available for anisotropic along-ray/lateral uncertainty.

`correlation_group_ids` define the multivariate likelihood blocks. Each block
has a frozen prior nominal probability and a composite-likelihood weight in
`(0, 1]`, allowing dense pixels or overlapping windows to be capped rather than
counted as independent samples.

## Bayesian-PhysTwin likelihood

`grouped_student_t_mixture_likelihood` evaluates, for every effective group,

```text
rho_g t_nu(r_g; 0, Psi_g)
+ (1-rho_g) t_nu(r_g; 0, lambda_out Psi_g)
```

where `Psi_g = (nu-2)/nu C_g`, so `C_g` is the covariance of the nominal
Student-t component. Block-diagonal local covariance plus shared low-rank
factors are evaluated with Cholesky solves and the Woodbury identity. Independent
factor groups are accumulated as separate rank-sized systems, so neither a dense
covariance matrix nor a dense all-groups factor matrix is formed.

The posterior nominal responsibility may depend on the residual. The supplied
prior nominal probability never does. Association support is reported as a
separate diagnostic. The prior-aware solver is checked against this density in
[the likelihood conformance suite](likelihood_conformance.md). The strict
minimax solver intentionally retains a different rowwise Student-t power
objective and reports that distinction in its diagnostics.

## Gauge-aware state adapter

`build_gauge_aware_batch_from_observation_belief` connects this neutral
artifact to the query-identifiable state update without collapsing its
uncertainty structure:

- `mean_xyz_m - physical_prediction_xyz_m` forms the innovation once;
- `local_covariance_m2` remains the conditional metric covariance;
- each `(factor_group_id, factor_name)` becomes an explicit standard-normal
  nuisance parameter, so `low_rank_factor_m` is not added to local covariance
  a second time;
- row reliability, group nominal probability, and composite-likelihood weight
  remain distinct residual-independent inputs;
- association probability is retained only as a diagnostic;
- the default shared bias is a global 3-D translation;
- default camera biases are zero-sum Helmert translation contrasts, avoiding a
  duplicate global-translation column.

An unanchored global state translation is therefore indistinguishable from the
default shared observation bias and triggers exact fallback. Independent
anchors or query-relevant state modes outside the nuisance subspace can make a
state update identifiable.

## Strict Prob4D causal stream

An artifact whose repository and stream identify
`FlorianPfaff/Prob4D` and `prob4d:causal-overlap-window-points` receives an
additional provider-independent admission check before the physical innovation
is formed. Bayesian-PhysTwin verifies that:

- the source revision is exact rather than `unknown`;
- the seven gauge factor names and factor-group/window mapping are unchanged;
- coordinates are metric and the declared world frame is nonempty;
- a fixed external metric anchor is bound to the first selected payload;
- the causal-lineage cutoff equals the descriptor cutoff;
- no future prediction payload was opened;
- the source product is independently decoded overlap windows;
- every selected window is in descriptor order, has valid source bounds before
  the cutoff, and contains all rows assigned to it; and
- the lineage source digest equals the descriptor source digest.

The validation result is copied into the gauge-aware batch metadata. A generic
ObservationBelief from another feeder remains governed by the neutral schema;
provider-specific claims are never inferred from its repository name alone.

## Commands

Validate an artifact:

```bash
bpt-validate-observation-belief observation_belief.npz
```

Score a prediction aligned row-for-row with the artifact:

```bash
bpt-validate-observation-belief observation_belief.npz \
  --predicted-npz physical_prediction.npz \
  --predicted-key predicted_xyz_m \
  --summary-json outputs/observation_score.json
```

The artifact digest covers the descriptor, every array name, dtype, shape, and
byte payload. Prob4D, Bayesian-PhysTwin, and Causal4D contain the same golden
contract fixture to detect incompatible schema changes.
