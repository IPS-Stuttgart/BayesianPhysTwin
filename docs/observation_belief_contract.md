# ObservationBeliefV1

`ObservationBeliefV1` is the narrow, versioned interface between a 4-D
perception feeder and Bayesian-PhysTwin. It is a content-addressed, non-pickled
NPZ artifact. The same schema is emitted by Prob4D and can be validated by
Causal4D without importing either provider.

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
factors are evaluated with the Woodbury identity; a dense covariance matrix is
not formed.

The posterior nominal responsibility may depend on the residual. The supplied
prior nominal probability never does. Association support is reported as a
separate diagnostic.

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
