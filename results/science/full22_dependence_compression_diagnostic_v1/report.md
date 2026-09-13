# Full-22 dependence/compression diagnostic v1

This diagnostic reads sealed covariance predictions only; it reads no future
truth, score, prefix array, protected target, or held-v8 artifact.

## Result

- Cases: `22`
- 3D covariance blocks: `3638553`
- Equal-case mean total correlation: `0.00155312532` nats
- 95% case-bootstrap interval: `[0.0010322945390123276, 0.0020976713899215032]`
- Median case rank-one relative total-correlation error: `0.000462294`

## Gate

- Dependence signal supported: `false`
- Local rank-one fidelity supported: `true`
- Whole-object dependence testable: `false`
- Fused headline supported: `false`

## Representation boundary

The archive contains independent per-point 3x3 covariance blocks, not one
dense covariance over a physical object. A symmetric 3x3 covariance needs six
parameters, and a free diagonal plus one 3-vector factor also needs six; local
rank-one fidelity therefore is not strict parameter compression by itself.

This target-free retrospective diagnostic can quantify the Gaussian dependence encoded inside each exported 3x3 point covariance. It cannot establish realized log-score superiority, joint whole-object tolerance value, regularized-empirical-covariance superiority, cross-track dependence, physical-cause identification, fresh transfer, or a Bayesian-specific paper claim.
