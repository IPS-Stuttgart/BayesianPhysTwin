# Tree-separator Gaussian shadow parity

`bayesian_phystwin.tree_separator_gaussian_parity` connects the independent
block-tree separator solver to the production `TreeBlockNormalSystemV1` without
changing the accepted robust-update path.

The production factorization remains authoritative for condition-number
admission, reusable covariance application, and existing result identities. The
independent solver is a shadow implementation that must agree on:

- every node and separator posterior mean;
- the separator marginal covariance;
- selected node marginal covariances;
- selected node-to-separator cross covariances;
- the precision log determinant; and
- the structured normal-equation residual.

The comparison never calls complete covariance materialization. For large trees,
it checks all posterior means and a deterministic set of at most eight node
covariance blocks. Callers can provide an explicit node roster for a frozen
experiment.

```python
from bayesian_phystwin.tree_separator_gaussian_parity import (
    require_tree_separator_gaussian_parity,
)

report = require_tree_separator_gaussian_parity(
    normal_system,
    maximum_condition_number=1.0e12,
)
```

The returned `TreeSeparatorGaussianParityV1` binds the exact normal-system
array bytes through `normal_system_id` and is content-addressed by `parity_id`.
It also records the production maximum node and separator condition numbers.
Scaled comparison metrics and the `passed` flag are validated for internal
consistency. The production condition-number gate runs first, so the shadow path
cannot authorize a system rejected by the historical node-elimination gate. The
adapter also covers the valid zero-separator case by constructing the equivalent
empty global factor after node elimination; it does not change the historical
production API.

This is numerical parity evidence only. It does not establish provider
competence, calibrated posterior uncertainty, physical-query benefit, Causal4D
intervention benefit, deployment safety, or state of the art.
