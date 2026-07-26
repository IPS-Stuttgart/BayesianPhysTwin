# Prob4D causal observation stream contract v2

Bayesian-PhysTwin validates a strict Prob4D observation artifact before forming
any physical innovation. The neutral archive remains
`phys4d.observation_belief` version 1; the provider-specific interpretation is
versioned separately by `prob4d_causal_stream_contract_version`.

## Supported forms

- **Legacy contract v1:** seven `gauge_latent_*` columns in an independent factor
  group for each retained window. This remains available for frozen experiments.
- **Joint contract v2:** canonical `joint_gauge_latent_####` columns in one shared
  factor group. The latent vector represents the joint cross-window `Sim(3)`
  covariance propagated from a fixed metric anchor through the selected causal
  sequential gauge tree.

Contract v2 is admitted only when the artifact agrees on its factor rank, full
`7K` gauge dimension, retained covariance-trace threshold, parent lineage,
sequential gauge mode, fixed metric anchor, and non-approximate boundary
covariance. Approximate fixed-lag covariance is rejected from the strict stream.

Prob4D 0.2.0 produced canonical joint factors before adding the explicit stream
version. Those transitional artifacts remain recognizable, and the validation
result records `stream_contract_version_inferred = true`. New Prob4D exports
carry the explicit version and complete anchor schema.

The validation result is copied into the gauge-aware observation batch metadata,
so Bayesian updates and downstream Causal4D lineage can identify the exact
provider contract that was admitted.
