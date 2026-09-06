# Nonlinear GP residual pilot on DLO2 development data

This isolated experiment does not change production inference or any registered
terminal protocol. It reads only the original 40 fitting and eight validation
trajectories from DLO2/train. Source-test and official-evaluation trajectories
are excluded. The historical validation set is already development evidence;
this is not fresh independent confirmation.

## Frozen comparison

The checkpoint, baseline rollout, causal feature builder, and clamped-node
values are shared. The frozen checkpoint SHA-256 is
`b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65`.
The current per-node ridge uses ridge=1 and shrinkage=0.25. A ridge shrinkage
control is selected on the same inner split as the GPs.

Two finite-rank Nystrom Matern-3/2 GPs are evaluated: independent-node (64
inducing variables per node and coordinate), and material-aware (512 shared
inducing variables per coordinate for DLO2's eight internal nodes). The latter
uses a product with a Matern kernel on the existing normalized material arc
coordinate, length 0.5. The total latent rank per output coordinate is matched;
this does not match every inductive bias or effective degree of freedom.
All fitting rows enter the likelihood; inducing selection is deterministic and
outcome-blind. Hyperparameters use only 32 inner-fit and eight inner-validation
trajectories, obtained by hashing names within the original fitting roster.
Length scales {0.5,1.5}, normalized noise variances {0.05,0.5}, and correction
shrinkages {0.25,0.5,1} are fixed before execution. Feature scaling and response
scaling are fitted inside each fit partition. Exact duplicate causal queries
are collapsed; cross-partition causal duplicates cause failure.

Preparation, prediction, and scoring are separate processes. The prediction
process receives validation queries without validation target arrays and writes
a prediction hash before scoring. Raw predictions, model arrays, inner-selection
records, per-trajectory errors, and bootstrap differences are retained.

The endpoint is equal-trajectory mean coordinate L1 over all nodes/horizons,
matching the baseline operator. Complete trajectories are bootstrap units.
Raw GP marginal uncertainty is reported only as an uncalibrated diagnostic:
the working likelihood assumes independent rows, the finite-rank prior omits
the full-kernel remainder, and no simultaneous pathwise guarantee is claimed.
A GP posterior mean has a deterministic kernel-ridge equivalent; point-error
improvement alone cannot establish uniquely Bayesian value.

## Execution

The GitHub Actions workflow uses gpuserver4090 and the original frozen runtime.
No robot, new acquisition, Causal4D action selection, source-test, official
holdout, or production-model replacement is part of this experiment.
