# Held-out deformation uncertainty v1

## Question and fixed information order

Does a joint predictive belief transfer to new deformation quantities better than separately calibrated diagonal uncertainty and a whole-trajectory residual bootstrap, at exactly the same predicted mean?

This is an offline, retrospective source-only pilot using real DEFORM DLO4/DLO5 recordings. It needs no robot, new recording, optional marker observation, active sensing, or counterfactual outcome. Each original source split has 39 fit, 9 calibration, and 8 held-out source-test trajectories. Official evaluation partitions and unrelated reserved campaigns remain closed.

The point predictor is the existing source-qualified DEFORM hybrid plus BayesianPhysTwin's frozen local residual correction. The physical checkpoint and point adapter are not retrained or selected. The exporter verifies their original source seals and reproduces the saved source-test candidate to absolute tolerance 1e-8 m.

**This tests a new covariance-only empirical-Bayes layer around that fixed twin. It is not an evaluation of the twin's original stored covariance.**

## Bayesian predictive layer

Let each complete residual trajectory yield a 96-dimensional vector: eight free nodes, three coordinates, and four fixed forecast horizons. The mean is fixed at zero to preserve the caller's point prediction. Use an inverse-Wishart covariance prior with `nu0 = d + 3` and `Psi0 = 2 D`, where `D` is the diagonal fit-trajectory residual second moment with floor 1e-8 m^2. Nine independent calibration trajectory vectors update this prior. The posterior predictive distribution is multivariate Student t with 13 degrees of freedom and covariance `(E.T E + 2 D) / 11`.

The empirical-Bayes prior uses only fit data. Raw covariance is never claimed calibrated automatically. Every method independently chooses its scalar scale by leave-one-calibration-trajectory-out centroid CRPS; the held-out trajectory in each inner fold is excluded from covariance and bootstrap construction.

## Matched controls

The two primary controls are a marginal-matched diagonal Student distribution and a centered **whole-trajectory** residual bootstrap. The latter transforms the same residual trajectories into every requested quantity; it is not denied the ability to answer new queries. Additional controls are the symmetric whole-trajectory bootstrap and a covariance-matched Gaussian. All five predictive means are identical.

Centering the empirical residual bank enforces the fixed mean. The symmetric bank provides a separate sensitivity analysis that preserves raw second moments without changing that mean. No coordinatewise or framewise resampling is used.

## Queries and endpoints

There are 99 declared linear quantities: 12 centroid quantities for scale calibration, plus 12 relative front/back displacements, 72 local second differences, and three late-minus-early centroid displacements held out from scale fitting. These are recorded-position readouts, not claims that physical curvature or material parameters are directly observed.

The primary outcome is Brier score for absolute quantity displacement exceeding 5, 10, or 20 mm. The three held-out families receive equal weight, then trajectories and DLOs receive equal weight. CRPS, coverage, interval width, log score, event prevalence, and identical point errors are supporting diagnostics. Student/Gaussian event probabilities are analytic and empirical probabilities exact; continuous CRPS uses a fixed 513-point quantile quadrature.

Success requires a negative paired Brier contrast on each DLO and a negative upper bound for both primary comparisons using 97.5% two-sided intervals (Bonferroni correction for two comparisons). The 10,000 bootstrap repetitions resample complete trajectories within each DLO. They are conditional on these **two objects**, not an independent-object population test.

## Execution and evidence

A request-only change to `.github/requests/heldout-deformation-uncertainty-v1.json` triggers the dedicated workflow on `[self-hosted, Linux, X64, gpuserver4090]`. Code and protocol digests are checked before export. After source fit and calibration for both objects, the scorer writes a complete method seal before loading any source-test arrays. Negative scientific results finish successfully and are retained; they do not authorize tuning or retry after scoring.

The artifact contains the request, protocol, logs, method seal, result JSON, per-trajectory scores, and per-event probabilities. Raw prediction/observation carriers remain on the runner, outside the uploaded artifact.

## Interpretation limits

This is a 16-trajectory source-test pilot on previously studied public objects. It cannot establish fresh confirmation, unseen-object transfer, superiority of every Bayesian formulation, calibrated robot safety, recovery of latent physical parameters, or counterfactual validity. A covariance benefit beyond diagonal uncertainty but not beyond a trajectory bootstrap supports a narrower conclusion than a Bayesian-specific advantage.
