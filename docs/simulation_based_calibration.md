# Simulation-based calibration

`bayesian_phystwin.simulation_based_calibration` provides target-free diagnostics
for checking an inference implementation against its own frozen simulator and
prior. It complements parameter error, NEES, predictive coverage, and proper
scores by testing where the generating truth falls inside posterior sample
approximations.

## Statistical unit

Each row is one independently generated object/session replicate. Frames, views,
tracks, points, particles, and tactile taxels are repeated observations within
that replicate; they must not be presented as extra independent calibration
units.

For posterior draws with normalized weights `w`, the randomized
probability-integral-transform value is

```text
mass(samples < truth) + tie_breaker * mass(samples tied with truth).
```

The tie breaker is an explicit input so discrete or repeated posterior draws can
be handled reproducibly and a claim-bearing run can bind the randomization seed.

```python
from bayesian_phystwin.simulation_based_calibration import (
    SimulationBasedCalibrationSummaryV1,
    posterior_pit_matrix,
)

pit = posterior_pit_matrix(
    posterior_samples,  # (independent replicate, draw, parameter)
    generating_truths,  # (independent replicate, parameter)
    weights=posterior_weights,
    tie_breakers=registered_tie_breakers,
)
summary = SimulationBasedCalibrationSummaryV1(
    group_ids=independent_replicate_ids,
    parameter_names=("stiffness", "damping", "control_scale"),
    pit_values=pit,
    metadata={
        "generator_revision": generator_revision,
        "inference_revision": inference_revision,
        "target_outcomes_used": False,
    },
)
```

## Reported diagnostics

The summary is content addressed and retains immutable arrays for:

- PIT histograms;
- mean PIT;
- Kolmogorov--Smirnov distance from a uniform distribution;
- Cramer--von Mises discrepancy;
- central 50%, 90%, and 95% posterior mass coverage; and
- lower and upper 5% tail rates.

No asymptotic p-value is reported. With finite simulation counts, the raw
diagnostics, replicate count, generator identity, and stratification should be
reported together. Conditional failures should be examined by preregistered
corruption, horizon, action-support, contact, and identifiability strata rather
than hidden by one aggregate statistic.

## Information and claim boundary

Simulation-based calibration can detect implementation errors, posterior bias,
underdispersion, overdispersion, and tail asymmetry under the declared generator.
It does not establish that the simulator matches reality, that real observations
are exchangeable, that a physical parameter is identifiable, or that deployment
intervals have frequentist coverage.

Real-data uncertainty claims still require independent physical objects or
sessions, source-frozen policy selection, the registered finite-group calibration
procedure, complete technical-failure accounting, and a prospective target
opening. SBC must not consume Deform360 confirmation outcomes or be used to tune
a method after those outcomes are opened.
