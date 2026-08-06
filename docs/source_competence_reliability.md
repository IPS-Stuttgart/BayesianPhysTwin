# Target-blind temporal source competence

## Motivation

`ObservationBeliefV1.prior_reliability` is a source-side prior about whether an
observation row is nominal. For Prob4D this can be obtained from a separately
calibrated source-reliability model using overlap disagreement, uncertainty,
window position, scene flow, validity, and related provider diagnostics.

A static row probability does not represent persistent source failure. Camera
occlusion, poor triangulation, gauge instability, mask drift, and provider
breakdown often continue across neighboring observations. BayesianPhysTwin
already contains a binary Markov reliability model, but before this interface
it was used only inside residual replay and synthetic benchmarks rather than at
the portable observation boundary.

`bayesian_phystwin.source_competence_reliability` adds a narrow target-blind
composition:

```text
source-calibrated row reliability
             +
target-blind competent/incompetent evidence
             +
source/track temporal persistence
             |
             v
Markov posterior competence
             |
             v
min(provider prior, Markov posterior)
```

The last operation is deliberate. Target-blind temporal evidence may reject or
downweight a provider observation, but it cannot promote a row beyond the
provider's independently calibrated prior reliability.

## Evidence contract

`SourceCompetenceEvidenceV1` binds:

- the exact `ObservationBeliefV1` artifact ID;
- the exact ordered `(frame, view, window, entity)` row-identity digest;
- the exclusive causal frame cutoff;
- a content-addressed source-feature artifact;
- a content-addressed source-reliability model;
- ordered sequence IDs and time values;
- competent and incompetent log densities for every row; and
- the feature names and finite JSON metadata used to interpret the sidecar.

The contract rejects evidence that declares use of:

- confirmation or target outcomes;
- the BayesianPhysTwin physical innovation;
- downstream posterior inlier responsibilities; or
- association probability as a competence label.

Those prohibitions keep three mechanisms separate:

```text
source competence       persistent provider condition
association probability material-identity uncertainty
robust responsibility   innovation-dependent local outlier handling
```

Using the same physical residual first to decide source competence and then
again in the robust likelihood would count downstream evidence twice. The
source-competence sidecar therefore has to be produced from source/calibration
features before the physical update is formed.

## Temporal model

`SourceCompetenceMarkovConfigV1` content-addresses the inlier and outlier
persistence probabilities, probability floor, time-delta semantics, time step,
and conservative composition rule.

The historical `order-only` mode applies one transition between neighboring
observations after stable per-sequence ordering. `integer-steps` raises the
transition matrix to the positive integer number of elapsed time steps. It is
appropriate for dropped frames or irregular sampling when the supplied times
are exact integer multiples of the registered `time_step`.

For each sequence, `smooth_markov_reliability` combines the provider prior with
the two unary log densities and returns the smoothed competence probability and
normalized sequence evidence. Deployment then uses

```text
r_deployed[i] = min(r_provider[i], p_competent[i] | source evidence).
```

This is not a second observation likelihood. It replaces only the row-level
prior reliability used by the later grouped robust physical update.

## Invariants

`refine_observation_source_competence(...)` creates a new content-addressed
`ObservationBeliefV1`. It verifies that the following remain byte-equivalent in
value:

- observation means and all row identities;
- local covariance and low-rank coherent factors;
- association probability;
- correlation and factor groups;
- group nominal probabilities and composite weights; and
- declared frames and the causal cutoff.

Only `prior_reliability` and metadata change. Metadata records the source
observation, evidence, feature, model, and Markov-config identities, the
conservative composition rule, every forbidden-information declaration, and
the fact that covariance and association probability were not changed.

The retained evidence and result arrays are backed by immutable bytes. Their
writeability cannot be restored with `array.setflags(write=True)` after content
identity construction.

## Usage

A provider or adapter first constructs or loads a sidecar from source-only
features:

```python
import numpy as np

from bayesian_phystwin.source_competence_reliability import (
    SourceCompetenceEvidenceV1,
    SourceCompetenceMarkovConfigV1,
    refine_observation_source_competence,
)

sidecar = SourceCompetenceEvidenceV1(
    observation_artifact_id=observation.artifact_id,
    observation_identity_sha256=ordered_row_identity_sha256,
    source_feature_artifact_id=feature_artifact_id,
    source_reliability_model_id=reliability_model_id,
    causal_frame_stop=observation.causal_frame_stop,
    feature_names=(
        "overlap_disagreement",
        "triangulation_condition",
        "track_age",
    ),
    sequence_ids=tuple(track_ids),
    time_values=np.asarray(frame_times, dtype=np.float64),
    log_competent_density=np.asarray(log_competent, dtype=np.float64),
    log_incompetent_density=np.asarray(log_incompetent, dtype=np.float64),
    metadata={
        "uses_truth": False,
        "feature_semantics": "source-only-provider-diagnostics-v1",
    },
)

update = refine_observation_source_competence(
    observation,
    sidecar,
    config=SourceCompetenceMarkovConfigV1(
        inlier_persistence=0.98,
        outlier_persistence=0.90,
        time_delta_mode="integer-steps",
        time_step=1.0,
    ),
)
refined_observation = update.refined_observation
```

The JSON sidecar can be retained with:

```python
from bayesian_phystwin.source_competence_reliability import (
    load_source_competence_evidence,
    write_source_competence_evidence,
)

write_source_competence_evidence(sidecar, "source-competence.json")
restored = load_source_competence_evidence("source-competence.json")
```

Loading rejects duplicate JSON keys, nonfinite values, schema drift, changed
content identities, and noncanonical booleans. Publication is atomic and does
not overwrite an existing path unless explicitly requested.

## Recommended real-feeder features

The contract is feature-agnostic, but suitable target-blind inputs include:

- independent-view count and triangulation condition;
- reprojection or multiview cycle residuals;
- Prob4D overlap disagreement and gauge-cycle diagnostics;
- track age, termination risk, and source-only association entropy;
- segmentation-boundary distance and occlusion status;
- source-only covariance and retained-support diagnostics;
- agreement with an action-conditioned physical-response direction, provided
  that this is frozen on source/calibration data and does not use the current
  downstream innovation; and
- presence of an independent tactile, depth, contact, or synchronization anchor.

The source-feature artifact and reliability model must be selected without
confirmation outcomes. Sequence definitions and persistence parameters must
also be frozen before confirmation access.

## Claim boundary

This interface establishes exact row binding, temporal source-competence
composition, conservative reliability capping, non-duplication of association
and covariance semantics, and target-blind provenance. It does not establish
that the feature model detects every coherent camera bias, that a provider is
competent on a fresh cohort, that raw posterior covariance is calibrated, that
a physical-state update is identifiable, that a guarded physical query
improves, that exact fallback is a universal safety theorem, that Causal4D
interventions improve, or that the system is state of the art.
