# Deform360 reusable-PhysTwin protocol v1

## Objective

The fastest credible route to a state-of-the-art result is to fill the missing
physics-based entry in Deform360's multi-episode benchmark. Deform360 reports
that per-episode PhysTwin is the strongest 3D predictor in its Table 3, with
future Chamfer distance 0.014 m and track error 0.025 m. PhysTwin is omitted
from Table 4 because it requires per-episode registration. ParticleFormer is
therefore the reported multi-episode reference at 0.051 m CD and 0.079 m track
error.

The proposed result is not a larger residual network. It is a reusable physical
twin that:

1. registers one canonical object graph automatically to each new initial state;
2. pools physical evidence over several source actions;
3. propagates a Bayesian parameter ensemble;
4. admits simulator response only through an outcome-independent trust policy;
5. returns exact persistence wherever the physical response is rejected.

This directly attacks the reason Deform360 excludes PhysTwin from
multi-episode evaluation. Automatic registration alone is not a novelty claim:
PGRD also constructs a canonical spring graph, registers it to each episode,
and fits physics across episodes. The distinct claim here is low-data transfer
on public Deform360 objects with Bayesian physical support, outcome-independent
admission, calibrated uncertainty, and exact fallback when the simulator is not
trusted. PGRD remains the required hybrid-residual comparator, although its
published numbers come from another benchmark.

## Prospective panel

The panel was selected from object names in the Deform360 Hugging Face snapshot
`7fea8e20231a47641d1d2bc8791920ec4e62ec5e` before downloading or processing
any listed object outcome.

| Role | 1D | 2D | 3D |
| --- | --- | --- | --- |
| Development | rubber band, paracord, shoelace, metal chain | pink cloth, shirt, bag, paper cloth | dog, sponge, rubber duck, octopus |
| Confirmatory | nylon rope, string | mask cloth, bubble-wrap cloth | sloth, rubber toy |

For every object, fit episodes are `1,3,4,6,7,9` and held episodes are
`0,2,5,8`. This gives three unimanual and three bimanual fit actions, followed
by two unimanual and two bimanual held actions. The confirmatory result contains
6 objects and 24 held episodes.

## Information boundary

The 12 development objects may be processed after this lock. Their results are
used to freeze registration QA, the physical candidate grid, Bayesian support,
and all trust thresholds.

For each confirmatory object, the six fit episodes may be opened only after the
method is frozen. They identify that object's reusable twin. A held episode may
provide only its initial object state and the released robot action trajectory
before prediction. Future object video, point clouds, tracks, and tactile data
remain unavailable until the prediction archive has been checksummed.

The checksummed [action-window addendum](deform360_reusable_sota_window_v1.md)
uses only robot action, aperture, and episode length to select an 81-frame
compute slice. It changes neither the object split nor the held-out information
boundary; the selected held episode still contributes only one object frame
before its prediction is sealed.

An early outcome reveal invalidates that object without replacement. Penguin
episodes `0,2,5,8` and the PokeFlex target remain under their existing seals.
Nothing in this protocol changes the frozen Causal4D claim.

## Required controls

The central scientific claim is that pooled object physics transfers across
actions. Four arms are therefore mandatory:

| Arm | Purpose |
| --- | --- |
| Constant persistence | no-dynamics baseline |
| Single-fit-episode physical selection | tests whether pooling adds value |
| Pooled physical twin without trust | measures raw simulator transfer |
| Pooled physical twin with frozen trust | prospective deployed method |

The single-episode control uses the same physical candidate grid independently
on every fit episode. All six choices are reported, along with their median; a
favorable source episode may not be selected using held outcomes.

Candidate selection uses the equal-weight dimensionless score
`0.5 * track/persistence_track + 0.5 * CD/persistence_CD`. Pooled selection
averages this score over all fit actions. The source diagnostic also performs
leave-one-fit-action-out selection and compares it with candidates selected
from each remaining single action. Held episodes evaluate those frozen indices
only; they cannot refit either the pooled or single-action choices.

The tested temporal learned residual is excluded from the primary arm. Its
5,000-step source-only run converged in training but failed every useful
transfer threshold, and its calibrated utility gate chose exact abstention.
That source test is not a faithful reproduction of PGRD's Point Transformer,
sliding-window transformer, data scale, or rollout training. It therefore
supports abstaining from this residual, not a claim of superiority to PGRD. A
faithful PGRD-style control is required once evaluator-compatible processed
annotations are available.

## Gates

The trusted pooled arm must satisfy every internal transfer gate:

- at least 5% lower future CD and track error than persistence;
- at least 2% better than the median single-episode control;
- wins on at least 16 of 24 held episodes;
- no held episode degrades by more than 10%;
- 90% coverage lies between 85% and 95%;
- energy score improves.

Inference is clustered by object, retaining all four episodes inside each
bootstrap draw. Episode and category tables are still reported so a favorable
object class cannot hide failures elsewhere.

## State-of-the-art claim boundary

Passing the internal gate justifies a strong independently preregistered public
benchmark result. It does not by itself justify saying that the method beats
Deform360 Table 4. A direct state-of-the-art claim additionally requires the
same object split, temporal horizon, particle construction, metric definition,
and aggregation as the published evaluator. Those details are not currently
released with the baseline training code.

The direct comparison becomes valid only after obtaining or reproducing that
evaluation contract. PGRD remains a required related-work comparison, but its
published scores are from a different dataset and cannot be ranked numerically
against this panel.

## Decision

If the development panel does not reproduce the existing source transfer, stop
without opening confirmatory fit episodes. If development passes, freeze the
pipeline, process the six confirmatory fit sets, seal all 24 predictions, and
only then reveal held outcomes. If confirmatory transfer passes but evaluator
parity is unavailable, report the result honestly as an independent protocol
and request the official split/evaluator before making a Table 4 claim.

References: [Deform360](https://arxiv.org/abs/2607.05390),
[PGRD](https://arxiv.org/abs/2607.13451).
