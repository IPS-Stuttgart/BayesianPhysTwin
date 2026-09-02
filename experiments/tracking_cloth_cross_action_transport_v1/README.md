# Tracking Cloth cross-action transport gate

This experiment is the first real-data consumer of the repository's
explain--transport--probe--abstain separation. It asks a narrower question than
online active manipulation:

> After observing one physical interaction with a cloth, can a response library
> predict a different held interaction without forcing a unique material-cause
> label?

The experiment uses scale-normalized pairwise-distance trajectories of the 20
tracked cloth points. This query is invariant to global translation, rotation,
and uniform scale. Repetition 1 supplies the complete source library. A single
interaction is selected by leave-one-material-out source performance. Only that
interaction is numerically parsed from repetition 2 before predictions for the
other two interactions are sealed.

## Source-only model

For each interaction, the four repetition-1 material trajectories are embedded in
a PCA basis fitted only to repetition 1. The diagnostic response is represented
as a deviation from the source action mean. Its source design is the matrix of
centered material responses. Centering creates an affine coefficient ambiguity:
adding the same constant to every material coefficient changes neither the
observed diagnostic response nor any centered held-action target. Pairwise
geometry also annihilates rigid pose gauge directions.

Consequently, a held-action query can be identifiable over the complete
coefficient ambiguity set even though the registered coefficient explanation is
not unique. The formal certificate therefore returns
`transport_without_cause` when the diagnostic response lies inside the frozen
source span and the target map annihilates every remaining null direction.

A repetition-2 diagnostic outside the source span returns `none_of_the_above`.
A target that is not invariant returns the exact action-mean fallback and a
source-only intervention recommendation, never a hypothetical post-probe
correction.

## Frozen controls

The scored prediction is compared with:

- the repetition-1 action mean;
- known-material repetition persistence, reported only as a reference ceiling;
- direct copying of the diagnostic interaction;
- the other held action used as a wrong-action mapping; and
- all 24 associations between the four diagnostic material prototypes and the
  four target material prototypes.

The last control gives an exact finite permutation test. Its smallest attainable
one-sided p-value is `1/24`.

## Information order

```text
repetition 1: all 12 material × interaction recordings
    -> PCA, diagnostic selection, source span, target maps, thresholds
repetition 2: selected diagnostic interaction only (4 recordings)
    -> coefficient sets, dispositions, eight held-action predictions
    -> write and hash model.npz + predictions.npz + seal.json
verify prediction seal
repetition 2: open the other two interactions (8 recordings)
    -> score once against frozen predictions and controls
repetition 3: zero numeric reads
```

This is a retrospective source-target development gate, not an independently
blind confirmation: repetition 2 was available to earlier project analyses. A
positive result can justify only review of a separate repetition-3 protocol. It
does not itself authorize repetition-3 access.

## Execution

```bash
PYTHONPATH=src python -m \
  experiments.tracking_cloth_cross_action_transport_v1.run \
  --stage seal \
  --dataset-root /path/to/tracking-cloth-deformation-v1-zenodo-14644526 \
  --work-dir /tmp/tracking-cloth-cross-action-v1

PYTHONPATH=src python -m \
  experiments.tracking_cloth_cross_action_transport_v1.run \
  --stage score \
  --dataset-root /path/to/tracking-cloth-deformation-v1-zenodo-14644526 \
  --work-dir /tmp/tracking-cloth-cross-action-v1 \
  --output result.json \
  --report report.md
```

The scientific decision is stored in `result.json`. A negative gate is a valid,
successfully executed result and therefore does not make the workflow fail.

## Claim boundary

The result concerns only the frozen trajectory-geometry query, three recorded
action families, four named materials, separate repetitions, source PCA basis,
local linear response library, and declared controls. It is not same-episode
probe--reset--act evidence, unrestricted causal identification, natural material
ground truth, nonlinear simulator closure, deployment validation, or state of
the art.
