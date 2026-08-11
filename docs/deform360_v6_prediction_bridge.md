# Deform360 v6 target-blind prediction bridge

## Purpose

The protected source producer first seals the complete 10-by-10 v5 nested
prediction batch. That artifact is necessary for nested source fitting, but it
is not itself the ten-unit v6 prediction batch because the v6 roster contains
the dynamic endpoint model average and three covariance interpretations of one
VT1 mean.

`deform360_fresh_object_session_v6_prediction_bridge` closes this custody gap
without reading a development suffix. It does not compute a candidate forecast;
it validates already-produced, source-only candidate artifacts and binds them to
the exact pre-existing v5 held-out prediction for the same physical object-session.

## Candidate-panel boundary

For each of the ten source units, one content-addressed panel binds:

- the exact v5 execution lock, 100-record prediction batch, and held-out v5 seal;
- the unchanged B0 and B1 prediction identities from that held-out seal;
- the frozen D1 native model-average prediction and covariance;
- the VT1 working-IRLS, observed-information, and group-sandwich covariance
  variants;
- the exact other-nine source fit roster, risk score, threshold, guard, interval,
  covariance, and source-artifact identities; and
- the candidate-production revision that also produced the sealed v5 batch.

All available VT1 covariance variants must share one point prediction, fit, risk
score, threshold, and guard decision. They may differ only in covariance and
interval identity. The D1 native variant cannot be silently marked unavailable.
A source-owned VT1 covariance failure remains an explicit unavailable variant
and therefore scores exact physical fallback later.

A panel fails closed when it substitutes either baseline, changes the held-out
object, changes the source fit roster, mixes implementation revisions, changes a
guard decision between VT1 covariance variants, or binds a different source
execution amendment.

## Ten-unit bridge

Exactly ten validated panels are converted into ten ordinary v6 prediction
seals and one existing-schema v6 prediction batch. The bridge receipt separately
binds:

- the bridge implementation revision;
- the v5 execution lock and 100-record prediction-batch identities;
- all ten candidate-panel identities;
- all ten v6 prediction-seal identities; and
- the final v6 prediction-batch identity.

The receipt fixes every target-access flag to `false`. Creating the batch does
not authorize source suffix scoring, source candidate selection, metadata-only
fresh-target selection, target payload opening, or target outcome access.

## Command line

```bash
python scripts/science/\
  run_deform360_fresh_object_session_v6_prediction_bridge.py bridge \
  --policy protocols/locks/\
    deform360_official_hub_fresh_object_session_v6.json \
  --covariance-amendment protocols/amendments/\
    deform360_official_hub_fresh_object_session_v6_source_covariance.json \
  --source-execution-amendment protocols/amendments/\
    deform360_official_hub_fresh_object_session_v6_source_prediction_execution.json \
  --selection protocols/locks/\
    deform360_official_hub_visuotactile_v1_selection.json \
  --v5-execution-lock protocols/locks/\
    deform360_official_hub_joint_sparse_source_execution_v5.json \
  --v5-prediction-batch /sealed/v5/source-prediction-batch.json \
  --candidate-panel /sealed/v6/panel-00.json \
  --candidate-panel /sealed/v6/panel-01.json \
  --candidate-panel /sealed/v6/panel-02.json \
  --candidate-panel /sealed/v6/panel-03.json \
  --candidate-panel /sealed/v6/panel-04.json \
  --candidate-panel /sealed/v6/panel-05.json \
  --candidate-panel /sealed/v6/panel-06.json \
  --candidate-panel /sealed/v6/panel-07.json \
  --candidate-panel /sealed/v6/panel-08.json \
  --candidate-panel /sealed/v6/panel-09.json \
  --bridge-revision "$(git rev-parse HEAD)" \
  --output-directory /sealed/v6/prediction-bridge
```

The output directory contains:

```text
source-seals/<object-id>.json
source-prediction-batch.json
bridge-receipt.json
```

Every file is create-once. An existing destination causes failure rather than
replacement.

## Scientific boundary

This bridge supplies the missing lossless transition between the already-sealed
nested v5 source panel and the frozen v6 candidate/covariance roster. It does not
show that D1 or VT1 is accurate, calibrated, stable, or physically beneficial.
Those claims require the registered source suffix gate and, after a source pass,
one independent evaluation on the separately selected fresh object-session
cohort. No opened source or target outcome may be used to alter a candidate
panel, covariance method, guard, fit roster, or threshold.
