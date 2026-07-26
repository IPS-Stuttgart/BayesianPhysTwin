# Fresh Deform360 Pairwise-Belief Outcome

## Status

The prospective fresh-object transfer gate failed. This is a model result, not
a pipeline failure:

- all 12 locked physical objects produced ordinary predictions;
- no case was replaced, retained as a technical-failure prediction, or left
  unsealable;
- the all-case completeness barrier passed before any `final_data.pkl` was
  deserialized;
- the analysis lock was frozen after that barrier and before outcome access;
- the one-shot scorer completed all 12 cases.

The 12 objects are now outcome-open and exhausted for subsequent model
selection.

## Primary Result

All values are object-balanced means in millimetres. There is one episode per
physical object.

| Arm | Hidden identity RMSE | Hidden symmetric Chamfer |
| --- | ---: | ---: |
| Persistence | 0.202 | 0.243 |
| Physical prior | 2.592 | 1.926 |
| Selected raw backbone | 0.899 | 0.772 |
| Pairwise-consensus RBF candidate | 2.709 | 2.477 |

The candidate:

- regressed against persistence on 12/12 objects for both metrics;
- regressed against the selected raw backbone on 12/12 objects for both
  metrics;
- beat the physical prior on only 2/12 identity-RMSE cases and 3/12 Chamfer
  cases;
- was worse than the physical prior in aggregate by 4.52% identity RMSE and
  28.64% Chamfer.

Against persistence, the object-bootstrap 95% difference intervals were:

- identity RMSE: `[+1.445, +3.635]` mm;
- Chamfer: `[+1.423, +3.097]` mm.

Against the selected raw backbone, they were:

- identity RMSE: `[+1.006, +2.683]` mm;
- Chamfer: `[+1.024, +2.445]` mm.

Every prospective comparison gate failed.

## Mechanism

The locked windows are close to static under the released object trajectories:
11/12 persistence identity errors are below 1 mm and the object-balanced mean
is 0.202 mm. Despite this:

- the current-observation selector chose the physical prior at 19/36 updates;
- the pairwise-consensus gate accepted 30/36 selected updates;
- the accepted RBF update then created future motion that was absent from the
  hidden material identities.

Pairwise distance preservation is therefore not a sufficient observation
admission criterion. A coherent common-mode triangulation bias can preserve
all pairwise distances and pass the clique gate while implying a false global
motion. Likewise, selecting a backbone by current sparse-observation Chamfer
does not bound its regret against an unchanged future baseline.

This independently reproduces the central limitation found by the selective
virtual-sensing study: under a near-static action window, camera-consistent
updates can be much worse than exact persistence.

## Consequence

Do not expand or retune this pairwise arm on these opened objects. The next
candidate must be a baseline-relative guarded Bayesian update with:

1. exact fallback to the unchanged baseline;
2. target-free dynamic-window admission based on registered contact, measured
   action, or predicted physical response;
3. a latent shared camera/time/spatial bias term;
4. observation redundancy that is independent of the common bias where
   possible;
5. a source-calibrated upper confidence bound on update regret relative to the
   baseline;
6. a sealed predictive covariance if calibration is to be claimed.

The fresh study does not establish official Deform360 SOTA parity. Its valid
claim is narrower: the previously positive open-27 pairwise-belief result does
not transfer to this prospectively locked, predominantly near-static
fresh-object cohort.

## Provenance

- Prediction protocol commit: `32f0fe13628336734f254bf350a9f2f4746372df`
- Outcome scorer commit: `0a01fb864d44feb8bf27be96bcbe339ebad45bd9`
- Completeness barrier result:
  `ac55fa454cf4351cca4ecaa680268607fc103bb33d58d382674f6e3a86ac6ca9`
- Outcome summary result:
  `18f4355bf2236e009532d24ceac3a47a349be151d6b317252d1d94817fdeae22`
- Native POSIX verification before outcome access: 844 passed, 3 skipped
- Changed-file Ruff verification: passed

Repository-wide Ruff also reports eight pre-existing findings in unrelated
legacy files; none is in the fresh protocol, scorer, adapter, or tests.
