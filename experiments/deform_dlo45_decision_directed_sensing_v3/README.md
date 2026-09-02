# DEFORM decision-directed virtual sensing v3

This study adds two pieces of evidence that were missing from v2:

1. **Non-overlapping source-test replication.** The complete v2 operating point
   is fixed: likelihood scale 2, action-prototype scale 1, support-regret
   tolerance 0.05, and four virtual measurements. A new deterministic
   fit/calibration/source-test split is used. Its source-test filenames must have
   zero overlap with the preceding v2 source-test cohort.
2. **Trajectory-level transport calibration.** The exact certificate controls
   regret only inside the registered finite local support. V3 estimates a
   separate additive transport slack from the 18 complete calibration
   trajectories. The nonconformity score is the trajectory mean of the positive
   difference between realized normalized regret and the finite-support
   certificate. A one-sided finite-sample split-conformal quantile is then
   evaluated on the 16 complete replication trajectories.

The physical replay remains unchanged. Each candidate virtual measurement
reveals the already recorded current line-relative 3-D position and one-frame
velocity of one internal DLO node. Future internal-node trajectories are opened
only after all acquisition paths and actions are frozen. Official DLO4/DLO5
evaluation files remain absent.

## Why this is stronger

The v2 result showed that expected decision-regret acquisition outperformed
state variance, query variance, posterior entropy, class entropy, Bayes risk,
fixed center-out, and random acquisition on one source-test cohort. V3 asks
whether the same fixed operating point replicates on a disjoint cohort and
separates two logically different uncertainties:

- **within-support decision ambiguity**, handled exactly by the quotient regret
  certificate; and
- **support-to-target transport error**, handled empirically at the complete
  trajectory level.

V3 also reports exact paired sign tests with Holm correction in addition to the
DLO-stratified paired trajectory bootstrap intervals inherited from v2.

## Interpretation boundary

The transport envelope targets trajectory-mean normalized regret under a
complete-trajectory exchangeability assumption. It is not a per-frame or
per-decision safety certificate. The replication split was chosen using file
identifiers only, but the study remains within the public DEFORM training
partition. It does not establish official evaluation performance, unseen-object
generalization, learned-sensor competence, continuous-control safety, or state
of the art.
