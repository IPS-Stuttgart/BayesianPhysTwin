# DEFORM decision-directed virtual sensing v1

This source-test-only pilot asks whether a physical twin can acquire **less
state information than a state-estimation policy** while still certifying the
same finite downstream decision.

No new data are collected. The pilot uses only the official DEFORM DLO4/DLO5
training trajectories. For each DLO, 48 trajectories fit the source support and
8 disjoint trajectories form the source-test cohort. The 14 official evaluation
trajectories are not opened.

## Virtual sensing task

The baseline observation contains:

- the two endpoint pairs at the current and previous frames;
- the registered future endpoint-action path over a 25-frame horizon.

The eight internal nodes are masked. Revealing one virtual sensor exposes that
node's current line-relative 3-D position and one-frame line-relative velocity,
all taken from the already recorded prefix. Future internal nodes remain hidden
until every acquisition path and finite action has been frozen.

The finite action portfolio is unchanged in spirit from the existing
decision-identifiability study:

1. exact endpoint-based physical fallback;
2. half of a source-derived residual correction;
3. the full source-derived residual correction.

A quotient posterior and the exact worst-compatible-belief regret certificate
determine whether an action is admissible. Candidate measurements are selected
by one of:

- expected decision-regret reduction;
- expected full-state variance reduction;
- expected task-query variance reduction;
- fixed center-out order;
- deterministic random order;
- a diagnostic oracle that knows the currently masked prefix readout, but never
  the future target.

## Primary question

At matched measurement budget, does decision-regret acquisition:

- certify more useful nonfallback actions;
- achieve lower realized regret or RMSE;
- require fewer internal-node measurements;
- act while more source hypotheses and future-state variance remain unresolved?

A positive pilot motivates one separately frozen replay on the official
evaluation split. A negative pilot is retained and no evaluation split is
opened.

## Claim boundary

This is source-test-only virtual sensing on existing public trajectories. It is
not unseen-object generalization, learned-vision validation, continuous-control
certification, deployment safety, or state of the art.
