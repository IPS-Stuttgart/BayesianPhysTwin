# DLO-Lab Slingshot posterior-aware policy certificate development v2

## Question

The first prospective policy-level certificate stopped before evaluation futures:
its five-neighbor geometry predictor admitted only 12 of 288 prefixes after a
global rank-88 conformal correction. This development study asks whether the
failure came from the policy-specific certificate or from an impoverished
competence feature.

All inputs are public DLO-Lab simulation artifacts that were already opened:
the original 51-world belief/control source panel and the failed study's 96
calibration worlds. Its 288 evaluation prefixes are used only for an admission
capacity check. Their futures remain absent and unread.

## Posterior-aware feature

The successor feature concatenates two causal blocks:

1. 51 shared-bias-invariant relative-geometry and temporal-increment values;
2. 110 posterior diagnostics: joint- and iid-bias particle weights, five
   incumbent-relative loss vectors, and the three registered raw regret bounds.

The feature does not use a future reward, hidden world parameter, or residual
against a future state. A deterministic distance-weighted nearest-neighbor
predictor estimates the gain of the already-fixed posterior policy action. An
exact feature match averages only exact reference matches; otherwise the seven
nearest rows receive inverse-distance weights.

The certificate itself remains ordinary one-sided split conformal. No local
variance rescaling is promoted: a source-only floor sweep reduced rather than
improved stable guarded value. Rejection still returns the incumbent action
exactly.

## Development design

Four fixed variants are compared over 30 deterministic seven-fold rotations.
Each rotation uses 105 worlds for fitting, 21 disjoint worlds for conformal
calibration, and 21 disjoint worlds for evaluation, with every source world
evaluated once per rotation. Model selection first minimizes median harmful
admissions among candidates with median coverage at least 0.85 and median harm
at most one, then maximizes median guarded gain.

The selected `combined_distance_k7` model has:

- median guarded gain `+0.004774` over the complete 147-world denominator;
- minimum guarded gain `+0.001604` across the 30 rotations;
- median 18 admitted worlds and zero harmful admissions;
- median marginal lower-bound coverage `90.48%`;
- median gain `+0.001406` above the geometry-only uniform-five predecessor.

The worst rotation contains four harmful admissions, so this is not safety
evidence. It is a model-selection diagnostic whose only purpose is deciding
whether a fresh prospective test is warranted.

## Prefix-only capacity

A separate descriptive check fits the selected model on all 147 opened source
worlds and obtains its correction from seven-fold out-of-fold residuals. It
admits 44 of the 288 already-opened prefixes and falls back on 244. No future of
that panel is generated or read. This clears the predeclared development
capacity threshold of 24, but it is not split-conformal coverage evidence
because the same opened source collection was used for model development.

## Decision and boundary

The source advancement gate passes and justifies one entirely fresh public
simulator protocol. The next protocol must train on these 147 worlds, use new
world and sensor seeds for calibration and evaluation, freeze the seven-neighbor
model before calibration, and seal all evaluation decisions before any fresh
evaluation future is generated.

This result does not alter the terminal v1 negative, certify Slingshot, establish
backend-wide competence, claim an official benchmark or state of the art, or
support robot safety. It uses no recordings, protected target, held-v8, DLO4,
or DLO5 material.
