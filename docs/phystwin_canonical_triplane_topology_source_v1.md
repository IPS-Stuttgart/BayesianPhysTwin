# Canonical tri-plane spring field with piecewise topology

Status: implementation and source-only protocol in progress. No benchmark
future or held-v8 artifact is authorized by this protocol.

## Why this route remains open

The published NeuSpring ablation reports future CD/track values of
`10.1/20.5 mm` for its neural spring field alone, `11.4/22.3 mm` for piecewise
topology alone, and `8.7/17.5 mm` for their combination. The interaction is
therefore the material claim. The public NeuSpring repository is still a
two-line placeholder at commit
`51d94f67ed1e2557fca29c1e86b418506e3d51ca`; no official code, checkpoint,
license, or executable data contract is available.

Bayesian-PhysTwin previously rejected several adjacent families:

- a rank-16 canonical RBF spring field;
- a fixed sparse topology;
- a 25-member regional topology/field bank; and
- a canonical tri-plane *output-residual* dynamics model.

Those are meaningful negative results, but none continuously optimizes a
spatial spring field on an object-specific piecewise topology. This protocol
tests that remaining mechanism directly inside the official Warp simulator.
It is an independent paper-described implementation, not an official
NeuSpring reproduction.

## Sparse tri-plane field

Object-spring midpoints are transformed into a deterministic right-handed PCA
frame and normalized to the canonical bounding box. Every spring queries four
bilinear neighbours from each of three scalar planes:

```text
log k_e = log k_e,teacher + (f_xy(e) + f_yz(e) + f_xz(e)) / 3.
```

Only twelve indices and weights are stored per spring. This avoids a dense
spring-by-parameter matrix when the paper rule
`N = round(0.85 * sqrt(number of object springs))` yields hundreds of grid
cells per axis. Controller springs use one separate coefficient. Zero plane
and controller coefficients exactly recover the supplied teacher field.

The first implementation deliberately uses scalar planes rather than
NeuSpring's unpublished 32-channel planes and three-layer MLP. It is a cheap
capacity gate: if even the direct spatial field cannot improve an untouched
prefix suffix, the larger nonlinear decoder is not justified. If it passes,
the nonlinear decoder becomes a preregistered follow-up rather than an
unbounded first attempt.

## Staged decision

Stage 0 verifies exact identity, future-mutation invariance, controller/object
separation, and a gradient direction. Stage 1 fits the field on the exact
released topology for the already-open `single_lift_sloth` development case.
It must improve both official prefix-suffix metrics, achieve at least 3%
balanced improvement, and stay within a factor-three stiffness correction.

Only a passing Stage 1 authorizes Stage 2. Stage 2 selects one five-region
topology from a fixed 25-member source-only bank, fits homogeneous object and
controller scales during topology selection, and then fits the unchanged
tri-plane field on the selected topology. The exact teacher remains a first-
class candidate and fallback.

The source panel contains four already-open transfer objects. At least three
must improve both metrics, equal-case CD and track must both improve by at
least 3% in balanced ratio, no case metric may regress by more than 5%, and
every selected field must pass the geometric plausibility bound. Failure
closes the family without a 19-case future run.

The first native stage-0 run exposed a PCA sign ambiguity for symmetric
spring-midpoint sets: orienting an eigenvector by its largest absolute
projection left an exact positive/negative tie to eigensolver rounding. The
implementation was amended before source transfer to orient each axis by the
first nonzero spring-identity projection instead. This is a technical repair
to the preregistered rigid-frame contract; no split, optimizer, field size, or
acceptance gate changed. The development result produced before this check
passed is inadmissible and is retained only as engineering provenance.

## Claim boundary

The development and source suffixes are model-development evidence. A passing
source gate would authorize only an exploratory run on the repeatedly examined
PhysTwin cohort. It would not establish state of the art. A SOTA claim would
still require a separately locked independent evaluation under the same
future-hidden metric contract.

Prob4D remains an optional observation feeder and Causal4D remains a separate
intervention-abduction project. Neither is used here. All held-v8 runtime,
target, query, score, barrier, and outcome artifacts remain outside this task.
