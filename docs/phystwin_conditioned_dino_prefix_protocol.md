# PhysTwin-conditioned DINO prefix competence protocol

Status: locked before the first DINO prediction.

This one-case source control tests a narrow missing capability identified by the
opened PhysTwin evidence: deployable material correspondence. It does not test a
new simulator, fit a future residual, or claim Bayesian-PhysTwin improvement.

## Prediction boundary

The predictor receives:

- manual identities 3, 4, 6, and 8 at frame 114 only;
- their released PhysTwin node trajectories over frames `[114, 121)`;
- causal RGB, depth, object masks, calibration, and intrinsics over the same
  prefix.

Manual identity values on frames 115--120 are stored in a separate hashed
artifact. They cannot be opened until the complete prediction archive and
report have been hashed and sealed. No frame at or after the released training
boundary 121 is admitted.

## New method under test

The physical prediction restricts each DINO descriptor search to a 56-pixel
neighborhood. It does not contribute confidence. Prior perception reliability
uses only residual-independent evidence:

- DINO descriptor similarity and assignment entropy;
- local normalized patch correlation and its assignment entropy;
- distance from the object-mask boundary;
- valid metric depth and local depth spread;
- agreement among independently calibrated views.

Descriptor-candidate and patch-candidate mixture spread remain in pixel
covariance and are propagated through RGB-D unprojection into square metres.
Views are combined by covariance intersection because their errors are
correlated and the correlation is unknown. A shared 5 mm common-mode bias floor
is added once after fusion. Duplicating a pixel block or camera therefore
cannot create arbitrary confidence.

Distance between an observation and the PhysTwin state is deliberately absent
from prior reliability. A later Bayesian assimilation may process that
innovation once through the existing robust mixture likelihood. This competence
control stops before assimilation.

Every rejected identity-frame uses the released physical prediction with
byte-exact fallback.

## Frozen gate

The route advances only if all conditions hold on frames 115--120:

- at least 50% of eligible identity-frames have accepted multiview support;
- accepted observations improve at least 10% over the physical prediction on
  those same rows;
- exact-fallback candidate output does not regress overall;
- accepted-observation RMSE is at most 15 mm;
- last-two-frame candidate RMSE is at most 15 mm.

A pass authorizes only a separately locked source-panel assimilation study. A
failure stops this descriptor, patch, depth, and covariance configuration
without tuning it against the withheld prefix.

## Claim boundary

This is an already-open source interaction and a sparse online-supervised
competence check. It cannot establish independent transfer, covariance
calibration, state-of-the-art performance, or a Bayesian-PhysTwin result. The
three released future interactions, PokeFlex targets, and all held-v8 artifacts
remain outside the experiment.
