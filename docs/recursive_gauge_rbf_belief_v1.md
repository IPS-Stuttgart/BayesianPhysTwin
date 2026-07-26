# Recursive gauge-aware RBF belief

Status: implemented inference primitive with synthetic controls. No empirical
accuracy or state-of-the-art claim is attached to this module yet.

## Motivation

The frozen Prob4D and Deform360 studies establish two constraints on a useful
online observation update:

1. a camera-consistent displacement can still be a shared observation bias;
2. a static endpoint correction cannot represent action-dependent contraction,
   rotation, reversal, or mode leakage.

`bayesian_phystwin.recursive_gauge_rbf_belief` combines the existing
query-identifiable gauge update with a full-covariance recursive spatial state.
It is designed for causal prefix observations and leaves the physical baseline
unchanged whenever the update is not identifiable or fails a safety gate.

## State and observation model

The latent correction state contains one global translation and one local
three-dimensional coefficient for every fixed material center:

```text
s_t = [g_t, a_1,t, ..., a_K,t].
```

A normalized Gaussian RBF decoder maps this state to an arbitrary material
query `q`:

```text
d_t(q) = g_t + local_blend * sum_k w_k(q) a_k,t.
```

The caller supplies an action-conditioned linearized transition and process
covariance:

```text
s_t^- = F_t s_(t-1)^+
P_t^- = F_t P_(t-1)^+ F_t^T + Q_t.
```

`F_t` may encode contraction, rotation, reversal, and leakage between global
and local modes. The implementation retains the complete covariance rather
than propagating independent marginal variances.

Each observation update is expressed as an increment around the predicted
state. `ObservationBeliefV1` supplies metric conditional covariance,
low-rank correlated factors, correlation groups, and residual-independent
reliability. The adapter exposes low-rank factors as explicit nuisance
variables and adds shared and centered per-view bias modes. The gauge-aware
solver then estimates only query-identifiable state directions using a robust
Student-t update.

Association probabilities remain diagnostic; they are not converted into
prior perception reliability. The physical innovation is processed once in
the robust likelihood.

## Independent anchors

Camera-only observations cannot distinguish a true global translation from an
equal shared camera bias. The update therefore abstains from a global state
change when that direction is unidentifiable. An independent sparse anchor,
such as trusted depth, tactile contact, or another held-out modality, may
identify the state translation because it observes state but not camera bias.

## Exact fallback

An update is rejected when it has no observation support, has no identifiable
query-state direction, exceeds the gauge-aware physical-response gate, or
exceeds the total query-correction cap. A rejected update:

- returns the physical candidate with identical dtype and bytes;
- does not assimilate the rejected measurement;
- retains the action-predicted latent belief internally, including process
  uncertainty, so a later valid observation can still update it.

The exact output fallback is a local safety property. It is not a prospective
non-regression certificate; a deployable selector still requires a locked,
source-calibrated regret guard.

## Implemented controls

The focused test suite verifies:

- separation of local physical deformation from shared camera bias;
- abstention for an unanchored global translation with byte-exact fallback;
- recovery of global translation when an independent anchor is supplied;
- action-conditioned reversal of retained local state;
- rejected-measurement behavior with continued process propagation; and
- full-covariance propagation through a non-orthogonal transition.

Existing gauge and adapter tests separately cover correlated duplicate
observations, covariance conservatism, residual-independent reliability, and
robust outlier handling.

## Evidence boundary

The three released PhysTwin interactions and all opened Prob4D/Deform360
cohorts are development evidence. Held-v8 artifacts are outside this module's
authority and must not be inspected or altered. The next empirical step is a
source-only smoke test whose observation stream, transition construction,
acceptance thresholds, and future metrics are locked before scoring.

Advancement requires improvement over the unchanged physical/persistence
baseline, acceptable calibration, and no material tail regression. Failure
must be archived as a negative result without retuning on the scored future.
