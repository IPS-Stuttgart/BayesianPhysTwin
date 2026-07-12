# Causal4D with PhysTwin and MolmoMotion

This milestone replaces the controlled Causal4D simulator with the pinned
official PhysTwin Warp backend while preserving the finite Bayesian inference
problem:

\[
p(x_{t+1:T}, z, a, \theta \mid x_{0:t}, \ell).
\]

Here `theta` is a Bayesian-PhysTwin spring-parameter particle, `z` is a
Causal4D realized-contact hypothesis, `a` is a candidate future controller
trajectory, and `ell` is language supplied to MolmoMotion.

## Division of labor

| Component | Role |
| --- | --- |
| PhysTwin | dense geometry, spring graph, contact dynamics, and Warp rollout |
| Bayesian-PhysTwin | weighted object/controller spring-scale particles |
| Causal4D | hypotheses, finite joint posterior, prefix-only update, controls |
| MolmoMotion | sparse language-conditioned proposal evidence over rollouts |

MolmoMotion never overwrites the physical state. Its eight predicted material
point trajectories robustly reweight complete PhysTwin rollouts. This keeps a
poor learned forecast from being laundered into a state correction and lets the
physics model fill in dense motion.

## Real contact hypotheses

The official spring graph is reconstructed from each released `final_data.pkl`
and `optimal_params.pkl`. For every hand, the attachment field can remain
nominal or move one object-graph hop in either principal-axis direction. Every
controller spring moves coherently, its rest length is recomputed, and spring
count/order remain compatible with the official checkpoint.

The contact beam also retains controller-spring gain, target delay, slip
attenuation, and control-frame rotation. It is prior-ranked but stratified: the
nominal state and every individual latent channel are retained before joint
combinations fill the remaining slots. Each state is crossed with weighted
Bayesian parameter particles and action proposals.

## Action settings

`known` uses the released future controller trajectory. It is the control where
MolmoMotion should add little once action is observed.

`hidden` discards every future controller value. Its finite action library is
built only from the final observed controller history: damped continuation,
persistence, reversal, and an orthogonal continuation. A leakage test changes
all withheld future controls and verifies byte-identical proposals.

`ambiguous` includes the released trajectory among the history-only proposals.
It tests whether MolmoMotion can identify the correct action family when the
answer is present but not given to the physics model.

## Raw MolmoMotion association

The adapter recovers the release preprocessing's exact correspondence from
processed object tracks to raw per-camera CoTracker queries. It chooses the
camera with the most tracks visible over the three-frame history, then applies
deterministic farthest-point sampling to choose eight spatially distributed
material identities.

Archived CoTracker coordinates use row/column order for the point-cloud lookup;
the adapter converts them to x/y pixels for MolmoMotion. The 3D history is
passed in the PhysTwin world frame together with calibrated camera-to-world.
Both camera-frame and transformed world-frame forecasts are saved. No manual
track identity or nearest-neighbor association is introduced.

## Inference boundary

Before motion is observed, prediction marginalizes action, contact, and
parameter support. MolmoMotion can reweight this prior using language and the
same observed RGB/3D history.

The online setting then uses only the first 10-20% of held-out object motion.
Position and velocity residuals robustly update the same joint support. Metrics
start at the first frame after this prefix. Changing later target frames leaves
the online posterior unchanged.

MolmoMotion evidence uses a heavy-tailed displacement score with a disclosed
scale. It is a product-of-experts factor, not a calibrated independent
likelihood: its RGB history overlaps information already used to initialize the
twin.

## Three-stage execution

MolmoMotion and Warp use separate pinned Python environments, so exchange is
through immutable NPZ artifacts.

### 1. MolmoMotion forecasts

```bash
causal4d-molmo-phystwin-forecast \
  CASE/final_data.pkl RAW_CASE MOLMO_CHECKPOINT molmo.npz \
  --train-end-frame 59 \
  --caption 'instruction=A person lifts the object upward with one hand.' \
  --caption 'shuffled=A person pushes the object sideways.' \
  --caption 'generic=The object moves.'
```

### 2. Physical rollout bank

```bash
causal4d-phystwin-rollout-bank \
  PHYSTWIN_REPO CASE parameter_profile.npz refit_checkpoint.pt hidden.npz \
  --action-setting hidden \
  --parameter-particles 4 \
  --maximum-contact-states 12
```

### 3. Evaluation

```bash
causal4d-evaluate-phystwin-molmo \
  hidden.npz CASE/final_data.pkl molmo.npz result.json
```

The result compares `physics_prior`, `molmo_*`, `online_prefix`, every
`molmo_*_plus_online` composition, and an evaluator-only best-rollout ceiling.
It reports full-object and eight-query-point horizon metrics, action and
attachment marginals, parameter weights, effective support, and direct
MolmoMotion displacement ADE/FDE.

## Claim boundary

The implementation establishes a real integration path. Whether MolmoMotion
helps is an empirical result and must be reported separately for known, hidden,
and ambiguous actions. A gain without the shuffled/generic language controls
does not establish a semantic contribution. A gain only when the true future
controller is in the candidate library does not establish hidden-action
control generation.

## Single-case pilot result

The first locked pilot used `single_lift_sloth` at the released training
endpoint (frame 58), twelve contact states, and the four highest-mass profile
particles. Those particles retain 42.33% of the 9 by 9 posterior mass, so this
is a go/no-go pilot rather than a final uncertainty study. Every reported
rollout uses the official 667-substep Warp configuration.

The scored window starts after a six-frame online prefix:

| Action setting | Physics prior | Molmo ranking | Change | Online | Molmo + online | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| known | 41.598 mm | 41.548 mm | -0.12% | 40.849 mm | 40.800 mm | -0.12% |
| hidden | 47.566 mm | 48.074 mm | +1.07% | 46.462 mm | 46.860 mm | +0.86% |
| ambiguous | 40.906 mm | 41.785 mm | +2.15% | 40.303 mm | 40.906 mm | +1.49% |

The tiny known-action change is not semantic: instruction and generic captions
produce identical forecasts, and the shuffled-caption physical RMSE differs by
less than 0.01 mm. In the ambiguous setting, MolmoMotion lowers posterior mass
on the released true action from 50.0% to 40.9%. After the observation prefix,
it lowers that mass from 54.9% to 46.1%.

The failure is visible directly in the learned forecast. The instruction output
moves the eight points by only 0.81 mm on average at the endpoint, while its
displacement ADE and FDE against the real tracks are 60.13 mm and 110.71 mm.
Instruction and generic outputs are byte-identical; the shuffled output differs
from them by only 0.35 mm coordinate RMS. Across a 3 by 4 sensitivity grid of
heavy-tail scales and positive evidence weights, every hidden-action setting
worsens the physics prior (range +0.02% to +10.49%) and online forecast (range
+0.02% to +11.07%).

This is not a dead checkpoint. Its bundled `egodex_clean_surface` prediction
moves points by 206.21 mm on average at the endpoint. The pilot therefore finds
an out-of-domain persistence collapse on this deformable-object view, not an
integration or model-loading failure.

**Decision:** the real Causal4D/PhysTwin/Bayesian-PhysTwin integration succeeds,
but MolmoMotion does not pass the scale-up gate. Do not run the 19-case cohort
or make a semantic-prior improvement claim from this checkpoint. A subsequent
attempt needs deformable-object adaptation or a proposal model that produces
nontrivial motion on this input before further physical sweeps.
