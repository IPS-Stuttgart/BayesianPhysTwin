# Tracking Cloth self-collision stable-bank source gate

## Scientific purpose

The preregistered self-collision confirmation originally evaluated an 18-member
contact-aware spring-model bank. Fresh workflow `33588967510` reached the first
rep1 source fit and stopped because one parameter member escaped the simulator's
registered numerical domain. The failure occurred before the source model,
rep2 policy, rep3 predictions, prediction seal, or target score existed.

This separately versioned protocol asks a narrower source-only question:

> Does a sufficiently broad, explicitly recorded stable subset of the
> preregistered physical bank support the same leave-one-material-out source
> qualification?

## Stable-bank rule

The only prunable failures are the two explicit numerical-domain messages
already emitted by the parent simulator:

```text
nonfinite contact rollout
contact rollout escaped the registered domain
```

All parsing, marker geometry, timestamp, dimensional, and other model errors
remain fatal. A source fit is admissible only when:

1. the registered nominal hypothesis `(400, 2, 200)` remains valid;
2. at least 50% of the original 18 hypotheses survive;
3. every rejected parameter and exact reason is retained in `physics_fits.json`;
4. posterior weights are renormalized only over that source-stable subset; and
5. the same sealed subset is used later without target-side pruning.

The wrapper changes no data split, prefix, prediction horizon, query, selector,
source gate, confirmation gate, loss, bootstrap, or fallback. It patches only
the three model interface names used by the existing staged runner:
`PhysicsFit`, `fit_physics`, and `all_predictions`.

## Information boundary

This branch runs only rep1 fitting and rep2 source policy selection. Rep3 future
cloth outcomes are not listed or read by the source evaluator. A positive source
gate cannot itself open rep3; target prediction and scoring require a separate
reviewed authorization. A source failure is retained as a scientific or model-
competence result rather than tuned away.

## Commands

```bash
PYTHONPATH=src:. python -m \
  experiments.tracking_cloth_self_collision_stable_bank_v1.entrypoint \
  --stage source \
  --dataset-root /home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526 \
  --protocol experiments/tracking_cloth_self_collision_stable_bank_v1/protocol.json \
  --output /tmp/tracking-cloth-stable-bank-source
```

## Claim boundary

This is a source-gated numerical-stability and competence experiment on public
real motion-capture trajectories. It does not validate a physical probe, expose
a fresh target result, establish online sensing, infer the support-miss rate used
by the support-robust act--sense theorem, or authorize calibration, deployment,
safety, or state-of-the-art claims.
