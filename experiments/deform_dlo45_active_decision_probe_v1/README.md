# DEFORM active decision-identifying probe-duration pilot

This retrospective development experiment asks whether a physical twin can
acquire **only enough response information to certify a residual-admission
decision**, rather than identifying the complete deformable state.

## Probe semantics

Every existing DEFORM decision already contains a registered 25-frame endpoint
motion. A candidate probe executes and observes the first `p` frames of that
same motion, with `p in {0, 3, 6, 12}`. The duration is selected from source
hypotheses before target internal-node response is read. After the prefix:

1. the observed internal-node response is converted to a residual relative to
   the existing kinematic baseline;
2. a source-local deterministic outcome model assigns one of three outcomes;
3. the exact active decision certificate chooses an outcome-contingent action
   from fallback, half correction, and full correction; and
4. only the internal-node configuration at the original 25-frame terminal
   horizon is scored.

Thus every method predicts the same absolute terminal time. The probe changes
available information, not the endpoint-motion direction or final query time.

## Scientific question

The primary mechanism endpoint is whether the minimum-cost exact selector:

- uses no probe when the passive decision is already certified;
- chooses a short probe when response information closes the regret certificate;
- retains multiple compatible source hypotheses after the selected outcome;
- improves terminal RMSE over exact fallback; and
- falls back when the observed response lies outside source outcome support.

Baselines include fixed probe durations, a maximum-outcome-entropy duration,
the passive no-probe certificate, exact fallback, and a hindsight probe/action
oracle. The entropy and oracle rows are diagnostics and do not provide the
registered all-compatible-beliefs guarantee.

## Evidence boundary

This is retrospective mechanism evidence on the public DLO4/DLO5 official held
trajectories. Probe observations are oracle-quality motion-capture internal
nodes. The experiment does not invent unobserved counterfactual probe outcomes,
but it also does not compare alternative physical probe directions or establish
a learned-perception or real-robot active policy.

The protocol is frozen in `protocol.json`. It uses all source training
trajectories to fit the local support model and does not tune on held outcomes.
The target response is sliced only after probe duration and contingent policies
have been constructed; terminal truth is sliced only after all non-oracle
choices are fixed.

## Run

```bash
python -m experiments.deform_dlo45_active_decision_probe_v1.run \
  --dataset-root /path/to/DEFORM/data_set \
  --output results/development/deform_dlo45_active_decision_probe_v1/result.json
```
