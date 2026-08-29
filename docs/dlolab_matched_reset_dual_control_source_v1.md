# Matched-reset Native Dual-control Source Protocol

## Purpose

This is a controlled source study of whether a reward-blind physical probe can
improve a subsequent deformable-object decision. It uses the public DLO-Lab
native ROD implementation on CPU. It is not a real-data result, an official
benchmark comparison, independent confirmation, or permission to inspect a
protected target.

The preceding wrapping probe study stopped because equal commanded endpoints
did not produce equal realized tool endpoints after different contact histories.
This study does not modify or retry that probe bank. Instead, every probe and
task action is evaluated from a byte-checked native snapshot branch. Probe
mechanics therefore cannot change the state from which task value is measured.

## Frozen Study

The hidden variable is continuous bending modulus. Nine fixed material particles
form the model bank; 72 continuous, deterministically seeded material values form
the source-test denominator. Initial geometry, velocity, gravity, solver settings,
and all other parameters are unchanged.

Four probe candidates are fixed from source geometry: hold, low-frequency lateral,
high-frequency lateral, and high-frequency vertical excitation. Each begins and
ends at the exact same clamp command. The selected probe maximizes expected
mutual information about the nine material particles under the registered metric
noise model. Probe selection receives no task reward, task future, source-test
material value, or source-test loss.

After the probe is selected, null and selected probe observations are generated
for all 72 source-test worlds with paired nuisance draws. The particle likelihood
marginalizes one shared xyz observation bias. Decisions are then sealed for:

- goal-conditioned best fixed action;
- null-probe Bayesian action;
- predeclared low-frequency fixed-probe Bayesian action;
- active-probe MAP action;
- active-probe posterior-mean action;
- active-probe guarded action with exact best-fixed fallback.

Only after the decision seal may the nine native task-action futures be generated.
The oracle is computed only during scoring. The task is a lateral pulse returning
both clamps to the initial command, with two signed terminal shape goals.

## Gates

The study stops before source-test observations unless the selected probe is
nonnull, improves expected material information by at least 0.10 nat over hold,
and improves material MAP accuracy by at least 0.10. It also stops unless each
goal has at least two material-dependent oracle actions and oracle task value is
at least 5% better than the goal-conditioned best fixed action.

The primary source-value gate requires the active Bayesian arm to improve mean
loss by at least 3% over best fixed, 1% over null Bayes, and 0.5% over the
predeclared fixed active probe, with positive paired bootstrap lower bounds
against all three, at least 12 decisions different from best fixed, no missing
or replaced episodes, and no more than 20% harmful decisions relative to null
Bayes. The information-selected probe must differ from that fixed probe.

Passing remains controlled source evidence only. Failing preserves the exact
best-fixed fallback and terminates this method family without retry or threshold
relaxation. No result automatically changes Bayesian-PhysTwin, DEFORM, Causal4D,
or any public claim.

## Information Boundary

The write-once stage order is:

1. particle probe and task bank;
2. reward-blind probe selection and task-headroom check;
3. source-test probe observations;
4. sealed decisions;
5. source-test task futures;
6. score and terminal decision.

All stages bind the clean Git revision, exact source bytes, upstream revision,
runtime, protocol, output root, array hashes, and parent seals. There is one
attempt, no replacements, no GPU, no new recording, no held-v8, no DLO4/DLO5,
and no official DLO3 evaluation.

Before freezing, one nominal null-probe runtime smoke verified byte-identical
snapshot restoration and ordinary geometry without running a task action,
material comparison, reward, information selector, or source-test world. Its
receipt is source-bound and is implementation evidence only.
