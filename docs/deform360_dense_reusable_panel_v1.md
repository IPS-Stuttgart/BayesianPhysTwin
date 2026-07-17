# Dense reusable PhysTwin panel v1

## Scientific target

Deform360 reports ParticleFormer at `0.051 m` symmetric Chamfer distance and
`0.079 m` material-track error in its multi-episode protocol. PhysTwin is not
reported in that table because its original pipeline registers a new twin for
every episode. This protocol tests the missing setting: one source-built,
dense PhysTwin graph per object, reused across unseen actions.

The five-object panel is a prerequisite result, not the full official
benchmark. Beating the published numbers here justifies a full-split run; it
does not by itself establish an unqualified state-of-the-art claim.

## Evidence history

The earlier sparse reusable backend used only 21--32 graph nodes and failed its
six-object source gate. The later `081-stripe-rope` development experiment
showed that the official PhysTwin path can retain 662 surface nodes and improve
three unseen calibration actions over persistence by `22.12%` in track error
and `17.13%` in Chamfer distance. That object has now been examined repeatedly,
so it is excluded from this panel and its target episode 5 remains sealed.

Two follow-up ideas are also frozen as negatives. Gibbs model averaging failed
to beat point selection on the source actions, and normalizing trust by only
the supported controller groups degraded track accuracy. The panel therefore
uses one source-selected tuple and rejects any episode whose commanded contact
groups do not all have graph support.

## Locked cohort

The five targets inherit their metadata-only selection from the original
replication protocol:

| Stratum | Object | Source episodes | Calibration episodes | Target |
| --- | --- | --- | --- | ---: |
| Filament | `002-rope-silk` | 0, 2, 5, 6, 7, 9 | 3, 4, 8 | 1 |
| Sheet | `085-scarf-cloth` | 1, 3, 4, 6, 8, 9 | 0, 5, 7 | 2 |
| Sheet | `083-blanket-cloth` | 1, 2, 4, 5, 8, 9 | 0, 3, 6 | 7 |
| Volumetric | `092-squirrel` | 0, 4, 5, 7, 8, 9 | 2, 3, 6 | 1 |
| Volumetric | `170-spider` | 0, 1, 3, 5, 8, 9 | 2, 4, 7 | 6 |

No target prefix, geometry, tactile stream, or outcome has been read. A
filesystem audit found all 15 target paths absent from the protected
`aligned`, `observations`, and `fits` trees.

## Frozen method

Each object receives one dense canonical graph reconstructed from its source
reference episode. Episode association uses the frozen appearance-first,
multiview feasibility method at frame zero only. It cannot use a post-action
object observation, simulator residual, or future object motion. Admission
requires at least 10 accepted cameras, `0.95` match fraction, and `0.8`
effective reliable match fraction.

All commanded gripper groups must attach within `15 mm`; a missing group rejects
the episode before Warp. The official PhysTwin revision and real-data
configuration are pinned. One tuple is selected jointly over all six source
episodes from a 24-member grid. The deployed prediction uses the source-frozen
trust rule from the development experiment, normalized by commanded group
count. Model averaging and supported-count renormalization are not allowed.

The original fixed interval `[110,191)` is superseded. Across all 30 source
actions it captured only `25.09%` of the best equal-length controller
displacement and `33.65%` of the best path on average. This made persistence
artificially strong and could not support a meaningful dynamics gate.

The first action-aligned rule maximized gripper displacement, but a source-only
smoke test falsified it: `002-rope-silk/0` contained `83.07 mm` mean gripper
displacement but only `0.597 mm` mean object displacement. The selected video
segment was an open-gripper approach. This negative is retained in the
milestone.

Each episode now uses an 81-frame interval selected from the known robot action
and aperture alone. Candidate starts are frame 8 and every sixth frame
thereafter. Per-gripper closure confidence robustly maps the episode's 90th
aperture percentile to zero and 10th percentile to one. The rule chooses the
earliest window maximizing gripper-centre path weighted by the minimum closure
confidence at each step. A static aperture falls back exactly to unweighted
path. This grid matches the independently generated source-QA mask timeline,
so every selected source start has an exact reviewed mask. Object geometry,
tactile values, simulator output, and outcome metrics cannot affect window
selection. Future action and aperture are conditioning inputs, so applying the
same frozen rule to an unseen episode remains causal.

The final five tracking frames are discarded, leaving 76 scored frames. An
action-only audit confirmed valid coverage for all 45 source/calibration
episodes: 22 unimanual and 23 bimanual, with original frame counts from 222 to
474. Calibration dynamics remain sealed until source admission passes.

## Evidence order

1. Select and checksum every source window from the known action, then run dense
   association, contact support, and the official Warp grid on all 30 source
   episodes.
2. Require every object to pass its six-action source gate. Any object failure
   seals all calibration dynamics and all five targets.
3. If source admission passes, evaluate the one frozen per-object predictor on
   all 15 calibration episodes as one conjunctive panel.
4. Calibration must improve execution-balanced track and Chamfer errors by at
   least `5%`, improve late errors by at least `3%`, and jointly win at least 10
   of 15 executions. Every listed gate is conjunctive.
5. Only after a pass, freeze prediction artifacts and open all five targets as
   one panel. Only each target's frame-zero observation and known robot action
   may initialize prediction; post-initial object observations remain sealed
   until predictions are frozen. Partial target opening is prohibited.

The clear-margin target is at most `0.0459 m` Chamfer and `0.0711 m` track error,
10% below ParticleFormer's published multi-episode values. The confirmatory
panel also requires joint wins over persistence in at least four of five target
episodes and no topology stratum with median degradation.

## Predictor strategy

The source candidate is an exact-fallback action-response model. For a target
initial state, matched driven and zero-action Warp rollouts isolate the causal
effect of the known robot trajectory. Source cross-fitting chooses physical
parameters and response trust. If an episode is unsupported or learned trust
is zero, prediction is exactly frame-zero persistence. Dynamic contact
activation and a low-rank source residual may be promoted only after they beat
this fallback out of sample; they are not included merely because they are
physically plausible.

## Claim boundary

This protocol evaluates reusable, source-built PhysTwin transfer. It does not
alter the frozen Causal4D claim, does not use target outcomes for selection, and
does not turn the existing one-rope development result into multi-object
evidence. An unqualified state-of-the-art claim requires a subsequent run over
the complete official Deform360 multi-episode split.

## Automatic-twin source diagnostic

After locking the contact-conditioned windows, a separate source-only
diagnostic tested an automatic frame-zero graph on three episodes. A
distributed contact patch and matched driven-minus-zero Warp response were
combined with an exponential material-graph support prior. Scarf episode 1 and
squirrel episode 0 selected a `0.12 m` support scale and `0.9` response gain on
training frames only. Their untouched tails improved by `23.26%` on the pooled
combined score. Frozen transfer to rope episode 0 improved untouched-tail track
error by `51.50%` and Chamfer distance by `57.49%`.

This diagnostic does not amend the reusable-graph claim above: it reconstructs
an automatic graph from each episode's allowed frame-zero observations. It is
instead a candidate causal prediction path for independent evaluation. The
three examined episodes are excluded from the new 27-episode source gate in
`deform360_graph_action_support_independent_source_v1.json`. No calibration
outcome or target observation was opened.
