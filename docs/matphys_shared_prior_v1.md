# MatPhys Shared-Prior Audit

## Goal

This branch asks whether MatPhys-style shared training can close the remaining
Bayesian-PhysTwin gap without giving a target model access to future visual
observations. The released PhysTwin spring field remains the exact coefficient-zero
fallback.

## Released-checkpoint competence result

The pinned MatPhys checkpoint was evaluated on `single_lift_sloth` with the
released PhysTwin graph and teacher field. Four input controls were run:

| Video / part input | Object prediction | Result |
| --- | ---: | --- |
| causal prefix, one global part | all 110,597 springs at 1,000 | reject |
| all-frame reconstruction control, one global part | identical to causal | reject |
| causal prefix, five DINO graph parts | identical to one part | reject |
| causal prefix, five graph parts with distinct valid material one-hots | identical to one part | reject |

The predicted object log stiffness is effectively constant at
`log(1000) = 6.907755`; its spatial standard deviation is below `1e-7`. The raw
proposal direction has mean `-2.05068`, standard deviation `0.23302`, and
correlation approximately `-1` with the teacher log-stiffness field. It therefore
does not provide a learned spatial proposal: it cancels the teacher variation and
pushes the object field to the decoder's lower bound. Causal and all-frame outputs
are numerically indistinguishable for this case.

The final row is a diagnostic capacity control, not a semantic material
prediction. It assigns the five graph parts maximally distinct valid MatPhys
material codes (`fabric`, `rubber`, `silk`, `denim`, and `fur`). The frozen
decoder still returns exactly `log(1000)` for every object spring. Thus the
remaining failure is not explained by missing per-part VLM material labels on
the public inference interface. Recovering or inventing such labels is not a
credible fast path for this checkpoint.

The exporter now records a target-independent competence gate. A proposal is not
allowed into a Warp family gate when at least 99% of object springs occupy either
decoder bound or when its spatial log-stiffness standard deviation is below
`1e-4`.

Raw summaries are archived under
`results/sota/matphys_shared_prior_checkpoint_audit_v1/`. This is a negative
checkpoint/interface result, not evidence that shared physical learning is
unhelpful.

## Source-supervised contract

The next model uses a different and stronger information structure:

1. Inputs for every interaction are sampled only from an early causal RGB prefix.
2. Complete trajectories may supervise registered source interactions.
3. Registered target videos, outcomes, and metrics are absent from training and
   checkpoint selection.
4. The checkpoint is selected at a fixed terminal epoch.
5. Predictions are bounded residuals around each target's released PhysTwin
   spring field, so zero remains an exact fallback.
6. A disjoint target-prefix interval chooses learned versus teacher before future
   target metrics are opened.

Using complete **source** outcomes is ordinary supervised transfer, not target
leakage. The typed `matphys-source-supervised-meta-audit-v1` artifact binds exact
source frame hashes, full source objective boundaries, source/target membership,
target non-access, proxy bytes, split bytes, and checkpoint bytes.

The registered run uses MatPhys's own 17/5 case split. The five targets have
already been examined, so the run is explicitly exploratory. The original
20-epoch budget was shortened to five epochs by a registered compute amendment
after the first epoch took substantially longer than the smoke estimate. A
positive result would still have required a frozen successor and an independent
PokeFlex or new physical evaluation.

## Stable v2b result

The first complete run exposed two distributed-training failures before target
future evaluation: non-finite Adam state under object-broadcast synchronization
and a second-backward incompatibility with uneven DDP inputs. The stable v2b run
uses one hooked backward pass, tensor-wise optimizer synchronization, gradient
clipping, and transactional finite-state rollback. Its terminal checkpoint is
fully finite:

- checkpoint SHA-256:
  `cb59db8bb8e7344337cbaab71f7f3b33fdb0067dba2ae45964f93dc8d2e84dd8`;
- rank 0: `5182/5182` accepted optimizer steps;
- rank 1: `5766/5766` accepted optimizer steps;
- zero rejected steps and zero non-finite model or optimizer values.

The fixed target-prefix gate selected the learned field on two of five cases,
`double_lift_zebra` and `single_push_sloth`, and selected the exact teacher on
the other three. The learned field mostly reduced to global softening: mean
object-spring ratios ranged from `0.550` to `0.760`, while the largest range
between five graph-part means was only `0.039`.

Within the MatPhys-fork replay, the selected family improved the raw five-case
mean from `76.122/105.055 mm` CD/track to `51.955/68.152 mm`. The Bayesian
overlay reached `49.242/64.752 mm`. These values are not competitive with the
matched Bayesian-PhysTwin future reference of `8.740/17.443 mm`.

## Simulator-parity failure

The target result cannot be interpreted as a clean physical-model comparison.
The coefficient-zero arm preserves the released spring parameters, but it runs
through MatPhys's modified PhysTwin fork rather than the pinned upstream
PhysTwin simulator. Direct trajectory comparison against the archived upstream
zero replay gives:

| Case | Mean node-frame difference | Final-frame mean difference |
| --- | ---: | ---: |
| `single_clift_cloth_1` | 0.771 mm | 4.478 mm |
| `single_clift_cloth_3` | 0.438 mm | 1.007 mm |
| `double_lift_cloth_1` | 3.338 mm | 24.547 mm |
| `double_lift_zebra` | 186.457 mm | 186.852 mm |
| `single_push_sloth` | 196.559 mm | 227.292 mm |

The fork changes the Warp spring/damping implementation and topology runtime.
On the last two cases the mismatch is much larger than the method effect. The
large apparent benefit from softening therefore includes numerical stabilization
of a non-parity replay. It is not evidence that the learned spring field improves
the released PhysTwin baseline.

## Verdict and successor

This branch is frozen as a useful negative result:

1. Source-supervised causal transfer can learn a target-prefix-useful direction.
2. The learned direction is overwhelmingly a global stiffness scale rather than
   the intended semantic part field.
3. The MatPhys-fork rollout is not a valid SOTA evaluator for that direction.
4. The selected stack is far behind the matched Bayesian-PhysTwin future result.

Do not continue the high-dimensional spring predictor on these five targets.
Any physical-prior successor must first apply its spring field inside the pinned
upstream PhysTwin runtime and require a coefficient-zero trajectory-parity gate.

The higher-leverage successor is source-trained graph-spectral discrepancy
dynamics. Prob4D supplies a covariance-aware estimate of the early residual,
while a source-only action-conditioned state-space model predicts how that
residual evolves under known future controls. This directly targets the failed
static-persistence assumption and leaves the exact Bayesian-PhysTwin trajectory
as a zero-correction fallback.

Machine-readable evidence, runtime logs, and the byte-matched implementation
snapshot are archived under
`results/sota/matphys_source_supervised_meta_stable_v2b/`.
