# PhysTwin full-22 evidence report

Run date: 2026-07-18

Status: complete for 3D trajectory metrics, released-render reproduction, and
corrected-render scoring.

## Verdict

The new methods improve on released PhysTwin under its official 22-case 3D
metric contract, but they are not state of the art against all later published
methods.

- Bayesian anchoring improves equal-case CD by 12.09% and track error by
  12.78% relative to released PhysTwin.
- The action-conditioned method improves CD by 9.53% and track error by 7.61%.
- Bayesian anchoring has the strongest equal-case local CD. The much simpler
  last-residual comparator has the strongest equal-case local track error.
- MatPhys and NeuSpring report lower CD and track error than every local method.
  Those external values have not been reproduced here.

The defensible claim is therefore: better than original PhysTwin, not overall
SOTA.

## Frozen evaluation contract

The main table uses the exact ordered 22-case PhysTwin Table-1 cohort and all
735 official held-out frames. Each case receives equal weight, matching the
paper-facing aggregation that reproduces the published PhysTwin values. CD is
the released one-way L1 nearest-surface distance. Track error uses the released
initial-nearest-node correspondence and Euclidean 3D error.

Three cases were used for development and 19 remained the locked confirmation
cohort. Confirmation-only and frame-weighted readouts are retained in the
machine-readable result.

## Reproduced 3D results

Lower is better. Changes are relative to the re-evaluated released PhysTwin
trajectories.

| Ownership | Method | CD (m) | Change | Track error (m) | Change |
|---|---|---:|---:|---:|---:|
| Original | Released PhysTwin | 0.011579 | - | 0.022019 | - |
| Ours | Action-conditioned residual | 0.010476 | -9.53% | 0.020344 | -7.61% |
| Ours | Bayesian anchor | **0.010180** | **-12.09%** | 0.019205 | -12.78% |
| Ours, comparator | Last residual | 0.010185 | -12.04% | **0.019156** | **-13.00%** |
| Ours, comparator | DMDc | 0.010431 | -9.92% | 0.020155 | -8.47% |
| Ours, comparator | Autonomous latent residual | 0.010586 | -8.58% | 0.020342 | -7.62% |

Bayesian anchoring does not dominate the matched simple baselines. It wins CD
over last residual by about 0.005 mm, while last residual wins track error by
about 0.048 mm. DMDc also outperforms the action-conditioned method on both
point metrics. A contribution claim should therefore emphasize belief
estimation, uncertainty, or recursive updating rather than merely the lowest
deterministic error.

The secondary frame-weighted diagnostic is:

| Method | CD (m) | Track error (m) |
|---|---:|---:|
| Released PhysTwin | 0.010784 | 0.022241 |
| Action-conditioned residual | 0.009688 | 0.020554 |
| Bayesian anchor | **0.009477** | **0.019763** |
| Last residual | 0.009513 | 0.019785 |
| DMDc | 0.009664 | 0.020386 |
| Autonomous latent residual | 0.009815 | 0.020823 |

The strict SciPy KD-tree and NumPy fallback evaluators produced identical
absolute values to machine precision.

## Confirmation uncertainty

On the 19 untouched cases, the paired case-and-frame bootstrap gives:

| Method | Metric | Observed macro change | 95% interval |
|---|---|---:|---:|
| Action-conditioned | CD | -7.21% | [-12.11%, -1.68%] |
| Action-conditioned | Track | -4.18% | [-9.99%, +2.96%] |
| Bayesian anchor | CD | -9.39% | [-13.15%, -5.08%] |
| Bayesian anchor | Track | -8.51% | [-13.32%, -3.66%] |

Thus the action tracking gain is unresolved on the confirmation cohort, while
both Bayesian point-metric intervals exclude zero. The reported Bayesian
point estimate does not make its raw covariance calibrated; the separate
calibration audit documents that limitation.

## Released-render reproduction

Released trajectories were rendered with the official Gaussian assets and
scored with the official frame-weighted evaluator. The reproduction matches
[PhysTwin Table 1](https://arxiv.org/html/2503.17973v1#S5.T1) after rounding.

| Split | PSNR | SSIM | LPIPS | IoU |
|---|---:|---:|---:|---:|
| Reconstruction/train, reproduced | 28.213905 | 0.944931 | 0.033607 | 0.843767 |
| Future/test, reproduced | 25.617270 | 0.940507 | 0.055103 | 0.725168 |
| Reconstruction/train, paper | 28.214 | 0.945 | 0.034 | 0.844 |
| Future/test, paper | 25.617 | 0.941 | 0.055 | 0.725 |

This validates the rendering and scoring path. It does not reproduce the
original inverse-physics optimization, because released `inference.pkl`
trajectories are the starting point.

### Corrected rendering

| Method | Test PSNR | Test SSIM | Test LPIPS | Test IoU |
|---|---:|---:|---:|---:|
| Action-conditioned residual | **26.046118** | **0.942526** | 0.054224 | 0.734781 |
| Bayesian anchor | 25.986869 | 0.942461 | **0.052503** | **0.735717** |

Relative to the reproduced released future render, the action correction raises
PSNR by 0.428848 dB, SSIM by 0.002019, and IoU by 0.009613, while lowering
LPIPS by 0.000879. Reconstruction/train values are unchanged because the
correction is applied only to the official future interval.

The Bayesian correction raises PSNR by 0.369599 dB, SSIM by 0.001954, and IoU
by 0.010549, while lowering LPIPS by 0.002600. Action is slightly stronger on
PSNR and SSIM; Bayesian is stronger on LPIPS and IoU. Both improve all four
future-render metrics over released PhysTwin. These image-space results do not
change the cross-paper 3D SOTA verdict below.

## Causal learned-backbone track

The fixed Bayesian corrections can now be applied to a hash-validated external
trajectory with the same PhysTwin vertex identities. This path was used to
test a future-blind public MatPhys ablation without changing the official
future evaluator.

An absolute one-part MatPhys spring field is a clear negative result: its raw
22-case future CD and track error are 0.014725 m and 0.028953 m. The Bayesian
overlay improves them to 0.012599 m and 0.025024 m, but both remain worse than
released PhysTwin. This rejects wholesale replacement of the fitted spring
field under the available public-artifact proxy.

The successor instead predicts a bounded log-stiffness residual around every
released per-edge PhysTwin value while freezing released contact and damping
parameters. Its zero-scale arm reproduces the released simulator within the
measured Warp replay floor, and its fixed `log(2)` scale passed the three-case,
single-object development gate. It then failed the frozen 19-case transfer:
the selected all-22 result is 0.010675 m CD and 0.020632 m track error, 4.98%
and 7.40% worse than the released selected stack. It wins only 9/19 and 6/19
transfer cases, violates the maximum-regression gate, and does not approach the
published frontier. No independent run is admitted. See
`docs/matphys_causal_backbone_v1.md`.

The strict calibration audit also accepts the external manifest and matching
overlay. It recomputes conformal coverage and manual-track NEES rather than
inheriting the released-backbone covariance claim. For this rejected backbone,
nominal-90% posterior-scaled coverage is 70.53% CD and 84.92% track, while the
operational pooled NEES is 3460.68 rather than 3. Calibration therefore fails
alongside point-estimate transfer.

## Published comparison boundary

These are external paper values rechecked on 2026-07-27, not locally reproduced
measurements.

| Method | Status | CD (m) | Track error (m) |
|---|---|---:|---:|
| [MatPhys](https://arxiv.org/html/2605.19386) | Published only | **0.0080** | **0.0150** |
| [NeuSpring](https://arxiv.org/html/2511.08310) | Published only | 0.0087 | 0.0175 |
| [PhySPRING L0](https://arxiv.org/html/2605.07687) | Published only | 0.0096 | 0.0197 |
| [PhysWorld](https://arxiv.org/html/2510.21447) | Published only | 0.0100 | 0.0210 |
| Ours, Bayesian anchor | Reproduced full 22 | 0.010180 | 0.019205 |
| Ours, last residual | Reproduced full 22 | 0.010185 | 0.019156 |
| [PhysTwin](https://arxiv.org/html/2503.17973) | Reproduced full 22 | 0.011579 | 0.022019 |

Against MatPhys's published values, the best local CD is 27.25% higher and the
best local track error is 27.71% higher; lower is better. Against NeuSpring,
the corresponding gaps are 17.01% and 9.47%. These are paper-value gaps, not
paired local comparisons.

MatPhys's current public
[training script](https://github.com/Yrainy0615/MatPhys/blob/main/scripts/ours/train_all.sh)
passes `--fit_all_frames`, while its released
[evaluator](https://github.com/Yrainy0615/MatPhys/blob/main/semantic/eval_simple_video.py)
scores the configured fitted range. Its paper value remains the best published
claim, but the public path does not provide a clean local reproduction of the
stated held-out result. This caveat does not make the present method SOTA:
NeuSpring also reports lower values on both metrics.

The dated [frontier audit](phystwin_sota_frontier_20260726.md) also checks
PGRD, DeformMaster, EgoPhys, BoxTwin, and the current NeuSpring repository
against the local source-gated experiments. DeformMaster reports `0.0114` m CD
and `0.0240` m track error on a 20-case subset, but its released checkpoint
bundle does not encode a reproducible observation/future split and its full
training code is not public. BoxTwin targets articulated elastoplastic objects
and has no released matched full-22 artifact. PGRD uses a different benchmark
and its current public commit is exactly the one already tested locally. None
of these results is inserted into the matched full-22 table above.

## Original versus ours

| Component | Origin |
|---|---|
| PhysTwin simulator, released `inference.pkl` trajectories, Gaussian assets, annotations, splits, and metric definitions | Original PhysTwin |
| Released `inference.pkl` row | Original output, re-evaluated here |
| Action-conditioned residual correction | Ours |
| Robust Bayesian random-walk endpoint anchoring | Ours |
| Last-residual, autonomous, and DMDc matched comparators | Ours |
| Frozen aggregator, locks, parallel execution, hashes, and render orchestration | Ours |
| Renderer trajectory override and evaluator path overrides | Ours, orchestration-only patches |
| Gaussian rendering and metric calculations | Original PhysTwin code |

We did not rerun or modify PhysTwin's inverse-physics optimization. The new
methods are causal correction layers over released outputs: each per-case
correction consumes only its training interval and is validation-gated before
the official future interval. Method choices were frozen on three development
cases before the 19-case confirmation cohort was evaluated.

These are Bayesian-PhysTwin results. No Molmo model or Molmo trajectory enters
this evaluation.

## Deform360 boundary

Deform360 is a separate external evaluation track. Its partially staged server
data and live worktree were not modified or used for this result. An earlier
single-rope pilot remains separate. The preregistered source-backend competence
decision admitted 0/6 objects to the six-object target phase, so no Deform360
target result is admissible under that protocol.

## Provenance

- 3D fitting: `gpuserver4090:/home/florianpfaff/phystwin-sota-eval-20260715`
- rendering: `gpuserver6000:/mnt/corsair/florianpfaff/phystwin-sota-render-20260715`
- schema-2 full comparison: `results/sota_comparison_schema2.json`
- schema-2 comparison SHA-256: `108f94bd6cc22640e614ca3ee1463e93f54d7fd4f573c403af5225e9fa87d413`
- schema-2 comparison log SHA-256: `7f5fadc28a0683c6f679de09f282a5e2ddbdf8c4709d5a329c36ad8089b8173e`
- original schema-1 comparison SHA-256: `7fc8d1ab33a4e0a5fc7e070d0b8d4f79013f42769ecfbaa031f17fee0fbff87a`
- schema-2 aggregator source SHA-256: `4a064f9f3886a1a3b1b4bc96bbe8182de6f5dda94b1a0ae3eacf9f8267564afe`
- independent NumPy comparison without action SHA-256: `a16f3d13a9d812e9dc6922e0be044bfd0c0caddd02aa06e6a11abe6e94a88af9`
- independent NumPy action comparison SHA-256: `98bad8ef63bbe87b414f4384af7e87328b832d9c6c27e671d233acc7f61689e1`
- data-manifest SHA-256: `c986f9fffe99e63f842bb48eb1d394a6b87663f5c4a4fb99f2a58855875fb125`
- official PhysTwin commit: `2b6630528141b9cba5a7677c8b88b2129b4a8390`
- official `data.zip`: `586a237a0f870b2a0a144ee69c3f04b636cac4fe251df969c1fd1feaf67eeae2`

The schema-2 comparison re-evaluated the same frozen inputs and reproduced all
schema-1 metric values exactly. It uses immutable pre-load byte snapshots,
requires every input hash and the evaluator implementation to remain unchanged
through scoring, and records Python 3.12.3, NumPy 2.0.2, SciPy 1.17.1, the
SciPy KD-tree backend, and its native-extension hash.

Locked protocol IDs:

- action-conditioned: `2a0507950ed40802756ad17e96de307a3640591b64ad585f6f5d9d235d84237d`
- Bayesian anchor: `ee11310a84b92ff2158018a13ef09989e641e7c0ea84733fe8a6abf267093c65`
- matched baselines: `508017b7bdaf67a24ca81639ba886147d218297adc0efed49b61ad1ab802da5a`
- repaired action confirmation summary SHA-256: `f3e8662a6dee69f34967fefcc5fab6fd6da113aa8b29ed8c2dedc42d0665fb83`
- action confirmation comparison SHA-256: `5619a9a7a811c5e2faedfa35c986ed8d76ff4b6021c2437331eb5013dce11029`
- action all-22 comparison SHA-256: `61c17603704208ed1270b386047de7beb653e1bf5ce348ce06e1e85367de5974`
- Bayesian confirmation comparison SHA-256: `d4f6c25b3520dd618d38485b2a62bccc33b3006bc118b78d2193eb659858d7b8`
- last-residual confirmation comparison SHA-256: `3dca633404d4e995941e26f4e63da48cdfcff89512fbdc55cf86ddc573729535`

The original action run loaded a transient parallel wrapper with SHA-256
`96df896bd90bf20740b1679986dd994e743511d46e6689efb02d0b67fe3a554e`.
Its fitting path was correct, but a misplaced insertion made the cohort readout
return null. The unchanged fitting core has SHA-256
`990308228405e5c79372e4bf3d33059992f9996fb0e54147c339d144598442c8`,
and the ordered 22-trajectory identity is
`c4a9b96d07a0dfc0e9277c3f15d92928c8f31145d49a7b774aa111d5814b0e95`.
The null summary was preserved with SHA-256
`12e46b48a6c99c378f0ec882f01bc13e4f0bd2f81ccd5ec966bf60d1a4707937`.
A cache-only regeneration from read-only source tree
`e11a0d03b5c8cb40efd2cb44a91806e57396bf3862f5455fc7a35a6ca5831917`
reproduced all three comparison files byte-for-byte and repaired only the
top-level readout; no trajectory was refit or modified.

Render evidence:

- patched renderer: `40d7a7d7c2713edeb3f1a4283c20405fe214ebdc7b7872067635f93973845129`
- patched evaluator: `69aa0b84dc48e285b6b9cd3465064258d01df0707d193a069cfbcb6ae9db12a2`
- evaluated schema-1 runner: `2b3cbb9120dcc6216fbc9e322e53a164ccb9d9f483d2e71c33dc7af88e4524fd`
- current schema-2 runner: `3e5cb434ea2a13127b63e0e3b1d190bd245e494da4ef0ef26ed58a2584be4b73`
- compiled `gsplat_cuda.so`: `6fd7d5f7c62aff18654ed154fae253f4ccb6f8f285055bbbbafa56c8e3e49b2c`
- released render manifest: `6de8a122bf371df2a19110219b9e2f32b9ee90bee13b0ce1c7d055a7665be926`
- released evaluator output: `3a5da38f1e62af085f9afef9402b1663924812ccc8c4df0668928002fb5ad5f1`
- released PNG-tree SHA-256: `ef75a3c19a9013dc472cc328fb3b84c6a17031e121f2fb74663cc558e61dc07c`
- action render manifest: `86a30437d3745d9ca006a0f420c49433e681b01dae2e20a9ce129ddd19d01a4b`
- action evaluator output: `a6f394049aec4378f8f81a7220409cd6c431634f478010975d462ff86269a842`
- action evaluator log: `ceda2c7b56f286b95b38b0285f1c99690763a70dfd43578a4459a8c80e7c74d9`
- action PNG-tree SHA-256: `6cbb8f0e779d47c8e33cab8337e6b4b44368a77c7b9ebe6531bb92ad3ebfdd91`
- Bayesian render manifest: `22836c3c2fbba39b46a4d439644d55a4dc14063f72629c36754ca1d9fed7aebd`
- Bayesian evaluator output: `abb21953050ed3acd1dd442dc34aab3c18507ea21f2d3441dd26eb11a2c897a2`
- Bayesian evaluator log: `51b9a7071d665dc673a396e2bc2861ec2e9f23cbc88a08027dfd73832dd66e66`
- Bayesian PNG-tree SHA-256: `7dc556551dcd29b34570a34c3f74a1df251d480ede41e3739073a620ebfd21b4`

The evaluated schema-1 sweep was fresh and separately preflighted for exact
frame sets, PNG integrity, dimensions, trajectory hashes, and source/model
asset hashes. The repository now contains a stricter schema-2 runner with an
exclusive output lock, staged per-case replacement, output-tree hashes, and
final code/runtime/input/output revalidation. Those later safeguards are not
retroactively claimed for this run.

The final repository tree also gives all three confirmation runners exclusive
output ownership and seals source, split, input, summary, and output identities.
On `gpuserver4090`, 514 tests passed and 5 were skipped; Ruff passed on every
changed Python file.
