# Causal MatPhys backbone experiments

Run date: 2026-07-18

Status: complete. Both the absolute-stiffness model and the bounded
teacher-residual model are frozen negative results. No independent evaluation
is admitted by the locked gate.

## Question

Can a learned spatial material model close the published PhysTwin accuracy gap
while Bayesian-PhysTwin's causal information boundary and discrepancy layer
remain unchanged?

The public MatPhys recipe cannot be used directly for this question. Its video
loader samples the complete video, and its repository does not publish the
per-case semantic and part artifacts used by the paper model. The wrapper in
`scripts/remote/run_matphys_causal.py` therefore enforces:

- 16 uniformly sampled RGB frames from the released observation prefix only;
- tracking, geometry, checkpoint, and selection objectives ending at the same
  released prefix boundary;
- a hard rejection of `--fit_all_frames`;
- byte-bound source, checkpoint, proxy, frame-index, and objective provenance;
- known future controller targets during simulation, as in the official
  action-conditioned PhysTwin task;
- no future object RGB, point, depth, mask, track, or metric access.

The pinned MatPhys source is commit
`c16b858dfb79bf21024ead24b45a710600de7b4f`. Gaussian rendering is disabled
during fitting. The physical track and geometry losses and official nonlinear
Warp rollout remain active.

## Public-artifact boundary

The public MatPhys repository omits its generated `node_sem.npz` and
`train_ready.pt` files. Both experiments therefore use the deterministic
`global-onehot-single-part-v1` proxy: the released object material label is a
one-hot class, all nodes share one part, and the simplified decoder's unused
semantic tensor is zero. These are causal global-material MatPhys ablations,
not reproductions of the paper's learned part decomposition.

## V1: absolute stiffness is a negative result

V1 lets MatPhys replace the spring field outright. Its fixed terminal epoch was
locked before the remaining 19 cases were run. On all 22 cases, lower is
better:

| Method | CD (mm) | Track error (mm) |
|---|---:|---:|
| Released PhysTwin, raw | 11.579 | 22.019 |
| Absolute MatPhys, raw | 14.725 | 28.953 |
| Absolute MatPhys + Bayesian anchor | **12.599** | **25.024** |
| Absolute MatPhys + last residual | 12.637 | 25.080 |

On the 19 non-development cases the raw model is 32.69% worse in CD and
35.74% worse in track error than released PhysTwin. It wins both metrics on
only 4/19 cases, with maximum regressions of 149.7% CD and 245.2% track error.
The Bayesian correction recovers part of the damage but does not make the
backbone competitive. The absolute one-part parameterization is rejected.

An observed-prefix family gate selects this MatPhys family on 4/22 cases and
improves the released selected stack by 2.54% CD and 2.05% track error. This is
not independent validation: the MatPhys objective used the complete prefix,
including the tail later scored by the family policy. The amendment in
`configs/sota/matphys_causal_exploratory_v1_amendment.json` locked that
interpretation before any future result from the 19 new cases was inspected.

## V2: bounded teacher residual

The concrete repair is to retain the fitted PhysTwin as a strong physical
teacher and let MatPhys predict only a bounded spatial correction:

```text
log(k_ij) = log(k_ij^PhysTwin) + log(2) * tanh(r_ij^MatPhys)
```

This gives every object and controller spring a ratio in `[0.5, 2.0]` relative
to the released fitted spring. Released collision and damping parameters are
frozen. The exact per-edge teacher comes from the released `best_*.pth` file;
the released CMA globals come from `optimal_params.pkl`. Both files are hashed
in the causal audit and revalidated before export.

At residual scale zero the parameterization is the teacher identity arm. Two
independent `single_lift_sloth` exports differ by only 0.00523 mm on average
over node-frames, and its raw future metrics reproduce released PhysTwin within
0.009 mm CD and 0.002 mm track error. The zero-scale code path and spring-count
contract also pass unit and remote CUDA tests.

### Frozen development result

The scale `log(2)` and terminal epoch 20 were locked on the three declared
development interactions. The selected trajectory uses the existing Bayesian
or last-residual overlay only when its prefix score passes the unchanged gate.

| Case | Released selected CD | V2 selected CD | Released selected track | V2 selected track |
|---|---:|---:|---:|---:|
| `single_lift_sloth` | 16.710 | **16.125** | 20.804 | **20.707** |
| `double_lift_sloth` | 15.938 | **12.291** | 22.463 | **20.681** |
| `double_stretch_sloth` | **4.568** | 4.612 | 8.304 | **8.296** |
| Equal-case mean | 12.405 | **11.009** | 17.190 | **16.561** |

All development gates passed. In particular, the catastrophic v1
`double_stretch_sloth` failure became a bounded 0.96% CD regression, below the
predeclared 5% maximum. This supports the teacher-centered model class; it is
not cohort or SOTA evidence.

## Frozen exploratory gate

`configs/sota/matphys_teacher_residual_exploratory_v1.json` was copied to the
compute host before any v1 or v2 future metric on the remaining 19 cases was
opened. It requires all of the following on those 19 cases:

1. improve both equal-case CD and track error over the released selected stack;
2. win or tie both metrics in at least 12/19 cases;
3. have no case regress by more than 10% in either metric;
4. close at least half of the remaining gap from released Bayesian-PhysTwin to
   the published MatPhys point in both metrics.

Passing can justify a fresh preregistered evaluation. It cannot convert these
previously examined 22 cases into confirmatory evidence.

### Frozen outcome

The teacher residual does not transfer from the three sloth development
interactions to the multi-object cohort. Lower is better:

| Cohort and method | CD (mm) | Track error (mm) |
|---|---:|---:|
| All 22, released validation-selected stack | **10.169** | **19.212** |
| All 22, V2 validation-selected stack | 10.675 | 20.632 |
| Confirmation 19, released validation-selected stack | **9.815** | **19.531** |
| Confirmation 19, V2 validation-selected stack | 10.622 | 21.275 |
| All 22, raw released PhysTwin | **11.579** | **22.019** |
| All 22, raw V2 backbone | 12.214 | 23.116 |

Relative to the released selected stack, V2 regresses by 4.98% CD and 7.40%
track error over all 22 cases, and by 8.22% / 8.93% on the 19-case transfer
cohort. It wins or ties only 9/19 CD cases and 6/19 track cases, below the
required 12/19. Maximum regressions are 64.10% CD on
`single_lift_cloth_1` and 60.54% track error on `single_push_sloth`; six CD
cases and seven track cases exceed the locked 10% limit. The all-22 half-gap
thresholds were 9.084 mm CD and 17.106 mm track error, so the frontier gate also
fails by a wide margin.

The bounded teacher removes the catastrophic behavior of absolute V1 on the
development object, but it does not make the unpublished one-part proxy a
transferable material model. The three-case positive result was a
single-object development result, not evidence across cloth, rope, plush, and
package objects.

The predeclared observed-prefix family gate selects V2 on 12/22 cases. Its
adaptive trajectory reaches 9.945 mm CD, a 2.20% improvement over the released
selected stack, but worsens track error to 19.589 mm by 1.97%. Because the
family-validation tail overlaps the prefix used to fit MatPhys, this remains a
secondary exploratory diagnostic and cannot rescue the transfer decision.

## Calibration boundary

Neither MatPhys trajectory export carries a predictive distribution. The
Bayesian anchor supplies a conditional discrepancy covariance, but replacing
the deterministic backbone does not make that covariance calibrated. The
calibration CLI therefore accepts an opt-in external manifest and matching
overlay:

```bash
bpt-audit-phystwin-calibration DATA_ROOT OUTPUT \
  --external-backbone-manifest MANIFEST \
  --external-overlay-dir OVERLAY
```

The command hash-validates the external trajectory and requires the overlay to
reference the same manifest before recomputing strict-split conformal coverage
and manual-track NEES. No calibration statement is allowed until this audit is
run for the frozen trajectory bank.

The audit rejects calibration of V2's operational posterior. On the 19-case
cohort, primary nominal-90% posterior-scaled coverage is 70.53% for CD and
84.92% for track error; late-horizon coverage falls to 64.37% and 73.68%.
Operational pooled 3D NEES is 3460.68 against an expectation of 3, with only
41.44% coverage under the nominal 90% ellipsoid. The strict fixed-process
posterior moves to the opposite failure mode: pooled NEES 0.681 and 99.44%
ellipsoid coverage. Thus neither the raw operational covariance nor the strict
fixed-process covariance supports a calibrated Bayesian claim for this
backbone.

## Current interpretation

The useful idea is not "replace PhysTwin with MatPhys." That failed twice. The
teacher-centered identity constraint remains valuable engineering, but the
one-part model class is rejected:

```text
released fitted PhysTwin spring field
+ bounded learned spatial stiffness residual
+ frozen causal Bayesian discrepancy overlay
= identity-preserving learned physical twin
```

The next admissible model family should combine the same bounded teacher
residual with **causal keyframe-only graph parts** and a truly disjoint family
gate:

1. compute DINO features only from frames before a frozen fit boundary;
2. cluster and graph-regularize node features into temporally stable parts;
3. condition the bounded spring residual on part and material features;
4. fit on the early prefix and choose exact-teacher versus learned-residual on
   a later, nonoverlapping validation prefix;
5. freeze on a multi-object source panel rather than three interactions of one
   object;
6. refit or conformally wrap uncertainty on disjoint calibration cases.

The part hypothesis is the missing component most strongly implicated by
[MatPhys's published ablation](https://arxiv.org/html/2605.19386#S5.T2):
removing part decomposition changes its future result from 8/15 mm CD/track to
10/20 mm.
Those are external paper values, not locally reproduced evidence, but they make
keyframe-only DINO part recovery the most defensible next model family. The
residual bound must not be widened after this result.

## Provenance

- compute root:
  `gpuserver6000:/mnt/corsair/florianpfaff/matphys-causal-sota-v1`
- v1 full manifest SHA-256:
  `c4da7fc524d5f5efca5af9d31b5f444d71bb368c176cbd872ab1d48bfb27624c`
- v1 overlay protocol ID:
  `6dc864acf5fec92e1fb7c279f89b2f348aa68f4956f425d4d87ae78296decc01`
- v1 family-gate protocol ID:
  `6667b3431faaf61a9f48e732c11a316b5efbc735b5b472218c70f2d2ca69f083`
- V2 development manifest SHA-256:
  `9bfe019dcd68d61484cf08a62f7d91ad13362f198cb6065207033c38fc38eb9f`
- V2 development overlay protocol ID:
  `bf810a91dd394d95aaf0e77828cd72879db0568e1cc77fb11aacd9626acee051`
- V2 exploratory protocol SHA-256:
  `7a89cdf78b713592fb63b9ad7ed9ef137dba74e08050f4dc5a4f7a525a788833`
- V2 full manifest SHA-256:
  `fa502660c532d4eb11bd40e535d69df7550656b705346361c02e40205f9e81f2`
- V2 overlay protocol ID:
  `a118e1ae77498db93607092947f480b052116d2e6a1449ed37536d042105f978`
- V2 calibration protocol ID:
  `fd27d6837dfc48daa14c995f7377c6c7bdb8afa3c1e66d5df8ac5a4ade0ca284`
- V2 secondary family-gate protocol ID:
  `f8ba99b2f34ea70177909f9216a3138196a85feb3c9ab20ee0dc5dd99d6d6793`
- frozen gate report: `results/sota/matphys_teacher_residual_exploratory_v1_gate.json`
- frozen gate report SHA-256:
  `e4a136143e53bc77f12fc7ef4f0444e39e50f11ef042908c2938a341780927ae`
- ordered Table-1 manifest SHA-256:
  `41bab11c174cc1551c53a719a8b41a4ec493c2d83c8ef2ef946d73ed01f46219`
