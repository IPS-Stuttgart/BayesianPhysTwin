# DEFORM DLO2 Validation-Selected Local Residual v6 Result

## Decision

The independent DLO2 source gate passed. The validation-selected shrinkage reduced source mean coordinate L1 by 9.26%, improved all eight source trajectories, and kept the worst candidate-to-baseline ratio at `0.9347`. Its 7.3150 mm source error is below the published DLO2 reference of 9.7 mm.

This authorizes a separately frozen all-training refit and one-shot official evaluation. It is not itself an official benchmark or state-of-the-art result, and no official evaluation file has been read.

## Exact Execution

- Source commit: `9e2bbd95` (`Freeze validation-selected DLO2 source transfer`)
- Source archive SHA-256: `ab68f0d1e21121cfe5c89dc90b7082728f38e3cbb1bcb94a1f0e88ba32a4620c`
- Protocol SHA-256: `82c4d07ce297bf27884e52f092d450ea38c4e62495695e97d69568c39c7ba470`
- Runtime: Python 3.10.12, Torch 2.0.1+cu118, CUDA 11.8
- Durable root: `/home/florianpfaff/source-only/deform-dlo2-local-residual-v6/source-evaluation-9e2bbd95`
- Exact-runtime focused tests: 17 passed
- Preflight reproduced fitted model: `ed4feae941ca9c293860d21f89846761ab2e5b9e6c64d97dc798d6b4db7acf10`
- DLO2 source opened: true, once, after the source-opening seal
- Official evaluation read: false
- Prob4D used: false

## Locked Results

| Stage | Baseline L1 | Candidate L1 | Relative change | Wins | Worst ratio |
|---|---:|---:|---:|---:|---:|
| Validation (8 trajectories) | 7.9120 mm | 7.2403 mm | -8.49% | 7/8 | 1.0480 |
| Source (8 trajectories) | 8.0619 mm | 7.3150 mm | -9.26% | 8/8 | 0.9347 |

The source gate required at least 1% improvement, at least 6/8 wins, worst ratio at most 1.10, and candidate mean strictly below 9.7 mm. All four conditions passed without changing the arm or thresholds after source opening.

Source mean coordinate NEES was `0.604` and empirical coordinate coverage was `95.54%` at nominal `90%`. The uncertainty is conservative on this eight-trajectory source partition; calibration on the official split remains unknown and must be reported without retuning.

## Interpretation

The v5 failure was a correction-strength problem rather than a lack of transferable residual structure. Reducing shrinkage from `0.5` to `0.25` retained most of the aggregate validation benefit, brought the lone validation regression inside the locked cap, and then transferred with eight source wins. The result supports a causal local residual conditioned on observed initial state, known clamped action, and the physical rollout.

The next admissible action is to freeze a deterministic refit on all 56 DLO2 training trajectories and a single official-evaluation operator. The official arm must keep ridge `1.0`, shrinkage `0.25`, the same checkpoint-training recipe, exact clamped nodes, and byte-exact fallback. No additional DLO2 tuning is authorized.

## Artifact Identities

- Development selection: `56365ef30f511e296ffbbb0d22001fd1bc07f655ed87656b60c23c0897a2bef1`
- Authoritative preflight: `6c573fe919633ef5f611a14536fa4e257afd97f37a783cb389c9c229cb1be31b`
- Source-opening seal: `97c29029b17362038881a90a3d2541ae7486653af7d2138c2ebae10c9d915a24`
- Source prediction: `802d2a457c8c5a1d6ab480f141e001573630e8bdb3b14cf86bdae8e760eb2660`
- Result: `e1a4bbf87a658c4e52527b3bacbfbb761e2fc08b723358d7f4f8dfedb49440ef`
