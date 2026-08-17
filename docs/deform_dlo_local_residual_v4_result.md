# DEFORM DLO Local-Residual v4 Result

## Decision

The frozen post-open DLO1 source gate passed. This authorizes a separately preregistered DLO2 source study. It does not authorize official DEFORM evaluation and is not a prospective or state-of-the-art result.

## Exact Execution

- Source commit: `68351843` (`Add frozen DEFORM local residual source gate`)
- Source archive SHA-256: `dbf6fa43d9bbe7cbaf12068dc681514347ee99ce3144126727ee659d9f9112d5`
- Protocol SHA-256: `64d13d8db565b4ba9c661a27f9e4b34209b8f8c45d941054460bd668d50d32d6`
- Runtime: Python 3.10.12, Torch 2.0.1+cu118, CUDA 11.8
- Durable root: `/home/florianpfaff/source-only/deform-dlo-local-residual-v4/run-68351843`
- Focused exact-runtime tests: 34 passed
- DLO2 read: false
- Official evaluation read: false

The selected arm was ridge `1.0` with correction shrinkage `0.5`.

## Locked Results

| Stage | Baseline L1 | Candidate L1 | Relative change | Wins | Worst ratio |
|---|---:|---:|---:|---:|---:|
| Validation (8 trajectories) | 9.2562 mm | 8.2888 mm | -10.45% | 8/8 | 0.9326 |
| Source test (8 trajectories) | 9.6424 mm | 9.1532 mm | -5.07% | 6/8 | 1.0132 |

The source gate required at least 1% improvement, six wins, worst ratio at most 1.10, and candidate mean at most 10.1 mm. All four conditions passed. Baseline reproduction differed from the archived values by less than the locked `1e-7` m tolerance.

For the source predictions, mean coordinate NEES was `0.391` and empirical coordinate coverage was `97.14%` at nominal `90%`. The posterior is therefore conservative on this opened split; this is a diagnostic, not a fresh calibration claim.

## Interpretation

The result supports the hypothesis behind v4: local baseline state, local geometry, and known clamped action explain transferable residual structure that whole-trajectory action analogs missed. The improvement is about five percentage points over the selected DEFORM baseline on the held source partition and is materially larger than the closed v3 action-analog gain of 0.165%.

Because DLO1 has been repeatedly examined, this result is method-development evidence only. The next admissible action is to freeze the same feature/model family and gates for a genuinely unread DLO2 source split, train the DLO2 physical baseline from scratch, and evaluate the source gate once. Official evaluation remains closed unless that independent source gate passes.

## Artifact Identities

- Preflight: `ebb95a72921ac8dab07a1d9bd09b091e553312a97972b4daeae284d517f89888`
- Validation selection: `6651b569de68ae1512b5561ce2bea1481bc0a2ac2b37bac401f2325762ab7f4f`
- Result: `2e7c10454e8787814ce9f723cf7505757f6360c7729980a53a0cb1c6e15b90a8`
- Source prediction: `91a83374f24839bd83ee9098da292c4947b259a67a16098892b126bf81b91aab`
- Selected model: `e762ffd0fe4e72f731f3e876698a9970deb5bece8289d7290be80ac2c94fb954`
