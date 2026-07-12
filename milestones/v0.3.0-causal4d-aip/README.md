# Causal4D AIP research milestone

`v0.3.0-causal4d-aip` freezes the implementation and evidence boundary for the
abduction-intervention-prediction architecture on 2026-07-12.

The milestone preserves two distinct findings:

- In the registered controlled benchmark, latent intervention inference reduces
  shifted-contact RMSE from 4.132 mm to 0.805 mm, attains 90.8% coverage at the
  nominal 90% level, closes 80.6% of the oracle gap, and passes all 13 gates.
- On the one real `single_lift_sloth` interaction, Causal4D improves future track
  error by 1.36%, while coverage is only 50.6% at the nominal 90% level. The
  oracle audit attributes 88.7% of total track-error diagnostic headroom to
  model discrepancy. This is a diagnosis, not a multi-action real-data claim.

## Authoritative boundary

- Git tag: `v0.3.0-causal4d-aip`
- Large-artifact vault:
  `/mnt/corsair/florianpfaff/research-milestones/v0.3.0-causal4d-aip`
- File inventory and SHA-256 values: `artifact-manifest.json`
- Paths used to capture the inventory: `artifact-paths.json`
- Source revisions and environment summary: `source-revisions.json`
- Registered gates: `results/controlled/success_gates.json`
- Test and reproduction record: `test-summary.json`
- Architecture-change rule: `FREEZE_POLICY.md`

The Git tree contains compact generated JSON, CSV, and typed NPZ artifacts. The
vault additionally contains the 17 GB MolmoMotion checkpoint, exact raw Molmo
input tar, external source bundles, editable source snapshot, and large rollout
banks referenced by `artifact-manifest.json`.

## Reproduce controlled results

Run this from a checkout of the tag on `gpuserver6000`:

```bash
./milestones/v0.3.0-causal4d-aip/reproduce_controlled.sh
```

The command runs both controlled benchmarks. Exact SHA-256 matches are accepted
directly; JSON/CSV values are otherwise compared structurally with a `1e-12`
absolute numeric tolerance to accommodate last-bit floating-point reduction
drift. Strings, keys, row structure, and nonnumeric fields must match exactly.

## Reproduce real chain

Run this from the same tagged checkout:

```bash
./milestones/v0.3.0-causal4d-aip/reproduce_real.sh
```

This single command regenerates the MolmoMotion forecast, four-particle
Bayesian-PhysTwin belief, factual intervention abduction, known and hidden
counterfactual banks, semantic trust decisions, and expanded-bank oracle audit.
It uses GPU 0 for PhysTwin and GPU 1 for MolmoMotion by default.

## Verify archived inputs

```bash
python scripts/release/capture_file_manifest.py verify \
  milestones/v0.3.0-causal4d-aip/artifact-manifest.json --location source
python scripts/release/capture_file_manifest.py verify \
  milestones/v0.3.0-causal4d-aip/artifact-manifest.json --location archive
```

The source check covers all 81 declared files. The archive check covers the 43
files with explicit second copies; the remaining entries are already
vault-resident archives or intentionally retained source datasets and banks.

## Restore external source

The vault contains self-contained Git bundles for MolmoMotion commit
`61f5b21b694ad8f854ec7ecd2400005acc73f685` and official PhysTwin commit
`2b6630528141b9cba5a7677c8b88b2129b4a8390`. The non-versioned editable
`MolmoMotion-Field` package is preserved as an exact source tar.

The checkpoint is `MolmoMotion-4B-H3-F30`; its model SHA-256 is
`506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205`.
The exact combined query/input/output NPZ SHA-256 is
`4eb7b89eafa6721d54214127317c36ccb4adbefbebfb3d3afbb4f535b880a7f8`.
