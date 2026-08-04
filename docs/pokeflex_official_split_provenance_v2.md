# PokeFlex official-split provenance audit v2

## Finding

The complete 18-object validation split used for PokeFlex's published
`6.498 mm` Kinect result is not materializable from exact public archive names.
Thirteen evaluator IDs have exact public counterparts after projecting legacy
object names into the public namespace. Five do not:

| Internal evaluator ID | Public identity projection | Highest public take |
| --- | --- | ---: |
| `Fjadrar_T8` | `Pillow_T8` | 7 |
| `Cylinder_T7` | `3dPrintedCylinder_T7` | 6 |
| `Heart_T14` | `3dPrintedHeart_T14` | 6 |
| `Sponge_T10` | `Sponge_T10` | 5 |
| `Pizza_T13` | `3dPrintedPizza_T13` | 6 |

The upstream `testing-scripts` branch provides a second, author-controlled
source boundary. Commit `fa484b0fa94f59f51e8c5f2293a6b1bc378b7375` is titled
"Update evaluation and preprocessing scripts to align with open-sourced
dataset." It changes the evaluator from the 18 internal recordings to only
`FoamDice_T3`, while changing preprocessing to the public object names. It does
not publish a mapping for the other 17 internal validation recordings.

This strengthens the earlier feasibility conclusion: lower-numbered public
takes must not be guessed as replacements. The 13-case robust result remains a
retrospective public-subset result, even though its `6.44785 mm` value is
numerically below the published aggregate.

## Executable audit

Run the source-only audit on a complete public archive:

```bash
python scripts/remote/audit_pokeflex_official_split.py \
  --public-root /path/to/pokeflex \
  --output /path/to/pokeflex_official_split_source_audit.json
```

For a filename-only inventory represented by empty `.zip` placeholders, add
`--allow-empty-inventory-files`. The audit reads filenames only. It never reads
target meshes, predictions, or metric outcomes.

The frozen 116-archive inventory audit is stored at
`results/sota/pokeflex_official_split_provenance_v2/public_inventory_audit.json`.
Its canonical audit SHA-256 is
`c62f496ffea3e0dc9551cd7c9eb993ddf50eedea5f407d5a4fb51c7b153095c9`;
the serialized file SHA-256 is
`fbc829724a8b6c53c44d28e6ec61dad3302fa39ed71ac369c5c900037b01b406`.

## Exact data request

A directly comparable 18-object run requires one of:

1. an author-supplied mapping from all 18 legacy evaluator IDs to public archive
   IDs, with SHA-256 for every archive; or
2. the processed validation set used for the paper, with a file manifest and
   SHA-256 digests.

At minimum, the response must resolve `Fjadrar_T8`, `Cylinder_T7`, `Heart_T14`,
`Sponge_T10`, and `Pizza_T13`. A guessed take, outcome-selected replacement, or
unverified alias is not sufficient.

## Next gate

If authoritative data arrive, regenerate this audit. Only
`decision=full_official_split_available` permits a new frozen 18-object
evaluation. Until then, the strongest reproducible evidence is the paired
13-case public-subset result; it must not be described as full-split state of
the art.

## Verification

An isolated Python environment with NumPy 2.2.6 and SciPy 1.15.3 passed all 48
official-split, target-custody, public-subset-result, and action-robust protocol
tests. Ruff passed for every added Python file, `compileall` passed, and
`git diff --check` reported no whitespace errors.

The host Python installation separately has an old SciPy binary built for NumPy
below 1.25 and cannot import `scipy.spatial.cKDTree` with the installed NumPy
2.2.6. That host-only ABI error was reproduced in two scorer tests and is not a
failure of this source audit.
