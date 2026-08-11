# Deform360 v6 generic-selector byte-identity repair

Date: **2026-08-11**  
Status: **pre-source-prediction technical repair; all suffix, confirmation, and target data remain closed**

## Failure that motivated the repair

Protected-main source run `31458096956` passed the corrected SAM 2.1 Hiera-Small
checkpoint stage and completed the frozen ten-object, 324-stream inventory. It
then stopped at `locate-frozen-generic-selector` because the registered SHA-256
for `src/causal4d_public/deform360_object_sam2.py` did not match any checked-out
file.

The bounded evidence is:

| Item | Value |
| --- | --- |
| Source revision | `67daacdaafe98b63b8aa0357dccdcd11b9a81d51` |
| Workflow run | `31458096956`, attempt `1` |
| Artifact | `deform360-v6-source-prediction-evidence-31458096956-1` |
| Artifact ID | `9088797337` |
| Artifact digest | `sha256:4438365b1664020f5398dfc1b6bdcd749b7499c8c1168ef62ca0fa49cb95d63a` |
| Receipt ID | `cfcfeab74ee9cc88002e398afa2655ccc1a56752787fe6b44a961061fb7cd040` |
| Prepared inventory ID | `b94c1c87d0a8c571d03cc4e222a20c2fa41b12236461c9e4134ba39bf41267bb` |
| Prepared objects / streams | `10 / 324` |
| Physical manifests | `0/10` |
| Source prediction seals | `0/100` |

No physical prediction or source forecast had been produced. The run is
technical readiness evidence, not evidence about model performance.

## Complete-history diagnostic

Workflow run `31458663573` checked the complete Causal4D history for the
registered selector digest.

At the already-pinned Causal4D revision
`50e3682a5dbf976b20cc9115b6e7a975d0144ea5`, the selector file has:

- byte count `17,310`; and
- SHA-256 `c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5`.

The execution amendment instead recorded:

`79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df`.

The registered digest matched **no revision in the complete Causal4D history**.
Consequently, a history-search fallback cannot reproduce the declared bytes.
The stale byte identity must be corrected while the repository revision, file
path, and selector semantics remain fixed.

## Repair

The original content-addressed execution amendment remains immutable and bound
to both failed runs. A separate repair is recorded at:

`protocols/amendments/deform360_official_hub_fresh_object_session_v6_generic_selector_identity_repair.json`

Repair ID:

`d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90`

The original reviewed runner after the SAM 2.1 correction is archived
byte-for-byte at:

`scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh`

Git blob identity:

`42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1`

The active entrypoint verifies the repair and archived blob, replaces exactly
one selector SHA occurrence in a temporary copy, executes the otherwise
unchanged runner, and binds the repair ID plus repository/revision/path/SHA/byte
identity into the resulting execution receipt.

## Frozen scope

Only `runtime_sources.generic_selector_source_sha256` changes. The following do
not change:

- Causal4D repository or revision;
- selector path or semantics;
- SAM 2.1 model, configuration, checkpoint, or revision;
- source objects, camera panel, candidates, losses, gates, or fallback;
- replacement policy;
- the required 100-record source prediction barrier; or
- claim authorization.

The development suffix, v5 confirmation payloads and outcomes, and all v6 target
selection, payloads, and outcomes remain unopened. A later run may be interpreted
scientifically only after it produces all ten physical manifests and seals all
100 source prediction records before any source suffix is accessed.
