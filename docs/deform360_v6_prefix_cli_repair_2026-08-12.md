# Deform360 v6 prefix CLI compatibility repair

## Retained failure

Protected-main workflow run `31510971371`, attempt `1`, executed exact source
revision `b0f6b46991a20c54260baf58ddf62fbb6dab7813`. Runtime construction passed,
but the first source `stage-prefix` command terminated before prediction because
the frozen archived runner supplied legacy `--repo` and `--role` arguments that
the checksum-bound prefix-stage parser does not accept.

The uploaded receipt is
`ea3856ed0084efd5e13357df877bc1e3bc0a64257c043a35490fda65054660b5`.
It records `source-technical-failure-retained`, zero of ten physical manifests,
zero of one hundred prediction seals, and false values for every source-suffix,
v5-confirmation, v6-target, replacement, and claim-authorization boundary. The
compact artifact ID is `9109136220`, with digest
`sha256:7e4bd7ba33db2985a2b8e768c1a489487d89b86f736276ed1d25d6cf9b3c73a1`.

## Correction

The content-addressed repair is
`protocols/amendments/deform360_official_hub_fresh_object_session_v6_prefix_cli_repair.json`,
repair ID
`88441357317afa7280513e67fe081dc3fafcd463e5cd3a0e2d32520a50db31ae`.

The frozen v5 stage implementation and archived v6 runner remain unchanged.
The outer execution Python shim validates exactly one `stage-prefix` binding,
requires `--repo` to equal the exact execution worktree, requires
`--role calibration`, and removes only those two legacy option-value pairs
before invoking the checksum-bound wrapper. Non-prefix commands retain their
original argument sequence. Missing, duplicate, or changed bindings fail
closed.

## Authorization boundary

After reviewed merge, this repair authorizes one protected-main source
prediction execution under the existing amendment. It does not authorize
development-suffix scoring, fresh-object selection, target payload access,
replacement, confirmation, or any scientific claim. The one-hundred-prediction
barrier remains unchanged.
