# Deform360 v6 source-plan run-root export repair

Protected-main workflow run `31566855876` completed all ten frozen physical
source carriers, then retained a technical failure at `materialize-source-plan`.
The compact receipt records a valid `source-plan-inputs.json` and no derived
`source-plan.json`, source prediction seals, suffix access, confirmation access,
or target access.

The archived launcher assigns `RUN_ROOT` as a shell variable. Its following
inline Python extractor reads the same value from `os.environ`. Because the
assignment was not exported, the extractor deterministically failed after the
valid wrapper had been written.

The repair seeds `RUN_ROOT` as an exported empty environment value when entering
the content-addressed launcher chain. Nested shells preserve the export
attribute, and the archived launcher's unchanged assignment supplies the exact
run-root value. The physical implementation, cohort, source-plan payload,
selector, covariance, prediction horizons, fallback, and all information
boundaries are unchanged.

The active launcher was also present in both the legacy and dual-runtime
workflow path filters. To prevent two protected-main empirical executions from
writing the same frozen lineage, the legacy self-hosted evidence job is
statically retired while its hosted contracts remain active. The dual-runtime
workflow is the sole empirical executor for this repair.

This is implementation evidence only. A later protected-main execution must
still produce and seal the registered source panel before any source decision
can be frozen. Confirmation and fresh-target access remain unauthorized.
