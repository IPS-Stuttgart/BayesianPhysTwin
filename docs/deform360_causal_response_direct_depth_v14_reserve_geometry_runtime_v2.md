# V14 reserve geometry runtime v2

The first reserve-geometry operator stopped before prefix materialization
because the unchanged parent builder requested the legacy
`parent_prefix_assets` view while the reserve child lock stores that binding
under `parent_artifacts.prefix_assets`.

This runtime-only child supplies the missing alias. It does not alter any mask,
rank, reconstruction parameter, runtime dependency, admission threshold, or
information boundary. The failed invocation produced neither a geometry
artifact nor a result artifact and remains preserved in the campaign log.
