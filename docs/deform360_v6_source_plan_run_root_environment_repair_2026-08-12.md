# Deform360 v6 source-plan `RUN_ROOT` environment repair

Protected-main dual-runtime run `31566855876` completed all ten registered physical cases. Its bounded artifact contains ten physical prediction manifests and a valid, serializable `source-plan-inputs.json`, but no `source-plan.json` or source-prediction seal. The retained terminal stage is `materialize-source-plan`.

The immutable archived science launcher assigns `RUN_ROOT` as a shell variable. The source-plan handoff then launches an inline Python child through `BPT_PYTHON` and reads `os.environ["RUN_ROOT"]`. Because the shell variable was not exported, the child fails after the valid input artifact has already been published.

The repair is deliberately process-local. The dual-runtime dispatcher derives the already established run root only when all of the following hold:

- the command is a generic stdin Python invocation;
- `RESULTS_ROOT`, `AMENDMENT_ID`, and `BPT_SOURCE_SHA` are present;
- the derived run root is a real directory;
- the exact `source-plan-inputs.json` file already exists and is not a symlink; and
- the caller has not already exported `RUN_ROOT`.

It then exports `RUN_ROOT` only to that child process and writes a bounded activation marker under the compact source-evidence root. Earlier generic probes, all per-case commands, and caller-owned exported values are unchanged. Unsafe run-root, input, output, evidence, or marker paths fail closed.

This repair changes no archived science runner, source-plan content, physical manifest, object, episode, camera, selector, physical model, covariance, endpoint, fallback, loss, gate, threshold, or target-access rule. Development suffixes, confirmation payloads, and fresh-target data remain closed until the complete 100-record source-prediction batch is sealed.
