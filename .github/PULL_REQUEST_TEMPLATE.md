## Summary

Describe the change and its root cause or scientific purpose.

## Change classification

Select every applicable category:

- [ ] Stable contract or supported public interface
- [ ] Prospective method for future evidence
- [ ] Diagnostic, ablation, or negative control
- [ ] Frozen reproduction or historical compatibility
- [ ] Infrastructure, CI, packaging, or documentation

## Scientific and information boundary

- Statistical unit and split, when applicable:
- Source/calibration/target access performed by this change:
- Frozen implementation, protocol, lock, or result files touched:
- Permitted claim:
- Explicit non-claims:

- [ ] No target outcomes were used for method, threshold, exclusion, or
      hyperparameter selection.
- [ ] Technical failures and preregistered exclusions remain visible.
- [ ] Rejected updates reproduce the declared exact fallback.
- [ ] A new version or amendment is used instead of rewriting frozen evidence.

## Compatibility and identities

List changed artifact schemas, provider capabilities, CLI routes, package
requirements, content identities, and cross-repository pins. State `none` where
applicable.

- [ ] Existing frozen artifacts retain their original interpretation.
- [ ] Stable changes have wheel and source-distribution coverage.
- [ ] Prob4D and Causal4D integration uses versioned public boundaries.

## Validation

Record exact commands, source revision, runtime, and relevant artifact or workflow
run IDs.

```text
# validation evidence
```

- [ ] Ruff lint passed.
- [ ] Ruff formatting passed.
- [ ] MyPy passed for the changed first-party path.
- [ ] Focused tests passed.
- [ ] Relevant adjacent or full-suite tests passed.
- [ ] Byte compilation and `pip check` passed.
- [ ] The validated checkout was clean and matched the reviewed head.

## Workflow and security review

Complete this section for workflow, download, archive, artifact, credential, or
self-hosted-runner changes.

- [ ] Permissions are least privilege.
- [ ] Actions are pinned by immutable commit SHA.
- [ ] Checkout credentials are not persisted unless a documented publication
      step requires them.
- [ ] Pull-request execution has no private dataset roots, confirmation payloads,
      SSH keys, publishing credentials, or writable Docker socket.
- [ ] Temporary writer or validation-only workflows are excluded from the final
      merge.
