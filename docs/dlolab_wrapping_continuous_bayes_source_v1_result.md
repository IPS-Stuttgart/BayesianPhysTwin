# DLO-Lab wrapping continuous-material Bayes source result v1

## Status

**Terminal technical failure; no scientific task-value result.**

The single registered attempt ran under frozen revision
`7ddd623b89867348bb4b8635bea03bc6e32f8421`. Its CPU/software-rendering
preflight passed, all four prefix-only batches sealed, and the pre-future gate
passed before any task future was generated.

The sealed decision barrier recorded:

- 253,848 posterior-Bayes decisions different from the source-best fixed action;
- 10,324 posterior-Bayes decisions different from plug-in MAP;
- three distinct posterior-Bayes actions.

All 32 registered continuous-material futures then completed with ordinary native
QA and zero per-world technical failures. While assembling the write-once generation
record, JSON canonicalization rejected one NumPy boolean in the QA metadata. The
runner wrote terminal failure artifact
`32f1da52f18bcddc1697931b139b1222692f8eb7b9839b2997b60b9328837692`
and did not write a generation seal or result artifact.

## Interpretation boundary

No method score, reward comparison, confidence interval, source-gate decision, or
advancement claim is reported. The already-generated futures are not resumed or
scored around the failure. Retry and replacement remain unauthorized.

This failure says nothing about whether posterior expected utility transfers to
continuous materials. It only identifies a compact-artifact serialization defect
after simulation and before scoring.

The implementation was subsequently hardened to cast every QA predicate to a
built-in JSON boolean and to label this point as the generation stage. That change
is a prospective software fix for distinct future protocols, not authorization to
continue this attempt.

## Evidence identities

| Artifact | Identity |
| --- | --- |
| Frozen source commit | `7ddd623b89867348bb4b8635bea03bc6e32f8421` |
| Runtime preflight result | `019c36ae8d28b0814ecdc3439a113f4780c25e03800772988fe79999c113818f` |
| Study lock | `3afda0772f04f3ef7850b1ada4304ade752f462e4b08ce1100fef6ed50768534` |
| Decision barrier | `00c7002573419dd987aeb04d3f158eded35468676cb15e957c85eded522c8214` |
| Terminal failure | `32f1da52f18bcddc1697931b139b1222692f8eb7b9839b2997b60b9328837692` |

The complete frozen simulator tree remains under
`/home/fpfaff/source-only/dlolab-wrapping-continuous-bayes-source-v1`.
