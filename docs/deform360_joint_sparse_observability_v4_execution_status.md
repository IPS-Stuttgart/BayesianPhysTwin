# Deform360 v4 execution status and custody boundary

This file records the operational state of the joint-sparse observability v4
implementation separately from any scientific result.

## Current state

- The v4 contracts, numerical evaluator, portable manifest loader, frozen policy,
  tests, package membership, and guarded workflow are implemented.
- No v4 manifest derived from runner-resident Deform360 factors has been
  evaluated by this implementation.
- No v4 scientific pass or support-negative result is claimed by the code-only
  pull request.

## Required execution input

A protected repository variable named
`DEFORM360_JOINT_SPARSE_V4_MANIFEST` must identify a regular manifest file under

```text
/mnt/lexar4tb/datasets/deform360/results/
```

on the sole `self-hosted` runner. The guarded workflow rejects a manifest under
either raw-data root and requires the exact runner name `workstation2`.

## Closed inputs

The v4 development workflow does not authorize traversal of:

```text
/mnt/lexar4tb/datasets/deform360/data-7fea8e2
/mnt/lexar4tb/datasets/deform360/adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370
```

It also does not authorize prediction point values, residuals, calibration
outcomes, future frames, confirmation payloads, adaptive-confirmation payloads,
target outcomes, replacement factors, replacement cameras, replacement objects,
or human selection.

## Result semantics

Exit code `0` denotes that the frozen development design gate passed. Exit code
`3` denotes a complete support-negative development result. Neither result
opens confirmation or establishes Prob4D calibration, BayesianPhysTwin benefit,
Causal4D benefit, deployment safety, or state of the art.
