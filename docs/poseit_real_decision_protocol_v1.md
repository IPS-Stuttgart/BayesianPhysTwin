# PoseIt real-object decision validation v1

## Purpose

This protocol tests a narrow real-data claim: on previously unseen household
objects, a downstream-decision-directed policy can choose logged physical
holding-pose probes more efficiently than a task-agnostic system-identification
policy, while abstaining when no pose is certified stable.

The evaluation is a logged-policy study. It does not claim that the selector was
deployed online or that the released robot setup defines safety outside PoseIt.

## Why PoseIt fits

PoseIt contains 1,840 physical grasp cycles over 26 household objects and 16
holding poses. Each cycle records RGB-D, tactile, wrist force/torque, robot state,
and gripper force before a physical shake tests grasp stability. The paper also
reports that all combinations of grasp point, two gripper forces, and holding
pose were collected. This creates repeated action menus nested inside real
objects rather than treating individual frames as independent samples.

The protocol treats pose 1 as the mandatory reference observation. A probe
reveals only pre-shake sensory data at another logged pose. The shake label is
never revealed during selection and is used only to score the final chosen pose.

Primary sources:

- Repository: <https://github.com/Robo-Touch/PoseIt>
- Paper: <https://arxiv.org/abs/2209.05022>
- Public archive folder: <https://drive.google.com/drive/u/2/folders/1CQiMPBEVvRMrDBSIRVeuwyuUOCOesfMc>

## Frozen design

- Statistical unit: one physical object.
- Nested repeat: one grasp-location by gripper-force family.
- Fit/calibration/source/confirmation counts: 10/5/5/6 objects.
- Split: domain-separated SHA-256 order of archive-derived canonical object
  tokens, assigned before any phase label is decoded.
- Probe budgets: 0, 1, 2, and 3 observations after the mandatory anchor.
- Primary contrast: object-level regret AUC for decision-directed selection
  minus the same AUC for task-agnostic latent-response information gain.
- Utilities: +1 for a stable chosen pose, -1 for an unstable chosen pose, and 0
  for abstention.
- Guard: one shared 80% object-level lower-stability certificate calibrated
  without confirmation objects.
- Confirmation: one attempt, exact paired object-level sign-flip test, no
  replacement or outcome-based method change.

The modest confirmation count makes this a demanding test: a statistical pass
requires a highly consistent object-level benefit. Nested grasps increase
precision within an object but never inflate the primary sample size.

## Current acquisition state

The public repository is locked to revision
`5e290eb024f25b1f4aa602724e6869e512aca434`. The primary data locator is the
official `gelsight.zip` Google Drive file. At protocol freeze, Google Drive
exposed the public locator but refused byte acquisition due to its download
quota. No archive member name, phase label, sensor payload, or shake outcome has
been opened.

The repository contains inconsistent license signals: its license file is CC0
1.0, while its README displays CC BY-SA 4.0 and MIT badges. This does not prevent
an attributed academic analysis, but raw archive bytes must not be redistributed
until the release terms are clarified.

## Gates

Before any scientific execution, the acquisition stage must:

1. acquire the exact public primary archive and publish its byte size and
   SHA-256;
2. inspect archive structure without decoding labels or sensor values;
3. confirm 26 canonical objects and freeze their hash-derived roles;
4. bind exact paths, pre-shake timestamps, resampling, missingness handling, and
   the fixed-dimensional feature map;
5. seal the implementation and pass independent source-only tests.

Only then may fit and calibration outcomes open. Source-test outcomes open once,
after predictions are sealed. Confirmation remains unauthorized until the
registered source gate passes.

## Structure-only custody tool

Once the official archive is available, the first admissible command is:

```bash
PYTHONPATH=src python \
  scripts/science/build_poseit_archive_structure_lock_v1.py \
  --archive /home/fpfaff/source-only/poseit-real-decision-v1/archives/gelsight.zip \
  --protocol protocols/poseit_real_decision_probe_v1.json \
  --expected-protocol-sha256 221803b109a82d3a2d923d5e0c18284b965a8848bcd69e25addd97409d31c5d4 \
  --private-member-manifest /home/fpfaff/source-only/poseit-real-decision-v1/structure/private-member-manifest-v1.json \
  --output /home/fpfaff/source-only/poseit-real-decision-v1/structure/archive-structure-lock-v1.json
```

The tool hashes the complete ZIP and reads its central directory. It rejects
encrypted, duplicate, traversing, absolute, linked, and special members. It does
not call `ZipFile.open`, decompress a member, verify member payload CRCs, decode a
phase label, or read sensor data. Member names stay in the local private manifest;
the compact lock contains only aggregate structure and domain-separated digests.
