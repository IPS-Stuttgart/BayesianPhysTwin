# Deform360 existing-data transport eligibility audit

## Purpose

This audit determines whether the runner-resident Deform360 fragments can support
an **existing-data, same-object cross-episode transport study**. It is a data
readiness decision, not an estimator result and not physical-transport evidence.

The intended downstream study separates episode-specific initial state from an
object-specific physical coefficient or parameter belief. A source episode may
inform the object-specific belief. A different target episode supplies its own
initial geometry and registered action, while the target future remains hidden
until all forecasts are sealed.

## Information boundary

The audit may inspect:

- directory and file names;
- ordinary file sizes;
- exact data/timestamp stem pairing; and
- JSON files of at most 1 MiB whose basename is explicitly allowlisted in the
  protocol.

It must not:

- decode RGB, depth, tactile, or other media;
- load `.npy`, `.npz`, `.h5`, `.ply`, or other numerical payloads;
- hash large dataset payloads;
- score predictions; or
- use a target outcome.

The result records these boundaries explicitly.

## Eligibility classes

- `processed_transport_ready`: at least two sufficiently multiview processed
  episodes, target and robot carriers, and at least two metadata-derived action
  values.
- `raw_visuotactile_transport_candidate`: at least two raw episodes with the
  frozen camera and tactile floors. Processing and action verification remain
  necessary.
- `raw_visual_transport_candidate`: at least two raw episodes with sufficient
  cameras but not the full tactile floor.
- `processed_rgb_transport_candidate`: at least two processed multiview RGB
  episodes, but the audit cannot establish complete action/target carriers.
- `single_episode_calibration_or_control`: usable for calibration, nuisance
  characterization, wrong-object donors, or provider diagnostics, but not for
  within-object cross-episode transport.
- `incomplete`: no episode meets the minimum structural requirements.

The aggregate decision is:

- `sufficient_for_bounded_existing_data_study` only when at least eight objects
  are immediately processed and action-labelled;
- `conditionally_sufficient_pending_processing_and_action_audit` when at least
  eight multi-episode objects are structurally processable;
- `sufficient_for_pilot_only` when five to seven objects are structurally
  processable; or
- `insufficient_for_cross_episode_transport` otherwise.

Even the strongest decision authorizes only a bounded study over the available
objects and action families. It does not establish representative coverage of
all 198 Deform360 objects, arbitrary-action generalization, unseen-object
transfer, unique causal identification, or state of the art.

## Execution

The permanent `Deform360 metadata-only preflight` workflow runs the audit on
`gpuserver6000` and `gpuserver4090` after the reviewed files are pushed to
`main`. Pull requests execute only the hosted synthetic contract checks. The
self-hosted jobs use the exact merged revision, install no dependencies, write no
dataset files, and upload only compact JSON/text evidence.
