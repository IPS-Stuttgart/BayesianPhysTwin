# Official-Hub Deform360 visuotactile validation v1

## Purpose

This protocol is the next independent-cohort gate for Bayesian-PhysTwin. It tests
whether an uncertainty-bearing visual update should be admitted only when an
independent tactile/proprioceptive contact factor makes the requested physical
query identifiable beyond camera gauge and camera bias.

The locked protocol is
[`protocols/deform360_official_hub_visuotactile_v1.json`](../protocols/deform360_official_hub_visuotactile_v1.json).
Its canonical JSON SHA-256 is:

```text
55534067fb0b3d7965eb66438cbec2ac5b85bcf5378abd1a73785479a5cdbeab
```

The protocol and Stage-0 workflow do not download or open selected camera,
tactile, robot, reconstruction, depth, tracking, point-cloud, or control-point
payloads. They open only the official raw object-directory names and the selected
objects' `metadata.json` files.

## Why this route is distinct from the mounted-cache failure

The target-blind cache preflight in PR #115 completed before numerical payload
access. It found that every locally present candidate-name trajectory path was
inside a prior-generated experiment root. Those paths cannot support a fresh
object/session claim and are excluded here by object ID and by the retained
cache-inventory content addresses.

The official Hugging Face dataset is a different source boundary. Stage 0 lists
the top-level `raw/` object directories from `brownu/deform360`, resolves the
exact dataset revision, and downloads only selected `raw/<object>/metadata.json`
files. Later stages must use the official preprocessing implementation pinned to
`lhy0807/deform360@d8522a4403b766aeb387510c04e89032a56fdf35`.

A resolved Hub revision and exact object/episode selection must be committed and
hash-bound before any selected raw payload is downloaded.

## Completed Stage-0 lock

The publication workflow completed on the hosted contract lane and on
`workstation2`. The committed lock is
[`protocols/locks/deform360_official_hub_visuotactile_v1_selection.json`](../protocols/locks/deform360_official_hub_visuotactile_v1_selection.json).
It binds:

- official dataset revision
  `f804696d7a133908c7497ffdab43819d879b5cbc`;
- selector implementation revision
  `b3d0b0657fc183381a5705a28a26e2ee3a701d5c`;
- content-selection SHA-256
  `f3d3ac25020ec85cad3fadf097259930437baae2b50b4c7f21f61d4823fc649b`;
- canonical selection SHA-256
  `b28daf8477e214cb74a4d250ef5eea8f9f1a014aec10487699ac0ce063961222`;
- complete selection-artifact SHA-256
  `47c577b6d08f8beba187a622b3555631a2a7f3d970cb1d2fde80fb5584173071`;
- five calibration and six confirmation objects in each of the sheet and
  volumetric strata; and
- the exact metadata-file SHA-256 for all 22 selected object/episode pairs.

The official inventory contained 192 raw object directories. Ninety-two objects
were excluded by the frozen prior-protocol and contaminated-cache boundaries.
Stage 0 opened the 22 selected `metadata.json` files only. Its recorded boundary
states that no camera media, tactile arrays, robot arrays, geometry annotations,
or target outcomes were opened. Replacement remains forbidden after payload
access.

## Scientific model

Visual rows retain the existing explicit-gauge model:

```text
r_visual = H_x alpha + H_g delta_g + H_s beta + H_v gamma + epsilon_v
```

The independent contact family is:

```text
r_contact = A_x alpha + A_b eta + epsilon_t
```

`delta_g`, `beta`, and `gamma` are camera-gauge and camera-bias nuisance
variables. `eta` is an optional tactile/proprioceptive sensor-family bias. The
contact factor contains no camera gauge. It can therefore identify a physical
state direction that is otherwise indistinguishable from coherent camera error,
but only when its own bias model leaves that direction observable.

The public
[`Deform360ContactAnchorV1`](../src/bayesian_phystwin/deform360_contact_anchor.py)
contract binds the mapped contact residual, covariance, state Jacobian, source
files, cutoff, sensor names, correlation groups, and optional anchor-bias prior.
It attaches to the already tested `GaugeAwareObservationBatch` anchor fields and
does not introduce a second estimator.

## Tactile semantics

Released synchronized tactile grids are unitless peak-relative responses. They
are not forces and their 16-by-32 taxels are not independent Cartesian
measurements. Before admission, calibration objects must freeze a contact or
gripper-proprioception linearization that maps reduced sensor features to:

- displacement-equivalent residuals in metres;
- physical-state Jacobians;
- positive-definite row covariance;
- sensor/contact correlation groups;
- source-only reliability and composite likelihood weights; and
- an optional sensor-family baseline/drift nuisance.

Neighboring taxels and repeated samples must share declared groups. The solver's
effective-sample cap then prevents dense tactile grids from manufacturing
confidence through duplication.

## Fresh cohort selection

The registered strata are `sheet` and `volumetric`. No filament claim is made:
the frozen historical filament vocabulary has no untouched object after prior
cohorts and the cache-preflight exclusions are applied.

For each registered stratum, Stage 0 selects by SHA-256 rank:

- five calibration objects;
- six separate confirmation objects; and
- one episode per object from the official metadata sequence IDs.

The object vocabulary comes from the already frozen v1 candidate pools. Every
historical open/reserved, calibration, target, and cache-touched object is
excluded. Listing order cannot affect selection. There is no replacement after
metadata selection or after payload access; technical failures remain in the
accounting.

## Compared methods

1. unchanged physical baseline;
2. last causal readout residual;
3. visual-only persistent factors with the complete explicit joint gauge;
4. contact-anchor-only physical update;
5. guarded explicit-gauge visuotactile update, the primary method;
6. the same visuotactile update without the regret guard; and
7. the same update with an explicit shared anchor-bias nuisance.

Every rejected method returns the physical baseline byte-for-byte. Direct
residual-as-position/velocity injection is not a candidate because the frozen
state-injection control was harmful.

## Endpoints and annotation independence

The statistical unit is the physical object. The first co-primary loss must use
future held-out-view depth or geometry whose defining target does not reuse a
tactile-refinement factor. Released particle or control-point trajectories may be
reported as a secondary endpoint, but cannot be the sole support for a tactile
benefit claim.

The complete report includes object-balanced paired differences, harmful
accepted objects, exact fallback, coverage and full interval width, tail
regression, nonlinear closure, and sheet/volumetric strata. Confirmation is
opened once after all contact mappings, priors, guards, and conformal choices are
serialized from calibration objects.

## Decision rule

A positive result requires all registered conditions, including:

- at least 10% improvement over the physical baseline;
- at least 5% improvement over last residual;
- paired object-bootstrap 95% upper bounds below zero against both;
- at most 5% harmful accepted objects;
- no stratum regression above 2%;
- byte-exact fallback for every rejection; and
- an improvement from the contact anchor over the otherwise identical visual-only
  arm in point loss or harmful-update control.

A completed negative result is admissible and should localize failure to visual
competence, contact-anchor informativeness, object transfer, physical-model
mismatch, or interval calibration. Confirmation-side rescue tuning is forbidden.

## Stage order

```text
merge protocol, contract, tests, and Stage-0 workflow
-> resolve exact official dataset revision
-> select objects from names and episodes from metadata.json
-> commit exact selection and content SHA-256
-> process calibration objects only
-> serialize contact mapping, guard, and intervals
-> open confirmation payloads exactly once
-> publish the object-level positive or negative result
```

The Stage-0 workflow is
`.github/workflows/deform360-official-hub-visuotactile.yml`. It checks out and
records the exact pull-request head rather than GitHub's synthetic merge commit.
The self-hosted metadata job is disabled for fork pull requests. A manual
`workflow_dispatch` run produces the metadata-only candidate lock and evidence
artifact. A same-repository pull-request run independently regenerates the
selection, requires the committed lock to match after removing only the
implementation-bound artifact fields, and verifies that the lock's exact
implementation revision is an ancestor of the current head. The workflow has
read-only contents permission, does not persist credentials, and never commits or
pushes to the pull-request branch. The uploaded Stage-0 artifact remains selection
and provenance evidence, not an empirical model result.
