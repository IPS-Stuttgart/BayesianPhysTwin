# Prospective Prob4D-to-BayesianPhysTwin protocol

The scientific question is not whether Prob4D can emit a valid observation
artifact. It is whether a causally valid, calibrated Prob4D feeder improves a
guarded physical-twin belief on held-out interactions. The protocol in
`bayesian_phystwin.prob4d_prospective_protocol` freezes that question before any
target outcome is opened.

The contract is deliberately ordered:

```text
source/calibration freeze
        |
        v
held-out Prob4D provider-competence gate
        | pass
        v
guarded BayesianPhysTwin physical-prediction gate
        | pass
        v
optional downstream Causal4D evaluation
```

A provider failure prevents the physical gate from being opened. Failure of the
physical gate prevents a Causal4D evaluation from being admitted. Infrastructure,
artifact validity, or exact fallback alone cannot promote the claim.

## Required split

The protocol uses complete physical objects or acquisition sessions as
indivisible units. It requires nonempty, mutually disjoint:

- `development` units for implementation and source-only debugging;
- `calibration` units for gauge covariance, point covariance, source reliability,
  tracklet policy, thresholds, and guard calibration; and
- `target` units that are opened only after the freeze is sealed.

Both `unit_id` and `group_id` must be disjoint across stages. The primary
statistical unit is `group_id`; dense pixels, frames, or tracks are not treated as
independent experimental replicates.

## Required method matrix

Every freeze contains at least these five methods:

| Method ID | Role | Gauge treatment |
| --- | --- | --- |
| `physical_baseline` | unchanged physical twin and sole exact fallback | none |
| `simple_visual` | non-Prob4D visual reference | fixed or predeclared |
| `prob4d_fused_gauge_marginalized` | fused Prob4D candidate | gauge marginalized |
| `prob4d_framewise_joint_gauge` | unfused framewise factors | explicit joint nuisance |
| `prob4d_tracklet_joint_gauge` | persistent tracklet factors | explicit joint nuisance |

Sensor-assisted variants may be added, but they must be separate method IDs with
`sensor_assisted=true`. They cannot silently replace the camera-only primary
method. The physical baseline is the only method that may declare
`exact_fallback=true`.

## Frozen software and calibration

The protocol binds exact identities for:

- Prob4D revision and wheel SHA-256;
- BayesianPhysTwin revision and wheel SHA-256;
- MotionCrafter revision and immutable model-set ID;
- stochastic seed policy;
- Python and NumPy versions;
- gauge-covariance calibration artifact;
- point-covariance calibration artifact;
- source-reliability calibration artifact; and
- persistent-tracklet policy.

Changing any bound implementation, seed semantics, covariance meaning, feature
set, grouping rule, or calibration split requires a new freeze.

The freeze also hashes source-side files such as the provider-evaluation
manifest, analysis manifest, and method-freeze record. Artifact roles or stages
containing target outcomes are rejected. Readiness validation rejects symlinks,
path escapes, missing files, and SHA-256 mismatches.

## Separate gates

Each primary Prob4D candidate must have at least one provider criterion and one
physical criterion.

Provider criteria compare the candidate with `simple_visual`. Typical frozen
statistics are paired equal-group differences in:

- metric point or flow error;
- endpoint error, seam error, and drift slope;
- coverage shortfall and covariance width;
- Gaussian NLL or Mahalanobis score; and
- selective risk at a fixed retention rate.

Physical criteria compare the guarded candidate with `physical_baseline`.
Typical frozen statistics are paired equal-group differences in:

- future track error;
- future geometry or Chamfer error;
- harmful accepted-update rate;
- interval coverage and width by horizon; and
- exact-fallback frequency.

Every criterion freezes:

```text
criterion ID
stage: provider or physical
candidate and reference method
metric
statistic: estimate, CI lower bound, or CI upper bound
comparison direction
numeric threshold
```

The contract supports several criteria per candidate. All frozen criteria must
pass. A result may not choose a method, threshold, metric, or reference after
opening target outcomes.

## Freeze and readiness

Prepare an unhashed JSON configuration and run:

```bash
bpt experiment run prob4d-prospective-protocol freeze \
  protocols/prob4d_bpt_v1_configuration.json \
  protocols/prob4d_bpt_v1_frozen.json
```

The command validates the split, methods, software, calibrations, analysis, gate
criteria, and source-side artifacts, then adds a canonical
`protocol_sha256`. Editing any field invalidates the content address.

Before target access, hash-verify the frozen source-side tree:

```bash
bpt experiment run prob4d-prospective-protocol validate \
  protocols/prob4d_bpt_v1_frozen.json \
  --artifact-root /data/prob4d-bpt-v1/source \
  --output /data/prob4d-bpt-v1/readiness.json \
  --require-ready
```

Exit code `3` means the protocol itself is valid but the source-side evidence is
not ready for target opening. A successful readiness report still sets
`causal4d_evaluation_admissible=false`; no target result has passed yet.

## Result and decision

The target result identifies the frozen protocol hash and supplies the exact
criterion statistics. It also declares that target outcomes were opened after
the freeze and that no method selection occurred afterward.

If the provider gate fails:

- physical criterion statistics must remain absent;
- `physical_update` must be `null`;
- the physical gate is marked inadmissible; and
- Causal4D remains inadmissible.

If the provider gate passes, the result must include the physical criteria and
exact-fallback accounting:

```json
{
  "fallback_method_id": "physical_baseline",
  "fallback_exact": true,
  "evaluated_group_count": 12,
  "accepted_update_count": 5,
  "harmful_accepted_update_count": 0
}
```

Adjudicate with:

```bash
bpt experiment run prob4d-prospective-protocol decide \
  protocols/prob4d_bpt_v1_frozen.json \
  results/prob4d_bpt_v1_target.json \
  results/prob4d_bpt_v1_decision.json
```

The decision is content-addressed and reports provider pass/fail, physical
admissibility and pass/fail, feeder support, exact fallback, and whether a
secondary Causal4D evaluation is admissible.

## Claim boundary

A successful provider gate establishes held-out observation competence under the
frozen split and semantics. It does not by itself establish an identifiable or
safe physical update.

A successful physical gate supports Prob4D as a feeder under that exact frozen
protocol. It does not establish general reconstruction state of the art or a
Causal4D intervention benefit.

A negative provider or physical result is complete. Target outcomes must not be
reused to select another feature set, gauge estimator, guard, or method under the
same confirmatory label.
