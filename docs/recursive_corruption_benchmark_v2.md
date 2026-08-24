# Recursive-corruption benchmark v2: matched guards and fresh domains

## Purpose

The original controlled benchmark showed that a guarded Gaussian discrepancy
belief can outperform unguarded residual persistence when provider reliability
and lineage cues identify corrupted observations. That comparison did not isolate
whether the gain came from Bayesian recursion or from giving only one method the
guard.

Version 2 resolves that confound. `guarded_last_residual` and
`guarded_recursive` receive the same:

- observation-availability flag;
- provider reliability;
- reported source timestamp;
- normalized-innovation gate;
- residual and update trust region; and
- byte-level complete scalar fallback identity check.

Their point-update rules differ: `guarded_last_residual` accepts the measured
residual directly, while `guarded_recursive` applies the Gaussian posterior
update. The matched comparison therefore tests recursive belief propagation
rather than unequal access to corruption metadata.

This remains controlled synthetic mechanism evidence. It is not a real-provider,
physical-object, Causal4D, calibration, deployment-safety, or state-of-the-art
experiment.

## Fresh evidence roster

The registered evidence run uses seeds `100000:100200`. Development tests use
small seeds below 1000 and must not use that evidence roster.

Each seed is an independent **seed-domain**. Before any condition is applied, it
draws and freezes:

- time step, stiffness, damping, and cubic stiffness;
- three action amplitudes, frequencies, and phases;
- discrepancy persistence, process noise, initial scale, and action coupling;
- observation noise;
- corruption start, duration, and recovery window;
- outlier, drift, identity, delay, and density-loss severity.

All registered conditions share the same underlying physical trajectory and truth
within a seed. Methods and conditions are paired within seed; the 200 seed-domains
are the independent statistical units.

## Methods

1. `physical_baseline`
2. `last_residual`
3. `guarded_last_residual`
4. `recursive_gaussian`
5. `guarded_recursive`

The materially harmful accepted-update endpoint is defined for every updating
method by comparing the accepted-update forecast with that method's own pre-update
forecast. It is undefined, and serialized as `null`, for `physical_baseline`.
Gaussian NLL, nominal-90% coverage, and interval width are defined only for the
two Gaussian methods; unsupported cells are `null`, never synthetic zeros.

## Conditions

The direct conditions are:

- clean observations;
- missing burst;
- outlier burst;
- coherent drift;
- identity substitution;
- delayed observations with correct lineage; and
- density drop.

The mandatory imperfect-cue conditions are:

- `reliability_false_negative`: corrupted observations receive clean-like
  reliability;
- `reliability_false_positive`: clean observations receive spuriously low
  reliability;
- `timestamp_jitter`: current observations carry noisy reported timestamps;
- `partial_stale`: current and delayed observations are mixed and timestamps are
  imperfect; and
- `mixed_identity`: observations are partial target/distractor mixtures with
  moderate reliability.

Generating truth and the actual source index are used only for scoring and
retained diagnostics. The guard sees only availability, declared reliability,
reported source time, innovation, and its frozen trust region.

## Primary endpoints

Exactly two endpoints are primary:

1. equal-seed mean full-sequence RMSE difference over all 11 stress conditions,
   `guarded_recursive - guarded_last_residual`;
2. equal-seed total materially harmful accepted-update difference over the same
   stress conditions, `guarded_recursive - recursive_gaussian`.

No other contrast may be described as primary.

## Co-equal companion review

A result is reviewed jointly with:

- corruption-window RMSE versus `guarded_last_residual`;
- recovery-window RMSE versus `guarded_last_residual`;
- clean-condition cost versus unguarded Gaussian recursion;
- harmful-seed fraction and Wilson interval;
- worst observed seed-domain regret;
- exact-fallback identity violations;
- NLL, coverage, and interval width; and
- condition-specific results, including all imperfect-cue cases.

The source-frozen review limits are:

- clean RMSE noninferiority margin: `0.100 mm`;
- harmful-seed regret margin: `0.500 mm`;
- maximum harmful-seed fraction: `0.25`; and
- maximum observed mean stress regret: `1.000 mm`.

The fresh roster must not be retuned if any criterion fails.

## Per-time-step retention

The evidence bundle retains deterministic compressed per-time-step arrays for:

- absolute forecast error;
- accepted updates;
- exact fallback;
- materially harmful updates;
- fallback reason;
- corruption masks;
- reliability; and
- reported source age.

The archive uses fixed ZIP metadata and canonical NumPy arrays, so its SHA-256 is
stable across exact reruns. `time-summary.csv` is regenerated from this archive
by aligning every seed to its own corruption start. Recovery plots therefore do
not depend on manually transcribed summary numbers.

## Reproducible analysis

Run the registered benchmark:

```bash
PYTHONPATH=src:scripts/science \
python scripts/science/run_recursive_corruption_v2.py \
  --seeds 100000:100200 \
  --output-dir outputs/recursive-corruption-v2
```

Regenerate every analysis product from retained inputs:

```bash
PYTHONPATH=src:scripts/science \
python scripts/science/analyze_recursive_corruption_v2.py \
  --result outputs/recursive-corruption-v2/result.json \
  --traces outputs/recursive-corruption-v2/traces.npz \
  --output-dir outputs/recursive-corruption-v2
```

Verify byte-exact regeneration:

```bash
PYTHONPATH=src:scripts/science \
python scripts/science/analyze_recursive_corruption_v2.py \
  --result outputs/recursive-corruption-v2/result.json \
  --traces outputs/recursive-corruption-v2/traces.npz \
  --output-dir outputs/recursive-corruption-v2 \
  --check
```

The analyzer first reduces conditions within seed and only then computes means,
SEMs, bootstrap intervals, harmful-seed fractions, and worst-seed diagnostics.
Frames, time steps, and conditions are not treated as independent replicates.

## Implementation validation

The reviewed v2 implementation, runner, analyzer, and focused regression tests
were reconstructed from a SHA-256-bound source archive and validated on Python
3.12 before publication to the branch. The validation comprised canonical Ruff
formatting, Ruff lint, five focused tests, Python byte-code compilation, and a
clean Git diff. This validates the implementation and regeneration contracts;
it does not itself promote the diagnostic dry-run numbers. Claim-bearing
numbers must be rerun from the exact merged source revision and retained in the
paper repository with their source and artifact identities.

## Scientific boundary

A positive v2 result establishes only that, under the registered family of
varying synthetic dynamics and corruption mechanisms, Gaussian recursive belief
propagation adds value beyond a deterministic residual method with the same
guard. It does not establish:

- real Prob4D provider competence;
- unseen physical-object or acquisition-session transfer;
- physical-data covariance calibration;
- validity of a latent physical-state interpretation;
- Causal4D intervention benefit;
- deployment safety;
- robustness to arbitrary unregistered corruptions; or
- deformable-object state of the art.
