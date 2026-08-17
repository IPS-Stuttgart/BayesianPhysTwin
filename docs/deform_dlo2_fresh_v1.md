# Fresh DEFORM DLO2 confirmation protocol

The DLO2 model is constructed with the DLO-specific rest geometry, rest lengths,
and stiffness parsed from the locked external upstream source under
`official-deform-dlo-initialization-v1`. The target-free correction and its
information boundary are recorded in
`docs/deform_dlo2_initialization_amendment_v1.md`.

DLO2 is the independent source confirmation for the long-budget DEFORM route.
Its training values, source outcomes, and official evaluation values remain
unopened until the frozen DLO1 long-run result explicitly authorizes this stage.

## Locked design

- Upstream DEFORM commit: `b73b8b8ecc033caefa693fab7898741d4e6dbeff`
- DLO2 node count: 12
- Official-train split: 40 fit, 8 validation, 8 source-test trajectories
- Split seed: `deform-dlo2-fresh-v1-20260731`
- Training: from scratch, batch size 32, horizon 50, 6400 updates
- Validation checkpoints: 0, 280, 640, 1280, 2560, 4000, 5200, 6040, 6400
- Source parity threshold: 1.1 times the published 9.7 mm reference
- Persistence gate: at least 6 of 8 paired wins

The source runner now supports protocol-declared node counts while preserving an
explicit DLO1 compatibility wrapper. DLO1 behavior is unchanged.

## Parent authorization

The wrapper refuses to enumerate or hash DLO2 source trajectories unless its
parent result has:

1. contract `deform-dlo-longrun-result-v2`;
2. `official_eval_read = false`;
3. a passed DLO1 source gate;
4. `checkpoint_posterior_authorized = true`; and
5. the exact executing long-run protocol SHA-256
   `a1c7ad23ccf6c83e0130efb9c9b172063892caf5b0066ca79cd6ad3b151515a0`.

Thus even DLO2 preflight remains sealed while the DLO1 run is unresolved.

## Posterior and evaluation

The parameter-mean and predictive-mean checkpoint arms, temperatures,
uncertainty floor, and exact fallback are copied byte-for-byte from the separate
DLO1 posterior policy. DLO2 source outcomes cannot change them.

After the single-checkpoint DLO2 source gate passes, the separate posterior
runner selects among those fixed arms using only the eight DLO2 validation
trajectories. It preserves the selected single checkpoint exactly unless the
best posterior arm improves validation L1 by at least 1%. A non-fallback arm
must then improve the untouched eight-case source panel by at least 1% and win
at least five paired cases before it can authorize an identical-information
official evaluation. The official evaluation directory remains read-guarded
throughout this stage.

Passing the DLO2 source gate can authorize an identical-information official
evaluation. It cannot authorize online-prefix assimilation in the SOTA table;
that remains a separately labeled information setting.

If the Bayesian posterior also passes its fresh transfer gate, the selected
operator and weights advance unchanged to the all-56 final refit described in
`deform_dlo2_alltrain_refit_v1.md`. This restores the full official training
budget before evaluation without reopening method selection.

## Command

```bash
python scripts/remote/run_deform_dlo2_fresh.py \
  --protocol configs/sota/deform_dlo2_fresh_v1.json \
  --parent-longrun-result /path/to/longrun_result.json \
  --upstream-root /path/to/DEFORM \
  --output-root /path/to/dlo2-fresh-v1 \
  --device cuda:0 \
  --mode run
```

Conditional posterior command:

```bash
python scripts/remote/run_deform_dlo2_posterior.py \
  --protocol configs/sota/deform_dlo2_fresh_v1.json \
  --source-result /path/to/dlo2-fresh-v1/source_run/source_result.json \
  --upstream-root /path/to/DEFORM \
  --output-root /path/to/dlo2-posterior-v1 \
  --device cuda:0
```
