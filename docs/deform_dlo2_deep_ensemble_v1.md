# Fresh DEFORM DLO2 two-seed confirmation v1

This route is an independent confirmation of the two-seed candidate registered
on DLO1. It is separate from the checkpoint-posterior route and cannot enumerate
or hash DLO2 training trajectories unless the DLO1 ensemble selected a
non-fallback arm, improved validation and source error by at least 1%, won at
least five of eight DLO1 source cases, and produced a finite validation-fitted
variance scale.

The two DLO2 members use seeds 42 and 43 with the same split, 50-frame horizon,
batch size 32, 6,400-update budget, checkpoint schedule, physical
initialization, and official SGD parameter groups. The only registered ensemble
operators are the equal-weight predictive mean and the validation-softmax
predictive mean with the DLO1-locked 1 mm temperature. A failed validation gate
falls back exactly to the lower-validation-error member.

The fresh source gate additionally requires the selected ensemble to improve on
that member by at least 1%, win at least five of eight source trajectories, stay
within 1.1 times the published 9.7 mm DLO2 reference, and beat persistence on at
least six of eight trajectories. Failure leaves the official evaluation closed.

The relevant frozen files are:

- `configs/sota/deform_dlo2_deep_seed42_v1.json`
- `configs/sota/deform_dlo2_deep_seed43_v1.json`
- `configs/sota/deform_dlo2_deep_ensemble_eval_v1.json`
- `scripts/remote/run_deform_dlo2_deep_seed.py`
- `scripts/remote/run_deform_dlo2_deep_ensemble.py`

Each seed wrapper verifies the checksummed DLO1 result and its selection seal
before invoking the generic source runner. Both source results must then bind the
same DLO1 parent digest and byte-identical DLO2 source partition. The evaluator
installs the official-evaluation read guard before loading any trajectory.

Passing this stage authorizes only a no-retuning, all-training-data DLO2 refit
with the same selected operator and weights. It is not itself a state-of-the-art
claim, and it does not authorize online observations or future-frame inputs.
