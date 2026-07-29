# Deform360 Adaptive Causal Direct Depth V14

## Status

V14 is implementation-locked before selecting a fresh source cohort. It is a
prospective source-development study, not a result, confirmation, or
state-of-the-art claim. Only a complete pass of the frozen source gates may
authorize a separately preregistered evaluation on independent fresh objects.

## Why This Is Not Another Tracker Arm

V13 established two separate facts. Its target-free camera carrier succeeded
on six of eight opened source cases, but its fixed frame-zero TAPNext++
identity provider retained only 5 of 96 endpoint identities. V14 keeps the
former and discards the latter.

The V14 carrier chooses eight complete cameras at frame zero and splits them
into disjoint four-camera proposal and validation panels. At every tested
causal endpoint, V14 re-associates local metric RGB-D geometry around the
selected physical or persistence backbone. It does not propagate a fixed
camera track identity through the prefix.

The strict arm requires three supporting views in each panel. The fallback arm
requires two views in each panel, multiplies local covariance by four, and
retains a separate 5 mm shared-camera-bias nuisance. Cases without either
carrier abstain before source locking.

## Causal Boundary

V14 scans only frames 0 through 57 and selects the earliest endpoint that
passes all registered conditions:

1. the released tactile stream supports contact;
2. the measured actuator has moved by the required amount;
3. observed nonrigid response is large enough;
4. the response aligns with the physical action-conditioning trajectory;
5. proposal and validation panels agree under the shared-bias covariance; and
6. the proposal correction improves the untouched validation-prefix residual.

The proposal panel compares physical motion with exact persistence. Physical
motion must win by 5% to become the prediction backbone; otherwise persistence
is selected. The physical trajectory remains a separate action-conditioning
signal in either case.

The proposal panel forms the candidate. The validation panel can only admit or
reject it. Every rejection returns the selected baseline byte for byte. No
object observation after frame 57, hidden identity, future geometry, or future
metric may be read before all twelve source predictions or exact fallbacks are
sealed.

## Reliability And Calibration Boundary

Association probability comes from local candidate geometry and mask/depth
support. Prior reliability comes from residual-independent redundancy and
view-scatter cues. The innovation against the selected twin trajectory is not
fed back into either prior quantity; it enters the robust mixture likelihood
once.

Metric covariance is carried in square metres. It includes local depth and
pixel uncertainty, assignment-mixture spread, temporal unknown-correlation
inflation, arm-specific two-view inflation, cross-view scatter, bounded
per-endpoint Sim(3) fit residual, and the shared-camera-bias variance.

The per-endpoint Sim(3) operation is a nuisance debiasing step, not evidence
that global scale or pose is known. It removes bounded global pose and scale
from the sparse endpoint displacement before the nonrigid readout discrepancy
is inferred.

The resulting candidate remains an observable/readout correction. V14 does
not claim to update the hidden Warp state or to identify a physical material
mechanism.

## Freshness And Admission

The source cohort will contain twelve physical objects, one episode per
object, split into three object-level cross-fit folds of four. The hash-only
exclusion union contains 138 physical objects previously opened, reserved,
selected, or technically dispositioned by relevant Bayesian-PhysTwin,
Prob4D, MolmoMotion-Field, and held-v8 work.

Before a case can enter the source lock, an outcome-blind preflight must verify:

- released metadata enums;
- 76-frame episode, robot, and tactile streams;
- the physical backend's 128-to-10,000-node contract;
- all twelve registered camera records;
- at least eight complete depth, mask, and calibration streams;
- complete source checksums; and
- an admitted strict or inflated adaptive carrier.

The preflight retains hashes rather than plaintext object or episode identity.
A rejected preflight is not a selected source case. Once selected, a case is
never silently replaced after any prediction or outcome disposition.

## Advancement Gate

The source study must seal twelve predictions or exact fallbacks without a
technical failure. At least six objects must admit an update. Relative to the
unchanged selected baseline, V14 must:

- improve object-balanced disjoint hidden-identity error by at least 5%;
- improve object-balanced Chamfer distance by at least 5%;
- jointly win on at least 8 of 12 objects;
- keep every single-object regression below 5%;
- keep false-safe admissions below 10%; and
- pass the registered three-fold source-calibrated upper-regret guard.

Early, middle, and late errors, NEES, coverage, and interval width are reported
regardless of the decision. Failure closes V14 without threshold changes.
Passing authorizes only a new independent target protocol; it is not itself a
confirmation.

## Synthetic Controls

Before source locking, the production V14 wrapper must pass frozen strict and
inflated positive controls plus rigid-bias, cross-panel-inconsistency, and
missing-contact placebos. These controls test implementation sensitivity,
specificity, covariance routing, and exact fallback. They are not real-data
evidence.

The frozen controls passed: 12 of 12 positive trials produced an update, none
of 12 placebos was admitted, all 12 placebos preserved the baseline exactly,
and mean synthetic continuation error improved by 11.64%. The source-lock
builder validates and binds this checksummed result; it cannot create a cohort
lock from a missing, altered, or failed control artifact.

## Claim Boundary

The strongest possible source-stage statement is:

> On twelve fresh development objects, a prospectively locked,
> tactile-supported adaptive direct-depth readout update passed or failed its
> registered transfer, safety, and calibration gates.

No SOTA language is permitted until a later independent target protocol is
locked and completed.
