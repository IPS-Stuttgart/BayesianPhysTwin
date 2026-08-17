# Learned-twin backend interface v1

This interface lets BayesianPhysTwin consume fixed-identity physical rollouts
from learned deformable-twin methods without conflating three different claims:

1. a method is described in a paper;
2. a public runtime and checkpoint can actually be executed; and
3. BayesianPhysTwin reproduced that runtime under a causal evaluation contract.

The command is an active research interface:

```bash
bpt experiment run materialize-learned-twin-backend profiles
```

## Frozen availability snapshot

The registry is frozen at `2026-08-18`. Upstream revisions are exact where a
public repository exists; an absent revision is represented as absent rather
than guessed.

| Profile | Public release at snapshot | BayesianPhysTwin native support | Portable intake |
| --- | --- | --- | --- |
| `matphys-v1` | Executable public source at `Yrainy0615/MatPhys@c16b858d...` | Guarded spring proposal replay through PhysTwin/Warp | Yes |
| `neuspring-v1` | Repository contains metadata only at `GhiXu/NeuSpring@51d94f67...` | Blocked on a public runtime and checkpoint | Yes |
| `physpring-v1` | Paper only; no public runtime registered | Blocked on a public runtime and checkpoint | Yes |
| `physworld-v1` | Repository contains metadata and figures only at `AlanYoung123/PhysWorld@157a309e...` | Blocked on a public runtime and checkpoint | Yes |
| `egophys-v1` | Project announces code and data as unavailable | Blocked on public runtime, data, and checkpoint | Yes |

“Portable intake” means an independent producer can export the strict six-array
[`physical_rollout_v1`](../src/bayesian_phystwin/physical_rollout_v1.py)
contract. It does **not** mean the repository can execute the named method. A
portable artifact always records `official_method_reproduction_claimed=false`
and `published_benchmark_parity_claimed=false`.

## Support levels

The interface makes backend progress incremental and inspectable:

| Level | Meaning |
| --- | --- |
| Registry | Paper identity, public-release status, and exact public revision are recorded. |
| Portable intake | A producer-generated fixed-identity rollout can be custody checked and consumed. |
| Native adapter | BayesianPhysTwin can invoke a public implementation under a source-specific contract. |
| Independent reproduction | A frozen run has been reproduced from public inputs and exact source/checkpoint identities. |
| Claim-bearing evaluation | A preregistered source/target protocol passes its transfer and calibration gates. |

This change raises every listed method to registry and portable-intake support.
It does not promote unavailable methods to native-adapter support.

This is a separate experimental interface catalog, not an admission into the
canonical `material_backend_v1` recommendation registry. Every profile remains
at `registered-adapter`, has `source_value_qualified=false`, and is not
recommended for claim-bearing evaluation. The
[evidence-first admission freeze](backend_admission_policy_v1.md) remains in
force.

## Portable contract

The input archive must contain exactly:

```text
prediction_m                 (T,N,3)
persistence_m                (T,N,3)
driven_readout_m             (T,N,3)
zero_action_readout_m        (T,N,3)
action_support               (N,)
frame_zero_points_m          (N,3)
```

All arrays use one floating dtype, finite metres, fixed material identity, and
the canonical `right-handed-z-up-world-v1` frame. The output contains the
deterministically encoded physical archive, the exact source archive under
`provenance/`, a content-addressed manifest, and `SHA256SUMS`.

Every model/checkpoint/configuration file is supplied as a logical path and an
external ordinary file. Its SHA-256 and byte count are bound at materialization.
`validate --verify-sources` rehashes those external files; ordinary validation
remains portable after the bundle is moved.

## Causal and parity modes

`causal-source-v1` requires:

- the target object is absent from the declared training objects;
- target evidence ends no later than rollout start;
- no target future observation is used; and
- prediction is sealed before future scoring.

`published-parity-v1` records an explicitly noncausal control. It can preserve
published preprocessing or target fitting, but its artifact is never marked
causal or an official reproduction by this generic intake.

## Example

```bash
bpt experiment run materialize-learned-twin-backend build \
  physical-rollout.npz output/neuspring-case \
  --profile neuspring-v1 \
  --mode causal-source-v1 \
  --model-artifact checkpoints/model.pt=/models/model.pt \
  --source-artifact producer.py=<sha256> \
  --producer-repository owner/portable-producer \
  --producer-revision <40-or-64-character-revision> \
  --case-id case-001 \
  --target-object-id target-object \
  --training-object-id source-object \
  --evidence-start 0 --evidence-stop 6 \
  --rollout-start 6 --rollout-stop 26

bpt experiment run materialize-learned-twin-backend validate \
  output/neuspring-case --verify-sources
```

## Advancement rule

A native adapter should only be added after an executable public runtime and
the necessary checkpoint/configuration artifacts exist. It must then bind the
upstream source and checkpoint, prove the causal input boundary, preserve
material identities, pass a zero/identity replay, and seal predictions before
outcome access. Registry or portable-intake support alone cannot enter a SOTA
table as a reproduced result.
