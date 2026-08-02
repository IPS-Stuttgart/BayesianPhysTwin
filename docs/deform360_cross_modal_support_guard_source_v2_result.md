# Deform360 cross-modal support guard source result v2

## Status

This is post-open source-development evidence, not a prospective confirmation or
state-of-the-art result. The v1 three-route guard is preserved separately at
commit `1e13e30`; v2 does not relabel its exact no-ops as admissions.

V2 adds one route:

```text
cumulative tactile energy does not increase
AND
the camera-derived correction field is spatially coherent
```

The conjunction is intended to describe a stable or releasing contact regime in
which a coherent correction may be propagated. Neither condition admits an
update through this route on its own. The original tactile, causal-response, and
temporal-consistency routes remain unchanged.

## Result

| Source panel | Identity change | Chamfer change | Beneficial updates | Regressive updates | Admitted objects |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open27 | -6.97% | -5.47% | 14 | 0 | 5 |
| Stress12 | 0.00% | 0.00% | 0 | 0 | 0 |
| Combined | -5.60% | -4.43% | 14 | 0 | 5 |

Nested leave-one-object-out fitting produced eight strict joint case wins and no
case regression. The conjunctive thresholds were stable across the 17 folds:

- maximum cumulative tactile-energy change: `-0.01670` to `-0.00091`;
- minimum correction coherence: `0.80145` to `0.84952`.

The full-source thresholds are `-0.00091054` and `0.80144528`. Every rejected
interval uses exact baseline fallback, and an exact candidate no-op is not an
admission. All registered source-development checks pass.

## Claim boundary

The route itself was discovered using these opened source outcomes, so the
nested result estimates object transfer within the development data but does not
remove route-selection bias. It also does not solve the camera common-mode-bias
identifiability limit: stable tactile loading and camera coherence can coexist
with a coherent camera bias.

The result justifies writing a new prospective protocol on genuinely fresh,
non-overlapping physical objects. That protocol must preserve exact fallback,
bind every threshold before prediction, and count ordinary predictions,
technical failures, and unsealable cases separately. It must not reuse the
closed 12-case tactile cohort or held-v8.

Registered result:
`results/sota/diagnostics/deform360_cross_modal_support_guard_source_v2/result.json`

Canonical artifact SHA-256:
`a8fad12b9df844bd2152eee660e6dcdfb545830191d65cafac5a5be081e6f148`

LF-normalized Git-text SHA-256:
`db991a8bbebf838371830204239b92ad00f6aa5b4d095ba34d0eb7673be07cf9`
