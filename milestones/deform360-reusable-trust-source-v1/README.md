# Deform360 reusable-twin trust source milestone

This milestone freezes the exhausted-source diagnosis and the fitted candidate
used by `deform360-reusable-trust-fresh-v1`. It contains no calibration or
target outcomes.

The cross-fitted closure-and-self-diagnostic arm improves the 27-episode source
panel by 5.65% future track error, 6.00% future Chamfer, 8.21% late track error,
and 9.62% late Chamfer. Its maximum future degradation is 2.71% in track error
and 2.60% in Chamfer. These are discovery results, not a state-of-the-art claim.

The standalone candidate uses known robot motion and openness, frame-zero
object geometry, and the simulator's predicted response. It uses no tactile,
symbolic action label, post-initial object observation, or held-out outcome. A
rejected closure gate returns exact persistence.

The fresh admission panel is locked in
`../../configs/causal4d_public/deform360_reusable_trust_fresh_v1.json`. All
twelve predictions must be generated and hashed before any held-out outcome is
opened.
