# Deform360 grounded frame-zero masks (v3)

The reusable-twin experiment requires one object observation at the selected
action-window start. Independent generic SAM2 masks admitted the scarf source
case but failed the unchanged 3 cm contact gate for cable and penguin before
any future object outcome was opened. Borrowing appearance from neighboring
objects also failed for cable.

V3 changes only candidate proposal. The object prompt already published in
Deform360 metadata is frozen per object. Grounding DINO
`IDEA-Research/grounding-dino-base` at revision
`12bdfa3120f3e7ec7b434d90674b3396eccf88eb`, through Transformers 4.57.3,
proposes frame-zero boxes. The pinned SAM2.1 small checkpoint produces masks
inside those boxes. The existing calibrated joint-multiview selector and all
camera-count, visual-hull, and 3 cm contact gates remain unchanged.

The candidate prior may use the text prompt, frame-zero RGB, camera
calibration, and the known robot trajectory for the final contact QA. It may
not use a simulator residual, tactile data, post-initial object imagery, or an
observed outcome. A failed gate returns exact persistence; the attachment
radius is never widened.

This is an observation admission repair, not a dynamics contribution. It does
not modify the canonical graph, physical parameter grid, trust rule, held
episodes, or success thresholds. The v2 artifacts remain immutable for audit.
