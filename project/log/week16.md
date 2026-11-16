# Week 16 / P3 — Causal RCA and negative controls

Executed on 2026-11-16 as a logical continuation milestone.

Implemented actual input gradients of the frozen detector score and channel
occlusion using train-normal medians/discrete modes. Occlusion batches independent
channel interventions; a test verifies equality with serial interventions. No
residual proxy is presented as gradient attribution.

R01–R05 now compare residual, gradient, occlusion, fusion without graph, and full
fusion. Snapshots use only available history at the first alarm and five-second
updates. Channel onset thresholds come from normal calibration; stability uses
recent actual top-three memberships. Full variable rankings, evidence components,
checkpoint hashes, evidence margins and availability are exported.

Strict directly observable targets and weak associated-loop targets remain
separate. End-to-end missed-event scores are zero; unavailable target sets remain
undefined. A regression test ensures a preexisting alarm's initial ranking cannot
be replaced by a later ranking within the attack. N01 graph rewiring, N02 target
permutation, N03 onset permutation, N04 PA comparison and N05 label/filename
invariance are registered and executed in the real smoke matrix.

The 1,900-row test1 fixture contains an attack on an internal, unobserved target.
Strict-target localization is therefore unavailable for that prefix. The weak
reference is evaluated explicitly; it is not relabeled as a strict success.
Broader strict-target performance and event coverage require the full HAI run.
