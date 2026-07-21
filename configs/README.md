# Frozen STAGE configurations

This directory stores small, reviewable parameter manifests, not model
checkpoints.  `scripts/stage_autopilot.py freeze` writes
`stage_locked_seed2026.json` only after all ten official TSB-AD Tuning subsets
have completed the same 24-candidate search budget.

The locked manifest records the exact Tuning files, objective, winning trial,
full-precision Tuning means, source hash, and a content fingerprint.  Eval and
later seeds may read it but may not modify it.  Checkpoints and score arrays
remain forbidden.
