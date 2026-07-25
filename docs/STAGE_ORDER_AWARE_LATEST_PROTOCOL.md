# Latest order-aware STAGE protocol

This branch executes only the standalone order-aware STAGE source supplied by
the user.  It never launches a baseline and never imports an older STAGE model.

Stage 1 compares six predeclared order-readout settings on all 22 official
TSB-AD Tuning series at seed 2026.  The dataset-specific training and memory
anchors come from the earlier Tuning-only STAGE lock, but every unit retrains
the new source.  Candidates are ranked separately for each of the ten datasets
by the unrounded macro mean of **VUS-PR only**.  The supplied LaTeX baseline
table is post-freeze Eval context and is not readable by the runner.

After Stage 1, the top candidates must be confirmed on seeds 2027 and 2028.
Only then may final-memory granularity and k be selected on official Tuning.
An independent Eval is allowed only after ten dataset-specific locks exist.

Safety rules:

- strict PyTorch deterministic algorithms and deterministic cuBLAS;
- source, protocol, official data, and evaluator hashes in the frozen plan;
- one atomic JSON per candidate/seed/series unit;
- no checkpoint, embedding, score-array, or baseline artifact;
- fail-fast controller that terminates peer workers after the first failure;
- no overwrite of a different plan, invalid unit, summary, or selection.
