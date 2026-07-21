# Experiment governance

`experiment_policy.json` is the machine-readable authority for execution and
hyperparameter selection. The important rules are:

1. All methods use the official TSB-AD Tuning/Eval partition and the
   filename-declared `tr_<N>` training prefix. Eval labels or metrics cannot
   influence training, method admission, configuration, or rerun selection.
2. Baselines receive no additional search. Use paper-reported settings first,
   then author-released settings, then official TSB-AD settings. If no matching
   setting exists, freeze one documented stability default before Eval.
3. STAGE receives the same 24-trial budget for every benchmark subset on
   official Tuning. A subset may select a different frozen configuration, but
   individual Eval series may not. Exathlon gets no extra budget and no Eval
   exception.
4. Run all admitted baselines first, with PaAno first on the GPU queue. Verify
   baseline completeness and zero errors before starting STAGE. The formal
   launcher defaults to `baseline`; `target` is blocked until all same-seed
   baseline JSONs are complete.
5. Store the six metrics at full precision. Three-decimal values are rendering
   artifacts only. Do not retain checkpoints or anomaly-score arrays.
6. Main-run timing is diagnostic. Report Average Inference Time only after a
   separate exclusive rerun with identical hardware and measurement rules.

The two recent AAAI baselines remain behind a configuration/admission gate.
Once both are frozen, the intended paper table contains sixteen baselines plus
STAGE.
