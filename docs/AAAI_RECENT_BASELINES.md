# AAAI 2024/2025 recent-baseline extension

This document records the selection decision for two recent AAAI baselines.
They are an **extension** to the active no-KNN 15-method comparison, not a
mutation of `protocol.json`. The base comparison therefore remains 15 methods,
193 series, and 2,895 method-series units; admitting both extensions would
produce 17 methods and 3,281 method-series units.

## Decision

| Year | Extension name | Paper | Why it is selected | Limitation |
| --- | --- | --- | --- | --- |
| 2024 | `DPAD_AAAI24` | [Dense Projection for Anomaly Detection](https://ojs.aaai.org/index.php/AAAI/article/view/28682) | AAAI-24 main-track deep unsupervised anomaly detector; learns a locally dense representation and scores with kNN in representation space. It can be trained only on the filename-declared prefix. | The paper is tabular rather than time-series specific and does not synthesize negative samples. A time-series window adapter is required. |
| 2025 | `DNE_AAAI25` | [Unsupervised Anomaly Detection for Tabular Data Using Deep Noise Evaluation](https://ojs.aaai.org/index.php/AAAI/article/view/33257) | AAAI-25 main-track deep detector that creates diverse noise-augmented samples from clean training data and learns to regress the element-wise noise magnitude. This directly matches the requested auxiliary-negative construction. | The paper is tabular rather than time-series specific. The project implementation must be identified as a paper-faithful reimplementation unless an author repository is pinned later. |

The papers and their supplements define the main mechanisms. DPAD optimizes
weighted pairwise representation distances, detaches the exponential weights,
regularizes layer norms to avoid collapse, and applies kNN after projection.
DNE trains on clean samples with a zero target and on generated noisy samples
with the absolute injected noise as target. The DNE paper reports PyTorch on an
NVIDIA Tesla V100, a four-layer MLP or residual MLP, AMSGrad, Gaussian noises,
and 500 epochs; the [full arXiv version](https://arxiv.org/abs/2412.11461)
contains the training algorithm and architecture appendix.

## Why the AAAI-24 time-series paper is not selected

[When Model Meets New Normals](https://ojs.aaai.org/index.php/AAAI/article/view/29210)
is a directly relevant AAAI-24 time-series anomaly-detection paper, but its
central contribution is test-time adaptation on test observations. The frozen
DuoBa comparison permits fitting only on the filename-declared `tr_<N>` prefix.
Using Eval observations for parameter updates would therefore make its results
incomparable with the active 15-method table. It remains a related-work item,
not a baseline in this protocol.

## Time-series adapter shared by both methods

The papers operate on fixed-dimensional samples. For a time series, the
extension will convert the prefix and full sequence into fixed-dimensional
windows without reading labels:

- `U`: use the same training-prefix-only, data-derived periodic window rule as
  the existing classical window baselines; no Eval value or label chooses it.
- `M`: use one timestamp's channel vector as one tabular instance unless a
  Tuning-only experiment freezes a different window before the formal run.
- normalize only from `values[:tr_N]` statistics;
- fit only on windows wholly contained in `values[:tr_N]`;
- score all full-sequence windows and aggregate overlapping window scores back
  to point scores;
- evaluate with the same six frozen metrics and store no checkpoint or score
  array.

Any engineering cap needed to keep DPAD's pairwise objective tractable must be
chosen on the official Tuning split, recorded in
`extensions/aaai_recent/protocol_extension.json`, and frozen before Eval. It
must not be chosen from Eval runtime or accuracy.

## Implementation status and provenance

The local implementation is now complete at the portability-smoke level:

- `extensions/aaai_recent/models.py` contains a bias-free DPAD projection,
  the detached dynamic pair weights and anti-collapse layer-norm penalty from
  the paper equation, plus the DNE four-layer MLP, five-block ResMLP, and
  Algorithm-1-style diverse Gaussian noise generator;
- `extensions/aaai_recent/run_one_recent.py` applies the shared prefix-only
  time-series adapter, scores the complete sequence, and invokes the unchanged
  six-metric evaluator;
- `validation/aaai_recent_local_00gwk_smoke.json` records 4/4 valid U/M smoke
  units without runtime fields.

No official public source URL was found for either selected method. The code
is therefore identified as a paper-derived reimplementation, not author code.
DPAD's paper gives its objective, no-bias constraint, 100-epoch experiments,
and kNN scoring, but does not fully specify a tabular network and optimizer.
Consequently its three-epoch local settings, hidden widths, Adam optimizer,
pairwise-mean scaling, and 512-window ceiling are explicitly smoke-only.

DNE is more fully specified: the implementation follows the appendix's hidden
width rule, ResMLP depth, AMSGrad learning rate and weight decay, Gaussian
noise level construction, three noise ratios, maximum aggregation, and the
epoch-100 learning-rate decay. Only the local epoch count is reduced from 500
to 3. Even so, the time-series window choice and final formal parameters must
still pass the Tuning-only freeze gate.

## Provenance and acceptance gates

The two implementations must pass all of the following before formal results
are accepted:

1. implementation provenance and paper equations are documented;
2. deterministic seed `2027` is honored on CPU and CUDA;
3. one official U series and one official M series each produce six finite
   metrics locally under `00gwk`;
4. the complete Tuning-only configuration is frozen and hashed;
5. no Eval label or Eval score influences configuration;
6. no persistent checkpoint or score array is produced;
7. formal results use a separate result root and never overwrite the active
   15-method results.

Gates 1-3 and 6 have passed for the smoke profile. Gates 4, 5, and 7 remain
mandatory before the 386-unit extension may run on Eval.

Runtime ordering is deliberately not claimed here. Whether either method is
slower than DuoBa and faster than GBOC must be measured later on exclusive,
identical hardware with the efficiency protocol, not inferred from model
architecture or shared-machine smoke tests.
