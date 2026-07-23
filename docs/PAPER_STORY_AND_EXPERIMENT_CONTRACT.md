# STAGE Paper Story and Experiment Contract

Status: frozen unless explicitly revised by the authors.

## Canonical title

Learning and Retaining Underrepresented Normal Dynamics for Debiased Time-Series Anomaly Detection

## Canonical abstract

Time-series anomaly detection (TSAD) supports monitoring in industrial, online, aerospace, and clinical systems, where anomaly labels are often scarce. Normal behavior is rarely homogeneous, instead comprising dominant dynamics and less frequent variations. Mainstream sliding windows extract substantially more patches from dominant dynamics while repeatedly placing shared temporal content within varying contexts. Patch abundance can therefore bias encoder training and finite-reference coverage toward prevalent dynamics, whereas changing contexts can disperse their representations and distort the geometry used to characterize normality. This two-sided bias may leave underrepresented normal dynamics insufficiently learned or retained, causing their recurrences to receive high anomaly scores, while context-expanded dominant regions may bring true anomalies closer to normal references and lower their scores. We therefore introduce Structure-Preserving Timestamp Alignment for Geometry Enhancement (STAGE), a framework for debiased TSAD. STAGE aligns shared timestamps across shifted windows to reduce context-induced representation variation. It further uses an adaptive intermediate partition to temper abundance-driven encoder updates and reconstructs the final embedding geometry to retain representative memory exemplars at an adaptive granularity. Requiring neither auxiliary negative samples nor a pre-specified prototype count, STAGE remains lightweight and adaptive. Extensive experiments against fifteen baselines on ten benchmark datasets confirm its superiority.

## Immutable problem statement

The imbalance studied by STAGE occurs among temporal dynamics within normality, not between normal and anomalous classes. Dominant normal dynamics persist or recur and therefore generate many highly overlapping patches. Underrepresented normal dynamics appear briefly and contribute fewer normal patches. Shifted windows additionally encode shared temporal content at different relative positions and within changing contexts.

The central hypothesis is:

> Normality may become too broad around context-distorted dominant dynamics and insufficiently supported around underrepresented ones.

This produces two risks in a shared learned normality space:

- context-expanded dominant regions may shorten anomaly-to-reference distances and increase missed detections;
- insufficiently learned or retained underrepresented dynamics may leave normal recurrences far from available references and increase false alarms.

## Method responsibilities

STAGE must retain three connected responsibilities:

1. Timestamp and interval alignment reduces representation variation for shared temporal content across shifted windows.
2. Adaptive intermediate geometry tempers abundance-driven encoder updates and increases the effective exposure of sparsely observed normal dynamics.
3. Independently reconstructed final geometry retains observed representative exemplars at adaptive granularity; anomaly scores are distances to the nearest retained exemplars.

These components share one encoder and one normal-reference scoring pipeline. Ablation switches are diagnostic controls and do not define alternative production architectures.

## Required evidence

The main claims require the following controlled evidence:

- no alignment, timestamp-token-only, interval-only, and matched-budget context-only controls;
- adaptive sampling versus uniform sampling;
- adaptive observed exemplars versus uncompressed memory and same-size uniform observed memory;
- shared-timestamp representation distances across shifted windows;
- exposure and partition diagnostics for dominant and underrepresented normal regions;
- reference coverage, normal-recurrence distances, anomaly-to-reference distances, false alarms, and missed detections;
- all six official TSB-AD metrics, with results reported per dataset as well as in aggregate.

Mechanism selection uses only the official TSB-AD Tuning split. Evaluation feedback must not select parameters for a confirmatory result. Exploratory or oracle analyses must remain explicitly separated from confirmatory comparisons.

## Comparison governance

- The complete audit retains all sixteen evaluated baseline methods.
- A paper table may show fifteen baselines only under a score-independent, predeclared relevance or provenance rule.
- Runtime observed under shared concurrency is a resource diagnostic and is not eligible for the paper efficiency table.
- Raw datasets, checkpoints, embeddings, score arrays, credentials, and large logs must not be committed.
