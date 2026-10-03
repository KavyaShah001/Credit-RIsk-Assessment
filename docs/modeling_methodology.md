# Phase 4 — Credit-risk modelling methodology

This document describes the legacy Polish experiment only. The annual-panel model has a different horizon and a chronological development design; see [its methodology](temporal_methodology.md).

## Outcome and prediction horizon

The binary target is the source's bankruptcy label for the 1st-year case, which UCI defines over the following five years. The model therefore estimates a **five-year bankruptcy probability proxy**. It does not estimate a lender-specific missed-payment/default event.

## Feature set and leakage controls

- Model inputs are the 64 source financial ratios. The target is kept separate and never enters the feature matrix.
- No reporting date or persistent company identifier is present. The system cannot perform a temporal or company-grouped holdout with this file.
- A fixed-seed, stratified random split allocates 60% to training, 20% to validation, and 20% to final test.
- Median imputation, missingness indicators, and scaling are fitted inside a scikit-learn pipeline using training rows only. Validation and test missing values do not influence fitted preprocessing.
- The held-out test set is reserved for reporting. Candidate selection, probability-calibration choice, and decision-threshold choice use validation data.
- Balanced class weights are used in training. Validation and test prevalence remain unchanged; no synthetic resampling is applied.

## Model comparison

The MVP compares:

1. Logistic regression with balanced class weights and a standardized feature pipeline.
2. Random forest with balanced subsample class weights and a conservative minimum leaf size.
3. Histogram gradient boosting with balanced class weights and regularization.

Validation **PR-AUC (average precision)** is the model-selection metric because the positive class is rare. ROC-AUC measures ranking across thresholds; Brier score measures probability error; precision/recall/F1 and the confusion matrix are reported at a stated threshold. Accuracy is supplementary.

## Calibration and operating threshold

The selected model is calibrated with sigmoid/Platt calibration using three-fold cross-validation within the training set. Validation Brier scores compare its calibrated and raw probabilities; use calibrated PD only when that calibration improves validation Brier score. Isotonic calibration is not part of the MVP because the held-out calibration folds contain relatively few positive events.

An illustrative F1 threshold is chosen on validation data using the selected probability output. The test set measures that fixed threshold once. It is not a cost-optimal lending threshold: different credit decisions have different false-positive and false-negative costs.

## Internal score and grade assumptions

- Score = round(100 × (1 - selected PD)); higher score means lower modeled risk.
- The Project Internal Credit Risk Rating bands are defined in the model code and displayed in the generated model-results document.
- Bands are transparent project assumptions; they are not mapped to any external rating scale or lending rule.

## Explainability

- Global feature importance is permutation importance on validation PR-AUC.
- The individual example uses a one-feature-at-a-time sensitivity: replace one observed ratio with its training-set median and measure the selected model PD change.
- These techniques describe model behavior in the observed data. They do not establish causation, and correlated ratios can make single-feature sensitivities misleading.

## Validation limitations

The dataset has no dates or persistent entity identifiers. A random split may not represent a future-period deployment and cannot rule out dependence among observations. UCI also reports different observation windows for bankrupt and still-operating observations. The test sample contains few bankruptcies, so performance and calibration estimates have substantial uncertainty. Treat every output as a student demonstration, not a deployable credit decision.

## Generated outputs

Run the model training script from the project root with the virtual environment active. The pipeline creates:

- validation model comparison CSV
- raw-versus-calibrated test metrics CSV
- test metrics at the validation-selected threshold CSV
- held-out ROC, precision-recall, calibration, and confusion-matrix plots
- permutation importance on validation data
- example-observation sensitivity explanation
- model card JSON
- a local fitted model
- model-results Markdown generated from the actual metrics and explanation outputs

Raw datasets, the local database, fitted model, and report artifacts stay out of Git. The generated model-results document is source-controlled so the documented results can be reviewed and reproduced.
