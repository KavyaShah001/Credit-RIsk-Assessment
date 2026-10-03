# Model Results

Generated from the current source dataset and model pipeline on 2026-10-03 19:19 UTC. Regenerate with `.venv/bin/python scripts/train_models.py`.

## Outcome, split, and selection

- Target: bankruptcy within five years, used as a default-risk proxy.
- Model selected by validation PR-AUC: **gradient_boosting**.
- Split: stratified random, 4,216 train / 1,405 validation / 1,406 test observations.
- Positive outcomes: 163 train / 54 validation / 54 test.
- The source provides no observation dates or persistent company keys, so a temporal or company-grouped holdout was not possible. The test set was not used for model selection.
- Class weights address training imbalance; validation and test retain the natural source prevalence.

## Validation model comparison

The table shows validation metrics at the default 0.50 threshold. The model is selected by PR-AUC, since accuracy alone can conceal poor default capture when outcomes are rare.

| Model (selected by validation PR-AUC) | ROC-AUC | PR-AUC (average precision) | Brier score | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | 0.9565 | 0.7825 | 0.0157 | 0.8409 | 0.6852 | 0.7551 |
| random_forest | 0.9391 | 0.6625 | 0.0242 | 0.9130 | 0.3889 | 0.5455 |
| logistic_regression | 0.8947 | 0.3973 | 0.0969 | 0.2176 | 0.7778 | 0.3401 |

## Selected model: raw versus calibrated probabilities on test

Sigmoid calibration improved validation Brier score and is selected for displayed PD. Calibration was fitted by cross-validation within training data. The table below uses a 0.50 threshold for precision/recall/F1; the ranking and probability metrics do not depend on that threshold.

| Probability | ROC-AUC | PR-AUC (average precision) | Brier score | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 0.9678 | 0.7872 | 0.0153 | 0.8372 | 0.6667 | 0.7423 |
| sigmoid_calibrated | 0.9678 | 0.7872 | 0.0145 | 0.9412 | 0.5926 | 0.7273 |

The validation-selected operating threshold is **0.3855**. At that threshold on the held-out test set: ROC-AUC 0.9678, PR-AUC 0.7872, precision 0.8750, recall 0.6481, and F1 0.7447. The threshold was selected on validation data, not on the test set.

## Internal grade mapping

This is an explicit project assumption for demonstration; it is not an external scale or lending policy.

| Five-year PD proxy band | Grade | Description |
| --- | --- | --- |
| <1.0% | I1 | Lowest modeled risk |
| 1.0% to <3.0% | I2 | Lower modeled risk |
| 3.0% to <7.5% | I3 | Moderate modeled risk |
| 7.5% to <15.0% | I4 | Elevated modeled risk |
| >=15.0% | I5 | Highest modeled risk |

## Global feature importance

Permutation importance is the average decrease in validation PR-AUC after shuffling a feature. It measures model reliance in this dataset, not causality.

| Feature | Category | Source definition | Mean PR-AUC decrease | Std. deviation |
| --- | --- | --- | --- | --- |
| ratio_27 | Coverage / obligations | Profit on operating activities / financial expenses | 0.6341 | 0.0121 |
| ratio_11 | Profitability | (Gross profit + extraordinary items + financial expenses) / total assets | 0.1125 | 0.0082 |
| ratio_34 | Coverage / obligations | Operating expenses / total liabilities | 0.0866 | 0.0122 |
| ratio_09 | Efficiency / other | Sales / total assets | 0.0599 | 0.0146 |
| ratio_46 | Liquidity / working capital | (Current assets - inventory) / short-term liabilities | 0.0571 | 0.0137 |
| ratio_05 | Liquidity / working capital | (Cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation) × 365 | 0.0280 | 0.0112 |
| ratio_29 | Efficiency / other | Logarithm of total assets | 0.0212 | 0.0051 |
| ratio_58 | Profitability | Total costs / total sales | 0.0206 | 0.0034 |
| ratio_56 | Profitability | (Sales - cost of products sold) / sales | 0.0100 | 0.0049 |
| ratio_13 | Profitability | (Gross profit + depreciation) / sales | 0.0073 | 0.0045 |

## Example local sensitivity

For observation **UCI-1YEAR-005354**, selected PD proxy is 0.46%, risk score 100, grade **I1**. The factors below show how the model score changes if one ratio at a time is replaced by its training-set median. They are sensitivity checks, not causal contributions.

| Direction relative to median scenario | Feature | Source definition | PD change |
| --- | --- | --- | --- |
| Higher than median-setting scenario | ratio_30 | (Total liabilities - cash) / sales | -0.224 pp |
| Higher than median-setting scenario | ratio_43 | Receivables plus inventory turnover in days | -0.140 pp |
| Higher than median-setting scenario | ratio_03 | Working capital / total assets | -0.053 pp |
| Higher than median-setting scenario | ratio_13 | (Gross profit + depreciation) / sales | -0.053 pp |
| Higher than median-setting scenario | ratio_46 | (Current assets - inventory) / short-term liabilities | -0.032 pp |
| Lower than median-setting scenario | ratio_25 | (Equity - share capital) / total assets | +0.714 pp |
| Lower than median-setting scenario | ratio_06 | Retained earnings / total assets | +0.309 pp |
| Lower than median-setting scenario | ratio_05 | (Cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation) × 365 | +0.296 pp |
| Lower than median-setting scenario | ratio_38 | Constant capital / total assets | +0.064 pp |
| Lower than median-setting scenario | ratio_11 | (Gross profit + extraordinary items + financial expenses) / total assets | +0.063 pp |

## Limitations

- The target is five-year bankruptcy, not a directly observed contractual default event.
- The random holdout is not a time-based backtest; source-period differences may affect results.
- The held-out test set contains 54 positive cases. Performance and calibration therefore have substantial statistical uncertainty.
- Calibration should be treated as a check, not proof that probabilities transfer to new sectors, geographies, or time periods.
- Project grades, score thresholds, model explanations, and any future LGD/EAD assumptions are educational outputs, not operational decisions.
