# Chronological validation results

Generated from run `b2e25b34bcd2f5b5`. Selected model: **random_forest**; probability source: **chronological sigmoid calibration**.

The outcome is a reconstructed next-fiscal-year bankruptcy proxy. The source has fiscal years, but no exact filing availability or event dates. These results measure calendar-year transfer under that limitation.

## Chronological periods

| split | first_year | last_year | rows | companies | events | event_rate | last_label_year |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1999 | 2007 | 32597 | 5569 | 210 | 0.0064 | 2008 |
| calibration | 2009 | 2010 | 5808 | 3138 | 49 | 0.0084 | 2011 |
| validation | 2012 | 2013 | 5562 | 3018 | 41 | 0.0074 | 2014 |
| test | 2015 | 2018 | 12282 | 3700 | 119 | 0.0097 | 2019 |

## Candidate selection on validation only

| model | roc_auc | pr_auc | brier | calibrated_brier | use_calibrated |
| --- | --- | --- | --- | --- | --- |
| random_forest | 0.8561 | 0.0478 | 0.0350 | 0.0072 | True |
| gradient_boosting | 0.7780 | 0.0403 | 0.0394 | 0.0072 | True |
| logistic_regression | 0.8252 | 0.0312 | 0.1907 | 0.0075 | True |
| decision_tree | 0.6904 | 0.0173 | 0.1729 | 0.0073 | True |

## Locked final test and constant-probability baselines

| probability_source | rows | events | mean_pd | roc_auc | pr_auc | brier | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 12282 | 119 | 0.1119 | 0.8327 | 0.0540 | 0.0390 | 0.0207 | 0.9244 |
| sigmoid | 12282 | 119 | 0.0093 | 0.8327 | 0.0540 | 0.0094 | 0.1045 | 0.1176 |
| selected | 12282 | 119 | 0.0093 | 0.8327 | 0.0540 | 0.0094 | 0.1045 | 0.1176 |
| training_prevalence_baseline | 12282 | 119 | 0.0064 | 0.5000 | 0.0097 | 0.0096 | 0.0000 | 0.0000 |
| calibration_prevalence_baseline | 12282 | 119 | 0.0084 | 0.5000 | 0.0097 | 0.0096 | 0.0000 | 0.0000 |

## Performance by test fiscal year

| fiscal_year | rows | events | event_rate | mean_pd | roc_auc | pr_auc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2015 | 3354 | 33 | 0.0098 | 0.0093 | 0.8551 | 0.0423 | 0.0096 |
| 2016 | 3191 | 29 | 0.0091 | 0.0097 | 0.7960 | 0.0482 | 0.0089 |
| 2017 | 3014 | 21 | 0.0070 | 0.0091 | 0.8337 | 0.0279 | 0.0069 |
| 2018 | 2723 | 36 | 0.0132 | 0.0090 | 0.8419 | 0.1272 | 0.0126 |

## Transfer to unseen companies

| cohort | companies | rows | events | roc_auc | pr_auc | brier |
| --- | --- | --- | --- | --- | --- | --- |
| naturally_new_entities | 577 | 1634 | 7 | 0.8254 | 0.0316 | 0.0043 |
| reserved_unseen_entities | 728 | 2419 | 23 | 0.8105 | 0.0597 | 0.0093 |
| seen_in_development | 2395 | 8229 | 89 | 0.8402 | 0.0615 | 0.0105 |

## 95% confidence intervals from entity-cluster bootstrap

| cohort | metric | estimate | lower_95 | upper_95 | valid_replicates |
| --- | --- | --- | --- | --- | --- |
| all | roc_auc | 0.8327 | 0.8058 | 0.8571 | 500 |
| all | pr_auc | 0.0540 | 0.0353 | 0.0831 | 500 |
| all | brier | 0.0094 | 0.0078 | 0.0112 | 500 |
| naturally_new_entities | roc_auc | 0.8254 | 0.6400 | 0.9246 | 499 |
| naturally_new_entities | pr_auc | 0.0316 | 0.0054 | 0.1780 | 499 |
| naturally_new_entities | brier | 0.0043 | 0.0017 | 0.0077 | 499 |
| reserved_unseen_entities | roc_auc | 0.8105 | 0.7464 | 0.8698 | 500 |
| reserved_unseen_entities | pr_auc | 0.0597 | 0.0251 | 0.1601 | 500 |
| reserved_unseen_entities | brier | 0.0093 | 0.0058 | 0.0132 | 500 |
| seen_in_development | roc_auc | 0.8402 | 0.8122 | 0.8691 | 500 |
| seen_in_development | pr_auc | 0.0615 | 0.0376 | 0.1070 | 500 |
| seen_in_development | brier | 0.0105 | 0.0085 | 0.0128 | 500 |

## Earlier walk-forward diagnostics

| test_year | train_end | calibration_year | validation_year | selected_model | rows | events | roc_auc | pr_auc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2009 | 2003 | 2005 | 2007 | random_forest | 2946 | 18 | 0.6935 | 0.0406 | 0.0061 |
| 2011 | 2005 | 2007 | 2009 | random_forest | 2788 | 20 | 0.8426 | 0.0860 | 0.0071 |
| 2013 | 2007 | 2009 | 2011 | gradient_boosting | 2789 | 21 | 0.7516 | 0.0613 | 0.0074 |

## What the evidence supports

Reserved unseen entities are excluded from every development stage. Repeated annual rows are resampled together for uncertainty estimates. Intervals condition on the trained model and historical data; they omit model-selection, event-reconstruction, and future-regime uncertainty.

Earlier walk-forward results are development diagnostics. Final test years are not used to choose models, calibration, feature preprocessing, or the classification threshold. Lower final metrics must be retained rather than used to retune against this test period.

## Data and timing limitations

- Fiscal years and reconstructed event years only. Filing availability, exact event dates, and historical restatement vintages are unavailable.
- Negative outcomes rely on author alive/event status; independent survival or delisting follow-up is unavailable.
- Label reconstruction matches published annual counts but individual filing dates are not independently verified.
- Fiscal-year chronology is not a verified as-of-filing backtest.
- Transfer to other countries, private firms, or contemporary conditions is untested.
- Internal one-year grades are project assumptions.

## Reproduction

Run `python scripts/prepare_temporal_dataset.py`, followed by `python scripts/run_temporal_pipeline.py`. Split configuration is in `config/temporal_validation.json`.

The original five-year Polish experiment remains a separate baseline. Its ROC/PR numbers cannot be used as a like-for-like comparison to this panel because the population and outcome horizon differ.
