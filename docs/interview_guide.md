# Explaining the project in an interview

## A short introduction

“I built a financial analytics pipeline that converts public annual statements into interpretable financial ratios and historical changes, estimates an annual bankruptcy probability proxy, and evaluates transfer across time and companies. The project includes SQL storage, calibration, uncertainty estimates, a portfolio dashboard and automated reports. A major part of the work was auditing labels and information timing before fitting models.”

Use the actual latest model and results in `docs/temporal_results.md`; do not memorize a result that may change after a documented pipeline revision.

## Why did you extend the first version?

The first source contained financial ratios and a five-year outcome but no dates or persistent entity IDs. A stratified holdout demonstrated the pipeline, but could not test future-period transfer. The extension uses fiscal years and repeated company identifiers. The original version remains an explicitly separate baseline.

## What was the most important data issue?

The annual source's company failure status repeats across historical records. Treating those records as separate next-year failures would predict eventual company status and exaggerate the annual event rate. I reconstructed a positive outcome only for the final financial year of each failed entity. The resulting annual event counts match all twenty years in the publication's event table.

This remains a reconstruction. I cannot independently verify filing timestamps, exact bankruptcy dates, historical restatements or negative follow-up. I therefore describe the output as an annual bankruptcy proxy, and the evaluation as fiscal-year validation.

## What does the chronological split protect against?

Training fits models and all learned preprocessing on early years. A later calibration period maps raw scores into probabilities. Validation then selects the model, decides whether calibration helps and fixes the threshold. The final 2015–2018 observations are used for reporting only.

A full-year gap between stages allows the preceding annual label window to mature. Feature construction uses only current statements and the same entity's preceding fiscal-year statements. It does not create calendar dates from row order.

## Why use a company holdout as well?

Performance on a future observation from a familiar company is different from performance on an unfamiliar company. A deterministic allocation reserves roughly twenty percent of identifiers from training, calibration, selection and development diagnostics. Final metrics report reserved companies separately. The model never uses the identifiers themselves as predictors.

## How do the financial decisions affect features?

- Current and quick ratios describe liquidity; they do not establish default by themselves.
- Liabilities/assets and long-term debt ratios describe different obligations. I do not relabel long-term debt as total debt.
- Assets minus liabilities provides a book-equity proxy. Negative equity gets its own indicator; equity-denominated ratios are missing when the denominator is nonpositive.
- Earnings plus depreciation is an earnings proxy, not observed cash flow.
- Annual differences and lags track changes without substituting future statements or filling missing years.

All monetary amounts within an observation must use a consistent scale. The model uses dimensionless ratios, indicators and growth changes, which helps make the input interface reusable. Accounting definitions still matter.

## Why does probability calibration matter?

Class weighting helps a model learn rare events, but raw probabilities can differ substantially from the actual event frequency. The project fits an unweighted sigmoid on a separate chronological calibration period. Validation Brier score decides whether to use it. Final evaluation compares both raw and calibrated probabilities with constant-prevalence baselines.

Calibration matters when a predicted probability is interpreted as a risk level or used in expected loss. A low Brier score alone can be misleading when almost all outcomes are zero; event capture and baseline improvement also matter.

## Why are the final results less impressive than the original demo?

The populations, event horizons and labels differ, so their metrics are not directly comparable. The chronological experiment is harder and reveals limitations that a small random holdout could miss. Report PR-AUC, event prevalence, threshold recall and false negatives alongside ROC-AUC. The actual final run has limited event detection at its validation-selected threshold. That is a finding to disclose, not a reason to optimize against the test set.

## Why bootstrap companies?

Several annual observations can belong to one company. Resampling individual rows would ignore that dependence. The project instead resamples entire company histories. These intervals still omit label error, model-selection variability and future economic-regime changes.

## Can this model score any company now?

It can accept additional datasets mapped to the canonical financial schema. That is software reuse, not proof of statistical transfer. Different geographies, industries, private-company accounting or current conditions require independent validation and potentially redevelopment. Missing-feature and training-range diagnostics help identify input issues; they do not guarantee validity.

## What would you improve next?

Acquire independently verified default dates and actual filing availability, retain historical statement vintages, confirm survival and delisting follow-up, and evaluate an external population. Then add real exposure and loss information for expected loss and concentration analysis. A production decision policy also requires economic costs, monitoring and governance beyond this educational project.
