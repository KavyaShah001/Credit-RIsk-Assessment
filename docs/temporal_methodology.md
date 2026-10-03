# Annual financial panel: chronology and generalization

This extension measures transfer across fiscal periods and companies. It does not establish universal validity for new borrowers. The original Polish five-year experiment remains available as a separate baseline.

## Source, version and outcome audit

Source: the authors' [annual financial panel](https://github.com/sowide/bankruptcy_dataset), pinned to the original release at commit `7f56a80b73c2f32582fcc01d28acb6adc665ae40`. Attribution: Lombardo et al. (2022), [doi:10.3390/fi14080244](https://doi.org/10.3390/fi14080244), CC BY 4.0. Source variables follow the paper's Table 2.

The downloaded file contains 78,682 rows, 8,971 anonymous entities, and fiscal years 1999–2018. These counts are audited from the file. The paper's prose gives a different entity count; we retain the file's observed count. A checksum locks the exact release, since the later repository release changes column order, field names and numeric representations. No silent column guessing or rescaling is performed.

A critical source issue: `status_label` is constant within each company. Its 5,220 `failed` rows describe 609 companies. Using all those rows as annual events would leak eventual status into earlier years. The adapter assigns a positive next-year target only to each failed entity's last fiscal-year row. That reconstruction reproduces **all 20 annual bankruptcy counts in Table 1**. The adapter refuses training if these counts change. Terminal year and terminal status never enter the feature matrix.

This reconstruction supports an annual bankruptcy **proxy**. There are no individual filing dates, publication dates, statement revision vintages, or independently verified survival/delisting histories. Negative outcomes rely on the authors' source status and event convention. Therefore this is a fiscal-year backtest under a documented timing assumption, rather than a verified as-of-filing backtest. Cash, interest expense, total interest-bearing debt, operating cash flow, capital expenditure and sector are absent from the pinned release.

## Canonical CSV interface

An additional dataset can be mapped to the statement fields in `credit_risk.ingestion.panel_loader.FIELD_MAP`. Inference requires `company_id`, integer `fiscal_year`, and the 18 named statement columns. Blank monetary fields are preserved as missing. The validator rejects invalid numeric values, infinite amounts, missing identifiers, duplicate company-years and entirely empty statement rows. Outcomes are not required for scoring.

All monetary values within a row must have a consistent currency and scale. Model inputs are dimensionless ratios, flags and annual ratio/growth changes. Raw monetary levels, entity IDs, fiscal year, terminal status, terminal year and market value are excluded. This makes the interface reusable; different accounting definitions or borrower populations still require external validation.

| Source | Canonical field | Use |
| --- | --- | --- |
| X1 | current_assets | Liquidity |
| X2 | cost_of_goods_sold | Statement display |
| X3 | depreciation_amortization | Earnings proxy |
| X4 | ebitda | Earnings/margin |
| X5 | inventory | Quick ratio |
| X6 | net_income | Profitability |
| X7 | receivables | Receivables/sales |
| X8 | market_value | Display only; valuation timestamp unavailable |
| X9 | net_sales | Sales denominator |
| X10 | total_assets | Assets denominator |
| X11 | long_term_debt | Long-term debt ratios; never labeled total debt |
| X12 | ebit | EBIT margin |
| X13 | gross_profit | Gross margin |
| X14 | current_liabilities | Liquidity denominator |
| X15 | retained_earnings | Retained earnings/assets |
| X16 | revenue | Margins and growth |
| X17 | total_liabilities | Liability ratios and derived book-equity proxy |
| X18 | operating_expenses | Statement display and scenario |

## Financial features and history

The machine-readable dictionary is `FEATURE_DESCRIPTIONS` in `src/credit_risk/features/panel_features.py`. Every feature is documented there. Seventeen base ratios include current and quick ratios, working capital/assets, liabilities/assets, long-term debt/assets, profitability margins, returns, retained earnings/assets, and an earnings-plus-depreciation proxy. Equity is derived as assets minus liabilities. It is not a separately observed source amount.

Nonpositive denominators are missing for these financial ratios; separate indicators preserve negative/zero equity, assets and EBITDA. This prevents negative equity from appearing as a reassuring low leverage ratio. These handling choices are project conventions, not universal accounting rules.

Four indicators also receive a one-year lag and an annual difference: current ratio, liabilities/assets, EBITDA margin and return on assets. Revenue growth and asset growth require a positive preceding-year denominator. Lags use the same entity and exactly the previous fiscal year. Missing years are not filled; future rows cannot alter prior features. Computing historical lags for an unseen entity uses that entity's available financial history without using its labels or fitting preprocessing on it.

Each candidate pipeline fits 0.5%/99.5% clipping bounds, median imputation, missing-value indicators and standardization on training rows only. Market prices are excluded because the observation timestamp is unknown. Feature distribution shift is measured against bins fitted on training, with missingness as its own bin.

## Chronological development and entity holdout

The committed configuration is `config/temporal_validation.json`:

| Stage | Fiscal years | Purpose |
| --- | --- | --- |
| Training | 1999–2007 | Fit model and preprocessing |
| Calibration | 2009–2010 | Fit unweighted sigmoid calibration |
| Validation | 2012–2013 | Select family, raw/calibrated output, and F1 threshold |
| Final test | 2015–2018 | Report once after choices are fixed |

Years 2008, 2011 and 2014 are excluded from adjacent fitting/evaluation stages, allowing preceding next-year outcomes to mature. They may supply historical feature values for later rows, which is legitimate for a backward feature. Maturity is checked using whole years, not invented filing dates.

A stable SHA-256 allocation reserves approximately 20% of entity IDs before fitting. Reserved entities are excluded from training, calibration, validation and all earlier development diagnostics. The final period includes them and reports their performance separately. Other test entities are split into those observed in development and naturally new entities. These cohorts answer different questions and are not pooled to claim new-company performance.

Earlier walk-forward diagnostics evaluate 2009, 2011 and 2013. Each fold has its own older training, later calibration and later validation periods, with a full-year gap between stages. These folds demonstrate sensitivity to different historical periods. The final 2015–2018 sample is excluded from them.

## Models, calibration and selection

Candidates: balanced logistic regression, a shallow decision tree, a random forest and histogram gradient boosting. Fixed, modest configurations avoid a large search against a small event sample. Gradient boosting's internal random early-stopping split is disabled. Natural prevalence is retained outside weighted model fitting.

Families are compared by validation average precision (reported as PR-AUC). For the selected family, an unweighted logistic sigmoid fitted on a separate later calibration period is retained only if it improves validation Brier score and preserves score ordering. This avoids stratified random calibration mixing future periods into earlier predictions. The threshold is chosen by validation F1 and is an illustrative operating point, not a lending policy or economic optimum.

Final reports include ROC-AUC, PR-AUC, Brier score, log loss, observed event rate, mean PD, precision, recall, F1 and confusion counts. Constant probabilities estimated from training and calibration prevalence provide meaningful benchmarks. Calibration tables include observation/event counts; a low overall Brier score alone is not proof of useful rare-event detection.

The original Polish test scores cannot be directly compared with this experiment: both the borrower population and outcome horizon changed.

## Uncertainty, drift and explanations

Five hundred bootstrap replicates resample whole entity histories. Thus several annual rows for the same company are not treated as independent observations. The generated 95% intervals are conditional on the fitted model and this historical sample. They omit model-selection uncertainty, reconstructed-label error and future macroeconomic regimes. Single-class replicates are skipped and the valid replicate count is disclosed.

PSI measures feature shift against training-defined bins. It is a diagnostic, not a hypothesis test or guarantee of validity. Inference also reports missing features and counts outside training percentile bounds. Global importance is measured on validation rows; local explanations use one-feature substitutions by training medians and do not imply causality.

## Dashboard, SQL and scenarios

The dashboard selector keeps the annual and five-year experiments distinct. Annual portfolio views use one fiscal year at a time, with one record per entity. SQLite stores entity keys, annual statements, derived features, model metadata, internal grades and predictions in separate tables. Queries join statements to predictions using observation keys.

Annual internal grades T1–T5 use these project cutoffs: <0.25%, 0.25–<1%, 1–<3%, 3–<10%, and ≥10%. They are deliberately distinct from the old five-year I1–I5 bands and carry no external-rating interpretation.

The annual scenario adds operating expense equal to revenue times a margin-compression assumption. EBITDA, EBIT and net income fall by that amount with no tax relief. Ratios and historical changes are rebuilt before model scoring. This is a partial income-statement sensitivity: it does not solve cash, financing or balance-sheet feedback. Both company and scenario HTML reports include the timing and outcome limitations.

## What is still needed for deployment

An independently curated source with actual default/event dates, verified filing publication dates, complete survival follow-up, historical statement vintages and relevant current borrowers would allow a stronger point-in-time evaluation. Exposure and LGD data would enable portfolio loss analysis. These are future data requirements, not missing values to invent in the current project.
