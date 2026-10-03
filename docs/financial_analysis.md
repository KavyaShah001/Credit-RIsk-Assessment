# Phase 3 — Financial analysis

## What the source supports

The selected file supplies financial ratios, not raw balance sheet, income statement, or cash-flow statement amounts. The analysis therefore:

1. Presents selected UCI ratios with their documented definitions.
2. Derives only two identities from source ratios: liabilities/equity = (liabilities/assets) ÷ (equity/assets), and ROE = (net profit/assets) ÷ (equity/assets).
3. Preserves an absent input as missing and treats a zero denominator as undefined.
4. Lists requested measures that cannot be calculated from this source as unavailable.

The two derived identities do not add new source information. They make the corresponding financial relationship easier to inspect. They are unavailable if an input is missing or the equity/assets denominator is zero.

## Supported indicators

| Category | Indicator | Source/formula | How to interpret it |
|---|---|---|---|
| Liquidity | Current ratio | `ratio_04`; current assets / short-term liabilities | Short-term asset coverage; inventory quality, receivable collection, seasonality, and operating cycle matter. |
| Liquidity | Quick ratio, source definition | `ratio_46`; (current assets - inventory) / short-term liabilities | A narrower asset-coverage view; the source's exact convention is shown because definitions can differ. |
| Liquidity | Working capital / assets | `ratio_03` | Relative working-capital position; needs context about business scale and cycle. |
| Leverage | Liabilities / assets | `ratio_02` | Share of assets financed by liabilities. This is not interest-bearing debt / assets. |
| Leverage | Liabilities / equity | `ratio_02 / ratio_10` | Derived from liabilities/assets and equity/assets; undefined if equity/assets is zero. |
| Leverage | Long-term liabilities / equity | `ratio_59` | Long-term liability burden relative to equity; maturity dates are not included. |
| Profitability | Return on assets | `ratio_01`; net profit / assets | Net profitability relative to assets; accounting choices and sector affect comparisons. |
| Profitability | EBIT / assets | `ratio_07` | Operating earnings relative to the asset base; not a sales margin. |
| Profitability | Operating margin | `ratio_42`; operating profit / sales | Operating profit relative to sales; mix and classification affect comparability. |
| Profitability | Net profit margin | `ratio_23`; net profit / sales | Net profit relative to sales; non-operating items can affect it. |
| Profitability | Return on equity | `ratio_01 / ratio_10` | Derived from net profit/assets and equity/assets; leverage can amplify it, and near-zero or negative equity limits interpretation. |
| Coverage | Operating profit / financial expenses | `ratio_27` | Source coverage indicator; financial expenses are not necessarily interest expense alone. |
| Cash-flow proxy | (Net profit + depreciation) / liabilities | `ratio_26` | Earnings-plus-depreciation proxy, not observed operating cash flow / debt. |
| Cash-flow proxy | (Gross profit + depreciation) / liabilities | `ratio_16` | Source proxy, not free cash flow or debt-service coverage. |

Ratio values are indicators, not universal decision thresholds. A low value can have different meanings across industries, accounting policies, business models, and reporting periods. No single ratio establishes that an observation will become bankrupt.

## Requested but unavailable from this source

- Cash ratio: raw cash and current-liability amounts are not available.
- Interest-bearing debt / EBITDA: the source does not separately identify interest-bearing debt and a comparable EBITDA amount.
- Operating cash flow / debt and free cash flow: raw cash flows and capital expenditure are unavailable.
- Debt service metrics: scheduled principal and interest obligations and maturity dates are unavailable.
- Full statement-level current assets, liabilities, debt, cash, revenue, EBITDA, EBIT, net income, and equity amounts: only selected precomputed ratios are supplied.

The code in `src/credit_risk/financial_analysis/ratio_analysis.py` is the source of truth for the displayed supported and unavailable metric records.
