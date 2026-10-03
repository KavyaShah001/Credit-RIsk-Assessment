"""Dimensionless statement ratios and strictly backward annual features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RATIO_FORMULAS = {
    'current_ratio': 'Current assets / current liabilities',
    'quick_ratio': '(Current assets - inventory) / current liabilities',
    'working_capital_to_assets': '(Current assets - current liabilities) / total assets',
    'liabilities_to_assets': 'Total liabilities / total assets',
    'long_term_debt_to_assets': 'Long-term debt / total assets',
    'liabilities_to_equity': 'Total liabilities / (assets - liabilities)',
    'long_term_debt_to_equity': 'Long-term debt / (assets - liabilities)',
    'long_term_debt_to_ebitda': 'Long-term debt / EBITDA; not total debt / EBITDA',
    'return_on_assets': 'Net income / total assets',
    'return_on_equity': 'Net income / (assets - liabilities)',
    'ebit_margin': 'EBIT / revenue',
    'ebitda_margin': 'EBITDA / revenue',
    'net_margin': 'Net income / revenue',
    'gross_margin': 'Gross profit / revenue',
    'retained_earnings_to_assets': 'Retained earnings / total assets',
    'earnings_plus_da_to_liabilities': '(Net income + depreciation/amortization) / liabilities; earnings proxy, not cash flow',
    'receivables_to_sales': 'Receivables / net sales',
}
TREND_RATIOS = ['current_ratio', 'liabilities_to_assets', 'ebitda_margin', 'return_on_assets']
FEATURE_DESCRIPTIONS = dict(RATIO_FORMULAS)
FEATURE_DESCRIPTIONS.update({
    'nonpositive_equity': '1 when assets minus liabilities is zero or negative',
    'nonpositive_assets': '1 when reported assets are zero or negative',
    'nonpositive_ebitda': '1 when EBITDA is zero or negative',
    'annual_history_missing': '1 when no observation exists for exactly the previous fiscal year',
    'revenue_growth': 'Revenue / preceding annual revenue - 1, positive prior denominator required',
    'assets_growth': 'Assets / preceding annual assets - 1, positive prior denominator required',
})
for _feature in TREND_RATIOS:
    FEATURE_DESCRIPTIONS[_feature + '_lag1'] = f'Previous fiscal-year {_feature}; no filling across gaps'
    FEATURE_DESCRIPTIONS[_feature + '_change1'] = f'Current minus previous fiscal-year {_feature}'
FEATURE_COLUMNS = list(FEATURE_DESCRIPTIONS)


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    # Nonpositive denominators have ambiguous financial interpretations and remain missing.
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = numerator / denominator.where(denominator.gt(0))
    return result.replace([np.inf, -np.inf], np.nan)


def build_panel_features(frame: pd.DataFrame) -> pd.DataFrame:
    """No labels, entity encodings, fiscal years, or terminal-history features enter X."""
    if frame.duplicated(['company_id', 'fiscal_year']).any():
        raise ValueError('Duplicate company-year keys cannot define annual lags.')
    f = frame.sort_values(['company_id', 'fiscal_year'])
    equity = f.total_assets - f.total_liabilities
    out = pd.DataFrame(index=f.index)
    operands = [
        (f.current_assets, f.current_liabilities),
        (f.current_assets - f.inventory, f.current_liabilities),
        (f.current_assets - f.current_liabilities, f.total_assets),
        (f.total_liabilities, f.total_assets), (f.long_term_debt, f.total_assets),
        (f.total_liabilities, equity), (f.long_term_debt, equity), (f.long_term_debt, f.ebitda),
        (f.net_income, f.total_assets), (f.net_income, equity), (f.ebit, f.revenue),
        (f.ebitda, f.revenue), (f.net_income, f.revenue), (f.gross_profit, f.revenue),
        (f.retained_earnings, f.total_assets),
        (f.net_income + f.depreciation_amortization, f.total_liabilities),
        (f.receivables, f.net_sales),
    ]
    for name, (num, den) in zip(RATIO_FORMULAS, operands):
        out[name] = _divide(num, den)
    out['nonpositive_equity'] = equity.le(0).astype(float).where(equity.notna())
    out['nonpositive_assets'] = f.total_assets.le(0).astype(float).where(f.total_assets.notna())
    out['nonpositive_ebitda'] = f.ebitda.le(0).astype(float).where(f.ebitda.notna())
    groups = f.groupby('company_id', sort=False)
    consecutive = f.fiscal_year.sub(groups.fiscal_year.shift()).eq(1)
    out['annual_history_missing'] = (~consecutive).astype(float)
    out['revenue_growth'] = (_divide(f.revenue, groups.revenue.shift()) - 1).where(consecutive)
    out['assets_growth'] = (_divide(f.total_assets, groups.total_assets.shift()) - 1).where(consecutive)
    for feature in TREND_RATIOS:
        lag = out[feature].groupby(f.company_id, sort=False).shift().where(consecutive)
        out[feature + '_lag1'] = lag
        out[feature + '_change1'] = out[feature] - lag
    return out[FEATURE_COLUMNS].reindex(frame.index)


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Fit tail limits on training only; preserve missingness for the imputer."""
    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.lower_bounds_ = frame.quantile(self.lower).fillna(-np.inf).to_numpy()
        self.upper_bounds_ = frame.quantile(self.upper).fillna(np.inf).to_numpy()
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError('Feature count changed after preprocessing was fitted.')
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)
