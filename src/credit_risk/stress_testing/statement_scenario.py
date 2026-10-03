"""Explicit partial income-statement sensitivity for the annual panel."""
import numpy as np


def apply_margin_compression(history, fiscal_year, percentage_points):
    if not np.isfinite(percentage_points) or not 0 <= percentage_points <= 20:
        raise ValueError('Margin compression must be between 0 and 20 percentage points.')
    result = history.copy()
    mask = result.fiscal_year.eq(fiscal_year)
    if mask.sum() != 1:
        raise ValueError('Select exactly one company-year.')
    revenue = float(result.loc[mask, 'revenue'].iloc[0])
    if not np.isfinite(revenue) or revenue <= 0:
        raise ValueError('Positive observed revenue is required for this scenario.')
    extra_cost = revenue * percentage_points / 100
    result.loc[mask, 'operating_expenses'] += extra_cost
    for column in ['ebitda', 'ebit', 'net_income']:
        result.loc[mask, column] -= extra_cost
    return result, extra_cost
