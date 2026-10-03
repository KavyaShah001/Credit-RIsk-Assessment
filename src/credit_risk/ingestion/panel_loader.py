"""Versioned adapter and audit for the authors' original annual financial panel."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_COMMIT = '7f56a80b73c2f32582fcc01d28acb6adc665ae40'
SOURCE_SHA256 = '1aff805af8d2f4368ffcd876814a4589e09905a9cbac289a7b780ed0f7a2478a'
SOURCE_URL = f'https://raw.githubusercontent.com/sowide/bankruptcy_dataset/{SOURCE_COMMIT}/american_bankruptcy_dataset.csv'
FIELD_MAP = dict(zip([f'X{i}' for i in range(1, 19)], [
    'current_assets', 'cost_of_goods_sold', 'depreciation_amortization', 'ebitda',
    'inventory', 'net_income', 'receivables', 'market_value', 'net_sales', 'total_assets',
    'long_term_debt', 'ebit', 'gross_profit', 'current_liabilities', 'retained_earnings',
    'revenue', 'total_liabilities', 'operating_expenses',
]))
# Table 1 of doi:10.3390/fi14080244: outcome year -> bankruptcy events.
PUBLISHED_EVENTS = dict(zip(range(2000, 2020), [3, 7, 10, 17, 29, 46, 40, 51, 59, 58,
                                              23, 35, 25, 26, 28, 33, 33, 29, 21, 36]))


def validate_panel(frame: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Canonical statement interface; entity/year are keys, never model inputs.

    No filling of dates, statement amounts, or labels. Missing amounts are allowed,
    invalid strings/infinity are rejected, and duplicate entity-years are errors.
    """
    required = ['company_id', 'fiscal_year', *FIELD_MAP.values()]
    if require_target:
        required += ['bankruptcy_next_year', 'label_observed_year']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing canonical columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('No financial observations supplied.')
    result = frame.copy()
    if result.company_id.isna().any() or result.company_id.astype(str).str.strip().eq('').any():
        raise ValueError('company_id must be a nonempty persistent identifier.')
    result['company_id'] = result.company_id.astype(str).str.strip()
    year = pd.to_numeric(result.fiscal_year, errors='raise')
    if year.isna().any() or not np.isfinite(year).all() or not year.eq(np.floor(year)).all():
        raise ValueError('fiscal_year must contain real integer fiscal years.')
    if not year.between(1900, 2200).all():
        raise ValueError('fiscal_year is outside the supported range.')
    result['fiscal_year'] = year.astype(int)
    if result.duplicated(['company_id', 'fiscal_year']).any():
        raise ValueError('Duplicate company_id / fiscal_year observations.')
    for column in FIELD_MAP.values():
        result[column] = pd.to_numeric(result[column], errors='raise')
        if np.isinf(result[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f'Infinite statement amount in {column}.')
    if result[list(FIELD_MAP.values())].isna().all(axis=1).any():
        raise ValueError('At least one observed statement amount is required per row.')
    if require_target:
        if not result.bankruptcy_next_year.isin([0, 1]).all():
            raise ValueError('bankruptcy_next_year must be binary with no missing labels.')
        maturity = pd.to_numeric(result.label_observed_year, errors='raise')
        if not maturity.eq(result.fiscal_year + 1).all():
            raise ValueError('Label observation year must equal fiscal_year + 1 for this horizon.')
    return result.sort_values(['company_id', 'fiscal_year']).reset_index(drop=True)


def load_author_panel(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Source checksum mismatch. Use the pinned 2022 release; later releases change the schema and values.')
    source = pd.read_csv(path)
    required = {'company_name', 'year', 'status_label', *FIELD_MAP}
    if set(source.columns) != required:
        raise ValueError('Unexpected original-source schema.')
    if not source.status_label.isin(['alive', 'failed']).all():
        raise ValueError('Unexpected source status label.')
    if not source.groupby('company_name').status_label.nunique().eq(1).all():
        raise ValueError('Source label semantics changed; review the adapter.')
    frame = source.rename(columns={'company_name': 'company_id', 'year': 'fiscal_year', **FIELD_MAP})
    last_year = frame.groupby('company_id').fiscal_year.transform('max')
    frame['bankruptcy_next_year'] = (frame.status_label.eq('failed') & frame.fiscal_year.eq(last_year)).astype('int8')
    frame['label_observed_year'] = frame.fiscal_year.astype(int) + 1
    events = frame.loc[frame.bankruptcy_next_year.eq(1)].groupby('label_observed_year').size().to_dict()
    if events != PUBLISHED_EVENTS:
        raise ValueError('Reconstructed annual events disagree with the source paper; training stopped.')
    frame = validate_panel(frame, require_target=True)
    audit = {
        'source_url': SOURCE_URL, 'source_commit': SOURCE_COMMIT, 'sha256': digest,
        'license': 'CC BY 4.0', 'rows': len(frame), 'companies': frame.company_id.nunique(),
        'first_fiscal_year': int(frame.fiscal_year.min()), 'last_fiscal_year': int(frame.fiscal_year.max()),
        'source_failed_rows': int(frame.status_label.eq('failed').sum()),
        'reconstructed_events': int(frame.bankruptcy_next_year.sum()),
        'published_annual_event_counts_match': True,
        'event_counts_by_outcome_year': {str(k): int(v) for k, v in events.items()},
        'missing_statement_cells': int(frame[list(FIELD_MAP.values())].isna().sum().sum()),
        'label_policy': 'Only final fiscal-year row of each failed company is positive; all 20 annual counts match paper Table 1.',
        'timing_limitation': 'Fiscal years and reconstructed event years only. Filing availability, exact event dates, and historical restatement vintages are unavailable.',
        'negative_followup_limitation': 'Negative outcomes rely on author alive/event status; independent survival or delisting follow-up is unavailable.',
        'scope': 'Historical listed-company panel; no claim of transfer to other populations.',
    }
    # Terminal status and last-year metadata are excluded from the canonical training frame.
    return frame.drop(columns='status_label'), audit
