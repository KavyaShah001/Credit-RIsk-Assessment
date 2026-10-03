"""Calendar ordering, outcome-maturity embargoes, and deterministic entity holdout."""
from __future__ import annotations
import hashlib
import pandas as pd


def reserved_entities(ids: pd.Series, fraction: float, seed: int) -> pd.Series:
    if not 0 <= fraction < 1:
        raise ValueError('Entity holdout fraction must be in [0, 1).')
    mapping = {key: int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16) / 2**64 < fraction
               for key in ids.unique()}
    return ids.map(mapping).astype(bool)


def make_temporal_split(panel: pd.DataFrame, config: dict) -> tuple[dict[str, pd.Index], pd.Series, list[dict]]:
    ordered = [config['train_end'], config['calibration_start'], config['calibration_end'],
               config['validation_start'], config['validation_end'], config['test_start'], config['test_end']]
    if not (ordered[0] + 1 < ordered[1] <= ordered[2] and ordered[2] + 1 < ordered[3] <= ordered[4]
            and ordered[4] + 1 < ordered[5] <= ordered[6]):
        raise ValueError('Periods must be chronological with an outcome-maturity year between stages.')
    held = reserved_entities(panel.company_id, config['unseen_entity_fraction'], config['random_seed'])
    year = panel.fiscal_year
    masks = {
        'train': year.le(config['train_end']) & ~held,
        'calibration': year.between(config['calibration_start'], config['calibration_end']) & ~held,
        'validation': year.between(config['validation_start'], config['validation_end']) & ~held,
        'test': year.between(config['test_start'], config['test_end']),
    }
    indices = {name: panel.index[mask] for name, mask in masks.items()}
    summary = []
    for name, index in indices.items():
        part = panel.loc[index]
        if part.empty or part.bankruptcy_next_year.nunique() < 2:
            raise ValueError(f'{name} must contain both observed outcome classes.')
        summary.append({'split': name, 'first_year': int(part.fiscal_year.min()),
                        'last_year': int(part.fiscal_year.max()), 'rows': len(part),
                        'companies': int(part.company_id.nunique()),
                        'events': int(part.bankruptcy_next_year.sum()),
                        'event_rate': float(part.bankruptcy_next_year.mean()),
                        'last_label_year': int(part.label_observed_year.max())})
    for earlier, later in zip(list(indices)[:-1], list(indices)[1:]):
        if panel.loc[indices[earlier], 'label_observed_year'].max() >= panel.loc[indices[later], 'fiscal_year'].min():
            raise ValueError('An outcome window crosses a later evaluation boundary.')
    fitted_ids = set(panel.loc[indices['train'].union(indices['calibration']).union(indices['validation']), 'company_id'])
    if fitted_ids.intersection(panel.loc[held, 'company_id']):
        raise AssertionError('Reserved entities leaked into development.')
    return indices, held, summary
