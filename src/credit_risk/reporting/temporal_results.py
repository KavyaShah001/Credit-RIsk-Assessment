"""Render the temporal validation evidence from run artifacts, never example metrics."""
import json
from pathlib import Path
import pandas as pd


def _table(frame):
    def value(item):
        if pd.isna(item):
            return 'unavailable'
        return f'{item:.4f}' if isinstance(item, float) else str(item)
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(value(item) for item in row) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def write_temporal_results(root):
    root = Path(root)
    reports = root / 'reports/temporal'
    card = json.loads((reports / 'model_card.json').read_text())
    parts = [
        '# Chronological validation results',
        f"Generated from run `{card['run_id']}`. Selected model: **{card['selected_model']}**; probability source: **{card['probability_source']}**.",
        'The outcome is a reconstructed next-fiscal-year bankruptcy proxy. The source has fiscal years, '
        'but no exact filing availability or event dates. These results measure calendar-year transfer under that limitation.',
    ]
    for heading, file, columns in [
        ('Chronological periods', 'split_summary.csv', None),
        ('Candidate selection on validation only', 'validation_comparison.csv',
         ['model', 'roc_auc', 'pr_auc', 'brier', 'calibrated_brier', 'use_calibrated']),
        ('Locked final test and constant-probability baselines', 'test_metrics.csv',
         ['probability_source', 'rows', 'events', 'mean_pd', 'roc_auc', 'pr_auc', 'brier', 'precision', 'recall']),
        ('Performance by test fiscal year', 'test_by_year.csv',
         ['fiscal_year', 'rows', 'events', 'event_rate', 'mean_pd', 'roc_auc', 'pr_auc', 'brier']),
        ('Transfer to unseen companies', 'test_by_cohort.csv',
         ['cohort', 'companies', 'rows', 'events', 'roc_auc', 'pr_auc', 'brier']),
        ('95% confidence intervals from entity-cluster bootstrap', 'bootstrap_intervals.csv', None),
        ('Earlier walk-forward diagnostics', 'walk_forward.csv',
         ['test_year', 'train_end', 'calibration_year', 'validation_year', 'selected_model', 'rows', 'events', 'roc_auc', 'pr_auc', 'brier']),
    ]:
        frame = pd.read_csv(reports / file)
        if columns:
            frame = frame[columns]
        parts += ['## ' + heading, _table(frame)]
    parts += ['## What the evidence supports',
              'Reserved unseen entities are excluded from every development stage. Repeated annual rows are resampled '
              'together for uncertainty estimates. Intervals condition on the trained model and historical data; '
              'they omit model-selection, event-reconstruction, and future-regime uncertainty.',
              'Earlier walk-forward results are development diagnostics. Final test years are not used to choose '
              'models, calibration, feature preprocessing, or the classification threshold. Lower final metrics '
              'must be retained rather than used to retune against this test period.',
              '## Data and timing limitations',
              '\n'.join('- ' + item for item in card['limitations']),
              '## Reproduction',
              'Run `python scripts/prepare_temporal_dataset.py`, followed by '
              '`python scripts/run_temporal_pipeline.py`. Split configuration is in `config/temporal_validation.json`.',
              'The original five-year Polish experiment remains a separate baseline. Its ROC/PR numbers cannot '
              'be used as a like-for-like comparison to this panel because the population and outcome horizon differ.']
    content = '\n\n'.join(parts) + '\n'
    (root / 'docs/temporal_results.md').write_text(content)
    (root / 'outputs').mkdir(exist_ok=True)
    (root / 'outputs/temporal_validation_summary.md').write_text(content)
