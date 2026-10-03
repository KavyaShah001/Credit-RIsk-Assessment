"""Locked chronological model development, backtesting, and final diagnostics."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_risk.evaluation.robustness import probability_metrics, reliability_table, entity_bootstrap, feature_drift
from credit_risk.evaluation.temporal_split import make_temporal_split
from credit_risk.features.panel_features import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS
from credit_risk.models.temporal_model import candidates, TemporalSigmoid, score_features


def _threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros_like(precision[:-1]), where=precision[:-1] + recall[:-1] > 0)
    return float(thresholds[np.argmax(f1)])


def fit_development(panel, features, config):
    split, held, split_rows = make_temporal_split(panel, config)
    y = panel.bankruptcy_next_year
    fitted, comparisons = {}, []
    for name, estimator in candidates(config['random_seed']).items():
        print(f'  Fitting {name} on fiscal years through {config["train_end"]}…', flush=True)
        estimator.fit(features.loc[split['train']], y.loc[split['train']])
        calibrated = TemporalSigmoid(estimator).fit(features.loc[split['calibration']], y.loc[split['calibration']])
        raw_p = estimator.predict_proba(features.loc[split['validation']])[:, 1]
        calibrated_p = calibrated.predict_proba(features.loc[split['validation']])[:, 1]
        raw_metrics = probability_metrics(y.loc[split['validation']], raw_p)
        cal_metrics = probability_metrics(y.loc[split['validation']], calibrated_p)
        use_calibrated = calibrated.monotone_ and cal_metrics['brier'] < raw_metrics['brier']
        fitted[name] = (estimator, calibrated, use_calibrated)
        comparisons.append({'model': name, **raw_metrics, 'calibrated_brier': cal_metrics['brier'],
                            'calibration_monotone': calibrated.monotone_, 'use_calibrated': use_calibrated})
    comparison = pd.DataFrame(comparisons).sort_values(['pr_auc', 'model'], ascending=[False, True])
    selected = str(comparison.iloc[0]['model'])
    raw_model, cal_model, use_cal = fitted[selected]
    active = cal_model if use_cal else raw_model
    validation_p = active.predict_proba(features.loc[split['validation']])[:, 1]
    operating_threshold = _threshold(y.loc[split['validation']], validation_p)
    train_features = features.loc[split['train']]
    bundle = {
        'model': active, 'raw_model': raw_model, 'calibrated_model': cal_model,
        'selected_model': selected, 'use_calibrated_probability': use_cal,
        'probability_source': 'chronological sigmoid calibration' if use_cal else 'raw probability',
        'operating_threshold': operating_threshold,
        'feature_columns': FEATURE_COLUMNS, 'feature_descriptions': FEATURE_DESCRIPTIONS,
        'training_medians': train_features.median(),
        'training_lower': train_features.quantile(.005), 'training_upper': train_features.quantile(.995),
        'horizon_years': 1, 'target': 'reconstructed next-fiscal-year bankruptcy proxy',
    }
    return bundle, split, held, split_rows, comparison


def evaluate_temporal(panel, features, config, audit, root: Path):
    report_dir, model_dir = root / 'reports/temporal', root / 'models/temporal'
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    walk_rows = []
    for test_year in config['walk_forward_test_years']:
        if test_year >= config['test_start']:
            raise ValueError('Walk-forward development diagnostics cannot consume final test years.')
        fold_config = {**config, 'train_end': test_year - 6,
                       'calibration_start': test_year - 4, 'calibration_end': test_year - 4,
                       'validation_start': test_year - 2, 'validation_end': test_year - 2,
                       'test_start': test_year, 'test_end': test_year}
        print(f'Walk-forward development diagnostic: fiscal {test_year}', flush=True)
        fold, split, held, _, _ = fit_development(panel, features, fold_config)
        evaluation_index = split['test'][~held.loc[split['test']]]
        p = fold['model'].predict_proba(features.loc[evaluation_index])[:, 1]
        walk_rows.append({'test_year': test_year, 'train_end': test_year - 6,
                          'calibration_year': test_year - 4, 'validation_year': test_year - 2,
                          'selected_model': fold['selected_model'], 'probability_source': fold['probability_source'],
                          **probability_metrics(panel.loc[evaluation_index, 'bankruptcy_next_year'], p, fold['operating_threshold'])})

    print('Final chronological experiment: model choices use validation only.', flush=True)
    bundle, split, held, split_rows, comparison = fit_development(panel, features, config)
    test_index = split['test']
    test = panel.loc[test_index, ['company_id', 'fiscal_year', 'bankruptcy_next_year']].copy()
    test['observation_id'] = test.company_id + ':' + test.fiscal_year.astype(str)
    scored = score_features(bundle, features.loc[test_index])
    for column in scored:
        test[column] = scored[column]
    test['raw_pd'] = bundle['raw_model'].predict_proba(features.loc[test_index])[:, 1]
    test['calibrated_pd'] = bundle['calibrated_model'].predict_proba(features.loc[test_index])[:, 1]
    development_ids = set(panel.loc[split['train'].union(split['calibration']).union(split['validation']), 'company_id'])
    test['cohort'] = np.where(held.loc[test_index], 'reserved_unseen_entities',
                              np.where(test.company_id.isin(development_ids), 'seen_in_development', 'naturally_new_entities'))
    y = test.bankruptcy_next_year
    threshold = bundle['operating_threshold']
    train_rate = float(panel.loc[split['train'], 'bankruptcy_next_year'].mean())
    calibration_rate = float(panel.loc[split['calibration'], 'bankruptcy_next_year'].mean())
    evaluated = []
    for source, p in [('raw', test.raw_pd), ('sigmoid', test.calibrated_pd), ('selected', test.selected_pd),
                      ('training_prevalence_baseline', np.full(len(test), train_rate)),
                      ('calibration_prevalence_baseline', np.full(len(test), calibration_rate))]:
        evaluated.append({'probability_source': source, **probability_metrics(y, p, threshold)})
    yearly = [{'fiscal_year': int(year), **probability_metrics(group.bankruptcy_next_year, group.selected_pd, threshold)}
              for year, group in test.groupby('fiscal_year')]
    cohorts = [{'cohort': name, 'companies': group.company_id.nunique(),
                **probability_metrics(group.bankruptcy_next_year, group.selected_pd, threshold)}
               for name, group in test.groupby('cohort')]
    intervals = []
    for name, group in [('all', test), *list(test.groupby('cohort'))]:
        interval = entity_bootstrap(group, config['bootstrap_repeats'], config['random_seed'])
        interval.insert(0, 'cohort', name)
        intervals.append(interval)
    print('Computing training-reference drift and validation feature importance…', flush=True)
    drift = feature_drift(features.loc[split['train']], features.loc[test_index])
    importance = permutation_importance(bundle['raw_model'], features.loc[split['validation']],
                                        panel.loc[split['validation'], 'bankruptcy_next_year'],
                                        scoring='average_precision', n_repeats=3, random_state=config['random_seed'], n_jobs=1)
    importance_frame = pd.DataFrame({'feature': FEATURE_COLUMNS, 'description': list(FEATURE_DESCRIPTIONS.values()),
                                    'pr_auc_decrease': importance.importances_mean, 'std': importance.importances_std})
    importance_frame = importance_frame.sort_values('pr_auc_decrease', ascending=False)
    reports = {
        'split_summary.csv': pd.DataFrame(split_rows), 'validation_comparison.csv': comparison,
        'walk_forward.csv': pd.DataFrame(walk_rows), 'test_predictions.csv': test,
        'test_metrics.csv': pd.DataFrame(evaluated), 'test_by_year.csv': pd.DataFrame(yearly),
        'test_by_cohort.csv': pd.DataFrame(cohorts), 'bootstrap_intervals.csv': pd.concat(intervals, ignore_index=True),
        'feature_drift.csv': drift, 'feature_importance.csv': importance_frame,
    }
    for name, frame in reports.items():
        frame.to_csv(report_dir / name, index=False)
    for source, p in [('raw', test.raw_pd), ('sigmoid', test.calibrated_pd), ('selected', test.selected_pd)]:
        reliability_table(y, p).to_csv(report_dir / f'calibration_{source}.csv', index=False)
    fpr, tpr, _ = roc_curve(y, test.selected_pd)
    precision, recall, _ = precision_recall_curve(y, test.selected_pd)
    pd.DataFrame({'false_positive_rate': fpr, 'true_positive_rate': tpr}).to_csv(report_dir / 'roc.csv', index=False)
    pd.DataFrame({'recall': recall, 'precision': precision}).to_csv(report_dir / 'precision_recall.csv', index=False)
    policy = 'annual-panel-v1-terminal-event-label-financial-ratios-backward-lags'
    source_code = b''.join(path.read_bytes() for path in sorted((root / 'src/credit_risk').rglob('*.py')))
    code_hash = hashlib.sha256(source_code).hexdigest()
    run_id = hashlib.sha256((audit['sha256'] + json.dumps(config, sort_keys=True) + code_hash).encode()).hexdigest()[:16]
    card = {
        'run_id': run_id, 'feature_policy': policy, 'code_sha256': code_hash, 'configuration': config, 'source_audit': audit,
        'selected_model': bundle['selected_model'], 'probability_source': bundle['probability_source'],
        'threshold': threshold, 'target': bundle['target'], 'horizon_years': 1,
        'split_summary': split_rows, 'test_metrics': probability_metrics(y, test.selected_pd, threshold),
        'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'unseen_entity_policy': 'SHA-256 allocation before fitting; reserved entities excluded from train, calibration, validation and development diagnostics.',
        'limitations': [audit['timing_limitation'], audit['negative_followup_limitation'],
                        'Label reconstruction matches published annual counts but individual filing dates are not independently verified.',
                        'Fiscal-year chronology is not a verified as-of-filing backtest.',
                        'Transfer to other countries, private firms, or contemporary conditions is untested.',
                        'Internal one-year grades are project assumptions.'],
    }
    bundle['run_id'] = run_id
    bundle['model_card'] = card
    joblib.dump(bundle, model_dir / 'credit_risk_model.joblib')
    (report_dir / 'model_card.json').write_text(json.dumps(card, indent=2))
    (report_dir / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    return bundle, test, card
