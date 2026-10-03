"""Probability diagnostics, entity-cluster uncertainty and training-reference drift."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)


def probability_metrics(y, p, threshold=0.5) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not len(y) or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError('Metrics require finite probabilities and nonempty observations.')
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        'rows': len(y), 'events': int(y.sum()), 'event_rate': float(y.mean()),
        'mean_pd': float(p.mean()),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'pr_auc': float(average_precision_score(y, p)) if y.sum() else None,
        'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'threshold': float(threshold), 'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def reliability_table(y, p, bins=8) -> pd.DataFrame:
    frame = pd.DataFrame({'target': np.asarray(y), 'pd': np.asarray(p)})
    edges = np.unique(np.quantile(frame.pd, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 1:
        frame['bin'] = 'all'
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        frame['bin'] = pd.cut(frame.pd, edges).astype(str)
    return frame.groupby('bin', observed=True).agg(
        observations=('target', 'size'), events=('target', 'sum'),
        mean_predicted_pd=('pd', 'mean'), observed_rate=('target', 'mean'),
    ).reset_index().sort_values('mean_predicted_pd')


def entity_bootstrap(predictions: pd.DataFrame, repeats=500, seed=20261004) -> pd.DataFrame:
    """Resample whole entity clusters, keeping all their annual rows together.

    Intervals condition on the fitted model and this historical sample; they do not
    include model-selection, label-reconstruction or macro-regime uncertainty.
    """
    codes, companies = pd.factorize(predictions.company_id)
    y = predictions.bankruptcy_next_year.to_numpy()
    p = predictions.selected_pd.to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        counts = np.bincount(rng.integers(0, len(companies), size=len(companies)), minlength=len(companies))
        weights = counts[codes]
        if weights[y == 1].sum() == 0 or weights[y == 0].sum() == 0:
            continue
        samples.append([
            roc_auc_score(y, p, sample_weight=weights),
            average_precision_score(y, p, sample_weight=weights),
            np.average((p - y) ** 2, weights=weights),
        ])
    if not samples:
        return pd.DataFrame(columns=['metric', 'estimate', 'lower_95', 'upper_95', 'valid_replicates'])
    boot = np.asarray(samples)
    point = probability_metrics(y, p)
    return pd.DataFrame([
        {'metric': name, 'estimate': point[name], 'lower_95': float(np.quantile(boot[:, i], .025)),
         'upper_95': float(np.quantile(boot[:, i], .975)), 'valid_replicates': len(boot)}
        for i, name in enumerate(['roc_auc', 'pr_auc', 'brier'])
    ])


def feature_drift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """PSI with cutpoints fitted on training; missingness is a separate bin."""
    records = []
    for name in train.columns:
        reference, current = train[name].dropna(), test[name].dropna()
        if reference.empty:
            records.append({'feature': name, 'psi': None, 'train_missing': 1.0,
                            'test_missing': float(test[name].isna().mean())})
            continue
        internal = np.unique(reference.quantile(np.linspace(.1, .9, 9)).to_numpy())
        edges = np.r_[-np.inf, internal, np.inf]
        a = np.r_[np.histogram(reference, edges)[0], train[name].isna().sum()].astype(float)
        b = np.r_[np.histogram(current, edges)[0], test[name].isna().sum()].astype(float)
        a, b = np.maximum(a / a.sum(), 1e-6), np.maximum(b / b.sum(), 1e-6)
        a, b = a / a.sum(), b / b.sum()
        records.append({'feature': name, 'psi': float(((b - a) * np.log(b / a)).sum()),
                        'train_missing': float(train[name].isna().mean()),
                        'test_missing': float(test[name].isna().mean())})
    return pd.DataFrame(records).sort_values('psi', ascending=False)
