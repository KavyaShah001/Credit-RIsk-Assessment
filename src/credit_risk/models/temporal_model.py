"""Independent annual-panel models with chronological probability calibration."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from credit_risk.features.panel_features import QuantileClipper


def candidates(seed=20261004):
    estimators = {
        'logistic_regression': LogisticRegression(C=.1, class_weight='balanced', max_iter=2000, random_state=seed),
        'decision_tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, class_weight='balanced', random_state=seed),
        'random_forest': RandomForestClassifier(n_estimators=160, max_depth=12, min_samples_leaf=20,
                            class_weight='balanced_subsample', n_jobs=1, random_state=seed),
        'gradient_boosting': HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, min_samples_leaf=40,
                            l2_regularization=5, class_weight='balanced', early_stopping=False, random_state=seed),
    }
    return {name: Pipeline([
        ('clip', QuantileClipper()),
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler()), ('model', estimator),
    ]) for name, estimator in estimators.items()}


class TemporalSigmoid:
    """Unweighted Platt logistic fit on a later, disjoint calibration period."""
    def __init__(self, estimator):
        self.estimator = estimator

    @staticmethod
    def _log_odds(probabilities):
        p = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, X, y):
        raw = self.estimator.predict_proba(X)[:, 1]
        self.sigmoid = LogisticRegression(C=1.0, max_iter=1000)
        self.sigmoid.fit(self._log_odds(raw), y)
        self.monotone_ = bool(self.sigmoid.coef_[0, 0] > 0)
        return self

    def predict_proba(self, X):
        return self.sigmoid.predict_proba(self._log_odds(self.estimator.predict_proba(X)[:, 1]))


# One-year demonstration bands, intentionally distinct from the legacy five-year bands.
PANEL_GRADE_BANDS = ((.0025, 'T1'), (.01, 'T2'), (.03, 'T3'), (.10, 'T4'), (1.000001, 'T5'))


def annual_grade(probability):
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('PD must be finite and in [0, 1].')
    return next(grade for bound, grade in PANEL_GRADE_BANDS if probability < bound)


def score_features(bundle, features):
    X = features[bundle['feature_columns']]
    p = bundle['model'].predict_proba(X)[:, 1]
    result = features.iloc[:, :0].copy()
    result['selected_pd'] = p
    result['internal_grade'] = [annual_grade(float(value)) for value in p]
    result['risk_score'] = np.rint(100 * (1 - p)).astype(int)
    result['missing_features'] = X.isna().sum(axis=1)
    result['outside_training_range'] = (X.lt(bundle['training_lower']) | X.gt(bundle['training_upper'])).sum(axis=1)
    return result
