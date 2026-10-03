"""Leakage-aware development, calibration, and evaluation on the UCI source."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "credit-risk-system-matplotlib"),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(os.cpu_count() or 2, 4)))

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.evaluation.metrics import binary_metrics
from credit_risk.financial_analysis.ratio_dictionary import RATIO_DICTIONARY
from credit_risk.ingestion.csv_loader import RATIO_COLUMNS, TARGET_COLUMN
from credit_risk.models.risk_grades import GRADE_BANDS, grade_from_pd

RANDOM_SEED = 20261004


def model_candidates() -> dict[str, Pipeline]:
    """Create the three modest baseline classifiers used by the MVP."""
    def preprocessor_steps():
        return [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]

    return {
        "logistic_regression": Pipeline(
            preprocessor_steps()
            + [
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2500,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        ),
        "random_forest": Pipeline(
            preprocessor_steps()
            + [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        ),
        "gradient_boosting": Pipeline(
            preprocessor_steps()
            + [
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=150,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        ),
    }


def stratified_data_split(
    features: pd.DataFrame,
    target: pd.Series,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create 60/20/20 train/validation/test partitions with label stratification."""
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        features,
        target,
        test_size=0.4,
        random_state=random_seed,
        stratify=target,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.5,
        random_state=random_seed,
        stratify=y_remaining,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def choose_f1_threshold(target: pd.Series, probabilities: np.ndarray) -> float:
    """Choose an illustrative F1 operating threshold on validation data only."""
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def _global_permutation_importance(
    estimator: Pipeline,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    repeats: int = 5,
) -> pd.DataFrame:
    importance = permutation_importance(
        estimator,
        x_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    descriptions = {item["code"]: item for item in RATIO_DICTIONARY}
    records = []
    for index, feature in enumerate(x_validation.columns):
        definition = descriptions.get(feature)
        records.append(
            {
                "feature": feature,
                "source_ratio": feature,
                "description": definition["description"] if definition else feature,
                "category": definition.get("category", "Other source ratio") if definition else "Other source ratio",
                "importance_mean_pr_auc_decrease": float(importance.importances_mean[index]),
                "importance_std": float(importance.importances_std[index]),
            }
        )
    return pd.DataFrame(records).sort_values("importance_mean_pr_auc_decrease", ascending=False)


def _calibration_rows(y_true: pd.Series, raw: np.ndarray, calibrated: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for label, probabilities in (("raw", raw), ("sigmoid_calibrated", calibrated)):
        observed, predicted = calibration_curve(
            y_true,
            probabilities,
            n_bins=8,
            strategy="quantile",
        )
        records.extend(
            {
                "probability_type": label,
                "mean_predicted_probability": float(predicted_value),
                "observed_bankruptcy_frequency": float(observed_value),
            }
            for observed_value, predicted_value in zip(observed, predicted)
        )
    return pd.DataFrame(records)


def _save_test_plots(
    y_test: pd.Series,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    selected_probabilities: np.ndarray,
    operating_threshold: float,
    report_dir: Path,
) -> None:
    from sklearn.metrics import precision_recall_curve as pr_curve
    from sklearn.metrics import roc_curve as receiver_operator_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, probabilities in (
        ("Raw", raw_probabilities),
        ("Sigmoid calibrated", calibrated_probabilities),
    ):
        fpr, tpr, _ = receiver_operator_curve(y_test, probabilities)
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve — held-out test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curve_test.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, probabilities in (
        ("Raw", raw_probabilities),
        ("Sigmoid calibrated", calibrated_probabilities),
    ):
        precision, recall, _ = pr_curve(y_test, probabilities)
        ax.plot(recall, precision, label=label)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve — held-out test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "precision_recall_curve_test.png", dpi=140)
    plt.close(fig)

    calibration = _calibration_rows(y_test, raw_probabilities, calibrated_probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, group in calibration.groupby("probability_type"):
        ax.plot(
            group["mean_predicted_probability"],
            group["observed_bankruptcy_frequency"],
            marker="o",
            label=label.replace("_", " "),
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed bankruptcy frequency",
        title="Calibration — held-out test",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "calibration_curve_test.png", dpi=140)
    plt.close(fig)

    test_predictions = (selected_probabilities >= operating_threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=[0, 1],
        display_labels=["No bankruptcy", "Bankruptcy"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Test confusion matrix (threshold={operating_threshold:.3f})")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix_test.png", dpi=140)
    plt.close(fig)


def train_and_evaluate(
    data: pd.DataFrame,
    model_dir: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    """Train candidate models, calibrate the selected model, and save review artifacts."""
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    features = data[RATIO_COLUMNS].copy()
    target = data[TARGET_COLUMN].astype(int).copy()
    x_train, x_validation, x_test, y_train, y_validation, y_test = stratified_data_split(features, target)

    candidates = model_candidates()
    comparison_rows: list[dict[str, Any]] = []
    for name, estimator in candidates.items():
        estimator.fit(x_train, y_train)
        validation_probabilities = estimator.predict_proba(x_validation)[:, 1]
        metrics = binary_metrics(y_validation, validation_probabilities, threshold=0.5)
        comparison_rows.append({"model": name, "split": "validation", **metrics})

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "pr_auc_average_precision", ascending=False
    )
    selected_name = str(comparison.iloc[0]["model"])
    selected_estimator = candidates[selected_name]

    calibration_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    calibrated_estimator = CalibratedClassifierCV(
        estimator=selected_estimator,
        method="sigmoid",
        cv=calibration_cv,
        ensemble=False,
    )
    calibrated_estimator.fit(x_train, y_train)

    raw_validation_probabilities = selected_estimator.predict_proba(x_validation)[:, 1]
    calibrated_validation_probabilities = calibrated_estimator.predict_proba(x_validation)[:, 1]
    raw_validation_brier = binary_metrics(
        y_validation, raw_validation_probabilities
    )["brier_score"]
    calibrated_validation_brier = binary_metrics(
        y_validation, calibrated_validation_probabilities
    )["brier_score"]
    use_calibrated_probability = calibrated_validation_brier <= raw_validation_brier
    probability_source = (
        "sigmoid_calibrated"
        if use_calibrated_probability
        else "raw_model_probability (sigmoid calibration did not improve validation Brier score)"
    )
    operating_validation_probabilities = (
        calibrated_validation_probabilities
        if use_calibrated_probability
        else raw_validation_probabilities
    )
    operating_threshold = choose_f1_threshold(y_validation, operating_validation_probabilities)

    raw_test_probabilities = selected_estimator.predict_proba(x_test)[:, 1]
    calibrated_test_probabilities = calibrated_estimator.predict_proba(x_test)[:, 1]
    selected_test_probabilities = (
        calibrated_test_probabilities if use_calibrated_probability else raw_test_probabilities
    )
    test_raw_metrics = binary_metrics(y_test, raw_test_probabilities, threshold=0.5)
    test_calibrated_metrics = binary_metrics(y_test, calibrated_test_probabilities, threshold=0.5)
    test_operating_metrics = binary_metrics(
        y_test, selected_test_probabilities, threshold=operating_threshold
    )
    test_index = x_test.index[0]
    example_observation_id = f"UCI-1YEAR-{int(test_index) + 1:06d}"
    example_grade = grade_from_pd(float(selected_test_probabilities[0]))

    test_scores = []
    for observation_index, raw_probability, calibrated_probability, selected_probability in zip(
        x_test.index,
        raw_test_probabilities,
        calibrated_test_probabilities,
        selected_test_probabilities,
    ):
        grade = grade_from_pd(float(selected_probability))
        test_scores.append(
            {
                "observation_id": f"UCI-1YEAR-{int(observation_index) + 1:06d}",
                "raw_pd": float(raw_probability),
                "calibrated_pd": float(calibrated_probability),
                "selected_pd": float(selected_probability),
                "risk_score": grade["risk_score"],
                "internal_grade": grade["internal_grade"],
                "selected_model": selected_name,
                "probability_source": probability_source,
            }
        )
    pd.DataFrame(test_scores).to_csv(report_dir / "test_observation_scores.csv", index=False)

    comparison.to_csv(report_dir / "model_comparison_validation.csv", index=False)
    pd.DataFrame(
        [
            {"probability_type": "raw", **test_raw_metrics},
            {"probability_type": "sigmoid_calibrated", **test_calibrated_metrics},
        ]
    ).to_csv(report_dir / "selected_model_raw_vs_calibrated_test.csv", index=False)
    pd.DataFrame([test_operating_metrics]).to_csv(
        report_dir / "selected_model_test_at_validation_threshold.csv", index=False
    )
    _calibration_rows(y_test, raw_test_probabilities, calibrated_test_probabilities).to_csv(
        report_dir / "calibration_curve_test.csv", index=False
    )
    _save_test_plots(
        y_test,
        raw_test_probabilities,
        calibrated_test_probabilities,
        selected_test_probabilities,
        operating_threshold,
        report_dir,
    )

    importance = _global_permutation_importance(selected_estimator, x_validation, y_validation)
    importance.to_csv(report_dir / "permutation_importance_validation.csv", index=False)

    example = x_test.loc[[test_index], RATIO_COLUMNS]
    train_medians = x_train.median()
    baseline_test_pd = float(
        calibrated_estimator.predict_proba(example)[:, 1][0]
        if use_calibrated_probability
        else selected_estimator.predict_proba(example)[:, 1][0]
    )
    active_explainer = calibrated_estimator if use_calibrated_probability else selected_estimator
    from credit_risk.explainability.local_sensitivity import local_ratio_sensitivity

    local_drivers = local_ratio_sensitivity(
        active_explainer,
        example,
        train_medians,
        RATIO_COLUMNS,
        baseline_pd=baseline_test_pd,
    )
    ratio_metadata = {item["code"]: item for item in RATIO_DICTIONARY}
    local_drivers["source_definition"] = local_drivers["feature"].map(
        lambda code: ratio_metadata.get(code, {}).get("description", code)
    )
    local_drivers["category"] = local_drivers["feature"].map(
        lambda code: ratio_metadata.get(code, {}).get("category", "Other source ratio")
    )
    local_drivers.insert(0, "observation_id", example_observation_id)
    local_drivers.to_csv(report_dir / "example_observation_sensitivity.csv", index=False)

    band_descriptions = [
        {
            "upper_pd_exclusive": upper,
            "internal_grade": grade,
            "label": label,
        }
        for upper, grade, label in GRADE_BANDS
    ]
    model_card = {
        "selected_model": selected_name,
        "target": "bankruptcy within five years (default-risk proxy)",
        "selected_probability_source": probability_source,
        "sigmoid_calibration_validation_brier": float(calibrated_validation_brier),
        "raw_validation_brier": float(raw_validation_brier),
        "calibration_improved_validation_brier": bool(use_calibrated_probability),
        "decision_threshold_selected_on_validation": operating_threshold,
        "example_observation_id": example_observation_id,
        "example_observation_risk_result": example_grade,
        "risk_score_formula": "round(100 * (1 - selected_probability))",
        "grade_bands": band_descriptions,
        "random_seed": RANDOM_SEED,
        "split": {
            "method": "stratified random split; no dates or stable company identifiers in selected source",
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "test_rows": len(x_test),
            "train_positive": int(y_train.sum()),
            "validation_positive": int(y_validation.sum()),
            "test_positive": int(y_test.sum()),
            "test_used_for_model_selection": False,
        },
        "class_imbalance_handling": (
            "Balanced class weights in logistic regression, random forest, and gradient boosting; "
            "validation and test retain source prevalence."
        ),
        "caveats": [
            "The target is five-year bankruptcy, not an observed contractual default event.",
            "The file has no company identifier or observation date; temporal/grouped holdout is unavailable.",
            "Validation/test default counts are small, so performance estimates have substantial sampling uncertainty.",
            "Internal grade cutoffs are project assumptions, not external ratings or a lending methodology.",
            "Local sensitivity and permutation importance are associations, not causal explanations.",
        ],
        "test_metrics_raw": test_raw_metrics,
        "test_metrics_sigmoid_calibrated": test_calibrated_metrics,
        "test_metrics_selected_probability_and_threshold": test_operating_metrics,
    }
    (report_dir / "model_card.json").write_text(
        json.dumps(model_card, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "raw_model": selected_estimator,
            "calibrated_model": calibrated_estimator,
            "use_calibrated_probability": use_calibrated_probability,
            "selected_model": selected_name,
            "training_medians": train_medians,
            "feature_columns": RATIO_COLUMNS,
            "operating_threshold": operating_threshold,
            "model_card": model_card,
        },
        model_dir / "credit_risk_model.joblib",
    )

    return {
        "selected_model": selected_name,
        "probability_source": probability_source,
        "comparison": comparison,
        "test_raw_metrics": test_raw_metrics,
        "test_calibrated_metrics": test_calibrated_metrics,
        "test_operating_metrics": test_operating_metrics,
        "operating_threshold": operating_threshold,
        "example_risk_grade": example_grade,
        "example_observation_id": example_observation_id,
    }
