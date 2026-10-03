"""Render tracked model-results documentation from generated run artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc_average_precision", "PR-AUC (average precision)"),
    ("brier_score", "Brier score"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return chr(10).join(output)


def _metric_cell(value: float) -> str:
    return f"{value:.4f}"


def render_model_results(report_dir: str | Path, output_path: str | Path) -> Path:
    """Generate a concise, source-controlled Markdown record of the latest run."""
    report_dir = Path(report_dir)
    output_path = Path(output_path)
    model_card = json.loads((report_dir / "model_card.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(report_dir / "model_comparison_validation.csv")
    raw_calibrated = pd.read_csv(report_dir / "selected_model_raw_vs_calibrated_test.csv")
    importance = pd.read_csv(report_dir / "permutation_importance_validation.csv").head(10)
    sensitivity = pd.read_csv(report_dir / "example_observation_sensitivity.csv")

    comparison_rows = [
        [row.model]
        + [_metric_cell(float(getattr(row, key))) for key, _ in METRIC_COLUMNS]
        for row in comparison.itertuples(index=False)
    ]
    comparison_table = _table(
        ["Model (selected by validation PR-AUC)"] + [label for _, label in METRIC_COLUMNS],
        comparison_rows,
    )
    test_rows = [
        [row.probability_type]
        + [_metric_cell(float(getattr(row, key))) for key, _ in METRIC_COLUMNS]
        for row in raw_calibrated.itertuples(index=False)
    ]
    test_table = _table(["Probability"] + [label for _, label in METRIC_COLUMNS], test_rows)

    split = model_card["split"]
    result = model_card["example_observation_risk_result"]
    grade_rows = []
    prior = 0.0
    for upper, grade, label in (
        (item["upper_pd_exclusive"], item["internal_grade"], item["label"])
        for item in model_card["grade_bands"]
    ):
        upper = float(upper)
        if grade == "I1":
            interval = "<1.0%"
        elif grade == "I5":
            interval = ">=15.0%"
        else:
            interval = f"{prior:.1%} to <{upper:.1%}"
        grade_rows.append([interval, grade, label])
        prior = upper

    importance_rows = [
        [
            row.feature,
            row.category,
            row.description,
            _metric_cell(float(row.importance_mean_pr_auc_decrease)),
            _metric_cell(float(row.importance_std)),
        ]
        for row in importance.itertuples(index=False)
    ]
    sensitivity_rows: list[list[object]] = []
    for direction, group in (
        ("Higher than median-setting scenario", sensitivity[sensitivity["pd_change"] < 0].head(5)),
        ("Lower than median-setting scenario", sensitivity[sensitivity["pd_change"] > 0].head(5)),
    ):
        for row in group.itertuples(index=False):
            sensitivity_rows.append(
                [
                    direction,
                    row.feature,
                    row.source_definition,
                    f"{float(row.pd_change) * 100:+.3f} pp",
                ]
            )

    calibration_state = (
        "Sigmoid calibration improved validation Brier score and is selected for displayed PD."
        if model_card["calibration_improved_validation_brier"]
        else "Sigmoid calibration did not improve validation Brier score; raw model probabilities are selected."
    )
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    operating = model_card["test_metrics_selected_probability_and_threshold"]
    text = f"""# Model Results

Generated from the current source dataset and model pipeline on {current_time}. Regenerate with `.venv/bin/python scripts/train_models.py`.

## Outcome, split, and selection

- Target: bankruptcy within five years, used as a default-risk proxy.
- Model selected by validation PR-AUC: **{model_card['selected_model']}**.
- Split: stratified random, {split['train_rows']:,} train / {split['validation_rows']:,} validation / {split['test_rows']:,} test observations.
- Positive outcomes: {split['train_positive']:,} train / {split['validation_positive']:,} validation / {split['test_positive']:,} test.
- The source provides no observation dates or persistent company keys, so a temporal or company-grouped holdout was not possible. The test set was not used for model selection.
- Class weights address training imbalance; validation and test retain the natural source prevalence.

## Validation model comparison

The table shows validation metrics at the default 0.50 threshold. The model is selected by PR-AUC, since accuracy alone can conceal poor default capture when outcomes are rare.

{comparison_table}

## Selected model: raw versus calibrated probabilities on test

{calibration_state} Calibration was fitted by cross-validation within training data. The table below uses a 0.50 threshold for precision/recall/F1; the ranking and probability metrics do not depend on that threshold.

{test_table}

The validation-selected operating threshold is **{model_card['decision_threshold_selected_on_validation']:.4f}**. At that threshold on the held-out test set: ROC-AUC {operating['roc_auc']:.4f}, PR-AUC {operating['pr_auc_average_precision']:.4f}, precision {operating['precision']:.4f}, recall {operating['recall']:.4f}, and F1 {operating['f1']:.4f}. The threshold was selected on validation data, not on the test set.

## Internal grade mapping

This is an explicit project assumption for demonstration; it is not an external scale or lending policy.

{_table(["Five-year PD proxy band", "Grade", "Description"], grade_rows)}

## Global feature importance

Permutation importance is the average decrease in validation PR-AUC after shuffling a feature. It measures model reliance in this dataset, not causality.

{_table(["Feature", "Category", "Source definition", "Mean PR-AUC decrease", "Std. deviation"], importance_rows)}

## Example local sensitivity

For observation **{model_card['example_observation_id']}**, selected PD proxy is {float(result['pd']):.2%}, risk score {result['risk_score']}, grade **{result['internal_grade']}**. The factors below show how the model score changes if one ratio at a time is replaced by its training-set median. They are sensitivity checks, not causal contributions.

{_table(["Direction relative to median scenario", "Feature", "Source definition", "PD change"], sensitivity_rows)}

## Limitations

- The target is five-year bankruptcy, not a directly observed contractual default event.
- The random holdout is not a time-based backtest; source-period differences may affect results.
- The held-out test set contains {split['test_positive']} positive cases. Performance and calibration therefore have substantial statistical uncertainty.
- Calibration should be treated as a check, not proof that probabilities transfer to new sectors, geographies, or time periods.
- Project grades, score thresholds, model explanations, and any future LGD/EAD assumptions are educational outputs, not operational decisions.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path
