"""Train baselines and write validation/test model artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.ingestion.csv_loader import read_financial_csv
from credit_risk.models.training import train_and_evaluate
from credit_risk.reporting.model_results import render_model_results

CSV_PATH = ROOT / "data" / "raw" / "polish_bankruptcy_1year.csv"


def main() -> None:
    data = read_financial_csv(CSV_PATH)
    result = train_and_evaluate(
        data,
        model_dir=ROOT / "models",
        report_dir=ROOT / "reports",
    )
    results_document = render_model_results(
        report_dir=ROOT / "reports",
        output_path=ROOT / "docs" / "model_results.md",
    )
    print(f"Selected on validation PR-AUC: {result['selected_model']}")
    print(f"PD source selected on validation Brier score: {result['probability_source']}")
    print("\nValidation model comparison (sorted by PR-AUC):")
    print(
        result["comparison"][
            ["model", "roc_auc", "pr_auc_average_precision", "brier_score", "precision", "recall", "f1"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nHeld-out test metrics at the validation-selected operating threshold:")
    for key, value in result["test_operating_metrics"].items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    grade = result["example_risk_grade"]
    print(
        f"\nExample held-out observation {result['example_observation_id']}: "
        f"PD proxy={grade['pd']:.2%}, score={grade['risk_score']}, "
        f"Project Internal Credit Risk Rating={grade['internal_grade']} ({grade['grade_label']})"
    )
    print("Artifacts saved under reports/; fitted model saved under models/ (both are Git-ignored).")
    print(f"Generated results document: {results_document.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
