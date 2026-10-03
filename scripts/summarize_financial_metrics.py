"""Create actual population summaries for supported financial metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.financial_analysis.ratio_analysis import analyze_financial_ratios
from credit_risk.ingestion.csv_loader import read_financial_csv

CSV_PATH = ROOT / "data" / "raw" / "polish_bankruptcy_1year.csv"
OUTPUT_PATH = ROOT / "reports" / "financial_metric_summary.csv"


def main() -> None:
    observations = read_financial_csv(CSV_PATH)
    rows: list[dict[str, object]] = []
    for record in observations.to_dict(orient="records"):
        for metric in analyze_financial_ratios(record):
            rows.append(metric.to_dict())

    metrics = pd.DataFrame(rows)
    summary: list[dict[str, object]] = []
    grouping = ["code", "name", "category", "formula", "source_fields", "interpretation", "limitation"]
    for _, group in metrics.groupby(grouping, sort=False, dropna=False):
        metric = group.iloc[0]
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        summary.append(
            {
                "code": metric["code"],
                "name": metric["name"],
                "category": metric["category"],
                "formula": metric["formula"],
                "source_fields": ", ".join(metric["source_fields"]),
                "observations": len(group),
                "available_observations": len(values),
                "unavailable_or_undefined": len(group) - len(values),
                "median": values.median() if len(values) else None,
                "p25": values.quantile(0.25) if len(values) else None,
                "p75": values.quantile(0.75) if len(values) else None,
                "interpretation": metric["interpretation"],
                "limitation": metric["limitation"],
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Observations summarized: {len(observations):,}")
    print("Supported metric distribution summary (median and interquartile range):")
    for item in summary:
        print(
            f"  {item['code']}: median={item['median']!s}, "
            f"IQR=({item['p25']!s}, {item['p75']!s}), "
            f"available={item['available_observations']:,}/{item['observations']:,}"
        )


if __name__ == "__main__":
    main()
