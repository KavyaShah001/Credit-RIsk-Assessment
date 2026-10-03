"""Validate the prepared CSV and load/query its records in local SQLite."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.database.sqlite_store import load_csv_to_sqlite
from credit_risk.ingestion.csv_loader import read_financial_csv

CSV_PATH = ROOT / "data" / "raw" / "polish_bankruptcy_1year.csv"
DATABASE_PATH = ROOT / "data" / "processed" / "credit_risk.sqlite3"


def main() -> None:
    frame = read_financial_csv(CSV_PATH)
    summary = load_csv_to_sqlite(frame, DATABASE_PATH)
    print(f"Loaded SQLite database: {DATABASE_PATH.relative_to(ROOT)}")
    print(f"Rows: {summary['rows_loaded']:,}")
    print(f"Bankruptcies in the five-year outcome window: {summary['defaults_loaded']:,}")
    print(f"Observed bankruptcy share: {summary['default_rate_pct']:.2f}%")
    print(f"Missing ratio cells: {summary['missing_ratio_cells']:,}")

    with sqlite3.connect(DATABASE_PATH) as connection:
        by_outcome = connection.execute(
            """SELECT d.defaulted, COUNT(*) AS observations
               FROM companies AS c
               JOIN default_outcomes AS d ON d.company_id = c.company_id
               GROUP BY d.defaulted
               ORDER BY d.defaulted"""
        ).fetchall()
        missing_by_ratio = connection.execute(
            """SELECT r.ratio_code, r.description, COUNT(*) AS missing_observations
               FROM financial_ratios AS f
               JOIN ratio_dictionary AS r ON r.ratio_code = f.ratio_code
               WHERE f.ratio_value IS NULL
               GROUP BY r.ratio_code, r.description
               ORDER BY missing_observations DESC, r.ratio_code
               LIMIT 5"""
        ).fetchall()

    print("SQL outcome aggregation (bankruptcy label, observations):")
    for defaulted, observations in by_outcome:
        print(f"  {defaulted}: {observations:,}")
    print("Ratios with the most missing observations:")
    for ratio_code, description, missing_count in missing_by_ratio:
        print(f"  {ratio_code}: {missing_count:,} missing — {description}")


if __name__ == "__main__":
    main()
