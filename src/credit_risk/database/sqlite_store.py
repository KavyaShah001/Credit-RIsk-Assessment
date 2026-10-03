"""SQLite schema and idempotent loader for the MVP source data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from credit_risk.ingestion.csv_loader import RATIO_COLUMNS, TARGET_COLUMN

DATASET_ID = "uci_polish_1year"
OUTCOME_NAME = "bankruptcy_within_5_years"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data"


def _create_schema(connection: sqlite3.Connection) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    connection.executescript(schema_path.read_text(encoding="utf-8"))


def _load_dictionary(connection: sqlite3.Connection) -> None:
    from credit_risk.financial_analysis.ratio_dictionary import RATIO_DICTIONARY

    connection.executemany(
        """INSERT INTO ratio_dictionary (ratio_code, source_attribute, category, description)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(ratio_code) DO UPDATE SET
             source_attribute=excluded.source_attribute,
             category=excluded.category,
             description=excluded.description""",
        [
            (item["code"], item["source_attribute"], item["category"], item["description"])
            for item in RATIO_DICTIONARY
        ],
    )


def load_csv_to_sqlite(frame: pd.DataFrame, database_path: str | Path) -> dict[str, int]:
    """Replace this single-source MVP dataset in SQLite and return load statistics."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_cells = int(frame[RATIO_COLUMNS].isna().sum().sum())
    default_count = int(frame[TARGET_COLUMN].sum())

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _create_schema(connection)
        connection.execute("DELETE FROM ingestion_runs WHERE source_dataset = ?", (DATASET_ID,))
        connection.execute("DELETE FROM companies WHERE source_dataset = ?", (DATASET_ID,))
        _load_dictionary(connection)

        company_records = [
            (f"UCI-1YEAR-{row_number:06d}", DATASET_ID, row_number, None, None)
            for row_number in range(1, len(frame) + 1)
        ]
        connection.executemany(
            """INSERT INTO companies
               (company_id, source_dataset, source_row_number, company_name, sector)
               VALUES (?, ?, ?, ?, ?)""",
            company_records,
        )

        connection.executemany(
            "INSERT INTO financial_observations (company_id, observation_case) VALUES (?, '1year')",
            [(f"UCI-1YEAR-{row_number:06d}",) for row_number in range(1, len(frame) + 1)],
        )
        ratio_records = (
            (row_number, ratio_code, None if pd.isna(value) else float(value))
            for row_number, row in enumerate(frame[RATIO_COLUMNS].itertuples(index=False, name=None), 1)
            for ratio_code, value in zip(RATIO_COLUMNS, row)
        )
        connection.executemany(
            "INSERT INTO financial_ratios (observation_id, ratio_code, ratio_value) VALUES (?, ?, ?)",
            ratio_records,
        )
        connection.executemany(
            """INSERT INTO default_outcomes (company_id, outcome_name, horizon_years, defaulted)
               VALUES (?, ?, 5, ?)""",
            [
                (f"UCI-1YEAR-{row_number:06d}", OUTCOME_NAME, int(defaulted))
                for row_number, defaulted in enumerate(frame[TARGET_COLUMN].tolist(), 1)
            ],
        )
        connection.execute(
            """INSERT INTO ingestion_runs
               (source_dataset, source_url, source_license, rows_loaded, defaults_loaded, missing_ratio_cells)
               VALUES (?, ?, 'CC BY 4.0', ?, ?, ?)""",
            (DATASET_ID, SOURCE_URL, len(frame), default_count, missing_cells),
        )
        summary = connection.execute(
            """SELECT COUNT(*) AS rows_loaded,
                      SUM(defaulted) AS defaults_loaded,
                      ROUND(100.0 * AVG(defaulted), 3) AS default_rate_pct
               FROM default_outcomes"""
        ).fetchone()

    return {
        "rows_loaded": int(summary[0]),
        "defaults_loaded": int(summary[1]),
        "default_rate_pct": float(summary[2]),
        "missing_ratio_cells": missing_cells,
    }
