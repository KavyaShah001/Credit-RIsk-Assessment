"""Small contract tests for CSV validation and the SQLite load path."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from credit_risk.database.sqlite_store import load_csv_to_sqlite
from credit_risk.ingestion.csv_loader import RATIO_COLUMNS, read_financial_csv


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(0.1, index=range(2), columns=RATIO_COLUMNS)
        frame.loc[0, "ratio_01"] = float("nan")
        frame["defaulted"] = [0, 1]
        return frame

    def test_csv_accepts_missing_ratios_and_binary_labels(self) -> None:
        path = self.root / "valid.csv"
        self.make_frame().to_csv(path, index=False)

        result = read_financial_csv(path)

        self.assertEqual(result.shape, (2, 65))
        self.assertEqual(result["defaulted"].tolist(), [0, 1])
        self.assertTrue(pd.isna(result.loc[0, "ratio_01"]))

    def test_csv_rejects_invalid_target_and_unexpected_columns(self) -> None:
        invalid_target = self.make_frame()
        invalid_target.loc[0, "defaulted"] = 2
        target_path = self.root / "invalid_target.csv"
        invalid_target.to_csv(target_path, index=False)
        with self.assertRaisesRegex(ValueError, "binary default labels"):
            read_financial_csv(target_path)

        extra_column = self.make_frame()
        extra_column["company_name"] = "invented"
        extra_path = self.root / "extra_column.csv"
        extra_column.to_csv(extra_path, index=False)
        with self.assertRaisesRegex(ValueError, "unexpected columns"):
            read_financial_csv(extra_path)

    def test_csv_rejects_observation_without_any_ratio(self) -> None:
        frame = self.make_frame()
        frame.loc[0, RATIO_COLUMNS] = float("nan")
        path = self.root / "empty_observation.csv"
        frame.to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "no numeric ratio values"):
            read_financial_csv(path)

    def test_sqlite_load_is_idempotent_and_counts_missingness(self) -> None:
        database = self.root / "nested" / "credit_risk.sqlite3"
        frame = self.make_frame()

        first = load_csv_to_sqlite(frame, database)
        second = load_csv_to_sqlite(frame, database)

        self.assertEqual(first, second)
        self.assertEqual(first["rows_loaded"], 2)
        self.assertEqual(first["defaults_loaded"], 1)
        self.assertEqual(first["missing_ratio_cells"], 1)
        with sqlite3.connect(database) as connection:
            company_count = connection.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
            ratio_count = connection.execute("SELECT COUNT(*) FROM financial_ratios").fetchone()[0]
            outcome_count = connection.execute("SELECT COUNT(*) FROM default_outcomes").fetchone()[0]
            self.assertEqual((company_count, ratio_count, outcome_count), (2, 128, 2))


if __name__ == "__main__":
    unittest.main()
