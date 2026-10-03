"""Validate the canonical CSV produced by the UCI source adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RATIO_COLUMNS = [f"ratio_{i:02d}" for i in range(1, 65)]
TARGET_COLUMN = "defaulted"
REQUIRED_COLUMNS = RATIO_COLUMNS + [TARGET_COLUMN]


def read_financial_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate one row per anonymous company observation."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CSV file not found: {source}")

    frame = pd.read_csv(source, na_values=["", "?"], keep_default_na=True)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(
            "CSV schema mismatch. "
            f"Missing columns: {missing_columns or 'none'}; "
            f"unexpected columns: {extra_columns or 'none'}."
        )
    frame = frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame.empty:
        raise ValueError("The CSV has no observations.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("The default outcome contains missing or non-numeric values.")
    labels = set(frame[TARGET_COLUMN].astype(int).unique())
    if not labels.issubset({0, 1}) or frame[TARGET_COLUMN].ne(frame[TARGET_COLUMN].astype(int)).any():
        raise ValueError(f"Expected binary default labels 0/1; found {sorted(labels)}.")
    if not frame[RATIO_COLUMNS].notna().any(axis=1).all():
        bad_count = int((~frame[RATIO_COLUMNS].notna().any(axis=1)).sum())
        raise ValueError(f"{bad_count} observations have no numeric ratio values at all.")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")
    return frame
