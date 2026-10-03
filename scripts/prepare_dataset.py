"""Download the UCI archive and convert its 1st-year ARFF file to CSV."""

from __future__ import annotations

import csv
import io
import re
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "uci_polish_bankruptcy.zip"
CSV_PATH = RAW_DIR / "polish_bankruptcy_1year.csv"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
EXPECTED_RATIO_COUNT = 64


def fetch_archive() -> bytes:
    """Fetch the public UCI archive and save a local, Git-ignored copy."""
    if ARCHIVE_PATH.exists():
        return ARCHIVE_PATH.read_bytes()

    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "credit-risk-education-project/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    if not content.startswith(b"PK"):
        raise ValueError("UCI response was not a ZIP archive; check the source URL.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_PATH.write_bytes(content)
    return content


def convert_arff_to_csv(arff_text: str, destination: Path) -> tuple[int, list[str]]:
    """Convert the simple numeric ARFF file to a CSV with documented field names."""
    source_attributes: list[str] = []
    data_lines: list[str] = []
    in_data = False
    for raw_line in arff_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if in_data:
            data_lines.append(line)
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        match = re.match(r"@attribute\s+([^\s]+)\s+", line, flags=re.IGNORECASE)
        if match:
            source_attributes.append(match.group(1))

    if not in_data:
        raise ValueError("The downloaded ARFF file has no @data section.")
    if len(source_attributes) != EXPECTED_RATIO_COUNT + 1:
        raise ValueError(
            f"Expected 64 ratios plus one target in the UCI 1st-year file; "
            f"found {len(source_attributes)} attributes: {source_attributes[-3:]}"
        )
    if source_attributes[-1].lower() != "class":
        raise ValueError(f"Expected final ARFF attribute 'class', got {source_attributes[-1]!r}.")

    ratio_names = [f"ratio_{i:02d}" for i in range(1, EXPECTED_RATIO_COUNT + 1)]
    csv_fields = ratio_names + ["defaulted"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_fields)
        row_count = 0
        for line_number, line in enumerate(data_lines, start=1):
            values = next(csv.reader([line]))
            if len(values) != len(csv_fields):
                raise ValueError(
                    f"ARFF data row {line_number} has {len(values)} fields; "
                    f"expected {len(csv_fields)}."
                )
            cleaned = ["" if value.strip() == "?" else value.strip() for value in values]
            if cleaned[-1] not in {"0", "1"}:
                raise ValueError(f"Unexpected target value on ARFF row {line_number}: {cleaned[-1]!r}")
            writer.writerow(cleaned)
            row_count += 1
    return row_count, csv_fields


def main() -> None:
    archive_bytes = fetch_archive()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if "1year.arff" not in archive.namelist():
            raise FileNotFoundError("The UCI archive does not contain 1year.arff.")
        arff_text = archive.read("1year.arff").decode("utf-8-sig")

    count, fields = convert_arff_to_csv(arff_text, CSV_PATH)
    print(f"Created: {CSV_PATH.relative_to(ROOT)}")
    print(f"Rows: {count:,}; ratio columns: {len(fields) - 1}; target: {fields[-1]}")
    print(f"Source archive: {ARCHIVE_PATH.relative_to(ROOT)}")
    print("Source: UCI Polish Companies Bankruptcy, 1st-year case, CC BY 4.0.")


if __name__ == "__main__":
    main()
