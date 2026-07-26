"""Convert the Taiwan credit card default source workbook to CSV.

Source: UCI "Default of Credit Card Clients", https://archive.ics.uci.edu/dataset/350
Downloaded artifact: data/raw/taiwan_credit/default of credit card clients.xls

The workbook has a two-row header: row 0 holds the anonymous X1..X23 labels and row 1
holds the real column names. Only the second row is meaningful, so the first is skipped.

Run once. Asserts row count and schema so a silently different source file fails loudly
rather than producing a quietly wrong dataset.
"""

import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402

SOURCE = RAW_DATA_DIR / "taiwan_credit" / "default of credit card clients.xls"
DESTINATION = PROCESSED_DATA_DIR / "taiwan_credit_default.csv"

EXPECTED_ROWS = 30_000
TARGET = "default payment next month"


def main() -> None:
    frame = pd.read_excel(SOURCE, header=1)

    # The ID column is a row identifier and must never become a feature.
    if "ID" in frame.columns:
        frame = frame.drop(columns=["ID"])

    assert len(frame) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(frame)}"
    assert TARGET in frame.columns, f"target column missing; got {list(frame.columns)}"
    assert frame[TARGET].isin([0, 1]).all(), "target is not binary 0/1"
    assert not frame.isnull().any().any(), "unexpected missing values in source"

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DESTINATION, index=False)

    print("wrote:", DESTINATION)
    print("rows :", len(frame), "| cols:", frame.shape[1])
    print("target balance:", frame[TARGET].value_counts().to_dict())


if __name__ == "__main__":
    main()
