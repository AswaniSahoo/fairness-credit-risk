"""Recover the code-to-integer mapping of the processed German Credit CSV.

The processed file `german_credit_numerical_final.csv` holds integers, but nothing in the
repository records which UCI code each integer came from. That mapping decides whether a
column may be treated as ordinal, so it has to be established from the raw file rather
than assumed. Run this whenever the processed file is regenerated.

Emits one line per column: the raw code order, the integer it maps to, and whether the
mapping is a bijection. Exits non-zero if any column fails to map one-to-one, which would
mean the processed file is not a faithful encoding of `german.data`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402

RAW_COLUMNS = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk",
]


def main() -> int:
    raw = pd.read_csv(RAW_DATA_DIR / "german.data", sep=" ", header=None, names=RAW_COLUMNS)
    processed = pd.read_csv(PROCESSED_DATA_DIR / "german_credit_numerical_final.csv")

    if len(raw) != len(processed):
        print(f"FAIL row count: raw {len(raw)} vs processed {len(processed)}")
        return 1

    failures = 0
    for column in RAW_COLUMNS:
        if column not in processed.columns:
            print(f"{column:24s} ABSENT from processed file")
            failures += 1
            continue

        pairs = pd.DataFrame({"raw": raw[column], "encoded": processed[column]})
        mapping = pairs.groupby("raw")["encoded"].unique()
        collisions = pairs.groupby("encoded")["raw"].nunique()

        one_to_one = all(len(v) == 1 for v in mapping) and (collisions == 1).all()
        if len(mapping) > 12:
            # Genuine numerics pass through unchanged; printing 900 identity mappings
            # would bury the categorical codes that actually need inspecting.
            identity = bool((pairs["raw"] == pairs["encoded"]).all())
            rendered = f"{len(mapping)} distinct values, identity={identity}"
        else:
            rendered = ", ".join(f"{k}->{v[0]}" for k, v in mapping.items())
        print(f"{column:24s} {'OK ' if one_to_one else 'BAD'} {rendered}")
        if not one_to_one:
            failures += 1

    # gender is derived, not present in the raw file; confirm it against attribute 9.
    female_codes = {"A92", "A95"}
    expected_gender = (~raw["personal_status_sex"].isin(female_codes)).astype(int)
    gender_matches = int((expected_gender == processed["gender"]).sum())
    print(f"{'gender (derived)':24s} {'OK ' if gender_matches == len(raw) else 'BAD'} "
          f"male=1 from A91/A93/A94, matches {gender_matches}/{len(raw)}")
    if gender_matches != len(raw):
        failures += 1

    print(f"\n{failures} column(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
