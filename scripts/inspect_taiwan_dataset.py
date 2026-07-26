"""Inspect the converted Taiwan dataset: schema, encodings, group balance, power.

Establishes the real column roles rather than trusting the UCI prose description, and
quantifies whether the larger sample actually enables method ranking.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PROCESSED_DATA_DIR  # noqa: E402

TARGET = "default payment next month"
df = pd.read_csv(PROCESSED_DATA_DIR / "taiwan_credit_default.csv")

print("shape:", df.shape)
print("\ncolumns:", list(df.columns))

print("\n--- candidate categorical / demographic encodings ---")
for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    counts = df[col].value_counts().sort_index()
    print(f"{col:10} {dict(counts)}")

print("\n--- PAY_0..PAY_6 repayment status codes ---")
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
print("distinct values across PAY_*:", sorted(pd.unique(df[pay_cols].values.ravel())))

print("\n--- numeric ranges ---")
for col in ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]:
    s = df[col]
    print(f"{col:12} min={s.min():>10.0f} max={s.max():>10.0f} mean={s.mean():>10.1f}")

print("\n--- target ---")
print("balance:", df[TARGET].value_counts().to_dict())
print(f"default rate: {df[TARGET].mean():.4f}")

print("\n--- protected attribute: SEX (1 = male, 2 = female per UCI docs) ---")
for sex, group in df.groupby("SEX"):
    fav = (group[TARGET] == 0).mean()   # favorable outcome = no default
    print(f"SEX={sex}  n={len(group):>6}  favorable_label_rate={fav:.4f}")

rate_male = (df.loc[df.SEX == 1, TARGET] == 0).mean()
rate_female = (df.loc[df.SEX == 2, TARGET] == 0).mean()
print(f"label-only disparate impact (female/male): {rate_female / rate_male:.4f}")
print(f"label-only statistical parity difference : {rate_female - rate_male:+.4f}")

print("\n--- statistical power: minority group size in a 20% test split ---")
n_test = int(0.20 * len(df))
share_female = (df.SEX == 2).mean()
n_female_test = int(n_test * share_female)
print(f"test rows           : {n_test}")
print(f"female share        : {share_female:.4f}")
print(f"female rows in test : {n_female_test}")
print(f"selection-rate step : {1 / n_female_test:.5f} per single prediction flip")

# German Credit comparison: 62 women in a 200-row test set.
german_step = 1 / 62
print(f"\nGerman Credit step  : {german_step:.5f} per flip (62 women)")
print(f"improvement factor  : {german_step / (1 / n_female_test):.1f}x finer resolution")

se_german = np.sqrt(0.5 * 0.5 / 62)
se_taiwan = np.sqrt(0.5 * 0.5 / n_female_test)
print(f"\nworst-case SE of female selection rate: German {se_german:.4f} -> Taiwan {se_taiwan:.4f}")
print(f"CI width shrinks by ~{se_german / se_taiwan:.1f}x")
