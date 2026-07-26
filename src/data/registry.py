"""Per-dataset specifications: column roles, protected attributes, provenance.

Two facts forced this module into existence.

First, the disparity direction reverses between the two datasets. In German Credit women
are approved less often, encoded ``gender`` 0 for female and 1 for male. In the Taiwan
dataset women default *less*, so men are the disadvantaged group, and the encoding is 1
for male and 2 for female. A global ``PRIVILEGED_VALUE = 1`` constant is therefore wrong
on one of the two datasets no matter which value it holds. Privileged and unprivileged
values belong to the dataset, not to the project.

Second, whether a column may be used as a feature is a legal question with a per-column
answer, not a modelling preference. Sex and national origin are prohibited bases under
ECOA and Regulation B, so they are excluded from the feature matrix and retained only for
measurement. Age is different: Regulation B permits age in an empirically derived,
demonstrably and statistically sound credit scoring system, so ``age`` stays a feature and
is still reported on. Encoding those distinctions in data rather than in prose is what
makes the encoder unable to leak them by accident.

Column roles are asserted, not guessed. The integer encodings are verified against
``data/raw/german.doc`` by ``tests/integration/test_german_provenance.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.paths import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class ProtectedAttribute:
    """A protected attribute and the direction of its disparity.

    ``privileged_value`` and ``unprivileged_value`` are the concrete encoded values, so
    the caller of any fairness metric never has to infer them. ``prohibited_basis`` marks
    attributes that may not enter the feature matrix at all.
    """

    column: str
    privileged_value: object
    unprivileged_value: object
    prohibited_basis: bool
    rationale: str
    # Populated where the group is too small to support an interval worth reporting.
    caveat: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the pipeline needs to know about one dataset.

    Roles partition the non-target columns:

    - ``numeric``: genuine continuous or count measurements. Scaled.
    - ``ordinal``: integer codes whose order is established by the codebook. Passed
      through and scaled, so the order is used but no false spacing is introduced beyond
      what the codes already imply.
    - ``nominal``: codes with no order, or with an off-scale category such as "unknown".
      One-hot encoded. Treating these as continuous was finding B8.
    - ``excluded``: columns dropped from features. Either a prohibited basis, a
      deterministic proxy for one, or a derived duplicate of another column.
    """

    name: str
    path: Path
    target: str
    favorable_label: int
    numeric: tuple[str, ...]
    ordinal: tuple[str, ...]
    nominal: tuple[str, ...]
    excluded: tuple[str, ...]
    protected: tuple[ProtectedAttribute, ...]
    primary_protected: str
    provenance: str
    # Category domain per nominal column, taken from the codebook rather than from
    # whatever values a given split happens to contain. Fixing the domain keeps the
    # encoded width identical across tracks and splits, which is a precondition for the
    # comparison being about the intervention and nothing else.
    nominal_categories: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # Undocumented category codes folded before encoding, per roadmap task 4.5.
    category_folds: dict[str, dict[int, int]] = field(default_factory=dict)

    @property
    def unfavorable_label(self) -> int:
        return 1 - self.favorable_label

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Columns entering the model, in role order. Excludes target and protected."""
        return self.numeric + self.ordinal + self.nominal

    @property
    def measurement_columns(self) -> tuple[str, ...]:
        """Columns needed at evaluation time but not for prediction."""
        return tuple(attribute.column for attribute in self.protected)

    def protected_attribute(self, column: str | None = None) -> ProtectedAttribute:
        """Look up a protected attribute, defaulting to the primary one.

        Raises:
            KeyError: If ``column`` is not a registered protected attribute.
        """
        wanted = column or self.primary_protected
        for attribute in self.protected:
            if attribute.column == wanted:
                return attribute
        raise KeyError(f"{wanted!r} is not a protected attribute of dataset {self.name!r}")

    def load(self) -> pd.DataFrame:
        """Read the processed file and validate it against this spec."""
        frame = pd.read_csv(self.path)
        self.validate(frame)
        return frame

    def validate(self, frame: pd.DataFrame) -> None:
        """Check the spec describes ``frame`` exactly.

        Guards against the silent-drift failure where a column is renamed or added and the
        feature matrix quietly changes shape.

        Raises:
            ValueError: On a missing column, an unclassified column, a role overlap, or a
                feature column that is also a protected attribute.
        """
        declared = (
            {self.target}
            | set(self.numeric)
            | set(self.ordinal)
            | set(self.nominal)
            | set(self.excluded)
        )
        present = set(frame.columns)

        if missing := sorted(declared - present):
            raise ValueError(f"{self.name}: declared columns absent from data: {missing}")
        if unclassified := sorted(present - declared):
            raise ValueError(f"{self.name}: data columns with no declared role: {unclassified}")

        roles = [self.numeric, self.ordinal, self.nominal, self.excluded]
        seen: set[str] = set()
        for role in roles:
            if overlap := sorted(seen & set(role)):
                raise ValueError(f"{self.name}: columns declared in two roles: {overlap}")
            seen |= set(role)

        prohibited = {a.column for a in self.protected if a.prohibited_basis}
        if leaked := sorted(prohibited & set(self.feature_columns)):
            raise ValueError(f"{self.name}: prohibited basis used as a feature: {leaked}")

        for column in self.nominal:
            declared = self.nominal_categories.get(column)
            if declared is None:
                raise ValueError(
                    f"{self.name}: nominal column {column!r} has no declared category domain"
                )
            observed = set(self.fold_categories(frame)[column].unique())
            if unexpected := sorted(observed - set(declared)):
                raise ValueError(
                    f"{self.name}: {column!r} contains values outside its declared "
                    f"domain {list(declared)}: {unexpected}"
                )

        for attribute in self.protected:
            if attribute.column not in present:
                raise ValueError(
                    f"{self.name}: protected attribute {attribute.column!r} absent from data"
                )
        self.protected_attribute()  # raises if primary_protected is not registered

    def fold_categories(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the declared undocumented-code folds, leaving other columns untouched.

        Returns a copy, so callers cannot accidentally mutate the loaded frame.
        """
        folded = frame.copy()
        for column, mapping in self.category_folds.items():
            if column in folded.columns:
                folded[column] = folded[column].replace(mapping)
        return folded


GERMAN_CREDIT = DatasetSpec(
    name="german_credit",
    path=PROCESSED_DATA_DIR / "german_credit_numerical_final.csv",
    target="credit_risk",
    # german.doc labels 1 = good, 2 = bad; the processed file shifts these to 0 and 1.
    favorable_label=0,
    numeric=("duration", "amount", "age", "number_credits", "people_liable"),
    # Codebook order survives the alphabetical label encoding for all four of these:
    # employment tenure A71..A75 increases, job skill A171..A174 increases, and
    # installment rate and residence tenure are already ordered integers.
    ordinal=("employment_duration", "installment_rate", "present_residence", "job"),
    # Each of these either has no order at all, or has an off-scale "unknown" category
    # that breaks the scale: savings A65, status A14 (no account), property A124.
    nominal=(
        "status",
        "credit_history",
        "purpose",
        "savings",
        "other_debtors",
        "property",
        "other_installment_plans",
        "housing",
        "telephone",
    ),
    excluded=(
        "gender",
        # Finding B5: code 1 is A92, the only female code present, so this column is a
        # perfect sex proxy. Marital status cannot be separated out, because A92 conflates
        # divorced, separated and married for women while the male codes keep them apart.
        "personal_status_sex",
        "foreign_worker",
        # A coarsening of `age`, which is already a feature. Keeping both would double
        # count and leak the bin edges.
        "age_group",
    ),
    protected=(
        ProtectedAttribute(
            column="gender",
            privileged_value=1,
            unprivileged_value=0,
            prohibited_basis=True,
            rationale=(
                "Sex is a prohibited basis under ECOA and Regulation B. Male 1, female 0. "
                "Label-only favorable rates 0.7232 male against 0.6484 female."
            ),
        ),
        ProtectedAttribute(
            column="foreign_worker",
            privileged_value=1,
            unprivileged_value=0,
            prohibited_basis=True,
            rationale=(
                "National origin is a prohibited basis. A201 yes maps to 0 and A202 no to "
                "1, so the privileged value is 1. Label-only favorable rates 0.8919 for "
                "the 37 non-foreign rows against 0.6926 for the 963 foreign rows."
            ),
            caveat=(
                "Only 37 rows in the privileged group, roughly 7 in a 200-row test split. "
                "Report the counts alongside any rate; intervals here are near useless."
            ),
        ),
    ),
    primary_protected="gender",
    provenance=(
        "UCI Statlog German Credit, Hofmann. data/raw/german.data label-encoded "
        "alphabetically per code. Mapping asserted in "
        "tests/integration/test_german_provenance.py."
    ),
    nominal_categories={
        # Domains follow german.doc. `purpose` has ten codes because A47 (vacation) is
        # marked as non-existent in the codebook and never occurs.
        "status": (0, 1, 2, 3),
        "credit_history": (0, 1, 2, 3, 4),
        "purpose": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        "savings": (0, 1, 2, 3, 4),
        "other_debtors": (0, 1, 2),
        "property": (0, 1, 2, 3),
        "other_installment_plans": (0, 1, 2),
        "housing": (0, 1, 2),
        "telephone": (0, 1),
    },
)


TAIWAN_CREDIT = DatasetSpec(
    name="taiwan_credit",
    path=PROCESSED_DATA_DIR / "taiwan_credit_default.csv",
    target="default payment next month",
    favorable_label=0,
    numeric=(
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ),
    # PAY_* are months of payment delay, ordered upward from -2. The -2 and 0 codes are
    # undocumented in the UCI description but sit at the "not delinquent" end of the
    # scale, so the order is retained and the ambiguity recorded rather than folded away.
    ordinal=("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    nominal=("EDUCATION", "MARRIAGE"),
    excluded=("SEX",),
    protected=(
        ProtectedAttribute(
            column="SEX",
            # Reversed relative to German Credit: women default less here, so men are the
            # disadvantaged group. This is the case that a global constant gets wrong.
            privileged_value=2,
            unprivileged_value=1,
            prohibited_basis=True,
            rationale=(
                "Sex is a prohibited basis. Encoded 1 male, 2 female. Label-only "
                "disparate impact female over male is 1.0447 with SPD +0.0339, so the "
                "unprivileged group is men."
            ),
        ),
    ),
    primary_protected="SEX",
    provenance=(
        "UCI dataset 350, default of credit card clients, Yeh and Lien. Converted from "
        "the published .xls by scripts/convert_taiwan_dataset.py; ID column dropped."
    ),
    category_folds={
        # UCI documents EDUCATION 1-4 and MARRIAGE 1-3. The codes below appear in the
        # data with no published meaning, so they are folded into the existing "other"
        # bucket rather than silently one-hot encoded as if they were real categories.
        "EDUCATION": {0: 4, 5: 4, 6: 4},
        "MARRIAGE": {0: 3},
    },
    # Declared after folding, which is the order `validate` checks them in.
    nominal_categories={
        "EDUCATION": (1, 2, 3, 4),
        "MARRIAGE": (1, 2, 3),
    },
)


DATASETS: dict[str, DatasetSpec] = {
    GERMAN_CREDIT.name: GERMAN_CREDIT,
    TAIWAN_CREDIT.name: TAIWAN_CREDIT,
}


def get_dataset(name: str) -> DatasetSpec:
    """Look up a dataset spec by name.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}") from None
