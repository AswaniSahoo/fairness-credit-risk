"""The README may not contain a number that is absent from the generated assets.

The project previously published a comparison chart whose values appear in no artifact at
all, which is finding I1. The rule that replaced it is that every published figure is
generated from `reports/track_comparison.json`. A rule stated in prose is not a rule, so it
is enforced here: the README's result tables must match the generated table files verbatim,
and the headline numbers quoted in the surrounding text must exist in the artifact.
"""

from __future__ import annotations

import json
import re

import pytest

from src.paths import PROJECT_ROOT, REPORTS_DIR

pytestmark = pytest.mark.integration

README = PROJECT_ROOT / "README.md"
TABLES_DIR = REPORTS_DIR / "tables"
DATASETS = ("german_credit", "taiwan_credit")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((REPORTS_DIR / "track_comparison.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("dataset", DATASETS)
def test_readme_tables_match_the_generated_tables(readme: str, dataset: str):
    """Every data row of each generated table must appear in the README verbatim.

    Regenerating the assets after a new run therefore forces the README to be updated, and a
    hand-edited cell fails here.
    """
    generated = (TABLES_DIR / f"track_comparison_{dataset}.md").read_text(encoding="utf-8")
    data_rows = [
        line.strip()
        for line in generated.splitlines()
        if line.startswith("| T") and "Track" not in line
    ]

    assert len(data_rows) == 4, f"expected four tracks for {dataset}"
    for row in data_rows:
        assert row in readme, f"{dataset} row absent from README: {row}"


def test_every_number_in_a_readme_table_row_exists_in_the_artifact(readme: str, artifact: dict):
    """Catches a fabricated row that happens not to be in a generated file either."""
    recorded: set[str] = set()
    for run in artifact["runs"].values():
        for interval in run["intervals"].values():
            recorded.update(f"{interval[key]:.4f}" for key in ("point", "ci_low", "ci_high"))

    table_rows = [line for line in readme.splitlines() if line.startswith("| T0 |")]
    assert table_rows, "no result rows found in README"

    for row in table_rows:
        for number in re.findall(r"-?\d+\.\d{4}", row):
            assert number in recorded, f"{number} appears in the README but in no artifact"


@pytest.mark.parametrize(
    ("claim", "dataset", "track", "path"),
    [
        # Numbers quoted in the README's prose, each tied to the key it came from.
        ("0.7263", "german_credit", "T0", ("fairness", "disparate_impact")),
        ("-0.1884", "german_credit", "T0", ("fairness", "statistical_parity_difference")),
        ("-0.0143", "german_credit", "T2", ("model", "train_statistical_parity_difference")),
        ("0.9767", "taiwan_credit", "T0", ("fairness", "disparate_impact")),
        ("0.6774", "german_credit", "T1", ("fairness", "disparate_impact")),
    ],
)
def test_prose_claims_trace_to_the_artifact(
    readme: str, artifact: dict, claim: str, dataset: str, track: str, path: tuple[str, ...]
):
    run = artifact["runs"][f"{dataset}|{track}"]
    value = run
    for key in path:
        value = value[key]

    assert f"{float(value):.4f}".rstrip("0").rstrip(".") in claim or claim in f"{float(value):.4f}"
    assert claim.lstrip("-") in readme or claim in readme


def test_readme_group_counts_match_the_recorded_split(readme: str, artifact: dict):
    """Rates on 62 people are not interchangeable with rates on 3,622, so the counts are
    quoted in the README and must match the runs they describe.

    Which side is which differs by dataset: German Credit's unprivileged group is women, and
    Taiwan's is men, because women there default less. Reading the same field for both would
    silently mislabel one of them.
    """
    german = artifact["runs"]["german_credit|T0"]["fairness"]
    taiwan = artifact["runs"]["taiwan_credit|T0"]["fairness"]

    assert f"{int(german['n_unprivileged'])} women" in readme
    assert f"{int(taiwan['n_privileged']):,} women" in readme
    assert f"{int(taiwan['n_unprivileged']):,} men" in readme


def test_dataset_level_claims_recompute_from_the_data(readme: str):
    """The README's dataset-bias figures are not model results, so they cannot come from the
    comparison artifact. They are recomputed from the processed CSV instead, so the rule that
    no published number is typed by hand holds for them too.

    The distinction is the point of finding B1: 0.8966 is a property of the 1,000 recorded
    outcomes and the model's 0.7263 is a different quantity entirely.
    """
    import pandas as pd

    from src.data.registry import GERMAN_CREDIT
    from src.evaluation.group_fairness import group_fairness

    frame = pd.read_csv(GERMAN_CREDIT.path)
    sex = next(a for a in GERMAN_CREDIT.protected if a.column == "gender")
    labels = frame[GERMAN_CREDIT.target].to_numpy()

    # Labels passed as decisions on purpose: this measures the data, not a model.
    dataset_bias = group_fairness(
        labels,
        labels,
        frame["gender"].to_numpy(),
        privileged_value=sex.privileged_value,
        unprivileged_value=sex.unprivileged_value,
        favorable_label=GERMAN_CREDIT.favorable_label,
    )

    for value in (
        dataset_bias.disparate_impact,
        dataset_bias.privileged.selection_rate,
        dataset_bias.unprivileged.selection_rate,
    ):
        assert f"{value:.4f}" in readme, (
            f"{value:.4f} is the recomputed dataset-level figure but is absent from the README"
        )
