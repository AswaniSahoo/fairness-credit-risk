"""Tests for the report asset generator.

Validates that the generated assets faithfully reproduce the artifact's values and that
missing keys surface as errors rather than defaulting to a literal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.generate_report_assets import (
    extract_runs,
    generate_markdown_table,
    load_artifact,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ARTIFACT_PATH = REPO_ROOT / "reports" / "track_comparison.json"


@pytest.fixture()
def artifact() -> dict[str, Any]:
    return load_artifact(ARTIFACT_PATH)


@pytest.fixture()
def german_runs(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return extract_runs(artifact, "german_credit")


def test_missing_key_in_run_raises(tmp_path: Path) -> None:
    """A run missing a required key must raise KeyError, not produce a default."""
    broken_run = {
        "track": "T0",
        "intervention": "none",
        "stage": "control",
        "deployable": True,
        # intervals key is missing entirely
    }
    artifact = {"runs": {"german_credit|T0": broken_run}, "schema_version": 2}
    artifact_path = tmp_path / "broken.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_artifact(artifact_path)
    runs = extract_runs(loaded, "german_credit")

    with pytest.raises(KeyError):
        generate_markdown_table(runs, tmp_path / "out.md")


def test_missing_nested_interval_key_raises(tmp_path: Path) -> None:
    """A run with intervals but missing 'point' raises."""
    broken_run = {
        "track": "T0",
        "intervention": "none",
        "stage": "control",
        "deployable": True,
        "intervals": {
            "roc_auc": {"ci_low": 0.7, "ci_high": 0.9},  # missing 'point'
            "balanced_accuracy": {"point": 0.7, "ci_low": 0.6, "ci_high": 0.8},
            "disparate_impact": {"point": 0.7, "ci_low": 0.5, "ci_high": 0.9},
            "statistical_parity_difference": {
                "point": -0.1,
                "ci_low": -0.3,
                "ci_high": 0.0,
            },
        },
    }
    artifact = {"runs": {"german_credit|T0": broken_run}, "schema_version": 2}
    artifact_path = tmp_path / "broken2.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_artifact(artifact_path)
    runs = extract_runs(loaded, "german_credit")

    with pytest.raises(KeyError):
        generate_markdown_table(runs, tmp_path / "out.md")


def test_every_track_present_in_table(
    german_runs: list[dict[str, Any]], tmp_path: Path
) -> None:
    """Every track in the artifact appears as a row in the generated table."""
    output = tmp_path / "track_comparison.md"
    generate_markdown_table(german_runs, output)

    content = output.read_text(encoding="utf-8")
    for run in german_runs:
        track = run["track"]
        assert f"| {track} |" in content, f"track {track} missing from table"


def test_table_numbers_match_artifact_exactly(
    german_runs: list[dict[str, Any]], tmp_path: Path
) -> None:
    """Point estimates in the table match the artifact's values to four decimals."""
    output = tmp_path / "track_comparison.md"
    generate_markdown_table(german_runs, output)

    content = output.read_text(encoding="utf-8")
    # Match only data rows (T0, T1, ...), not the header row starting with "| Track"
    lines = [
        line
        for line in content.strip().split("\n")
        if line.startswith("| T") and not line.startswith("| Track")
    ]

    assert len(lines) == len(german_runs)

    for run, line in zip(german_runs, lines, strict=True):
        # Verify the ROC-AUC point estimate appears in the row
        roc_point = run["intervals"]["roc_auc"]["point"]
        assert f"{roc_point:.4f}" in line, (
            f"ROC-AUC {roc_point:.4f} not found in row for {run['track']}"
        )

        # Verify the disparate impact point estimate appears in the row
        di_point = run["intervals"]["disparate_impact"]["point"]
        assert f"{di_point:.4f}" in line, (
            f"DI {di_point:.4f} not found in row for {run['track']}"
        )

        # Verify balanced accuracy
        ba_point = run["intervals"]["balanced_accuracy"]["point"]
        assert f"{ba_point:.4f}" in line, (
            f"BA {ba_point:.4f} not found in row for {run['track']}"
        )

        # Verify statistical parity difference
        spd_point = run["intervals"]["statistical_parity_difference"]["point"]
        assert f"{spd_point:.4f}" in line, (
            f"SPD {spd_point:.4f} not found in row for {run['track']}"
        )


def test_interval_bounds_in_table(
    german_runs: list[dict[str, Any]], tmp_path: Path
) -> None:
    """Interval bounds appear in the generated markdown alongside point estimates."""
    output = tmp_path / "track_comparison.md"
    generate_markdown_table(german_runs, output)

    content = output.read_text(encoding="utf-8")
    lines = [
        line
        for line in content.strip().split("\n")
        if line.startswith("| T") and not line.startswith("| Track")
    ]

    for run, line in zip(german_runs, lines, strict=True):
        roc_low = run["intervals"]["roc_auc"]["ci_low"]
        roc_high = run["intervals"]["roc_auc"]["ci_high"]
        assert f"{roc_low:.4f}" in line
        assert f"{roc_high:.4f}" in line
