"""Generate report tables and figures from the track comparison artifact.

Every number that appears in a chart, table, or the model card originates here from
reports/track_comparison.json. A missing key raises rather than defaulting, because a
default would recreate the defect this script exists to fix: numbers appearing in
published assets that exist in no artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = REPO_ROOT / "reports" / "track_comparison.json"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
TABLES_DIR = REPO_ROOT / "reports" / "tables"


def load_artifact(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    """Load and return the comparison artifact, raising on missing file."""
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def extract_runs(artifact: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    """Extract runs for one dataset, ordered by track name."""
    runs_dict: dict[str, Any] = artifact["runs"]
    runs = [
        run for key, run in runs_dict.items() if key.startswith(f"{dataset}|")
    ]
    runs.sort(key=lambda r: r["track"])
    return runs


def _require(mapping: dict[str, Any], *keys: str) -> Any:
    """Traverse nested keys, raising KeyError on any missing level."""
    current: Any = mapping
    for key in keys:
        current = current[key]
    return current


def _fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def _fmt_interval(interval: dict[str, float], decimals: int = 4) -> str:
    """Format a point estimate with its 95% interval as 'point [low, high]'."""
    point = _require(interval, "point")
    low = _require(interval, "ci_low")
    high = _require(interval, "ci_high")
    return f"{_fmt(point, decimals)} [{_fmt(low, decimals)}, {_fmt(high, decimals)}]"


def generate_markdown_table(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Write a GitHub-flavoured markdown table of track results."""
    header = (
        "| Track | Intervention | Stage | Deployable | ROC-AUC | "
        "Balanced Accuracy | Disparate Impact | Stat. Parity Diff. |"
    )
    separator = "|---|---|---|---|---|---|---|---|"

    rows: list[str] = []
    footnoted: list[str] = []
    for run in runs:
        track = _require(run, "track")
        intervention = _require(run, "intervention")
        stage = _require(run, "stage")
        deployable = "yes" if _require(run, "deployable") else "no"

        roc_auc_iv = _require(run, "intervals", "roc_auc")
        bal_acc_iv = _require(run, "intervals", "balanced_accuracy")
        di_iv = _require(run, "intervals", "disparate_impact")
        spd_iv = _require(run, "intervals", "statistical_parity_difference")

        # A track whose model emits a decision rather than a graded score has an ROC-AUC
        # that equals its balanced accuracy by construction. Left unmarked, a reader
        # compares it with the control's AUC and concludes the intervention destroyed
        # ranking ability, which is not what happened.
        marker = ""
        if run.get("model", {}).get("scores_are_graded") is False:
            marker = " [*]"
            footnoted.append(track)

        row = (
            f"| {track} | {intervention} | {stage} | {deployable} | "
            f"{_fmt_interval(roc_auc_iv)}{marker} | {_fmt_interval(bal_acc_iv)} | "
            f"{_fmt_interval(di_iv)} | {_fmt_interval(spd_iv)} |"
        )
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header, separator, *rows, ""]
    if footnoted:
        lines += [
            f"[*] {', '.join(footnoted)} emits a decision, not a graded score. Its ROC-AUC "
            "therefore equals its balanced accuracy by construction, and its Brier score "
            "is computed on a two-valued score, so neither is comparable with the other "
            "tracks' ranking or calibration quality.",
            "",
        ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_path}")


def generate_tradeoff_plot(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Scatter: disparate impact (x) vs ROC-AUC (y) with error bars."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for run in runs:
        track = _require(run, "track")
        deployable = _require(run, "deployable")
        di_iv = _require(run, "intervals", "disparate_impact")
        auc_iv = _require(run, "intervals", "roc_auc")

        x = _require(di_iv, "point")
        y = _require(auc_iv, "point")
        xerr_low = x - _require(di_iv, "ci_low")
        xerr_high = _require(di_iv, "ci_high") - x
        yerr_low = y - _require(auc_iv, "ci_low")
        yerr_high = _require(auc_iv, "ci_high") - y

        marker = "o" if deployable else "x"
        color = "black" if deployable else "gray"
        ax.errorbar(
            x,
            y,
            xerr=[[xerr_low], [xerr_high]],
            yerr=[[yerr_low], [yerr_high]],
            fmt=marker,
            color=color,
            capsize=3,
            markersize=8,
            label=track,
        )

    # Four-fifths rule line
    ax.axvline(x=0.8, color="red", linestyle="--", linewidth=0.8, label="0.8 threshold")

    ax.set_xlabel("Disparate Impact")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Performance-Fairness Tradeoff by Track")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def generate_group_rates_plot(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Grouped bar chart of per-group selection rate, TPR and FPR per track."""
    metrics = [
        ("Selection Rate", "selection_rate_privileged", "selection_rate_unprivileged"),
        ("TPR", "tpr_privileged", "tpr_unprivileged"),
        ("FPR", "fpr_privileged", "fpr_unprivileged"),
    ]
    n_metrics = len(metrics)
    n_tracks = len(runs)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    # All runs share the same group sizes; read from first run.
    n_priv = int(_require(runs[0], "fairness", "n_privileged"))
    n_unpriv = int(_require(runs[0], "fairness", "n_unprivileged"))

    bar_width = 0.35
    x = np.arange(n_tracks)
    track_labels = [_require(r, "track") for r in runs]

    for ax, (title, priv_key, unpriv_key) in zip(axes, metrics, strict=True):
        priv_vals = [_require(r, "fairness", priv_key) for r in runs]
        unpriv_vals = [_require(r, "fairness", unpriv_key) for r in runs]

        ax.bar(
            x - bar_width / 2,
            priv_vals,
            bar_width,
            label=f"Privileged (n={n_priv})",
            color="dimgray",
        )
        ax.bar(
            x + bar_width / 2,
            unpriv_vals,
            bar_width,
            label=f"Unprivileged (n={n_unpriv})",
            color="silver",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(track_labels)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Per-Group Rates by Track", y=1.02)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def generate_intervals_plot(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Forest plot of fairness metrics with their 95% intervals."""
    fairness_keys = [
        ("disparate_impact", "Disparate Impact"),
        ("statistical_parity_difference", "Stat. Parity Diff."),
        ("equal_opportunity_difference", "Equal Opportunity Diff."),
    ]

    n_panels = len(fairness_keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    track_labels = [_require(r, "track") for r in runs]
    y_pos = np.arange(len(runs))

    for ax, (key, title) in zip(axes, fairness_keys, strict=True):
        points = []
        lows = []
        highs = []
        for run in runs:
            iv = _require(run, "intervals", key)
            points.append(_require(iv, "point"))
            lows.append(_require(iv, "ci_low"))
            highs.append(_require(iv, "ci_high"))

        points_arr = np.array(points)
        lows_arr = np.array(lows)
        highs_arr = np.array(highs)

        xerr = np.array([points_arr - lows_arr, highs_arr - points_arr])
        ax.errorbar(
            points_arr,
            y_pos,
            xerr=xerr,
            fmt="o",
            color="black",
            capsize=4,
            markersize=6,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(track_labels)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Reference line at the legally relevant threshold or zero
        if key == "disparate_impact":
            ax.axvline(x=0.8, color="red", linestyle="--", linewidth=0.8)
        else:
            ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Fairness Metric Intervals by Track", y=1.02)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def datasets_in(artifact: dict[str, Any]) -> list[str]:
    """Dataset names present in the artifact, in stable order.

    Derived from the artifact rather than hardcoded, so adding a dataset to the comparison
    cannot leave its figures silently ungenerated.
    """
    keys = _require(artifact, "runs").keys()
    return sorted({key.split("|", 1)[0] for key in keys})


def main() -> None:
    artifact = load_artifact()
    datasets = datasets_in(artifact)
    if not datasets:
        print("artifact contains no runs", file=sys.stderr)
        sys.exit(1)

    for dataset in datasets:
        runs = extract_runs(artifact, dataset)
        generate_markdown_table(runs, TABLES_DIR / f"track_comparison_{dataset}.md")
        generate_tradeoff_plot(runs, FIGURES_DIR / f"tradeoff_{dataset}.png")
        generate_group_rates_plot(runs, FIGURES_DIR / f"group_rates_{dataset}.png")
        generate_intervals_plot(runs, FIGURES_DIR / f"intervals_{dataset}.png")
        print(f"generated {len(runs)} tracks for {dataset}")


if __name__ == "__main__":
    main()
