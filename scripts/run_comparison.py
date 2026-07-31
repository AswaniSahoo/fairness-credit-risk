"""Run the four comparison tracks on one dataset and record the measured results.

Usage:
    python scripts/run_comparison.py --dataset german_credit --trials 60 --cv-folds 5

T0 runs first because T2 reuses its selected configuration and T3 wraps its fitted model.
Fitted models are written to `artifacts/tracks/` and metrics are merged into
`reports/track_comparison.json`, which is the only source published numbers may come from.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import config  # noqa: E402
from src.data.registry import get_dataset  # noqa: E402
from src.pipelines.tracks import (  # noqa: E402
    TrackRun,
    load_inputs,
    record_run,
    run_baseline_track,
    run_constrained_track,
    run_reweighing_track,
    run_threshold_track,
    save_track_model,
)

logger = logging.getLogger("run_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="german_credit")
    parser.add_argument("--trials", type=int, default=config.N_TRIALS)
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--tracks",
        default="T0,T1,T2,T3",
        help="Comma-separated subset. T2 and T3 require T0 in the same run.",
    )
    return parser.parse_args()


def report(run: TrackRun) -> None:
    """Print one track's measured result."""
    deployable = "deployable" if run.deployable else "NOT DEPLOYABLE"
    threshold = "per-group" if run.threshold is None else f"{run.threshold:.4f}"
    print(f"\n{run.track}  {run.intervention}  [{run.stage}, {deployable}]")
    print(f"  model {run.model['model_type']}, threshold {threshold}, "
          f"selection rate {run.selection_rate:.4f}")
    for name, interval in run.intervals.items():
        print(
            f"  {name:32s} {interval['point']:+.4f} "
            f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
        )
    if run.matched:
        matched = run.matched
        print(f"  at T0's approval rate {matched['reference_selection_rate']:.4f} "
              f"(threshold {matched['threshold']:.4f}):")
        print(f"    disparate impact {matched['fairness']['disparate_impact']:+.4f}, "
              f"ROC-AUC {matched['performance']['roc_auc']:+.4f}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    spec = get_dataset(args.dataset)
    wanted = [name.strip() for name in args.tracks.split(",") if name.strip()]

    if {"T2", "T3"} & set(wanted) and "T0" not in wanted:
        raise SystemExit("T2 reuses T0's configuration and T3 wraps its model; include T0")

    started = time.perf_counter()
    inputs = load_inputs(
        spec,
        seed=args.seed,
        test_size=config.TEST_SIZE,
        calibration_size=config.CALIBRATION_SIZE,
    )
    shared = {"seed": args.seed, "n_bootstrap": args.bootstrap}
    searched = {"n_trials": args.trials, "cv_folds": args.cv_folds}

    baseline_run = baseline_estimator = baseline_search = None
    if "T0" in wanted:
        baseline_run, baseline_estimator, baseline_search, background = run_baseline_track(
            spec, inputs, **shared, **searched
        )
        save_track_model(spec, baseline_run, baseline_estimator, background=background)
        record_run(baseline_run)
        report(baseline_run)

    reference_rate = baseline_run.selection_rate if baseline_run else None

    if "T1" in wanted:
        run, estimator, _, bg = run_reweighing_track(
            spec, inputs, **shared, **searched, reference_selection_rate=reference_rate
        )
        save_track_model(spec, run, estimator, background=bg)
        record_run(run)
        report(run)

    if "T2" in wanted:
        run, mitigator = run_constrained_track(
            spec,
            inputs,
            baseline_search=baseline_search,
            baseline_estimator=baseline_estimator,
            reference_selection_rate=reference_rate,
            **shared,
        )
        save_track_model(spec, run, mitigator)
        record_run(run)
        report(run)

    if "T3" in wanted:
        run, postprocessor = run_threshold_track(
            spec,
            inputs,
            baseline_estimator=baseline_estimator,
            baseline_search=baseline_search,
            **shared,
        )
        save_track_model(spec, run, postprocessor)
        record_run(run)
        report(run)

    print(f"\n{len(wanted)} tracks on {args.dataset} in {time.perf_counter() - started:.1f}s")
    print("recorded in reports/track_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
