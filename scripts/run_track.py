"""Run one comparison track and record its measured result.

Usage:
    python scripts/run_track.py --dataset german_credit --track T0 --trials 60

Writes the fitted model to `artifacts/tracks/` and merges the metrics into
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
from src.pipelines.tracks import record_run, run_baseline_track  # noqa: E402

TRACKS = {"T0": run_baseline_track}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="german_credit")
    parser.add_argument("--track", default="T0", choices=sorted(TRACKS))
    parser.add_argument("--trials", type=int, default=config.N_TRIALS)
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    spec = get_dataset(args.dataset)

    started = time.perf_counter()
    run, _, split = TRACKS[args.track](
        spec,
        seed=args.seed,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
        test_size=config.TEST_SIZE,
        calibration_size=config.CALIBRATION_SIZE,
        n_bootstrap=args.bootstrap,
    )
    elapsed = time.perf_counter() - started

    path = record_run(run)
    print(f"\n{args.dataset} {args.track}  ({elapsed:.1f}s, {args.trials} trials)")
    print(f"split: {split.sizes}, encoded features: {run.n_encoded_features}")
    print(f"model: {run.model['model_type']}  cv {run.model['scoring']} "
          f"{run.model['cv_score']:.4f}")

    print("\ntest metrics, 95% bootstrap interval")
    for name, interval in run.intervals.items():
        print(
            f"  {name:32s} {interval['point']:+.4f} "
            f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
        )

    print("\ngroup rates on test")
    for key in (
        "selection_rate_privileged", "selection_rate_unprivileged",
        "tpr_privileged", "tpr_unprivileged",
        "fpr_privileged", "fpr_unprivileged",
        "n_privileged", "n_unprivileged",
    ):
        print(f"  {key:32s} {run.fairness[key]:.4f}")

    print(f"\nrecorded in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
