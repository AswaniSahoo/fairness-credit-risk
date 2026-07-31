"""Add the SHAP background sample to track artifacts fitted before the field existed.

The committed T0 models predate the explanation work, so ``Predictor`` refuses to build an
explainer against them and adverse-action reason codes are unavailable on a clean checkout.

Re-running the tracks would fix that, but it would also re-fit the models, and every
published number traces to those exact artifacts. This script computes only what is missing.
The model is loaded and written back unchanged; the background is derived from the committed
split artifact and the encoder already inside the model, so the result is identical to what a
fresh run would have stored, and no metric can move.

The same computation yields the global feature importance the run record was designed to
carry, so it is written into ``reports/track_comparison.json`` in the same pass. Tracks whose
artifact is not an encoder-plus-classifier pipeline are skipped: T2 is a randomised ensemble
whose per-feature attribution is not defined for the mixture, and T3 wraps T0's model
unchanged. Both already record a null importance by design.

Usage:
    python scripts/backfill_shap_background.py [--dataset german_credit] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import config  # noqa: E402
from src.data.registry import DATASETS, DatasetSpec, get_dataset  # noqa: E402
from src.paths import ARTIFACTS_DIR  # noqa: E402
from src.pipelines.tracks import (  # noqa: E402
    COMPARISON_PATH,
    compute_shap_importance,
    load_inputs,
)

logger = logging.getLogger("backfill_shap_background")

BACKFILLABLE_TRACKS = ("T0", "T1")


def _is_encoder_pipeline(model: Any) -> bool:
    """True if the artifact is the fitted encoder-plus-classifier pipeline."""
    return (
        isinstance(model, Pipeline)
        and "encoder" in model.named_steps
        and "classifier" in model.named_steps
    )


def backfill(spec: DatasetSpec, track: str, *, dry_run: bool) -> bool:
    """Add background and feature importance for one (dataset, track). True if written.

    The seed comes from the run record rather than from an argument, so the split rebuilt
    here is the one the model was fitted under and not whatever the caller passed.
    """
    path = ARTIFACTS_DIR / "tracks" / f"{spec.name}_{track}.joblib"
    if not path.exists():
        logger.info("%s %s: no artifact at %s, skipped", spec.name, track, path)
        return False

    artifact = joblib.load(path)
    model = artifact["model"]

    if not _is_encoder_pipeline(model):
        logger.info(
            "%s %s: artifact is %s, not an encoder pipeline, skipped",
            spec.name, track, type(model).__name__,
        )
        return False

    if "background" in artifact:
        logger.info("%s %s: background already present, skipped", spec.name, track)
        return False

    run = artifact["run"]
    inputs = load_inputs(
        spec,
        seed=run["seed"],
        test_size=config.TEST_SIZE,
        calibration_size=config.CALIBRATION_SIZE,
    )

    # The background is only equivalent to a fresh run's if it comes from the same training
    # rows the model was fitted on. The fingerprint identifies the data the split was drawn
    # from and the sizes identify the blocks, so together they settle it. Refuse rather than
    # write a background derived from a different split.
    if inputs.split.fingerprint != run["split_fingerprint"]:
        raise ValueError(
            f"{spec.name} {track}: rebuilt split fingerprint {inputs.split.fingerprint} does "
            f"not match the run record's {run['split_fingerprint']}. The data changed since "
            "the model was fitted; re-run the track instead of backfilling."
        )
    if inputs.split.sizes != run["split_sizes"]:
        raise ValueError(
            f"{spec.name} {track}: rebuilt split sizes {inputs.split.sizes} do not match the "
            f"run record's {run['split_sizes']}."
        )

    importance, background = compute_shap_importance(model, inputs.X_train, run["seed"])

    logger.info(
        "%s %s: %d training rows, background %s, top feature %s (mean |SHAP| %.4f)",
        spec.name, track, len(inputs.X_train), background.shape,
        importance[0]["feature"], importance[0]["mean_abs_shap"],
    )

    if dry_run:
        return False

    artifact["background"] = background
    joblib.dump(artifact, path)

    if COMPARISON_PATH.exists():
        comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
        key = f"{spec.name}|{track}"
        run = comparison.get("runs", {}).get(key)
        if run is None:
            logger.warning("%s: no run record in %s, importance not recorded",
                           key, COMPARISON_PATH)
        else:
            run["feature_importance"] = importance
            COMPARISON_PATH.write_text(
                json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
            )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                        help="Backfill one dataset. Default: every registered dataset.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report without writing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    names = [args.dataset] if args.dataset else sorted(DATASETS)
    written = 0
    for name in names:
        spec = get_dataset(name)
        for track in BACKFILLABLE_TRACKS:
            if backfill(spec, track, dry_run=args.dry_run):
                written += 1

    logger.info("%d artifact(s) updated", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
