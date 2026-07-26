"""The shared three-way split: train, calibration, test.

Two problems are solved here.

The first is finding B4. Group thresholds were previously fitted on probabilities from the
same rows the model was trained on, where a tree ensemble is close to memorising its
training set. Any threshold chosen there is tuned to noise the model has already absorbed
and does not transfer. Post-processing needs its own held-out block, so the split is
three-way: models fit on train, post-processing fits on calibration, and every reported
number comes from test.

The second is comparability. Tracks are only comparable if nothing but the intervention
differs, so the split is produced once, persisted with a fingerprint of the data it was
derived from, and reloaded by every track. A track that regenerates its own split would
silently be answering a different question.

Stratification is joint on the target and the primary protected attribute. Stratifying on
the target alone leaves the protected group counts to chance, which on German Credit means
the number of women in the test set varies by roughly plus or minus 8 between seeds, and
every fairness metric moves with it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from src.data.registry import DatasetSpec
from src.paths import ARTIFACTS_DIR


@dataclass(frozen=True)
class DataSplit:
    """Row positions for each block, plus the provenance needed to trust them.

    Indices are positional, not label-based, so they stay valid regardless of the frame's
    index. ``fingerprint`` identifies the exact data the split was drawn from.
    """

    dataset: str
    seed: int
    train: NDArray[np.int_]
    calibration: NDArray[np.int_]
    test: NDArray[np.int_]
    fingerprint: str
    stratified_on: tuple[str, ...]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "calibration": int(self.calibration.size),
            "test": int(self.test.size),
        }

    def frame(self, frame: pd.DataFrame, block: str) -> pd.DataFrame:
        """Return one block of ``frame``.

        Raises:
            ValueError: If ``block`` is not train, calibration or test, or if ``frame`` is
                not the data this split was derived from.
        """
        if block not in {"train", "calibration", "test"}:
            raise ValueError(f"unknown block {block!r}")
        if fingerprint(frame) != self.fingerprint:
            raise ValueError(
                f"frame does not match the split fingerprint for {self.dataset}; "
                "the data changed after the split was made"
            )
        return frame.iloc[getattr(self, block)]


def fingerprint(frame: pd.DataFrame) -> str:
    """Content hash of a dataframe, stable across processes.

    ``pandas.util.hash_pandas_object`` is used rather than ``hash`` so the value does not
    depend on Python's per-process string hash seed.
    """
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    digest = hashlib.sha256(row_hashes.tobytes())
    digest.update(",".join(map(str, frame.columns)).encode())
    return digest.hexdigest()[:16]


def split_path(spec: DatasetSpec, seed: int) -> Path:
    return ARTIFACTS_DIR / "splits" / f"{spec.name}_seed{seed}.joblib"


def make_split(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    *,
    test_size: float,
    calibration_size: float,
    seed: int,
) -> DataSplit:
    """Draw the three-way split.

    ``test_size`` and ``calibration_size`` are fractions of the whole dataset, so 0.2 and
    0.2 give 60/20/20. The calibration block is carved out of the non-test remainder at the
    corresponding relative rate.

    Raises:
        ValueError: If the requested fractions do not leave a training block.
    """
    if not 0 < test_size < 1 or not 0 <= calibration_size < 1:
        raise ValueError("test_size and calibration_size must be fractions of the dataset")
    if test_size + calibration_size >= 1:
        raise ValueError(
            f"test_size {test_size} plus calibration_size {calibration_size} leaves no "
            "training rows"
        )

    protected = spec.protected_attribute()
    strata = (
        frame[spec.target].astype(str) + "|" + frame[protected.column].astype(str)
    ).to_numpy()
    positions = np.arange(len(frame))

    remainder, test = train_test_split(
        positions, test_size=test_size, stratify=strata, random_state=seed
    )
    if calibration_size == 0:
        train, calibration = remainder, np.array([], dtype=int)
    else:
        train, calibration = train_test_split(
            remainder,
            test_size=calibration_size / (1.0 - test_size),
            stratify=strata[remainder],
            random_state=seed,
        )

    return DataSplit(
        dataset=spec.name,
        seed=seed,
        train=np.sort(train),
        calibration=np.sort(calibration),
        test=np.sort(test),
        fingerprint=fingerprint(frame),
        stratified_on=(spec.target, protected.column),
    )


def save_split(split: DataSplit, spec: DatasetSpec) -> Path:
    """Persist a split so every track loads the identical one."""
    path = split_path(spec, split.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(split, path)
    return path


def load_split(spec: DatasetSpec, seed: int) -> DataSplit:
    """Load a persisted split.

    Raises:
        FileNotFoundError: If no split has been generated for this dataset and seed.
    """
    path = split_path(spec, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"no split artifact at {path}; generate it before running any track"
        )
    return joblib.load(path)


def get_or_create_split(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    *,
    test_size: float,
    calibration_size: float,
    seed: int,
) -> DataSplit:
    """Load the persisted split, regenerating it only if the data has changed.

    Regenerating on a fingerprint mismatch is deliberate: a stale split against changed
    data would quietly mix blocks, which is worse than losing comparability with earlier
    runs that are in any case no longer valid.
    """
    path = split_path(spec, seed)
    if path.exists():
        existing: DataSplit = joblib.load(path)
        if existing.fingerprint == fingerprint(frame):
            return existing

    split = make_split(
        spec,
        frame,
        test_size=test_size,
        calibration_size=calibration_size,
        seed=seed,
    )
    save_split(split, spec)
    return split
