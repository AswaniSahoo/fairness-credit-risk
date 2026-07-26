"""Single inference path for the served credit risk model.

This module is the only code that may call the model at prediction time. The API, the
Streamlit demo, and all tests route through ``Predictor.predict``. That invariant is
enforced by tests/unit/serving/test_predictor.py and tests/integration/test_api.py.

SECURITY: This module performs no authentication. Any caller that can import it can score
an applicant. The service that exposes it over the network must provide an auth layer;
see api/main.py for the documented absence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.registry import GERMAN_CREDIT, DatasetSpec
from src.paths import ARTIFACTS_DIR
from src.preprocessing.features import extract_features
from src.training.search import positive_class_probabilities

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT = ARTIFACTS_DIR / "tracks" / "german_credit_T0.joblib"


class Predictor:
    """Loads a track artifact and scores applicants against a single global threshold.

    The threshold is read from the run record at load time. No group-keyed threshold
    exists; every applicant is scored identically regardless of protected attributes.

    Raises:
        FileNotFoundError: If the artifact path does not exist.
        KeyError: If the artifact lacks expected keys.
    """

    def __init__(self, artifact_path: Path | None = None, spec: DatasetSpec | None = None):
        self._path = artifact_path or DEFAULT_ARTIFACT
        self._spec = spec or GERMAN_CREDIT

        if not self._path.exists():
            raise FileNotFoundError(
                f"Track artifact not found at {self._path}. Run the pipeline first."
            )

        artifact = joblib.load(self._path)
        self._model = artifact["model"]
        self._run: dict[str, Any] = artifact["run"]
        self._threshold: float = self._run["threshold"]

        logger.info(
            "loaded %s track %s, threshold %.4f",
            self._run["dataset"],
            self._run["track"],
            self._threshold,
        )

    @property
    def run(self) -> dict[str, Any]:
        return self._run

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def predict(self, applicant: dict[str, Any]) -> dict[str, Any]:
        """Score one applicant and return a decision dict.

        Args:
            applicant: Field names matching ``spec.feature_columns``, values in codebook
                domain. Protected attributes must not be present.

        Returns:
            Dict with probability_of_default, decision, threshold, track, dataset.

        Raises:
            KeyError: If a required feature column is missing from ``applicant``.
            ValueError: If a prohibited-basis attribute is supplied, or if the model
                rejects the input (out-of-domain category, for example).
        """
        prohibited = sorted(
            column for column in self._spec.prohibited_inputs if column in applicant
        )
        if prohibited:
            # extract_features would drop these silently, which would let a caller believe
            # the model considers them. Refusing is the only way the contract stays legible:
            # this service does not accept a prohibited basis, it does not merely ignore one.
            raise ValueError(
                f"prohibited-basis attributes are not accepted as input: {prohibited}"
            )

        frame = pd.DataFrame([applicant])
        features = extract_features(self._spec, frame)

        scores = positive_class_probabilities(self._model, features)
        probability = float(scores[0])

        # Unfavorable label (default) is the positive class; score >= threshold means
        # the applicant is predicted to default.
        decision = "decline" if probability >= self._threshold else "approve"

        return {
            "probability_of_default": probability,
            "decision": decision,
            "threshold": self._threshold,
            "track": self._run["track"],
            "dataset": self._run["dataset"],
        }
