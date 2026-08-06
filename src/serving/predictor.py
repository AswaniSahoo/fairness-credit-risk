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
import numpy as np
import pandas as pd

from src.data.registry import GERMAN_CREDIT, DatasetSpec
from src.evaluation.explanations import (
    ReasonCode,
    build_explainer,
    explain_single,
)
from src.paths import ARTIFACTS_DIR
from src.preprocessing.features import (
    encoded_feature_names,
    extract_features,
    indicator_feature_names,
)
from src.training.search import positive_class_probabilities

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT = ARTIFACTS_DIR / "tracks" / "german_credit_T0.joblib"


class Predictor:
    """Loads a track artifact and scores applicants against a single global threshold.

    The threshold is read from the run record at load time. No group-keyed threshold
    exists; every applicant is scored identically regardless of protected attributes.

    When ``include_reasons=True`` is passed to ``predict`` and the decision is "decline",
    the response includes SHAP-based reason codes for the adverse-action notice required
    by Regulation B (12 CFR 1002.9).

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
        self._background = artifact.get("background")
        self._explainer: Any | None = None

        # Serve the threshold the run record chose, not the 0.5 it was scored at for
        # comparison. A cost-minimising point that is computed and then not applied would be
        # decoration; the whole argument for choosing an operating point is that decisions
        # are made at it. The nominal figure stays in the run record beside it.
        operating_point = self._run.get("operating_point")
        if operating_point is not None:
            self._threshold = float(operating_point["threshold"])
            self._threshold_basis = (
                f"cost-minimising at a {operating_point['cost_ratio']:.0f}:1 "
                f"false-negative to false-positive ratio, fitted on "
                f"{operating_point['fitted_on']}"
            )
        else:
            self._threshold = float(self._run["threshold"])
            self._threshold_basis = (
                "nominal, no operating point recorded for this track"
            )

        logger.info(
            "loaded %s track %s, threshold %.4f (%s)",
            self._run["dataset"],
            self._run["track"],
            self._threshold,
            self._threshold_basis,
        )

    @property
    def run(self) -> dict[str, Any]:
        return self._run

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def threshold_basis(self) -> str:
        """Why the served threshold has the value it has."""
        return self._threshold_basis

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def _get_explainer(self) -> Any:
        """Lazily build the SHAP explainer on first use."""
        if self._explainer is None:
            if self._background is None:
                raise RuntimeError(
                    "No background data in the track artifact. Re-run the pipeline "
                    "to generate an artifact with SHAP background."
                )
            self._explainer = build_explainer(self._model, self._background)
            logger.info("SHAP explainer built for %s", type(self._model.named_steps["classifier"]).__name__)
        return self._explainer

    def _get_feature_names(self) -> list[str]:
        """Encoded feature names from the fitted pipeline's encoder step."""
        return encoded_feature_names(self._model.named_steps["encoder"])

    def predict(
        self,
        applicant: dict[str, Any],
        *,
        include_reasons: bool = False,
    ) -> dict[str, Any]:
        """Score one applicant and return a decision dict.

        Args:
            applicant: Field names matching ``spec.feature_columns``, values in codebook
                domain. Protected attributes must not be present.
            include_reasons: If True and the decision is "decline", attach SHAP-based
                reason codes to the response. Approved applications do not require
                adverse-action notices under ECOA.

        Returns:
            Dict with probability_of_default, decision, threshold, track, dataset,
            and optionally reason_codes (list of dicts, or None).

        Raises:
            KeyError: If a required feature column is missing from ``applicant``.
            ValueError: If a prohibited-basis attribute is supplied, or if the model
                rejects the input (out-of-domain category, for example).
        """
        prohibited = sorted(
            column for column in self._spec.prohibited_inputs if column in applicant
        )
        if prohibited:
            raise ValueError(
                f"prohibited-basis attributes are not accepted as input: {prohibited}"
            )

        frame = pd.DataFrame([applicant])
        features = extract_features(self._spec, frame)

        scores = positive_class_probabilities(self._model, features)
        probability = float(scores[0])

        decision = "decline" if probability >= self._threshold else "approve"

        result: dict[str, Any] = {
            "probability_of_default": probability,
            "decision": decision,
            "threshold": self._threshold,
            "threshold_basis": self._threshold_basis,
            "track": self._run["track"],
            "dataset": self._run["dataset"],
        }

        if include_reasons and decision == "decline":
            result["reason_codes"] = self._compute_reasons(features)
        elif include_reasons:
            result["reason_codes"] = None

        return result

    def explain(self, applicant: dict[str, Any]) -> dict[str, Any]:
        """Score and explain, regardless of decision.

        Always returns reason codes, even for approved applicants. Intended for internal
        audit and model monitoring, not for applicant-facing notices.
        """
        prohibited = sorted(
            column for column in self._spec.prohibited_inputs if column in applicant
        )
        if prohibited:
            raise ValueError(
                f"prohibited-basis attributes are not accepted as input: {prohibited}"
            )

        frame = pd.DataFrame([applicant])
        features = extract_features(self._spec, frame)

        scores = positive_class_probabilities(self._model, features)
        probability = float(scores[0])
        decision = "decline" if probability >= self._threshold else "approve"

        return {
            "probability_of_default": probability,
            "decision": decision,
            "threshold": self._threshold,
            "threshold_basis": self._threshold_basis,
            "track": self._run["track"],
            "dataset": self._run["dataset"],
            "reason_codes": self._compute_reasons(features),
        }

    def _compute_reasons(self, features: pd.DataFrame) -> list[dict[str, Any]]:
        """SHAP reason codes for the encoded features of one applicant."""
        explainer = self._get_explainer()
        feature_names = self._get_feature_names()

        # Transform through the encoder to get the encoded row the model actually sees.
        encoded = self._model.named_steps["encoder"].transform(features)
        encoded_arr = np.asarray(encoded, dtype=np.float64)

        reasons: list[ReasonCode] = explain_single(
            explainer,
            encoded_arr,
            feature_names,
            indicator_features=indicator_feature_names(self._model),
        )
        return [reason.as_dict() for reason in reasons]
