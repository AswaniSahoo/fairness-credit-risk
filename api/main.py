"""Credit risk scoring REST API.

SECURITY: This service has NO authentication. It must not be exposed to the public
internet without an authenticating reverse proxy (e.g. AWS ALB + Cognito, nginx +
OAuth2-proxy). The absence is deliberate: adding a placeholder auth layer would create a
false sense of security while obscuring the real deployment requirement.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas.prediction import (
    CreditApplicationRequest,
    HealthResponse,
    PredictionResponse,
)
from src.paths import REPORTS_DIR
from src.serving.predictor import Predictor

logger = logging.getLogger(__name__)

_predictor: Predictor | None = None
_comparison: dict[str, Any] | None = None


def _load_comparison() -> dict[str, Any]:
    path = REPORTS_DIR / "track_comparison.json"
    return json.loads(path.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:  # noqa: ARG001
    global _predictor, _comparison  # noqa: PLW0603
    try:
        _predictor = Predictor()
        _comparison = _load_comparison()
        logger.info("startup: predictor and comparison artifact loaded")
    except Exception:
        logger.exception("startup: failed to load artifacts")
        _predictor = None
        _comparison = None
    yield
    _predictor = None
    _comparison = None


app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Single-threshold credit risk prediction. No authentication is provided; "
        "deploy behind an authenticating proxy before exposing publicly."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# allow_credentials=True requires an explicit origin list; the wildcard '*' is rejected
# by browsers when credentials are included (Fetch spec). Since this API carries no
# session cookies, credentials are disabled and the wildcard is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _predictor is None:
        return HealthResponse(
            status="unavailable", track=None, dataset=None, artifact_loaded=False
        )
    return HealthResponse(
        status="ok",
        track=_predictor.run["track"],
        dataset=_predictor.run["dataset"],
        artifact_loaded=True,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CreditApplicationRequest) -> PredictionResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = _predictor.predict(request.model_dump())
    return PredictionResponse(**result)


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Test-block metrics and bootstrap intervals from the track comparison artifact."""
    if _predictor is None or _comparison is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    key = f"{_predictor.run['dataset']}|{_predictor.run['track']}"
    run = _comparison["runs"].get(key)
    if run is None:
        raise HTTPException(
            status_code=404, detail=f"No comparison entry for {key}"
        )
    return {
        "track": run["track"],
        "dataset": run["dataset"],
        "performance": run["performance"],
        "fairness": run["fairness"],
        "intervals": run["intervals"],
    }


@app.get("/model-info")
async def model_info() -> dict[str, Any]:
    """Model type, hyperparameters, split sizes and fingerprint from the run record."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    run = _predictor.run
    return {
        "model_type": run["model"]["model_type"],
        "params": run["model"]["params"],
        "split_sizes": run["split_sizes"],
        "n_encoded_features": run["n_encoded_features"],
        "split_fingerprint": run["split_fingerprint"],
        "threshold": run["threshold"],
        "track": run["track"],
        "dataset": run["dataset"],
    }
