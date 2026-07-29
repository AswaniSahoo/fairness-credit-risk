"""Request and response schemas for the credit risk prediction API.

Field names match ``GERMAN_CREDIT.feature_columns`` exactly. Domain constraints come from
the UCI german.doc codebook. Protected attributes (gender, personal_status_sex,
foreign_worker) are absent because they are prohibited bases under ECOA.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CreditApplicationRequest(BaseModel):
    """One applicant's feature vector, validated against the codebook domain."""

    # --- numeric (5) ---
    duration: int = Field(..., ge=4, le=72)
    amount: int = Field(..., ge=250, le=20000)
    age: int = Field(..., ge=19, le=75)
    number_credits: int = Field(..., ge=1, le=4)
    people_liable: int = Field(..., ge=1, le=2)

    # --- ordinal (4) ---
    # A71=0 .. A75=4
    employment_duration: int = Field(..., ge=0, le=4)
    # 1 to 4, percentage of disposable income
    installment_rate: int = Field(..., ge=1, le=4)
    # 1 to 4, years at current residence
    present_residence: int = Field(..., ge=1, le=4)
    # A171=0 .. A174=3
    job: int = Field(..., ge=0, le=3)

    # --- nominal (9) ---
    # A11=0 .. A14=3
    status: int = Field(..., ge=0, le=3)
    # A30=0 .. A34=4
    credit_history: int = Field(..., ge=0, le=4)
    # A40=0 .. A49=9 (A47 never occurs but is in the domain)
    purpose: int = Field(..., ge=0, le=9)
    # A61=0 .. A65=4
    savings: int = Field(..., ge=0, le=4)
    # A101=0 .. A103=2
    other_debtors: int = Field(..., ge=0, le=2)
    # A121=0 .. A124=3
    property: int = Field(..., ge=0, le=3)
    # A141=0 .. A143=2
    other_installment_plans: int = Field(..., ge=0, le=2)
    # A151=0 .. A153=2
    housing: int = Field(..., ge=0, le=2)
    # A191=0, A192=1
    telephone: Literal[0, 1] = Field(...)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "duration": 24,
                    "amount": 5951,
                    "age": 35,
                    "number_credits": 1,
                    "people_liable": 1,
                    "employment_duration": 2,
                    "installment_rate": 2,
                    "present_residence": 3,
                    "job": 2,
                    "status": 1,
                    "credit_history": 2,
                    "purpose": 3,
                    "savings": 0,
                    "other_debtors": 0,
                    "property": 1,
                    "other_installment_plans": 2,
                    "housing": 1,
                    "telephone": 0,
                }
            ]
        }
    }


class ReasonCodeResponse(BaseModel):
    """One line of an adverse-action notice."""

    feature: str
    shap_value: float
    direction: str
    contribution: str


class PredictionResponse(BaseModel):
    """Decision returned by POST /predict."""

    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["approve", "decline"]
    threshold: float
    track: str
    dataset: str
    reason_codes: list[ReasonCodeResponse] | None = None


class ExplainResponse(BaseModel):
    """Decision with reason codes, returned by POST /explain."""

    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["approve", "decline"]
    threshold: float
    track: str
    dataset: str
    reason_codes: list[ReasonCodeResponse]


class HealthResponse(BaseModel):
    """GET /health response."""

    status: Literal["ok", "unavailable"]
    track: str | None
    dataset: str | None
    artifact_loaded: bool
