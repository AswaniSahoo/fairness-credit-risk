from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any

class CreditApplicationRequest(BaseModel):
    """Input schema for credit risk prediction"""
    
    # Account status
    status: int = Field(..., description="Status of existing checking account (A11-A14)", ge=0)
    
    # Credit details
    duration: int = Field(..., description="Duration in months", ge=1, le=72)
    credit_history: int = Field(..., description="Credit history code (A30-A34)", ge=0)
    purpose: int = Field(..., description="Purpose of credit (A40-A410)", ge=0)
    amount: int = Field(..., description="Credit amount", ge=250, le=20000)
    
    # Savings and employment
    savings: int = Field(..., description="Savings account/bonds (A61-A65)", ge=0)
    employment_duration: int = Field(..., description="Present employment since (A71-A75)", ge=0)
    installment_rate: int = Field(..., description="Installment rate as % of disposable income", ge=1, le=4)
    
    # Personal information
    personal_status_sex: int = Field(..., description="Personal status and sex (A91-A95)", ge=0)
    other_debtors: int = Field(..., description="Other debtors/guarantors (A101-A103)", ge=0)
    present_residence: int = Field(..., description="Present residence since (years)", ge=1, le=4)
    property: int = Field(..., description="Property type (A121-A124)", ge=0)
    age: int = Field(..., description="Age in years", ge=18, le=100)
    
    # Other credits and housing
    other_installment_plans: int = Field(..., description="Other installment plans (A141-A143)", ge=0)
    housing: int = Field(..., description="Housing type (A151-A153)", ge=0)
    number_credits: int = Field(..., description="Number of existing credits at this bank", ge=1, le=4)
    
    # Job and other
    job: int = Field(..., description="Job category (A171-A174)", ge=0)
    people_liable: int = Field(..., description="Number of people being liable", ge=1, le=2)
    telephone: int = Field(..., description="Telephone (A191-A192)", ge=0)
    foreign_worker: int = Field(..., description="Foreign worker (A201-A202)", ge=0)
    
    # Protected attributes (for fairness monitoring only - NOT used in prediction)
    gender: Optional[int] = Field(None, description="Gender (0=female, 1=male) - for fairness monitoring only")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": 0,
                "duration": 12,
                "credit_history": 2,
                "purpose": 3,
                "amount": 5000,
                "savings": 0,
                "employment_duration": 2,
                "installment_rate": 2,
                "personal_status_sex": 2,
                "other_debtors": 0,
                "present_residence": 3,
                "property": 1,
                "age": 35,
                "other_installment_plans": 0,
                "housing": 1,
                "number_credits": 1,
                "job": 2,
                "people_liable": 1,
                "telephone": 0,
                "foreign_worker": 0,
                "gender": 1
            }
        }

class CreditPredictionResponse(BaseModel):
    """Output schema for credit risk prediction"""
    
    prediction: int = Field(..., description="Predicted class: 0=Good Credit, 1=Default Risk")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability_default: float = Field(..., description="Probability of default (0-1)")
    probability_good: float = Field(..., description="Probability of good credit (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")
    fairness_adjusted: bool = Field(..., description="Whether fairness post-processing was applied")
    model_version: str = Field(default="1.0.0", description="Model version")
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    fairness_module_loaded: bool
    version: str

class ModelMetricsResponse(BaseModel):
    """Model performance metrics"""
    roc_auc: float
    balanced_accuracy: float
    f1_score: float
    precision: float
    recall: float
    disparate_impact: float
    statistical_parity_difference: float
    equal_opportunity_difference: float
    fairness_compliant: bool