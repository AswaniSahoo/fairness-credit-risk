# Fairness-Aware AutoML for Credit Risk Scoring

**A production-ready, bias-mitigating credit risk prediction system with automated model selection and REST API deployment.**

## 📊 Phase 1: Bias Detection & Analysis

Before building any models, I conducted comprehensive fairness analysis:

**Key Discoveries:**
- 🔍 **7.5% gender approval gap** (males: 72.4%, females: 64.9%)
- 📊 **27.4% intersectional variance** across gender-age groups
- ⚖️ **Disparate Impact: 0.897** (borderline legal compliance)
- 📈 **30% default rate** (class imbalance requiring mitigation)

**Analysis Artifacts:**
- 6-panel visualization dashboard
- Intersectional bias heatmaps
- Feature correlation analysis
- Protected attribute impact assessment

**Tools Used:** AIF360, Matplotlib, Pandas, NumPy


[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project implements an **end-to-end fairness-aware machine learning pipeline** for credit risk assessment, addressing critical challenges in financial AI:

- **Class Imbalance** (70-30 split): Handled via `class_weight='balanced'`
- **Algorithmic Bias**: Mitigated using AIF360 reweighting
- **Model Selection**: Automated hyperparameter tuning with Optuna (50 trials)
- **Legal Compliance**: Disparate Impact > 0.8 (80% rule)

---

## 🏆 Key Results

| Metric | Value | Status |
|--------|-------|--------|
| **ROC-AUC** | 0.840 | ✅ Excellent |
| **Balanced Accuracy** | 0.726 | ✅ Good |
| **F1-Score** | 0.614 | ✅ Solid |
| **Disparate Impact** | 0.890 | ✅ Legal (>0.8) |
| **Statistical Parity** | -0.079 | ✅ Fair (±0.1) |

---

## 📁 Project Structure

automl/
├── api/ # REST API
│ ├── main.py # FastAPI application
│ ├── schemas/ # Pydantic models
│ └── utils/ # Model loader
├── src/
│ ├── preprocessing/ # Data processing
│ ├── training/ # AutoML tuner
│ ├── evaluation/ # Fairness metrics
│ └── models/ # Model wrappers
├── config/
│ └── config.py # Configuration
├── artifacts/ # Saved models
├── reports/ # Evaluation reports
├── Dockerfile # Container definition
├── docker-compose.yml # Orchestration
└── requirements_api.txt # Dependencies


---

## 🚀 Quick Start

### **1. Run with Docker (Recommended)**

```bash
# Build and start the API
docker-compose up --build -d

# Test the API
python test_api.py

# Access Swagger UI
open http://localhost:8000/docs
```

🔬 Methodology
Phase 1: Bias Detection
Protected attributes identified: gender, age, foreign_worker
Initial Disparate Impact: 0.897 (borderline)
7.5% approval gap between genders detected
Phase 2: Fairness Mitigation
Pre-processing: AIF360 Reweighing (sample weights: 0.855-1.082)
In-processing: class_weight='balanced' for imbalance
Post-processing: Threshold optimization (attempted)
Phase 3: AutoML Optimization
Models tested: Random Forest, XGBoost, LightGBM, Logistic Regression
Trials: 50 (Optuna TPE sampler)
Objective: Composite score (70% performance + 30% fairness)
Winner: Random Forest (0.634 composite score)
Phase 4: Deployment
FastAPI REST API with Pydantic validation
Docker containerization with health checks
Automatic fairness adjustment via threshold optimizer

📈 Model Performance
Confusion Matrix (Test Set):

```
                Predicted
              Good | Bad
Actual Good    133 |  28
       Bad      12 |  75
```
Key Insights:

Precision: 53.7% (of predicted defaults, 53.7% actually defaulted)
Recall: 71.7% (caught 71.7% of actual defaults)
Trade-off: Model prioritizes catching defaults (high recall) over precision
⚖️ Fairness Analysis
Disparate Impact by Gender:

Male approval rate: 73.4%
Female approval rate: 65.3%
Disparate Impact: 0.890 ✅ (above 0.8 legal threshold)
Limitations:

Equal Opportunity metric (-0.225) outside ideal range
Future work: Calibration methods, more balanced training data

🛠️ Technologies Used
Category	Technology
ML Framework	scikit-learn, XGBoost, LightGBM
Fairness	AIF360 (IBM)
AutoML	Optuna
API	FastAPI, Pydantic
Deployment	Docker, Docker Compose
Monitoring	Logging, Health Checks
📚 Dataset
German Credit Dataset (UCI ML Repository)

1,000 loan applications
20 features (7 numerical, 13 categorical)
70% good credit, 30% default
Protected attributes: gender, age, foreign worker status
🔮 Future Improvements
Fairness: Adversarial debiasing, calibration
Performance: Ensemble methods, feature engineering
Deployment: Kubernetes, A/B testing, model monitoring
Explainability: SHAP integration for loan decisions
Data: Active learning for underrepresented groups
📖 References
AIF360 Documentation(https://aif360.readthedocs.io/en/stable/)
Fairlearn
German Credit Dataset
Optuna
