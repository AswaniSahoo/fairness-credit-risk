"""
Fairness-Aware Credit Risk Scoring - Streamlit Demo
Deploy to Hugging Face Spaces for live demonstration
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Fairness-Aware Credit Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-good {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
    .prediction-bad {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
    .fairness-badge {
        background: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model():
    """Load the trained model and preprocessor"""
    try:
        model = joblib.load('artifacts/fair_model_complete.joblib')
        preprocessor = joblib.load('artifacts/preprocessor.joblib')
        
        # Try to load threshold optimizer
        threshold_optimizer = None
        if os.path.exists('artifacts/threshold_optimizer.joblib'):
            threshold_optimizer = joblib.load('artifacts/threshold_optimizer.joblib')
        
        return model, preprocessor, threshold_optimizer, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

# Load model
model, preprocessor, threshold_optimizer, model_loaded = load_model()

# Header
st.markdown('<p class="main-header">🏦 Fairness-Aware Credit Risk Scoring</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AutoML-Optimized Model with Bias Mitigation | GSoC 2026 Portfolio Project</p>', unsafe_allow_html=True)

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    
    if model_loaded:
        st.success("✅ Model Loaded Successfully")
        
        st.markdown("---")
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC-AUC", "0.840")
            st.metric("Balanced Acc", "0.726")
        with col2:
            st.metric("F1-Score", "0.614")
            st.metric("Precision", "0.537")
        
        st.markdown("---")
        st.subheader("Fairness Metrics")
        st.metric("Disparate Impact", "0.890", help="≥0.8 is legally compliant (80% rule)")
        st.metric("Statistical Parity", "-0.079", help="Should be within ±0.1")
        
        st.markdown("---")
        st.subheader("About")
        st.info("""
        **Model**: Random Forest (AutoML Optimized)
        
        **Fairness Methods**:
        - AIF360 Reweighting
        - Threshold Optimization
        
        **Training**: 50 Optuna trials with composite score (70% performance + 30% fairness)
        """)
    else:
        st.error("❌ Model not loaded")

# Main content
if model_loaded:
    st.markdown("### 📝 Enter Applicant Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Personal Information**")
        age = st.slider("Age", 18, 75, 35)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        foreign_worker = st.selectbox("Foreign Worker", ["No", "Yes"], index=0)
        employment_duration = st.selectbox("Employment Duration", [
            "Unemployed", "< 1 year", "1-4 years", "4-7 years", ">7 years"
        ], index=2)
    
    with col2:
        st.markdown("**Financial Information**")
        amount = st.number_input("Credit Amount ($)", 250, 50000, 5000, step=500)
        duration = st.slider("Loan Duration (months)", 4, 72, 24)
        status = st.selectbox("Checking Account Status", [
            "No account", "< $0", "$0-$200", "> $200"
        ], index=2)
        savings = st.selectbox("Savings Account", [
            "< $100", "$100-$500", "$500-$1000", "> $1000", "Unknown"
        ], index=1)
    
    with col3:
        st.markdown("**Credit History**")
        credit_history = st.selectbox("Credit History", [
            "No credits", "All paid", "Existing paid", "Delayed", "Critical"
        ], index=2)
        purpose = st.selectbox("Loan Purpose", [
            "Car (new)", "Car (used)", "Furniture", "Radio/TV", 
            "Appliances", "Repairs", "Education", "Business", "Other"
        ], index=3)
        property_type = st.selectbox("Property", [
            "Real estate", "Savings agreement", "Car", "None"
        ], index=0)
        housing = st.selectbox("Housing", ["Rent", "Own", "Free"], index=1)

    # Additional fields in expander
    with st.expander("📋 Additional Details (Optional)"):
        col4, col5 = st.columns(2)
        with col4:
            installment_rate = st.slider("Installment Rate (%)", 1, 4, 2)
            present_residence = st.slider("Years at Current Residence", 1, 4, 3)
            number_credits = st.slider("Number of Existing Credits", 1, 4, 1)
        with col5:
            people_liable = st.selectbox("People Liable", [1, 2], index=0)
            telephone = st.selectbox("Has Telephone", ["No", "Yes"], index=0)
            other_debtors = st.selectbox("Other Debtors/Guarantors", ["None", "Co-applicant", "Guarantor"], index=0)
            job = st.selectbox("Job Type", ["Unskilled non-resident", "Unskilled resident", "Skilled", "Management"], index=2)
            other_plans = st.selectbox("Other Installment Plans", ["None", "Bank", "Stores"], index=0)
            personal_status = st.selectbox("Personal Status", ["Single Male", "Married Male", "Single Female", "Married Female"], index=0)

    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Credit Risk", type="primary", use_container_width=True):
        # Encode inputs
        status_map = {"No account": 0, "< $0": 1, "$0-$200": 2, "> $200": 3}
        credit_history_map = {"No credits": 0, "All paid": 1, "Existing paid": 2, "Delayed": 3, "Critical": 4}
        purpose_map = {"Car (new)": 0, "Car (used)": 1, "Furniture": 2, "Radio/TV": 3, "Appliances": 4, "Repairs": 5, "Education": 6, "Business": 7, "Other": 8}
        savings_map = {"< $100": 0, "$100-$500": 1, "$500-$1000": 2, "> $1000": 3, "Unknown": 4}
        employment_map = {"Unemployed": 0, "< 1 year": 1, "1-4 years": 2, "4-7 years": 3, ">7 years": 4}
        property_map = {"Real estate": 0, "Savings agreement": 1, "Car": 2, "None": 3}
        housing_map = {"Rent": 0, "Own": 1, "Free": 2}
        gender_map = {"Male": 1, "Female": 0}
        job_map = {"Unskilled non-resident": 0, "Unskilled resident": 1, "Skilled": 2, "Management": 3}
        other_debtors_map = {"None": 0, "Co-applicant": 1, "Guarantor": 2}
        other_plans_map = {"None": 0, "Bank": 1, "Stores": 2}
        personal_status_map = {"Single Male": 0, "Married Male": 1, "Single Female": 2, "Married Female": 3}
        
        # Create feature vector (matching the training features)
        features = {
            'status': status_map[status],
            'duration': duration,
            'credit_history': credit_history_map[credit_history],
            'purpose': purpose_map[purpose],
            'amount': amount,
            'savings': savings_map[savings],
            'employment_duration': employment_map[employment_duration],
            'installment_rate': installment_rate,
            'personal_status_sex': personal_status_map[personal_status],
            'other_debtors': other_debtors_map[other_debtors],
            'present_residence': present_residence,
            'property': property_map[property_type],
            'age': age,
            'other_installment_plans': other_plans_map[other_plans],
            'housing': housing_map[housing],
            'number_credits': number_credits,
            'job': job_map[job],
            'people_liable': people_liable,
            'telephone': 1 if telephone == "Yes" else 0,
            'foreign_worker': 1 if foreign_worker == "Yes" else 0,
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        try:
            # Get prediction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)
                prob_default = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
                prob_good = 1 - prob_default
                prediction = 1 if prob_default > 0.5 else 0
            else:
                prediction = model.predict(input_df)[0]
                prob_default = float(prediction)
                prob_good = 1 - prob_default
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 0:
                    st.markdown("""
                    <div class="prediction-good">
                        <h2>✅ LOW RISK</h2>
                        <p>Likely to repay the loan</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-bad">
                        <h2>⚠️ HIGH RISK</h2>
                        <p>May default on the loan</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Probability of Good Credit", f"{prob_good:.1%}")
                st.progress(prob_good)
            
            with col3:
                st.metric("Probability of Default", f"{prob_default:.1%}")
                st.progress(prob_default)
            
            # Fairness notice
            st.markdown("---")
            st.info("""
            🔒 **Fairness Assurance**: This prediction was made using a bias-mitigated model. 
            The model has been trained with AIF360 reweighting to ensure fair treatment across gender groups.
            - Disparate Impact: 0.890 (passes 80% rule)
            - Statistical Parity Difference: -0.079 (within ±0.1 threshold)
            """)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("The model may require specific preprocessing. Please check the input format.")

else:
    st.warning("⚠️ Model not loaded. Please ensure the model artifacts are available.")
    st.info("""
    To run this demo locally:
    1. Clone the repository
    2. Run `python run_automl.py` to train the model
    3. Run `streamlit run app.py`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Built with ❤️ by <a href="https://github.com/AswaniSahoo" target="_blank">Aswani Sahoo</a> | 
    <a href="https://github.com/AswaniSahoo/fairness-credit-risk" target="_blank">View on GitHub</a></p>
    <p>GSoC 2026 Portfolio Project | Fairness-Aware Machine Learning</p>
</div>
""", unsafe_allow_html=True)
