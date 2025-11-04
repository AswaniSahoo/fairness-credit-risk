"""
Model Loader Module
Handles loading and managing ML artifacts for the API
Loads models, preprocessors, and fairness optimization components
"""

import joblib
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


class ModelLoader:
    """
    Manages loading and utilizing ML artifacts for inference
    Handles both standard models and fairness-aware models with post-processing
    """
    
    def __init__(self, artifacts_path: str = "artifacts"):
        """
        Initialize model loader
        
        Args:
            artifacts_path: Directory containing saved model artifacts
        """
        self.artifacts_path = artifacts_path
        self.model = None
        self.preprocessor = None
        self.threshold_optimizer = None
        self.model_metadata = None
        self.feature_columns = None
        
    def load_all(self):
        """
        Load all required artifacts for model inference
        Includes model, preprocessor, threshold optimizer, and metadata
        """
        print("Loading model artifacts...")
        
        # Attempt to load fair model with post-processing
        fair_model_path = os.path.join(self.artifacts_path, "fair_model_complete.joblib")
        if os.path.exists(fair_model_path):
            self.model = joblib.load(fair_model_path)
            print("Fair model loaded successfully")
        else:
            # Fallback to base model if fair model not found
            base_model_path = os.path.join(self.artifacts_path, "best_automl_model.joblib")
            self.model = joblib.load(base_model_path)
            print("Base model loaded (fairness post-processing not available)")
        
        # Load preprocessor and feature column information
        preprocessor_data = joblib.load(os.path.join(self.artifacts_path, "preprocessor.joblib"))
        self.preprocessor = preprocessor_data['preprocessor']
        self.feature_columns = preprocessor_data['feature_columns']
        print("Preprocessor loaded successfully")
        
        # Load threshold optimizer if available
        threshold_path = os.path.join(self.artifacts_path, "threshold_optimizer.joblib")
        if os.path.exists(threshold_path):
            self.threshold_optimizer = joblib.load(threshold_path)
            print("Threshold optimizer loaded successfully")
        
        # Load model metadata (performance metrics, etc.)
        metadata_path = os.path.join(self.artifacts_path, "final_model_report.joblib")
        if os.path.exists(metadata_path):
            self.model_metadata = joblib.load(metadata_path)
            print("Model metadata loaded successfully")
        
        print("All artifacts loaded successfully\n")
        
    def predict(self, input_data: Dict[str, Any], apply_fairness: bool = True) -> Dict[str, Any]:
        """
        Make prediction on input data with optional fairness adjustment
        
        Args:
            input_data: Dictionary containing feature values for prediction
            apply_fairness: Whether to apply fairness post-processing (default: True)
            
        Returns:
            Dictionary containing:
                - prediction: Binary prediction (0=good, 1=default)
                - prediction_label: Human-readable prediction label
                - probability_default: Probability of default
                - probability_good: Probability of good credit
                - risk_level: Risk category (Low/Medium/High)
                - fairness_adjusted: Whether fairness adjustment was applied
        """
        # Extract protected attribute (gender) for fairness adjustment
        # Remove from features since it shouldn't be used for prediction
        gender = input_data.pop('gender', None)
        
        # Convert input to DataFrame with correct feature order
        df = pd.DataFrame([input_data])
        df = df[self.feature_columns]  # Ensure features are in training order
        
        # Apply preprocessing (scaling, imputation, etc.)
        X_processed = self.preprocessor.transform(df)
        
        # Get probability predictions from base model
        if hasattr(self.model, 'get_base_model'):
            # Model is wrapped, extract base model
            base_model = self.model.get_base_model()
            y_proba = base_model.predict_proba(X_processed)[0]
        else:
            # Model is not wrapped, use directly
            y_proba = self.model.predict_proba(X_processed)[0]
        
        # Extract probabilities for each class
        probability_good = y_proba[0]      # Class 0: Good credit
        probability_default = y_proba[1]   # Class 1: Default risk
        
        # Make final prediction with optional fairness adjustment
        fairness_adjusted = False
        if apply_fairness and self.threshold_optimizer and gender is not None:
            # Apply group-specific threshold for fairness
            protected_df = pd.DataFrame({'gender': [gender]})
            prediction = self.threshold_optimizer.predict(
                np.array([probability_default]), 
                protected_df
            )[0]
            fairness_adjusted = True
        else:
            # Use default threshold (0.5) without fairness adjustment
            prediction = int(probability_default >= 0.5)
        
        # Categorize risk level based on default probability
        if probability_default < 0.3:
            risk_level = "Low"
        elif probability_default < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Create human-readable prediction label
        prediction_label = "Good Credit" if prediction == 0 else "Default Risk"
        
        # Return comprehensive prediction result
        return {
            "prediction": int(prediction),
            "prediction_label": prediction_label,
            "probability_default": float(probability_default),
            "probability_good": float(probability_good),
            "risk_level": risk_level,
            "fairness_adjusted": fairness_adjusted
        }
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve model performance and fairness metrics
        
        Returns:
            Dictionary containing model metadata and metrics, or None if not available
        """
        if self.model_metadata:
            return self.model_metadata
        return None