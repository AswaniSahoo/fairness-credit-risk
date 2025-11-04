# src/models/fair_model_wrapper.py
"""
Fair Model Wrapper Module
Combines a base ML model with optional fairness post-processing
Allows seamless integration of threshold optimization for fair predictions
"""

import numpy as np
import pandas as pd


class FairModelWrapper:
    """
    Wrapper class that integrates fairness post-processing with standard ML models
    Enables fair predictions by applying group-specific thresholds when needed
    """
    
    def __init__(self, base_model, threshold_optimizer=None):
        """
        Initialize fair model wrapper
        
        Args:
            base_model: Trained ML model (scikit-learn compatible)
            threshold_optimizer: Optional ThresholdOptimizer for fairness adjustment
        """
        self.base_model = base_model
        self.threshold_optimizer = threshold_optimizer
        
    def predict_proba(self, X):
        """
        Get probability predictions from the base model
        
        Args:
            X: Feature matrix
            
        Returns:
            np.array: Probability predictions for each class
        """
        return self.base_model.predict_proba(X)
    
    def predict(self, X, protected_attributes=None):
        """
        Make predictions with optional fairness adjustment
        If threshold optimizer is available and protected attributes are provided,
        applies group-specific thresholds for fairer predictions
        
        Args:
            X: Feature matrix
            protected_attributes: DataFrame with protected attributes 
                                 (required if threshold_optimizer is set)
            
        Returns:
            np.array: Binary predictions (0 or 1)
        """
        # Get probability predictions for positive class (class 1)
        y_proba = self.base_model.predict_proba(X)[:, 1]
        
        # Apply fairness post-processing if threshold optimizer is available
        if self.threshold_optimizer is not None and protected_attributes is not None:
            # Use group-specific thresholds for fair predictions
            return self.threshold_optimizer.predict(y_proba, protected_attributes)
        else:
            # Use default threshold (0.5) if no fairness adjustment
            return (y_proba >= 0.5).astype(int)
    
    def get_base_model(self):
        """
        Get the underlying base model
        Useful for accessing model-specific attributes like feature importances
        
        Returns:
            Base ML model object
        """
        return self.base_model