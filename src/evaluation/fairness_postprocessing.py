"""
Fairness Post-Processing Module
Implements threshold optimization techniques to improve fairness after model training
Uses group-specific classification thresholds to achieve equal opportunity
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize_scalar


class ThresholdOptimizer:
    """
    Post-processing technique to achieve fairness through threshold adjustment
    Optimizes classification thresholds separately for different demographic groups
    to equalize true positive rates (equal opportunity)
    """
    
    def __init__(self, protected_attribute='gender'):
        """
        Initialize threshold optimizer
        
        Args:
            protected_attribute: Name of the protected attribute (e.g., 'gender', 'age')
        """
        self.protected_attribute = protected_attribute
        self.thresholds = {}  # Store optimal thresholds for each group
        
    def fit(self, y_true, y_proba, protected_attributes, target_metric='equal_opportunity'):
        """
        Find optimal classification thresholds for each protected group
        Adjusts thresholds to minimize disparity in true positive rates
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities (for positive class)
            protected_attributes: DataFrame with protected attributes
            target_metric: Fairness criterion ('equal_opportunity' or 'equalized_odds')
            
        Returns:
            self: Fitted optimizer
        """
        print(f"\nOptimizing classification thresholds for {target_metric}...")
        
        protected_series = protected_attributes[self.protected_attribute]
        unique_groups = protected_series.unique()
        
        # Calculate baseline metrics using default threshold (0.5)
        baseline_metrics = {}
        for group in unique_groups:
            mask = protected_series == group
            baseline_metrics[group] = self._calculate_tpr_fpr(
                y_true[mask], (y_proba[mask] >= 0.5).astype(int)
            )
        
        print(f"Baseline TPR by group (threshold=0.5): {baseline_metrics}")
        
        # Define privileged and unprivileged groups
        privileged_group = 1    # e.g., Male
        unprivileged_group = 0  # e.g., Female
        
        # Get target TPR from privileged group
        priv_mask = protected_series == privileged_group
        target_tpr = baseline_metrics[privileged_group]['tpr']
        
        # Optimize threshold for unprivileged group to match privileged group's TPR
        def objective(threshold):
            """Objective function: minimize TPR difference between groups"""
            y_pred_unpriv = (y_proba[~priv_mask] >= threshold).astype(int)
            metrics = self._calculate_tpr_fpr(y_true[~priv_mask], y_pred_unpriv)
            return abs(metrics['tpr'] - target_tpr)
        
        # Find optimal threshold using bounded scalar minimization
        result = minimize_scalar(objective, bounds=(0.1, 0.9), method='bounded')
        optimal_threshold_unpriv = result.x
        
        # Store optimal thresholds for each group
        self.thresholds[privileged_group] = 0.5  # Keep default for privileged group
        self.thresholds[unprivileged_group] = optimal_threshold_unpriv
        
        # Verify the improvement
        y_pred_adjusted = self.predict(y_proba, protected_attributes)
        final_metrics = {}
        for group in unique_groups:
            mask = protected_series == group
            final_metrics[group] = self._calculate_tpr_fpr(y_true[mask], y_pred_adjusted[mask])
        
        print(f"Optimized thresholds: {self.thresholds}")
        print(f"Final TPR by group: {final_metrics}")
        print(f"TPR difference after optimization: {abs(final_metrics[1]['tpr'] - final_metrics[0]['tpr']):.3f}")
        
        return self
    
    def predict(self, y_proba, protected_attributes):
        """
        Apply group-specific thresholds to generate fair predictions
        
        Args:
            y_proba: Predicted probabilities for positive class
            protected_attributes: DataFrame with protected attributes
            
        Returns:
            np.array: Adjusted binary predictions
            
        Raises:
            ValueError: If optimizer hasn't been fitted yet
        """
        if not self.thresholds:
            raise ValueError("ThresholdOptimizer not fitted. Call fit() first.")
        
        protected_series = protected_attributes[self.protected_attribute]
        y_pred = np.zeros(len(y_proba), dtype=int)
        
        # Apply group-specific thresholds
        for group, threshold in self.thresholds.items():
            mask = protected_series == group
            y_pred[mask] = (y_proba[mask] >= threshold).astype(int)
        
        return y_pred
    
    def _calculate_tpr_fpr(self, y_true, y_pred):
        """
        Calculate True Positive Rate and False Positive Rate
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            dict: Contains TPR, FPR, and confusion matrix components
        """
        # Get confusion matrix for binary classification (0=good credit, 1=bad credit)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calculate rates for favorable outcome (class 0 = good credit)
        # TPR = how many actual good credits are correctly predicted as good
        tpr = tn / (tn + fp) if (tn + fp) > 0 else 0
        # FPR = how many actual bad credits are incorrectly predicted as good
        fpr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'tpr': tpr, 
            'fpr': fpr, 
            'tn': tn, 
            'fp': fp, 
            'fn': fn, 
            'tp': tp
        }