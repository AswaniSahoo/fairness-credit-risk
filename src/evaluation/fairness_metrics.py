"""
Fairness Metrics Evaluation Module
Calculates performance and fairness metrics for model evaluation
Handles class imbalance and provides comprehensive fairness assessment
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, balanced_accuracy_score, average_precision_score
)
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import config


class FairnessEvaluator:
    """
    Evaluates model fairness and performance metrics
    Uses AIF360 library for fairness calculations and sklearn for performance
    """
    
    def __init__(self, protected_attribute=config.fairness.PRIMARY_PROTECTED_ATTRIBUTE):
        """
        Initialize fairness evaluator
        
        Args:
            protected_attribute: Name of the protected attribute to evaluate (e.g., 'gender')
        """
        self.protected_attribute = protected_attribute
        
    def calculate_performance_metrics(self, y_true, y_pred, y_proba=None):
        """
        Calculate performance metrics with focus on imbalanced datasets
        Uses balanced metrics like balanced_accuracy and F1 score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC metrics)
            
        Returns:
            dict: Performance metrics including accuracy, F1, precision, recall, AUC
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),  # Adjusted for class imbalance
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            # Use probability of positive class (class 1 = default/bad credit)
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_proba)
        else:
            metrics['average_precision'] = 0.0
            
        return metrics
    
    def calculate_fairness_metrics(self, y_true, y_pred, protected_attributes):
        """
        Calculate comprehensive fairness metrics using AIF360
        Measures disparate impact, statistical parity, and equal opportunity
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: DataFrame with protected attribute values
            
        Returns:
            dict: Fairness metrics including disparate impact and statistical parity
        """
        # Convert predictions to AIF360 dataset format
        dataset = self._create_aif360_dataset(y_true, y_pred, protected_attributes)
        
        # Initialize AIF360 metric calculator
        metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=[{self.protected_attribute: 0}],  # e.g., Female = 0
            privileged_groups=[{self.protected_attribute: 1}]     # e.g., Male = 1
        )
        
        # Calculate fairness metrics
        fairness_metrics = {
            'disparate_impact': metric.disparate_impact(),  # Ratio of favorable outcomes
            'statistical_parity_difference': metric.statistical_parity_difference(),  # Difference in positive rates
            'equal_opportunity_difference': self._calculate_equal_opportunity(y_true, y_pred, protected_attributes)
        }
        
        return fairness_metrics
    
    def _create_aif360_dataset(self, y_true, y_pred, protected_attributes):
        """
        Convert predictions to AIF360 StandardDataset format
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: DataFrame with protected attribute values
            
        Returns:
            StandardDataset: AIF360 dataset object
        """
        df = pd.DataFrame({
            'prediction': y_pred,
            'true_label': y_true
        })
        df[self.protected_attribute] = protected_attributes[self.protected_attribute].values
        
        # Create AIF360 dataset
        dataset = StandardDataset(
            df=df,
            label_name='true_label',
            favorable_classes=[0],  # 0 = Good credit (no default)
            protected_attribute_names=[self.protected_attribute],
            privileged_classes=[[1]]  # 1 = Privileged group (e.g., Male)
        )
        
        return dataset
    
    def _calculate_equal_opportunity(self, y_true, y_pred, protected_attributes):
        """
        Calculate equal opportunity difference between privileged and unprivileged groups
        Measures difference in true positive rates for the favorable outcome
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: DataFrame with protected attribute values
            
        Returns:
            float: Equal opportunity difference (closer to 0 is fairer)
        """
        protected_series = protected_attributes[self.protected_attribute]
        
        # Calculate true positive rates for favorable outcome (no default = 0)
        # Privileged group (e.g., Male = 1)
        mask_priv = (protected_series == 1) & (y_true == 0)
        # Unprivileged group (e.g., Female = 0)
        mask_unpriv = (protected_series == 0) & (y_true == 0)
        
        # Safe calculation with zero-division handling
        tpr_privileged = np.mean(y_pred[mask_priv] == 0) if mask_priv.sum() > 0 else 0.0
        tpr_unprivileged = np.mean(y_pred[mask_unpriv] == 0) if mask_unpriv.sum() > 0 else 0.0
        
        # Return difference (ideally should be close to 0)
        return tpr_unprivileged - tpr_privileged
    
    def calculate_composite_score(self, performance_metrics, fairness_metrics, fairness_weight=0.3):
        """
        Calculate composite score balancing performance and fairness
        Uses metrics appropriate for imbalanced datasets
        
        Args:
            performance_metrics: Dict of performance metrics
            fairness_metrics: Dict of fairness metrics
            fairness_weight: Weight given to fairness (0 to 1, default 0.3)
            
        Returns:
            float: Composite score (higher is better)
        """
        # Performance component - focus on imbalance-appropriate metrics
        performance_score = (
            performance_metrics['balanced_accuracy'] +
            performance_metrics['f1_score'] +
            performance_metrics.get('roc_auc', 0.5)
        ) / 3
        
        # Fairness component - penalize unfairness
        # Disparate impact should be close to 1.0 (threshold is 0.8)
        disparate_impact_penalty = max(0, config.fairness.DISPARATE_IMPACT_THRESHOLD - 
                                      fairness_metrics['disparate_impact'])
        # Statistical parity difference should be close to 0
        statistical_parity_penalty = abs(fairness_metrics['statistical_parity_difference'])
        
        # Combined fairness score (1.0 is perfectly fair)
        fairness_score = 1 - (disparate_impact_penalty + statistical_parity_penalty)
        
        # Weighted composite score
        composite_score = (
            (1 - fairness_weight) * performance_score +
            fairness_weight * fairness_score
        )
        
        return composite_score
    
    def generate_fairness_report(self, model, X_test, y_test, protected_test, model_name="Model"):
        """
        Generate comprehensive fairness and performance report for a model
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            protected_test: Protected attributes for test set
            model_name: Name of the model for display
            
        Returns:
            dict: Complete evaluation results including performance, fairness, and composite score
        """
        print(f"\n{'='*50}")
        print(f"FAIRNESS REPORT: {model_name}")
        print(f"{'='*50}")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Get probability of positive class (class 1 = default/bad credit)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            proba_matrix = model.predict_proba(X_test)
            # Ensure we have binary classification
            if proba_matrix.shape[1] == 2:
                y_proba = proba_matrix[:, 1]  # Probability of class 1 (default)
            else:
                print("Warning: Model has unexpected number of classes")
        
        # Calculate all metrics
        performance_metrics = self.calculate_performance_metrics(y_test, y_pred, y_proba)
        fairness_metrics = self.calculate_fairness_metrics(y_test, y_pred, protected_test)
        composite_score = self.calculate_composite_score(performance_metrics, fairness_metrics)
        
        # Display performance metrics
        print("\nPERFORMANCE METRICS:")
        for metric, value in performance_metrics.items():
            print(f"   {metric:20}: {value:.3f}")
        
        # Display fairness metrics with pass/fail indicators
        print("\nFAIRNESS METRICS:")
        for metric, value in fairness_metrics.items():
            status = "PASS" if self._is_fair(metric, value) else "FAIL"
            print(f"   [{status}] {metric:30}: {value:.3f}")
        
        print(f"\nCOMPOSITE SCORE: {composite_score:.3f}")
        
        # Overall fairness assessment
        print(f"\nFAIRNESS ASSESSMENT:")
        if all(self._is_fair(metric, value) for metric, value in fairness_metrics.items()):
            print("   Model meets fairness criteria")
        else:
            print("   Model may require fairness mitigation")
        
        return {
            'performance': performance_metrics,
            'fairness': fairness_metrics,
            'composite_score': composite_score
        }
    
    def _is_fair(self, metric, value):
        """
        Check if a fairness metric meets the fairness threshold
        
        Args:
            metric: Name of the fairness metric
            value: Value of the metric
            
        Returns:
            bool: True if metric meets fairness criteria
        """
        if metric == 'disparate_impact':
            # Should be between 0.8 and 1.25 (reciprocal of 0.8)
            return (config.fairness.DISPARATE_IMPACT_THRESHOLD <= value <= 
                   (1 / config.fairness.DISPARATE_IMPACT_THRESHOLD))
        elif metric == 'statistical_parity_difference':
            # Absolute difference should be below threshold
            return abs(value) <= config.fairness.STATISTICAL_PARITY_THRESHOLD
        elif metric == 'equal_opportunity_difference':
            # Absolute difference should be below threshold
            return abs(value) <= config.fairness.STATISTICAL_PARITY_THRESHOLD
        return True
    
    def plot_fairness_comparison(self, models_results, save_path=None):
        """
        Create visualization comparing fairness metrics across multiple models
        
        Args:
            models_results: Dict mapping model names to their evaluation results
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract metrics for all models
        model_names = list(models_results.keys())
        disparate_impacts = [results['fairness']['disparate_impact'] for results in models_results.values()]
        statistical_parities = [results['fairness']['statistical_parity_difference'] for results in models_results.values()]
        composite_scores = [results['composite_score'] for results in models_results.values()]
        
        # Plot 1: Disparate Impact comparison
        bars1 = axes[0].bar(model_names, disparate_impacts, color='lightblue', alpha=0.7)
        axes[0].axhline(y=0.8, color='red', linestyle='--', label='Fairness Threshold (0.8)')
        axes[0].axhline(y=1.0, color='green', linestyle='-', alpha=0.5, label='Perfect Fairness (1.0)')
        axes[0].set_title('Disparate Impact by Model')
        axes[0].set_ylabel('Disparate Impact')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, disparate_impacts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Statistical Parity Difference comparison
        bars2 = axes[1].bar(model_names, statistical_parities, color='lightcoral', alpha=0.7)
        axes[1].axhline(y=0.0, color='green', linestyle='-', alpha=0.5, label='Perfect Fairness (0.0)')
        axes[1].axhline(y=0.1, color='red', linestyle='--', label='Threshold (+0.1)')
        axes[1].axhline(y=-0.1, color='red', linestyle='--', label='Threshold (-0.1)')
        axes[1].set_title('Statistical Parity Difference by Model')
        axes[1].set_ylabel('Statistical Parity Difference')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, statistical_parities):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, 
                        height + 0.01 if height >= 0 else height - 0.02, 
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fairness comparison plot saved to {save_path}")
        
        plt.close('all')  # Close figure to free memory
        
        # Print composite scores summary
        print("\nCOMPOSITE SCORES (Performance + Fairness):")
        for name, score in zip(model_names, composite_scores):
            print(f"   {name:20}: {score:.3f}")