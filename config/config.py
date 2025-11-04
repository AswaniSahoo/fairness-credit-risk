# config/config.py
"""
Configuration module for AutoML Fairness-Aware System
Contains all configuration settings for data processing, model training, and fairness evaluation
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DataConfig:
    """
    Data-related configuration settings
    Handles paths, target columns, and preprocessing strategies
    """
    
    # File paths for raw and processed data
    RAW_DATA_PATH: str = "/home/aswani/automl/data/raw/german.data"
    PROCESSED_DATA_PATH: str = "/home/aswani/automl/data/processed/german_credit_numerical_final.csv"
    
    # Target variable for prediction
    TARGET_COLUMN: str = "credit_risk"
    
    # Protected attributes for fairness evaluation (gender, age, etc.)
    PROTECTED_ATTRIBUTES: List[str] = None
    
    # Strategy to handle class imbalance in dataset
    # Options: 'smote', 'class_weight', 'undersampling', 'none'
    CLASS_IMBALANCE_STRATEGY: str = 'class_weight'
    
    # Apply fairness-aware preprocessing techniques
    APPLY_FAIRNESS_PREPROCESSING: bool = True
    
    # Method for reweighting samples to mitigate bias
    # Options: 'aif360' or 'manual'
    REWEIGHTING_METHOD: str = 'aif360'
    
    def __post_init__(self):
        """Initialize protected attributes if not provided"""
        if self.PROTECTED_ATTRIBUTES is None:
            self.PROTECTED_ATTRIBUTES = ['gender', 'age', 'foreign_worker']


@dataclass
class ModelConfig:
    """
    Model training and hyperparameter configuration
    Defines which models to train and their parameter spaces
    """
    
    # List of machine learning models to evaluate
    MODELS_TO_TRY: List[str] = None
    
    # Hyperparameter search space for each model
    HYPERPARAMETER_SPACE: Dict[str, Any] = None
    
    # Weight given to fairness metrics vs performance metrics (0 to 1)
    FAIRNESS_WEIGHT: float = 0.3
    
    # Evaluation metrics specifically designed for imbalanced datasets
    IMBALANCE_METRICS: List[str] = None
    
    def __post_init__(self):
        """Initialize default models and hyperparameter spaces if not provided"""
        
        if self.MODELS_TO_TRY is None:
            self.MODELS_TO_TRY = [
                'random_forest', 
                'xgboost', 
                'logistic_regression', 
                'lightgbm'
            ]
        
        if self.HYPERPARAMETER_SPACE is None:
            self.HYPERPARAMETER_SPACE = {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced', None]
                },
                'xgboost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    # scale_pos_weight: ratio of negative to positive class (70/30 = 2.33)
                    'scale_pos_weight': [1, 2.33]
                },
                'logistic_regression': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'class_weight': ['balanced', None]
                },
                'lightgbm': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'class_weight': ['balanced', None]
                }
            }
        
        if self.IMBALANCE_METRICS is None:
            self.IMBALANCE_METRICS = [
                'balanced_accuracy',  # Accuracy adjusted for class imbalance
                'f1',                 # Harmonic mean of precision and recall
                'precision',          # True positives / (True positives + False positives)
                'recall',             # True positives / (True positives + False negatives)
                'roc_auc',           # Area under ROC curve
                'average_precision'   # Area under precision-recall curve
            ]


@dataclass
class FairnessConfig:
    """
    Fairness evaluation configuration
    Defines fairness metrics and threshold values
    """
    
    # Primary protected attribute to focus fairness analysis on
    PRIMARY_PROTECTED_ATTRIBUTE: str = 'gender'
    
    # List of fairness metrics to calculate
    FAIRNESS_METRICS: List[str] = None
    
    # Minimum acceptable ratio for disparate impact (0.8 = 80% rule)
    DISPARATE_IMPACT_THRESHOLD: float = 0.8
    
    # Maximum acceptable difference in statistical parity
    STATISTICAL_PARITY_THRESHOLD: float = 0.1
    
    def __post_init__(self):
        """Initialize default fairness metrics if not provided"""
        if self.FAIRNESS_METRICS is None:
            self.FAIRNESS_METRICS = [
                'disparate_impact',                # Ratio of favorable outcomes between groups
                'statistical_parity_difference',   # Difference in positive prediction rates
                'equal_opportunity_difference'     # Difference in true positive rates
            ]


@dataclass
class Config:
    """
    Main configuration class that combines all config components
    """
    
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    fairness: FairnessConfig = FairnessConfig()
    
    # Random seed for reproducibility
    RANDOM_STATE: int = 42
    
    # Proportion of data to use for testing
    TEST_SIZE: float = 0.2
    
    # Number of optimization trials for Optuna hyperparameter search
    N_TRIALS: int = 50


# Global configuration instance - import this in other modules
config = Config()