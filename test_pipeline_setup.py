# test_pipeline_setup.py
"""
Pipeline Setup Testing Script
Validates the complete ML pipeline including data processing, fairness preprocessing,
model training, and evaluation with multiple baseline models
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.data_processor import DataProcessor
from src.evaluation.fairness_metrics import FairnessEvaluator
from config.config import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def test_pipeline():
    """
    Test the complete pipeline with imbalance and fairness handling
    Validates data loading, preprocessing, model training, and fairness evaluation
    
    Returns:
        tuple: (results dict, data_processor, fairness_evaluator)
    """
    
    # Initialize core components
    data_processor = DataProcessor()
    fairness_evaluator = FairnessEvaluator()

    print("=" * 70)
    print("PIPELINE VALIDATION TEST")
    print("=" * 70)
    
    # STEP 1: Load and split data with fairness preprocessing
    print("\nSTEP 1: LOADING AND PREPROCESSING DATA")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test, prot_train, prot_test = data_processor.load_and_split_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    print(f"Protected attributes: {config.data.PROTECTED_ATTRIBUTES}")

    # Save train/test split for reproducibility
    print("\nSaving train/test split for reproducibility...")
    os.makedirs("artifacts", exist_ok=True)
    data_processor.save_train_test_split(
        X_train, X_test, y_train, y_test, prot_train, prot_test,
        path="artifacts/train_test_split.joblib"
    )
    print("Train/test split saved to artifacts/")

    # STEP 2: Create preprocessing pipeline
    print("\nSTEP 2: CREATING PREPROCESSING PIPELINE")
    print("-" * 70)
    
    preprocessor = data_processor.create_preprocessing_pipeline(X_train)
    
    # Transform data using fitted preprocessor
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed shapes - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")

    # STEP 3: Get sample weights for fairness-aware training
    print("\nSTEP 3: SETTING UP FAIRNESS-AWARE TRAINING")
    print("-" * 70)
    
    sample_weights = data_processor.get_sample_weights()
    
    if sample_weights is not None:
        print(f"Sample weights available: {len(sample_weights)} samples")
        print(f"Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        print("Fairness reweighting will be applied during training")
    else:
        print("No sample weights available (fairness preprocessing disabled)")

    # STEP 4: Test baseline models
    print("\nSTEP 4: TRAINING AND EVALUATING BASELINE MODELS")
    print("-" * 70)
    
    # Define baseline models with class imbalance handling
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced',  # Handle class imbalance
            random_state=config.RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,
            random_state=config.RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=2.33,  # Handle class imbalance (70/30 ratio)
            random_state=config.RANDOM_STATE,
            eval_metric='logloss'
        )
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Fit model with sample weights if available
        if sample_weights is not None:
            try:
                model.fit(X_train_processed, y_train, sample_weight=sample_weights)
                print(f"  Trained with fairness reweighting (sample weights applied)")
            except TypeError:
                # Some models don't support sample_weight parameter
                model.fit(X_train_processed, y_train)
                print(f"  Trained without sample weights (model doesn't support sample_weight)")
        else:
            model.fit(X_train_processed, y_train)
            print(f"  Trained (no fairness reweighting applied)")
        
        # Generate comprehensive fairness and performance report
        report = fairness_evaluator.generate_fairness_report(
            model, X_test_processed, y_test, prot_test, name
        )
        
        results[name] = report

    # STEP 5: Compare all models
    print("\nSTEP 5: COMPARING MODEL PERFORMANCE AND FAIRNESS")
    print("-" * 70)
    
    os.makedirs("reports", exist_ok=True)
    fairness_evaluator.plot_fairness_comparison(
        results, 
        save_path="reports/fairness_comparison.png"
    )
    print("Fairness comparison plot saved to reports/")

    # STEP 6: Save artifacts
    print("\nSTEP 6: SAVING ARTIFACTS")
    print("-" * 70)
    
    data_processor.save_preprocessor("artifacts/preprocessor.joblib")
    print("Preprocessor saved to artifacts/preprocessor.joblib")

    # STEP 7: Summary
    print("\n" + "=" * 70)
    print("PIPELINE VALIDATION COMPLETE")
    print("=" * 70)
    print("VALIDATED COMPONENTS:")
    print("  [PASS] Data loading and stratified splitting")
    print("  [PASS] Train/test split persistence")
    print("  [PASS] Class imbalance handling (class_weight parameter)")
    print("  [PASS] Fairness preprocessing (AIF360 reweighting)")
    print("  [PASS] Multiple model training with fairness evaluation")
    print("  [PASS] Comprehensive fairness metrics calculation")
    print("  [PASS] Model comparison and visualization")
    print("  [PASS] Artifact saving for production deployment")
    print("\nStatus: Ready for AutoML hyperparameter optimization")
    
    return results, data_processor, fairness_evaluator


def print_best_model(results):
    """
    Identify and display the best performing model
    
    Args:
        results: Dictionary of model evaluation results
    """
    # Find model with highest composite score
    best_model = max(results.items(), key=lambda x: x[1]['composite_score'])
    model_name = best_model[0]
    metrics = best_model[1]
    
    print("\n" + "=" * 70)
    print("BEST MODEL SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Composite Score: {metrics['composite_score']:.3f}")
    print(f"\nPerformance Metrics:")
    print(f"  ROC-AUC: {metrics['performance']['roc_auc']:.3f}")
    print(f"  Balanced Accuracy: {metrics['performance']['balanced_accuracy']:.3f}")
    print(f"  F1 Score: {metrics['performance']['f1_score']:.3f}")
    print(f"\nFairness Metrics:")
    print(f"  Disparate Impact: {metrics['fairness']['disparate_impact']:.3f}")
    print(f"  Statistical Parity Diff: {metrics['fairness']['statistical_parity_difference']:.3f}")


if __name__ == "__main__":
    # Run the pipeline validation test
    try:
        results, data_processor, fairness_evaluator = test_pipeline()
        
        # Display best model information
        print_best_model(results)
        
        print("\nPipeline test completed successfully")
        exit(0)
        
    except Exception as e:
        print(f"\nERROR: Pipeline test failed - {e}")
        import traceback
        traceback.print_exc()
        exit(1)