# run_automl.py
"""
Fairness-Aware AutoML Pipeline
Complete pipeline for training, evaluating, and deploying fair credit risk models
Includes preprocessing, hyperparameter optimization, fairness post-processing, and artifact saving
"""

import sys
import os
import pandas as pd
import joblib
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score
)

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.automl_tuner import FairnessAutoMLTuner
from src.preprocessing.data_processor import DataProcessor
from src.evaluation.fairness_metrics import FairnessEvaluator
from src.evaluation.fairness_postprocessing import ThresholdOptimizer
from src.models.fair_model_wrapper import FairModelWrapper
from config.config import config


def main():
    """
    Main AutoML pipeline execution function
    Orchestrates the complete workflow from data loading to model deployment
    """
    print("=" * 70)
    print("FAIRNESS-AWARE AUTOML FOR CREDIT RISK SCORING")
    print("=" * 70)
    
    # Initialize core components
    data_processor = DataProcessor()
    fairness_evaluator = FairnessEvaluator()
    
    # STEP 1: Load or create train/test split
    print("\nSTEP 1: LOADING TRAIN/TEST SPLIT")
    print("-" * 70)
    try:
        # Attempt to load saved split for reproducibility
        X_train, X_test, y_train, y_test, prot_train, prot_test = data_processor.load_train_test_split(
            "artifacts/train_test_split.joblib"
        )
        print("Using saved train/test split (ensures reproducibility)")
    except FileNotFoundError:
        # Create new split if saved version doesn't exist
        print("No saved split found - creating new train/test split")
        X_train, X_test, y_train, y_test, prot_train, prot_test = data_processor.load_and_split_data()
        os.makedirs("artifacts", exist_ok=True)
        data_processor.save_train_test_split(
            X_train, X_test, y_train, y_test, prot_train, prot_test
        )
    
    # STEP 2: Load or create preprocessor
    print("\nSTEP 2: LOADING PREPROCESSOR")
    print("-" * 70)
    try:
        # Load saved preprocessor
        preprocessor_data = joblib.load("artifacts/preprocessor.joblib")
        preprocessor = preprocessor_data['preprocessor']
        print("Preprocessor loaded successfully")
    except FileNotFoundError:
        # Create and save new preprocessor
        print("No saved preprocessor found - creating new preprocessor")
        preprocessor = data_processor.create_preprocessing_pipeline(X_train)
        data_processor.save_preprocessor("artifacts/preprocessor.joblib")
    
    # Transform training and test data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Training data shape: {X_train_processed.shape}")
    print(f"Test data shape: {X_test_processed.shape}")
    
    # Check for fairness-aware sample weights
    sample_weights = data_processor.get_sample_weights()
    if sample_weights is not None:
        print(f"Sample weights available: {len(sample_weights)} samples")
        print(f"Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
    else:
        print("No sample weights available (fairness reweighting not applied)")
    
    # STEP 3: AutoML hyperparameter optimization
    print("\nSTEP 3: FAIRNESS-AWARE AUTOML OPTIMIZATION")
    print("-" * 70)
    print(f"Models to evaluate: {config.model.MODELS_TO_TRY}")
    print(f"Number of trials: {config.N_TRIALS}")
    print(f"Fairness weight: {config.model.FAIRNESS_WEIGHT}")
    
    # Initialize AutoML tuner with fairness awareness
    automl_tuner = FairnessAutoMLTuner(
        fairness_weight=config.model.FAIRNESS_WEIGHT,
        n_trials=config.N_TRIALS,
        cv_folds=3
    )
    
    # Run optimization to find best model and hyperparameters
    best_model, best_params = automl_tuner.optimize(
        X_train_processed, 
        y_train.values if hasattr(y_train, 'values') else y_train,
        prot_train, 
        sample_weights
    )

    # STEP 4: Apply fairness post-processing
    print("\nSTEP 4: APPLYING FAIRNESS POST-PROCESSING")
    print("-" * 70)
    
    # Get probability predictions on training data
    y_train_proba = best_model.predict_proba(X_train_processed)[:, 1]

    # Optimize classification thresholds for equal opportunity
    threshold_optimizer = ThresholdOptimizer(protected_attribute='gender')
    threshold_optimizer.fit(
        y_train.values if hasattr(y_train, 'values') else y_train,
        y_train_proba,
        prot_train,
        target_metric='equal_opportunity'
    )

    # Save threshold optimizer
    joblib.dump(threshold_optimizer, "artifacts/threshold_optimizer.joblib")
    print("Threshold optimizer saved to artifacts/")

    # STEP 5: Evaluate model with fairness post-processing
    print("\nSTEP 5: EVALUATING MODEL ON TEST SET")
    print("-" * 70)

    # Create fair model wrapper combining base model and threshold optimizer
    fair_model = FairModelWrapper(best_model, threshold_optimizer)

    print("Evaluation with fairness post-processing:")

    # Generate predictions with adjusted thresholds
    y_test_proba = best_model.predict_proba(X_test_processed)[:, 1]
    y_test_pred_adjusted = threshold_optimizer.predict(y_test_proba, prot_test)

    # Calculate performance metrics
    final_performance = {
        'accuracy': accuracy_score(y_test, y_test_pred_adjusted),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred_adjusted),
        'precision': precision_score(y_test, y_test_pred_adjusted, zero_division=0),
        'recall': recall_score(y_test, y_test_pred_adjusted, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred_adjusted, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'average_precision': average_precision_score(y_test, y_test_proba)
    }

    # Calculate fairness metrics
    final_fairness = fairness_evaluator.calculate_fairness_metrics(
        y_test, y_test_pred_adjusted, prot_test
    )

    # Calculate composite score balancing performance and fairness
    final_composite = fairness_evaluator.calculate_composite_score(
        final_performance, final_fairness
    )

    # Display comprehensive evaluation report
    print("\n" + "=" * 70)
    print("FINAL EVALUATION REPORT: AutoML + Fairness Post-processing")
    print("=" * 70)

    print("\nPERFORMANCE METRICS:")
    for metric, value in final_performance.items():
        print(f"   {metric:20}: {value:.3f}")

    print("\nFAIRNESS METRICS:")
    for metric, value in final_fairness.items():
        # Check if metric passes fairness threshold
        status = "PASS" if fairness_evaluator._is_fair(metric, value) else "FAIL"
        print(f"   [{status}] {metric:30}: {value:.3f}")

    print(f"\nCOMPOSITE SCORE: {final_composite:.3f}")

    # Overall fairness assessment
    all_fair = all(fairness_evaluator._is_fair(m, v) for m, v in final_fairness.items())
    if all_fair:
        print("\nRESULT: Model meets all fairness criteria")
    else:
        print("\nWARNING: Model may need additional fairness mitigation")

    # Compile final report
    final_report = {
        'performance': final_performance,
        'fairness': final_fairness,
        'composite_score': final_composite
    }

    # STEP 6: Analyze optimization process
    print("\nSTEP 6: ANALYZING OPTIMIZATION PROCESS")
    print("-" * 70)
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Generate optimization history plot
    automl_tuner.plot_optimization_history("reports/automl_optimization_history.png")

    # Get optimization history data
    optimization_history = automl_tuner.get_optimization_history()

    # Display top performing trials
    print("\nTOP 5 TRIALS:")
    top_trials = optimization_history.nlargest(5, 'score')[['trial_number', 'model_type', 'score']]
    print(top_trials.to_string(index=False))

    # Analyze performance by model type
    print("\nMODEL TYPE PERFORMANCE SUMMARY:")
    model_stats = optimization_history.groupby('model_type')['score'].agg(['mean', 'max', 'count'])
    print(model_stats)

    # STEP 7: Save all artifacts
    print("\nSTEP 7: SAVING ARTIFACTS")
    print("-" * 70)
    
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)

    # Save models and optimizers
    joblib.dump(fair_model, "artifacts/fair_model_complete.joblib")
    joblib.dump(best_model, "artifacts/best_automl_model.joblib")
    joblib.dump(threshold_optimizer, "artifacts/threshold_optimizer.joblib")
    joblib.dump(best_params, "artifacts/best_parameters.joblib")
    joblib.dump(final_report, "artifacts/final_model_report.joblib")

    # Save optimization history
    optimization_history.to_csv("reports/automl_optimization_history.csv", index=False)

    # Create comprehensive summary report
    summary_report = {
        'best_model_type': best_params['model_type'],
        'best_parameters': best_params,
        'composite_score': final_composite,
        'performance_metrics': final_performance,
        'fairness_metrics': final_fairness,
        'total_trials': len(optimization_history),
        'data_shape': {
            'train': X_train_processed.shape,
            'test': X_test_processed.shape
        }
    }

    # Save summary as JSON
    with open("reports/automl_summary.json", 'w') as f:
        json.dump(summary_report, f, indent=4, default=str)

    print("All artifacts saved successfully")
    print("   - Models: artifacts/")
    print("   - Reports: reports/")

    # FINAL SUMMARY
    print("\n" + "=" * 70)
    print("AUTOML PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Best Model: {best_params['model_type'].upper()}")
    print(f"Composite Score: {final_composite:.3f}")
    print(f"Fairness Status: {'COMPLIANT' if all_fair else 'NEEDS REVIEW'}")
    print("\nAll results saved to artifacts/ and reports/ directories")
    
    return fair_model, best_params, final_report, optimization_history


if __name__ == "__main__":
    # Execute the AutoML pipeline
    fair_model, best_params, final_report, optimization_history = main()