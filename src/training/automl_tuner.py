"""
Fairness-Aware AutoML Hyperparameter Tuner
Uses Optuna for optimization while balancing model performance and fairness metrics
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from config.config import config
from src.evaluation.fairness_metrics import FairnessEvaluator


class FairnessAutoMLTuner:
    """
    AutoML tuner that optimizes both model performance and fairness
    Uses Optuna for Bayesian hyperparameter optimization with cross-validation
    """
    
    def __init__(self, fairness_weight=0.3, n_trials=50, cv_folds=3):
        """
        Initialize the AutoML tuner
        
        Args:
            fairness_weight: Weight given to fairness metrics (0-1, default 0.3)
            n_trials: Number of optimization trials to run
            cv_folds: Number of cross-validation folds
        """
        self.fairness_weight = fairness_weight
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.fairness_evaluator = FairnessEvaluator()
        self.study = None
        self.best_model = None
        self.best_score = None
        self.trial_results = []
        
    def objective(self, trial, X, y, protected_attributes, sample_weights=None):
        """
        Optuna objective function that balances performance and fairness
        Evaluates a model configuration using cross-validation
        
        Args:
            trial: Optuna trial object for hyperparameter suggestions
            X: Training features
            y: Training labels
            protected_attributes: Protected attribute values (gender, age, etc.)
            sample_weights: Optional sample weights from fairness preprocessing
            
        Returns:
            float: Composite score (higher is better)
        """
        
        # Let Optuna suggest which model type to try
        model_type = trial.suggest_categorical('model_type', config.model.MODELS_TO_TRY)
        
        # Create model with suggested hyperparameters based on type
        if model_type == 'random_forest':
            model = self._suggest_rf_params(trial)
        elif model_type == 'xgboost':
            model = self._suggest_xgb_params(trial)
        elif model_type == 'lightgbm':
            model = self._suggest_lgbm_params(trial)
        elif model_type == 'logistic_regression':
            model = self._suggest_lr_params(trial)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform cross-validation with fairness evaluation
        cv_scores = []
        
        # Use stratified K-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                             random_state=config.RANDOM_STATE)
        
        for train_idx, val_idx in skf.split(X, y):
            # Split data for this fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            prot_train = protected_attributes.iloc[train_idx]
            prot_val = protected_attributes.iloc[val_idx]
            
            # Get sample weights for this fold if fairness preprocessing was applied
            fold_sample_weights = (sample_weights[train_idx] 
                                 if sample_weights is not None else None)
            
            # Train model with error handling
            try:
                if fold_sample_weights is not None and hasattr(model, 'fit'):
                    model.fit(X_train, y_train, sample_weight=fold_sample_weights)
                else:
                    model.fit(X_train, y_train)
            except Exception as e:
                # If training fails, return poor score to skip this configuration
                return 0.0
            
            # Generate predictions on validation set
            y_pred = model.predict(X_val)
            y_proba = (model.predict_proba(X_val)[:, 0] 
                      if hasattr(model, 'predict_proba') else None)
            
            # Calculate performance metrics (accuracy, F1, AUC, etc.)
            perf_metrics = self.fairness_evaluator.calculate_performance_metrics(
                y_val, y_pred, y_proba
            )
            
            # Calculate fairness metrics (disparate impact, statistical parity, etc.)
            fair_metrics = self.fairness_evaluator.calculate_fairness_metrics(
                y_val, y_pred, prot_val
            )
            
            # Combine performance and fairness into a single score
            fold_score = self.fairness_evaluator.calculate_composite_score(
                perf_metrics, fair_metrics, self.fairness_weight
            )
            
            cv_scores.append(fold_score)
        
        # Return average score across all folds
        mean_score = np.mean(cv_scores)
        
        # Store trial information for later analysis
        self.trial_results.append({
            'trial_number': trial.number,
            'model_type': model_type,
            'score': mean_score,
            'params': trial.params
        })
        
        return mean_score
    
    def _suggest_rf_params(self, trial):
        """
        Suggest Random Forest hyperparameters for this trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            RandomForestClassifier: Model with suggested parameters
        """
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_n_estimators', 100, 500),
            max_depth=trial.suggest_int('rf_max_depth', 5, 30),
            min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
            class_weight='balanced',  # Handle class imbalance
            random_state=config.RANDOM_STATE,
            n_jobs=-1  # Use all CPU cores
        )
    
    def _suggest_xgb_params(self, trial):
        """
        Suggest XGBoost hyperparameters for this trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            XGBClassifier: Model with suggested parameters
        """
        return XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_estimators', 100, 500),
            max_depth=trial.suggest_int('xgb_max_depth', 3, 10),
            learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('xgb_subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            scale_pos_weight=2.33,  # Handle 70-30 class imbalance
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    def _suggest_lgbm_params(self, trial):
        """
        Suggest LightGBM hyperparameters for this trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            LGBMClassifier: Model with suggested parameters
        """
        return LGBMClassifier(
            n_estimators=trial.suggest_int('lgbm_n_estimators', 100, 500),
            max_depth=trial.suggest_int('lgbm_max_depth', 3, 15),
            learning_rate=trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int('lgbm_num_leaves', 20, 100),
            subsample=trial.suggest_float('lgbm_subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
            class_weight='balanced',  # Handle class imbalance
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
    
    def _suggest_lr_params(self, trial):
        """
        Suggest Logistic Regression hyperparameters for this trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            LogisticRegression: Model with suggested parameters
        """
        penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2', 'elasticnet'])
        
        params = {
            'C': trial.suggest_float('lr_C', 0.001, 10.0, log=True),  # Inverse regularization strength
            'penalty': penalty,
            'solver': 'saga',  # Supports all penalty types
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': config.RANDOM_STATE
        }
        
        # ElasticNet requires l1_ratio parameter
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('lr_l1_ratio', 0.0, 1.0)
        
        return LogisticRegression(**params)
    
    def optimize(self, X, y, protected_attributes, sample_weights=None):
        """
        Run the complete AutoML optimization process
        
        Args:
            X: Training features
            y: Training labels
            protected_attributes: Protected attribute values
            sample_weights: Optional sample weights from preprocessing
            
        Returns:
            tuple: (best_model, best_params)
        """
        print("Starting Fairness-Aware AutoML Optimization...")
        print(f"Models to evaluate: {config.model.MODELS_TO_TRY}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Cross-validation folds: {self.cv_folds}")
        print(f"Fairness weight: {self.fairness_weight}")
        
        # Create Optuna study with Tree-structured Parzen Estimator sampler
        self.study = optuna.create_study(
            direction='maximize',  # Maximize composite score
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
        )
        
        # Run optimization trials
        self.study.optimize(
            lambda trial: self.objective(trial, X, y, protected_attributes, sample_weights),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Train best model on complete training dataset
        print("\nTraining best model on full dataset...")
        self.best_model = self._train_best_model(X, y, sample_weights)
        self.best_score = self.study.best_value
        
        print(f"AutoML optimization complete!")
        print(f"Best composite score: {self.best_score:.3f}")
        print(f"Best model type: {self.study.best_params['model_type']}")
        
        return self.best_model, self.study.best_params
    
    def _train_best_model(self, X, y, sample_weights):
        """
        Train the best model configuration on the full training dataset
        
        Args:
            X: Full training features
            y: Full training labels
            sample_weights: Optional sample weights
            
        Returns:
            Trained model with best hyperparameters
        """
        best_params = self.study.best_params
        model_type = best_params['model_type']
        
        # Reconstruct the best model with optimal hyperparameters
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=best_params['rf_n_estimators'],
                max_depth=best_params['rf_max_depth'],
                min_samples_split=best_params['rf_min_samples_split'],
                min_samples_leaf=best_params['rf_min_samples_leaf'],
                class_weight='balanced',
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=best_params['xgb_n_estimators'],
                max_depth=best_params['xgb_max_depth'],
                learning_rate=best_params['xgb_learning_rate'],
                subsample=best_params.get('xgb_subsample', 1.0),
                colsample_bytree=best_params.get('xgb_colsample_bytree', 1.0),
                scale_pos_weight=2.33,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif model_type == 'lightgbm':
            model = LGBMClassifier(
                n_estimators=best_params['lgbm_n_estimators'],
                max_depth=best_params['lgbm_max_depth'],
                learning_rate=best_params['lgbm_learning_rate'],
                num_leaves=best_params['lgbm_num_leaves'],
                subsample=best_params.get('lgbm_subsample', 1.0),
                colsample_bytree=best_params.get('lgbm_colsample_bytree', 1.0),
                class_weight='balanced',
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                C=best_params['lr_C'],
                penalty=best_params['lr_penalty'],
                solver='saga',
                class_weight='balanced',
                max_iter=1000,
                random_state=config.RANDOM_STATE
            )
        
        # Train with sample weights if available
        if sample_weights is not None and hasattr(model, 'fit'):
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)
            
        return model
    
    def get_optimization_history(self):
        """
        Get optimization history as a DataFrame for analysis
        
        Returns:
            pd.DataFrame: Trial history with scores and parameters
        """
        return pd.DataFrame(self.trial_results)
    
    def plot_optimization_history(self, save_path=None):
        """
        Visualize the optimization process
        Creates two plots: score progression and model type distribution
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if not self.trial_results:
            print("No optimization results available")
            return
        
        history_df = self.get_optimization_history()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Score progression over trials
        ax1.plot(history_df['trial_number'], history_df['score'], 'b-', alpha=0.5)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Composite Score')
        ax1.set_title('AutoML Optimization Progress')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of model types explored
        model_counts = history_df['model_type'].value_counts()
        ax2.bar(model_counts.index, model_counts.values, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('Number of Trials')
        ax2.set_title('Model Types Explored')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization history plot saved to {save_path}")
        
        plt.show()