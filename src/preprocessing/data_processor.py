"""
Data Processing Module
Handles data loading, preprocessing, train-test splitting, and fairness-aware transformations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
import joblib

from config.config import config


class DataProcessor:
    """
    Main class for data preprocessing and transformation
    Supports fairness-aware preprocessing through reweighting techniques
    """
    
    def __init__(self):
        """Initialize data processor with default attributes"""
        self.preprocessor = None
        self.feature_columns = None
        self.protected_columns = config.data.PROTECTED_ATTRIBUTES
        self.sample_weights = None

    def load_and_split_data(self):
        """
        Load dataset and split into training and testing sets
        Applies stratified splitting based on target and protected attributes
        
        Returns:
            tuple: X_train, X_test, y_train, y_test, prot_train, prot_test
        """
        print("Loading and preparing data...")

        # Load processed data from CSV
        df = pd.read_csv(config.data.PROCESSED_DATA_PATH)

        # Remove age_group if present (keeping only numeric age column)
        if 'age_group' in df.columns:
            df = df.drop(columns=['age_group'])
            print("Dropped 'age_group' column (non-numeric, redundant with age)")

        # Separate features from protected attributes and target
        feature_columns = [
            col for col in df.columns
            if col not in self.protected_columns + [config.data.TARGET_COLUMN]
        ]

        X = df[feature_columns]
        y = df[config.data.TARGET_COLUMN]
        protected_attributes = df[self.protected_columns]

        # Create stratification key combining target and primary protected attribute
        # This ensures balanced representation across both dimensions
        stratification_data = (
            y.astype(str) + "_" + 
            protected_attributes[config.fairness.PRIMARY_PROTECTED_ATTRIBUTE].astype(str)
        )

        # Perform stratified train-test split
        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
            X, y, protected_attributes,
            test_size=config.TEST_SIZE,
            stratify=stratification_data,
            random_state=config.RANDOM_STATE
        )

        self.feature_columns = feature_columns

        # Display split statistics
        print(f"Data split complete: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print(f"Class distribution in training set: {y_train.value_counts().to_dict()}")

        # Apply fairness preprocessing if enabled
        if config.data.APPLY_FAIRNESS_PREPROCESSING:
            self.sample_weights = self._apply_reweighting(X_train, y_train, prot_train)

        return X_train, X_test, y_train, y_test, prot_train, prot_test

    def _apply_reweighting(self, X_train, y_train, prot_train):
        """
        Apply AIF360 reweighting algorithm to mitigate bias in training data
        Assigns weights to samples to balance representation across protected groups
        
        Args:
            X_train: Training features
            y_train: Training labels
            prot_train: Protected attributes for training set
            
        Returns:
            np.array: Sample weights for each training instance
        """
        print("Applying fairness preprocessing through reweighting...")

        # Reconstruct dataframe with features, target, and protected attribute
        df_train = pd.DataFrame(X_train, columns=self.feature_columns)
        df_train[config.data.TARGET_COLUMN] = y_train.values
        df_train[config.fairness.PRIMARY_PROTECTED_ATTRIBUTE] = (
            prot_train[config.fairness.PRIMARY_PROTECTED_ATTRIBUTE].values
        )

        # Convert to AIF360 StandardDataset format
        aif_dataset = StandardDataset(
            df=df_train,
            label_name=config.data.TARGET_COLUMN,
            favorable_classes=[0],  # 0 represents good credit risk
            protected_attribute_names=[config.fairness.PRIMARY_PROTECTED_ATTRIBUTE],
            privileged_classes=[[1]]  # 1 represents privileged group
        )

        # Initialize and apply reweighting algorithm
        reweighter = Reweighing(
            unprivileged_groups=[{config.fairness.PRIMARY_PROTECTED_ATTRIBUTE: 0}],
            privileged_groups=[{config.fairness.PRIMARY_PROTECTED_ATTRIBUTE: 1}]
        )

        transformed_dataset = reweighter.fit_transform(aif_dataset)
        sample_weights = transformed_dataset.instance_weights

        print(f"Reweighting complete. Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

        return sample_weights

    def create_preprocessing_pipeline(self, X_train):
        """
        Create and fit sklearn preprocessing pipeline
        Handles missing values through imputation and scales numerical features
        
        Args:
            X_train: Training features to fit the pipeline
            
        Returns:
            ColumnTransformer: Fitted preprocessing pipeline
        """
        print("Creating preprocessing pipeline...")

        # Identify numerical features for transformation
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Define transformation pipeline for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
            ('scaler', StandardScaler())  # Standardize features (mean=0, std=1)
        ])

        # Combine transformers using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features)
            ],
            remainder='passthrough'  # Keep other columns unchanged
        )

        # Fit the preprocessor on training data
        self.preprocessor.fit(X_train)
        
        print(f"Preprocessor fitted on {len(numerical_features)} numerical features")
        print(f"Class imbalance will be handled via model class_weight parameter")

        return self.preprocessor

    def transform_data(self, X):
        """
        Transform data using the fitted preprocessing pipeline
        
        Args:
            X: Features to transform
            
        Returns:
            np.array: Transformed features
            
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call create_preprocessing_pipeline first.")
        return self.preprocessor.transform(X)

    def get_sample_weights(self):
        """
        Retrieve sample weights from fairness reweighting
        
        Returns:
            np.array: Sample weights or None if reweighting wasn't applied
        """
        return self.sample_weights

    def save_preprocessor(self, path="artifacts/preprocessor.joblib"):
        """
        Save the fitted preprocessor to disk for later use
        
        Args:
            path: File path to save the preprocessor
        """
        if self.preprocessor:
            joblib.dump({
                'preprocessor': self.preprocessor,
                'feature_columns': self.feature_columns
            }, path)
            print(f"Preprocessor saved to {path}")

    def load_preprocessor(self, path="artifacts/preprocessor.joblib"):
        """
        Load a previously saved preprocessor from disk
        
        Args:
            path: File path to load the preprocessor from
            
        Returns:
            ColumnTransformer: Loaded preprocessor
        """
        data = joblib.load(path)
        self.preprocessor = data['preprocessor']
        self.feature_columns = data['feature_columns']
        print(f"Preprocessor loaded from {path}")
        return self.preprocessor

    def save_train_test_split(self, X_train, X_test, y_train, y_test,
                              prot_train, prot_test, path="artifacts/train_test_split.joblib"):
        """
        Save train-test split to disk for reproducibility
        Useful for consistent evaluation across different model runs
        
        Args:
            X_train, X_test: Training and testing features
            y_train, y_test: Training and testing labels
            prot_train, prot_test: Protected attributes for both sets
            path: File path to save the split data
        """
        split_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'prot_train': prot_train,
            'prot_test': prot_test,
            'feature_columns': self.feature_columns,
            'protected_columns': self.protected_columns
        }
        joblib.dump(split_data, path)
        print(f"Train/test split saved to {path}")

    def load_train_test_split(self, path="artifacts/train_test_split.joblib"):
        """
        Load previously saved train-test split from disk
        
        Args:
            path: File path to load the split data from
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, prot_train, prot_test
        """
        split_data = joblib.load(path)
        self.feature_columns = split_data['feature_columns']
        self.protected_columns = split_data['protected_columns']

        print(f"Train/test split loaded from {path}")
        print(f"Train shape: {split_data['X_train'].shape}, Test shape: {split_data['X_test'].shape}")

        return (
            split_data['X_train'],
            split_data['X_test'],
            split_data['y_train'],
            split_data['y_test'],
            split_data['prot_train'],
            split_data['prot_test']
        )