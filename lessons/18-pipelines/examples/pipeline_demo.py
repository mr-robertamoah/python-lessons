#!/usr/bin/env python3
"""
ML Pipelines Demo
Comprehensive pipeline examples
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

def create_sample_data():
    """Create sample dataset for pipeline demo"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master'], n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], n_samples),
        'experience': np.random.randint(0, 40, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable
    target_prob = (
        0.1 + 
        0.3 * (df['age'] > 40) + 
        0.2 * (df['income'] > 50000) +
        0.1 * (df['education'] == 'Master')
    )
    df['approved'] = np.random.binomial(1, target_prob)
    
    return df

def basic_pipeline_demo():
    """Demonstrate basic pipeline construction"""
    print("=== Basic Pipeline Demo ===")
    
    df = create_sample_data()
    
    # Separate features and target
    X = df[['age', 'income', 'experience']]
    y = df['approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=2)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    print(f"Pipeline accuracy: {pipeline.score(X_test, y_test):.3f}")
    print(f"Selected features: {X.columns[pipeline.named_steps['selector'].get_support()]}")

def column_transformer_demo():
    """Demonstrate ColumnTransformer for mixed data types"""
    print("\n=== Column Transformer Demo ===")
    
    df = create_sample_data()
    
    # Define feature types
    numeric_features = ['age', 'income', 'experience']
    categorical_features = ['education', 'city']
    
    X = df[numeric_features + categorical_features]
    y = df['approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Train and evaluate
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    
    print(f"Mixed data pipeline accuracy: {accuracy:.3f}")
    
    # Show feature names after preprocessing
    feature_names = (numeric_features + 
                    list(pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical_features)))
    print(f"Total features after preprocessing: {len(feature_names)}")

class CustomTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer example"""
    
    def __init__(self, feature_name):
        self.feature_name = feature_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy[f'{self.feature_name}_log'] = np.log1p(X_copy[self.feature_name])
        return X_copy

def custom_transformer_demo():
    """Demonstrate custom transformer in pipeline"""
    print("\n=== Custom Transformer Demo ===")
    
    df = create_sample_data()
    
    X = df[['age', 'income', 'experience']]
    y = df['approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline with custom transformer
    pipeline = Pipeline([
        ('custom', CustomTransformer('income')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Train and evaluate
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    
    print(f"Custom transformer pipeline accuracy: {accuracy:.3f}")
    print(f"Features after custom transform: {pipeline.named_steps['custom'].transform(X_train).columns.tolist()}")

def hyperparameter_tuning_demo():
    """Demonstrate hyperparameter tuning with pipelines"""
    print("\n=== Hyperparameter Tuning Demo ===")
    
    df = create_sample_data()
    
    X = df[['age', 'income', 'experience']]
    y = df['approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'selector__k': [1, 2, 3],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5, None]
    }
    
    # Grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    print(f"Test set accuracy: {grid_search.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    basic_pipeline_demo()
    column_transformer_demo()
    custom_transformer_demo()
    hyperparameter_tuning_demo()
