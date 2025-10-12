# Lesson 16: ML Pipelines - Automating Machine Learning Workflows

## Learning Objectives
By the end of this lesson, you will be able to:
- Create automated ML pipelines using scikit-learn
- Implement preprocessing and modeling in a single workflow
- Perform hyperparameter tuning systematically
- Save and load trained models for production use
- Build reproducible machine learning workflows
- Handle complex data preprocessing scenarios

## What are ML Pipelines?

### The Problem with Manual Workflows
```python
# Manual approach (error-prone and repetitive):
# 1. Load data
# 2. Split train/test
# 3. Scale features
# 4. Train model
# 5. Make predictions
# 6. Evaluate

# Problems:
# - Easy to forget steps
# - Data leakage risks
# - Hard to reproduce
# - Difficult to deploy
```

### Pipeline Solution
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Automated pipeline:
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# One command does everything:
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Basic Pipelines

### Simple Pipeline Example
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                          n_redundant=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
simple_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Train pipeline
simple_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = simple_pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Pipeline Accuracy: {accuracy:.3f}")

# Access pipeline components
print("Pipeline steps:")
for name, step in simple_pipeline.named_steps.items():
    print(f"  {name}: {step}")
```

### Pipeline with Multiple Preprocessing Steps
```python
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

# Complex preprocessing pipeline
complex_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('selector', SelectKBest(score_func=f_classif, k=10)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Train and evaluate
complex_pipeline.fit(X_train, y_train)
y_pred_complex = complex_pipeline.predict(X_test)
accuracy_complex = accuracy_score(y_test, y_pred_complex)

print(f"Complex Pipeline Accuracy: {accuracy_complex:.3f}")

# Inspect transformed features
X_transformed = complex_pipeline[:-1].transform(X_test)  # All steps except classifier
print(f"Original features: {X_test.shape[1]}")
print(f"Transformed features: {X_transformed.shape[1]}")
```

## Column Transformer for Mixed Data Types

### Handling Different Data Types
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Create mixed dataset
np.random.seed(42)
n_samples = 1000

mixed_data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
    'experience': np.random.randint(0, 30, n_samples),
    'satisfaction': np.random.uniform(1, 10, n_samples)
})

# Add target variable
mixed_data['high_performer'] = (
    (mixed_data['satisfaction'] > 7) & 
    (mixed_data['experience'] > 5)
).astype(int)

# Add some missing values
missing_indices = np.random.choice(n_samples, 100, replace=False)
mixed_data.loc[missing_indices, 'income'] = np.nan

# Separate features and target
X_mixed = mixed_data.drop('high_performer', axis=1)
y_mixed = mixed_data['high_performer']

# Split data
X_train_mixed, X_test_mixed, y_train_mixed, y_test_mixed = train_test_split(
    X_mixed, y_mixed, test_size=0.2, random_state=42)

# Define column types
numeric_features = ['age', 'income', 'experience', 'satisfaction']
categorical_features = ['education', 'department']

# Create preprocessing pipelines for each data type
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train and evaluate
full_pipeline.fit(X_train_mixed, y_train_mixed)
y_pred_mixed = full_pipeline.predict(X_test_mixed)
accuracy_mixed = accuracy_score(y_test_mixed, y_pred_mixed)

print(f"Mixed Data Pipeline Accuracy: {accuracy_mixed:.3f}")

# Get feature names after preprocessing
feature_names = (numeric_features + 
                list(full_pipeline.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names_out(categorical_features)))

print(f"Final feature count: {len(feature_names)}")
print("Feature names:", feature_names[:10], "...")  # Show first 10
```

## Hyperparameter Tuning with Pipelines

### Grid Search with Pipelines
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid for pipeline
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5, 10]
}

# Create grid search
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Perform grid search
print("Performing grid search...")
grid_search.fit(X_train_mixed, y_train_mixed)

# Best parameters and score
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
print("Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate best model on test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test_mixed, y_test_mixed)
print(f"Test accuracy with best model: {test_accuracy:.3f}")
```

### Random Search for Efficiency
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'classifier__n_estimators': randint(50, 300),
    'classifier__max_depth': [5, 10, 15, 20, None],
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 10),
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Random search
random_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Performing random search...")
random_search.fit(X_train_mixed, y_train_mixed)

print(f"Random search best score: {random_search.best_score_:.3f}")
print("Random search best parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
```

## Custom Transformers

### Creating Custom Preprocessing Steps
```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Custom transformer to remove outliers using IQR method"""
    
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def fit(self, X, y=None):
        # Calculate IQR bounds for each feature
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR
        
        return self
    
    def transform(self, X):
        # Remove outliers by clipping values
        X_transformed = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_transformed

class FeatureCreator(BaseEstimator, TransformerMixin):
    """Custom transformer to create interaction features"""
    
    def __init__(self, feature_pairs=None):
        self.feature_pairs = feature_pairs or []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Create interaction features
        for i, j in self.feature_pairs:
            if i < X.shape[1] and j < X.shape[1]:
                interaction = X[:, i] * X[:, j]
                X_new = np.column_stack([X_new, interaction])
        
        return X_new

# Pipeline with custom transformers
custom_pipeline = Pipeline([
    ('outlier_remover', OutlierRemover(factor=2.0)),
    ('feature_creator', FeatureCreator(feature_pairs=[(0, 1), (2, 3)])),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train custom pipeline
custom_pipeline.fit(X_train, y_train)
y_pred_custom = custom_pipeline.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

print(f"Custom Pipeline Accuracy: {accuracy_custom:.3f}")
```

## Model Persistence

### Saving and Loading Models
```python
import joblib
import pickle
from datetime import datetime

# Train a model
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

final_model.fit(X_train_mixed, y_train_mixed)

# Method 1: Using joblib (recommended for scikit-learn)
model_filename = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
joblib.dump(final_model, model_filename)
print(f"Model saved as {model_filename}")

# Load model
loaded_model = joblib.load(model_filename)

# Verify loaded model works
test_predictions = loaded_model.predict(X_test_mixed)
loaded_accuracy = accuracy_score(y_test_mixed, test_predictions)
print(f"Loaded model accuracy: {loaded_accuracy:.3f}")

# Method 2: Using pickle
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('model_pickle.pkl', 'rb') as f:
    loaded_model_pickle = pickle.load(f)

print("Model successfully saved and loaded with pickle")
```

### Model Metadata and Versioning
```python
import json

class ModelManager:
    """Class to manage model metadata and versioning"""
    
    def __init__(self):
        self.metadata = {}
    
    def save_model_with_metadata(self, model, filename, **metadata):
        """Save model with metadata"""
        
        # Add default metadata
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model.named_steps['classifier']).__name__,
            'preprocessing_steps': list(model.named_steps.keys())[:-1],
            'feature_count': None,
            'performance_metrics': {},
            **metadata
        }
        
        # Save model
        joblib.dump(model, filename)
        
        # Save metadata
        metadata_filename = filename.replace('.joblib', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Model and metadata saved: {filename}, {metadata_filename}")
    
    def load_model_with_metadata(self, filename):
        """Load model with metadata"""
        
        # Load model
        model = joblib.load(filename)
        
        # Load metadata
        metadata_filename = filename.replace('.joblib', '_metadata.json')
        try:
            with open(metadata_filename, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}
        
        return model, metadata

# Use model manager
manager = ModelManager()

# Save model with metadata
manager.save_model_with_metadata(
    final_model,
    'production_model.joblib',
    version='1.0',
    training_samples=len(X_train_mixed),
    test_accuracy=loaded_accuracy,
    features_used=feature_names,
    description='Employee performance prediction model'
)

# Load model with metadata
loaded_model, metadata = manager.load_model_with_metadata('production_model.joblib')

print("Model metadata:")
for key, value in metadata.items():
    if key != 'features_used':  # Skip long feature list
        print(f"  {key}: {value}")
```

## Pipeline Validation and Testing

### Cross-Validation with Pipelines
```python
from sklearn.model_selection import cross_validate

def evaluate_pipeline(pipeline, X, y, cv=5):
    """Comprehensive pipeline evaluation"""
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate statistics
    results = {}
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        results[metric] = {
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std(),
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'overfitting': train_scores.mean() - test_scores.mean()
        }
    
    return results

# Evaluate different pipelines
pipelines = {
    'Simple': simple_pipeline,
    'Complex': complex_pipeline,
    'Full': full_pipeline
}

print("Pipeline Comparison:")
print("=" * 60)

for name, pipeline in pipelines.items():
    print(f"\n{name} Pipeline:")
    
    # Use appropriate data for each pipeline
    if name == 'Full':
        X_eval, y_eval = X_mixed, y_mixed
    else:
        X_eval, y_eval = X, y
    
    results = evaluate_pipeline(pipeline, X_eval, y_eval)
    
    for metric, scores in results.items():
        print(f"  {metric.capitalize()}:")
        print(f"    Train: {scores['train_mean']:.3f} ± {scores['train_std']:.3f}")
        print(f"    Test:  {scores['test_mean']:.3f} ± {scores['test_std']:.3f}")
        print(f"    Overfitting: {scores['overfitting']:.3f}")
```

## Production Pipeline Considerations

### Robust Pipeline Design
```python
class ProductionPipeline:
    """Production-ready ML pipeline with error handling and logging"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def validate_input(self, X):
        """Validate input data"""
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("Input must be pandas DataFrame or numpy array")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.is_fitted and X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        
        return X
    
    def fit(self, X, y):
        """Fit the pipeline"""
        X = self.validate_input(X)
        
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Make predictions with error handling"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        X = self.validate_input(X)
        
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        X = self.validate_input(X)
        
        try:
            probabilities = self.model.predict_proba(X)
            return probabilities
        except Exception as e:
            print(f"Probability prediction error: {e}")
            return None
    
    def save_model(self, filepath):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")
        
        joblib.dump({
            'model': self.model,
            'n_features': self.n_features_,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load_model(self, filepath):
        """Load a fitted model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.n_features_ = data['n_features']
        self.is_fitted = data['is_fitted']

# Example usage
prod_pipeline = ProductionPipeline()
prod_pipeline.fit(X_train_mixed, y_train_mixed)

# Make predictions
predictions = prod_pipeline.predict(X_test_mixed)
probabilities = prod_pipeline.predict_proba(X_test_mixed)

print(f"Production pipeline predictions shape: {predictions.shape}")
print(f"Production pipeline probabilities shape: {probabilities.shape}")

# Save for production
prod_pipeline.save_model('production_pipeline.joblib')
```

## Best Practices for ML Pipelines

### 1. Pipeline Design Principles
```python
# Good pipeline design principles:

# 1. Modularity - Each step has a single responsibility
# 2. Reproducibility - Same inputs always produce same outputs
# 3. Testability - Each component can be tested independently
# 4. Maintainability - Easy to modify and extend
# 5. Documentation - Clear naming and documentation

# Example of well-designed pipeline
well_designed_pipeline = Pipeline([
    ('missing_value_imputer', SimpleImputer(strategy='median')),
    ('outlier_handler', OutlierRemover(factor=2.0)),
    ('feature_scaler', StandardScaler()),
    ('feature_selector', SelectKBest(k=10)),
    ('final_estimator', RandomForestClassifier(random_state=42))
])
```

### 2. Error Handling and Validation
```python
def safe_pipeline_prediction(pipeline, X, return_probabilities=False):
    """Make predictions with comprehensive error handling"""
    
    try:
        # Validate input
        if X is None or len(X) == 0:
            raise ValueError("Input data is empty")
        
        # Make predictions
        if return_probabilities:
            if hasattr(pipeline, 'predict_proba'):
                return pipeline.predict_proba(X)
            else:
                raise ValueError("Pipeline does not support probability predictions")
        else:
            return pipeline.predict(X)
    
    except ValueError as e:
        print(f"Validation error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during prediction: {e}")
        return None

# Test error handling
safe_predictions = safe_pipeline_prediction(final_model, X_test_mixed)
safe_probabilities = safe_pipeline_prediction(final_model, X_test_mixed, return_probabilities=True)
```

## Key Terminology

- **Pipeline**: Sequence of data processing steps ending with an estimator
- **Transformer**: Object that transforms data (preprocessing steps)
- **Estimator**: Object that can fit to data and make predictions
- **ColumnTransformer**: Applies different transformers to different columns
- **GridSearchCV**: Exhaustive search over parameter grid
- **RandomizedSearchCV**: Random sampling of parameter space
- **Cross-Validation**: Technique to assess model performance across multiple data splits
- **Hyperparameters**: Parameters that control the learning process
- **Model Persistence**: Saving and loading trained models

## Looking Ahead

In Lesson 17, we'll learn about:
- **Data Sources**: Working with APIs, databases, and web scraping
- **Real-time Data**: Handling streaming and live data feeds
- **Data Integration**: Combining multiple data sources
- **ETL Processes**: Extract, Transform, Load workflows
- **Data Quality**: Monitoring and ensuring data reliability
