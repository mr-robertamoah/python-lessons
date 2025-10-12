# Lesson 16 Solutions: Machine Learning Pipelines

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Basic Pipeline Construction
print("Exercise 1: Basic Pipeline Construction")
print("-" * 50)

# Create sample data
np.random.seed(42)
n_samples = 1000

X = np.random.randn(n_samples, 5)
y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build basic pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=3)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate
accuracy = pipeline.score(X_test, y_test)
print(f"Pipeline accuracy: {accuracy:.3f}")

# Show selected features
selected_features = pipeline.named_steps['selector'].get_support()
print(f"Selected features: {np.where(selected_features)[0]}")

# Exercise 2: Column Transformer
print("\n" + "="*50)
print("Exercise 2: Column Transformer")
print("-" * 50)

# Create mixed data
data = {
    'age': np.random.randint(18, 80, 500),
    'income': np.random.lognormal(10, 1, 500),
    'education': np.random.choice(['High School', 'Bachelor', 'Master'], 500),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 500),
    'experience': np.random.randint(0, 40, 500)
}

df = pd.DataFrame(data)

# Create target
target_prob = 0.1 + 0.3 * (df['age'] > 40) + 0.2 * (df['income'] > 50000)
df['approved'] = np.random.binomial(1, target_prob)

# Define feature types
numeric_features = ['age', 'income', 'experience']
categorical_features = ['education', 'city']

X = df[numeric_features + categorical_features]
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

print(f"Column transformer pipeline accuracy: {accuracy:.3f}")
print(f"Original features: {len(X.columns)}")

# Get feature names after preprocessing
try:
    feature_names = (numeric_features + 
                    list(pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical_features)))
    print(f"Features after preprocessing: {len(feature_names)}")
except:
    print("Feature names extraction not available in this sklearn version")

# Exercise 3: Hyperparameter Tuning
print("\n" + "="*50)
print("Exercise 3: Hyperparameter Tuning")
print("-" * 50)

# Use the same data from Exercise 2
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test accuracy: {grid_search.score(X_test, y_test):.3f}")

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: End-to-End Customer Analysis Pipeline
print("Exercise 4: End-to-End Customer Analysis Pipeline")
print("-" * 50)

# Create comprehensive customer dataset
np.random.seed(42)
n_customers = 2000

customer_data = {
    'age': np.random.randint(18, 80, n_customers),
    'income': np.random.lognormal(10.5, 0.8, n_customers),
    'credit_score': np.random.randint(300, 850, n_customers),
    'years_employed': np.random.uniform(0, 40, n_customers),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_customers),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_customers),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_customers),
    'num_accounts': np.random.randint(1, 8, n_customers),
    'total_balance': np.random.uniform(0, 100000, n_customers)
}

customer_df = pd.DataFrame(customer_data)

# Create target variable (loan approval)
approval_prob = (
    0.1 + 
    0.3 * (customer_df['credit_score'] > 650) + 
    0.2 * (customer_df['income'] > 50000) + 
    0.1 * (customer_df['years_employed'] > 2) +
    0.1 * (customer_df['education'].isin(['Master', 'PhD']))
)
customer_df['loan_approved'] = np.random.binomial(1, approval_prob)

print(f"Customer dataset shape: {customer_df.shape}")
print(f"Loan approval rate: {customer_df['loan_approved'].mean():.3f}")

# Define feature types
numeric_features = ['age', 'income', 'credit_score', 'years_employed', 'num_accounts', 'total_balance']
categorical_features = ['education', 'marital_status', 'city']

X = customer_df[numeric_features + categorical_features]
y = customer_df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom feature engineering transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        # Create new features
        X_new['debt_to_income'] = X_new['total_balance'] / (X_new['income'] + 1)
        X_new['balance_per_account'] = X_new['total_balance'] / (X_new['num_accounts'] + 1)
        X_new['income_per_year_employed'] = X_new['income'] / (X_new['years_employed'] + 1)
        return X_new

# Build comprehensive pipeline
numeric_transformer = Pipeline([
    ('engineer', FeatureEngineer()),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(drop='first', sparse=False)

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train and evaluate
full_pipeline.fit(X_train, y_train)
accuracy = full_pipeline.score(X_test, y_test)

print(f"End-to-end pipeline accuracy: {accuracy:.3f}")

# Feature importance
feature_importance = full_pipeline.named_steps['classifier'].feature_importances_
selected_features = full_pipeline.named_steps['selector'].get_support()

print(f"Selected {sum(selected_features)} features out of {len(selected_features)}")

# Exercise 5: Model Comparison Pipeline
print("\n" + "="*50)
print("Exercise 5: Model Comparison Pipeline")
print("-" * 50)

# Use the same preprocessor from Exercise 4
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
}

# Compare models
results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std()
    }
    
    print(f"{name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Select best model
best_model_name = max(results.keys(), key=lambda k: results[k]['mean_cv_score'])
print(f"\nBest model: {best_model_name}")

# Exercise 6: Production-Ready Pipeline
print("\n" + "="*50)
print("Exercise 6: Production-Ready Pipeline")
print("-" * 50)

class ProductionPipeline:
    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self.is_fitted = False
    
    def build_pipeline(self, numeric_features, categorical_features):
        """Build the ML pipeline"""
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])
        
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        self.feature_names = numeric_features + categorical_features
    
    def fit(self, X, y):
        """Train the pipeline"""
        try:
            # Validate input
            if X.empty or len(y) == 0:
                raise ValueError("Empty input data")
            
            if len(X) != len(y):
                raise ValueError("X and y must have same length")
            
            # Check for required features
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Train pipeline
            self.pipeline.fit(X[self.feature_names], y)
            self.is_fitted = True
            
            print("Pipeline training completed successfully")
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        try:
            if not self.is_fitted:
                raise ValueError("Pipeline not fitted yet")
            
            # Validate input
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            predictions = self.pipeline.predict(X[self.feature_names])
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise
    
    def save_pipeline(self, filepath):
        """Save pipeline to disk"""
        try:
            joblib.dump({
                'pipeline': self.pipeline,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }, filepath)
            print(f"Pipeline saved to {filepath}")
        except Exception as e:
            print(f"Save error: {str(e)}")
    
    def load_pipeline(self, filepath):
        """Load pipeline from disk"""
        try:
            data = joblib.load(filepath)
            self.pipeline = data['pipeline']
            self.feature_names = data['feature_names']
            self.is_fitted = data['is_fitted']
            print(f"Pipeline loaded from {filepath}")
        except Exception as e:
            print(f"Load error: {str(e)}")

# Demo production pipeline
prod_pipeline = ProductionPipeline()
prod_pipeline.build_pipeline(numeric_features, categorical_features)

# Train
prod_pipeline.fit(X_train, y_train)

# Make predictions
predictions = prod_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Production pipeline accuracy: {accuracy:.3f}")

# Save pipeline (commented out to avoid file creation)
# prod_pipeline.save_pipeline('customer_loan_pipeline.pkl')

print("\n" + "="*50)
print("ML Pipelines exercise solutions complete!")
print("Key concepts demonstrated:")
print("- Basic pipeline construction with preprocessing and modeling")
print("- Column transformers for mixed data types")
print("- Hyperparameter tuning within pipelines")
print("- End-to-end customer analysis pipeline")
print("- Model comparison using consistent preprocessing")
print("- Production-ready pipeline with error handling")
