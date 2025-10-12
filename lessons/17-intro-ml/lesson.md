# Lesson 15: Introduction to Machine Learning

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand different types of machine learning problems
- Split data into training and testing sets
- Train basic supervised learning models
- Evaluate model performance using appropriate metrics
- Apply cross-validation for robust model assessment
- Avoid common pitfalls like overfitting and data leakage

## What is Machine Learning?

### Definition and Intuition
Machine Learning is the science of getting computers to learn patterns from data without being explicitly programmed for every scenario.

```python
# Traditional Programming:
def calculate_house_price(size, bedrooms, location):
    if location == "downtown":
        base_price = 500000
    else:
        base_price = 300000
    return base_price + (size * 200) + (bedrooms * 10000)

# Machine Learning:
# Algorithm learns from data: [size, bedrooms, location] → price
# No explicit rules needed!
```

### Types of Machine Learning

#### 1. Supervised Learning
Learning with labeled examples (input-output pairs)

```python
# Classification: Predict categories
# Email → Spam/Not Spam
# Image → Cat/Dog/Bird
# Customer → Will Buy/Won't Buy

# Regression: Predict continuous values  
# House features → Price
# Weather data → Temperature
# Marketing spend → Sales
```

#### 2. Unsupervised Learning
Finding patterns in data without labels

```python
# Clustering: Group similar items
# Customer segmentation
# Gene sequencing
# Market basket analysis

# Dimensionality Reduction: Simplify data
# Data visualization
# Feature extraction
# Noise reduction
```

#### 3. Reinforcement Learning
Learning through trial and error with rewards

```python
# Examples:
# Game playing (Chess, Go)
# Autonomous vehicles
# Trading algorithms
# Recommendation systems
```

## Setting Up for Machine Learning

### Essential Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
```

## Train-Test Split

### Why Split Data?
```python
# The fundamental principle: Never test on training data!

# Bad approach:
# 1. Train model on all data
# 2. Test model on same data
# 3. Get artificially high performance

# Good approach:
# 1. Split data into train/test
# 2. Train model on training data only
# 3. Test model on unseen test data
# 4. Get realistic performance estimate
```

### Implementing Train-Test Split
```python
# Generate sample dataset
from sklearn.datasets import make_classification, make_regression

# Classification dataset
X_class, y_class = make_classification(n_samples=1000, n_features=10, 
                                      n_informative=5, n_redundant=2, 
                                      random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y_class    # Maintain class proportions
)

print(f"Total samples: {len(X_class)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Testing class distribution: {np.bincount(y_test)}")
```

### Different Split Strategies
```python
# Time series data: Use temporal split
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
time_series_data = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(1000).cumsum()
})

# Split by time (last 20% for testing)
split_date = time_series_data['date'].quantile(0.8)
train_ts = time_series_data[time_series_data['date'] <= split_date]
test_ts = time_series_data[time_series_data['date'] > split_date]

print(f"Time series split:")
print(f"Training period: {train_ts['date'].min()} to {train_ts['date'].max()}")
print(f"Testing period: {test_ts['date'].min()} to {test_ts['date'].max()}")
```

## Supervised Learning: Classification

### Logistic Regression
```python
# Binary classification example
from sklearn.datasets import make_classification

# Generate binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Visualize decision boundary
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(10, 8))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(X, y, log_reg, 'Logistic Regression Decision Boundary')
```

### Decision Tree Classifier
```python
# Train decision tree
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

# Make predictions
y_pred_tree = tree_clf.predict(X_test)

# Evaluate
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree:.3f}")

# Visualize decision boundary
plot_decision_boundary(X, y, tree_clf, 'Decision Tree Decision Boundary')

# Feature importance
feature_importance = tree_clf.feature_importances_
print(f"Feature Importance: {feature_importance}")
```

### Random Forest Classifier
```python
# Train random forest
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_clf.predict(X_test)

# Evaluate
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.3f}")

# Compare all models
models_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy, accuracy_tree, accuracy_rf]
})
print("\nModel Comparison:")
print(models_comparison)
```

## Supervised Learning: Regression

### Linear Regression
```python
# Generate regression dataset
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Train linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = lin_reg.predict(X_test_reg)

# Evaluate regression
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("Linear Regression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# Visualize regression
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test_reg, y_test_reg, alpha=0.6, label='Actual')
plt.scatter(X_test_reg, y_pred_reg, alpha=0.6, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Actual vs Predicted')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Prediction vs Actual')

plt.tight_layout()
plt.show()
```

## Model Evaluation Metrics

### Classification Metrics
```python
from sklearn.metrics import confusion_matrix, classification_report

# Generate multi-class classification data
X_multi, y_multi = make_classification(n_samples=1000, n_features=10, 
                                      n_classes=3, n_informative=5, random_state=42)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

# Train classifier
rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = rf_multi.predict(X_test_multi)

# Confusion Matrix
cm = confusion_matrix(y_test_multi, y_pred_multi)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test_multi, y_pred_multi))

# Metrics explanation
print("\nMetrics Explanation:")
print("Accuracy: (TP + TN) / (TP + TN + FP + FN)")
print("Precision: TP / (TP + FP) - Of predicted positives, how many were correct?")
print("Recall: TP / (TP + FN) - Of actual positives, how many were found?")
print("F1-Score: 2 * (Precision * Recall) / (Precision + Recall)")
```

### Regression Metrics
```python
def regression_metrics_explained(y_true, y_pred):
    """Calculate and explain regression metrics"""
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("Regression Metrics:")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"MSE (Mean Squared Error): {mse:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R² (R-squared): {r2:.3f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    print("\nInterpretation:")
    print("MAE: Average absolute difference between actual and predicted")
    print("RMSE: Square root of average squared differences (penalizes large errors)")
    print("R²: Proportion of variance explained (1.0 = perfect, 0.0 = no better than mean)")
    print("MAPE: Average percentage error")
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# Apply to our regression example
metrics = regression_metrics_explained(y_test_reg, y_pred_reg)
```

## Cross-Validation

### K-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# K-Fold Cross-Validation for classification
cv_scores = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')

print("5-Fold Cross-Validation Results:")
print(f"Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Standard Deviation: {cv_scores.std():.3f}")
print(f"95% Confidence Interval: {cv_scores.mean():.3f} ± {1.96 * cv_scores.std():.3f}")

# Stratified K-Fold (maintains class proportions)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(rf_clf, X, y, cv=skf, scoring='accuracy')

print(f"\nStratified K-Fold Mean: {stratified_scores.mean():.3f}")
```

### Cross-Validation for Model Selection
```python
from sklearn.model_selection import GridSearchCV

# Compare multiple models with cross-validation
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }

# Display results
results_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'Mean CV Score': [cv_results[name]['mean'] for name in cv_results.keys()],
    'Std CV Score': [cv_results[name]['std'] for name in cv_results.keys()]
})

print("Cross-Validation Model Comparison:")
print(results_df.round(3))

# Visualize CV results
plt.figure(figsize=(10, 6))
model_names = list(cv_results.keys())
means = [cv_results[name]['mean'] for name in model_names]
stds = [cv_results[name]['std'] for name in model_names]

plt.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
plt.ylabel('Cross-Validation Accuracy')
plt.title('Model Comparison with Cross-Validation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Overfitting and Underfitting

### Demonstrating Overfitting
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Generate simple dataset
np.random.seed(42)
X_simple = np.linspace(0, 1, 100).reshape(-1, 1)
y_simple = 1.5 * X_simple.ravel() + 0.5 * np.sin(2 * np.pi * X_simple.ravel()) + np.random.normal(0, 0.1, 100)

# Split data
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.3, random_state=42)

# Test different polynomial degrees
degrees = [1, 3, 9, 15]
plt.figure(figsize=(16, 4))

for i, degree in enumerate(degrees, 1):
    # Create polynomial features
    poly_reg = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Fit model
    poly_reg.fit(X_train_simple, y_train_simple)
    
    # Predictions
    y_train_pred = poly_reg.predict(X_train_simple)
    y_test_pred = poly_reg.predict(X_test_simple)
    
    # Calculate scores
    train_score = r2_score(y_train_simple, y_train_pred)
    test_score = r2_score(y_test_simple, y_test_pred)
    
    # Plot
    plt.subplot(1, 4, i)
    
    # Generate smooth curve for visualization
    X_plot = np.linspace(0, 1, 300).reshape(-1, 1)
    y_plot = poly_reg.predict(X_plot)
    
    plt.scatter(X_train_simple, y_train_simple, alpha=0.6, label='Training')
    plt.scatter(X_test_simple, y_test_simple, alpha=0.6, label='Testing')
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
    
    plt.title(f'Degree {degree}\nTrain R²: {train_score:.3f}\nTest R²: {test_score:.3f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()

print("Overfitting Analysis:")
print("Degree 1: Underfitting (high bias, low variance)")
print("Degree 3: Good fit (balanced bias-variance)")
print("Degree 9: Starting to overfit (lower bias, higher variance)")
print("Degree 15: Severe overfitting (very low bias, very high variance)")
```

### Learning Curves
```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """Plot learning curve to diagnose overfitting"""
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy' if hasattr(y, 'dtype') and y.dtype == int else 'r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot learning curves for different models
plot_learning_curve(DecisionTreeClassifier(max_depth=3, random_state=42), 
                   X, y, "Decision Tree (max_depth=3) - Good Fit")

plot_learning_curve(DecisionTreeClassifier(max_depth=20, random_state=42), 
                   X, y, "Decision Tree (max_depth=20) - Overfitting")
```

## Common Pitfalls and Best Practices

### Data Leakage
```python
# Example of data leakage
def demonstrate_data_leakage():
    """Show how data leakage can inflate performance"""
    
    # Generate dataset where future information leaks into features
    np.random.seed(42)
    n_samples = 1000
    
    # Target variable
    y = np.random.binomial(1, 0.3, n_samples)
    
    # Legitimate features
    X_legit = np.random.randn(n_samples, 5)
    
    # Leaked feature (directly related to target)
    X_leaked = y + np.random.normal(0, 0.1, n_samples)
    
    # Combine features
    X_with_leakage = np.column_stack([X_legit, X_leaked])
    X_without_leakage = X_legit
    
    # Train models
    rf_with_leak = RandomForestClassifier(random_state=42)
    rf_without_leak = RandomForestClassifier(random_state=42)
    
    # Cross-validation scores
    scores_with_leak = cross_val_score(rf_with_leak, X_with_leakage, y, cv=5)
    scores_without_leak = cross_val_score(rf_without_leak, X_without_leakage, y, cv=5)
    
    print("Data Leakage Demonstration:")
    print(f"With leaked feature: {scores_with_leak.mean():.3f} ± {scores_with_leak.std():.3f}")
    print(f"Without leaked feature: {scores_without_leak.mean():.3f} ± {scores_without_leak.std():.3f}")
    print("\nThe suspiciously high performance with leaked feature indicates data leakage!")

demonstrate_data_leakage()
```

### Proper Validation Strategy
```python
def proper_ml_workflow(X, y):
    """Demonstrate proper ML workflow"""
    
    print("Proper Machine Learning Workflow:")
    print("1. Split data into train/validation/test")
    
    # First split: separate test set (never touch until final evaluation)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Second split: separate train/validation from remaining data
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    print("\n2. Train models and tune hyperparameters using train/validation")
    
    # Train different models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    val_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_scores[name] = accuracy_score(y_val, val_pred)
    
    print("Validation scores:")
    for name, score in val_scores.items():
        print(f"{name}: {score:.3f}")
    
    print("\n3. Select best model and evaluate on test set (only once!)")
    
    # Select best model
    best_model_name = max(val_scores, key=val_scores.get)
    best_model = models[best_model_name]
    
    # Final evaluation on test set
    test_pred = best_model.predict(X_test)
    test_score = accuracy_score(y_test, test_pred)
    
    print(f"Best model: {best_model_name}")
    print(f"Final test score: {test_score:.3f}")
    
    return best_model, test_score

# Apply proper workflow
best_model, final_score = proper_ml_workflow(X, y)
```

## Key Terminology

- **Supervised Learning**: Learning with labeled examples
- **Unsupervised Learning**: Finding patterns without labels
- **Training Set**: Data used to train the model
- **Test Set**: Data used to evaluate final model performance
- **Validation Set**: Data used for model selection and hyperparameter tuning
- **Overfitting**: Model performs well on training data but poorly on new data
- **Underfitting**: Model is too simple to capture underlying patterns
- **Cross-Validation**: Technique to assess model performance using multiple train/test splits
- **Data Leakage**: When future or target information inappropriately influences features

## Looking Ahead

In Lesson 16, we'll learn about:
- **ML Pipelines**: Automating the entire machine learning workflow
- **Hyperparameter Tuning**: Optimizing model parameters systematically
- **Model Persistence**: Saving and loading trained models
- **Production Considerations**: Deploying models in real-world applications
- **Advanced Evaluation**: ROC curves, precision-recall curves, and more sophisticated metrics
