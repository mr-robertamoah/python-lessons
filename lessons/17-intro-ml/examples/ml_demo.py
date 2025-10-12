#!/usr/bin/env python3
"""
Introduction to Machine Learning Demo
Basic ML concepts and implementations
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt

def classification_demo():
    """Demonstrate basic classification"""
    print("=== Classification Demo ===")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")

def regression_demo():
    """Demonstrate basic regression"""
    print("\n=== Regression Demo ===")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = reg.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = reg.score(X_test, y_test)
    
    print(f"MSE: {mse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Coefficients: {reg.coef_[0]:.3f}")
    print(f"Intercept: {reg.intercept_:.3f}")

def clustering_demo():
    """Demonstrate unsupervised learning"""
    print("\n=== Clustering Demo ===")
    
    # Generate synthetic data
    X, _ = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                              n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    print(f"Data points: {X.shape[0]}")
    print(f"Clusters found: {len(np.unique(clusters))}")
    print(f"Cluster centers:\n{kmeans.cluster_centers_}")

def cross_validation_demo():
    """Demonstrate cross-validation"""
    print("\n=== Cross-Validation Demo ===")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Compare models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

if __name__ == "__main__":
    classification_demo()
    regression_demo()
    clustering_demo()
    cross_validation_demo()
