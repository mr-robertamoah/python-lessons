# Lesson 15 Exercises: Introduction to Machine Learning

## Guided Exercises (Do with Instructor)

### Exercise 1: Classification with Scikit-Learn
**Goal**: Build your first classification model

**Tasks**:
1. Load the iris dataset
2. Split data into training and testing sets
3. Train a decision tree classifier
4. Evaluate model performance

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and explore the iris dataset
iris = load_iris()
# Build and evaluate classifier
```

---

### Exercise 2: Regression Analysis
**Goal**: Predict continuous values using regression

**Tasks**:
1. Load Boston housing dataset
2. Perform exploratory data analysis
3. Train linear regression model
4. Calculate regression metrics

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Build regression model
```

---

### Exercise 3: Model Evaluation and Cross-Validation
**Goal**: Properly evaluate model performance

**Tasks**:
1. Implement k-fold cross-validation
2. Calculate multiple evaluation metrics
3. Compare different models
4. Understand overfitting vs underfitting

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Customer Churn Prediction
**Goal**: Build a binary classification model

**Scenario**: Predict which customers will cancel their subscription

**Tasks**:
1. Create synthetic customer data
2. Engineer relevant features
3. Try multiple classification algorithms
4. Compare model performance
5. Interpret feature importance

---

### Exercise 5: Sales Forecasting
**Goal**: Build a regression model for sales prediction

**Tasks**:
1. Generate time-series sales data
2. Create seasonal and trend features
3. Build multiple regression models
4. Evaluate forecasting accuracy
5. Make future predictions

---

### Exercise 6: Clustering Analysis
**Goal**: Discover patterns in unlabeled data

**Tasks**:
1. Generate customer transaction data
2. Apply K-means clustering
3. Determine optimal number of clusters
4. Interpret cluster characteristics
5. Visualize clustering results

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: End-to-End ML Pipeline
**Goal**: Build complete machine learning workflow

**Components**:
1. Data collection and cleaning
2. Feature engineering and selection
3. Model training and validation
4. Hyperparameter tuning
5. Model deployment preparation

### Challenge 2: Ensemble Methods
**Goal**: Combine multiple models for better performance

**Tasks**:
1. Implement voting classifier
2. Build bagging ensemble
3. Create boosting model
4. Compare ensemble vs individual models

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Understand supervised vs unsupervised learning
- [ ] Build classification and regression models
- [ ] Evaluate model performance using appropriate metrics
- [ ] Apply cross-validation for robust evaluation
- [ ] Handle overfitting and underfitting
- [ ] Compare different machine learning algorithms
- [ ] Interpret model results and feature importance

## Git Reminder

Save your work:
```bash
git add .
git commit -m "Complete Lesson 15: Introduction to Machine Learning"
git push
```
