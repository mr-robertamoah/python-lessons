# Lesson 16 Exercises: Machine Learning Pipelines

## Guided Exercises (Do with Instructor)

### Exercise 1: Basic Pipeline Construction
**Goal**: Build your first ML pipeline

**Tasks**:
1. Create preprocessing pipeline with scaling
2. Add feature selection step
3. Include model training
4. Evaluate pipeline performance

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=5)),
    ('classifier', RandomForestClassifier())
])
```

---

### Exercise 2: Column Transformer
**Goal**: Handle different data types in one pipeline

**Tasks**:
1. Separate numerical and categorical features
2. Apply different preprocessing to each type
3. Combine transformations using ColumnTransformer
4. Add model to complete pipeline

---

### Exercise 3: Hyperparameter Tuning
**Goal**: Optimize pipeline parameters

**Tasks**:
1. Define parameter grid for pipeline
2. Use GridSearchCV for optimization
3. Evaluate best parameters
4. Compare performance before/after tuning

---

## Independent Exercises (Try on Your Own)

### Exercise 4: End-to-End Customer Analysis Pipeline
**Goal**: Build complete analysis pipeline

**Tasks**:
1. Create synthetic customer dataset
2. Build preprocessing pipeline for mixed data types
3. Add feature engineering steps
4. Include model training and evaluation
5. Make predictions on new data

---

### Exercise 5: Model Comparison Pipeline
**Goal**: Compare multiple models systematically

**Tasks**:
1. Create pipeline template
2. Test different algorithms
3. Use cross-validation for fair comparison
4. Select best performing model
5. Analyze feature importance

---

### Exercise 6: Production-Ready Pipeline
**Goal**: Build pipeline for deployment

**Tasks**:
1. Add data validation steps
2. Include error handling
3. Add logging and monitoring
4. Create prediction interface
5. Save and load pipeline

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Custom Transformer
**Goal**: Create reusable custom preprocessing steps

**Tasks**:
1. Build custom transformer class
2. Implement fit and transform methods
3. Integrate into pipeline
4. Test with different datasets

### Challenge 2: Automated ML Pipeline
**Goal**: Build self-optimizing pipeline

**Tasks**:
1. Automatic feature selection
2. Model selection based on data characteristics
3. Hyperparameter optimization
4. Performance monitoring

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Build ML pipelines with multiple steps
- [ ] Handle mixed data types with ColumnTransformer
- [ ] Optimize hyperparameters within pipelines
- [ ] Create custom transformers
- [ ] Build production-ready pipelines
- [ ] Compare models systematically

## Git Reminder

Save your work:
```bash
git add .
git commit -m "Complete Lesson 16: ML Pipelines"
git push
```
