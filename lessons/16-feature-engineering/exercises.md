# Lesson 14 Exercises: Feature Engineering

## Guided Exercises (Do with Instructor)

### Exercise 1: Categorical Encoding Practice
**Goal**: Apply different encoding techniques to categorical data

**Dataset**:
```python
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'product': ['A', 'B', 'C', 'A', 'B', 'C', 'A'],
    'size': ['Small', 'Medium', 'Large', 'Medium', 'Small', 'Large', 'Small'],
    'region': ['North', 'South', 'East', 'North', 'West', 'South', 'East'],
    'sales': [100, 150, 200, 120, 90, 180, 110]
})
```

**Tasks**:
1. Apply one-hot encoding to 'product' and 'region'
2. Apply label encoding to 'size' (Small=1, Medium=2, Large=3)
3. Apply target encoding to 'product' based on 'sales'
4. Compare the different encoding results

---

### Exercise 2: Feature Scaling Comparison
**Goal**: Understand when to use different scaling methods

**Dataset**:
```python
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 200],  # 200 is outlier
    'salary': [50000, 60000, 75000, 90000, 85000, 95000, 150000],
    'experience': [2, 5, 8, 12, 10, 15, 25]
})
```

**Tasks**:
1. Apply StandardScaler, MinMaxScaler, and RobustScaler
2. Compare how each handles the outlier (age=200)
3. Visualize the scaled data using box plots
4. Determine which scaler is most appropriate for this data

---

### Exercise 3: DateTime Feature Engineering
**Goal**: Extract useful features from datetime data

**Generate datetime data**:
```python
dates = pd.date_range('2024-01-01', periods=100, freq='D')
sales = 1000 + 200 * np.sin(2 * np.pi * np.arange(100) / 7) + np.random.normal(0, 50, 100)
df = pd.DataFrame({'date': dates, 'sales': sales})
```

**Tasks**:
1. Extract year, month, day, day_of_week
2. Create is_weekend feature
3. Create cyclical encoding for month and day_of_week
4. Analyze which features correlate most with sales

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Customer Data Feature Engineering
**Goal**: Create comprehensive features for customer analysis

**Generate customer data**:
```python
np.random.seed(42)
customers = pd.DataFrame({
    'customer_id': range(1000),
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'last_purchase_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
    'total_purchases': np.random.poisson(5, 1000),
    'total_spent': np.random.normal(2000, 800, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 1000)
})
```

**Create features**:
1. Days since last purchase
2. Average purchase amount
3. Purchase frequency (purchases per day since first purchase)
4. Age groups (18-25, 26-35, 36-50, 51+)
5. Income percentile ranking
6. One-hot encode education and city
7. Create interaction features (age × income, education × income)

---

### Exercise 5: Text Feature Engineering
**Goal**: Extract features from text data for analysis

**Sample text data**:
```python
reviews = [
    "This product is amazing! Great quality and fast shipping.",
    "Terrible experience. Poor quality and slow delivery.",
    "Good value for money. Recommended!",
    "Excellent customer service. Very helpful staff.",
    "Not worth the price. Disappointed with purchase.",
    "Outstanding product! Exceeded my expectations completely.",
    "Average quality. Nothing special but does the job.",
    "Fantastic! Will definitely buy again. Highly recommended!"
]
```

**Create features**:
1. Text length (characters and words)
2. Sentiment indicators (positive/negative word counts)
3. Exclamation mark count
4. Capital letter percentage
5. Bag of words representation
6. TF-IDF features
7. Simple sentiment score

---

### Exercise 6: Financial Data Feature Engineering
**Goal**: Create features for financial analysis

**Generate stock data**:
```python
dates = pd.date_range('2024-01-01', periods=252, freq='D')  # Trading year
np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
volume = np.random.lognormal(10, 0.5, 252)

stock_data = pd.DataFrame({
    'date': dates,
    'price': prices,
    'volume': volume
})
```

**Create features**:
1. Daily returns (percentage change)
2. Moving averages (5, 10, 20 days)
3. Volatility (rolling standard deviation of returns)
4. Price momentum indicators
5. Volume-weighted average price (VWAP)
6. Bollinger Bands (price relative to moving average ± 2 std)
7. RSI (Relative Strength Index) approximation

---

### Exercise 7: E-commerce Feature Engineering
**Goal**: Create features for e-commerce recommendation system

**Generate e-commerce data**:
```python
np.random.seed(42)
transactions = pd.DataFrame({
    'user_id': np.random.randint(1, 101, 1000),
    'product_id': np.random.randint(1, 51, 1000),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 1000),
    'price': np.random.uniform(10, 500, 1000),
    'quantity': np.random.randint(1, 5, 1000),
    'rating': np.random.randint(1, 6, 1000),
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H')
})
```

**Create user features**:
1. Total number of purchases per user
2. Average purchase amount per user
3. Favorite category per user
4. Average rating given per user
5. Days since last purchase
6. Purchase frequency
7. Price sensitivity (preference for low/high priced items)

**Create product features**:
1. Average rating per product
2. Number of purchases per product
3. Price percentile within category
4. Purchase recency

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Advanced Feature Selection
**Goal**: Implement sophisticated feature selection techniques

**Tasks**:
1. Generate dataset with 50 features, only 5 are truly predictive
2. Implement recursive feature elimination
3. Use LASSO regularization for feature selection
4. Compare mutual information vs correlation-based selection
5. Create feature importance visualization

### Challenge 2: Automated Feature Engineering
**Goal**: Build automated feature engineering pipeline

**Requirements**:
1. Automatically detect data types
2. Apply appropriate encoding for categorical variables
3. Create polynomial features for numeric variables
4. Generate interaction terms
5. Apply feature selection
6. Create pipeline that can handle new data

### Challenge 3: Time Series Feature Engineering
**Goal**: Advanced time series feature creation

**Tasks**:
1. Create lag features (1, 7, 30 days)
2. Rolling statistics (mean, std, min, max)
3. Seasonal decomposition features
4. Fourier transform features for periodicity
5. Change point detection features

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Apply one-hot encoding to categorical variables
- [ ] Use label encoding for ordinal categories
- [ ] Implement target encoding with proper validation
- [ ] Choose appropriate scaling methods for different data types
- [ ] Extract meaningful features from datetime data
- [ ] Create interaction and polynomial features
- [ ] Handle text data for machine learning
- [ ] Select important features using statistical methods
- [ ] Build automated feature engineering pipelines

## Best Practices Learned

### Feature Engineering Guidelines
1. **Domain knowledge is crucial** - understand your data context
2. **Start simple** - basic features often work well
3. **Validate features** - ensure they don't leak future information
4. **Handle missing values** appropriately for each feature type
5. **Scale features** when using distance-based algorithms
6. **Document transformations** for reproducibility

### Common Pitfalls to Avoid
1. **Data leakage** - using future information to predict past
2. **Overfitting** - creating too many features relative to samples
3. **Ignoring business logic** - features that don't make business sense
4. **Inconsistent preprocessing** - different handling for train/test data
5. **Feature explosion** - creating too many irrelevant features

## Git Reminder

Save your work:
1. Create `lesson-14-feature-engineering` folder in your repository
2. Save exercise solutions as `.py` files
3. Include feature engineering pipelines
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 14: Feature Engineering"
   git push
   ```

## Next Lesson Preview

In Lesson 15, we'll learn about:
- **Machine Learning Fundamentals**: Supervised vs unsupervised learning
- **Model Training**: Fitting algorithms to data
- **Model Evaluation**: Metrics and validation techniques
- **Cross-Validation**: Robust performance assessment
