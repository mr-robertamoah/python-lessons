# Lesson 14: Feature Engineering - Preparing Data for Machine Learning

## Learning Objectives
By the end of this lesson, you will be able to:
- Transform categorical variables into numerical formats
- Scale and normalize numerical features
- Create new features from existing data
- Handle datetime features effectively
- Select the most important features for modeling
- Apply dimensionality reduction techniques

## What is Feature Engineering?

### The Art and Science of Features
Feature engineering is the process of creating, transforming, and selecting variables (features) that make machine learning algorithms work better.

```python
# Raw data might look like this:
raw_data = {
    'name': 'John Doe',
    'birth_date': '1990-05-15',
    'salary': '$75,000',
    'department': 'Engineering'
}

# Engineered features might be:
engineered_features = {
    'age': 34,
    'salary_numeric': 75000,
    'salary_log': 11.225,
    'dept_engineering': 1,
    'dept_sales': 0,
    'salary_above_median': 1
}
```

### Why Feature Engineering Matters
- **Garbage In, Garbage Out**: Poor features lead to poor models
- **Algorithm Performance**: Good features can make simple algorithms outperform complex ones
- **Interpretability**: Well-engineered features are easier to understand
- **Domain Knowledge**: Incorporates business understanding into models

## Handling Categorical Variables

### One-Hot Encoding
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Sample categorical data
df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'A', 'B'],
    'region': ['North', 'South', 'East', 'North', 'West'],
    'size': ['Small', 'Medium', 'Large', 'Medium', 'Small']
})

# Method 1: Using pandas get_dummies
encoded_df = pd.get_dummies(df, columns=['product', 'region'], prefix=['prod', 'reg'])
print("One-Hot Encoded Data:")
print(encoded_df)

# Method 2: Using sklearn OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity
encoded_array = encoder.fit_transform(df[['product', 'region']])
feature_names = encoder.get_feature_names_out(['product', 'region'])

encoded_sklearn = pd.DataFrame(encoded_array, columns=feature_names)
print("\nSklearn One-Hot Encoded:")
print(encoded_sklearn)
```

### Label Encoding
```python
# For ordinal categories (order matters)
size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
df['size_encoded'] = df['size'].map(size_mapping)

# Using sklearn LabelEncoder
label_encoder = LabelEncoder()
df['product_label'] = label_encoder.fit_transform(df['product'])

print("Label Encoded Data:")
print(df[['size', 'size_encoded', 'product', 'product_label']])
```

### Target Encoding
```python
# Encode categories based on target variable relationship
def target_encode(df, categorical_col, target_col, smoothing=1):
    """
    Target encoding with smoothing to prevent overfitting
    """
    # Calculate global mean
    global_mean = df[target_col].mean()
    
    # Calculate category means and counts
    category_stats = df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
    
    # Apply smoothing
    smoothed_means = (category_stats['mean'] * category_stats['count'] + 
                     global_mean * smoothing) / (category_stats['count'] + smoothing)
    
    # Map back to original dataframe
    return df[categorical_col].map(smoothed_means)

# Example with sales data
sales_df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'] * 10,
    'sales': np.random.normal([100, 150, 200], 20, 80)
})

sales_df['product_target_encoded'] = target_encode(sales_df, 'product', 'sales')
print("Target Encoded Products:")
print(sales_df.groupby('product')['product_target_encoded'].first())
```

## Feature Scaling and Normalization

### StandardScaler (Z-score normalization)
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Sample numerical data
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50],
    'salary': [50000, 60000, 75000, 90000, 85000, 95000],
    'experience': [2, 5, 8, 12, 10, 15]
})

# StandardScaler: (x - mean) / std
scaler_standard = StandardScaler()
data_standard = pd.DataFrame(
    scaler_standard.fit_transform(data),
    columns=[f'{col}_standard' for col in data.columns]
)

print("Original Data:")
print(data)
print("\nStandardized Data (mean=0, std=1):")
print(data_standard.round(3))
print(f"Means: {data_standard.mean().round(3).tolist()}")
print(f"Stds: {data_standard.std().round(3).tolist()}")
```

### MinMaxScaler (0-1 normalization)
```python
# MinMaxScaler: (x - min) / (max - min)
scaler_minmax = MinMaxScaler()
data_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(data),
    columns=[f'{col}_minmax' for col in data.columns]
)

print("Min-Max Scaled Data (range 0-1):")
print(data_minmax.round(3))
print(f"Mins: {data_minmax.min().tolist()}")
print(f"Maxs: {data_minmax.max().tolist()}")
```

### RobustScaler (median and IQR)
```python
# RobustScaler: (x - median) / IQR (less sensitive to outliers)
scaler_robust = RobustScaler()
data_robust = pd.DataFrame(
    scaler_robust.fit_transform(data),
    columns=[f'{col}_robust' for col in data.columns]
)

print("Robust Scaled Data:")
print(data_robust.round(3))
```

### When to Use Each Scaler
```python
# Demonstration with outliers
data_with_outliers = pd.DataFrame({
    'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'with_outlier': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
})

scalers = {
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler()
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original data
axes[0].boxplot([data_with_outliers['normal'], data_with_outliers['with_outlier']], 
                labels=['Normal', 'With Outlier'])
axes[0].set_title('Original Data')

# Compare scalers
for i, (name, scaler) in enumerate(scalers.items(), 1):
    scaled_data = scaler.fit_transform(data_with_outliers)
    axes[i].boxplot([scaled_data[:, 0], scaled_data[:, 1]], 
                    labels=['Normal', 'With Outlier'])
    axes[i].set_title(f'{name} Scaler')

plt.tight_layout()
plt.show()

print("Scaler Recommendations:")
print("- StandardScaler: Normal distribution, no outliers")
print("- MinMaxScaler: Bounded range needed, no outliers")
print("- RobustScaler: Presence of outliers")
```

## Creating New Features

### Mathematical Transformations
```python
# Sample dataset
df = pd.DataFrame({
    'price': [10, 20, 30, 40, 50],
    'quantity': [100, 80, 60, 40, 20],
    'cost': [8, 15, 22, 30, 38]
})

# Create new features
df['revenue'] = df['price'] * df['quantity']
df['profit'] = df['revenue'] - df['cost']
df['profit_margin'] = df['profit'] / df['revenue']
df['price_per_unit_cost'] = df['price'] / df['cost']

# Log transformations (for skewed data)
df['log_price'] = np.log(df['price'])
df['log_quantity'] = np.log(df['quantity'])

# Polynomial features
df['price_squared'] = df['price'] ** 2
df['price_quantity_interaction'] = df['price'] * df['quantity']

print("Original and Engineered Features:")
print(df.round(3))
```

### Binning and Discretization
```python
# Convert continuous variables to categorical
ages = np.random.randint(18, 80, 100)

# Equal-width binning
age_bins_equal = pd.cut(ages, bins=5, labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])

# Custom bins
age_bins_custom = pd.cut(ages, 
                        bins=[0, 25, 35, 50, 65, 100], 
                        labels=['Gen Z', 'Millennial', 'Gen X', 'Boomer', 'Silent'])

# Quantile-based binning
age_bins_quantile = pd.qcut(ages, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

binning_df = pd.DataFrame({
    'age': ages,
    'equal_width': age_bins_equal,
    'custom': age_bins_custom,
    'quantile': age_bins_quantile
})

print("Binning Examples:")
print(binning_df.head(10))

# Analyze bin distributions
print("\nBin Distributions:")
for col in ['equal_width', 'custom', 'quantile']:
    print(f"{col}: {binning_df[col].value_counts().to_dict()}")
```

### Text Feature Engineering
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample text data
texts = [
    "I love this product amazing quality",
    "Terrible service very disappointed",
    "Good value for money recommended",
    "Excellent customer support helpful staff",
    "Poor quality not worth the price"
]

# Bag of Words
vectorizer_bow = CountVectorizer(stop_words='english', max_features=10)
bow_features = vectorizer_bow.fit_transform(texts)
bow_df = pd.DataFrame(bow_features.toarray(), 
                     columns=vectorizer_bow.get_feature_names_out())

print("Bag of Words Features:")
print(bow_df)

# TF-IDF
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=10)
tfidf_features = vectorizer_tfidf.fit_transform(texts)
tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                       columns=vectorizer_tfidf.get_feature_names_out())

print("\nTF-IDF Features:")
print(tfidf_df.round(3))

# Simple text features
text_df = pd.DataFrame({'text': texts})
text_df['length'] = text_df['text'].str.len()
text_df['word_count'] = text_df['text'].str.split().str.len()
text_df['exclamation_count'] = text_df['text'].str.count('!')
text_df['positive_words'] = text_df['text'].str.count('good|great|excellent|amazing|love')
text_df['negative_words'] = text_df['text'].str.count('bad|terrible|poor|disappointed')

print("\nSimple Text Features:")
print(text_df)
```

## DateTime Feature Engineering

### Extracting Time Components
```python
# Sample datetime data
dates = pd.date_range('2024-01-01', periods=100, freq='D')
df_time = pd.DataFrame({'date': dates})

# Add some sample values
np.random.seed(42)
df_time['sales'] = 1000 + 200 * np.sin(2 * np.pi * np.arange(100) / 7) + np.random.normal(0, 50, 100)

# Extract datetime features
df_time['year'] = df_time['date'].dt.year
df_time['month'] = df_time['date'].dt.month
df_time['day'] = df_time['date'].dt.day
df_time['day_of_week'] = df_time['date'].dt.dayofweek
df_time['day_name'] = df_time['date'].dt.day_name()
df_time['week_of_year'] = df_time['date'].dt.isocalendar().week
df_time['quarter'] = df_time['date'].dt.quarter
df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
df_time['is_month_start'] = df_time['date'].dt.is_month_start.astype(int)
df_time['is_month_end'] = df_time['date'].dt.is_month_end.astype(int)

# Cyclical encoding for periodic features
df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
df_time['day_of_week_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
df_time['day_of_week_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)

print("DateTime Features:")
print(df_time.head(10))

# Visualize cyclical encoding
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Month cyclical encoding
axes[0].scatter(df_time['month_cos'], df_time['month_sin'], c=df_time['month'], cmap='viridis')
axes[0].set_xlabel('Month Cosine')
axes[0].set_ylabel('Month Sine')
axes[0].set_title('Cyclical Encoding: Month')

# Day of week cyclical encoding
axes[1].scatter(df_time['day_of_week_cos'], df_time['day_of_week_sin'], 
               c=df_time['day_of_week'], cmap='viridis')
axes[1].set_xlabel('Day of Week Cosine')
axes[1].set_ylabel('Day of Week Sine')
axes[1].set_title('Cyclical Encoding: Day of Week')

plt.tight_layout()
plt.show()
```

## Feature Selection

### Statistical Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=100, n_features=20, n_informative=5, 
                      noise=0.1, random_state=42)
feature_names = [f'feature_{i}' for i in range(20)]
X_df = pd.DataFrame(X, columns=feature_names)

# Method 1: Univariate feature selection (F-test)
selector_f = SelectKBest(score_func=f_regression, k=5)
X_selected_f = selector_f.fit_transform(X, y)
selected_features_f = [feature_names[i] for i in selector_f.get_support(indices=True)]

print("Top 5 features (F-test):")
print(selected_features_f)

# Method 2: Mutual information
selector_mi = SelectKBest(score_func=mutual_info_regression, k=5)
X_selected_mi = selector_mi.fit_transform(X, y)
selected_features_mi = [feature_names[i] for i in selector_mi.get_support(indices=True)]

print("Top 5 features (Mutual Information):")
print(selected_features_mi)

# Visualize feature scores
scores_f = selector_f.scores_
scores_mi = selector_mi.scores_

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(range(len(scores_f)), scores_f)
axes[0].set_title('F-test Scores')
axes[0].set_xlabel('Feature Index')
axes[0].set_ylabel('Score')

axes[1].bar(range(len(scores_mi)), scores_mi)
axes[1].set_title('Mutual Information Scores')
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Score')

plt.tight_layout()
plt.show()
```

### Correlation-Based Feature Selection
```python
def remove_correlated_features(df, threshold=0.95):
    """Remove highly correlated features"""
    corr_matrix = df.corr().abs()
    
    # Find pairs of highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_triangle.columns 
              if any(upper_triangle[column] > threshold)]
    
    return df.drop(columns=to_drop), to_drop

# Example with correlated features
np.random.seed(42)
corr_df = pd.DataFrame({
    'feature_1': np.random.randn(100),
    'feature_2': np.random.randn(100),
})
corr_df['feature_3'] = corr_df['feature_1'] + np.random.randn(100) * 0.1  # Highly correlated
corr_df['feature_4'] = corr_df['feature_2'] * 2 + np.random.randn(100) * 0.1  # Highly correlated
corr_df['feature_5'] = np.random.randn(100)  # Independent

print("Correlation Matrix:")
print(corr_df.corr().round(3))

cleaned_df, dropped_features = remove_correlated_features(corr_df, threshold=0.8)
print(f"\nDropped features: {dropped_features}")
print(f"Remaining features: {cleaned_df.columns.tolist()}")
```

## Dimensionality Reduction

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
X_iris = iris.data
feature_names = iris.feature_names

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_iris)

# Analyze explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("PCA Analysis:")
print(f"Original features: {len(feature_names)}")
print(f"Explained variance ratio: {explained_variance_ratio.round(3)}")
print(f"Cumulative variance: {cumulative_variance.round(3)}")

# Visualize explained variance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Explained Variance by Component')

axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()

plt.tight_layout()
plt.show()

# Choose number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Apply PCA with selected components
pca_reduced = PCA(n_components=n_components_95)
X_reduced = pca_reduced.fit_transform(X_iris)

print(f"Original shape: {X_iris.shape}")
print(f"Reduced shape: {X_reduced.shape}")
```

## Complete Feature Engineering Pipeline

### Comprehensive Example
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Create sample dataset
np.random.seed(42)
n_samples = 1000

sample_data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
    'experience': np.random.randint(0, 30, n_samples),
    'satisfaction': np.random.uniform(1, 10, n_samples),
    'date_hired': pd.date_range('2010-01-01', periods=n_samples, freq='D')
})

# Add some missing values
sample_data.loc[np.random.choice(n_samples, 50, replace=False), 'income'] = np.nan
sample_data.loc[np.random.choice(n_samples, 30, replace=False), 'satisfaction'] = np.nan

def comprehensive_feature_engineering(df):
    """Complete feature engineering pipeline"""
    df_processed = df.copy()
    
    # 1. DateTime features
    df_processed['years_employed'] = (pd.Timestamp.now() - df_processed['date_hired']).dt.days / 365.25
    df_processed['hire_month'] = df_processed['date_hired'].dt.month
    df_processed['hire_quarter'] = df_processed['date_hired'].dt.quarter
    
    # 2. Derived features
    df_processed['income_per_year_exp'] = df_processed['income'] / (df_processed['experience'] + 1)
    df_processed['age_when_hired'] = df_processed['age'] - df_processed['years_employed']
    
    # 3. Binning
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                      bins=[0, 30, 45, 60, 100], 
                                      labels=['Young', 'Middle', 'Senior', 'Elder'])
    
    # 4. Interaction features
    df_processed['education_experience'] = (df_processed['education'].astype('category').cat.codes * 
                                          df_processed['experience'])
    
    return df_processed

# Apply feature engineering
processed_data = comprehensive_feature_engineering(sample_data)

print("Original features:", sample_data.columns.tolist())
print("Processed features:", processed_data.columns.tolist())
print(f"Feature count: {len(sample_data.columns)} → {len(processed_data.columns)}")

# Create preprocessing pipeline
numeric_features = ['age', 'income', 'experience', 'satisfaction', 'years_employed', 
                   'income_per_year_exp', 'age_when_hired', 'education_experience']
categorical_features = ['education', 'department', 'age_group']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(processed_data)
feature_names_processed = (numeric_features + 
                          list(preprocessor.named_transformers_['cat']
                              .named_steps['onehot'].get_feature_names_out(categorical_features)))

print(f"Final processed shape: {X_processed.shape}")
print(f"Final feature names: {len(feature_names_processed)} features")
```

## Best Practices for Feature Engineering

### 1. Domain Knowledge Integration
```python
# Example: E-commerce features
def ecommerce_features(df):
    """Domain-specific features for e-commerce"""
    df_new = df.copy()
    
    # Recency, Frequency, Monetary (RFM) analysis
    df_new['days_since_last_purchase'] = (pd.Timestamp.now() - df_new['last_purchase_date']).dt.days
    df_new['avg_order_value'] = df_new['total_spent'] / df_new['num_orders']
    df_new['customer_lifetime_value'] = df_new['total_spent'] / df_new['days_since_first_purchase']
    
    # Behavioral features
    df_new['purchase_frequency'] = df_new['num_orders'] / df_new['days_since_first_purchase']
    df_new['cart_abandonment_rate'] = df_new['carts_abandoned'] / df_new['carts_created']
    
    return df_new
```

### 2. Feature Validation
```python
def validate_features(X_train, X_test, feature_names):
    """Validate engineered features"""
    validation_results = {}
    
    # Check for constant features
    constant_features = []
    for i, name in enumerate(feature_names):
        if len(np.unique(X_train[:, i])) == 1:
            constant_features.append(name)
    
    # Check for high correlation
    if X_train.shape[1] < 100:  # Only for manageable number of features
        corr_matrix = np.corrcoef(X_train.T)
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.95:
                    high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    # Check for data leakage (future information)
    # This would be domain-specific
    
    validation_results['constant_features'] = constant_features
    validation_results['high_correlation_pairs'] = high_corr_pairs if X_train.shape[1] < 100 else "Too many features to check"
    
    return validation_results
```

## Key Terminology

- **Feature**: An individual measurable property of observed phenomena
- **One-Hot Encoding**: Converting categorical variables to binary columns
- **Target Encoding**: Encoding categories based on target variable statistics
- **Feature Scaling**: Transforming features to similar scales
- **Dimensionality Reduction**: Reducing number of features while preserving information
- **Feature Selection**: Choosing subset of relevant features
- **Interaction Features**: Features created by combining existing features
- **Cyclical Encoding**: Representing cyclical data (time, angles) as sine/cosine

## Looking Ahead

In Lesson 15, we'll learn about:
- **Machine Learning Basics**: Supervised vs unsupervised learning
- **Model Training**: Fitting algorithms to data
- **Train/Test Split**: Evaluating model performance
- **Cross-Validation**: Robust model evaluation techniques
- **Model Selection**: Choosing the right algorithm for your problem
