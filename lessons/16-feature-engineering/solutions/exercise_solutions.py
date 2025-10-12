# Lesson 14 Solutions: Feature Engineering and Selection

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Basic Feature Creation
print("Exercise 1: Basic Feature Creation")
print("-" * 50)

# Sample customer data
customer_data = {
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 80, 100),
    'income': np.random.normal(50000, 20000, 100),
    'purchase_date': pd.date_range('2023-01-01', periods=100, freq='3D'),
    'purchase_amount': np.random.uniform(10, 500, 100),
    'previous_purchases': np.random.randint(0, 20, 100)
}

df = pd.DataFrame(customer_data)
df['income'] = np.maximum(df['income'], 20000)  # Ensure positive income

print("Original dataset:")
print(df.head())

# Create new features
print(f"\nCreating new features...")

# 1. Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])

# 2. Income brackets
df['income_bracket'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# 3. Days since last purchase
current_date = pd.Timestamp('2024-01-01')
df['days_since_purchase'] = (current_date - df['purchase_date']).dt.days

# 4. Purchase frequency (purchases per year)
df['purchase_frequency'] = df['previous_purchases'] / ((df['days_since_purchase'] + 1) / 365.25)

# 5. Customer lifetime value estimate
df['estimated_clv'] = df['purchase_amount'] * df['previous_purchases'] * 1.2

print(f"New features created:")
print(f"  - age_group: {df['age_group'].nunique()} categories")
print(f"  - income_bracket: {df['income_bracket'].nunique()} categories")
print(f"  - days_since_purchase: numerical")
print(f"  - purchase_frequency: numerical")
print(f"  - estimated_clv: numerical")

print(f"\nSample of new features:")
new_features = ['age_group', 'income_bracket', 'days_since_purchase', 'purchase_frequency', 'estimated_clv']
print(df[new_features].head())

# Exercise 2: Categorical Feature Encoding
print("\n" + "="*50)
print("Exercise 2: Categorical Feature Encoding")
print("-" * 50)

# Sample data with categorical variables
np.random.seed(42)
data = {
    'product_category': ['Electronics', 'Clothing', 'Books', 'Electronics', 'Home'] * 20,
    'customer_segment': ['Premium', 'Standard', 'Basic', 'Premium', 'Standard'] * 20,
    'satisfaction_level': ['Very Low', 'Low', 'Medium', 'High', 'Very High'] * 20,
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'] * 20,
    'sales': np.random.uniform(100, 1000, 100)
}

categorical_df = pd.DataFrame(data)

print("Original categorical data:")
print(categorical_df.head())
print(f"Data types:\n{categorical_df.dtypes}")

# 1. One-hot encoding for nominal categories
print(f"\n1. One-Hot Encoding:")
product_encoded = pd.get_dummies(categorical_df['product_category'], prefix='product')
segment_encoded = pd.get_dummies(categorical_df['customer_segment'], prefix='segment')
city_encoded = pd.get_dummies(categorical_df['city'], prefix='city')

print(f"Product categories encoded: {product_encoded.shape[1]} columns")
print(f"Customer segments encoded: {segment_encoded.shape[1]} columns")
print(f"Cities encoded: {city_encoded.shape[1]} columns")

# 2. Label encoding for ordinal categories
print(f"\n2. Label Encoding (Ordinal):")
satisfaction_order = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
categorical_df['satisfaction_encoded'] = categorical_df['satisfaction_level'].map(satisfaction_order)

print(f"Satisfaction levels encoded:")
print(categorical_df[['satisfaction_level', 'satisfaction_encoded']].drop_duplicates())

# 3. Target encoding for high-cardinality categories
print(f"\n3. Target Encoding:")
target_mean_product = categorical_df.groupby('product_category')['sales'].mean()
categorical_df['product_target_encoded'] = categorical_df['product_category'].map(target_mean_product)

target_mean_city = categorical_df.groupby('city')['sales'].mean()
categorical_df['city_target_encoded'] = categorical_df['city'].map(target_mean_city)

print(f"Target encoding results:")
print(f"Product category means: {target_mean_product.to_dict()}")
print(f"City means: {target_mean_city.to_dict()}")

# 4. Frequency encoding
print(f"\n4. Frequency Encoding:")
product_counts = categorical_df['product_category'].value_counts()
categorical_df['product_frequency'] = categorical_df['product_category'].map(product_counts)

city_counts = categorical_df['city'].value_counts()
categorical_df['city_frequency'] = categorical_df['city'].map(city_counts)

print(f"Frequency encoding results:")
print(f"Product frequencies: {product_counts.to_dict()}")
print(f"City frequencies: {city_counts.to_dict()}")

# Exercise 3: Feature Scaling and Normalization
print("\n" + "="*50)
print("Exercise 3: Feature Scaling and Normalization")
print("-" * 50)

# Sample data with different scales
np.random.seed(42)
features = {
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'years_employed': np.random.uniform(0, 40, 1000)
}

scaling_df = pd.DataFrame(features)
scaling_df['income'] = np.maximum(scaling_df['income'], 10000)  # Ensure positive

print("Original data statistics:")
print(scaling_df.describe())

# 1. StandardScaler (Z-score normalization)
print(f"\n1. StandardScaler (Z-score normalization):")
scaler_standard = StandardScaler()
scaled_standard = pd.DataFrame(
    scaler_standard.fit_transform(scaling_df),
    columns=[f'{col}_standard' for col in scaling_df.columns]
)

print("After StandardScaler (mean ≈ 0, std ≈ 1):")
print(scaled_standard.describe())

# 2. MinMaxScaler (0-1 normalization)
print(f"\n2. MinMaxScaler (0-1 normalization):")
scaler_minmax = MinMaxScaler()
scaled_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(scaling_df),
    columns=[f'{col}_minmax' for col in scaling_df.columns]
)

print("After MinMaxScaler (min = 0, max = 1):")
print(scaled_minmax.describe())

# 3. RobustScaler (robust to outliers)
print(f"\n3. RobustScaler (robust to outliers):")
from sklearn.preprocessing import RobustScaler
scaler_robust = RobustScaler()
scaled_robust = pd.DataFrame(
    scaler_robust.fit_transform(scaling_df),
    columns=[f'{col}_robust' for col in scaling_df.columns]
)

print("After RobustScaler (median-based scaling):")
print(scaled_robust.describe())

# Compare the effects
print(f"\n4. Comparison of scaling methods:")
comparison_data = {
    'Original': [scaling_df['income'].mean(), scaling_df['income'].std()],
    'StandardScaler': [scaled_standard['income_standard'].mean(), scaled_standard['income_standard'].std()],
    'MinMaxScaler': [scaled_minmax['income_minmax'].mean(), scaled_minmax['income_minmax'].std()],
    'RobustScaler': [scaled_robust['income_robust'].mean(), scaled_robust['income_robust'].std()]
}

comparison_df = pd.DataFrame(comparison_data, index=['Mean', 'Std Dev'])
print("Income feature comparison:")
print(comparison_df.round(4))

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: E-commerce Feature Engineering
print("Exercise 4: E-commerce Feature Engineering")
print("-" * 50)

# Generate e-commerce transaction data
np.random.seed(42)

transactions = []
customers = range(1, 1001)
products = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty']

for customer in customers:
    num_transactions = np.random.poisson(5) + 1
    
    for _ in range(num_transactions):
        transaction = {
            'customer_id': customer,
            'transaction_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'product_category': np.random.choice(products),
            'amount': np.random.uniform(10, 500),
            'quantity': np.random.randint(1, 5),
            'discount_used': np.random.choice([True, False], p=[0.3, 0.7]),
            'payment_method': np.random.choice(['Credit', 'Debit', 'PayPal', 'Cash'])
        }
        transactions.append(transaction)

transactions_df = pd.DataFrame(transactions)

print(f"Generated {len(transactions_df)} transactions for {len(customers)} customers")
print("Sample transactions:")
print(transactions_df.head())

# Feature Engineering
print(f"\nFeature Engineering:")

# 1. Recency Features
current_date = pd.Timestamp('2024-01-01')
customer_recency = transactions_df.groupby('customer_id').agg({
    'transaction_date': ['min', 'max']
}).round(2)

customer_recency.columns = ['first_purchase', 'last_purchase']
customer_recency['days_since_first'] = (current_date - customer_recency['first_purchase']).dt.days
customer_recency['days_since_last'] = (current_date - customer_recency['last_purchase']).dt.days

# 2. Frequency Features
customer_frequency = transactions_df.groupby('customer_id').agg({
    'transaction_date': 'count',
    'amount': 'count'
}).round(2)

customer_frequency.columns = ['transaction_count', 'purchase_count']
customer_frequency['avg_days_between_purchases'] = (
    customer_recency['days_since_first'] / customer_frequency['transaction_count']
)

# 3. Monetary Features
customer_monetary = transactions_df.groupby('customer_id').agg({
    'amount': ['sum', 'mean', 'max', 'std'],
    'quantity': 'sum'
}).round(2)

customer_monetary.columns = ['total_spent', 'avg_order_value', 'max_purchase', 'spending_std', 'total_quantity']

# 4. Behavioral Features
# Favorite category
favorite_category = transactions_df.groupby(['customer_id', 'product_category']).size().reset_index(name='count')
favorite_category = favorite_category.loc[favorite_category.groupby('customer_id')['count'].idxmax()]
favorite_category = favorite_category.set_index('customer_id')['product_category']

# Payment method preference
payment_preference = transactions_df.groupby(['customer_id', 'payment_method']).size().reset_index(name='count')
payment_preference = payment_preference.loc[payment_preference.groupby('customer_id')['count'].idxmax()]
payment_preference = payment_preference.set_index('customer_id')['payment_method']

# Discount usage rate
discount_usage = transactions_df.groupby('customer_id')['discount_used'].mean()

# 5. Seasonal Features
transactions_df['month'] = transactions_df['transaction_date'].dt.month
transactions_df['day_of_week'] = transactions_df['transaction_date'].dt.dayofweek
transactions_df['season'] = transactions_df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

seasonal_patterns = transactions_df.groupby(['customer_id', 'season']).size().unstack(fill_value=0)

# Combine all features
customer_features = pd.concat([
    customer_recency[['days_since_first', 'days_since_last']],
    customer_frequency,
    customer_monetary,
    favorite_category.rename('favorite_category'),
    payment_preference.rename('preferred_payment'),
    discount_usage.rename('discount_usage_rate'),
    seasonal_patterns
], axis=1)

print(f"Customer features created: {customer_features.shape[1]} features")
print(f"Sample customer features:")
print(customer_features.head())

# RFM Analysis
print(f"\nRFM Analysis:")
customer_features['R_score'] = pd.qcut(customer_features['days_since_last'], 5, labels=[5,4,3,2,1])
customer_features['F_score'] = pd.qcut(customer_features['transaction_count'].rank(method='first'), 5, labels=[1,2,3,4,5])
customer_features['M_score'] = pd.qcut(customer_features['total_spent'], 5, labels=[1,2,3,4,5])

customer_features['RFM_score'] = (
    customer_features['R_score'].astype(str) + 
    customer_features['F_score'].astype(str) + 
    customer_features['M_score'].astype(str)
)

print(f"RFM segments created:")
print(customer_features['RFM_score'].value_counts().head(10))

# Exercise 5: Time Series Feature Engineering
print("\n" + "="*50)
print("Exercise 5: Time Series Feature Engineering")
print("-" * 50)

# Generate time series sales data
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
trend = np.linspace(1000, 1500, len(dates))
seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 50, len(dates))
sales = trend + seasonal + noise

ts_df = pd.DataFrame({
    'date': dates,
    'sales': sales
})

print(f"Time series data: {len(ts_df)} daily observations")
print("Sample data:")
print(ts_df.head())

# Time series feature engineering
print(f"\nTime Series Feature Engineering:")

# 1. Lag features
for lag in [1, 7, 30]:
    ts_df[f'sales_lag_{lag}'] = ts_df['sales'].shift(lag)

# 2. Rolling statistics
for window in [7, 30]:
    ts_df[f'sales_rolling_mean_{window}'] = ts_df['sales'].rolling(window=window).mean()
    ts_df[f'sales_rolling_std_{window}'] = ts_df['sales'].rolling(window=window).std()
    ts_df[f'sales_rolling_min_{window}'] = ts_df['sales'].rolling(window=window).min()
    ts_df[f'sales_rolling_max_{window}'] = ts_df['sales'].rolling(window=window).max()

# 3. Date-based features
ts_df['year'] = ts_df['date'].dt.year
ts_df['month'] = ts_df['date'].dt.month
ts_df['day'] = ts_df['date'].dt.day
ts_df['day_of_week'] = ts_df['date'].dt.dayofweek
ts_df['day_of_year'] = ts_df['date'].dt.dayofyear
ts_df['week_of_year'] = ts_df['date'].dt.isocalendar().week
ts_df['quarter'] = ts_df['date'].dt.quarter
ts_df['is_weekend'] = ts_df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# 4. Cyclical features
ts_df['month_sin'] = np.sin(2 * np.pi * ts_df['month'] / 12)
ts_df['month_cos'] = np.cos(2 * np.pi * ts_df['month'] / 12)
ts_df['day_of_year_sin'] = np.sin(2 * np.pi * ts_df['day_of_year'] / 365.25)
ts_df['day_of_year_cos'] = np.cos(2 * np.pi * ts_df['day_of_year'] / 365.25)

# 5. Trend and change features
ts_df['sales_diff'] = ts_df['sales'].diff()
ts_df['sales_pct_change'] = ts_df['sales'].pct_change()
ts_df['sales_diff_7'] = ts_df['sales'].diff(7)  # Week-over-week change

# 6. Seasonal decomposition features
from statsmodels.tsa.seasonal import seasonal_decompose

# Use a subset for decomposition (seasonal_decompose needs complete data)
decomposition_data = ts_df.dropna().copy()
if len(decomposition_data) > 365 * 2:  # Need at least 2 years for annual seasonality
    decomp = seasonal_decompose(decomposition_data['sales'], model='additive', period=365)
    
    # Add decomposition components back to original dataframe
    ts_df.loc[decomposition_data.index, 'trend'] = decomp.trend
    ts_df.loc[decomposition_data.index, 'seasonal'] = decomp.seasonal
    ts_df.loc[decomposition_data.index, 'residual'] = decomp.resid

print(f"Time series features created: {ts_df.shape[1] - 2} features")  # -2 for original date and sales
print(f"Sample features:")
feature_cols = ['sales_lag_1', 'sales_rolling_mean_7', 'month', 'day_of_week', 'is_weekend', 'sales_diff']
print(ts_df[feature_cols].head(10))

print("\n=== CHALLENGE EXERCISE SOLUTIONS ===\n")

# Challenge 1: Advanced Feature Selection
print("Challenge 1: Advanced Feature Selection")
print("-" * 50)

# Create a dataset for feature selection
np.random.seed(42)
n_samples = 1000
n_features = 20

# Generate features with different levels of importance
X = np.random.randn(n_samples, n_features)
# Make some features more predictive
X[:, 0] = X[:, 0] + 2 * np.random.randn(n_samples)  # Important feature
X[:, 1] = X[:, 1] + 1.5 * np.random.randn(n_samples)  # Moderately important
X[:, 2] = X[:, 2] + np.random.randn(n_samples)  # Somewhat important

# Create target variable
y = (2 * X[:, 0] + 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) > 0).astype(int)

feature_names = [f'feature_{i}' for i in range(n_features)]
X_df = pd.DataFrame(X, columns=feature_names)

print(f"Dataset for feature selection: {X_df.shape[0]} samples, {X_df.shape[1]} features")

# 1. Filter Methods
print(f"\n1. Filter Methods:")

# Correlation with target
correlations = []
for col in X_df.columns:
    corr = np.corrcoef(X_df[col], y)[0, 1]
    correlations.append((col, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)
print(f"Top 5 features by correlation:")
for feature, corr in correlations[:5]:
    print(f"  {feature}: {corr:.4f}")

# Mutual information
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_df, y, random_state=42)
mi_ranking = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)

print(f"\nTop 5 features by mutual information:")
for feature, score in mi_ranking[:5]:
    print(f"  {feature}: {score:.4f}")

# 2. Wrapper Methods
print(f"\n2. Wrapper Methods:")

# Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=5)
rfe.fit(X_df, y)

selected_features_rfe = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
feature_ranking = [(feature_names[i], rfe.ranking_[i]) for i in range(len(feature_names))]
feature_ranking.sort(key=lambda x: x[1])

print(f"RFE selected features: {selected_features_rfe}")
print(f"Feature ranking (top 5):")
for feature, rank in feature_ranking[:5]:
    print(f"  {feature}: rank {rank}")

# 3. Embedded Methods
print(f"\n3. Embedded Methods:")

# LASSO feature selection
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_df, y)

lasso_coefs = [(feature_names[i], abs(lasso.coef_[i])) for i in range(len(feature_names))]
lasso_coefs.sort(key=lambda x: x[1], reverse=True)

print(f"LASSO coefficients (top 5):")
for feature, coef in lasso_coefs[:5]:
    print(f"  {feature}: {coef:.4f}")

# Tree-based feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_df, y)

feature_importance = [(feature_names[i], rf.feature_importances_[i]) for i in range(len(feature_names))]
feature_importance.sort(key=lambda x: x[1], reverse=True)

print(f"\nRandom Forest feature importance (top 5):")
for feature, importance in feature_importance[:5]:
    print(f"  {feature}: {importance:.4f}")

# 4. Hybrid approach - combine multiple methods
print(f"\n4. Hybrid Feature Selection:")

# Create scoring system
feature_scores = {}
for feature in feature_names:
    feature_scores[feature] = 0

# Add scores from different methods
for feature, corr in correlations:
    feature_scores[feature] += corr * 10  # Weight correlation

for feature, mi in mi_ranking:
    feature_scores[feature] += mi * 10  # Weight mutual information

for feature in selected_features_rfe:
    feature_scores[feature] += 5  # Bonus for RFE selection

for feature, coef in lasso_coefs:
    feature_scores[feature] += coef * 10  # Weight LASSO coefficient

for feature, importance in feature_importance:
    feature_scores[feature] += importance * 10  # Weight RF importance

# Sort by combined score
hybrid_ranking = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

print(f"Hybrid feature ranking (top 10):")
for i, (feature, score) in enumerate(hybrid_ranking[:10]):
    print(f"  {i+1}. {feature}: {score:.2f}")

# Final feature selection
top_features = [feature for feature, score in hybrid_ranking[:5]]
print(f"\nFinal selected features: {top_features}")

# Evaluate feature selection
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Compare performance with all features vs selected features
models = {
    'All Features': LogisticRegression(random_state=42),
    'Selected Features': LogisticRegression(random_state=42)
}

print(f"\n5. Feature Selection Evaluation:")
for name, model in models.items():
    if name == 'All Features':
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        n_features = X_train.shape[1]
    else:
        model.fit(X_train[top_features], y_train)
        score = model.score(X_test[top_features], y_test)
        n_features = len(top_features)
    
    print(f"  {name}: {score:.4f} accuracy with {n_features} features")

print("\n" + "="*50)
print("Feature engineering exercise solutions complete!")
print("\nKey techniques demonstrated:")
print("- Basic feature creation from existing data")
print("- Comprehensive categorical encoding methods")
print("- Multiple feature scaling approaches")
print("- E-commerce customer behavior features")
print("- Time series feature engineering")
print("- Advanced statistical feature selection")
print("- Hybrid feature selection combining multiple methods")
print("- Feature selection evaluation and validation")
