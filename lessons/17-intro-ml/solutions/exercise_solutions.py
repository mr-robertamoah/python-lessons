# Lesson 15 Solutions: Introduction to Machine Learning

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Classification with Scikit-Learn
print("Exercise 1: Classification with Scikit-Learn")
print("-" * 50)

iris = load_iris()
X, y = iris.data, iris.target

print(f"Dataset shape: {X.shape}")
print(f"Target classes: {iris.target_names}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Exercise 2: Regression Analysis
print("\n" + "="*50)
print("Exercise 2: Regression Analysis")
print("-" * 50)

# Generate synthetic housing data
np.random.seed(42)
n_samples = 500
X = np.random.randn(n_samples, 3)  # 3 features
y = 2*X[:, 0] + 1.5*X[:, 1] - 0.5*X[:, 2] + np.random.randn(n_samples)*0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"Coefficients: {reg.coef_}")

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Customer Churn Prediction
print("Exercise 4: Customer Churn Prediction")
print("-" * 50)

# Create synthetic customer data
np.random.seed(42)
n_customers = 1000

customer_data = {
    'tenure': np.random.randint(1, 72, n_customers),
    'monthly_charges': np.random.uniform(20, 120, n_customers),
    'total_charges': np.random.uniform(100, 8000, n_customers),
    'contract_length': np.random.choice([1, 12, 24], n_customers),
    'support_calls': np.random.poisson(2, n_customers)
}

df = pd.DataFrame(customer_data)

# Create target variable (churn) with some logic
churn_prob = (
    0.1 + 
    0.3 * (df['tenure'] < 12) + 
    0.2 * (df['monthly_charges'] > 80) + 
    0.1 * (df['support_calls'] > 3)
)
df['churn'] = np.random.binomial(1, churn_prob)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['churn'].mean():.3f}")

# Prepare features
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.3f}")

# Exercise 5: Sales Forecasting
print("\n" + "="*50)
print("Exercise 5: Sales Forecasting")
print("-" * 50)

# Generate time series sales data
dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
trend = np.linspace(1000, 1200, len(dates))
seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 50, len(dates))
sales = trend + seasonal + noise

sales_df = pd.DataFrame({
    'date': dates,
    'sales': sales
})

# Create features
sales_df['day_of_year'] = sales_df['date'].dt.dayofyear
sales_df['month'] = sales_df['date'].dt.month
sales_df['trend'] = np.arange(len(sales_df))

# Prepare for modeling
X = sales_df[['day_of_year', 'month', 'trend']]
y = sales_df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Sales Forecasting MSE: {mse:.2f}")
print(f"Sales Forecasting R²: {r2:.3f}")

# Exercise 6: Clustering Analysis
print("\n" + "="*50)
print("Exercise 6: Clustering Analysis")
print("-" * 50)

# Generate customer transaction data
np.random.seed(42)
n_customers = 300

transaction_data = {
    'avg_purchase_amount': np.random.lognormal(4, 1, n_customers),
    'purchase_frequency': np.random.poisson(5, n_customers),
    'days_since_last_purchase': np.random.exponential(30, n_customers)
}

cluster_df = pd.DataFrame(transaction_data)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

cluster_df['cluster'] = clusters

print(f"Clustering completed with {len(np.unique(clusters))} clusters")
print(f"Cluster distribution:")
print(cluster_df['cluster'].value_counts().sort_index())

# Analyze clusters
print(f"\nCluster characteristics:")
for cluster in sorted(cluster_df['cluster'].unique()):
    cluster_data = cluster_df[cluster_df['cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(f"  Avg purchase amount: ${cluster_data['avg_purchase_amount'].mean():.2f}")
    print(f"  Purchase frequency: {cluster_data['purchase_frequency'].mean():.1f}")
    print(f"  Days since last purchase: {cluster_data['days_since_last_purchase'].mean():.1f}")

print("\n" + "="*50)
print("Machine Learning exercise solutions complete!")
print("Key concepts demonstrated:")
print("- Classification with decision trees")
print("- Regression analysis and evaluation")
print("- Customer churn prediction")
print("- Sales forecasting with time series features")
print("- Customer segmentation with clustering")
