#!/usr/bin/env python3
"""
Feature Engineering and Selection Demo
Comprehensive examples of feature engineering techniques
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_dataset():
    """Create a comprehensive sample dataset for feature engineering"""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Generate base features
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'years_employed': np.random.uniform(0, 40, n_samples),
        'num_accounts': np.random.randint(1, 10, n_samples),
        'total_balance': np.random.uniform(0, 100000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1]),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'signup_date': pd.date_range('2020-01-01', periods=n_samples, freq='1D'),
        'last_login': pd.date_range('2024-01-01', periods=n_samples, freq='2H')
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (loan approval) with some logic
    df['loan_approved'] = (
        (df['credit_score'] > 650) & 
        (df['income'] > 40000) & 
        (df['years_employed'] > 2)
    ).astype(int)
    
    # Add some noise to make it more realistic
    noise = np.random.random(n_samples) < 0.1
    df.loc[noise, 'loan_approved'] = 1 - df.loc[noise, 'loan_approved']
    
    return df

def numerical_feature_engineering(df):
    """Demonstrate numerical feature engineering techniques"""
    print("=== Numerical Feature Engineering ===")
    
    df_features = df.copy()
    
    # 1. Binning/Discretization
    df_features['age_group'] = pd.cut(df_features['age'], 
                                     bins=[0, 25, 35, 50, 65, 100], 
                                     labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
    
    df_features['income_bracket'] = pd.qcut(df_features['income'], 
                                           q=5, 
                                           labels=['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High'])
    
    # 2. Mathematical transformations
    df_features['log_income'] = np.log1p(df_features['income'])
    df_features['sqrt_age'] = np.sqrt(df_features['age'])
    df_features['income_squared'] = df_features['income'] ** 2
    
    # 3. Ratios and derived features
    df_features['debt_to_income'] = df_features['total_balance'] / df_features['income']
    df_features['balance_per_account'] = df_features['total_balance'] / df_features['num_accounts']
    df_features['income_per_year_employed'] = df_features['income'] / (df_features['years_employed'] + 1)
    
    # 4. Interaction features
    df_features['age_income_interaction'] = df_features['age'] * df_features['income']
    df_features['credit_employment_interaction'] = df_features['credit_score'] * df_features['years_employed']
    
    # 5. Polynomial features (degree 2)
    df_features['age_squared'] = df_features['age'] ** 2
    df_features['credit_score_squared'] = df_features['credit_score'] ** 2
    
    print(f"Original features: {df.shape[1]}")
    print(f"After numerical engineering: {df_features.shape[1]}")
    print(f"New numerical features created: {df_features.shape[1] - df.shape[1]}")
    
    # Show some statistics
    print(f"\nSample of new features:")
    new_features = ['age_group', 'income_bracket', 'log_income', 'debt_to_income', 'balance_per_account']
    print(df_features[new_features].head())
    
    return df_features

def categorical_feature_engineering(df):
    """Demonstrate categorical feature engineering techniques"""
    print("\n=== Categorical Feature Engineering ===")
    
    df_encoded = df.copy()
    
    # 1. One-Hot Encoding for nominal categories
    education_encoded = pd.get_dummies(df_encoded['education'], prefix='education')
    marital_encoded = pd.get_dummies(df_encoded['marital_status'], prefix='marital')
    city_encoded = pd.get_dummies(df_encoded['city'], prefix='city')
    
    # 2. Label Encoding for ordinal categories
    education_order = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df_encoded['education_ordinal'] = df_encoded['education'].map(education_order)
    
    # 3. Target Encoding (mean encoding)
    target_mean_education = df_encoded.groupby('education')['loan_approved'].mean()
    df_encoded['education_target_encoded'] = df_encoded['education'].map(target_mean_education)
    
    target_mean_city = df_encoded.groupby('city')['loan_approved'].mean()
    df_encoded['city_target_encoded'] = df_encoded['city'].map(target_mean_city)
    
    # 4. Frequency Encoding
    education_counts = df_encoded['education'].value_counts()
    df_encoded['education_frequency'] = df_encoded['education'].map(education_counts)
    
    city_counts = df_encoded['city'].value_counts()
    df_encoded['city_frequency'] = df_encoded['city'].map(city_counts)
    
    # Combine all encoded features
    df_final = pd.concat([df_encoded, education_encoded, marital_encoded, city_encoded], axis=1)
    
    print(f"One-hot encoded features: {education_encoded.shape[1] + marital_encoded.shape[1] + city_encoded.shape[1]}")
    print(f"Target encoded features: 2")
    print(f"Frequency encoded features: 2")
    print(f"Total categorical features added: {df_final.shape[1] - df_encoded.shape[1]}")
    
    return df_final

def datetime_feature_engineering(df):
    """Demonstrate datetime feature engineering"""
    print("\n=== Datetime Feature Engineering ===")
    
    df_time = df.copy()
    
    # Extract components from signup_date
    df_time['signup_year'] = df_time['signup_date'].dt.year
    df_time['signup_month'] = df_time['signup_date'].dt.month
    df_time['signup_day'] = df_time['signup_date'].dt.day
    df_time['signup_dayofweek'] = df_time['signup_date'].dt.dayofweek
    df_time['signup_quarter'] = df_time['signup_date'].dt.quarter
    df_time['signup_is_weekend'] = df_time['signup_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Extract components from last_login
    df_time['login_hour'] = df_time['last_login'].dt.hour
    df_time['login_dayofweek'] = df_time['last_login'].dt.dayofweek
    df_time['login_is_business_hours'] = df_time['last_login'].dt.hour.between(9, 17).astype(int)
    
    # Calculate time differences
    current_date = pd.Timestamp('2024-01-01')
    df_time['days_since_signup'] = (current_date - df_time['signup_date']).dt.days
    df_time['hours_since_login'] = (current_date - df_time['last_login']).dt.total_seconds() / 3600
    
    # Create seasonal features
    df_time['signup_season'] = df_time['signup_month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Cyclical encoding for circular features
    df_time['month_sin'] = np.sin(2 * np.pi * df_time['signup_month'] / 12)
    df_time['month_cos'] = np.cos(2 * np.pi * df_time['signup_month'] / 12)
    df_time['hour_sin'] = np.sin(2 * np.pi * df_time['login_hour'] / 24)
    df_time['hour_cos'] = np.cos(2 * np.pi * df_time['login_hour'] / 24)
    
    datetime_features = [
        'signup_year', 'signup_month', 'signup_day', 'signup_dayofweek', 'signup_quarter',
        'signup_is_weekend', 'login_hour', 'login_dayofweek', 'login_is_business_hours',
        'days_since_signup', 'hours_since_login', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'
    ]
    
    print(f"Datetime features created: {len(datetime_features)}")
    print(f"Sample datetime features:")
    print(df_time[datetime_features[:5]].head())
    
    return df_time

def feature_scaling_demo(df):
    """Demonstrate different feature scaling techniques"""
    print("\n=== Feature Scaling Demonstration ===")
    
    # Select numerical features for scaling
    numerical_features = ['age', 'income', 'credit_score', 'years_employed', 'total_balance']
    X = df[numerical_features].copy()
    
    print("Original feature statistics:")
    print(X.describe())
    
    # 1. StandardScaler (Z-score normalization)
    scaler_standard = StandardScaler()
    X_standard = pd.DataFrame(
        scaler_standard.fit_transform(X),
        columns=[f'{col}_standard' for col in X.columns]
    )
    
    # 2. MinMaxScaler (0-1 normalization)
    scaler_minmax = MinMaxScaler()
    X_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(X),
        columns=[f'{col}_minmax' for col in X.columns]
    )
    
    # 3. RobustScaler (robust to outliers)
    from sklearn.preprocessing import RobustScaler
    scaler_robust = RobustScaler()
    X_robust = pd.DataFrame(
        scaler_robust.fit_transform(X),
        columns=[f'{col}_robust' for col in X.columns]
    )
    
    print(f"\nStandardScaler results (mean=0, std=1):")
    print(X_standard.describe())
    
    print(f"\nMinMaxScaler results (min=0, max=1):")
    print(X_minmax.describe())
    
    print(f"\nRobustScaler results (median=0, IQR-based scaling):")
    print(X_robust.describe())
    
    return X_standard, X_minmax, X_robust

def feature_selection_demo(df):
    """Demonstrate feature selection techniques"""
    print("\n=== Feature Selection Demonstration ===")
    
    # Prepare features and target
    # Select numerical features for demonstration
    feature_columns = ['age', 'income', 'credit_score', 'years_employed', 'num_accounts', 'total_balance']
    X = df[feature_columns]
    y = df['loan_approved']
    
    print(f"Starting with {X.shape[1]} features")
    
    # 1. Univariate Feature Selection (SelectKBest)
    selector_univariate = SelectKBest(score_func=f_classif, k=4)
    X_univariate = selector_univariate.fit_transform(X, y)
    
    selected_features_univariate = [feature_columns[i] for i in selector_univariate.get_support(indices=True)]
    scores_univariate = selector_univariate.scores_
    
    print(f"\nUnivariate Selection (SelectKBest):")
    print(f"Selected features: {selected_features_univariate}")
    print(f"Feature scores: {dict(zip(feature_columns, scores_univariate))}")
    
    # 2. Recursive Feature Elimination (RFE)
    estimator = LogisticRegression(random_state=42)
    selector_rfe = RFE(estimator=estimator, n_features_to_select=4)
    X_rfe = selector_rfe.fit_transform(X, y)
    
    selected_features_rfe = [feature_columns[i] for i in selector_rfe.get_support(indices=True)]
    feature_ranking_rfe = selector_rfe.ranking_
    
    print(f"\nRecursive Feature Elimination (RFE):")
    print(f"Selected features: {selected_features_rfe}")
    print(f"Feature ranking: {dict(zip(feature_columns, feature_ranking_rfe))}")
    
    # 3. Tree-based Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = rf.feature_importances_
    importance_dict = dict(zip(feature_columns, feature_importance))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTree-based Feature Importance (Random Forest):")
    for feature, importance in sorted_importance:
        print(f"  {feature}: {importance:.4f}")
    
    # 4. Correlation-based Feature Selection
    correlation_matrix = X.corr()
    correlation_with_target = X.corrwith(pd.Series(y, index=X.index)).abs().sort_values(ascending=False)
    
    print(f"\nCorrelation with Target:")
    for feature, corr in correlation_with_target.items():
        print(f"  {feature}: {corr:.4f}")
    
    return selected_features_univariate, selected_features_rfe, sorted_importance

def advanced_feature_engineering(df):
    """Demonstrate advanced feature engineering techniques"""
    print("\n=== Advanced Feature Engineering ===")
    
    df_advanced = df.copy()
    
    # 1. Clustering-based features
    from sklearn.cluster import KMeans
    
    # Create customer segments based on age and income
    features_for_clustering = df_advanced[['age', 'income']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_clustering)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_advanced['customer_segment'] = kmeans.fit_predict(features_scaled)
    
    # 2. Distance-based features
    # Distance from cluster centers
    cluster_centers = kmeans.cluster_centers_
    distances = []
    for i, point in enumerate(features_scaled):
        cluster_id = df_advanced.iloc[i]['customer_segment']
        center = cluster_centers[cluster_id]
        distance = np.linalg.norm(point - center)
        distances.append(distance)
    
    df_advanced['distance_to_cluster_center'] = distances
    
    # 3. Rank-based features
    df_advanced['income_rank'] = df_advanced['income'].rank(pct=True)
    df_advanced['credit_score_rank'] = df_advanced['credit_score'].rank(pct=True)
    df_advanced['age_rank'] = df_advanced['age'].rank(pct=True)
    
    # 4. Statistical aggregations by groups
    # Group statistics by education level
    education_stats = df_advanced.groupby('education').agg({
        'income': ['mean', 'std', 'median'],
        'credit_score': ['mean', 'std'],
        'age': 'mean'
    }).round(2)
    
    # Map group statistics back to individual records
    education_income_mean = df_advanced.groupby('education')['income'].mean()
    df_advanced['education_income_mean'] = df_advanced['education'].map(education_income_mean)
    
    education_credit_mean = df_advanced.groupby('education')['credit_score'].mean()
    df_advanced['education_credit_mean'] = df_advanced['education'].map(education_credit_mean)
    
    # 5. Deviation from group mean
    df_advanced['income_deviation_from_education_mean'] = df_advanced['income'] - df_advanced['education_income_mean']
    df_advanced['credit_deviation_from_education_mean'] = df_advanced['credit_score'] - df_advanced['education_credit_mean']
    
    print(f"Advanced features created:")
    print(f"  - Customer segments: {df_advanced['customer_segment'].nunique()} clusters")
    print(f"  - Distance to cluster center")
    print(f"  - Rank-based features: 3")
    print(f"  - Group statistics features: 2")
    print(f"  - Deviation features: 2")
    
    print(f"\nCustomer segment distribution:")
    print(df_advanced['customer_segment'].value_counts().sort_index())
    
    return df_advanced

def create_feature_engineering_pipeline():
    """Create a comprehensive feature engineering pipeline"""
    print("\n=== Feature Engineering Pipeline ===")
    
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # Define feature types
    numerical_features = ['age', 'income', 'credit_score', 'years_employed', 'num_accounts', 'total_balance']
    categorical_features = ['education', 'marital_status', 'city']
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create full pipeline with model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    print("Pipeline created with:")
    print(f"  - Numerical features: {len(numerical_features)}")
    print(f"  - Categorical features: {len(categorical_features)}")
    print("  - StandardScaler for numerical features")
    print("  - OneHotEncoder for categorical features")
    print("  - LogisticRegression classifier")
    
    return full_pipeline, numerical_features, categorical_features

def main():
    """Run comprehensive feature engineering demonstration"""
    print("Feature Engineering and Selection Demonstration")
    print("=" * 60)
    
    # Create sample dataset
    df = create_sample_dataset()
    print(f"Created sample dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Numerical feature engineering
    df_numerical = numerical_feature_engineering(df)
    
    # Categorical feature engineering
    df_categorical = categorical_feature_engineering(df)
    
    # Datetime feature engineering
    df_datetime = datetime_feature_engineering(df)
    
    # Feature scaling demonstration
    X_standard, X_minmax, X_robust = feature_scaling_demo(df)
    
    # Feature selection demonstration
    selected_univariate, selected_rfe, importance_ranking = feature_selection_demo(df)
    
    # Advanced feature engineering
    df_advanced = advanced_feature_engineering(df)
    
    # Create pipeline
    pipeline, num_features, cat_features = create_feature_engineering_pipeline()
    
    print("\n" + "=" * 60)
    print("Feature engineering demonstration complete!")
    print("\nKey techniques demonstrated:")
    print("- Numerical transformations and interactions")
    print("- Categorical encoding methods")
    print("- Datetime feature extraction")
    print("- Feature scaling techniques")
    print("- Statistical feature selection")
    print("- Tree-based feature importance")
    print("- Advanced clustering and ranking features")
    print("- End-to-end feature engineering pipeline")
    
    return df, df_advanced, pipeline

if __name__ == "__main__":
    original_df, engineered_df, feature_pipeline = main()
