# Lesson 20: Real-World Data Project

## Learning Objectives
By the end of this lesson, you will be able to:
- Plan and execute a complete data analysis project
- Build end-to-end data pipelines
- Create professional documentation and presentations
- Apply all learned concepts in a real-world scenario
- Develop a portfolio-worthy project

## Project Overview: E-commerce Sales Analysis

### Business Problem
You're a data analyst for an e-commerce company. The business wants to:
- Understand sales patterns and trends
- Identify top-performing products and regions
- Predict future sales for inventory planning
- Optimize marketing strategies based on customer behavior

### Project Structure
```
ecommerce_analysis/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── analysis.py
│   ├── modeling.py
│   └── visualization.py
├── tests/
├── reports/
├── requirements.txt
└── README.md
```

## Phase 1: Data Collection and Setup

### Project Setup
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class EcommerceDataGenerator:
    """Generate realistic e-commerce data for analysis"""
    
    def __init__(self, start_date='2023-01-01', end_date='2024-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
    def generate_products(self, n_products=100):
        """Generate product catalog"""
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty']
        
        products = []
        for i in range(n_products):
            category = np.random.choice(categories)
            base_price = np.random.uniform(10, 500)
            
            # Category-specific price adjustments
            if category == 'Electronics':
                base_price *= np.random.uniform(2, 10)
            elif category == 'Books':
                base_price = np.random.uniform(5, 50)
            
            products.append({
                'product_id': f'PROD_{i:04d}',
                'product_name': f'{category} Product {i}',
                'category': category,
                'price': round(base_price, 2),
                'cost': round(base_price * np.random.uniform(0.3, 0.7), 2)
            })
        
        return pd.DataFrame(products)
    
    def generate_customers(self, n_customers=1000):
        """Generate customer data"""
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        customers = []
        for i in range(n_customers):
            age = np.random.normal(35, 12)
            age = max(18, min(80, int(age)))
            
            customers.append({
                'customer_id': f'CUST_{i:05d}',
                'age': age,
                'region': np.random.choice(regions),
                'registration_date': self.start_date + timedelta(
                    days=np.random.randint(0, (self.end_date - self.start_date).days)
                ),
                'customer_segment': np.random.choice(['Premium', 'Standard', 'Budget'], 
                                                   p=[0.2, 0.6, 0.2])
            })
        
        return pd.DataFrame(customers)
    
    def generate_sales(self, products_df, customers_df, n_transactions=10000):
        """Generate sales transactions"""
        transactions = []
        
        for i in range(n_transactions):
            # Random date with seasonal patterns
            days_from_start = np.random.randint(0, (self.end_date - self.start_date).days)
            transaction_date = self.start_date + timedelta(days=days_from_start)
            
            # Seasonal adjustment (higher sales in Nov-Dec)
            seasonal_multiplier = 1.0
            if transaction_date.month in [11, 12]:
                seasonal_multiplier = 1.5
            elif transaction_date.month in [6, 7]:
                seasonal_multiplier = 1.2
            
            # Select customer and product
            customer = customers_df.sample(1).iloc[0]
            product = products_df.sample(1).iloc[0]
            
            # Quantity based on product category and customer segment
            base_quantity = np.random.poisson(2) + 1
            if customer['customer_segment'] == 'Premium':
                base_quantity = int(base_quantity * 1.5)
            
            quantity = max(1, int(base_quantity * seasonal_multiplier))
            
            # Calculate totals
            unit_price = product['price']
            total_amount = unit_price * quantity
            
            # Apply discounts occasionally
            discount = 0
            if np.random.random() < 0.15:  # 15% chance of discount
                discount = np.random.uniform(0.05, 0.25)
                total_amount *= (1 - discount)
            
            transactions.append({
                'transaction_id': f'TXN_{i:06d}',
                'date': transaction_date,
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'quantity': quantity,
                'unit_price': unit_price,
                'discount': discount,
                'total_amount': round(total_amount, 2)
            })
        
        return pd.DataFrame(transactions)

# Generate datasets
generator = EcommerceDataGenerator()
products_df = generator.generate_products(100)
customers_df = generator.generate_customers(1000)
sales_df = generator.generate_sales(products_df, customers_df, 10000)

print("Generated datasets:")
print(f"Products: {len(products_df)} records")
print(f"Customers: {len(customers_df)} records")
print(f"Sales: {len(sales_df)} records")
```

## Phase 2: Data Exploration and Cleaning

### Exploratory Data Analysis
```python
class DataExplorer:
    """Comprehensive data exploration toolkit"""
    
    def __init__(self, sales_df, products_df, customers_df):
        self.sales = sales_df
        self.products = products_df
        self.customers = customers_df
        
    def basic_info(self):
        """Display basic information about datasets"""
        print("=== DATASET OVERVIEW ===")
        
        for name, df in [('Sales', self.sales), ('Products', self.products), ('Customers', self.customers)]:
            print(f"\n{name} Dataset:")
            print(f"Shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"Missing values: {df.isnull().sum().sum()}")
            
    def sales_overview(self):
        """Analyze sales data patterns"""
        # Merge with product and customer data
        sales_enriched = self.sales.merge(self.products, on='product_id')
        sales_enriched = sales_enriched.merge(self.customers, on='customer_id')
        
        print("=== SALES ANALYSIS ===")
        print(f"Date range: {self.sales['date'].min()} to {self.sales['date'].max()}")
        print(f"Total revenue: ${self.sales['total_amount'].sum():,.2f}")
        print(f"Average order value: ${self.sales['total_amount'].mean():.2f}")
        print(f"Total transactions: {len(self.sales):,}")
        
        # Top categories by revenue
        category_revenue = sales_enriched.groupby('category')['total_amount'].sum().sort_values(ascending=False)
        print(f"\nTop categories by revenue:")
        for category, revenue in category_revenue.head().items():
            print(f"  {category}: ${revenue:,.2f}")
        
        return sales_enriched
    
    def create_visualizations(self, sales_enriched):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('E-commerce Sales Analysis Dashboard', fontsize=16)
        
        # 1. Sales over time
        daily_sales = sales_enriched.groupby('date')['total_amount'].sum()
        axes[0, 0].plot(daily_sales.index, daily_sales.values)
        axes[0, 0].set_title('Daily Sales Trend')
        axes[0, 0].set_ylabel('Sales ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Sales by category
        category_sales = sales_enriched.groupby('category')['total_amount'].sum()
        axes[0, 1].bar(category_sales.index, category_sales.values)
        axes[0, 1].set_title('Sales by Category')
        axes[0, 1].set_ylabel('Sales ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Sales by region
        region_sales = sales_enriched.groupby('region')['total_amount'].sum()
        axes[0, 2].pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Sales Distribution by Region')
        
        # 4. Customer segment analysis
        segment_stats = sales_enriched.groupby('customer_segment').agg({
            'total_amount': ['sum', 'mean', 'count']
        }).round(2)
        
        segment_revenue = sales_enriched.groupby('customer_segment')['total_amount'].sum()
        axes[1, 0].bar(segment_revenue.index, segment_revenue.values)
        axes[1, 0].set_title('Revenue by Customer Segment')
        axes[1, 0].set_ylabel('Revenue ($)')
        
        # 5. Monthly sales pattern
        sales_enriched['month'] = sales_enriched['date'].dt.month
        monthly_sales = sales_enriched.groupby('month')['total_amount'].sum()
        axes[1, 1].plot(monthly_sales.index, monthly_sales.values, marker='o')
        axes[1, 1].set_title('Monthly Sales Pattern')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Sales ($)')
        
        # 6. Price vs Quantity relationship
        axes[1, 2].scatter(sales_enriched['unit_price'], sales_enriched['quantity'], alpha=0.5)
        axes[1, 2].set_title('Price vs Quantity')
        axes[1, 2].set_xlabel('Unit Price ($)')
        axes[1, 2].set_ylabel('Quantity')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Perform exploration
explorer = DataExplorer(sales_df, products_df, customers_df)
explorer.basic_info()
sales_enriched = explorer.sales_overview()
dashboard = explorer.create_visualizations(sales_enriched)
```

## Phase 3: Advanced Analysis

### Statistical Analysis and Insights
```python
class BusinessAnalyzer:
    """Advanced business analysis and insights"""
    
    def __init__(self, sales_enriched):
        self.data = sales_enriched
        
    def customer_segmentation_analysis(self):
        """Analyze customer behavior by segment"""
        
        # RFM Analysis (Recency, Frequency, Monetary)
        current_date = self.data['date'].max()
        
        rfm = self.data.groupby('customer_id').agg({
            'date': lambda x: (current_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'total_amount': 'sum'  # Monetary
        }).round(2)
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Score customers (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Combine scores
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Define customer segments
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            else:
                return 'Others'
        
        rfm['Segment'] = rfm.apply(segment_customers, axis=1)
        
        # Segment analysis
        segment_analysis = rfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean'
        }).round(2)
        
        segment_counts = rfm['Segment'].value_counts()
        
        print("=== CUSTOMER SEGMENTATION ANALYSIS ===")
        print("\nSegment Distribution:")
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm)) * 100
            print(f"{segment}: {count} customers ({percentage:.1f}%)")
        
        print("\nSegment Characteristics:")
        print(segment_analysis)
        
        return rfm
    
    def product_performance_analysis(self):
        """Analyze product performance and profitability"""
        
        # Product metrics
        product_metrics = self.data.groupby(['product_id', 'product_name', 'category']).agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        product_metrics.columns = ['product_id', 'product_name', 'category', 
                                 'total_quantity', 'total_revenue', 'total_orders']
        
        # Add profitability
        product_costs = self.data.groupby('product_id')['cost'].first()
        product_metrics = product_metrics.merge(
            product_costs.reset_index(), on='product_id'
        )
        
        product_metrics['total_profit'] = (
            product_metrics['total_revenue'] - 
            (product_metrics['total_quantity'] * product_metrics['cost'])
        )
        
        product_metrics['profit_margin'] = (
            product_metrics['total_profit'] / product_metrics['total_revenue'] * 100
        )
        
        # Rank products
        product_metrics['revenue_rank'] = product_metrics['total_revenue'].rank(ascending=False)
        product_metrics['profit_rank'] = product_metrics['total_profit'].rank(ascending=False)
        
        print("=== PRODUCT PERFORMANCE ANALYSIS ===")
        
        print("\nTop 10 Products by Revenue:")
        top_revenue = product_metrics.nlargest(10, 'total_revenue')[
            ['product_name', 'category', 'total_revenue', 'profit_margin']
        ]
        print(top_revenue.to_string(index=False))
        
        print("\nTop 10 Products by Profit:")
        top_profit = product_metrics.nlargest(10, 'total_profit')[
            ['product_name', 'category', 'total_profit', 'profit_margin']
        ]
        print(top_profit.to_string(index=False))
        
        return product_metrics
    
    def seasonal_analysis(self):
        """Analyze seasonal patterns and trends"""
        
        # Add time features
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['quarter'] = self.data['date'].dt.quarter
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['week_of_year'] = self.data['date'].dt.isocalendar().week
        
        # Monthly analysis
        monthly_analysis = self.data.groupby(['year', 'month']).agg({
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum'
        }).round(2)
        
        # Quarterly analysis
        quarterly_analysis = self.data.groupby(['year', 'quarter']).agg({
            'total_amount': 'sum',
            'quantity': 'sum'
        }).round(2)
        
        # Day of week analysis
        dow_analysis = self.data.groupby('day_of_week').agg({
            'total_amount': 'mean',
            'transaction_id': 'count'
        }).round(2)
        
        dow_analysis.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        print("=== SEASONAL ANALYSIS ===")
        print("\nAverage Daily Sales by Day of Week:")
        print(dow_analysis)
        
        # Visualize seasonal patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Monthly sales
        monthly_sales = self.data.groupby('month')['total_amount'].sum()
        axes[0, 0].bar(monthly_sales.index, monthly_sales.values)
        axes[0, 0].set_title('Sales by Month')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Sales ($)')
        
        # Quarterly sales
        quarterly_sales = self.data.groupby('quarter')['total_amount'].sum()
        axes[0, 1].bar(quarterly_sales.index, quarterly_sales.values)
        axes[0, 1].set_title('Sales by Quarter')
        axes[0, 1].set_xlabel('Quarter')
        axes[0, 1].set_ylabel('Sales ($)')
        
        # Day of week sales
        axes[1, 0].bar(dow_analysis.index, dow_analysis['total_amount'])
        axes[1, 0].set_title('Average Sales by Day of Week')
        axes[1, 0].set_ylabel('Average Sales ($)')
        
        # Weekly trend
        weekly_sales = self.data.groupby('week_of_year')['total_amount'].sum()
        axes[1, 1].plot(weekly_sales.index, weekly_sales.values)
        axes[1, 1].set_title('Weekly Sales Trend')
        axes[1, 1].set_xlabel('Week of Year')
        axes[1, 1].set_ylabel('Sales ($)')
        
        plt.tight_layout()
        plt.show()
        
        return monthly_analysis, quarterly_analysis, dow_analysis

# Perform advanced analysis
analyzer = BusinessAnalyzer(sales_enriched)
rfm_analysis = analyzer.customer_segmentation_analysis()
product_analysis = analyzer.product_performance_analysis()
seasonal_data = analyzer.seasonal_analysis()
```

## Phase 4: Predictive Modeling

### Sales Forecasting Model
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class SalesForecaster:
    """Sales forecasting using machine learning"""
    
    def __init__(self, sales_data):
        self.data = sales_data
        self.model = None
        self.features = None
        
    def prepare_features(self):
        """Create features for forecasting"""
        
        # Aggregate daily sales
        daily_sales = self.data.groupby('date').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        daily_sales.columns = ['date', 'daily_revenue', 'daily_quantity', 'daily_transactions']
        
        # Create time-based features
        daily_sales['year'] = daily_sales['date'].dt.year
        daily_sales['month'] = daily_sales['date'].dt.month
        daily_sales['day'] = daily_sales['date'].dt.day
        daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
        daily_sales['day_of_year'] = daily_sales['date'].dt.dayofyear
        daily_sales['week_of_year'] = daily_sales['date'].dt.isocalendar().week
        daily_sales['quarter'] = daily_sales['date'].dt.quarter
        
        # Lag features (previous days' sales)
        for lag in [1, 7, 30]:
            daily_sales[f'revenue_lag_{lag}'] = daily_sales['daily_revenue'].shift(lag)
        
        # Rolling averages
        for window in [7, 14, 30]:
            daily_sales[f'revenue_ma_{window}'] = daily_sales['daily_revenue'].rolling(window).mean()
        
        # Seasonal indicators
        daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
        daily_sales['is_holiday_season'] = (daily_sales['month'].isin([11, 12])).astype(int)
        
        # Remove rows with NaN (due to lag features)
        daily_sales = daily_sales.dropna()
        
        self.daily_sales = daily_sales
        return daily_sales
    
    def train_model(self, target_col='daily_revenue'):
        """Train forecasting model"""
        
        # Prepare features
        feature_cols = [col for col in self.daily_sales.columns 
                       if col not in ['date', target_col]]
        
        X = self.daily_sales[feature_cols]
        y = self.daily_sales[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Time series split
        )
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"\n{name} Results:")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  RMSE: ${np.sqrt(mse):,.2f}")
            print(f"  R²: {r2:.3f}")
        
        # Select best model (lowest MAE)
        best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
        self.model = results[best_model_name]['model']
        self.features = feature_cols
        
        print(f"\nBest model: {best_model_name}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
        
        return results, X_test, y_test
    
    def forecast_future(self, days_ahead=30):
        """Forecast future sales"""
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get last known data point
        last_date = self.daily_sales['date'].max()
        last_row = self.daily_sales.iloc[-1].copy()
        
        forecasts = []
        
        for i in range(1, days_ahead + 1):
            # Create future date
            future_date = last_date + timedelta(days=i)
            
            # Create features for future date
            future_features = last_row[self.features].copy()
            
            # Update time-based features
            future_features['year'] = future_date.year
            future_features['month'] = future_date.month
            future_features['day'] = future_date.day
            future_features['day_of_week'] = future_date.dayofweek
            future_features['day_of_year'] = future_date.timetuple().tm_yday
            future_features['week_of_year'] = future_date.isocalendar()[1]
            future_features['quarter'] = (future_date.month - 1) // 3 + 1
            future_features['is_weekend'] = int(future_date.dayofweek >= 5)
            future_features['is_holiday_season'] = int(future_date.month in [11, 12])
            
            # Make prediction
            prediction = self.model.predict([future_features])[0]
            
            forecasts.append({
                'date': future_date,
                'predicted_revenue': prediction
            })
        
        return pd.DataFrame(forecasts)

# Build forecasting model
forecaster = SalesForecaster(sales_enriched)
daily_data = forecaster.prepare_features()
model_results, X_test, y_test = forecaster.train_model()

# Generate forecasts
future_forecasts = forecaster.forecast_future(30)
print("\nNext 30 Days Revenue Forecast:")
print(future_forecasts.head(10))

# Save model
joblib.dump(forecaster.model, 'sales_forecast_model.pkl')
print("\nModel saved as 'sales_forecast_model.pkl'")
```

## Key Terminology

- **End-to-End Project**: Complete data analysis workflow from data collection to insights
- **Business Intelligence**: Using data analysis to inform business decisions
- **RFM Analysis**: Customer segmentation based on Recency, Frequency, Monetary value
- **Feature Engineering**: Creating predictive variables from raw data
- **Time Series Forecasting**: Predicting future values based on historical patterns
- **Model Validation**: Testing model performance on unseen data
- **Business Metrics**: Key performance indicators relevant to business goals

## Looking Ahead

In Lesson 21, we'll learn about:
- **Portfolio Development**: Creating a professional data science portfolio
- **Project Documentation**: Writing clear, comprehensive project documentation
- **Presentation Skills**: Communicating technical results to non-technical audiences
- **Career Preparation**: Preparing for data analyst roles and interviews
- **Continuous Learning**: Resources and strategies for ongoing skill development
