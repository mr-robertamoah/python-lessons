#!/usr/bin/env python3
"""
Pandas Fundamentals Demo
Comprehensive examples of pandas operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def basic_dataframe_operations():
    """Demonstrate basic DataFrame creation and operations"""
    print("=== Basic DataFrame Operations ===")
    
    # Create DataFrame from dictionary
    data = {
        'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eva'],
        'Age': [25, 30, 35, 28, 32],
        'City': ['NYC', 'LA', 'Chicago', 'Houston', 'Miami'],
        'Salary': [70000, 80000, 90000, 75000, 85000]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # Basic statistics
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Selection examples
    print(f"\nNames column: {list(df['Name'])}")
    print(f"First 3 rows:\n{df.head(3)}")
    print(f"High earners (>80k):\n{df[df['Salary'] > 80000]}")

def data_manipulation_demo():
    """Demonstrate data manipulation techniques"""
    print("\n=== Data Manipulation Demo ===")
    
    # Create sample sales data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    sales_df = pd.DataFrame({
        'Date': np.random.choice(dates, 200),
        'Product': np.random.choice(['Laptop', 'Mouse', 'Keyboard'], 200),
        'Quantity': np.random.randint(1, 10, 200),
        'Price': np.random.uniform(20, 1000, 200)
    })
    
    # Add calculated column
    sales_df['Revenue'] = sales_df['Quantity'] * sales_df['Price']
    
    print("Sales data sample:")
    print(sales_df.head())
    
    # Grouping and aggregation
    product_summary = sales_df.groupby('Product').agg({
        'Quantity': 'sum',
        'Revenue': ['sum', 'mean'],
        'Price': 'mean'
    }).round(2)
    
    print(f"\nProduct summary:")
    print(product_summary)
    
    # Sorting
    top_sales = sales_df.nlargest(5, 'Revenue')[['Product', 'Quantity', 'Price', 'Revenue']]
    print(f"\nTop 5 sales by revenue:")
    print(top_sales)

def time_series_demo():
    """Demonstrate time series operations"""
    print("\n=== Time Series Demo ===")
    
    # Create time series data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    
    # Simulate stock price with trend and noise
    trend = np.linspace(100, 120, 365)
    noise = np.random.normal(0, 2, 365)
    prices = trend + noise
    
    stock_df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Volume': np.random.randint(1000, 10000, 365)
    })
    
    stock_df.set_index('Date', inplace=True)
    
    print("Stock data sample:")
    print(stock_df.head())
    
    # Calculate returns
    stock_df['Daily_Return'] = stock_df['Price'].pct_change()
    
    # Moving averages
    stock_df['MA_7'] = stock_df['Price'].rolling(window=7).mean()
    stock_df['MA_30'] = stock_df['Price'].rolling(window=30).mean()
    
    print(f"\nStock statistics:")
    print(f"Average price: ${stock_df['Price'].mean():.2f}")
    print(f"Price volatility: {stock_df['Daily_Return'].std():.4f}")
    print(f"Best day return: {stock_df['Daily_Return'].max():.4f}")
    print(f"Worst day return: {stock_df['Daily_Return'].min():.4f}")
    
    # Monthly resampling
    monthly_stats = stock_df.resample('M').agg({
        'Price': ['mean', 'min', 'max'],
        'Volume': 'sum'
    })
    
    print(f"\nMonthly statistics (first 3 months):")
    print(monthly_stats.head(3))

def data_cleaning_demo():
    """Demonstrate data cleaning techniques"""
    print("\n=== Data Cleaning Demo ===")
    
    # Create messy data
    messy_data = {
        'Name': ['John Doe', 'jane smith', 'ALICE JOHNSON', '', 'Bob Wilson'],
        'Email': ['john@email.com', 'jane@email', 'alice@email.com', 'invalid', ''],
        'Age': [25, 30, np.nan, 35, 28],
        'Salary': ['$50,000', '60000', '$70,000', '80000', np.nan],
        'Date': ['2024-01-15', '2024/02/20', '15-03-2024', '2024-04-01', '']
    }
    
    df = pd.DataFrame(messy_data)
    print("Original messy data:")
    print(df)
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Clean names
    df['Name_Clean'] = df['Name'].str.title().str.strip()
    df['Name_Clean'] = df['Name_Clean'].replace('', np.nan)
    
    # Clean salary
    df['Salary_Clean'] = df['Salary'].astype(str).str.replace('$', '').str.replace(',', '')
    df['Salary_Clean'] = pd.to_numeric(df['Salary_Clean'], errors='coerce')
    
    # Clean dates
    df['Date_Clean'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Salary_Clean'].fillna(df['Salary_Clean'].mean(), inplace=True)
    
    print(f"\nCleaned data:")
    print(df[['Name_Clean', 'Age', 'Salary_Clean', 'Date_Clean']])

def merging_demo():
    """Demonstrate merging and joining operations"""
    print("\n=== Merging and Joining Demo ===")
    
    # Create related datasets
    customers = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eva'],
        'City': ['NYC', 'LA', 'Chicago', 'Houston', 'Miami']
    })
    
    orders = pd.DataFrame({
        'OrderID': [101, 102, 103, 104, 105, 106],
        'CustomerID': [1, 2, 1, 3, 2, 6],  # Note: CustomerID 6 doesn't exist
        'Amount': [100, 200, 150, 300, 250, 400],
        'Date': pd.date_range('2024-01-01', periods=6, freq='D')
    })
    
    print("Customers:")
    print(customers)
    print("\nOrders:")
    print(orders)
    
    # Inner join
    inner_join = pd.merge(customers, orders, on='CustomerID', how='inner')
    print(f"\nInner join (customers with orders):")
    print(inner_join)
    
    # Left join
    left_join = pd.merge(customers, orders, on='CustomerID', how='left')
    print(f"\nLeft join (all customers):")
    print(left_join)
    
    # Customer summary
    customer_summary = inner_join.groupby(['CustomerID', 'Name']).agg({
        'Amount': ['sum', 'count', 'mean']
    }).round(2)
    
    print(f"\nCustomer order summary:")
    print(customer_summary)

def advanced_operations_demo():
    """Demonstrate advanced pandas operations"""
    print("\n=== Advanced Operations Demo ===")
    
    # Create multi-level data
    np.random.seed(42)
    
    # Sales data by region and product
    regions = ['North', 'South', 'East', 'West']
    products = ['A', 'B', 'C']
    months = pd.date_range('2024-01-01', periods=12, freq='M')
    
    data = []
    for region in regions:
        for product in products:
            for month in months:
                data.append({
                    'Region': region,
                    'Product': product,
                    'Month': month,
                    'Sales': np.random.randint(1000, 5000)
                })
    
    sales_df = pd.DataFrame(data)
    
    # Pivot table
    pivot_table = sales_df.pivot_table(
        values='Sales',
        index='Region',
        columns='Product',
        aggfunc='sum'
    )
    
    print("Pivot table (Sales by Region and Product):")
    print(pivot_table)
    
    # Cross-tabulation
    monthly_sales = sales_df.groupby(['Region', sales_df['Month'].dt.month])['Sales'].sum().unstack()
    print(f"\nMonthly sales by region:")
    print(monthly_sales.head())
    
    # Apply custom function
    def categorize_sales(sales):
        if sales > 15000:
            return 'High'
        elif sales > 10000:
            return 'Medium'
        else:
            return 'Low'
    
    region_totals = sales_df.groupby('Region')['Sales'].sum()
    region_categories = region_totals.apply(categorize_sales)
    
    print(f"\nRegion performance categories:")
    print(region_categories)

def performance_comparison():
    """Compare different pandas operations for performance"""
    print("\n=== Performance Comparison ===")
    
    # Create large dataset
    n = 100000
    df = pd.DataFrame({
        'A': np.random.randn(n),
        'B': np.random.randn(n),
        'C': np.random.choice(['X', 'Y', 'Z'], n)
    })
    
    print(f"Dataset size: {len(df):,} rows")
    
    # Compare vectorized vs apply
    import time
    
    # Vectorized operation
    start = time.time()
    result1 = df['A'] + df['B']
    vectorized_time = time.time() - start
    
    # Apply operation
    start = time.time()
    result2 = df.apply(lambda row: row['A'] + row['B'], axis=1)
    apply_time = time.time() - start
    
    print(f"Vectorized operation: {vectorized_time:.4f} seconds")
    print(f"Apply operation: {apply_time:.4f} seconds")
    print(f"Vectorized is {apply_time/vectorized_time:.1f}x faster")
    
    # Memory usage
    print(f"\nMemory usage:")
    print(df.memory_usage(deep=True))

def main():
    """Run all demonstrations"""
    print("Pandas Fundamentals Demonstration")
    print("=" * 50)
    
    basic_dataframe_operations()
    data_manipulation_demo()
    time_series_demo()
    data_cleaning_demo()
    merging_demo()
    advanced_operations_demo()
    performance_comparison()
    
    print("\n" + "=" * 50)
    print("Pandas demonstration complete!")

if __name__ == "__main__":
    main()
