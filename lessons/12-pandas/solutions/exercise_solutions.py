# Lesson 10 Solutions: Pandas Fundamentals

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: DataFrame Creation and Basic Operations
print("Exercise 1: DataFrame Creation and Basic Operations")
print("-" * 50)

student_data = {
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eva'],
    'Age': [20, 19, 21, 20, 22],
    'Grade': ['A', 'B', 'A', 'C', 'B'],
    'Score': [95, 87, 92, 78, 89]
}

df = pd.DataFrame(student_data)

print("DataFrame:")
print(df)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")
print(f"\nDataFrame info:")
df.info()
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"Last 2 rows:\n{df.tail(2)}")
print(f"\nBasic statistics:\n{df.describe()}")

# Exercise 2: Data Selection and Indexing
print("\n" + "="*50)
print("Exercise 2: Data Selection and Indexing")
print("-" * 50)

# Single column
print("Names column:")
print(df['Name'])

# Multiple columns
print(f"\nName and Score columns:")
print(df[['Name', 'Score']])

# Boolean indexing
high_scorers = df[df['Score'] > 85]
print(f"\nStudents with Score > 85:")
print(high_scorers)

# Using loc and iloc
print(f"\nFirst student (using loc):")
print(df.loc[0])

print(f"\nFirst two students, Name and Grade (using loc):")
print(df.loc[0:1, ['Name', 'Grade']])

print(f"\nSecond student (using iloc):")
print(df.iloc[1])

# Grade A students
grade_a_students = df[df['Grade'] == 'A']['Name'].tolist()
print(f"\nStudents with Grade A: {grade_a_students}")

# Exercise 3: Data Manipulation Basics
print("\n" + "="*50)
print("Exercise 3: Data Manipulation Basics")
print("-" * 50)

# Add new column
df['Score_Category'] = df['Score'].apply(lambda x: 'Excellent' if x >= 90 else 'Good' if x >= 80 else 'Average')
print("DataFrame with Score Category:")
print(df)

# Modify existing values
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 21 else 'Older')
print(f"\nDataFrame with Age Group:")
print(df[['Name', 'Age', 'Age_Group']])

# Sort by Score
df_sorted = df.sort_values('Score', ascending=False)
print(f"\nStudents sorted by Score (descending):")
print(df_sorted[['Name', 'Score']])

# Group by Grade
grade_stats = df.groupby('Grade').agg({
    'Score': ['mean', 'count'],
    'Age': 'mean'
}).round(2)
print(f"\nStatistics by Grade:")
print(grade_stats)

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Sales Data Analysis
print("Exercise 4: Sales Data Analysis")
print("-" * 50)

# Generate sample sales data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')
products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones']
regions = ['North', 'South', 'East', 'West']

sales_data = []
for date in dates:
    for _ in range(np.random.randint(5, 15)):
        sales_data.append({
            'Date': date,
            'Product': np.random.choice(products),
            'Region': np.random.choice(regions),
            'Quantity': np.random.randint(1, 10),
            'Price': np.random.uniform(20, 1000),
            'Salesperson': f'Sales_{np.random.randint(1, 11)}'
        })

sales_df = pd.DataFrame(sales_data)
sales_df['Revenue'] = sales_df['Quantity'] * sales_df['Price']

print(f"Sales dataset shape: {sales_df.shape}")
print("Sample data:")
print(sales_df.head())

# 1. Total revenue by product
product_revenue = sales_df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
print(f"\n1. Total revenue by product:")
print(product_revenue.round(2))

# 2. Top 5 sales days
daily_revenue = sales_df.groupby('Date')['Revenue'].sum().sort_values(ascending=False)
print(f"\n2. Top 5 sales days:")
print(daily_revenue.head().round(2))

# 3. Regional performance
regional_performance = sales_df.groupby('Region').agg({
    'Revenue': ['sum', 'mean', 'count'],
    'Quantity': 'sum'
}).round(2)
print(f"\n3. Regional performance:")
print(regional_performance)

# 4. Monthly trends
sales_df['Month'] = sales_df['Date'].dt.to_period('M')
monthly_trends = sales_df.groupby('Month')['Revenue'].sum()
print(f"\n4. Monthly revenue trends:")
print(monthly_trends.round(2))

# 5. Best performing salesperson
salesperson_performance = sales_df.groupby('Salesperson')['Revenue'].sum().sort_values(ascending=False)
print(f"\n5. Top 3 salespeople:")
print(salesperson_performance.head(3).round(2))

# Exercise 5: Employee HR Dataset
print("\n" + "="*50)
print("Exercise 5: Employee HR Dataset")
print("-" * 50)

# Create employee dataset
np.random.seed(42)
employee_data = {
    'EmployeeID': range(1, 101),
    'Name': [f'Employee_{i}' for i in range(1, 101)],
    'Department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], 100),
    'HireDate': pd.date_range('2020-01-01', periods=100, freq='30D'),
    'Salary': np.random.normal(60000, 15000, 100),
    'Age': np.random.randint(22, 65, 100)
}

emp_df = pd.DataFrame(employee_data)

# Introduce missing values
emp_df.loc[np.random.choice(emp_df.index, 10, replace=False), 'Salary'] = np.nan
emp_df.loc[np.random.choice(emp_df.index, 5, replace=False), 'Department'] = np.nan

print("Employee dataset with missing values:")
print(f"Shape: {emp_df.shape}")
print(f"Missing values:\n{emp_df.isnull().sum()}")

# Handle missing values
emp_df['Salary'].fillna(emp_df['Salary'].median(), inplace=True)
emp_df['Department'].fillna('Unknown', inplace=True)

# Convert data types
emp_df['HireDate'] = pd.to_datetime(emp_df['HireDate'])
emp_df['Department'] = emp_df['Department'].astype('category')

# Calculate tenure
emp_df['Tenure_Years'] = (datetime.now() - emp_df['HireDate']).dt.days / 365.25

# Age groups
emp_df['Age_Group'] = pd.cut(emp_df['Age'], bins=[20, 30, 40, 50, 70], labels=['20-30', '31-40', '41-50', '51+'])

print(f"\nCleaned dataset info:")
print(emp_df.dtypes)

# Salary analysis by department
dept_salary = emp_df.groupby('Department')['Salary'].agg(['mean', 'median', 'std']).round(2)
print(f"\nSalary statistics by department:")
print(dept_salary)

# Exercise 6: Time Series Analysis
print("\n" + "="*50)
print("Exercise 6: Time Series Analysis")
print("-" * 50)

# Generate stock price data
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
initial_price = 100
prices = [initial_price]

np.random.seed(42)
for i in range(1, len(dates)):
    change = np.random.normal(0, 0.02)
    new_price = prices[-1] * (1 + change)
    prices.append(new_price)

stock_df = pd.DataFrame({
    'Date': dates,
    'Price': prices,
    'Volume': np.random.randint(1000, 10000, len(dates))
})

stock_df.set_index('Date', inplace=True)

print("Stock data sample:")
print(stock_df.head())

# Calculate returns
stock_df['Daily_Return'] = stock_df['Price'].pct_change()

# Moving averages
stock_df['MA_7'] = stock_df['Price'].rolling(window=7).mean()
stock_df['MA_30'] = stock_df['Price'].rolling(window=30).mean()

# Volatility (30-day rolling)
stock_df['Volatility_30'] = stock_df['Daily_Return'].rolling(window=30).std()

print(f"\nStock analysis:")
print(f"Average price: ${stock_df['Price'].mean():.2f}")
print(f"Price range: ${stock_df['Price'].min():.2f} - ${stock_df['Price'].max():.2f}")
print(f"Average daily return: {stock_df['Daily_Return'].mean():.4f}")
print(f"Daily volatility: {stock_df['Daily_Return'].std():.4f}")

# Monthly resampling
monthly_stats = stock_df.resample('M').agg({
    'Price': ['first', 'last', 'min', 'max', 'mean'],
    'Volume': 'sum',
    'Daily_Return': 'mean'
})

print(f"\nMonthly statistics (first 6 months):")
print(monthly_stats.head(6).round(4))

# Exercise 7: Customer Transaction Analysis
print("\n" + "="*50)
print("Exercise 7: Customer Transaction Analysis")
print("-" * 50)

# Generate customer transaction data
np.random.seed(42)
customers = [f'Customer_{i}' for i in range(1, 101)]
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

transactions = []
for customer in customers:
    # Each customer has 1-20 transactions
    num_transactions = np.random.randint(1, 21)
    customer_start = start_date + timedelta(days=np.random.randint(0, 30))
    
    for i in range(num_transactions):
        transaction_date = customer_start + timedelta(days=np.random.randint(0, 700))
        if transaction_date <= end_date:
            transactions.append({
                'CustomerID': customer,
                'Date': transaction_date,
                'Amount': np.random.uniform(10, 500),
                'Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Food'])
            })

trans_df = pd.DataFrame(transactions)
trans_df['Date'] = pd.to_datetime(trans_df['Date'])

print(f"Transaction dataset: {len(trans_df)} transactions")
print("Sample transactions:")
print(trans_df.head())

# Customer lifetime value
clv = trans_df.groupby('CustomerID').agg({
    'Amount': ['sum', 'mean', 'count'],
    'Date': ['min', 'max']
}).round(2)

clv.columns = ['Total_Spent', 'Avg_Transaction', 'Transaction_Count', 'First_Purchase', 'Last_Purchase']
clv['Days_Active'] = (clv['Last_Purchase'] - clv['First_Purchase']).dt.days

print(f"\nTop 10 customers by total spending:")
print(clv.sort_values('Total_Spent', ascending=False).head(10))

# Customer segments based on RFM
current_date = trans_df['Date'].max()
rfm = trans_df.groupby('CustomerID').agg({
    'Date': lambda x: (current_date - x.max()).days,  # Recency
    'Amount': ['count', 'sum']  # Frequency, Monetary
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Create segments
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print(f"\nCustomer segments (RFM analysis):")
segment_counts = rfm['RFM_Score'].value_counts().head(10)
print(segment_counts)

# Exercise 8: Data Merging and Joining
print("\n" + "="*50)
print("Exercise 8: Data Merging and Joining")
print("-" * 50)

# Create related datasets
customers = pd.DataFrame({
    'CustomerID': range(1, 101),
    'Name': [f'Customer_{i}' for i in range(1, 101)],
    'City': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 100),
    'SignupDate': pd.date_range('2023-01-01', periods=100, freq='3D')
})

orders = pd.DataFrame({
    'OrderID': range(1, 501),
    'CustomerID': np.random.choice(range(1, 101), 500),
    'OrderDate': pd.date_range('2023-01-01', periods=500, freq='1D'),
    'Amount': np.random.uniform(10, 500, 500)
})

print("Customers dataset:")
print(customers.head())
print(f"\nOrders dataset:")
print(orders.head())

# Merge customers with orders
customer_orders = pd.merge(customers, orders, on='CustomerID', how='left')
print(f"\nMerged dataset shape: {customer_orders.shape}")

# Calculate total spending per customer
customer_spending = customer_orders.groupby(['CustomerID', 'Name', 'City']).agg({
    'Amount': ['sum', 'count', 'mean'],
    'OrderDate': ['min', 'max']
}).round(2)

customer_spending.columns = ['Total_Spent', 'Order_Count', 'Avg_Order', 'First_Order', 'Last_Order']

print(f"\nTop 10 customers by spending:")
print(customer_spending.sort_values('Total_Spent', ascending=False).head(10))

# Customers with no orders
customers_no_orders = customers[~customers['CustomerID'].isin(orders['CustomerID'])]
print(f"\nCustomers with no orders: {len(customers_no_orders)}")

# Spending by city
city_spending = customer_orders.groupby('City')['Amount'].agg(['sum', 'mean', 'count']).round(2)
print(f"\nSpending by city:")
print(city_spending)

print("\n=== CHALLENGE EXERCISE SOLUTIONS ===\n")

# Challenge 1: Advanced Data Cleaning Pipeline
print("Challenge 1: Advanced Data Cleaning Pipeline")
print("-" * 50)

# Create messy dataset
messy_data = {
    'customer_id': ['001', '002', '003', '004', '005', '001', '002'],
    'Customer Name': ['John Doe', 'jane smith', 'ALICE JOHNSON', 'bob WILSON', 'Carol Brown', 'John Doe', 'Jane Smith'],
    'email': ['john@email.com', 'jane@email', 'alice@email.com', '', 'carol@email.com', 'john@email.com', 'jane@email.com'],
    'phone': ['123-456-7890', '(555) 123-4567', '555.123.4567', '123456789', '555-123-4567', '123-456-7890', '555-123-4567'],
    'purchase_date': ['2024-01-15', '2024/02/20', '15-03-2024', '2024-04-01', '2024-05-10', '2024-01-20', '2024-02-25'],
    'amount': ['$100.50', '75.25', '$200', '150.75', '$300.00', '$125.00', '$85.50']
}

messy_df = pd.DataFrame(messy_data)
print("Original messy data:")
print(messy_df)

# Cleaning pipeline
def clean_data(df):
    df_clean = df.copy()
    
    # 1. Standardize names
    df_clean['Customer Name'] = df_clean['Customer Name'].str.title().str.strip()
    df_clean['Customer Name'] = df_clean['Customer Name'].replace('', np.nan)
    
    # 2. Clean emails
    df_clean['email_valid'] = df_clean['email'].str.contains('@.*\.', na=False)
    df_clean['email'] = df_clean['email'].where(df_clean['email_valid'], np.nan)
    
    # 3. Normalize phone numbers
    df_clean['phone_clean'] = df_clean['phone'].str.replace(r'[^\d]', '', regex=True)
    df_clean['phone_clean'] = df_clean['phone_clean'].apply(
        lambda x: f"{x[:3]}-{x[3:6]}-{x[6:]}" if len(x) == 10 else np.nan
    )
    
    # 4. Parse dates
    df_clean['purchase_date'] = pd.to_datetime(df_clean['purchase_date'], errors='coerce', infer_datetime_format=True)
    
    # 5. Clean amounts
    df_clean['amount_clean'] = df_clean['amount'].astype(str).str.replace('$', '').str.replace(',', '')
    df_clean['amount_clean'] = pd.to_numeric(df_clean['amount_clean'], errors='coerce')
    
    # 6. Handle duplicates
    df_clean['is_duplicate'] = df_clean.duplicated(subset=['customer_id', 'Customer Name'], keep='first')
    
    return df_clean

cleaned_df = clean_data(messy_df)
print(f"\nCleaned data:")
print(cleaned_df[['customer_id', 'Customer Name', 'email', 'phone_clean', 'purchase_date', 'amount_clean', 'is_duplicate']])

print(f"\nData quality summary:")
print(f"Valid emails: {cleaned_df['email'].notna().sum()}/{len(cleaned_df)}")
print(f"Valid phones: {cleaned_df['phone_clean'].notna().sum()}/{len(cleaned_df)}")
print(f"Valid dates: {cleaned_df['purchase_date'].notna().sum()}/{len(cleaned_df)}")
print(f"Duplicates found: {cleaned_df['is_duplicate'].sum()}")

print("\n" + "="*50)
print("Pandas exercise solutions complete!")
print("Key skills demonstrated:")
print("- DataFrame creation and manipulation")
print("- Data selection and filtering")
print("- Grouping and aggregation")
print("- Time series analysis")
print("- Data cleaning and validation")
print("- Merging and joining datasets")
print("- Advanced analytics and customer segmentation")
