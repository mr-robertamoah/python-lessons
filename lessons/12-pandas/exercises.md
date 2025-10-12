# Lesson 10 Exercises: Pandas Fundamentals

## Guided Exercises (Do with Instructor)

### Exercise 1: DataFrame Creation and Basic Operations
**Goal**: Create DataFrames and understand basic properties

**Tasks**:
1. Create a DataFrame from a dictionary with student data
2. Display basic information (shape, columns, dtypes, info)
3. View first/last few rows
4. Get basic statistics

```python
import pandas as pd

# Create student data
student_data = {
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eva'],
    'Age': [20, 19, 21, 20, 22],
    'Grade': ['A', 'B', 'A', 'C', 'B'],
    'Score': [95, 87, 92, 78, 89]
}

df = pd.DataFrame(student_data)
# Explore the DataFrame
```

---

### Exercise 2: Data Selection and Indexing
**Goal**: Master different ways to select data

**Tasks**:
1. Select single columns and multiple columns
2. Select rows by index and boolean conditions
3. Use .loc and .iloc for selection
4. Filter data based on conditions

```python
# Using the student DataFrame from Exercise 1
# Select students with Score > 85
# Get names of students with Grade 'A'
```

---

### Exercise 3: Data Manipulation Basics
**Goal**: Modify and transform data

**Tasks**:
1. Add new columns with calculations
2. Modify existing column values
3. Sort data by different columns
4. Group data and calculate aggregates

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Sales Data Analysis
**Goal**: Analyze comprehensive sales dataset

**Create sales data**:
```python
import pandas as pd
import numpy as np

# Generate sample sales data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')
products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones']
regions = ['North', 'South', 'East', 'West']

sales_data = []
for date in dates:
    for _ in range(np.random.randint(5, 15)):  # 5-15 sales per day
        sales_data.append({
            'Date': date,
            'Product': np.random.choice(products),
            'Region': np.random.choice(regions),
            'Quantity': np.random.randint(1, 10),
            'Price': np.random.uniform(20, 1000),
            'Salesperson': f'Sales_{np.random.randint(1, 11)}'
        })

df = pd.DataFrame(sales_data)
```

**Analysis Tasks**:
1. Calculate total revenue by product
2. Find top 5 sales days
3. Analyze regional performance
4. Calculate monthly trends
5. Identify best performing salesperson

---

### Exercise 5: Employee HR Dataset
**Goal**: Process employee data with missing values and data types

**Tasks**:
1. Load employee data with mixed data types
2. Handle missing values appropriately
3. Convert data types (dates, categories)
4. Calculate tenure and age groups
5. Analyze salary distributions by department

**Sample Data Creation**:
```python
# Create employee dataset with realistic issues
employee_data = {
    'EmployeeID': range(1, 101),
    'Name': [f'Employee_{i}' for i in range(1, 101)],
    'Department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], 100),
    'HireDate': pd.date_range('2020-01-01', periods=100, freq='30D'),
    'Salary': np.random.normal(60000, 15000, 100),
    'Age': np.random.randint(22, 65, 100)
}

# Introduce missing values
df = pd.DataFrame(employee_data)
df.loc[np.random.choice(df.index, 10), 'Salary'] = np.nan
df.loc[np.random.choice(df.index, 5), 'Department'] = np.nan
```

---

### Exercise 6: Time Series Analysis
**Goal**: Work with time-based data

**Tasks**:
1. Create daily stock price data
2. Calculate moving averages
3. Find daily returns and volatility
4. Resample data to weekly/monthly
5. Identify trends and patterns

**Stock Data Simulation**:
```python
# Generate realistic stock price data
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
initial_price = 100
prices = [initial_price]

for i in range(1, len(dates)):
    change = np.random.normal(0, 0.02)  # 2% daily volatility
    new_price = prices[-1] * (1 + change)
    prices.append(new_price)

stock_df = pd.DataFrame({
    'Date': dates,
    'Price': prices,
    'Volume': np.random.randint(1000, 10000, len(dates))
})
```

---

### Exercise 7: Customer Transaction Analysis
**Goal**: Analyze customer behavior patterns

**Tasks**:
1. Calculate customer lifetime value
2. Identify customer segments
3. Analyze purchase patterns
4. Find seasonal trends
5. Detect churned customers

---

### Exercise 8: Data Merging and Joining
**Goal**: Combine multiple datasets

**Create related datasets**:
```python
# Customers
customers = pd.DataFrame({
    'CustomerID': range(1, 101),
    'Name': [f'Customer_{i}' for i in range(1, 101)],
    'City': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 100),
    'SignupDate': pd.date_range('2023-01-01', periods=100, freq='3D')
})

# Orders
orders = pd.DataFrame({
    'OrderID': range(1, 501),
    'CustomerID': np.random.choice(range(1, 101), 500),
    'OrderDate': pd.date_range('2023-01-01', periods=500, freq='1D'),
    'Amount': np.random.uniform(10, 500, 500)
})

# Products
products = pd.DataFrame({
    'ProductID': range(1, 21),
    'ProductName': [f'Product_{i}' for i in range(1, 21)],
    'Category': np.random.choice(['Electronics', 'Clothing', 'Books'], 20),
    'Price': np.random.uniform(5, 200, 20)
})
```

**Tasks**:
1. Merge customers with their orders
2. Calculate total spending per customer
3. Find customers with no orders
4. Analyze spending by city
5. Create comprehensive customer profiles

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Advanced Data Cleaning Pipeline
**Goal**: Build robust data cleaning system

**Messy Dataset**:
```python
# Create intentionally messy data
messy_data = {
    'customer_id': ['001', '002', '003', '004', '005', '001', '002'],
    'Customer Name': ['John Doe', 'jane smith', 'ALICE JOHNSON', 'bob WILSON', 'Carol Brown', 'John Doe', 'Jane Smith'],
    'email': ['john@email.com', 'jane@email', 'alice@email.com', '', 'carol@email.com', 'john@email.com', 'jane@email.com'],
    'phone': ['123-456-7890', '(555) 123-4567', '555.123.4567', '123456789', '555-123-4567', '123-456-7890', '555-123-4567'],
    'purchase_date': ['2024-01-15', '2024/02/20', '15-03-2024', '2024-04-01', '2024-05-10', '2024-01-20', '2024-02-25'],
    'amount': ['$100.50', '75.25', '$200', '150.75', '$300.00', '$125.00', '$85.50']
}
```

**Cleaning Tasks**:
1. Standardize name formats
2. Clean and validate email addresses
3. Normalize phone number formats
4. Parse and standardize dates
5. Convert amounts to numeric
6. Handle duplicates intelligently
7. Validate data consistency

---

### Challenge 2: Financial Portfolio Analysis
**Goal**: Comprehensive investment analysis

**Tasks**:
1. Load multiple stock datasets
2. Calculate portfolio returns and risk
3. Perform correlation analysis
4. Implement portfolio optimization
5. Generate performance reports

---

### Challenge 3: E-commerce Analytics Dashboard
**Goal**: Create business intelligence insights

**Multi-table Analysis**:
1. Customer segmentation (RFM analysis)
2. Product performance metrics
3. Seasonal trend analysis
4. Cohort analysis for retention
5. Revenue forecasting

---

## Advanced Pandas Techniques

### Exercise 9: Multi-level Indexing
**Goal**: Work with hierarchical data

**Tasks**:
1. Create MultiIndex DataFrames
2. Perform operations across index levels
3. Reshape data with stack/unstack
4. Aggregate data at different levels

---

### Exercise 10: Custom Functions and Apply
**Goal**: Apply custom logic to data

**Tasks**:
1. Use apply() with custom functions
2. Apply functions to groups
3. Create vectorized operations
4. Handle complex transformations

---

### Exercise 11: Performance Optimization
**Goal**: Optimize pandas operations

**Tasks**:
1. Compare different approaches for speed
2. Use vectorized operations vs loops
3. Optimize memory usage
4. Profile pandas operations

---

## Real-World Applications

### Exercise 12: Web Analytics
**Goal**: Analyze website traffic data

**Tasks**:
1. Process web server logs
2. Calculate page views and sessions
3. Analyze user behavior flows
4. Identify popular content
5. Track conversion funnels

---

### Exercise 13: Survey Data Analysis
**Goal**: Process survey responses

**Tasks**:
1. Handle categorical responses
2. Calculate response rates
3. Perform cross-tabulation analysis
4. Handle Likert scale data
5. Generate summary statistics

---

### Exercise 14: Scientific Data Processing
**Goal**: Analyze experimental data

**Tasks**:
1. Process measurement data
2. Handle experimental conditions
3. Calculate statistical significance
4. Perform quality control checks
5. Generate research reports

---

## Data Import/Export Exercises

### Exercise 15: Multiple File Formats
**Goal**: Work with various data formats

**Tasks**:
1. Read/write CSV with different encodings
2. Process Excel files with multiple sheets
3. Handle JSON data structures
4. Work with database connections
5. Process compressed files

---

### Exercise 16: Large Dataset Handling
**Goal**: Process datasets too large for memory

**Tasks**:
1. Use chunking to process large files
2. Implement streaming data processing
3. Optimize memory usage
4. Use appropriate data types
5. Implement progress tracking

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Create DataFrames from various data sources
- [ ] Select and filter data using multiple methods
- [ ] Handle missing values appropriately
- [ ] Perform data type conversions
- [ ] Group data and calculate aggregates
- [ ] Merge and join multiple datasets
- [ ] Work with time series data
- [ ] Apply custom functions to data
- [ ] Reshape data between wide and long formats
- [ ] Handle categorical data effectively
- [ ] Optimize pandas operations for performance
- [ ] Export data to various formats
- [ ] Debug common pandas issues

## Common Pandas Patterns

### Data Selection
```python
# Column selection
df['column']                    # Single column
df[['col1', 'col2']]           # Multiple columns

# Row selection
df.loc[0]                      # By label
df.iloc[0]                     # By position
df[df['column'] > value]       # Boolean indexing

# Combined selection
df.loc[df['col1'] > value, 'col2']
```

### Data Manipulation
```python
# Adding columns
df['new_col'] = df['col1'] + df['col2']

# Grouping and aggregation
df.groupby('category').agg({'value': ['mean', 'sum']})

# Sorting
df.sort_values('column', ascending=False)

# Handling missing values
df.fillna(method='forward')
df.dropna(subset=['important_col'])
```

### Data Cleaning
```python
# Remove duplicates
df.drop_duplicates(subset=['key_column'])

# String operations
df['text_col'].str.lower().str.strip()

# Data type conversion
df['date_col'] = pd.to_datetime(df['date_col'])
df['category_col'] = df['category_col'].astype('category')
```

## Performance Tips

1. **Use vectorized operations** instead of loops
2. **Choose appropriate data types** (category for strings, int32 vs int64)
3. **Use query() method** for complex filtering
4. **Avoid chained indexing** (use .loc instead)
5. **Use inplace=True** when appropriate to save memory
6. **Consider chunking** for large datasets
7. **Use categorical data** for repeated string values

## Git Reminder

Save your work:
1. Create `lesson-10-pandas` folder in your repository
2. Save exercise solutions as `.py` files
3. Include sample datasets as CSV files
4. Document your analysis approach
5. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 10: Pandas Fundamentals"
   git push
   ```

## Next Lesson Preview

In Lesson 11, we'll learn about:
- **Data Cleaning**: Handling messy real-world data
- **Missing value strategies**: Different approaches for different scenarios
- **Data validation**: Ensuring data quality and consistency
- **Outlier detection**: Identifying and handling anomalous data
