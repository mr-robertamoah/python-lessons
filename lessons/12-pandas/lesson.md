# Lesson 10: Pandas Basics - Data Manipulation and Analysis

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand pandas Series and DataFrame structures
- Read data from various file formats (CSV, Excel, JSON)
- Select, filter, and manipulate data efficiently
- Perform basic data aggregation and grouping
- Handle missing data appropriately
- Export processed data to different formats

## Why Pandas is Essential for Data Analysis

### The Problem with Raw Data
Real-world data comes in messy formats:
- CSV files with mixed data types
- Missing values scattered throughout
- Date columns as strings
- Inconsistent formatting
- Need for filtering, grouping, and aggregation

### Pandas Solution
```python
import pandas as pd

# Read messy CSV data
df = pd.read_csv('sales_data.csv')

# Clean and analyze in a few lines
df['date'] = pd.to_datetime(df['date'])
monthly_sales = df.groupby(df['date'].dt.month)['sales'].sum()
top_products = df.groupby('product')['sales'].sum().sort_values(ascending=False)
```

### Key Benefits
- **Intuitive**: Works like Excel but programmable
- **Powerful**: Built on NumPy for performance
- **Flexible**: Handles various data formats and types
- **Complete**: Data cleaning, analysis, and export in one library

## Installing and Importing Pandas

### Installation
```bash
# In your virtual environment
pip install pandas

# Often installed with other data science packages
pip install pandas numpy matplotlib
```

### Import Convention
```python
import pandas as pd
import numpy as np  # Often used together

# Check version
print(pd.__version__)
```

## Pandas Data Structures

### Series - 1D Labeled Array
```python
import pandas as pd

# Create Series from list
temperatures = pd.Series([72, 75, 68, 80, 77])
print(temperatures)
# 0    72
# 1    75
# 2    68
# 3    80
# 4    77
# dtype: int64

# Series with custom index
cities = pd.Series([72, 75, 68, 80, 77], 
                  index=['NYC', 'LA', 'Chicago', 'Miami', 'Seattle'])
print(cities)
# NYC        72
# LA         75
# Chicago    68
# Miami      80
# Seattle    77
# dtype: int64

# Access values
print(cities['NYC'])        # 72
print(cities[['NYC', 'LA']]) # Multiple values
```

### DataFrame - 2D Labeled Data Structure
```python
# Create DataFrame from dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['NYC', 'LA', 'Chicago', 'Miami'],
    'Salary': [70000, 80000, 90000, 75000]
}

df = pd.DataFrame(data)
print(df)
#      Name  Age     City  Salary
# 0   Alice   25      NYC   70000
# 1     Bob   30       LA   80000
# 2 Charlie   35  Chicago   90000
# 3   Diana   28    Miami   75000

# DataFrame properties
print(f"Shape: {df.shape}")        # (4, 4)
print(f"Columns: {df.columns}")    # Index(['Name', 'Age', 'City', 'Salary'])
print(f"Index: {df.index}")        # RangeIndex(start=0, stop=4, step=1)
print(f"Data types:\n{df.dtypes}") # Data type of each column
```

## Reading Data from Files

### CSV Files
```python
# Basic CSV reading
df = pd.read_csv('data.csv')

# Common parameters
df = pd.read_csv('data.csv',
                 sep=',',           # Separator (default: comma)
                 header=0,          # Row to use as column names
                 index_col=0,       # Column to use as row index
                 na_values=['N/A', 'NULL'],  # Additional NA values
                 parse_dates=['date_column'], # Parse dates
                 encoding='utf-8')   # File encoding
```

### Excel Files
```python
# Read Excel file
df = pd.read_excel('data.xlsx')

# Specific sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Multiple sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)
```

### JSON Files
```python
# Read JSON
df = pd.read_json('data.json')

# Different JSON orientations
df = pd.read_json('data.json', orient='records')
```

### Other Formats
```python
# SQL databases
df = pd.read_sql('SELECT * FROM table', connection)

# HTML tables
df = pd.read_html('webpage.html')[0]  # First table

# Clipboard (copy from Excel/web)
df = pd.read_clipboard()
```

## Data Inspection and Information

### Basic Information
```python
# Sample dataset
sales_data = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    'product': ['A', 'B', 'A', 'C'],
    'sales': [100, 150, 120, 200],
    'region': ['North', 'South', 'North', 'East']
})

# First/last few rows
print(df.head())      # First 5 rows (default)
print(df.head(3))     # First 3 rows
print(df.tail())      # Last 5 rows

# Basic info
print(df.info())      # Data types, non-null counts, memory usage
print(df.describe())  # Statistical summary for numeric columns
print(df.shape)       # (rows, columns)
print(len(df))        # Number of rows
```

### Column and Index Operations
```python
# Column information
print(df.columns)           # Column names
print(df.columns.tolist())  # As Python list

# Rename columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# Set new column names
df.columns = ['col1', 'col2', 'col3', 'col4']

# Reset index
df.reset_index(drop=True, inplace=True)

# Set column as index
df.set_index('date', inplace=True)
```

## Data Selection and Filtering

### Column Selection
```python
# Single column (returns Series)
names = df['Name']

# Multiple columns (returns DataFrame)
subset = df[['Name', 'Age']]

# All columns except some
df_subset = df.drop(['Salary'], axis=1)
```

### Row Selection
```python
# By position (iloc)
first_row = df.iloc[0]           # First row
first_three = df.iloc[0:3]       # First 3 rows
last_row = df.iloc[-1]           # Last row

# By label (loc)
df_indexed = df.set_index('Name')
alice_data = df_indexed.loc['Alice']

# Boolean indexing
high_earners = df[df['Salary'] > 75000]
young_high_earners = df[(df['Age'] < 30) & (df['Salary'] > 70000)]
```

### Advanced Selection
```python
# Query method (more readable for complex conditions)
result = df.query('Age > 25 and Salary < 85000')

# isin method
cities_of_interest = df[df['City'].isin(['NYC', 'LA'])]

# String operations
ny_people = df[df['City'].str.contains('NY')]

# Sample random rows
sample = df.sample(n=2)  # 2 random rows
sample_pct = df.sample(frac=0.5)  # 50% of rows
```

## Data Manipulation

### Adding and Modifying Columns
```python
# Add new column
df['Bonus'] = df['Salary'] * 0.1

# Conditional column
df['Seniority'] = df['Age'].apply(lambda x: 'Senior' if x > 30 else 'Junior')

# Using np.where for conditions
df['Tax_Bracket'] = np.where(df['Salary'] > 80000, 'High', 'Standard')

# Multiple conditions with np.select
conditions = [
    df['Age'] < 25,
    df['Age'] < 35,
    df['Age'] >= 35
]
choices = ['Young', 'Mid-career', 'Experienced']
df['Career_Stage'] = np.select(conditions, choices)
```

### Sorting Data
```python
# Sort by single column
df_sorted = df.sort_values('Age')

# Sort by multiple columns
df_sorted = df.sort_values(['City', 'Age'], ascending=[True, False])

# Sort by index
df_sorted = df.sort_index()
```

### Removing Data
```python
# Drop rows
df_clean = df.drop([0, 2])  # Drop rows by index

# Drop columns
df_clean = df.drop(['Bonus'], axis=1)

# Drop duplicates
df_unique = df.drop_duplicates()
df_unique = df.drop_duplicates(subset=['Name'])  # Based on specific columns
```

## Handling Missing Data

### Detecting Missing Data
```python
# Create data with missing values
data_with_na = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, np.nan]
})

# Check for missing values
print(data_with_na.isnull())     # Boolean DataFrame
print(data_with_na.isnull().sum())  # Count of NAs per column
print(data_with_na.info())       # Shows non-null counts
```

### Handling Missing Data
```python
# Drop rows with any NA
df_dropped = data_with_na.dropna()

# Drop rows with all NA
df_dropped = data_with_na.dropna(how='all')

# Drop columns with any NA
df_dropped = data_with_na.dropna(axis=1)

# Fill missing values
df_filled = data_with_na.fillna(0)  # Fill with 0
df_filled = data_with_na.fillna(method='ffill')  # Forward fill
df_filled = data_with_na.fillna(data_with_na.mean())  # Fill with mean

# Fill different columns with different values
fill_values = {'A': 0, 'B': data_with_na['B'].mean(), 'C': 'Unknown'}
df_filled = data_with_na.fillna(fill_values)
```

## Data Aggregation and Grouping

### Basic Aggregation
```python
# Summary statistics
print(df['Salary'].mean())    # Average salary
print(df['Salary'].median())  # Median salary
print(df['Salary'].std())     # Standard deviation
print(df['Age'].min())        # Minimum age
print(df['Age'].max())        # Maximum age

# Multiple statistics at once
print(df['Salary'].describe())
```

### GroupBy Operations
```python
# Sample sales data
sales_df = pd.DataFrame({
    'Region': ['North', 'South', 'North', 'East', 'South', 'East'],
    'Product': ['A', 'B', 'A', 'C', 'B', 'A'],
    'Sales': [100, 150, 120, 200, 180, 90],
    'Quantity': [10, 15, 12, 20, 18, 9]
})

# Group by single column
region_sales = sales_df.groupby('Region')['Sales'].sum()
print(region_sales)

# Group by multiple columns
product_region = sales_df.groupby(['Region', 'Product'])['Sales'].sum()
print(product_region)

# Multiple aggregations
agg_result = sales_df.groupby('Region').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Quantity': ['sum', 'mean']
})
print(agg_result)
```

### Pivot Tables
```python
# Create pivot table
pivot = sales_df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    fill_value=0
)
print(pivot)
```

## Data Transformation

### Apply Functions
```python
# Apply function to column
df['Salary_K'] = df['Salary'].apply(lambda x: x / 1000)

# Apply function to multiple columns
def calculate_tax(row):
    if row['Salary'] > 80000:
        return row['Salary'] * 0.25
    else:
        return row['Salary'] * 0.20

df['Tax'] = df.apply(calculate_tax, axis=1)

# Apply to entire DataFrame
df_normalized = df[['Age', 'Salary']].apply(lambda x: (x - x.mean()) / x.std())
```

### String Operations
```python
# String methods (for text columns)
df['Name_Upper'] = df['Name'].str.upper()
df['Name_Length'] = df['Name'].str.len()
df['First_Letter'] = df['Name'].str[0]

# String splitting
df[['First_Name', 'Last_Name']] = df['Full_Name'].str.split(' ', expand=True)

# String replacement
df['City_Clean'] = df['City'].str.replace('NYC', 'New York City')
```

### Date and Time Operations
```python
# Convert to datetime
df['Date'] = pd.to_datetime(df['Date_String'])

# Extract date components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.day_name()

# Date arithmetic
df['Days_Since'] = (pd.Timestamp.now() - df['Date']).dt.days
```

## Merging and Joining Data

### Concatenation
```python
# Concatenate DataFrames vertically
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
combined = pd.concat([df1, df2], ignore_index=True)

# Concatenate horizontally
combined = pd.concat([df1, df2], axis=1)
```

### Merging
```python
# Sample data for merging
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 30],
    'dept_name': ['Engineering', 'Sales', 'Marketing']
})

# Inner join (default)
merged = pd.merge(employees, departments, on='dept_id')

# Left join
merged = pd.merge(employees, departments, on='dept_id', how='left')

# Outer join
merged = pd.merge(employees, departments, on='dept_id', how='outer')
```

## Exporting Data

### Save to Files
```python
# Save to CSV
df.to_csv('output.csv', index=False)

# Save to Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Save to JSON
df.to_json('output.json', orient='records')

# Save multiple sheets to Excel
with pd.ExcelWriter('multi_sheet.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

## Practical Examples

### Example 1: Sales Analysis
```python
# Create sample sales data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
products = ['Product A', 'Product B', 'Product C', 'Product D']
regions = ['North', 'South', 'East', 'West']

sales_data = pd.DataFrame({
    'date': np.random.choice(dates, 500),
    'product': np.random.choice(products, 500),
    'region': np.random.choice(regions, 500),
    'sales_amount': np.random.normal(1000, 200, 500),
    'quantity': np.random.randint(1, 20, 500)
})

print("Sales Analysis:")
print(f"Total sales: ${sales_data['sales_amount'].sum():,.2f}")
print(f"Average order: ${sales_data['sales_amount'].mean():.2f}")

# Top products by sales
top_products = sales_data.groupby('product')['sales_amount'].sum().sort_values(ascending=False)
print(f"\nTop products:\n{top_products}")

# Regional performance
regional_sales = sales_data.groupby('region').agg({
    'sales_amount': ['sum', 'mean', 'count'],
    'quantity': 'sum'
}).round(2)
print(f"\nRegional performance:\n{regional_sales}")

# Monthly trends
sales_data['month'] = sales_data['date'].dt.to_period('M')
monthly_sales = sales_data.groupby('month')['sales_amount'].sum()
print(f"\nMonthly sales:\n{monthly_sales}")
```

### Example 2: Customer Data Cleaning
```python
# Messy customer data
messy_data = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['  Alice Johnson  ', 'bob smith', 'CHARLIE BROWN', None, 'diana PRINCE'],
    'email': ['alice@email.com', 'BOB@EMAIL.COM', 'charlie@email', 'diana@email.com', ''],
    'age': [25, 30, 'thirty-five', 28, None],
    'purchase_amount': ['$1,200.50', '$800', '1500.75', '$2,000.00', '$500.25']
})

print("Original messy data:")
print(messy_data)

# Clean the data
cleaned_data = messy_data.copy()

# Clean names
cleaned_data['name'] = cleaned_data['name'].str.strip().str.title()

# Clean emails
cleaned_data['email'] = cleaned_data['email'].str.lower()
cleaned_data['email'] = cleaned_data['email'].replace('', np.nan)

# Clean age (convert to numeric, handle non-numeric)
cleaned_data['age'] = pd.to_numeric(cleaned_data['age'], errors='coerce')

# Clean purchase amounts (remove $ and commas, convert to float)
cleaned_data['purchase_amount'] = (cleaned_data['purchase_amount']
                                  .str.replace('$', '')
                                  .str.replace(',', '')
                                  .astype(float))

# Handle missing values
cleaned_data['name'].fillna('Unknown', inplace=True)
cleaned_data['age'].fillna(cleaned_data['age'].median(), inplace=True)

print("\nCleaned data:")
print(cleaned_data)
print(f"\nData types:\n{cleaned_data.dtypes}")
```

### Example 3: Time Series Analysis
```python
# Create time series data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
np.random.seed(42)

# Simulate daily website visits with trend and seasonality
trend = np.linspace(1000, 1500, 365)
seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly pattern
noise = np.random.normal(0, 50, 365)
visits = trend + seasonal + noise

web_data = pd.DataFrame({
    'date': dates,
    'visits': visits.astype(int)
})

web_data.set_index('date', inplace=True)

print("Time Series Analysis:")
print(f"Total visits: {web_data['visits'].sum():,}")
print(f"Average daily visits: {web_data['visits'].mean():.0f}")

# Resample to different frequencies
weekly_visits = web_data.resample('W')['visits'].sum()
monthly_visits = web_data.resample('M')['visits'].sum()

print(f"\nWeekly visits (first 5 weeks):\n{weekly_visits.head()}")
print(f"\nMonthly visits:\n{monthly_visits}")

# Rolling statistics
web_data['7_day_avg'] = web_data['visits'].rolling(window=7).mean()
web_data['30_day_avg'] = web_data['visits'].rolling(window=30).mean()

print(f"\nRecent data with moving averages:")
print(web_data.tail())
```

## Performance Tips

### Efficient Operations
```python
# Use vectorized operations instead of loops
# Slow:
result = []
for value in df['column']:
    result.append(value * 2)

# Fast:
result = df['column'] * 2

# Use .loc and .iloc for selection
# Slow:
df[df['column'] > 5]['other_column']

# Fast:
df.loc[df['column'] > 5, 'other_column']
```

### Memory Optimization
```python
# Check memory usage
print(df.memory_usage(deep=True))

# Optimize data types
df['category_col'] = df['category_col'].astype('category')
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')
```

## Key Terminology

- **DataFrame**: 2D labeled data structure with columns of potentially different types
- **Series**: 1D labeled array capable of holding any data type
- **Index**: Labels for rows in a DataFrame or Series
- **GroupBy**: Split-apply-combine operations for data aggregation
- **Pivot Table**: Reshape data based on column values
- **Merge/Join**: Combine DataFrames based on common columns or indices
- **Vectorization**: Operations applied to entire arrays/columns at once
- **Broadcasting**: Automatic alignment of operations between different-sized objects

## Looking Ahead

In Lesson 11, we'll learn about:
- **Data Cleaning**: Advanced techniques for messy real-world data
- **Data Types**: Converting and validating data types
- **Outlier Detection**: Finding and handling unusual values
- **Data Validation**: Ensuring data quality and consistency
- **Text Processing**: Advanced string manipulation for data cleaning
