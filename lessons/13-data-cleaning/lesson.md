# Lesson 11: Data Cleaning - Preparing Real-World Data

## Learning Objectives
By the end of this lesson, you will be able to:
- Identify and handle different types of data quality issues
- Clean and standardize messy text data
- Detect and handle outliers appropriately
- Convert and validate data types
- Create robust data cleaning pipelines
- Document data cleaning decisions

## Why Data Cleaning Matters

### Real-World Data is Messy
```python
# Typical messy dataset
messy_data = pd.DataFrame({
    'Name': ['  Alice Johnson  ', 'bob smith', 'CHARLIE', None, 'diana PRINCE'],
    'Email': ['alice@email.com', 'BOB@EMAIL.COM', 'charlie@email', '', 'diana@email.com'],
    'Age': [25, 30, 'thirty-five', 28, None],
    'Salary': ['$50,000', '60000', '$70,000.50', 'N/A', '80000'],
    'Date': ['2024-01-15', '01/20/2024', '2024/02/10', 'Feb 25, 2024', '']
})
```

### The 80/20 Rule
- 80% of data analysis time is spent cleaning data
- 20% is spent on actual analysis
- Clean data = reliable insights

## Common Data Quality Issues

### 1. Missing Values
```python
import pandas as pd
import numpy as np

# Different representations of missing data
df = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': [1, 2, np.nan, 4, 5],
    'C': [1, 2, 'N/A', 4, 5],
    'D': [1, 2, '', 4, 5],
    'E': [1, 2, 'NULL', 4, 5]
})

# Standardize missing values
df.replace(['N/A', '', 'NULL', 'null', 'None'], np.nan, inplace=True)
```

### 2. Inconsistent Formatting
```python
# Text inconsistencies
names = pd.Series(['  Alice Johnson  ', 'bob smith', 'CHARLIE BROWN'])

# Standardize
names_clean = names.str.strip().str.title()
```

### 3. Data Type Issues
```python
# Mixed types in numeric columns
mixed_numbers = pd.Series(['100', '200.5', '$300', '400.00', 'N/A'])

# Clean and convert
numeric_clean = (mixed_numbers
                .str.replace('$', '')
                .str.replace(',', '')
                .replace('N/A', np.nan)
                .astype(float))
```

## Handling Missing Data

### Detection Strategies
```python
# Comprehensive missing data analysis
def analyze_missing_data(df):
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    return missing_data.sort_values('Missing_Percentage', ascending=False)

# Usage
missing_analysis = analyze_missing_data(df)
print(missing_analysis)
```

### Handling Strategies
```python
# 1. Remove rows/columns with too many missing values
df_clean = df.dropna(thresh=len(df.columns) * 0.7)  # Keep rows with 70% data

# 2. Fill with appropriate values
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill numeric with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical with mode
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# 3. Forward/backward fill for time series
df_time = df.sort_values('date').fillna(method='ffill')
```

## Text Data Cleaning

### String Standardization
```python
def clean_text_column(series):
    """Comprehensive text cleaning function"""
    return (series
            .str.strip()                    # Remove leading/trailing spaces
            .str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
            .str.title()                    # Title case
            .str.replace(r'[^\w\s]', '', regex=True)  # Remove special chars
           )

# Apply to text columns
df['Name_Clean'] = clean_text_column(df['Name'])
```

### Email Validation and Cleaning
```python
import re

def clean_email(email):
    """Clean and validate email addresses"""
    if pd.isna(email) or email == '':
        return np.nan
    
    email = email.lower().strip()
    
    # Basic email pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(pattern, email):
        return email
    else:
        return np.nan

df['Email_Clean'] = df['Email'].apply(clean_email)
```

### Phone Number Standardization
```python
def clean_phone(phone):
    """Standardize phone numbers to (XXX) XXX-XXXX format"""
    if pd.isna(phone):
        return np.nan
    
    # Remove all non-digits
    digits = re.sub(r'\D', '', str(phone))
    
    # Handle different lengths
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    else:
        return np.nan

# Test
phone_numbers = pd.Series(['555-123-4567', '(555) 123-4567', '15551234567', '555.123.4567'])
clean_phones = phone_numbers.apply(clean_phone)
```

## Outlier Detection and Handling

### Statistical Methods
```python
def detect_outliers_iqr(df, column):
    """Detect outliers using Interquartile Range method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    return outliers

# Example usage
sales_data = pd.DataFrame({'sales': [100, 120, 110, 105, 1000, 115, 108]})
outliers, lower, upper = detect_outliers_iqr(sales_data, 'sales')
print(f"Outliers: {outliers}")
print(f"Normal range: {lower:.2f} to {upper:.2f}")
```

### Handling Outliers
```python
def handle_outliers(df, column, method='cap'):
    """Handle outliers using different methods"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if method == 'remove':
        # Remove outliers
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'cap':
        # Cap outliers at bounds
        df_capped = df.copy()
        df_capped[column] = np.where(df_capped[column] < lower_bound, lower_bound, df_capped[column])
        df_capped[column] = np.where(df_capped[column] > upper_bound, upper_bound, df_capped[column])
        return df_capped
    
    elif method == 'transform':
        # Log transformation for right-skewed data
        df_transformed = df.copy()
        df_transformed[column + '_log'] = np.log1p(df_transformed[column])
        return df_transformed
```

## Data Type Conversion and Validation

### Robust Type Conversion
```python
def safe_convert_numeric(series, target_type='float'):
    """Safely convert series to numeric type"""
    # Remove common non-numeric characters
    cleaned = (series.astype(str)
               .str.replace('$', '')
               .str.replace(',', '')
               .str.replace('%', '')
               .str.strip())
    
    # Convert to numeric, coercing errors to NaN
    numeric = pd.to_numeric(cleaned, errors='coerce')
    
    if target_type == 'int':
        # Only convert to int if no NaN values
        if numeric.isna().sum() == 0:
            return numeric.astype(int)
        else:
            print(f"Warning: {numeric.isna().sum()} values couldn't be converted to int")
            return numeric
    
    return numeric

# Example
messy_prices = pd.Series(['$100.50', '200', '$300.75', 'N/A', '400.00'])
clean_prices = safe_convert_numeric(messy_prices)
```

### Date Parsing
```python
def parse_dates_flexible(date_series):
    """Parse dates in multiple formats"""
    parsed_dates = []
    
    for date_str in date_series:
        if pd.isna(date_str) or date_str == '':
            parsed_dates.append(pd.NaT)
            continue
        
        # Try multiple date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%b %d, %Y']
        
        parsed = None
        for fmt in formats:
            try:
                parsed = pd.to_datetime(date_str, format=fmt)
                break
            except:
                continue
        
        if parsed is None:
            try:
                # Let pandas infer the format
                parsed = pd.to_datetime(date_str, infer_datetime_format=True)
            except:
                parsed = pd.NaT
        
        parsed_dates.append(parsed)
    
    return pd.Series(parsed_dates)

# Example
mixed_dates = pd.Series(['2024-01-15', '01/20/2024', '2024/02/10', 'Feb 25, 2024', ''])
clean_dates = parse_dates_flexible(mixed_dates)
```

## Data Validation

### Validation Rules
```python
def validate_data(df):
    """Comprehensive data validation"""
    validation_results = {}
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    validation_results['duplicates'] = duplicates
    
    # Check data ranges
    if 'age' in df.columns:
        invalid_ages = df[(df['age'] < 0) | (df['age'] > 150)].shape[0]
        validation_results['invalid_ages'] = invalid_ages
    
    # Check email format
    if 'email' in df.columns:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = df[~df['email'].str.match(email_pattern, na=False)].shape[0]
        validation_results['invalid_emails'] = invalid_emails
    
    # Check for negative values in positive-only columns
    positive_columns = ['salary', 'price', 'quantity']
    for col in positive_columns:
        if col in df.columns:
            negative_values = df[df[col] < 0].shape[0]
            validation_results[f'negative_{col}'] = negative_values
    
    return validation_results
```

## Complete Data Cleaning Pipeline

### Comprehensive Cleaning Function
```python
def clean_dataset(df, config=None):
    """
    Comprehensive data cleaning pipeline
    
    Parameters:
    df: DataFrame to clean
    config: Dictionary with cleaning configuration
    """
    if config is None:
        config = {
            'missing_threshold': 0.5,  # Drop columns with >50% missing
            'outlier_method': 'cap',   # How to handle outliers
            'standardize_text': True,  # Standardize text columns
            'validate_emails': True,   # Validate email format
            'parse_dates': True        # Parse date columns
        }
    
    df_clean = df.copy()
    cleaning_log = []
    
    # 1. Handle missing values
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_pct[missing_pct > config['missing_threshold']].index
    
    if len(cols_to_drop) > 0:
        df_clean = df_clean.drop(columns=cols_to_drop)
        cleaning_log.append(f"Dropped columns with >{config['missing_threshold']*100}% missing: {list(cols_to_drop)}")
    
    # 2. Standardize text columns
    if config['standardize_text']:
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col not in ['email', 'phone']:  # Skip special columns
                df_clean[col] = clean_text_column(df_clean[col])
                cleaning_log.append(f"Standardized text in column: {col}")
    
    # 3. Clean numeric columns
    numeric_columns = df_clean.select_dtypes(include=['object']).columns
    for col in numeric_columns:
        if any(keyword in col.lower() for keyword in ['price', 'salary', 'amount', 'cost']):
            df_clean[col] = safe_convert_numeric(df_clean[col])
            cleaning_log.append(f"Converted {col} to numeric")
    
    # 4. Handle outliers in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if config['outlier_method'] != 'none':
            df_clean = handle_outliers(df_clean, col, config['outlier_method'])
            cleaning_log.append(f"Handled outliers in {col} using {config['outlier_method']} method")
    
    # 5. Fill remaining missing values
    for col in df_clean.columns:
        if df_clean[col].dtype in ['object']:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col].fillna(mode_val.iloc[0], inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    cleaning_log.append("Filled remaining missing values with mode/median")
    
    return df_clean, cleaning_log

# Usage example
sample_data = pd.DataFrame({
    'name': ['  Alice  ', 'bob smith', 'CHARLIE', None],
    'email': ['alice@test.com', 'BOB@TEST', 'charlie@test.com', ''],
    'age': [25, 30, 200, 28],  # 200 is outlier
    'salary': ['$50,000', '60000', '$70,000', 'N/A']
})

cleaned_data, log = clean_dataset(sample_data)
print("Cleaning Log:")
for entry in log:
    print(f"- {entry}")
```

## Best Practices

### 1. Document Everything
```python
# Keep track of cleaning decisions
cleaning_decisions = {
    'missing_values': 'Filled numeric with median, categorical with mode',
    'outliers': 'Capped at 1.5 * IQR',
    'text_standardization': 'Applied title case and removed special characters',
    'date_parsing': 'Used flexible parsing for multiple formats'
}
```

### 2. Preserve Original Data
```python
# Always work on copies
df_original = df.copy()
df_working = df.copy()

# Keep backup of raw data
df.to_csv('data_backup_raw.csv', index=False)
```

### 3. Validate Results
```python
# Compare before and after
print("Before cleaning:")
print(df_original.info())
print("\nAfter cleaning:")
print(df_clean.info())

# Check data quality improvements
print(f"Missing values reduced from {df_original.isnull().sum().sum()} to {df_clean.isnull().sum().sum()}")
```

## Key Terminology

- **Data Quality**: Measure of how well data serves its intended purpose
- **Missing Values**: Absent or null data points
- **Outliers**: Data points significantly different from others
- **Data Validation**: Process of ensuring data meets quality standards
- **Standardization**: Converting data to consistent format
- **Imputation**: Filling in missing values with estimated values
- **Data Pipeline**: Automated sequence of data processing steps

## Looking Ahead

In Lesson 12, we'll learn about:
- **Data Visualization**: Creating charts and graphs with matplotlib and seaborn
- **Exploratory Data Analysis**: Understanding data through visual exploration
- **Statistical Plots**: Histograms, box plots, scatter plots, and more
- **Customization**: Making professional-looking visualizations
