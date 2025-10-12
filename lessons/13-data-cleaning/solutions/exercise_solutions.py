# Lesson 11 Solutions: Data Cleaning and Preprocessing

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Identifying Data Quality Issues
print("Exercise 1: Identifying Data Quality Issues")
print("-" * 50)

messy_data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Name': ['John Doe', 'jane smith', 'ALICE', '', 'Bob Wilson', 'Carol', 'john doe', 'David'],
    'Age': [25, 30, -5, 150, 28, np.nan, 25, 35],
    'Email': ['john@email.com', 'jane@invalid', 'alice@email.com', '', 'bob@email.com', 'carol@email', 'john@email.com', 'david@email.com'],
    'Salary': [50000, 60000, 0, 1000000, 55000, np.nan, 50000, 65000],
    'Date_Joined': ['2023-01-15', '2023/02/20', '15-03-2023', 'invalid', '2023-04-01', '', '2023-01-15', '2023-05-10']
}

df = pd.DataFrame(messy_data)

def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    print("Data Quality Assessment Report")
    print("=" * 40)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Missing values
    print(f"\nMissing Values Analysis:")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    for col in df.columns:
        if missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} missing ({missing_pct[col]:.1f}%)")
    
    # Data type issues
    print(f"\nData Type Issues:")
    for col in df.columns:
        dtype = df[col].dtype
        sample_values = df[col].dropna().head(3).tolist()
        print(f"  {col} ({dtype}): {sample_values}")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Records: {duplicates}")
    
    # Value range issues
    print(f"\nValue Range Issues:")
    
    # Age validation
    if 'Age' in df.columns:
        invalid_ages = df[(df['Age'] < 0) | (df['Age'] > 120)]['Age'].dropna()
        if len(invalid_ages) > 0:
            print(f"  Invalid ages: {invalid_ages.tolist()}")
    
    # Salary validation
    if 'Salary' in df.columns:
        extreme_salaries = df[(df['Salary'] <= 0) | (df['Salary'] > 500000)]['Salary'].dropna()
        if len(extreme_salaries) > 0:
            print(f"  Extreme salaries: {extreme_salaries.tolist()}")
    
    # Email validation
    if 'Email' in df.columns:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = df[~df['Email'].str.match(email_pattern, na=False)]['Email'].dropna()
        if len(invalid_emails) > 0:
            print(f"  Invalid emails: {invalid_emails.tolist()}")
    
    # Inconsistent formatting
    print(f"\nFormatting Issues:")
    if 'Name' in df.columns:
        name_cases = df['Name'].dropna().apply(lambda x: 'mixed' if x != x.upper() and x != x.lower() and x != x.title() else 'consistent')
        inconsistent_names = sum(name_cases == 'mixed')
        print(f"  Names with inconsistent casing: {inconsistent_names}")

assess_data_quality(df)

# Exercise 2: Handling Missing Values
print("\n" + "="*50)
print("Exercise 2: Handling Missing Values")
print("-" * 50)

def handle_missing_values_comprehensive(df):
    """Apply different missing value strategies"""
    df_clean = df.copy()
    
    print("Missing Value Strategies Applied:")
    
    # Strategy 1: Age - use median imputation
    if df_clean['Age'].isnull().sum() > 0:
        age_median = df_clean['Age'].median()
        df_clean['Age'].fillna(age_median, inplace=True)
        print(f"  Age: Filled {df['Age'].isnull().sum()} missing values with median ({age_median})")
    
    # Strategy 2: Salary - use mean imputation for reasonable values
    valid_salaries = df_clean[(df_clean['Salary'] > 0) & (df_clean['Salary'] <= 500000)]['Salary']
    if df_clean['Salary'].isnull().sum() > 0:
        salary_mean = valid_salaries.mean()
        df_clean['Salary'].fillna(salary_mean, inplace=True)
        print(f"  Salary: Filled {df['Salary'].isnull().sum()} missing values with mean (${salary_mean:,.2f})")
    
    # Strategy 3: Name - use forward fill or mark as unknown
    if df_clean['Name'].isnull().sum() > 0 or (df_clean['Name'] == '').sum() > 0:
        df_clean['Name'] = df_clean['Name'].replace('', np.nan)
        df_clean['Name'].fillna('Unknown', inplace=True)
        print(f"  Name: Filled missing/empty values with 'Unknown'")
    
    # Strategy 4: Email - leave as NaN for invalid/missing
    print(f"  Email: Keeping {df_clean['Email'].isnull().sum()} missing values as NaN")
    
    # Strategy 5: Date - leave as NaN for invalid dates
    print(f"  Date_Joined: Keeping invalid dates as NaN for later processing")
    
    return df_clean

df_with_missing_handled = handle_missing_values_comprehensive(df)

# Exercise 3: Data Type Conversion and Validation
print("\n" + "="*50)
print("Exercise 3: Data Type Conversion and Validation")
print("-" * 50)

def convert_and_validate_types(df):
    """Convert data types and validate values"""
    df_clean = df.copy()
    
    print("Data Type Conversions:")
    
    # Convert and validate Age
    def clean_age(age):
        if pd.isna(age):
            return np.nan
        if 18 <= age <= 100:
            return int(age)
        return np.nan
    
    df_clean['Age_Clean'] = df_clean['Age'].apply(clean_age)
    valid_ages = df_clean['Age_Clean'].notna().sum()
    print(f"  Age: Converted to int, {valid_ages} valid values")
    
    # Convert and validate Salary
    def clean_salary(salary):
        if pd.isna(salary):
            return np.nan
        if 1000 <= salary <= 500000:  # Reasonable salary range
            return float(salary)
        return np.nan
    
    df_clean['Salary_Clean'] = df_clean['Salary'].apply(clean_salary)
    valid_salaries = df_clean['Salary_Clean'].notna().sum()
    print(f"  Salary: Converted to float, {valid_salaries} valid values")
    
    # Convert dates
    df_clean['Date_Joined_Clean'] = pd.to_datetime(df_clean['Date_Joined'], errors='coerce')
    valid_dates = df_clean['Date_Joined_Clean'].notna().sum()
    print(f"  Date_Joined: Converted to datetime, {valid_dates} valid values")
    
    # Validate emails
    def validate_email(email):
        if pd.isna(email) or email == '':
            return np.nan
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return email if re.match(pattern, email) else np.nan
    
    df_clean['Email_Clean'] = df_clean['Email'].apply(validate_email)
    valid_emails = df_clean['Email_Clean'].notna().sum()
    print(f"  Email: Validated format, {valid_emails} valid values")
    
    # Standardize names
    def clean_name(name):
        if pd.isna(name) or name == '':
            return np.nan
        return name.strip().title()
    
    df_clean['Name_Clean'] = df_clean['Name'].apply(clean_name)
    print(f"  Name: Standardized to title case")
    
    return df_clean

df_typed = convert_and_validate_types(df_with_missing_handled)
print(f"\nCleaned dataset shape: {df_typed.shape}")

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Customer Database Cleaning
print("Exercise 4: Customer Database Cleaning")
print("-" * 50)

# Generate messy customer data
np.random.seed(42)

customer_data = {
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C001', 'C006', 'C007'],
    'first_name': ['John', 'jane', 'ALICE', '', 'Bob', 'John', 'Carol', 'david'],
    'last_name': ['Doe', 'smith', 'JOHNSON', 'Wilson', '', 'Doe', 'Brown', 'MILLER'],
    'email': ['john.doe@email.com', 'jane@invalid', 'alice@email.com', 'bob.wilson@email', 'bob@email.com', 'john.doe@email.com', 'carol.brown@email.com', 'david@email'],
    'phone': ['123-456-7890', '(555) 123-4567', '555.123.4567', '1234567890', '', '123-456-7890', '555-987-6543', '555 123 4567'],
    'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '', '321 Elm St', '123 Main St', '654 Maple Dr', '987 Cedar Ln'],
    'city': ['New York', 'los angeles', 'CHICAGO', 'Houston', 'Miami', 'New York', 'Boston', 'seattle'],
    'state': ['NY', 'ca', 'IL', 'TX', 'FL', 'NY', 'MA', 'WA'],
    'zip_code': ['10001', '90210', '60601', '77001', '33101', '10001', '02101', '98101'],
    'registration_date': ['2023-01-15', '2023/02/20', '15-03-2023', '2023-04-01', '2023-05-10', '2023-01-15', '2023-06-15', '2023-07-20'],
    'age': [25, 30, 35, 28, 32, 25, 29, 31],
    'income': ['$50,000', '60000', '$70,000', '80000', '', '$50,000', '$90,000', '100000']
}

customer_df = pd.DataFrame(customer_data)

def clean_customer_database(df):
    """Comprehensive customer database cleaning"""
    df_clean = df.copy()
    
    print("Customer Database Cleaning Steps:")
    
    # 1. Standardize names
    df_clean['first_name_clean'] = df_clean['first_name'].str.strip().str.title().replace('', np.nan)
    df_clean['last_name_clean'] = df_clean['last_name'].str.strip().str.title().replace('', np.nan)
    print("  ✓ Standardized name formats")
    
    # 2. Clean emails
    def validate_email(email):
        if pd.isna(email) or email == '':
            return np.nan
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return email if re.match(pattern, email) else np.nan
    
    df_clean['email_clean'] = df_clean['email'].apply(validate_email)
    valid_emails = df_clean['email_clean'].notna().sum()
    print(f"  ✓ Validated emails: {valid_emails}/{len(df_clean)} valid")
    
    # 3. Normalize phone numbers
    def clean_phone(phone):
        if pd.isna(phone) or phone == '':
            return np.nan
        digits = re.sub(r'[^\d]', '', str(phone))
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        return np.nan
    
    df_clean['phone_clean'] = df_clean['phone'].apply(clean_phone)
    valid_phones = df_clean['phone_clean'].notna().sum()
    print(f"  ✓ Normalized phone numbers: {valid_phones}/{len(df_clean)} valid")
    
    # 4. Standardize addresses
    df_clean['city_clean'] = df_clean['city'].str.title()
    df_clean['state_clean'] = df_clean['state'].str.upper()
    print("  ✓ Standardized city and state formats")
    
    # 5. Handle duplicates
    duplicate_mask = df_clean.duplicated(subset=['customer_id'], keep='first')
    duplicates_found = duplicate_mask.sum()
    df_clean = df_clean[~duplicate_mask]
    print(f"  ✓ Removed {duplicates_found} duplicate records")
    
    # 6. Clean income
    def clean_income(income):
        if pd.isna(income) or income == '':
            return np.nan
        cleaned = str(income).replace('$', '').replace(',', '')
        try:
            value = float(cleaned)
            return value if 0 < value <= 1000000 else np.nan
        except ValueError:
            return np.nan
    
    df_clean['income_clean'] = df_clean['income'].apply(clean_income)
    print("  ✓ Cleaned income values")
    
    # 7. Parse dates
    df_clean['registration_date_clean'] = pd.to_datetime(df_clean['registration_date'], errors='coerce')
    print("  ✓ Parsed registration dates")
    
    return df_clean

customer_cleaned = clean_customer_database(customer_df)

print(f"\nCleaning Results:")
print(f"Original records: {len(customer_df)}")
print(f"Cleaned records: {len(customer_cleaned)}")
print(f"Records removed: {len(customer_df) - len(customer_cleaned)}")

# Data quality metrics
def calculate_data_quality_metrics(df_original, df_clean):
    """Calculate data quality improvement metrics"""
    
    metrics = {}
    
    # Completeness
    original_completeness = (1 - df_original.isnull().sum() / len(df_original)) * 100
    
    # For cleaned data, use cleaned columns where available
    clean_columns = [col for col in df_clean.columns if col.endswith('_clean')]
    
    print(f"\nData Quality Metrics:")
    print(f"{'Column':<20} {'Original %':<12} {'Cleaned %':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for col in df_original.columns:
        clean_col = f"{col}_clean"
        if clean_col in df_clean.columns:
            orig_complete = original_completeness[col]
            clean_complete = (1 - df_clean[clean_col].isnull().sum() / len(df_clean)) * 100
            improvement = clean_complete - orig_complete
            print(f"{col:<20} {orig_complete:>8.1f}%    {clean_complete:>8.1f}%    {improvement:>+8.1f}%")

calculate_data_quality_metrics(customer_df, customer_cleaned)

# Exercise 5: Sales Transaction Cleaning
print("\n" + "="*50)
print("Exercise 5: Sales Transaction Cleaning")
print("-" * 50)

# Create messy sales data
np.random.seed(42)
sales_data = {
    'transaction_id': [f'T{i:03d}' for i in range(1, 21)],
    'product_code': ['P001', 'P002', 'INVALID', 'P003', '', 'P001', 'P004', 'P002', 'P999', 'P003'] * 2,
    'quantity': [5, -2, 10, 0, 3, 8, 15, -1, 7, 12] * 2,
    'price': ['$25.99', '-10.50', '150.00', '$0', '75.25', '$30.00', '200.50', '$-5.00', '99.99', '$125.00'] * 2,
    'customer_id': ['C001', 'C002', '', 'C003', 'C004', 'C001', 'C005', 'C002', 'INVALID', 'C003'] * 2,
    'transaction_date': ['2024-01-15', '2024-02-20', '2024-03-25', '2024-04-30', '2024-05-15', 
                        '2024-06-20', '2024-07-25', '2024-08-30', '2024-09-15', '2024-10-20'] * 2
}

sales_df = pd.DataFrame(sales_data)

def clean_sales_transactions(df):
    """Clean sales transaction data"""
    df_clean = df.copy()
    
    print("Sales Transaction Cleaning:")
    
    # Valid product codes (master list)
    valid_products = ['P001', 'P002', 'P003', 'P004', 'P005']
    
    # 1. Validate product codes
    df_clean['product_code_valid'] = df_clean['product_code'].isin(valid_products)
    invalid_products = (~df_clean['product_code_valid']).sum()
    print(f"  ✓ Found {invalid_products} invalid product codes")
    
    # 2. Clean quantities (must be positive)
    df_clean['quantity_clean'] = df_clean['quantity'].where(df_clean['quantity'] > 0, np.nan)
    invalid_quantities = (df_clean['quantity'] <= 0).sum()
    print(f"  ✓ Found {invalid_quantities} invalid quantities")
    
    # 3. Clean prices
    def clean_price(price):
        if pd.isna(price):
            return np.nan
        cleaned = str(price).replace('$', '').replace(',', '')
        try:
            value = float(cleaned)
            return value if value > 0 else np.nan
        except ValueError:
            return np.nan
    
    df_clean['price_clean'] = df_clean['price'].apply(clean_price)
    invalid_prices = df_clean['price_clean'].isnull().sum()
    print(f"  ✓ Cleaned prices, {invalid_prices} invalid values")
    
    # 4. Validate customer IDs
    valid_customers = [f'C{i:03d}' for i in range(1, 101)]  # C001 to C100
    df_clean['customer_id_valid'] = df_clean['customer_id'].isin(valid_customers)
    invalid_customers = (~df_clean['customer_id_valid']).sum()
    print(f"  ✓ Found {invalid_customers} invalid customer IDs")
    
    # 5. Calculate revenue
    df_clean['revenue'] = df_clean['quantity_clean'] * df_clean['price_clean']
    
    # 6. Detect outliers in revenue
    Q1 = df_clean['revenue'].quantile(0.25)
    Q3 = df_clean['revenue'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    
    outliers = df_clean[df_clean['revenue'] > outlier_threshold]
    print(f"  ✓ Detected {len(outliers)} revenue outliers (>${outlier_threshold:.2f})")
    
    return df_clean

sales_cleaned = clean_sales_transactions(sales_df)

print(f"\nSales Data Summary:")
print(f"Total transactions: {len(sales_cleaned)}")
print(f"Valid transactions: {sales_cleaned['revenue'].notna().sum()}")
print(f"Total revenue: ${sales_cleaned['revenue'].sum():,.2f}")

print("\n=== CHALLENGE EXERCISE SOLUTIONS ===\n")

# Challenge 1: Multi-Source Data Integration
print("Challenge 1: Multi-Source Data Integration")
print("-" * 50)

def integrate_multi_source_data():
    """Integrate customer data from multiple sources"""
    
    # Source 1: CRM System
    crm_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004'],
        'name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Wilson'],
        'email': ['john@email.com', 'jane@email.com', 'alice@email.com', 'bob@email.com'],
        'phone': ['123-456-7890', '555-123-4567', '555-987-6543', '555-111-2222'],
        'source': 'CRM'
    })
    
    # Source 2: Web Analytics
    web_data = pd.DataFrame({
        'user_id': ['U001', 'U002', 'U003', 'U005'],
        'email': ['john@email.com', 'jane@email.com', 'alice@email.com', 'carol@email.com'],
        'last_visit': ['2024-01-15', '2024-01-20', '2024-01-25', '2024-01-30'],
        'page_views': [25, 15, 30, 10],
        'source': 'Web'
    })
    
    # Source 3: Transaction System
    transaction_data = pd.DataFrame({
        'customer_ref': ['C001', 'C002', 'C004', 'C006'],
        'total_spent': [1500.00, 2300.00, 800.00, 1200.00],
        'last_purchase': ['2024-01-10', '2024-01-18', '2024-01-22', '2024-01-28'],
        'source': 'Transactions'
    })
    
    print("Integrating data from multiple sources:")
    print(f"  CRM records: {len(crm_data)}")
    print(f"  Web records: {len(web_data)}")
    print(f"  Transaction records: {len(transaction_data)}")
    
    # Create master customer record
    # Start with CRM as primary source
    master_df = crm_data.copy()
    
    # Add web analytics data
    web_matched = web_data.merge(master_df[['email']], on='email', how='inner')
    master_df = master_df.merge(web_matched[['email', 'last_visit', 'page_views']], 
                               on='email', how='left')
    
    # Add transaction data
    master_df = master_df.merge(transaction_data[['customer_ref', 'total_spent', 'last_purchase']], 
                               left_on='customer_id', right_on='customer_ref', how='left')
    
    # Handle conflicts and missing data
    master_df['data_completeness'] = master_df.notna().sum(axis=1) / len(master_df.columns)
    
    print(f"\nMaster customer records created: {len(master_df)}")
    print(f"Average data completeness: {master_df['data_completeness'].mean():.2%}")
    
    return master_df

integrated_data = integrate_multi_source_data()

print("\n" + "="*50)
print("Data cleaning exercise solutions complete!")
print("\nKey techniques demonstrated:")
print("- Systematic data quality assessment")
print("- Multiple missing value strategies")
print("- Data type validation and conversion")
print("- Text standardization and cleaning")
print("- Duplicate detection and handling")
print("- Outlier detection using statistical methods")
print("- Multi-source data integration")
print("- Data quality metrics and reporting")
