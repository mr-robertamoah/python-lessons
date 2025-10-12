#!/usr/bin/env python3
"""
Data Cleaning and Preprocessing Demo
Comprehensive examples of data cleaning techniques
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

def create_messy_dataset():
    """Create a realistic messy dataset for demonstration"""
    np.random.seed(42)
    
    # Create intentionally messy data
    messy_data = {
        'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 11],  # Duplicate ID
        'Name': ['John Doe', 'jane smith', 'ALICE JOHNSON', '', 'Bob Wilson', 
                'Carol Brown', 'john doe', 'David Miller', 'Eva Davis', 'Frank Wilson', 'John Doe', 'Grace Lee'],
        'Age': [25, 30, -5, 150, 28, np.nan, 25, 35, 29, 45, 25, 27],  # Invalid ages
        'Email': ['john@email.com', 'jane@invalid', 'alice@email.com', '', 'bob@email.com', 
                 'carol@email', 'john@email.com', 'david@email.com', 'eva@email.com', 'frank@email.com', 'john@email.com', 'grace@email.com'],
        'Salary': ['$50,000', '60000', '$0', '1000000', '55000', np.nan, '$50,000', '65000', '70000', '80000', '$50,000', '45000'],
        'Date_Joined': ['2023-01-15', '2023/02/20', '15-03-2023', 'invalid', '2023-04-01', '', 
                       '2023-01-15', '2023-05-10', '2023-06-15', '2023-07-20', '2023-01-15', '2023-08-01'],
        'Department': ['IT', 'sales', 'MARKETING', 'HR', 'it', 'Finance', 'IT', 'Sales', 'Marketing', 'HR', 'IT', 'Finance'],
        'Phone': ['123-456-7890', '(555) 123-4567', '555.123.4567', '1234567890', '', 
                 '123-456-7890', '555-987-6543', '555 123 4567', '(555)987-6543', '555.456.7890', '123-456-7890', '555-111-2222']
    }
    
    return pd.DataFrame(messy_data)

def data_quality_assessment(df):
    """Perform comprehensive data quality assessment"""
    print("=== Data Quality Assessment ===")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Missing values analysis
    print(f"\nMissing Values:")
    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / len(df)) * 100
    
    for col in df.columns:
        if missing_stats[col] > 0:
            print(f"  {col}: {missing_stats[col]} ({missing_pct[col]:.1f}%)")
    
    # Data types
    print(f"\nData Types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Unique values per column
    print(f"\nUnique values per column:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count}")
    
    return {
        'missing_values': missing_stats,
        'duplicates': duplicates,
        'data_types': df.dtypes
    }

def clean_names(df):
    """Clean and standardize name fields"""
    print("\n=== Cleaning Names ===")
    
    df_clean = df.copy()
    
    # Original names
    print("Original names:")
    print(df_clean['Name'].tolist())
    
    # Clean names
    df_clean['Name_Clean'] = (df_clean['Name']
                             .str.strip()  # Remove leading/trailing spaces
                             .str.title()  # Convert to title case
                             .replace('', np.nan))  # Replace empty strings with NaN
    
    print("\nCleaned names:")
    print(df_clean['Name_Clean'].tolist())
    
    return df_clean

def validate_and_clean_emails(df):
    """Validate and clean email addresses"""
    print("\n=== Cleaning Email Addresses ===")
    
    df_clean = df.copy()
    
    def is_valid_email(email):
        if pd.isna(email) or email == '':
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    # Check email validity
    df_clean['Email_Valid'] = df_clean['Email'].apply(is_valid_email)
    df_clean['Email_Clean'] = df_clean['Email'].where(df_clean['Email_Valid'], np.nan)
    
    print("Email validation results:")
    for i, (original, valid, clean) in enumerate(zip(df_clean['Email'], df_clean['Email_Valid'], df_clean['Email_Clean'])):
        status = "✓" if valid else "✗"
        print(f"  {status} {original} -> {clean}")
    
    return df_clean

def clean_phone_numbers(df):
    """Standardize phone number formats"""
    print("\n=== Cleaning Phone Numbers ===")
    
    df_clean = df.copy()
    
    def clean_phone(phone):
        if pd.isna(phone) or phone == '':
            return np.nan
        
        # Remove all non-digit characters
        digits = re.sub(r'[^\d]', '', str(phone))
        
        # Check if we have 10 digits (US format)
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        else:
            return np.nan
    
    df_clean['Phone_Clean'] = df_clean['Phone'].apply(clean_phone)
    
    print("Phone number cleaning results:")
    for original, clean in zip(df_clean['Phone'], df_clean['Phone_Clean']):
        print(f"  {original} -> {clean}")
    
    return df_clean

def handle_numeric_data(df):
    """Clean and validate numeric fields"""
    print("\n=== Cleaning Numeric Data ===")
    
    df_clean = df.copy()
    
    # Clean salary field
    def clean_salary(salary):
        if pd.isna(salary):
            return np.nan
        
        # Remove currency symbols and commas
        cleaned = str(salary).replace('$', '').replace(',', '')
        
        try:
            value = float(cleaned)
            # Validate reasonable salary range
            if 0 < value <= 500000:
                return value
            else:
                return np.nan
        except ValueError:
            return np.nan
    
    df_clean['Salary_Clean'] = df_clean['Salary'].apply(clean_salary)
    
    # Clean age field
    def validate_age(age):
        if pd.isna(age):
            return np.nan
        
        if 18 <= age <= 100:
            return age
        else:
            return np.nan
    
    df_clean['Age_Clean'] = df_clean['Age'].apply(validate_age)
    
    print("Salary cleaning results:")
    for original, clean in zip(df_clean['Salary'], df_clean['Salary_Clean']):
        print(f"  {original} -> {clean}")
    
    print("\nAge validation results:")
    for original, clean in zip(df_clean['Age'], df_clean['Age_Clean']):
        print(f"  {original} -> {clean}")
    
    return df_clean

def clean_dates(df):
    """Parse and standardize date fields"""
    print("\n=== Cleaning Dates ===")
    
    df_clean = df.copy()
    
    # Parse dates with multiple formats
    df_clean['Date_Joined_Clean'] = pd.to_datetime(df_clean['Date_Joined'], 
                                                  errors='coerce', 
                                                  infer_datetime_format=True)
    
    print("Date parsing results:")
    for original, clean in zip(df_clean['Date_Joined'], df_clean['Date_Joined_Clean']):
        print(f"  {original} -> {clean}")
    
    return df_clean

def standardize_categories(df):
    """Standardize categorical data"""
    print("\n=== Standardizing Categories ===")
    
    df_clean = df.copy()
    
    # Standardize department names
    dept_mapping = {
        'it': 'IT',
        'IT': 'IT',
        'sales': 'Sales',
        'Sales': 'Sales',
        'marketing': 'Marketing',
        'MARKETING': 'Marketing',
        'hr': 'HR',
        'HR': 'HR',
        'finance': 'Finance',
        'Finance': 'Finance'
    }
    
    df_clean['Department_Clean'] = df_clean['Department'].map(dept_mapping)
    
    print("Department standardization:")
    for original, clean in zip(df_clean['Department'], df_clean['Department_Clean']):
        print(f"  {original} -> {clean}")
    
    return df_clean

def handle_duplicates(df):
    """Identify and handle duplicate records"""
    print("\n=== Handling Duplicates ===")
    
    df_clean = df.copy()
    
    # Identify duplicates based on key fields
    duplicate_mask = df_clean.duplicated(subset=['ID'], keep='first')
    
    print(f"Found {duplicate_mask.sum()} duplicate records")
    
    if duplicate_mask.sum() > 0:
        print("Duplicate records:")
        duplicates = df_clean[duplicate_mask]
        for idx, row in duplicates.iterrows():
            print(f"  Row {idx}: ID={row['ID']}, Name={row['Name']}")
    
    # Remove duplicates
    df_clean = df_clean[~duplicate_mask]
    
    print(f"Dataset shape after removing duplicates: {df_clean.shape}")
    
    return df_clean

def handle_missing_values(df):
    """Apply different strategies for missing values"""
    print("\n=== Handling Missing Values ===")
    
    df_clean = df.copy()
    
    # Strategy 1: Fill with median for numeric columns
    if 'Age_Clean' in df_clean.columns:
        age_median = df_clean['Age_Clean'].median()
        df_clean['Age_Clean'].fillna(age_median, inplace=True)
        print(f"Filled missing ages with median: {age_median}")
    
    # Strategy 2: Fill with mean for salary
    if 'Salary_Clean' in df_clean.columns:
        salary_mean = df_clean['Salary_Clean'].mean()
        df_clean['Salary_Clean'].fillna(salary_mean, inplace=True)
        print(f"Filled missing salaries with mean: ${salary_mean:,.2f}")
    
    # Strategy 3: Fill with mode for categorical
    if 'Department_Clean' in df_clean.columns:
        dept_mode = df_clean['Department_Clean'].mode()[0]
        df_clean['Department_Clean'].fillna(dept_mode, inplace=True)
        print(f"Filled missing departments with mode: {dept_mode}")
    
    # Strategy 4: Forward fill for names (if similar records exist)
    if 'Name_Clean' in df_clean.columns:
        df_clean['Name_Clean'].fillna(method='ffill', inplace=True)
    
    return df_clean

def detect_outliers(df):
    """Detect outliers using statistical methods"""
    print("\n=== Outlier Detection ===")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                print(f"\nOutliers in {col}:")
                print(f"  Valid range: {lower_bound:.2f} to {upper_bound:.2f}")
                print(f"  Outlier values: {outliers[col].tolist()}")

def create_data_quality_report(df_original, df_cleaned):
    """Generate comprehensive data quality report"""
    print("\n=== Data Quality Report ===")
    
    print(f"Original dataset: {df_original.shape[0]} rows, {df_original.shape[1]} columns")
    print(f"Cleaned dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
    
    # Calculate improvement metrics
    original_missing = df_original.isnull().sum().sum()
    cleaned_missing = df_cleaned.isnull().sum().sum()
    
    print(f"\nMissing values:")
    print(f"  Original: {original_missing}")
    print(f"  Cleaned: {cleaned_missing}")
    print(f"  Improvement: {original_missing - cleaned_missing} fewer missing values")
    
    # Data completeness
    original_completeness = (1 - df_original.isnull().sum() / len(df_original)) * 100
    cleaned_completeness = (1 - df_cleaned.isnull().sum() / len(df_cleaned)) * 100
    
    print(f"\nData completeness by column:")
    for col in df_original.columns:
        if col in df_cleaned.columns:
            orig_comp = original_completeness[col]
            clean_comp = cleaned_completeness[col]
            improvement = clean_comp - orig_comp
            print(f"  {col}: {orig_comp:.1f}% -> {clean_comp:.1f}% ({improvement:+.1f}%)")

def main():
    """Run comprehensive data cleaning demonstration"""
    print("Data Cleaning and Preprocessing Demonstration")
    print("=" * 60)
    
    # Create messy dataset
    df_original = create_messy_dataset()
    print("Created messy dataset for demonstration")
    
    # Assess data quality
    quality_issues = data_quality_assessment(df_original)
    
    # Apply cleaning steps
    df = df_original.copy()
    
    # Step 1: Clean names
    df = clean_names(df)
    
    # Step 2: Validate emails
    df = validate_and_clean_emails(df)
    
    # Step 3: Clean phone numbers
    df = clean_phone_numbers(df)
    
    # Step 4: Handle numeric data
    df = handle_numeric_data(df)
    
    # Step 5: Clean dates
    df = clean_dates(df)
    
    # Step 6: Standardize categories
    df = standardize_categories(df)
    
    # Step 7: Handle duplicates
    df = handle_duplicates(df)
    
    # Step 8: Handle missing values
    df = handle_missing_values(df)
    
    # Step 9: Detect outliers
    detect_outliers(df)
    
    # Generate final report
    create_data_quality_report(df_original, df)
    
    print(f"\nFinal cleaned dataset:")
    print(df.head())
    
    print("\n" + "=" * 60)
    print("Data cleaning demonstration complete!")
    
    return df_original, df

if __name__ == "__main__":
    original_data, cleaned_data = main()
