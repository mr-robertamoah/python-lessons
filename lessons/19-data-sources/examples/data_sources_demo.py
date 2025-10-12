#!/usr/bin/env python3
"""
Data Sources Demo
Examples of working with various data sources
"""

import requests
import pandas as pd
import sqlite3
import json
import time
from urllib.parse import urljoin

def api_data_demo():
    """Demonstrate API data fetching"""
    print("=== API Data Demo ===")
    
    # JSONPlaceholder API example
    base_url = "https://jsonplaceholder.typicode.com"
    
    try:
        # Fetch posts
        posts_response = requests.get(f"{base_url}/posts")
        posts_response.raise_for_status()
        posts_data = posts_response.json()
        
        print(f"Fetched {len(posts_data)} posts")
        
        # Convert to DataFrame
        posts_df = pd.DataFrame(posts_data)
        print(f"Posts DataFrame shape: {posts_df.shape}")
        print(f"Columns: {posts_df.columns.tolist()}")
        
        # Fetch users
        users_response = requests.get(f"{base_url}/users")
        users_response.raise_for_status()
        users_data = users_response.json()
        
        users_df = pd.DataFrame(users_data)
        print(f"Users DataFrame shape: {users_df.shape}")
        
        # Merge data
        merged_df = posts_df.merge(users_df, left_on='userId', right_on='id', suffixes=('_post', '_user'))
        print(f"Merged DataFrame shape: {merged_df.shape}")
        
        return merged_df
        
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return None

def database_demo():
    """Demonstrate database operations"""
    print("\n=== Database Demo ===")
    
    # Create in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    
    try:
        # Create sample table
        conn.execute('''
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                salary REAL,
                hire_date TEXT
            )
        ''')
        
        # Insert sample data
        employees = [
            (1, 'Alice Johnson', 'Engineering', 75000, '2022-01-15'),
            (2, 'Bob Smith', 'Marketing', 65000, '2022-03-20'),
            (3, 'Carol Wilson', 'Engineering', 80000, '2021-11-10'),
            (4, 'David Brown', 'Sales', 70000, '2022-02-28'),
            (5, 'Eva Davis', 'Marketing', 68000, '2021-12-05')
        ]
        
        conn.executemany(
            'INSERT INTO employees VALUES (?, ?, ?, ?, ?)',
            employees
        )
        conn.commit()
        
        # Query data using pandas
        df = pd.read_sql_query(
            "SELECT * FROM employees WHERE salary > 70000",
            conn
        )
        
        print(f"High-salary employees: {len(df)}")
        print(df[['name', 'department', 'salary']])
        
        # Aggregate query
        dept_stats = pd.read_sql_query('''
            SELECT department, 
                   COUNT(*) as employee_count,
                   AVG(salary) as avg_salary,
                   MAX(salary) as max_salary
            FROM employees 
            GROUP BY department
        ''', conn)
        
        print(f"\nDepartment statistics:")
        print(dept_stats)
        
        return df
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()

def file_processing_demo():
    """Demonstrate file data processing"""
    print("\n=== File Processing Demo ===")
    
    # Create sample CSV data
    sample_data = {
        'product_id': range(1, 1001),
        'product_name': [f'Product_{i}' for i in range(1, 1001)],
        'category': ['Electronics', 'Clothing', 'Books'] * 334,
        'price': [round(10 + i * 0.5, 2) for i in range(1000)],
        'stock': [100 - i % 50 for i in range(1000)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    csv_file = 'sample_products.csv'
    df.to_csv(csv_file, index=False)
    print(f"Created sample CSV with {len(df)} records")
    
    # Read in chunks (useful for large files)
    chunk_size = 200
    chunks_processed = 0
    total_value = 0
    
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        chunks_processed += 1
        chunk_value = (chunk['price'] * chunk['stock']).sum()
        total_value += chunk_value
        
        if chunks_processed <= 2:  # Show first 2 chunks
            print(f"Chunk {chunks_processed}: {len(chunk)} records, value: ${chunk_value:,.2f}")
    
    print(f"Processed {chunks_processed} chunks, total inventory value: ${total_value:,.2f}")
    
    # Clean up
    import os
    os.remove(csv_file)
    
    return df

class DataConnector:
    """Generic data connector with retry logic"""
    
    def __init__(self, max_retries=3, retry_delay=1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def fetch_with_retry(self, fetch_func, *args, **kwargs):
        """Execute function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return fetch_func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)
    
    def fetch_api_data(self, url, params=None):
        """Fetch data from API with retry"""
        def _fetch():
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        
        return self.fetch_with_retry(_fetch)

def data_connector_demo():
    """Demonstrate robust data connector"""
    print("\n=== Data Connector Demo ===")
    
    connector = DataConnector(max_retries=2, retry_delay=0.5)
    
    try:
        # This should work
        data = connector.fetch_api_data("https://jsonplaceholder.typicode.com/posts/1")
        print(f"Successfully fetched post: {data['title']}")
        
        # This will fail and demonstrate retry
        try:
            bad_data = connector.fetch_api_data("https://nonexistent-api.com/data")
        except Exception as e:
            print(f"Failed after retries: {type(e).__name__}")
            
    except Exception as e:
        print(f"Connector error: {e}")

def multi_source_integration_demo():
    """Demonstrate integrating multiple data sources"""
    print("\n=== Multi-Source Integration Demo ===")
    
    # Source 1: API data
    try:
        api_response = requests.get("https://jsonplaceholder.typicode.com/users")
        api_data = pd.DataFrame(api_response.json())
        print(f"API data: {len(api_data)} users")
    except:
        # Fallback to mock data
        api_data = pd.DataFrame({
            'id': range(1, 6),
            'name': ['Alice', 'Bob', 'Carol', 'David', 'Eva'],
            'email': [f'user{i}@example.com' for i in range(1, 6)]
        })
        print(f"Using mock API data: {len(api_data)} users")
    
    # Source 2: Database data (simulated)
    db_data = pd.DataFrame({
        'user_id': range(1, 6),
        'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Marketing'],
        'salary': [75000, 65000, 70000, 80000, 68000]
    })
    print(f"Database data: {len(db_data)} employee records")
    
    # Source 3: File data (simulated)
    file_data = pd.DataFrame({
        'employee_id': range(1, 6),
        'performance_score': [4.2, 3.8, 4.5, 4.1, 3.9],
        'last_review': ['2023-12-01', '2023-11-15', '2023-12-10', '2023-11-20', '2023-12-05']
    })
    print(f"File data: {len(file_data)} performance records")
    
    # Integrate all sources
    # Step 1: Merge API and database data
    integrated_df = api_data.merge(
        db_data, 
        left_on='id', 
        right_on='user_id', 
        how='inner'
    )
    
    # Step 2: Add file data
    integrated_df = integrated_df.merge(
        file_data,
        left_on='id',
        right_on='employee_id',
        how='left'
    )
    
    print(f"Integrated dataset: {integrated_df.shape}")
    print(f"Columns: {integrated_df.columns.tolist()}")
    
    # Show sample of integrated data
    print(f"\nSample integrated data:")
    print(integrated_df[['name', 'department', 'salary', 'performance_score']].head())
    
    return integrated_df

if __name__ == "__main__":
    api_data_demo()
    database_demo()
    file_processing_demo()
    data_connector_demo()
    multi_source_integration_demo()
