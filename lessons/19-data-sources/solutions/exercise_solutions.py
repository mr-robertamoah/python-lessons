# Lesson 17 Solutions: Working with Data Sources

import requests
import pandas as pd
import sqlite3
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import os

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: API Data Collection
print("Exercise 1: API Data Collection")
print("-" * 50)

def fetch_api_data_with_error_handling():
    """Fetch data from API with proper error handling"""
    
    base_url = "https://jsonplaceholder.typicode.com"
    
    def safe_api_call(endpoint, params=None):
        """Make API call with error handling"""
        try:
            url = f"{base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Timeout error for {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error for {endpoint}: {e}")
            return None
        except json.JSONDecodeError:
            print(f"JSON decode error for {endpoint}")
            return None
    
    # Fetch posts with pagination
    all_posts = []
    page = 1
    while True:
        posts = safe_api_call("posts", params={"_page": page, "_limit": 10})
        if not posts:
            break
        all_posts.extend(posts)
        page += 1
        if page > 10:  # Limit to prevent infinite loop
            break
        time.sleep(0.1)  # Rate limiting
    
    print(f"Fetched {len(all_posts)} posts")
    
    # Fetch users
    users = safe_api_call("users")
    if users:
        print(f"Fetched {len(users)} users")
        
        # Convert to DataFrames
        posts_df = pd.DataFrame(all_posts)
        users_df = pd.DataFrame(users)
        
        # Merge data
        if not posts_df.empty and not users_df.empty:
            merged_df = posts_df.merge(users_df, left_on='userId', right_on='id', suffixes=('_post', '_user'))
            print(f"Merged dataset shape: {merged_df.shape}")
            return merged_df
    
    return None

api_data = fetch_api_data_with_error_handling()

# Exercise 2: Database Connections
print("\n" + "="*50)
print("Exercise 2: Database Connections")
print("-" * 50)

def database_operations_demo():
    """Comprehensive database operations"""
    
    # Create database and tables
    conn = sqlite3.connect(':memory:')
    
    try:
        # Create tables
        conn.execute('''
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                city TEXT,
                signup_date TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                product TEXT,
                amount REAL,
                order_date TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (id)
            )
        ''')
        
        # Insert sample data
        customers = [
            (1, 'Alice Johnson', 'alice@email.com', 'NYC', '2023-01-15'),
            (2, 'Bob Smith', 'bob@email.com', 'LA', '2023-02-20'),
            (3, 'Carol Wilson', 'carol@email.com', 'Chicago', '2023-03-10'),
            (4, 'David Brown', 'david@email.com', 'Houston', '2023-04-05'),
            (5, 'Eva Davis', 'eva@email.com', 'Phoenix', '2023-05-12')
        ]
        
        orders = [
            (1, 1, 'Laptop', 999.99, '2023-06-01'),
            (2, 1, 'Mouse', 29.99, '2023-06-02'),
            (3, 2, 'Keyboard', 79.99, '2023-06-03'),
            (4, 3, 'Monitor', 299.99, '2023-06-04'),
            (5, 2, 'Headphones', 149.99, '2023-06-05'),
            (6, 4, 'Tablet', 399.99, '2023-06-06'),
            (7, 5, 'Phone', 699.99, '2023-06-07')
        ]
        
        conn.executemany('INSERT INTO customers VALUES (?, ?, ?, ?, ?)', customers)
        conn.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?)', orders)
        conn.commit()
        
        print("Database tables created and populated")
        
        # Query 1: Customer order summary
        customer_orders = pd.read_sql_query('''
            SELECT c.name, c.city, 
                   COUNT(o.id) as order_count,
                   SUM(o.amount) as total_spent,
                   AVG(o.amount) as avg_order_value
            FROM customers c
            LEFT JOIN orders o ON c.id = o.customer_id
            GROUP BY c.id, c.name, c.city
            ORDER BY total_spent DESC
        ''', conn)
        
        print(f"\nCustomer Order Summary:")
        print(customer_orders)
        
        # Query 2: Top products
        top_products = pd.read_sql_query('''
            SELECT product, 
                   COUNT(*) as order_count,
                   SUM(amount) as total_revenue,
                   AVG(amount) as avg_price
            FROM orders
            GROUP BY product
            ORDER BY total_revenue DESC
        ''', conn)
        
        print(f"\nTop Products:")
        print(top_products)
        
        # Update operation
        conn.execute('''
            UPDATE customers 
            SET city = 'San Francisco' 
            WHERE name = 'Alice Johnson'
        ''')
        conn.commit()
        
        print(f"\nUpdated Alice's city to San Francisco")
        
        return customer_orders, top_products
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None, None
    finally:
        conn.close()

customer_summary, product_summary = database_operations_demo()

print("\n=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: Multi-Source Data Integration
print("Exercise 4: Multi-Source Data Integration")
print("-" * 50)

class MultiSourceIntegrator:
    """Integrate data from multiple sources"""
    
    def __init__(self):
        self.data_sources = {}
    
    def add_api_source(self, name, url, transform_func=None):
        """Add API data source"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if transform_func:
                data = transform_func(data)
            
            self.data_sources[name] = pd.DataFrame(data)
            print(f"Added API source '{name}': {len(self.data_sources[name])} records")
            
        except Exception as e:
            print(f"Failed to add API source '{name}': {e}")
    
    def add_csv_source(self, name, data):
        """Add CSV data source (simulated)"""
        self.data_sources[name] = pd.DataFrame(data)
        print(f"Added CSV source '{name}': {len(self.data_sources[name])} records")
    
    def add_database_source(self, name, query_result):
        """Add database query result"""
        self.data_sources[name] = query_result
        print(f"Added database source '{name}': {len(self.data_sources[name])} records")
    
    def integrate_sources(self, join_config):
        """Integrate all sources based on join configuration"""
        if not self.data_sources:
            print("No data sources available")
            return None
        
        # Start with first source
        source_names = list(self.data_sources.keys())
        integrated_df = self.data_sources[source_names[0]].copy()
        
        print(f"Starting integration with '{source_names[0]}'")
        
        # Join with remaining sources
        for source_name in source_names[1:]:
            if source_name in join_config:
                join_info = join_config[source_name]
                integrated_df = integrated_df.merge(
                    self.data_sources[source_name],
                    left_on=join_info['left_on'],
                    right_on=join_info['right_on'],
                    how=join_info.get('how', 'inner'),
                    suffixes=join_info.get('suffixes', ('', '_y'))
                )
                print(f"Joined with '{source_name}': {integrated_df.shape}")
        
        return integrated_df

# Demo multi-source integration
integrator = MultiSourceIntegrator()

# Add mock API source
try:
    integrator.add_api_source(
        'users', 
        'https://jsonplaceholder.typicode.com/users',
        transform_func=lambda data: [{'id': u['id'], 'name': u['name'], 'email': u['email']} for u in data[:5]]
    )
except:
    # Fallback to mock data
    mock_users = [
        {'id': 1, 'name': 'Alice Johnson', 'email': 'alice@email.com'},
        {'id': 2, 'name': 'Bob Smith', 'email': 'bob@email.com'},
        {'id': 3, 'name': 'Carol Wilson', 'email': 'carol@email.com'}
    ]
    integrator.add_csv_source('users', mock_users)

# Add CSV source (simulated)
csv_data = [
    {'user_id': 1, 'department': 'Engineering', 'salary': 75000},
    {'user_id': 2, 'department': 'Marketing', 'salary': 65000},
    {'user_id': 3, 'department': 'Sales', 'salary': 70000}
]
integrator.add_csv_source('employees', csv_data)

# Add database source (simulated)
db_data = pd.DataFrame([
    {'emp_id': 1, 'performance': 4.2, 'bonus': 5000},
    {'emp_id': 2, 'performance': 3.8, 'bonus': 3000},
    {'emp_id': 3, 'performance': 4.5, 'bonus': 6000}
])
integrator.add_database_source('performance', db_data)

# Configure joins
join_config = {
    'employees': {
        'left_on': 'id',
        'right_on': 'user_id',
        'how': 'inner'
    },
    'performance': {
        'left_on': 'id',
        'right_on': 'emp_id',
        'how': 'left'
    }
}

# Integrate all sources
integrated_data = integrator.integrate_sources(join_config)

if integrated_data is not None:
    print(f"\nIntegrated dataset:")
    print(integrated_data)

# Exercise 5: Real-Time Data Pipeline
print("\n" + "="*50)
print("Exercise 5: Real-Time Data Pipeline")
print("-" * 50)

class RealTimeDataPipeline:
    """Real-time data collection and processing pipeline"""
    
    def __init__(self, storage_file='realtime_data.csv'):
        self.storage_file = storage_file
        self.is_running = False
        self.data_buffer = []
        self.error_count = 0
        self.success_count = 0
    
    def fetch_data_point(self):
        """Simulate fetching a single data point"""
        import random
        
        # Simulate API call with occasional failures
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Simulated API failure")
        
        # Generate mock data point
        data_point = {
            'timestamp': pd.Timestamp.now(),
            'value': random.uniform(10, 100),
            'category': random.choice(['A', 'B', 'C']),
            'status': 'active'
        }
        
        return data_point
    
    def process_data_point(self, data_point):
        """Process individual data point"""
        # Add derived fields
        data_point['hour'] = data_point['timestamp'].hour
        data_point['value_squared'] = data_point['value'] ** 2
        
        return data_point
    
    def store_data(self, data_points):
        """Store data points to file"""
        df = pd.DataFrame(data_points)
        
        # Append to existing file or create new
        if os.path.exists(self.storage_file):
            df.to_csv(self.storage_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.storage_file, index=False)
    
    def run_pipeline(self, duration_seconds=10, batch_size=5):
        """Run the real-time pipeline"""
        print(f"Starting real-time pipeline for {duration_seconds} seconds...")
        
        self.is_running = True
        start_time = time.time()
        
        while self.is_running and (time.time() - start_time) < duration_seconds:
            try:
                # Fetch data point
                data_point = self.fetch_data_point()
                
                # Process data point
                processed_point = self.process_data_point(data_point)
                
                # Add to buffer
                self.data_buffer.append(processed_point)
                self.success_count += 1
                
                # Store batch when buffer is full
                if len(self.data_buffer) >= batch_size:
                    self.store_data(self.data_buffer)
                    print(f"Stored batch of {len(self.data_buffer)} records")
                    self.data_buffer = []
                
            except Exception as e:
                self.error_count += 1
                print(f"Error fetching data: {e}")
            
            time.sleep(0.5)  # Wait before next fetch
        
        # Store remaining data in buffer
        if self.data_buffer:
            self.store_data(self.data_buffer)
            print(f"Stored final batch of {len(self.data_buffer)} records")
        
        self.is_running = False
        
        print(f"Pipeline completed:")
        print(f"  Success: {self.success_count}")
        print(f"  Errors: {self.error_count}")
        print(f"  Success rate: {self.success_count/(self.success_count+self.error_count)*100:.1f}%")
        
        # Show stored data summary
        if os.path.exists(self.storage_file):
            stored_df = pd.read_csv(self.storage_file)
            print(f"  Total stored records: {len(stored_df)}")
            
            # Clean up
            os.remove(self.storage_file)

# Demo real-time pipeline
pipeline = RealTimeDataPipeline()
pipeline.run_pipeline(duration_seconds=5, batch_size=3)

# Exercise 6: Large Dataset Processing
print("\n" + "="*50)
print("Exercise 6: Large Dataset Processing")
print("-" * 50)

def process_large_dataset_demo():
    """Demonstrate efficient large dataset processing"""
    
    # Create large sample dataset
    print("Creating large sample dataset...")
    
    large_data = {
        'id': range(1, 10001),
        'category': ['A', 'B', 'C', 'D'] * 2500,
        'value': [i * 0.1 + (i % 100) for i in range(10000)],
        'date': pd.date_range('2020-01-01', periods=10000, freq='1H'),
        'status': ['active', 'inactive'] * 5000
    }
    
    large_df = pd.DataFrame(large_data)
    csv_file = 'large_dataset.csv'
    large_df.to_csv(csv_file, index=False)
    
    print(f"Created dataset with {len(large_df)} records")
    
    # Process in chunks
    chunk_size = 1000
    chunk_results = []
    
    print(f"Processing in chunks of {chunk_size}...")
    
    for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
        # Process chunk
        chunk_summary = {
            'chunk_id': i,
            'record_count': len(chunk),
            'avg_value': chunk['value'].mean(),
            'max_value': chunk['value'].max(),
            'active_count': (chunk['status'] == 'active').sum()
        }
        
        chunk_results.append(chunk_summary)
        
        if i < 3:  # Show first 3 chunks
            print(f"  Chunk {i}: {chunk_summary}")
    
    # Aggregate results
    results_df = pd.DataFrame(chunk_results)
    
    print(f"\nProcessed {len(chunk_results)} chunks")
    print(f"Total records: {results_df['record_count'].sum()}")
    print(f"Overall average value: {(results_df['avg_value'] * results_df['record_count']).sum() / results_df['record_count'].sum():.2f}")
    print(f"Total active records: {results_df['active_count'].sum()}")
    
    # Parallel processing demo
    def process_chunk_parallel(chunk_data):
        """Process chunk in parallel"""
        chunk_id, chunk = chunk_data
        
        # Simulate processing time
        time.sleep(0.1)
        
        return {
            'chunk_id': chunk_id,
            'processed_records': len(chunk),
            'sum_value': chunk['value'].sum()
        }
    
    print(f"\nDemonstrating parallel processing...")
    
    # Read chunks for parallel processing
    chunks = [(i, chunk) for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size))]
    
    # Process first 5 chunks in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        parallel_results = list(executor.map(process_chunk_parallel, chunks[:5]))
    
    print(f"Parallel processing results:")
    for result in parallel_results:
        print(f"  Chunk {result['chunk_id']}: {result['processed_records']} records, sum: {result['sum_value']:.2f}")
    
    # Clean up
    os.remove(csv_file)

process_large_dataset_demo()

print("\n" + "="*50)
print("Data Sources exercise solutions complete!")
print("Key concepts demonstrated:")
print("- API data fetching with error handling and rate limiting")
print("- Database operations with complex queries and joins")
print("- Multi-source data integration with configurable joins")
print("- Real-time data pipeline with buffering and error handling")
print("- Large dataset processing with chunking and parallel processing")
