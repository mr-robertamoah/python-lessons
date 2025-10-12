# Lesson 17: Working with Data Sources

## Learning Objectives
By the end of this lesson, you will be able to:
- Read data from various file formats (CSV, Excel, JSON, XML)
- Connect to and query databases using Python
- Fetch data from web APIs and handle authentication
- Perform basic web scraping for data collection
- Implement ETL (Extract, Transform, Load) processes
- Handle real-time data streams

## File Formats and Data Import

### CSV and Excel Files
```python
import pandas as pd
import numpy as np

# CSV files
df_csv = pd.read_csv('data.csv')
df_csv_custom = pd.read_csv('data.csv', 
                           sep=';',           # Different separator
                           encoding='utf-8',  # Specify encoding
                           parse_dates=['date_column'],
                           na_values=['N/A', 'NULL', ''])

# Excel files
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df_all_sheets = pd.read_excel('data.xlsx', sheet_name=None)  # All sheets

# Writing data
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', sheet_name='Data', index=False)
```

### JSON Data
```python
import json
import requests

# Read JSON file
with open('data.json', 'r') as f:
    json_data = json.load(f)

df_json = pd.read_json('data.json')

# Handle nested JSON
def flatten_json(nested_json, separator='_'):
    """Flatten nested JSON structure"""
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                items.extend(_flatten(v, new_key).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(nested_json)

# Example with nested JSON
nested_data = {
    "user": {
        "name": "John Doe",
        "address": {
            "street": "123 Main St",
            "city": "New York"
        },
        "hobbies": ["reading", "swimming"]
    }
}

flattened = flatten_json(nested_data)
print(flattened)
```

## Database Connections

### SQLite Database
```python
import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect('example.db')

# Create sample table
conn.execute('''
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    salary REAL,
    hire_date DATE
)
''')

# Insert sample data
sample_data = [
    (1, 'Alice Johnson', 'Engineering', 75000, '2022-01-15'),
    (2, 'Bob Smith', 'Marketing', 65000, '2022-03-20'),
    (3, 'Charlie Brown', 'Sales', 55000, '2021-11-10')
]

conn.executemany('INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?)', sample_data)
conn.commit()

# Query data with pandas
df_employees = pd.read_sql_query("SELECT * FROM employees", conn)
print(df_employees)

# Query with parameters
high_salary = pd.read_sql_query(
    "SELECT * FROM employees WHERE salary > ?", 
    conn, 
    params=[60000]
)

conn.close()
```

### PostgreSQL/MySQL Connection
```python
from sqlalchemy import create_engine
import pandas as pd

# Database connection strings
# PostgreSQL: 'postgresql://username:password@host:port/database'
# MySQL: 'mysql+pymysql://username:password@host:port/database'

def connect_to_database(connection_string):
    """Create database connection"""
    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Example usage (replace with actual credentials)
# engine = connect_to_database('postgresql://user:pass@localhost:5432/mydb')
# df = pd.read_sql_query("SELECT * FROM sales_data", engine)
```

## Web APIs

### REST API Requests
```python
import requests
import pandas as pd
from datetime import datetime
import time

class APIClient:
    """Generic API client with error handling and rate limiting"""
    
    def __init__(self, base_url, api_key=None, rate_limit=1):
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def get(self, endpoint, params=None):
        """Make GET request with error handling"""
        self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return None

# Example: Weather API
def get_weather_data(city, api_key):
    """Fetch weather data from OpenWeatherMap API"""
    client = APIClient('https://api.openweathermap.org/data/2.5')
    
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    data = client.get('weather', params)
    
    if data:
        return {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'timestamp': datetime.now()
        }
    return None

# Example: Financial data API
def get_stock_data(symbol, api_key):
    """Fetch stock data from Alpha Vantage API"""
    client = APIClient('https://www.alphavantage.co/query')
    
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'compact'
    }
    
    data = client.get('', params)
    
    if data and 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        
        return df.sort_index()
    
    return None
```

## Web Scraping

### Basic Web Scraping with BeautifulSoup
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin, urlparse

class WebScraper:
    """Basic web scraper with ethical practices"""
    
    def __init__(self, delay=1):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.delay = delay
    
    def get_page(self, url):
        """Fetch webpage with error handling"""
        try:
            time.sleep(self.delay)  # Be respectful to servers
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def scrape_table(self, url, table_selector='table'):
        """Scrape HTML table from webpage"""
        soup = self.get_page(url)
        
        if not soup:
            return None
        
        table = soup.select_one(table_selector)
        if not table:
            print("No table found with given selector")
            return None
        
        # Extract table data
        rows = []
        for tr in table.find_all('tr'):
            row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if row:  # Skip empty rows
                rows.append(row)
        
        if rows:
            # Use first row as headers if it looks like headers
            if len(rows) > 1:
                df = pd.DataFrame(rows[1:], columns=rows[0])
            else:
                df = pd.DataFrame(rows)
            
            return df
        
        return None

# Example: Scrape financial data
def scrape_stock_info(symbol):
    """Scrape basic stock information"""
    scraper = WebScraper(delay=2)
    
    # This is just an example - always check robots.txt and terms of service
    url = f"https://finance.yahoo.com/quote/{symbol}"
    
    soup = scraper.get_page(url)
    if not soup:
        return None
    
    try:
        # Extract basic information (selectors may change)
        price_elem = soup.select_one('[data-symbol="' + symbol + '"] [data-field="regularMarketPrice"]')
        price = price_elem.get_text() if price_elem else "N/A"
        
        return {
            'symbol': symbol,
            'price': price,
            'scraped_at': datetime.now()
        }
    
    except Exception as e:
        print(f"Error parsing stock data: {e}")
        return None
```

## ETL Processes

### Extract, Transform, Load Pipeline
```python
import logging
from datetime import datetime, timedelta

class ETLPipeline:
    """ETL pipeline for data processing"""
    
    def __init__(self, name="ETL_Pipeline"):
        self.name = name
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for ETL process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.name)
    
    def extract(self, sources):
        """Extract data from multiple sources"""
        self.logger.info("Starting data extraction")
        extracted_data = {}
        
        for source_name, source_config in sources.items():
            try:
                if source_config['type'] == 'csv':
                    data = pd.read_csv(source_config['path'])
                
                elif source_config['type'] == 'database':
                    engine = create_engine(source_config['connection_string'])
                    data = pd.read_sql_query(source_config['query'], engine)
                
                elif source_config['type'] == 'api':
                    # Implement API extraction
                    data = self.extract_from_api(source_config)
                
                else:
                    raise ValueError(f"Unknown source type: {source_config['type']}")
                
                extracted_data[source_name] = data
                self.logger.info(f"Extracted {len(data)} records from {source_name}")
            
            except Exception as e:
                self.logger.error(f"Error extracting from {source_name}: {e}")
                extracted_data[source_name] = pd.DataFrame()
        
        return extracted_data
    
    def transform(self, data_dict):
        """Transform extracted data"""
        self.logger.info("Starting data transformation")
        transformed_data = {}
        
        for source_name, df in data_dict.items():
            try:
                # Apply transformations
                df_transformed = df.copy()
                
                # Clean column names
                df_transformed.columns = df_transformed.columns.str.lower().str.replace(' ', '_')
                
                # Handle missing values
                numeric_columns = df_transformed.select_dtypes(include=[np.number]).columns
                df_transformed[numeric_columns] = df_transformed[numeric_columns].fillna(0)
                
                categorical_columns = df_transformed.select_dtypes(include=['object']).columns
                df_transformed[categorical_columns] = df_transformed[categorical_columns].fillna('Unknown')
                
                # Add metadata
                df_transformed['etl_processed_at'] = datetime.now()
                df_transformed['source'] = source_name
                
                transformed_data[source_name] = df_transformed
                self.logger.info(f"Transformed {source_name} data")
            
            except Exception as e:
                self.logger.error(f"Error transforming {source_name}: {e}")
                transformed_data[source_name] = df
        
        return transformed_data
    
    def load(self, data_dict, destination):
        """Load transformed data to destination"""
        self.logger.info("Starting data loading")
        
        try:
            if destination['type'] == 'csv':
                # Combine all data and save to CSV
                combined_df = pd.concat(data_dict.values(), ignore_index=True)
                combined_df.to_csv(destination['path'], index=False)
                self.logger.info(f"Loaded {len(combined_df)} records to {destination['path']}")
            
            elif destination['type'] == 'database':
                engine = create_engine(destination['connection_string'])
                
                for source_name, df in data_dict.items():
                    table_name = destination.get('table_prefix', '') + source_name
                    df.to_sql(table_name, engine, if_exists='replace', index=False)
                    self.logger.info(f"Loaded {len(df)} records to table {table_name}")
            
            else:
                raise ValueError(f"Unknown destination type: {destination['type']}")
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def run(self, sources, destination):
        """Run complete ETL pipeline"""
        start_time = datetime.now()
        self.logger.info(f"Starting ETL pipeline: {self.name}")
        
        try:
            # Extract
            extracted_data = self.extract(sources)
            
            # Transform
            transformed_data = self.transform(extracted_data)
            
            # Load
            self.load(transformed_data, destination)
            
            duration = datetime.now() - start_time
            self.logger.info(f"ETL pipeline completed successfully in {duration}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {e}")
            return False

# Example ETL configuration
etl_config = {
    'sources': {
        'sales_data': {
            'type': 'csv',
            'path': 'sales_data.csv'
        },
        'customer_data': {
            'type': 'csv', 
            'path': 'customer_data.csv'
        }
    },
    'destination': {
        'type': 'csv',
        'path': 'processed_data.csv'
    }
}

# Run ETL pipeline
# etl = ETLPipeline("Sales_ETL")
# success = etl.run(etl_config['sources'], etl_config['destination'])
```

## Real-time Data Processing

### Streaming Data Simulation
```python
import threading
import queue
import time
from datetime import datetime
import json

class DataStream:
    """Simulate real-time data stream"""
    
    def __init__(self, stream_name, data_generator, interval=1):
        self.stream_name = stream_name
        self.data_generator = data_generator
        self.interval = interval
        self.is_running = False
        self.data_queue = queue.Queue()
        self.thread = None
    
    def start(self):
        """Start the data stream"""
        self.is_running = True
        self.thread = threading.Thread(target=self._generate_data)
        self.thread.start()
        print(f"Started data stream: {self.stream_name}")
    
    def stop(self):
        """Stop the data stream"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        print(f"Stopped data stream: {self.stream_name}")
    
    def _generate_data(self):
        """Generate data in separate thread"""
        while self.is_running:
            try:
                data_point = self.data_generator()
                data_point['timestamp'] = datetime.now().isoformat()
                data_point['stream'] = self.stream_name
                
                self.data_queue.put(data_point)
                time.sleep(self.interval)
            
            except Exception as e:
                print(f"Error generating data: {e}")
    
    def get_data(self, timeout=1):
        """Get data from stream"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# Data generators
def generate_sensor_data():
    """Generate simulated sensor data"""
    return {
        'sensor_id': np.random.choice(['sensor_1', 'sensor_2', 'sensor_3']),
        'temperature': np.random.normal(25, 5),
        'humidity': np.random.uniform(30, 80),
        'pressure': np.random.normal(1013, 10)
    }

def generate_sales_data():
    """Generate simulated sales data"""
    products = ['Product_A', 'Product_B', 'Product_C']
    return {
        'product': np.random.choice(products),
        'quantity': np.random.randint(1, 10),
        'price': np.random.uniform(10, 100),
        'customer_id': f"customer_{np.random.randint(1, 1000)}"
    }

# Stream processor
class StreamProcessor:
    """Process real-time data streams"""
    
    def __init__(self):
        self.streams = {}
        self.processed_data = []
        self.is_processing = False
    
    def add_stream(self, stream):
        """Add a data stream"""
        self.streams[stream.stream_name] = stream
    
    def start_processing(self):
        """Start processing all streams"""
        self.is_processing = True
        
        # Start all streams
        for stream in self.streams.values():
            stream.start()
        
        # Process data
        processing_thread = threading.Thread(target=self._process_streams)
        processing_thread.start()
    
    def stop_processing(self):
        """Stop processing all streams"""
        self.is_processing = False
        
        # Stop all streams
        for stream in self.streams.values():
            stream.stop()
    
    def _process_streams(self):
        """Process data from all streams"""
        while self.is_processing:
            for stream_name, stream in self.streams.items():
                data = stream.get_data(timeout=0.1)
                
                if data:
                    # Process the data point
                    processed = self.process_data_point(data)
                    self.processed_data.append(processed)
                    
                    # Keep only recent data (last 100 points)
                    if len(self.processed_data) > 100:
                        self.processed_data = self.processed_data[-100:]
    
    def process_data_point(self, data):
        """Process individual data point"""
        # Add processing logic here
        processed = data.copy()
        processed['processed_at'] = datetime.now().isoformat()
        
        # Example: Calculate derived metrics
        if 'temperature' in data and 'humidity' in data:
            # Heat index calculation (simplified)
            processed['heat_index'] = data['temperature'] + (data['humidity'] * 0.1)
        
        return processed
    
    def get_recent_data(self, n=10):
        """Get recent processed data"""
        return self.processed_data[-n:] if self.processed_data else []

# Example usage
if __name__ == "__main__":
    # Create streams
    sensor_stream = DataStream("sensors", generate_sensor_data, interval=2)
    sales_stream = DataStream("sales", generate_sales_data, interval=3)
    
    # Create processor
    processor = StreamProcessor()
    processor.add_stream(sensor_stream)
    processor.add_stream(sales_stream)
    
    # Process for a short time
    print("Starting real-time processing...")
    processor.start_processing()
    
    # Let it run for 10 seconds
    time.sleep(10)
    
    # Stop processing
    processor.stop_processing()
    
    # Show results
    recent_data = processor.get_recent_data(5)
    print(f"\nProcessed {len(processor.processed_data)} data points")
    print("Recent data:")
    for data_point in recent_data:
        print(json.dumps(data_point, indent=2))
```

## Key Terminology

- **ETL**: Extract, Transform, Load - data integration process
- **API**: Application Programming Interface for data access
- **REST**: Representational State Transfer - web service architecture
- **JSON**: JavaScript Object Notation - data interchange format
- **Web Scraping**: Extracting data from websites
- **Rate Limiting**: Controlling request frequency to APIs
- **Data Stream**: Continuous flow of data in real-time
- **Database Connection**: Link between application and database
- **Authentication**: Verifying identity for data access

## Looking Ahead

This completes Phase 3 of our Python course! You now have comprehensive skills in:
- Data visualization and statistical analysis
- Feature engineering and machine learning
- Automated ML pipelines and data sources

In Phase 4 (Lessons 18-21), we'll cover:
- Advanced Python concepts and optimization
- Real-world project development
- Production deployment considerations
- Capstone project to demonstrate all skills
