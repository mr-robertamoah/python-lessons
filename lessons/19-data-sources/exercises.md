# Lesson 17 Exercises: Working with Data Sources

## Guided Exercises (Do with Instructor)

### Exercise 1: API Data Collection
**Goal**: Fetch data from REST APIs

**Tasks**:
1. Make GET requests to public APIs
2. Handle JSON responses
3. Parse and structure API data
4. Handle API errors and rate limits

```python
import requests
import pandas as pd
import time

# Example: JSONPlaceholder API
url = "https://jsonplaceholder.typicode.com/posts"
response = requests.get(url)
# Process response data
```

---

### Exercise 2: Database Connections
**Goal**: Connect to and query databases

**Tasks**:
1. Connect to SQLite database
2. Execute SELECT queries
3. Insert and update data
4. Use pandas for database operations

```python
import sqlite3
import pandas as pd

# Create connection and query data
conn = sqlite3.connect('example.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)
```

---

### Exercise 3: Web Scraping Basics
**Goal**: Extract data from web pages

**Tasks**:
1. Parse HTML with BeautifulSoup
2. Extract specific elements
3. Handle different page structures
4. Respect robots.txt and rate limits

```python
from bs4 import BeautifulSoup
import requests

# Scrape web page data
url = "https://example.com"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Multi-Source Data Integration
**Goal**: Combine data from multiple sources

**Tasks**:
1. Fetch data from 2-3 different APIs
2. Load data from CSV files
3. Query database for additional data
4. Merge and align all data sources
5. Handle missing data and conflicts

---

### Exercise 5: Real-Time Data Pipeline
**Goal**: Build streaming data collection

**Tasks**:
1. Set up periodic API calls
2. Store data incrementally
3. Handle data updates and duplicates
4. Monitor data quality
5. Create alerts for data issues

---

### Exercise 6: Large Dataset Processing
**Goal**: Handle big data efficiently

**Tasks**:
1. Process large CSV files in chunks
2. Use database pagination
3. Implement parallel data fetching
4. Optimize memory usage
5. Create progress monitoring

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Custom Data Connector
**Goal**: Build reusable data access layer

**Tasks**:
1. Create abstract base class for data sources
2. Implement specific connectors (API, DB, files)
3. Add caching and retry logic
4. Include data validation
5. Build configuration management

### Challenge 2: Data Lake Architecture
**Goal**: Design scalable data storage

**Tasks**:
1. Implement data ingestion pipeline
2. Create data cataloging system
3. Add data lineage tracking
4. Implement access controls
5. Build monitoring dashboard

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Fetch data from REST APIs
- [ ] Connect to and query databases
- [ ] Scrape data from web pages responsibly
- [ ] Integrate data from multiple sources
- [ ] Handle large datasets efficiently
- [ ] Build real-time data pipelines
- [ ] Implement error handling and retries

## Git Reminder

Save your work:
```bash
git add .
git commit -m "Complete Lesson 17: Data Sources"
git push
```
