# Lesson 11 Exercises: Data Cleaning and Preprocessing

## Guided Exercises (Do with Instructor)

### Exercise 1: Identifying Data Quality Issues
**Goal**: Learn to systematically assess data quality

**Tasks**:
1. Load a messy dataset and perform initial exploration
2. Identify different types of data quality issues
3. Create a data quality report
4. Prioritize cleaning tasks

```python
import pandas as pd
import numpy as np

# Sample messy data
messy_data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Name': ['John Doe', 'jane smith', 'ALICE', '', 'Bob Wilson', 'Carol', 'john doe', 'David'],
    'Age': [25, 30, -5, 150, 28, np.nan, 25, 35],
    'Email': ['john@email.com', 'jane@invalid', 'alice@email.com', '', 'bob@email.com', 'carol@email', 'john@email.com', 'david@email.com'],
    'Salary': [50000, 60000, 0, 1000000, 55000, np.nan, 50000, 65000],
    'Date_Joined': ['2023-01-15', '2023/02/20', '15-03-2023', 'invalid', '2023-04-01', '', '2023-01-15', '2023-05-10']
}

df = pd.DataFrame(messy_data)
# Analyze data quality issues
```

---

### Exercise 2: Handling Missing Values
**Goal**: Apply different strategies for missing data

**Tasks**:
1. Identify patterns in missing data
2. Apply different imputation strategies
3. Compare results of different approaches
4. Document decisions and rationale

**Missing Value Strategies**:
- Deletion (listwise, pairwise)
- Mean/median/mode imputation
- Forward/backward fill
- Interpolation
- Predictive imputation

---

### Exercise 3: Data Type Conversion and Validation
**Goal**: Ensure correct data types and valid values

**Tasks**:
1. Convert string dates to datetime
2. Handle categorical variables
3. Validate numeric ranges
4. Clean and standardize text fields

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Customer Database Cleaning
**Goal**: Clean a realistic customer database

**Scenario**: You've received a customer database from multiple sources with various quality issues.

```python
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
```

**Cleaning Tasks**:
1. Standardize name formats (proper case)
2. Validate and clean email addresses
3. Normalize phone number formats
4. Standardize address components
5. Handle duplicate records
6. Validate age and income ranges
7. Parse and validate dates
8. Create data quality metrics

---

### Exercise 5: Sales Transaction Cleaning
**Goal**: Clean transactional data with various anomalies

**Data Issues to Address**:
- Negative quantities and prices
- Invalid product codes
- Missing customer information
- Inconsistent date formats
- Currency symbols in numeric fields
- Outliers in transaction amounts

**Tasks**:
1. Identify and handle negative values
2. Validate product codes against master list
3. Standardize currency formats
4. Detect and handle outliers
5. Fill missing customer data using lookup tables
6. Create transaction validation rules

---

### Exercise 6: Survey Data Preprocessing
**Goal**: Clean survey responses with text and categorical data

**Challenges**:
- Free-text responses with typos
- Inconsistent categorical responses
- Partial responses and skip patterns
- Scale validation (1-5, 1-10)
- Multiple choice formatting issues

**Tasks**:
1. Standardize categorical responses
2. Clean and categorize free-text fields
3. Handle skip logic and partial responses
4. Validate scale responses
5. Create derived variables from multiple questions

---

### Exercise 7: Financial Data Validation
**Goal**: Clean and validate financial datasets

**Data Types**:
- Stock prices with splits and dividends
- Currency conversions
- Financial ratios with extreme values
- Date alignment across different markets

**Tasks**:
1. Adjust prices for stock splits
2. Handle currency conversion rates
3. Validate financial ratio calculations
4. Align data across different time zones
5. Detect and handle market holidays

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Multi-Source Data Integration
**Goal**: Clean and merge data from multiple sources

**Scenario**: Combine customer data from:
- CRM system (structured)
- Web analytics (semi-structured)
- Social media (unstructured)
- Transaction logs (structured)

**Tasks**:
1. Standardize identifiers across sources
2. Resolve conflicting information
3. Handle different data formats
4. Create master customer record
5. Implement data lineage tracking

---

### Challenge 2: Real-Time Data Cleaning Pipeline
**Goal**: Build automated cleaning pipeline

**Requirements**:
1. Process streaming data
2. Apply validation rules in real-time
3. Handle data quality alerts
4. Maintain cleaning statistics
5. Implement rollback mechanisms

---

### Challenge 3: Advanced Outlier Detection
**Goal**: Implement sophisticated outlier detection

**Methods**:
1. Statistical methods (Z-score, IQR)
2. Machine learning approaches (Isolation Forest)
3. Domain-specific rules
4. Multivariate outlier detection
5. Time series anomaly detection

---

## Specialized Cleaning Scenarios

### Exercise 8: Text Data Cleaning
**Goal**: Clean unstructured text data

**Tasks**:
1. Remove HTML tags and special characters
2. Standardize encoding (UTF-8)
3. Handle different languages
4. Extract structured information from text
5. Normalize abbreviations and acronyms

```python
# Sample messy text data
text_data = [
    "John Doe - CEO @ ABC Corp. (john.doe@abc.com)",
    "jane smith, VP Sales, XYZ Inc. jane@xyz.com",
    "Dr. Alice Johnson, MD - Chief Medical Officer",
    "Bob Wilson Jr., Software Engineer @ Tech Co.",
    "Carol Brown, Ph.D. - Research Director"
]
```

---

### Exercise 9: Geographic Data Cleaning
**Goal**: Clean and standardize location data

**Tasks**:
1. Standardize address formats
2. Geocode addresses to coordinates
3. Validate zip codes and postal codes
4. Handle international address formats
5. Resolve ambiguous location names

---

### Exercise 10: Time Series Data Cleaning
**Goal**: Clean temporal data with gaps and irregularities

**Issues**:
- Missing time periods
- Irregular sampling intervals
- Time zone inconsistencies
- Daylight saving time adjustments
- Leap years and leap seconds

**Tasks**:
1. Identify and fill time gaps
2. Resample irregular time series
3. Handle time zone conversions
4. Detect and correct time shifts
5. Validate temporal sequences

---

## Data Quality Assessment

### Exercise 11: Comprehensive Data Profiling
**Goal**: Create detailed data quality reports

**Metrics to Calculate**:
1. Completeness (missing value rates)
2. Validity (format compliance)
3. Accuracy (correctness checks)
4. Consistency (cross-field validation)
5. Uniqueness (duplicate detection)
6. Timeliness (data freshness)

---

### Exercise 12: Data Quality Monitoring
**Goal**: Implement ongoing quality monitoring

**Tasks**:
1. Create data quality dashboards
2. Set up automated quality checks
3. Implement alerting systems
4. Track quality trends over time
5. Generate quality scorecards

---

## Advanced Techniques

### Exercise 13: Machine Learning for Data Cleaning
**Goal**: Use ML techniques for automated cleaning

**Applications**:
1. Duplicate detection using similarity measures
2. Missing value imputation using predictive models
3. Outlier detection using clustering
4. Data type inference using pattern recognition
5. Entity resolution using fuzzy matching

---

### Exercise 14: Data Cleaning at Scale
**Goal**: Handle large datasets efficiently

**Techniques**:
1. Chunk processing for memory efficiency
2. Parallel processing for speed
3. Sampling for quality assessment
4. Incremental cleaning for streaming data
5. Distributed cleaning using frameworks

---

## Industry-Specific Scenarios

### Exercise 15: Healthcare Data Cleaning
**Goal**: Clean medical data with privacy constraints

**Challenges**:
- HIPAA compliance requirements
- Medical coding standardization
- Date of birth vs age consistency
- Drug name standardization
- Lab result validation

---

### Exercise 16: E-commerce Data Cleaning
**Goal**: Clean product and transaction data

**Tasks**:
1. Standardize product names and descriptions
2. Clean and categorize product attributes
3. Validate pricing and inventory data
4. Handle seasonal and promotional pricing
5. Clean customer review text

---

### Exercise 17: IoT Sensor Data Cleaning
**Goal**: Clean sensor data streams

**Issues**:
- Sensor drift and calibration
- Communication errors
- Environmental interference
- Battery-related data degradation
- Synchronization across sensors

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Systematically assess data quality issues
- [ ] Choose appropriate strategies for missing values
- [ ] Validate and convert data types correctly
- [ ] Standardize text and categorical data
- [ ] Detect and handle outliers appropriately
- [ ] Clean and validate dates and times
- [ ] Handle duplicate records intelligently
- [ ] Implement data validation rules
- [ ] Create data quality metrics and reports
- [ ] Build automated cleaning pipelines
- [ ] Handle domain-specific cleaning challenges
- [ ] Scale cleaning operations for large datasets

## Data Cleaning Best Practices

### 1. Documentation and Reproducibility
```python
# Always document cleaning decisions
cleaning_log = {
    'step': 'Remove outliers',
    'method': 'IQR method',
    'threshold': '1.5 * IQR',
    'records_affected': 23,
    'justification': 'Values beyond 3 standard deviations from mean'
}
```

### 2. Validation and Testing
```python
# Implement validation checks
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Test cleaning functions
assert validate_email('test@example.com') == True
assert validate_email('invalid-email') == False
```

### 3. Preserve Original Data
```python
# Always keep original data
df_original = df.copy()
df_cleaned = clean_data(df_original)

# Track changes
changes_made = compare_dataframes(df_original, df_cleaned)
```

### 4. Iterative Approach
1. **Assess** - Understand data quality issues
2. **Plan** - Develop cleaning strategy
3. **Clean** - Apply cleaning operations
4. **Validate** - Check results
5. **Document** - Record decisions and changes
6. **Iterate** - Refine based on results

## Common Data Quality Issues

### Missing Data Patterns
- **MCAR** (Missing Completely At Random)
- **MAR** (Missing At Random)
- **MNAR** (Missing Not At Random)

### Data Type Issues
- Numeric data stored as strings
- Dates in various formats
- Boolean values as text
- Mixed data types in columns

### Consistency Issues
- Different units of measurement
- Varying precision levels
- Inconsistent categorization
- Conflicting information across sources

## Git Reminder

Save your work:
1. Create `lesson-11-data-cleaning` folder in your repository
2. Save cleaning scripts with clear documentation
3. Include before/after data samples
4. Document cleaning decisions and rationale
5. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 11: Data Cleaning"
   git push
   ```

## Next Lesson Preview

In Lesson 12, we'll learn about:
- **Data Visualization**: Creating effective charts and graphs
- **Matplotlib and Seaborn**: Python visualization libraries
- **Dashboard creation**: Interactive visualizations
- **Storytelling with data**: Communicating insights effectively
