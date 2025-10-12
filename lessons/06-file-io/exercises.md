# Lesson 06 Exercises: File Input/Output and CSV Processing

## Guided Exercises (Do with Instructor)

### Exercise 1: Basic File Operations
**Goal**: Practice reading and writing text files

**Tasks**:
1. Create a text file with your favorite quotes
2. Read the file and display contents
3. Append a new quote to the file
4. Count lines and words in the file

```python
# Create and write to file
with open("quotes.txt", "w") as file:
    # Write your quotes here
    pass

# Read and display
# Add your code here
```

---

### Exercise 2: Error Handling with Files
**Goal**: Handle common file errors gracefully

**Tasks**:
1. Try to open a non-existent file
2. Handle the FileNotFoundError
3. Try to write to a read-only file
4. Use try-except blocks properly

```python
try:
    # Attempt file operations
    pass
except FileNotFoundError:
    # Handle missing file
    pass
except PermissionError:
    # Handle permission issues
    pass
```

---

### Exercise 3: CSV File Basics
**Goal**: Read and write CSV data

**Tasks**:
1. Create a CSV file with student data
2. Read the CSV and display as a table
3. Add new student records
4. Calculate averages from numeric columns

```python
import csv

# Sample data structure
students = [
    ["Name", "Age", "Grade", "Score"],
    ["Alice", 20, "A", 95],
    # Add more students
]
```

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Personal Journal Application
**Goal**: Build a simple journal that saves to files

**Requirements**:
1. Create daily journal entries
2. Save entries with timestamps
3. Read and display past entries
4. Search entries by keyword

**Features**:
- Each entry saved to separate file or append to one file
- Include date/time stamps
- Allow viewing entries by date range

---

### Exercise 5: Grade Book File Manager
**Goal**: Manage student grades using CSV files

**Create a system that**:
1. Loads student data from CSV
2. Adds new students and grades
3. Calculates class statistics
4. Exports reports to new CSV files

**CSV Structure**:
```
Student_ID,Name,Math,Science,English,History
001,Alice Johnson,85,92,78,88
002,Bob Smith,76,84,82,79
```

---

### Exercise 6: Inventory System with File Storage
**Goal**: Create persistent inventory management

**Requirements**:
1. Load inventory from CSV on startup
2. Track product additions/removals
3. Save changes back to CSV
4. Generate inventory reports
5. Handle low stock alerts

**Features**:
- Product ID, name, quantity, price, supplier
- Transaction logging to separate file
- Backup system for data safety

---

### Exercise 7: Log File Analyzer
**Goal**: Process and analyze log files

**Tasks**:
1. Read web server log files
2. Extract IP addresses, timestamps, status codes
3. Count requests per IP
4. Find most common error codes
5. Generate summary report

**Sample Log Format**:
```
192.168.1.1 - - [01/Jan/2024:12:00:00] "GET /index.html HTTP/1.1" 200 1234
192.168.1.2 - - [01/Jan/2024:12:01:00] "POST /login HTTP/1.1" 404 567
```

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Configuration File Manager
**Goal**: Create a system to manage application settings

**Requirements**:
1. Read configuration from JSON/INI files
2. Validate configuration values
3. Update settings programmatically
4. Create backup before changes
5. Support different config formats

**Config Example**:
```json
{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "myapp"
    },
    "logging": {
        "level": "INFO",
        "file": "app.log"
    }
}
```

---

### Challenge 2: Data Migration Tool
**Goal**: Convert data between different file formats

**Features**:
1. Read CSV and convert to JSON
2. Read JSON and convert to CSV
3. Handle nested data structures
4. Validate data during conversion
5. Generate conversion reports

**Support formats**: CSV, JSON, XML (basic)

---

### Challenge 3: File Synchronization System
**Goal**: Keep files synchronized between directories

**Requirements**:
1. Compare files in two directories
2. Identify new, modified, and deleted files
3. Copy changes from source to destination
4. Log all synchronization activities
5. Handle conflicts (same file modified in both)

---

## Real-World Application Exercises

### Exercise 8: Sales Report Generator
**Goal**: Process sales data and generate reports

**Tasks**:
1. Read sales data from CSV
2. Calculate monthly/quarterly totals
3. Find top-selling products
4. Generate formatted reports
5. Export charts data for visualization

**Data Structure**:
```csv
Date,Product,Quantity,Price,Salesperson
2024-01-15,Laptop,2,999.99,Alice
2024-01-15,Mouse,5,29.99,Bob
```

---

### Exercise 9: Contact Import/Export System
**Goal**: Handle contact data in multiple formats

**Features**:
1. Import contacts from CSV
2. Export to different formats (CSV, JSON, vCard)
3. Merge duplicate contacts
4. Validate email addresses and phone numbers
5. Generate mailing lists

---

### Exercise 10: Backup and Archive System
**Goal**: Create automated backup solution

**Requirements**:
1. Backup specified directories
2. Create compressed archives
3. Maintain backup history
4. Clean up old backups
5. Verify backup integrity

**Use Python's zipfile and shutil modules**

---

## File Processing Patterns

### Exercise 11: Large File Processing
**Goal**: Handle files too large to fit in memory

**Techniques**:
1. Process files line by line
2. Use generators for memory efficiency
3. Implement progress tracking
4. Handle interrupted processing

```python
def process_large_file(filename):
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            # Process one line at a time
            yield process_line(line)
            
            if line_num % 10000 == 0:
                print(f"Processed {line_num} lines")
```

---

### Exercise 12: File Monitoring System
**Goal**: Monitor files for changes

**Features**:
1. Watch directory for new files
2. Process files as they appear
3. Move processed files to archive
4. Log all activities
5. Handle errors gracefully

**Use os.path.getmtime() to detect changes**

---

## Data Validation Exercises

### Exercise 13: CSV Data Validator
**Goal**: Validate CSV data integrity

**Validation Rules**:
1. Check required columns exist
2. Validate data types (numbers, dates, emails)
3. Check for duplicate records
4. Verify referential integrity
5. Generate validation reports

**Example Validations**:
- Email format: contains @ and valid domain
- Phone format: proper number of digits
- Date format: valid date strings
- Numeric ranges: within expected bounds

---

### Exercise 14: Data Cleaning Pipeline
**Goal**: Clean messy data files

**Cleaning Tasks**:
1. Remove empty rows and columns
2. Standardize text formatting (case, spacing)
3. Fix common data entry errors
4. Handle missing values
5. Export cleaned data

**Common Issues to Fix**:
- Inconsistent date formats
- Mixed case in categorical data
- Extra whitespace
- Invalid characters
- Duplicate entries

---

## Error Handling and Recovery

### Exercise 15: Robust File Operations
**Goal**: Handle all possible file operation errors

**Error Scenarios**:
1. File doesn't exist
2. Permission denied
3. Disk full
4. Network drive disconnected
5. File locked by another process

```python
import os
import shutil
from pathlib import Path

def safe_file_operation(source, destination):
    try:
        # Attempt file operation
        pass
    except FileNotFoundError:
        # Handle missing file
        pass
    except PermissionError:
        # Handle permission issues
        pass
    except OSError as e:
        # Handle other OS-related errors
        pass
    finally:
        # Cleanup code
        pass
```

---

## Performance Optimization

### Exercise 16: File I/O Performance Testing
**Goal**: Compare different file reading methods

**Methods to Test**:
1. Reading entire file at once
2. Reading line by line
3. Reading in chunks
4. Using buffered I/O
5. Memory mapping for large files

**Measure**:
- Processing time
- Memory usage
- CPU utilization

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Open, read, and write text files
- [ ] Use context managers (with statements) properly
- [ ] Handle file-related exceptions gracefully
- [ ] Read and write CSV files using csv module
- [ ] Process large files efficiently
- [ ] Work with file paths using pathlib
- [ ] Create backup and recovery systems
- [ ] Validate and clean data from files
- [ ] Convert between different file formats
- [ ] Implement file monitoring and processing systems

## Common File Operations Reference

### Text Files
```python
# Reading
with open('file.txt', 'r') as f:
    content = f.read()          # Entire file
    lines = f.readlines()       # List of lines
    for line in f:              # Line by line

# Writing
with open('file.txt', 'w') as f:
    f.write("Hello World")      # Write string
    f.writelines(lines)         # Write list of lines

# Appending
with open('file.txt', 'a') as f:
    f.write("New content")
```

### CSV Files
```python
import csv

# Reading CSV
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# Writing CSV
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age'])
    writer.writerow(['Alice', 25])
```

### JSON Files
```python
import json

# Reading JSON
with open('data.json', 'r') as f:
    data = json.load(f)

# Writing JSON
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)
```

## Git Reminder

Save your work:

1. Create folder `lesson-06-file-io` in your repository
2. Save all exercise files and data files
3. Include sample data files for testing
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 06: File I/O"
   git push
   ```

## Next Lesson Preview

In Lesson 07, we'll learn about:
- **Testing**: Writing unit tests for your code
- **Debugging**: Finding and fixing bugs systematically
- **Test-driven development**: Writing tests before code
- **Debugging tools**: Using debugger and print statements effectively
